// RecompCore: StaticRecomp CPU core - Memory and instruction fallback HLE hooks.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Common/RunSettings.h"
#include "Core/System.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/JitInterface.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompLockstep.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/Memmap.h"
#include "Core/HW/SystemTimers.h"
#include "Common/Logging/Log.h"

#include <cstdio>
#include <cstdlib>
#include <intrin.h>
#include "Core/FzgxProbe.h"
#include "Core/HostHooks.h"
namespace { struct FzTimer { u64 t = __rdtsc(); ~FzTimer() { FzgxProbe::g_interp_tsc += __rdtsc() - t; } }; }

namespace
{
inline StaticRecompCore* core_of(CPUState* cpu) { return static_cast<StaticRecompCore*>(cpu->external_user_data); }
constexpr u32 LOCKED_CACHE_BASE = 0xE0000000u;
}

bool StaticRecompCore::HookHostCall(CPUState* cpu, u32 address)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  return core->m_module_source.host_call &&
         core->m_module_source.host_call(cpu, address, core->m_module_source.host_call_user);
}

u64 StaticRecompCore::HookExternalRead(CPUState* cpu, u32 ea, u8 size)
{
  ++FzgxProbe::g_ext[ea >> 28];
  // Spike fast path: the locked L1 cache (0xE0000000, GX's scratchpad) straight from Dolphin's backing store.
  if ((ea >> 28) == 0xEu)
  {
    auto& memory = core_of(cpu)->m_system.GetMemory();
    const u32 off = ea - 0xE0000000u;
    if (memory.GetL1Cache() && off + size <= memory.GetL1CacheSize())
    {
      const u8* p = memory.GetL1Cache() + off;
      u64 v = 0;
      for (u8 i = 0; i < size; ++i)
        v = (v << 8) | p[i];
      FzgxProbe::Census(3);
      return v;
    }
  }
  // The host program's window (Core/HostHooks.h): its data is the host's, answered here before anything of the
  // emulated machine is asked (the MMU would answer the same, after the bookkeeping of a hardware access).
  if (const HostHooks::Hooks& hooks = HostHooks::Get(); hooks.InWindow(ea))
    return hooks.window_read(ea, size);
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
    std::fprintf(stderr, "[zero-access] read size=%u guest_pc=%08x ppc_pc=%08x lr=%08x\n", size,
                 cpu->pc, core->m_system.GetPPCState().pc, cpu->lr);
  FzgxProbe::Census(2);
  FzgxProbe::CensusAccess(cpu->pc, ea, false);
  struct ReadTimer { u64 t = __rdtsc(); ~ReadTimer() { if (FzgxProbe::g_census_task >= 0) FzgxProbe::g_census_read_tsc[FzgxProbe::g_census_task] += __rdtsc() - t; } } read_timer;
  core->PropagateGuestMSR();
  core->FlushDispatchCharge();
  struct Refresh { StaticRecompCore* c; ~Refresh() { c->RefreshDispatchBudget(); } } refresh{core};
  auto& mmu = core->m_system.GetMMU();
  u64 value;
  switch (size)
  {
  case 1:
    value = mmu.Read<u8>(ea);
    break;
  case 2:
    value = mmu.Read<u16>(ea);
    break;
  case 4:
    value = mmu.Read<u32>(ea);
    break;
  case 8:
    value = mmu.Read<u64>(ea);
    break;
  default:
    ERROR_LOG_FMT(POWERPC, "StaticRecomp: external read of bad size {} at 0x{:08X}", size, ea);
    return 0;
  }
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_reads.push_back({ea, static_cast<u32>(value), size});
  }
  return value;
}

void StaticRecompCore::HookExternalWrite(CPUState* cpu, u32 ea, u64 value, u8 size)
{
  struct WriteTimer
  {
    u64 t = FzgxProbe::g_census_on ? __rdtsc() : 0;
    bool gather = false;
    ~WriteTimer()
    {
      if (!FzgxProbe::g_census_on)
        return;
      const u64 d = __rdtsc() - t;
      if (gather) { FzgxProbe::g_gather_tsc += d; ++FzgxProbe::g_gather_writes; }
      else { FzgxProbe::g_hwwrite_tsc += d; ++FzgxProbe::g_hwwrite_count; }
    }
  } write_timer;
  ++FzgxProbe::g_ext[16 + (ea >> 28)];
  if ((ea >> 28) == 0xEu)
  {
    auto& memory = core_of(cpu)->m_system.GetMemory();
    const u32 off = ea - 0xE0000000u;
    if (memory.GetL1Cache() && off + size <= memory.GetL1CacheSize())
    {
      u8* p = memory.GetL1Cache() + off;
      for (u8 i = size; i > 0; --i)
      {
        p[i - 1] = static_cast<u8>(value);
        value >>= 8;
      }
      FzgxProbe::Census(6);
      return;
    }
  }
  // the host program's window: see HookExternalRead
  if (const HostHooks::Hooks& hooks = HostHooks::Get(); hooks.InWindow(ea))
  {
    hooks.window_write(ea, value, size);
    return;
  }
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  if (ea == 0)
    std::fprintf(stderr, "[zero-access] write size=%u guest_pc=%08x ppc_pc=%08x lr=%08x\n", size,
                 cpu->pc, core->m_system.GetPPCState().pc, cpu->lr);

  // Gather-pipe fast path: stores to the write-gather pipe page at effective
  // 0xCC008000 go straight to GPFifo, mirroring the MMU's masked-write
  // special case without an MMU round trip. Keying on the effective page is
  // the same shortcut Dolphin's JITs take (optimizeGatherPipe). GPFifo
  // maintains ppc_state.gather_pipe_ptr internally.
  core->FlushDispatchCharge();
  struct Refresh { StaticRecompCore* c; ~Refresh() { c->RefreshDispatchBudget(); } } refresh{core};
  if ((ea & 0xFFFFF000) == 0xCC008000u)
  {
    FzgxProbe::Census(4);
    write_timer.gather = true;
    if (core->m_lockstep_verifier->m_ls_journaling)
      core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, static_cast<u32>(value), size});
    auto& gpfifo = core->m_system.GetGPFifo();
    switch (size)
    {
    case 1:
      gpfifo.Write8(static_cast<u8>(value));
      return;
    case 2:
      gpfifo.Write16(static_cast<u16>(value));
      return;
    case 4:
      gpfifo.Write32(static_cast<u32>(value));
      return;
    default:
      for (u32 i = size * 8u; i > 0;)
      {
        i -= 8;
        gpfifo.Write8(static_cast<u8>(value >> i));
      }
      return;
    }
  }

  core->PropagateGuestMSR();
  auto& mmu = core->m_system.GetMMU();
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, static_cast<u32>(value), size});
  }
  FzgxProbe::Census(5);
  FzgxProbe::CensusAccess(cpu->pc, ea, true);
  switch (size)
  {
  case 1:
    mmu.Write<u8>(static_cast<u8>(value), ea);
    break;
  case 2:
    mmu.Write<u16>(static_cast<u16>(value), ea);
    break;
  case 4:
    mmu.Write<u32>(static_cast<u32>(value), ea);
    break;
  case 8:
    mmu.Write<u64>(value, ea);
    break;
  default:
    ERROR_LOG_FMT(POWERPC, "StaticRecomp: external write of bad size {} at 0x{:08X}", size, ea);
    break;
  }
}

void StaticRecompCore::HookGatherPipeFlush(CPUState* cpu)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ++FzgxProbe::g_gather_flushes;
  core->FlushDispatchCharge();
  core->m_system.GetGPFifo().FastCheckGatherPipe();
  core->RefreshDispatchBudget();
}

// Hands the module the write-gather pipe (or takes it away): the address of Dolphin's pipe pointer, the start of
// the pipe buffer and the flush hook. Not while the lockstep verifier journals hardware writes, which must all
// pass through HookExternalWrite. FZGX_GATHER_INLINE=0 keeps every pipe write on the hook (for comparisons).
void StaticRecompCore::PublishGatherPipe()
{
  if (!m_set_gather_pipe)
    return;
  auto& ppc = m_system.GetPPCState();
  const char* off = RunSettings::Get("FZGX_GATHER_INLINE");
  const bool inline_ok = !(off && off[0] == '0') && !m_lockstep_verifier->IsEnabled() &&
                         ppc.gather_pipe_base_ptr != nullptr;
  if (inline_ok)
    m_set_gather_pipe(&ppc.gather_pipe_ptr, ppc.gather_pipe_base_ptr, &HookGatherPipeFlush);
  else
    m_set_gather_pipe(nullptr, nullptr, nullptr);
  std::fprintf(stderr, "[staticrecomp] gather pipe written by generated code: %s\n", inline_ok ? "yes" : "no");
}

u32 StaticRecompCore::HookExternalRead32(CPUState* cpu, u32 ea, u8 rid)
{
  // eciwx external-control read. EAR-enable and alignment were checked by the
  // generated helper; Dolphin's interpreter services the access as a plain
  // MMU read (the rid is carried in EAR only).
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  core->FlushDispatchCharge();
  struct Refresh { StaticRecompCore* c; ~Refresh() { c->RefreshDispatchBudget(); } } refresh{core};
  auto& mmu = core->m_system.GetMMU();
  const u32 value = mmu.Read<u32>(ea);
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_reads.push_back({ea, value, 4});
  }
  return value;
}

void StaticRecompCore::HookExternalWrite32(CPUState* cpu, u32 ea, u32 value, u8 rid)
{
  // ecowx external-control write; see HookExternalRead32.
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  core->FlushDispatchCharge();
  struct Refresh { StaticRecompCore* c; ~Refresh() { c->RefreshDispatchBudget(); } } refresh{core};
  auto& mmu = core->m_system.GetMMU();
  if (core->m_lockstep_verifier->m_ls_journaling &&
      StaticRecompLockstep::LsHwAccessInScope(mmu, ea))
  {
    core->m_lockstep_verifier->m_journal.native_mmio.push_back({ea, value, 4});
  }
  mmu.Write<u32>(value, ea);
}

void* StaticRecompCore::HookExternalPointer(CPUState* cpu, u32 ea, u32 size)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& memory = core->m_system.GetMemory();
  if (ea >= LOCKED_CACHE_BASE && size != 0 &&
      (ea - LOCKED_CACHE_BASE) + size <= memory.GetL1CacheSize())
  {
    return memory.GetL1Cache() + (ea - LOCKED_CACHE_BASE);
  }
  // Everything else stays on the per-access MMU hooks: this hook receives
  // *effective* addresses, and whether one maps to RAM depends on live
  // MSR/BAT state that only the MMU can answer. Handing out a raw pointer
  // here would bypass MMIO and translation. (Memory::GetPointerForRange was
  // considered and rejected for exactly that reason.)
  return nullptr;
}

u32 StaticRecompCore::HookSPRRead(CPUState* cpu, u16 spr, u32 cia)
{
  FzgxProbe::Census(8);
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& ppc = core->m_system.GetPPCState();
  if (spr >= 1024)
  {
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    return 0;
  }

  switch (spr)
  {
  case SPR_DEC:
    if ((ppc.spr[SPR_DEC] & 0x80000000u) == 0)
      ppc.spr[SPR_DEC] = core->m_system.GetSystemTimers().GetFakeDecrementer();
    break;
  case SPR_WPAR:
    if (core->m_system.GetGPFifo().IsBNE())
      ppc.spr[SPR_WPAR] |= 1;
    else
      ppc.spr[SPR_WPAR] &= ~1u;
    break;
  case SPR_UPMC1:
    return ppc.spr[SPR_PMC1];
  case SPR_UPMC2:
    return ppc.spr[SPR_PMC2];
  case SPR_UPMC3:
    return ppc.spr[SPR_PMC3];
  case SPR_UPMC4:
    return ppc.spr[SPR_PMC4];
  case SPR_IABR:
    return ppc.spr[SPR_IABR] & ~1u;
  default:
    break;
  }
  return ppc.spr[spr];
}

void StaticRecompCore::HookSPRWrite(CPUState* cpu, u16 spr, u32 value, u32 cia)
{
  FzgxProbe::Census(9);
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  auto& system = core->m_system;
  auto& ppc = system.GetPPCState();
  if (spr >= 1024)
  {
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    return;
  }

  const u32 old_value = ppc.spr[spr];
  ppc.spr[spr] = value;

  switch (spr)
  {
  case SPR_TL_W:
    TL(ppc) = value;
    system.GetSystemTimers().TimeBaseSet();
    return;
  case SPR_TU_W:
    TU(ppc) = value;
    system.GetSystemTimers().TimeBaseSet();
    return;
  case SPR_PVR:
    ppc.spr[SPR_PVR] = old_value;
    return;
  case SPR_HID0:
  {
    if (HID0(ppc).ICFI)
    {
      HID0(ppc).ICFI = 0;
      ppc.iCache.Reset(system.GetJitInterface());
    }
    return;
  }
  case SPR_HID1:
    ppc.spr[SPR_HID1] &= 0xF8000000u;
    return;
  case SPR_HID4:
    if (old_value != value)
    {
      system.GetMMU().IBATUpdated();
      system.GetMMU().DBATUpdated();
    }
    return;
  case SPR_WPAR:
    system.GetGPFifo().ResetGatherPipe();
    return;
  case SPR_DMAL:
    if (DMAL(ppc).DMA_T)
    {
      const u32 mem_address = DMAU(ppc).MEM_ADDR << 5;
      const u32 cache_address = DMAL(ppc).LC_ADDR << 5;
      u32 length = (DMAU(ppc).DMA_LEN_U << 2) | DMAL(ppc).DMA_LEN_L;
      if (length == 0)
        length = 128;
      if (DMAL(ppc).DMA_LD)
        system.GetMMU().DMA_MemoryToLC(cache_address, mem_address, length);
      else
        system.GetMMU().DMA_LCToMemory(mem_address, cache_address, length);
    }
    DMAL(ppc).DMA_T = 0;
    return;
  case SPR_DEC:
    if ((old_value >> 31) == 0 && (value >> 31) != 0)
      ppc.Exceptions |= EXCEPTION_DECREMENTER;
    system.GetSystemTimers().DecrementerSet();
    return;
  case SPR_SDR:
    system.GetMMU().SDRUpdated();
    return;
  case SPR_MMCR0:
  case SPR_MMCR1:
    PowerPC::MMCRUpdated(ppc);
    return;
  case SPR_THRM1:
  case SPR_THRM2:
  case SPR_THRM3:
  {
    const auto update = [&ppc](UReg_THRM12* reg) {
      if (!THRM3(ppc).E || !reg->V)
      {
        reg->TIV = 0;
      }
      else
      {
        reg->TIV = 1;
        reg->TIN = reg->TID ? 42 < reg->THRESHOLD : 42 > reg->THRESHOLD;
      }
    };
    update(&THRM1(ppc));
    update(&THRM2(ppc));
    return;
  }
  default:
    break;
  }

  const bool ibat = (spr >= SPR_IBAT0U && spr <= SPR_IBAT3L) ||
                    (spr >= SPR_IBAT4U && spr <= SPR_IBAT7L);
  const bool dbat = (spr >= SPR_DBAT0U && spr <= SPR_DBAT3L) ||
                    (spr >= SPR_DBAT4U && spr <= SPR_DBAT7L);
  if (old_value != value && ibat)
    system.GetMMU().IBATUpdated();
  else if (old_value != value && dbat)
    system.GetMMU().DBATUpdated();
}

void StaticRecompCore::HookCacheControl(CPUState* cpu, u8 operation, u32 ea, u32 cia)
{
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  ea = core->TranslateRelAddress(ea);
  core->PropagateGuestMSR();
  auto& ppc = core->m_system.GetPPCState();
  auto& mmu = core->m_system.GetMMU();

  if (operation == PPC_CACHE_ICBI)
  {
    ppc.iCache.Invalidate(core->m_system.GetMemory(), core->m_system.GetJitInterface(), ea);
    return;
  }

  if (!ppc.m_enable_dcache)
  {
    core->m_system.GetJitInterface().InvalidateICacheLine(ea);
    return;
  }

  switch (operation)
  {
  case PPC_CACHE_DCBST:
    mmu.StoreDCacheLine(ea);
    break;
  case PPC_CACHE_DCBF:
    mmu.FlushDCacheLine(ea);
    break;
  case PPC_CACHE_DCBI:
    mmu.InvalidateDCacheLine(ea);
    break;
  default:
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    break;
  }
}

void StaticRecompCore::HookInstructionFallback(CPUState* cpu, u32 raw, u32 cia)
{
  FzgxProbe::Census(7);
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  cia = core->TranslateRelAddress(cia);
  ++core->m_hook_fallback_instructions;
  FzgxProbe::PaceEvent(1, cia);

  // Lockstep: a block that fell back to the interpreter for an unmodeled
  // instruction (DMA mtspr, cache op, ...) performed side effects not captured
  // by the RAM journal / MMIO hooks, so re-running it on the shadow would
  // double-issue them. Mark it unsafe to differentially check.
  if (core->m_lockstep_verifier->m_ls_journaling)
    core->m_lockstep_verifier->m_ls_fallback_seen = true;

  auto& system = core->m_system;
  auto& ppc = system.GetPPCState();

  // Fast path for dcbf/dcbst/dcbi/icbi: streaming code flushes caches in
  // 32-byte loops (thousands per frame), and these ops read two GPRs and
  // change no CPU state, so they run straight off ctx without the full
  // SyncOut/interpreter/SyncIn round trip. This mirrors Dolphin's
  // interpreter with dcache emulation off: every one funnels into
  // InvalidateICacheLine (keeping the SMC guard exact). dcbi's PR!=0
  // privilege trap and dcache-on configs take the slow path.
  if ((raw >> 26) == 31u && !ppc.m_enable_dcache)
  {
    const u32 xo = (raw >> 1) & 0x3FFu;
    if (xo == 86u || xo == 54u || xo == 982u || (xo == 470u && (cpu->msr & 0x4000u) == 0))
    {
      const u32 ra = (raw >> 16) & 31u;
      const u32 rb = (raw >> 11) & 31u;
      const u32 ea = (ra ? cpu->gpr[ra] : 0u) + cpu->gpr[rb];
      if (xo == 982u)
        ppc.iCache.Invalidate(system.GetMemory(), system.GetJitInterface(), ea);
      else
        system.GetJitInterface().InvalidateICacheLine(ea);
      // These bypass SingleStepInner, so charge Dolphin's PPCTables cost
      // here (icbi 4, dcbf/dcbst/dcbi 5); their emitted block cost is zero.
      ppc.downcount -= (xo == 982u) ? 4 : 5;
      cpu->pc = cia + 4u;
      return;
    }
  }

  // Spike fast path: mfspr of BATs, SPRGs, HID0 and HID1 only copies a value
  // that the native state does not mirror; no SyncOut/interpreter round trip.
  if ((raw >> 26) == 31u && ((raw >> 1) & 0x3FFu) == 339u)
  {
    const u32 spr = (((raw >> 11) & 0x1Fu) << 5) | ((raw >> 16) & 0x1Fu);
    if ((spr >= 528u && spr <= 543u) || (spr >= 560u && spr <= 575u) ||
        (spr >= 272u && spr <= 275u) || spr == 1008u || spr == 1009u)
    {
      cpu->gpr[(raw >> 21) & 31u] = ppc.spr[spr];
      ppc.downcount -= 1;
      cpu->pc = cia + 4u;
      return;
    }
  }

  // The recompiled segment resumes via the dispatcher at the PC this leaves
  // behind, so this must execute exactly the instruction at cia via
  // Dolphin's interpreter and hand the register state back.
  core->SyncOut();
  ppc.pc = cia;
  ppc.npc = cia + 4;
  ppc.downcount -= system.GetInterpreter().SingleStepInner();
  core->SyncIn();
}
