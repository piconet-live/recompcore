// RecompCore: StaticRecomp CPU core - State synchronization.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>
#include "Core/CoreTiming.h"
#include "Core/HW/SystemTimers.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"

void StaticRecompCore::SetPPCStateFromGuestState(const CPUState& s, PowerPC::PowerPCState& ppc)
{
  std::memcpy(ppc.gpr, s.gpr, sizeof(ppc.gpr));
  for (int i = 0; i < 32; ++i)
  {
    std::memcpy(&ppc.ps[i].ps0, &s.fpr[i], sizeof(u64));
    std::memcpy(&ppc.ps[i].ps1, &s.ps1[i], sizeof(u64));
  }
  ppc.pc = s.pc;
  ppc.npc = s.pc;
  ppc.spr[SPR_LR] = s.lr;
  ppc.spr[SPR_CTR] = s.ctr;
  ppc.cr.Set(s.cr);
  ppc.SetXER(UReg_XER{s.xer});
  ppc.fpscr.Hex = s.fpscr;
  ppc.spr[SPR_SRR0] = s.srr0;
  ppc.spr[SPR_SRR1] = s.srr1;
  ppc.spr[SPR_DAR] = s.dar;
  ppc.spr[SPR_DSISR] = s.dsisr;
  ppc.spr[SPR_EAR] = s.ear;
  ppc.spr[SPR_HID2] = s.hid2;
  for (int i = 0; i < 16; ++i)
    ppc.sr[i] = s.sr[i];
  for (int i = 0; i < 8; ++i)
    ppc.spr[SPR_GQR0 + i] = s.gqr[i];
  ppc.reserve_address = s.reserve_addr;
  ppc.reserve = s.reserve_valid;
}

// The time base counts emulated ticks, twelve to a step. What generated code charges are the cycles of the
// instructions it ran, which is Dolphin's downcount: with the emulated CPU clock raised (Core.Overclock) that is the
// clock's factor times the ticks that passed. Counting the charge as ticks made the guest's time base run that many
// times too fast inside a burst and fall back at the next one (it is set from the emulator's clock at every
// SyncIn): a guest loop that reads the time saw it jump forward and back. The quotient is kept as a fraction of a
// tick, so nothing is lost between dispatches; with the clock at 1 the arithmetic is the old one.
// Test switch FZGX_TIMEBASE_BY_CHARGE=1: the fault as it was (the charge counted as ticks), to show whether a
// player's report from before the fix was this fault. Never set for a player. StaticRecompCore::Init sets it from
// this run's settings (Common/RunSettings.h): it is no static initialiser's, which would run before they are known.
bool g_staticrecomp_timebase_by_charge = false;

void StaticRecompCore::AdvanceGuestTimebase(u64 cpu_cycles)
{
  // The factor in effect for this slice, as CoreTiming's own DowncountToCycles uses it.
  const double per_cycle = m_system.GetCoreTiming().GetGlobals().last_OC_factor_inverted;
  if (per_cycle > 0.0 && per_cycle != 1.0 && !g_staticrecomp_timebase_by_charge)
  {
    const double ticks = static_cast<double>(cpu_cycles) * per_cycle + m_timebase_tick_fraction;
    cpu_cycles = static_cast<u64>(ticks);
    m_timebase_tick_fraction = ticks - static_cast<double>(cpu_cycles);
  }
  const u64 total_cycles = m_timebase_cycle_remainder + cpu_cycles;
  m_guest.timebase += total_cycles / SystemTimers::TIMER_RATIO;
  m_timebase_cycle_remainder = total_cycles % SystemTimers::TIMER_RATIO;
}

void StaticRecompCore::BeginDispatchBudget(bool waiting)
{
  const s64 left = m_system.GetPPCState().downcount;
  m_dispatch_flushed = 0;
  m_guest.cycle_budget = waiting ? 256 : (left > 0 ? left : 1);
}

void StaticRecompCore::FlushDispatchCharge()
{
  if (!m_hook_flush)
    return;
  const s64 charge = -m_guest.downcount;
  if (charge <= 0)
    return;
  m_guest.downcount = 0;
  m_system.GetPPCState().downcount -= static_cast<int>(charge);
  m_charged_cycles += static_cast<u64>(charge);
  m_dispatch_flushed += static_cast<u64>(charge);
  AdvanceGuestTimebase(static_cast<u64>(charge));
}

void StaticRecompCore::RefreshDispatchBudget()
{
  if (m_wait_streak != 0)
    return;
  // Generated code compares the cycles since the dispatch began with the budget. Those already charged are in
  // m_dispatch_flushed; Dolphin's downcount is what is left after them.
  const s64 left = m_system.GetPPCState().downcount;
  m_guest.cycle_budget = static_cast<s64>(m_dispatch_flushed) + (left > 0 ? left : 0);
}

void StaticRecompCore::SyncIn()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();

  std::memcpy(m_guest.gpr, ppc.gpr, sizeof(m_guest.gpr));
  for (int i = 0; i < 32; ++i)
  {
    std::memcpy(&m_guest.fpr[i], &ppc.ps[i].ps0, sizeof(u64));
    std::memcpy(&m_guest.ps1[i], &ppc.ps[i].ps1, sizeof(u64));
  }
  m_guest.pc = ppc.pc;
  m_guest.lr = ppc.spr[SPR_LR];
  m_guest.ctr = ppc.spr[SPR_CTR];
  m_guest.cr = ppc.cr.Get();
  m_guest.xer = ppc.GetXER().Hex;
  m_guest.fpscr = ppc.fpscr.Hex;
  m_guest.msr = ppc.msr.Hex;
  m_guest.srr0 = ppc.spr[SPR_SRR0];
  m_guest.srr1 = ppc.spr[SPR_SRR1];
  m_guest.dar = ppc.spr[SPR_DAR];
  m_guest.dsisr = ppc.spr[SPR_DSISR];
  m_guest.ear = ppc.spr[SPR_EAR];
  m_guest.hid2 = ppc.spr[SPR_HID2];
  for (int i = 0; i < 16; ++i)
    m_guest.sr[i] = ppc.sr[i];
  for (int i = 0; i < 8; ++i)
    m_guest.gqr[i] = ppc.spr[SPR_GQR0 + i];
  // Dolphin materializes TB lazily on read (spr[TL/TU] is a stale cache);
  // GetFakeTimeBase() is the live value, matching the interpreter's mftb.
  m_guest.timebase = m_system.GetSystemTimers().GetFakeTimeBase();
  const auto& core_timing = m_system.GetCoreTiming();
  m_timebase_cycle_remainder =
      (core_timing.GetTicks() - core_timing.GetFakeTBStartTicks()) % SystemTimers::TIMER_RATIO;
  m_timebase_tick_fraction = 0.0;
  m_guest.reserve_addr = ppc.reserve_address;
  m_guest.reserve_valid = ppc.reserve;
  m_guest.exception = 0;
  m_guest.program_exception = 0;
  m_guest.downcount = 0;  // charge accumulator, not a copy of ppc.downcount

  if (m_module && m_module->on_state_loaded)
    m_module->on_state_loaded(&m_guest);
}

void StaticRecompCore::SyncOut()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();

  // Flush any cycle charge the module accumulated since the last flush
  // (matters on the mid-dispatch fallback path, where the interpreter step
  // that follows must see up-to-date time).
  ppc.downcount += static_cast<int>(m_guest.downcount);
  m_guest.downcount = 0;

  SetPPCStateFromGuestState(m_guest, ppc);
  // Timebase is owned by Dolphin's CoreTiming; native code cannot write it.

  if (ppc.msr.Hex != m_guest.msr)
  {
    ppc.msr.Hex = m_guest.msr;
    power_pc.MSRUpdated();
  }
  PowerPC::RoundingModeUpdated(ppc);
}

void StaticRecompCore::PropagateGuestMSR()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();
  if (ppc.msr.Hex != m_guest.msr)
  {
    ppc.msr.Hex = m_guest.msr;
    power_pc.MSRUpdated();
  }
}
