// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// DL facts:
//  Ikaruga uses (nearly) NO display lists!
//  Zelda WW uses TONS of display lists
//  Zelda TP uses almost 100% display lists except menus (we like this!)
//  Super Mario Galaxy has nearly all geometry and more than half of the state in DLs (great!)

// Note that it IS NOT GENERALLY POSSIBLE to precompile display lists! You can compile them as they
// are while interpreting them, and hope that the vertex format doesn't change, though, if you do
// it right when they are called. The reason is that the vertex format affects the sizes of the
// vertices.

#include "VideoCommon/OpcodeDecoding.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>

#include <fmt/format.h>

#include "Common/Assert.h"
#include "Common/Logging/Log.h"
#include "Core/FifoPlayer/FifoRecorder.h"
#include "Core/HW/Memmap.h"
#include "Core/HW/ProcessorInterface.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"
#include "VideoCommon/AuroraPresent.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/CommandProcessor.h"
#include "VideoCommon/DataReader.h"
#include "VideoCommon/Fifo.h"
#include "VideoCommon/FzgxFifoTest.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/VertexLoaderBase.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/XFMemory.h"
#include "VideoCommon/XFStateManager.h"

namespace OpcodeDecoder
{
bool g_record_fifo_data = false;

template <bool is_preprocess>
class RunCallback final : public Callback
{
public:
  OPCODE_CALLBACK(void OnXF(u16 address, u8 count, const u8* data))
  {
    m_cycles += 18 + 6 * count;

    if constexpr (!is_preprocess)
    {
      LoadXFReg(address, count, data);

      INCSTAT(g_stats.this_frame.num_xf_loads);
    }
  }
  OPCODE_CALLBACK(void OnCP(u8 command, u32 value))
  {
    m_cycles += 12;
    const u8 sub_command = command & CP_COMMAND_MASK;
    if constexpr (!is_preprocess)
    {
      if (sub_command == MATINDEX_A)
      {
        VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
        auto& system = Core::System::GetInstance();
        system.GetXFStateManager().SetTexMatrixChangedA(value);
      }
      else if (sub_command == MATINDEX_B)
      {
        VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
        auto& system = Core::System::GetInstance();
        system.GetXFStateManager().SetTexMatrixChangedB(value);
      }
      else if (sub_command == VCD_LO || sub_command == VCD_HI)
      {
        VertexLoaderManager::g_main_vat_dirty = BitSet8::AllTrue(CP_NUM_VAT_REG);
        VertexLoaderManager::g_bases_dirty = true;
        VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
      }
      else if (sub_command == CP_VAT_REG_A || sub_command == CP_VAT_REG_B ||
               sub_command == CP_VAT_REG_C)
      {
        VertexLoaderManager::g_main_vat_dirty[command & CP_VAT_MASK] = true;
        VertexLoaderManager::g_needs_cp_xf_consistency_check = true;
      }
      else if (sub_command == ARRAY_BASE)
      {
        VertexLoaderManager::g_bases_dirty = true;
      }

      INCSTAT(g_stats.this_frame.num_cp_loads);
    }
    else if constexpr (is_preprocess)
    {
      if (sub_command == VCD_LO || sub_command == VCD_HI)
      {
        VertexLoaderManager::g_preprocess_vat_dirty = BitSet8::AllTrue(CP_NUM_VAT_REG);
      }
      else if (sub_command == CP_VAT_REG_A || sub_command == CP_VAT_REG_B ||
               sub_command == CP_VAT_REG_C)
      {
        VertexLoaderManager::g_preprocess_vat_dirty[command & CP_VAT_MASK] = true;
      }
    }
    GetCPState().LoadCPReg(command, value);
  }
  OPCODE_CALLBACK(void OnBP(u8 command, u32 value))
  {
    m_cycles += 12;

    if constexpr (is_preprocess)
    {
      LoadBPRegPreprocess(command, value, m_cycles);
    }
    else
    {
      LoadBPReg(command, value, m_cycles);
      INCSTAT(g_stats.this_frame.num_bp_loads);
    }
  }
  OPCODE_CALLBACK(void OnIndexedLoad(CPArray array, u32 index, u16 address, u8 size))
  {
    m_cycles += 6;

    if constexpr (is_preprocess)
      PreprocessIndexedXF(array, index, address, size);
    else
      LoadIndexedXF(array, index, address, size);
  }
  OPCODE_CALLBACK(void OnPrimitiveCommand(OpcodeDecoder::Primitive primitive, u8 vat,
                                          u32 vertex_size, u16 num_vertices, const u8* vertex_data))
  {
    // load vertices
    const u32 size = vertex_size * num_vertices;

    const u32 bytes =
        VertexLoaderManager::RunVertices<is_preprocess>(vat, primitive, num_vertices, vertex_data);

    ASSERT(bytes == size);

    // 4 GPU ticks per vertex, 3 CPU ticks per GPU tick
    m_cycles += num_vertices * 4 * 3 + 6;
  }
  // This can't be inlined since it calls Run, which makes it recursive
  // m_in_display_list prevents it from actually recursing infinitely, but there's no real benefit
  // to inlining Run for the display list directly.
  OPCODE_CALLBACK_NOINLINE(void OnDisplayList(u32 address, u32 size))
  {
    m_cycles += 6;

    if (m_in_display_list)
    {
      WARN_LOG_FMT(VIDEO, "recursive display list detected");
    }
    else
    {
      m_in_display_list = true;

      auto& system = Core::System::GetInstance();

      if constexpr (is_preprocess)
      {
        auto& memory = system.GetMemory();
        const u8* const start_address = memory.GetPointerForRange(address, size);

        system.GetFifo().PushFifoAuxBuffer(start_address, size);

        if (start_address != nullptr)
        {
          Run(start_address, size, *this);
        }
      }
      else
      {
        m_list_address = address;
        m_list_size = size;
        system.GetFifo().RecordDisplayList(address, size);
        if (FzgxFifoTest::On())
        {
          FzgxFifoTest::g_lists.Note({address, size, 0, 0, 0, 0, 0, 0});
          if (FzgxFifoTest::Watching() && FzgxFifoTest::InWatch(address, 1))
            FzgxFifoTest::Line("gpu calls list %08x size %x", address, size);
          FzgxFifoTest::Stall(FzgxFifoTest::Where::DisplayList);
        }
        const u8* start_address;

        auto& fifo = system.GetFifo();
        if (fifo.UseDeterministicGPUThread())
        {
          start_address = static_cast<u8*>(fifo.PopFifoAuxBuffer(size));
        }
        else
        {
          auto& memory = system.GetMemory();
          start_address = memory.GetPointerForRange(address, size);
        }

        // Avoid the crash if memory.GetPointerForRange failed ..
        if (start_address != nullptr)
        {
          VideoCommon::AuroraPresent::OnDisplayList(start_address, size);

          // temporarily swap dl and non-dl (small "hack" for the stats)
          g_stats.SwapDL();

          m_list_start = start_address;
          m_list_good = size;
          const u32 read = Run(start_address, size, *this);
          // Aurora is handed what the emulator read as commands: see AuroraPresent::OnListRead
          VideoCommon::AuroraPresent::OnListRead(std::min(read, m_list_good));
          INCSTAT(g_stats.this_frame.num_dlists_called);

          // un-swap
          g_stats.SwapDL();
        }
      }

      m_in_display_list = false;
    }
  }
  // A test's record of an unknown opcode (FzgxFifoTest.h): is it in a display list or the stream, and whose bytes
  void ReportUnknown(u8 opcode, const u8* data)
  {
    static std::atomic<int> reports{0};
    if (reports.fetch_add(1) >= 3)
      return;
    namespace T = FzgxFifoTest;
    auto& system = Core::System::GetInstance();
    auto& memory = system.GetMemory();
    auto& pi = system.GetProcessorInterface();
    const auto& fifo = system.GetCommandProcessor().GetFifo();
    const u64 now = T::NowMicros();
    const u32 burst = T::g_burst_guest.load(std::memory_order_relaxed);
    const u8* burst_host = T::g_burst_host.load(std::memory_order_relaxed);
    u32 guest = 0;
    if (m_in_display_list)
    {
      const u8* list = memory.GetPointerForRange(m_list_address, m_list_size);
      guest = m_list_address + static_cast<u32>(data - list);
    }
    else
    {
      guest = burst + static_cast<u32>(data - burst_host);
    }
    const u8* ram_now = memory.GetPointerForRange(guest, 128);
    ERROR_LOG_FMT(VIDEO,
                  "fzgx test: bad opcode {:#04x} in {}: list {:08x} size {:#x}; byte at guest {:08x}; burst guest "
                  "{:08x}, byte is {} bytes after the burst's start; {} us after the last reads on; stalls {} ({} us)",
                  opcode, m_in_display_list ? "a DISPLAY LIST" : "the STREAM", m_in_display_list ? m_list_address : 0,
                  m_in_display_list ? m_list_size : 0, guest, burst, data - burst_host,
                  now - T::g_list_on_us.load(std::memory_order_relaxed),
                  T::g_stalls.load(std::memory_order_relaxed), T::g_stalled_us.load(std::memory_order_relaxed));
    ERROR_LOG_FMT(VIDEO,
                  "fzgx test: CP base {:08x} end {:08x} read {:08x} write {:08x} distance {:#x}; CPU fifo base {:08x} "
                  "end {:08x} write {:08x}; PC {:08x} LR {:08x}",
                  fifo.CPBase.load(), fifo.CPEnd.load(), fifo.CPReadPointer.load(), fifo.CPWritePointer.load(),
                  fifo.CPReadWriteDistance.load(), pi.m_fifo_cpu_base, pi.m_fifo_cpu_end,
                  pi.m_fifo_cpu_write_pointer, system.GetPPCState().pc, LR(system.GetPPCState()));
    ERROR_LOG_FMT(VIDEO, "fzgx test: before: {}", T::Hex(data - 64, 64));
    ERROR_LOG_FMT(VIDEO, "fzgx test: read:   {}", T::Hex(data, 128));
    if (ram_now)
    {
      ERROR_LOG_FMT(VIDEO, "fzgx test: RAM now {} what was read: {}",
                    std::memcmp(ram_now, data, 128) ? "DIFFERS from" : "equals", T::Hex(ram_now, 128));
    }
    T::g_handovers.Each([&](const auto& e) {
      ERROR_LOG_FMT(VIDEO,
                    "fzgx test: reads {} -{} us: read {:08x} write {:08x} distance {:#x} base {:08x} end {:08x} cpu "
                    "write {:08x} cpu base {:08x}",
                    e.v[0] ? "on " : "off", now - e.us, e.v[1], e.v[2], e.v[3], e.v[4], e.v[5], e.v[6], e.v[7]);
    });
    std::string lists;
    T::g_lists.Each([&](const auto& e) {
      lists += fmt::format("-{}us:{:08x}+{:#x} ", now - e.us, e.v[0], e.v[1]);
    });
    ERROR_LOG_FMT(VIDEO, "fzgx test: lists: {}", lists);
    std::string aram;
    T::g_aram.Each([&](const auto& e) {
      const bool touches = e.v[3] && e.v[0] < guest + 128 && guest < e.v[0] + e.v[2];
      if (e.v[3])
        aram += fmt::format("-{}us:{:08x}+{:#x}{} ", now - e.us, e.v[0], e.v[2], touches ? "(OVER THE BYTE)" : "");
    });
    ERROR_LOG_FMT(VIDEO, "fzgx test: ARAM to RAM: {}", aram);
    std::string commands;
    T::g_commands.Each([&](const auto& e) {
      commands += fmt::format("{:02x}{}+{} ", e.v[0], e.v[2] ? "d" : "s", e.v[1]);
    });
    ERROR_LOG_FMT(VIDEO, "fzgx test: commands (opcode, d list / s stream, size): {}", commands);
  }

  OPCODE_CALLBACK(void OnNop(u32 count))
  {
    m_cycles += 6 * count;  // Hm, this means that we scan over nop streams pretty slowly...
  }
  OPCODE_CALLBACK(void OnUnknown(u8 opcode, const u8* data))
  {
    if (static_cast<Opcode>(opcode) == Opcode::GX_CMD_UNKNOWN_METRICS)
    {
      // 'Zelda Four Swords' calls it and checks the metrics registers after that
      m_cycles += 6;
      DEBUG_LOG_FMT(VIDEO, "GX 0x44");
    }
    else if (static_cast<Opcode>(opcode) == Opcode::GX_CMD_INVL_VC)
    {
      // Invalidate Vertex Cache
      m_cycles += 6;
      DEBUG_LOG_FMT(VIDEO, "Invalidate (vertex cache?)");
    }
    else
    {
      if constexpr (!is_preprocess)
      {
        if (m_in_display_list)
          m_list_good = std::min(m_list_good, static_cast<u32>(data - m_list_start));
        if (FzgxFifoTest::On())
          ReportUnknown(opcode, data);
      }
      auto& system = Core::System::GetInstance();
      system.GetCommandProcessor().HandleUnknownOpcode(opcode, data, is_preprocess);
      m_cycles += 1;
    }
  }

  OPCODE_CALLBACK(void OnCommand(const u8* data, u32 size))
  {
    ASSERT(size >= 1);
    if constexpr (!is_preprocess)
    {
      if (FzgxFifoTest::On())
        FzgxFifoTest::g_commands.Note({data[0], size, m_in_display_list, 0, 0, 0, 0, 0});
      // an XF load over 16 words is no command the hardware takes: the emulator reads 16 of them, Aurora the length
      if (m_in_display_list && static_cast<Opcode>(data[0]) == Opcode::GX_LOAD_XF_REG &&
          (data[1] | (data[2] & 0xf0)) != 0)
        m_list_good = std::min(m_list_good, static_cast<u32>(data - m_list_start));
      // Display lists get added directly into the FIFO stream since this same callback is used to
      // process them.
      if (g_record_fifo_data && static_cast<Opcode>(data[0]) != Opcode::GX_CMD_CALL_DL)
      {
        Core::System::GetInstance().GetFifoRecorder().WriteGPCommand(data, size);
      }
      // AuroraPresent: an XFB copy's command ended (of the FIFO stream itself: a display list's copy cuts at the
      // end of its call)
      if (VideoCommon::AuroraPresent::Pending() && !m_in_display_list)
        VideoCommon::AuroraPresent::OnCommandEnd(data, size);
    }
  }

  OPCODE_CALLBACK(CPState& GetCPState())
  {
    if constexpr (is_preprocess)
      return g_preprocess_cp_state;
    else
      return g_main_cp_state;
  }

  OPCODE_CALLBACK(u32 GetVertexSize(u8 vat))
  {
    VertexLoaderBase* loader = VertexLoaderManager::RefreshLoader<is_preprocess>(vat);
    return loader->m_vertex_size;
  }

  u32 m_cycles = 0;
  bool m_in_display_list = false;
  const u8* m_list_start = nullptr;
  u32 m_list_good = 0;
  u32 m_list_address = 0;
  u32 m_list_size = 0;
};

template <bool is_preprocess>
u8* RunFifo(DataReader src, u32* cycles)
{
  using CallbackT = RunCallback<is_preprocess>;
  auto callback = CallbackT{};
  u32 size = Run(src.GetPointer(), static_cast<u32>(src.size()), callback);

  if (cycles != nullptr)
    *cycles = callback.m_cycles;

  src.Skip(size);
  return src.GetPointer();
}

template u8* RunFifo<true>(DataReader src, u32* cycles);
template u8* RunFifo<false>(DataReader src, u32* cycles);

}  // namespace OpcodeDecoder
