// Spike probe (not upstream): per-field log of F-Zero GX state for comparing
// the StaticRecomp core with the JIT. Enabled by FZGX_PROBE=<csv path>.
// Columns: field, ticks, module, mode, submode, race_frame, racers_hash,
//          rel_hash, cpu_us (CPU-thread time since the previous field, probe excluded),
//          pad_reads (SI pad polls since the previous field), last_pad_tick.
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <unordered_map>
#include "Core/FzgxProbe.h"
#include "Common/RunSettings.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <timeapi.h>
#else
#include <ctime>
#include <sys/syscall.h>
#include <unistd.h>
#include <x86intrin.h>
namespace
{
// What the probe takes from Windows, on other systems: the local time of a log's header, a sleep and a thread id.
struct SYSTEMTIME
{
  unsigned wYear, wMonth, wDay, wHour, wMinute, wSecond, wMilliseconds;
};
void GetLocalTime(SYSTEMTIME* t)
{
  timespec now{};
  clock_gettime(CLOCK_REALTIME, &now);
  tm local{};
  localtime_r(&now.tv_sec, &local);
  *t = {unsigned(local.tm_year + 1900), unsigned(local.tm_mon + 1), unsigned(local.tm_mday), unsigned(local.tm_hour),
        unsigned(local.tm_min), unsigned(local.tm_sec), unsigned(now.tv_nsec / 1000000)};
}
void Sleep(unsigned ms)
{
  usleep(ms * 1000u);
}
unsigned long GetCurrentThreadId()
{
  return static_cast<unsigned long>(syscall(SYS_gettid));
}
}  // namespace
#endif

#include "Common/CommonTypes.h"
#include "Core/Config/MainSettings.h"
#include "Core/CoreTiming.h"
#include "Core/HW/Memmap.h"
#include "Core/System.h"
#include "Core/HLE/HLE.h"
#include "Core/PowerPC/PowerPC.h"
#include "InputCommon/GCPadStatus.h"
#include "VideoCommon/PerformanceMetrics.h"
#include "VideoCommon/Statistics.h"

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <cstdarg>
#include <mutex>
#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
#endif

namespace FzgxProbe
{
unsigned long long g_ext[32] = {};
unsigned long long g_task_emu_tsc = 0;
bool g_task_active = false;
int g_census_task = -1;
bool g_census_on = false;  // FZGX_CENSUS: ReadSettings()
unsigned long long g_census[3][12] = {};
unsigned long long g_census_read_tsc[3] = {};
unsigned long long g_gather_flushes = 0;
unsigned long long g_gather_tsc = 0, g_gather_writes = 0, g_hwwrite_tsc = 0, g_hwwrite_count = 0;
unsigned long long g_native_tsc = 0, g_jit_tsc = 0, g_interp_tsc = 0, g_dispatches = 0, g_idles = 0;
void (*g_core_counters)(unsigned long long out[4]) = nullptr;
namespace
{
std::FILE* s_file = nullptr;
bool s_checked = false;
bool s_hash = true;
u64 s_field = 0;
u64 s_last_end_cpu = 0;
u64 s_pad_reads = 0;
u64 s_last_pad_tick = 0;

// CPU-thread cycles (TSC reference cycles), not 15.6 ms GetThreadTimes ticks.
u64 ThreadCpu100ns()
{
#ifdef _WIN32
  ULONG64 cycles = 0;
  QueryThreadCycleTime(GetCurrentThread(), &cycles);
  return cycles;
#else
  return 0;
#endif
}

u64 Tsc()
{
  return __rdtsc();
}

u64 QpcUs()
{
#ifdef _WIN32
  LARGE_INTEGER c, f;
  QueryPerformanceCounter(&c);
  QueryPerformanceFrequency(&f);
  return static_cast<u64>(c.QuadPart * 1000000.0 / f.QuadPart);
#else
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<u64>(now.tv_sec) * 1000000u + static_cast<u64>(now.tv_nsec) / 1000u;
#endif
}

u32 BE32(const u8* p)
{
  return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | p[3];
}

u64 Hash(const u8* p, u32 n)
{
  u64 h = 0xcbf29ce484222325ull;
  u32 i = 0;
  for (; i + 8 <= n; i += 8)
  {
    u64 w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x100000001b3ull;
    h ^= h >> 29;
  }
  for (; i < n; ++i)
    h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}
struct PadStep
{
  u64 field;
  u16 buttons;
  u8 sx, sy;
  u8 cx = 0x80, cy = 0x80;  // the C-stick, centred unless the step gives it
};
std::vector<PadStep> s_pad_script;
bool s_pad_checked = false;
bool s_pad_active = false;
// Task timing: main.rel's task manager fn_1_407C calls each task with bctrl at text+0x411C and resumes at
// text+0x4120 (as in the fork's resim profile). Wall time per call of fn_1_83858, fn_1_850C4, fn_1_C4ABC and all tasks.
constexpr u32 TEXT_OFF = 0xDC;
u32 s_text = 0;
bool s_hooks = false;
u32 s_task_target = 0;
u64 s_task_t0 = 0;
unsigned long long s_te[4] = {};  // of s_tt, the part spent in CoreTiming::Advance
unsigned long long s_tt[4] = {}, s_tn[4] = {}, s_tc[4] = {};  // 83858, 850C4, C4ABC, all (tsc, calls, emulated ticks)
u64 s_task_c0 = 0;
u32 s_mxcsr = 0;
u64 Ticks() { return Core::System::GetInstance().GetCoreTiming().GetTicks(); }
void TaskEnd()
{
  if (!s_task_target)
    return;
  const u64 dt = __rdtsc() - s_task_t0;
  const u32 off = s_task_target - s_text;
  const int k = off == 0x83858 ? 0 : off == 0x850C4 ? 1 : off == 0xC4ABC ? 2 : -1;
  const u64 dc = Ticks() - s_task_c0;
  if (k >= 0) { s_tt[k] += dt; ++s_tn[k]; s_tc[k] += dc; s_te[k] += g_task_emu_tsc; }
  s_tt[3] += dt; ++s_tn[3]; s_tc[3] += dc; s_te[3] += g_task_emu_tsc;
  g_task_active = false;
  g_census_task = -1;
  s_task_target = 0;
}
// FZGX_TASKDUMP="N,M,...": full RAM at the first task call of race frame N (a fixed guest point in the main
// thread, unlike the VI), to <probe>.task<N>.ram
void TaskPointDump()
{
  static std::vector<u32> frames;
  static bool checked = false;
  if (!checked)
  {
    checked = true;
    if (const char* d = RunSettings::Get("FZGX_TASKDUMP"))
      for (const char* p = d; *p;)
      {
        char* e = nullptr;
        frames.push_back(static_cast<u32>(std::strtoul(p, &e, 10)));
        if (e == p) break;
        p = e;
        while (*p == ',') ++p;
      }
  }
  if (frames.empty() || !s_text)
    return;
  auto& memory = Core::System::GetInstance().GetMemory();
  const u8* ram = memory.GetRAM();
  const u32 rf = BE32(ram + (s_text - TEXT_OFF - 0x80000000u) + 0x1BA040u + 0x50A4u);
  for (u32& f : frames)
    if (f && f == rf)
    {
      char name[600];
      std::snprintf(name, sizeof(name), "%s.task%u.ram", RunSettings::Get("FZGX_PROBE"), f);
      if (std::FILE* out = std::fopen(name, "wb"))
      {
        std::fwrite(ram, 1, memory.GetRamSizeReal(), out);
        std::fclose(out);
      }
      f = 0;
    }
}
}  // namespace

static void ApplyTaskPokes(u32 task);
void HLE_TaskStart(const Core::CPUThreadGuard&)
{
  TaskPointDump();
  s_task_target = Core::System::GetInstance().GetPPCState().spr[SPR_CTR];
  ApplyTaskPokes(s_task_target);
  s_task_c0 = Ticks();
  s_mxcsr = _mm_getcsr();
  g_task_emu_tsc = 0;
  g_task_active = true;
  s_task_t0 = __rdtsc();
}

void HLE_TaskEnd(const Core::CPUThreadGuard&)
{
  TaskEnd();
}

static std::unordered_map<unsigned int, unsigned long long> s_census_pcs[3];
void CensusDispatch(unsigned int address)
{
  if (g_census_task < 0)
    return;
  ++g_census[g_census_task][1];
  ++s_census_pcs[g_census_task][address];
}
static std::unordered_map<unsigned int, std::pair<unsigned long long, unsigned int>> s_census_access[2];
void CensusAccess(unsigned int pc, unsigned int ea, bool write)
{
  if (g_census_task < 0)
    return;
  auto& entry = s_census_access[write ? 1 : 0][pc];
  ++entry.first;
  entry.second = ea;
}
bool g_trace_on = false;  // FZGX_TRACE: ReadSettings()
unsigned int g_trace_r1 = 0, g_trace_r3 = 0;
namespace
{
constexpr unsigned kShift = 2;
constexpr unsigned kBlocks = 0x1800000 >> kShift;  // 24 MB of RAM in 4-byte words
std::vector<unsigned> s_readers, s_writers;   // per block: bit i = call i of this frame
std::vector<unsigned> s_touched;              // blocks touched this frame
int s_trace_call = -1;                        // index of the running fn_1_83858 call in this frame, or -1
unsigned s_trace_sp[32], s_trace_arg[32];
unsigned long long s_trace_frames = 0, s_trace_calls = 0, s_frames_with_conflict = 0;
struct Conflict { unsigned long long frames = 0, ww = 0, rw = 0; unsigned readers = 0, writers = 0; int kind = 0; };
std::unordered_map<unsigned, Conflict> s_conflicts;  // block -> totals
unsigned long long s_kind_frames[4] = {};
bool s_trace_noirq = false;  // FZGX_TRACE=noirq: ReadSettings()
bool s_in_irq = false;
unsigned s_irq_pc = 0, s_irq_sp = 0;
unsigned long long s_irqs_in_calls = 0, s_irq_accesses = 0;
unsigned s_last_sp = 0;
}
void TraceInterrupt(unsigned int interrupted_pc, unsigned int stack_pointer)
{
  if (!g_trace_on || s_in_irq)
    return;
  s_in_irq = true;
  s_irq_pc = interrupted_pc;
  s_irq_sp = stack_pointer;
  if (s_trace_call >= 0 && g_census_task == 0)
    ++s_irqs_in_calls;
}
void TraceCallBegin(unsigned int stack_pointer, unsigned int first_argument)
{
  if (!g_trace_on)
    return;
  if (s_readers.empty()) { s_readers.assign(kBlocks, 0); s_writers.assign(kBlocks, 0); }
  if (s_trace_call < 31) ++s_trace_call;
  s_trace_sp[s_trace_call] = stack_pointer;
  s_last_sp = stack_pointer;
  s_trace_arg[s_trace_call] = first_argument;
  ++s_trace_calls;
}
void TraceAccess(unsigned int address, unsigned int width_and_write)
{
  if (s_trace_call < 0 || g_census_task != 0)
    return;
  if (s_in_irq)
  {
    ++s_irq_accesses;
    if (s_trace_noirq)
      return;
  }
  const unsigned physical = address & 0x3FFFFFFFu;
  if ((address & 0xB0000000u) != 0x80000000u || physical >= 0x1800000u)
    return;
  const unsigned width = width_and_write & 0xFFu;
  const unsigned bit = 1u << s_trace_call;
  const bool write = (width_and_write & 0x100u) != 0;
  for (unsigned block = physical >> kShift; block <= (physical + width - 1) >> kShift && block < kBlocks; ++block)
  {
    if (!s_readers[block] && !s_writers[block])
      s_touched.push_back(block);
    // s_readers holds exposed reads only: a read of a word this call has not written itself.
    if (write)
      s_writers[block] |= bit;
    else if (!(s_writers[block] & bit))
      s_readers[block] |= bit;
  }
}
void TraceFrameEnd()
{
  if (!g_trace_on || s_trace_call < 0)
    return;
  bool any[4] = {};
  for (unsigned block : s_touched)
  {
    const unsigned w = s_writers[block], r = s_readers[block];
    s_writers[block] = s_readers[block] = 0;
    if (!w)
      continue;
    const unsigned all = w | r;
    if ((all & (all - 1)) == 0)
      continue;  // one call only
    // Kind: 0 = inside the calls' shared guest stack, 1 = inside one of the machines' own argument blocks (first
    // argument .. +0x1000 of any call), 2 = other RAM.
    const unsigned address = 0x80000000u + (block << kShift);
    int kind = 2;
    for (int i = 0; i <= s_trace_call; ++i)
    {
      if (address + 4 > s_trace_sp[i] - 0x8000 && address < s_trace_sp[i] + 0x400) { kind = 0; break; }
    }
    auto& c = s_conflicts[block];
    ++c.frames; c.kind = kind; c.readers |= r; c.writers |= w;
    // read-write: some call reads (exposed) a word that an earlier call of the frame wrote
    bool carried = false;
    for (int j = 1; j <= s_trace_call && !carried; ++j)
      carried = ((r >> j) & 1u) && (w & ((1u << j) - 1u));
    if (carried) ++c.rw;
    if (w & (w - 1)) ++c.ww;
    any[kind] = true;
  }
  s_touched.clear();
  ++s_trace_frames;
  for (int k = 0; k < 3; ++k) s_kind_frames[k] += any[k];
  s_frames_with_conflict += any[2];
  s_trace_call = -1;
}
void PrintTrace()
{
  if (!g_trace_on)
    return;
  std::fprintf(stderr, "[trace] frames=%llu calls=%llu frames with a conflict on the shared stack=%llu, outside it=%llu\n",
               s_trace_frames, s_trace_calls, s_kind_frames[0], s_kind_frames[2]);
  std::vector<std::pair<unsigned, Conflict>> blocks(s_conflicts.begin(), s_conflicts.end());
  std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) { return a.second.frames > b.second.frames; });
  unsigned long long stack_blocks = 0, other_blocks = 0;
  for (const auto& b : blocks) (b.second.kind == 0 ? stack_blocks : other_blocks)++;
  std::fprintf(stderr, "[trace] conflicting blocks: %llu on the stack, %llu elsewhere\n", stack_blocks, other_blocks);
  std::fprintf(stderr, "[trace] external interrupts taken inside fn_1_83858 calls: %llu; accesses made before the "
               "interrupted code resumed: %llu (%s)\n", s_irqs_in_calls, s_irq_accesses,
               s_trace_noirq ? "left out" : "counted");
  // Stack words that a call read before writing them itself after an earlier call of the frame had written them
  // (state carried through the shared guest stack), relative to the stack pointer at the last call's entry.
  unsigned long long stack_carried = 0;
  for (const auto& b : blocks)
  {
    if (b.second.kind != 0 || b.second.rw == 0)
      continue;
    if (++stack_carried <= 200)
      std::fprintf(stderr, "[trace-stack] sp%+d frames=%llu read-write=%llu writers=%08x readers=%08x\n",
                   static_cast<int>(0x80000000u + (b.first << kShift) - s_last_sp), b.second.frames, b.second.rw,
                   b.second.writers, b.second.readers);
  }
  std::fprintf(stderr, "[trace] stack words carried from one call to a later one: %llu\n", stack_carried);
  int shown = 0;
  for (const auto& b : blocks)
  {
    if (b.second.kind == 0)
      continue;
    std::fprintf(stderr, "[trace-block] %08x frames=%llu write-write=%llu read-write=%llu writers=%08x readers=%08x\n",
                 0x80000000u + (b.first << kShift), b.second.frames, b.second.ww, b.second.rw, b.second.writers,
                 b.second.readers);
    if (++shown >= 3000)
      break;
  }
}
void PrintTaskList();
void PrintCensus()
{
  PrintTrace();
  PrintTaskList();
  if (g_gather_flushes)
    std::fprintf(stderr, "[staticrecomp] gather-pipe flushes asked for by generated code: %llu\n", g_gather_flushes);
  if (!g_census_on)
    return;
  std::fprintf(stderr, "[census] whole run: gather-pipe writes=%llu, %.1f ns each inside the hook, %.3f ms in all; other "
               "hardware writes=%llu, %.1f ns each\n", g_gather_writes,
               g_gather_writes ? g_gather_tsc / 3.394 / g_gather_writes : 0.0, g_gather_tsc / 3394000.0,
               g_hwwrite_count, g_hwwrite_count ? g_hwwrite_tsc / 3.394 / g_hwwrite_count : 0.0);
  for (int w = 0; w < 2; ++w)
  {
    std::vector<std::pair<unsigned int, std::pair<unsigned long long, unsigned int>>> sites(
        s_census_access[w].begin(), s_census_access[w].end());
    std::sort(sites.begin(), sites.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    for (std::size_t i = 0; i < sites.size() && i < 60; ++i)
      std::fprintf(stderr, "[census-access] %s pc=%08x count=%llu last_ea=%08x\n", w ? "write" : "read",
                   sites[i].first, sites[i].second.first, sites[i].second.second);
  }
  static const char* const names[3] = {"fn_1_83858", "fn_1_850C4", "fn_1_C4ABC"};
  for (int k = 0; k < 3; ++k)
  {
    const auto& c = g_census[k];
    std::fprintf(stderr,
                 "[census] %s calls=%llu dispatches=%llu mmu_reads=%llu l1_reads=%llu gather_writes=%llu "
                 "mmu_writes=%llu l1_writes=%llu fallback=%llu spr_reads=%llu spr_writes=%llu exceptions=%llu "
                 "advances=%llu\n",
                 names[k], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
    std::fprintf(stderr, "[census] %s mmu read time: %.1f ns per read, %.2f us per task call\n", names[k],
                 c[2] ? g_census_read_tsc[k] / 3.394 / c[2] : 0.0, c[0] ? g_census_read_tsc[k] / 3394.0 / c[0] : 0.0);
    std::vector<std::pair<unsigned int, unsigned long long>> pcs(s_census_pcs[k].begin(), s_census_pcs[k].end());
    std::sort(pcs.begin(), pcs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (std::size_t i = 0; i < pcs.size() && i < 400; ++i)
      std::fprintf(stderr, "[census-pc] %s %08x %llu\n", names[k], pcs[i].first, pcs[i].second);
  }
}

// FZGX_POKE8=ADDR:VALUE[@FROM-TO][,...] (address and value hex, fields decimal): bytes written to main RAM at every
// field, or at fields FROM..TO of the run. FZGX_POKE8_TASK is the same list, written at the start of every
// per-machine task call instead (after the game has read the pads). For demonstration clips only (the game's own
// debug byte and debug pad, by which the player's machine is handed to the AI); never in a gate run.
static unsigned long s_poke_field = 0;
static Core::System* s_poke_system = nullptr;
static void ApplyPokes(Core::System& system, const char* poke, unsigned long field)
{
  s_poke_system = &system;
  for (const char* p = poke; p && *p;)
  {
    char* end = nullptr;
    const unsigned long address = std::strtoul(p, &end, 16);
    if (end == p || *end != ':')
      break;
    const unsigned long value = std::strtoul(end + 1, &end, 16);
    unsigned long from = 0, to = ~0ul;
    if (*end == '@')
    {
      from = std::strtoul(end + 1, &end, 10);
      to = *end == '-' ? std::strtoul(end + 1, &end, 10) : from;
    }
    auto& memory = system.GetMemory();
    if (field >= from && field <= to && address >= 0x80000000ul &&
        address - 0x80000000ul < memory.GetRamSizeReal())
      memory.GetRAM()[address - 0x80000000ul] = static_cast<u8>(value);
    p = *end == ',' ? end + 1 : end;
  }
}
// FZGX_POKE8_TASK at the start of a per-machine task call: from the run loop on native code, from the task runner's
// hook under the JIT (which needs FZGX_TASKPROF, as the hooks do). Its FROM-TO are fields of the run, or race frames
// with FZGX_POKE8_TASK_BY=frame: a field can hold no task call or two (under the JIT the main loop's phase against
// the VI wanders), so only a race frame names the same call in two runs.
static void ApplyTaskPokes(u32 task)
{
  if (task != s_text + 0x83858u || !s_poke_system)
    return;
  static const char* task_poke = RunSettings::Get("FZGX_POKE8_TASK");
  if (!task_poke)
    return;
  static const char* by = RunSettings::Get("FZGX_POKE8_TASK_BY");
  unsigned long index = s_poke_field;
  if (by && by[0] == 'f')
  {
    const u8* ram = s_poke_system->GetMemory().GetRAM();
    index = BE32(ram + (s_text - TEXT_OFF - 0x80000000u) + 0x1BA040u + 0x50A4u);
  }
  ApplyPokes(*s_poke_system, task_poke, index);
}
// FZGX_TASKLIST=1: every function the task runner calls (a dispatch whose return address is the runner's), with
// its call count, printed at shutdown. The roots for closed bodies beyond the physics come from this list.
static std::unordered_map<unsigned int, unsigned long long> s_task_entries;
static bool s_task_list = false;  // ReadSettings()
// The settings above that the native core tests on its fast paths, read once when that core starts: after the
// host program has given this run's settings (Common/RunSettings.h), never before main().
void ReadSettings()
{
  const char* trace = RunSettings::Get("FZGX_TRACE");
  g_census_on = RunSettings::Get("FZGX_CENSUS") != nullptr;
  g_trace_on = trace != nullptr;
  s_trace_noirq = trace && std::string(trace) == "noirq";
  s_task_list = RunSettings::Get("FZGX_TASKLIST") != nullptr;
}
void PrintTaskList()
{
  for (const auto& e : s_task_entries)
    std::fprintf(stderr, "[task-entry] %08x %llu\n", e.first, e.second);
}
void OnNativeDispatch(unsigned int address, unsigned int lr)
{
  if (s_in_irq && address == s_irq_pc && g_trace_r1 == s_irq_sp)
    s_in_irq = false;
  if (!s_text)
    return;
  if (s_task_list && lr == s_text + 0x4120u && address != s_text + 0x4120u)
    ++s_task_entries[address];
  ApplyTaskPokes(address);
  if (lr == s_text + 0x4120u)
    TaskPointDump();
  if (address == s_text + 0x4120u)
    TaskEnd();
  else if (!s_task_target && (address == s_text + 0x83858u || address == s_text + 0x850C4u ||
                              address == s_text + 0xC4ABCu))
  {
    if (g_census_on || g_trace_on)
    {
      g_census_task = address == s_text + 0x83858u ? 0 : address == s_text + 0x850C4u ? 1 : 2;
      ++g_census[g_census_task][0];
      if (g_census_task == 0)
        TraceCallBegin(g_trace_r1, g_trace_r3);
      else if (g_census_task == 1)
        TraceFrameEnd();  // the big task follows the 30 per-machine calls
    }
    s_task_target = address;
    s_task_c0 = Ticks();
    s_mxcsr = _mm_getcsr();
    g_task_emu_tsc = 0;
    g_task_active = true;
    s_task_t0 = __rdtsc();
  }
}

// FZGX_PAD="neutral" or "FIELD:BUTTONS_HEX[:SX:SY[:CX:CY]],..." -- port 0 is driven only by the script,
// so inputs land on exact fields in every run. A step lasts until the next one.
void OverridePad(int port, GCPadStatus* pad)
{
  if (!s_pad_checked)
  {
    s_pad_checked = true;
    const char* script = RunSettings::Get("FZGX_PAD");
    if (script && *script)
    {
      s_pad_active = true;
      const char* p = script;
      while (*p)
      {
        char* end = nullptr;
        PadStep step{std::strtoull(p, &end, 10), 0, 0x80, 0x80, 0x80, 0x80};
        if (end == p)
          break;
        p = end;
        if (*p == ':')
        {
          step.buttons = static_cast<u16>(std::strtoul(p + 1, &end, 16));
          p = end;
          if (*p == ':')
          {
            step.sx = static_cast<u8>(std::strtoul(p + 1, &end, 10));
            p = end;
            if (*p == ':')
            {
              step.sy = static_cast<u8>(std::strtoul(p + 1, &end, 10));
              p = end;
              if (*p == ':')
              {
                step.cx = static_cast<u8>(std::strtoul(p + 1, &end, 10));
                p = end;
                if (*p == ':')
                {
                  step.cy = static_cast<u8>(std::strtoul(p + 1, &end, 10));
                  p = end;
                }
              }
            }
          }
        }
        s_pad_script.push_back(step);
        while (*p == ',' || *p == ' ')
          ++p;
        if (*p && (*p < '0' || *p > '9'))
          break;
      }
    }
  }
  if (!s_pad_active || port != 0)
    return;
  PadStep cur{0, 0, 0x80, 0x80, 0x80, 0x80};
  for (const PadStep& step : s_pad_script)
    if (step.field <= s_field)
      cur = step;
  if (cur.buttons == 0xFFFF)  // hand port 0 back to the normal input path
    return;
  GCPadStatus out{};
  out.button = cur.buttons;
  out.stickX = cur.sx;
  out.stickY = cur.sy;
  out.substickX = cur.cx;
  out.substickY = cur.cy;
  out.analogA = (cur.buttons & PAD_BUTTON_A) ? 0xFF : 0;
  out.analogB = (cur.buttons & PAD_BUTTON_B) ? 0xFF : 0;
  out.triggerLeft = (cur.buttons & PAD_TRIGGER_L) ? 0xFF : 0;
  out.triggerRight = (cur.buttons & PAD_TRIGGER_R) ? 0xFF : 0;
  out.isConnected = true;
  *pad = out;
}

#ifdef _WIN32
// FZGX_SAMPLE=<path>: sampling profiler of the CPU thread (RIP every ~1 ms while a race runs), written as
// "base,<exe>,<recomp dll>" then one RIP per line.
namespace
{
std::atomic<bool> s_sampling{false};
void StartSampler()
{
  static bool started = false;
  const char* path = RunSettings::Get("FZGX_SAMPLE");
  if (started || !path || !*path)
    return;
  started = true;
  HANDLE target = nullptr;
  DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &target,
                  THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
  std::string out = path;
  std::thread([target, out] {
    timeBeginPeriod(1);
    std::FILE* f = std::fopen(out.c_str(), "w");
    if (!f)
      return;
    std::fprintf(f, "base,%llx,%llx\n", (unsigned long long)GetModuleHandleA(nullptr),
                 (unsigned long long)GetModuleHandleA("gGFZE01_recomp.dll"));
    for (;;)
    {
      Sleep(1);
      if (!s_sampling.load(std::memory_order_relaxed) || (RunSettings::Get("FZGX_SAMPLE_TASKS") && !s_task_target))
        continue;
      // FZGX_SAMPLE_PHYSICS: only inside the three physics tasks (the JIT's hooks see every task).
      if (RunSettings::Get("FZGX_SAMPLE_PHYSICS"))
      {
        const u32 off = s_task_target - s_text;
        if (off != 0x83858u && off != 0x850C4u && off != 0xC4ABCu)
          continue;
      }
      if (SuspendThread(target) == (DWORD)-1)
        break;
      CONTEXT c{};
      c.ContextFlags = CONTEXT_CONTROL;
      const bool ok = GetThreadContext(target, &c);
      ResumeThread(target);
      if (ok)
      {
        std::fprintf(f, "%llx\n", (unsigned long long)c.Rip);
        static int n = 0;
        if ((++n & 1023) == 0)
          std::fflush(f);
      }
    }
  }).detach();
}
}  // namespace
#endif

// ---- the slow-second line ---------------------------------------------------------------------------------------------
// Always on, for a player's report: one line on standard error (the launcher has the runner write that to
// <user folder>/runner.log) for every second of real time in which the game made fewer than 55 fields a second, or,
// with a race running through the whole second, fewer than 55 race frames or pictures. The line says what is at hand
// for nothing, so that "it slowed down here" comes with numbers: the place in the game, how the second was spent
// (this thread, the whole process, the whole PC) and what the CPU core did that is slow by nature (trips through
// the run loop, steps on the fallback JIT, interpreted instructions, chunks failing their check).
// Cost: per field one clock read, three reads of guest memory and a dozen additions; once a second three system
// calls. FZGX_SLOW_LOG=0 turns it off (for the measurement that shows it costs nothing); FZGX_SLOW_LOG=all writes
// the line for every second.
namespace
{
struct SlowSecond
{
  int mode = -1;  // -1 not read yet, 0 off, 1 slow seconds, 2 every second
  bool started = false;
  u64 start_us = 0, last_us = 0, longest_us = 0;
  u32 fields = 0;
  u32 last_race_frame = 0, race_frames = 0, first_advance = 0, last_advance = 0;
  bool racing = true;                       // the game's mode was "race" at every field of the second
  u32 game_mode = 0, game_submode = 0;
  u64 trips = 0, idles = 0, seen_trips = 0, seen_idles = 0;
  unsigned long long core[4] = {};
  u64 thread_100ns = 0, process_100ns = 0, pc_idle_100ns = 0, pc_all_100ns = 0;
} s_slow;

#ifdef _WIN32
u64 Filetime(const FILETIME& t)
{
  return (u64(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}
#endif

// CPU time so far: this thread's, the whole process's, and the PC's idle and total time over all its processors.
void CpuTimes(u64* thread, u64* process, u64* pc_idle, u64* pc_all)
{
  *thread = *process = *pc_idle = *pc_all = 0;
#ifdef _WIN32
  FILETIME created, exited, kernel, user, idle;
  if (GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
    *thread = Filetime(kernel) + Filetime(user);
  if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user))
    *process = Filetime(kernel) + Filetime(user);
  if (GetSystemTimes(&idle, &kernel, &user))  // the kernel's time includes the idle time
  {
    *pc_idle = Filetime(idle);
    *pc_all = Filetime(kernel) + Filetime(user);
  }
#endif
}

void SlowSecondField(Core::System& system)
{
  SlowSecond& s = s_slow;
  if (s.mode < 0)
  {
    const char* v = RunSettings::Get("FZGX_SLOW_LOG");
    s.mode = !v ? 1 : v[0] == '0' ? 0 : v[0] == 'a' ? 2 : 1;
  }
  if (s.mode == 0)
    return;
  const u64 now = QpcUs();
  // the place in the game: main.rel's mode words and its race frame counter
  u32 mode = 0, submode = 0, race_frame = 0;
  const u8* ram = system.GetMemory().GetRAM();
  const u32 off = (ram ? BE32(ram + 0x30C8) : 0) - 0x80000000u;
  if (ram && off < 0x01800000u && off + 0x1BA040u + 0x8FEA4u <= system.GetMemory().GetRamSizeReal())
  {
    const u8* bss = ram + off + 0x1BA040u;
    mode = (u32(bss[0x960]) << 8) | bss[0x961];
    submode = (u32(bss[0x962]) << 8) | bss[0x963];
    race_frame = mode == 2 ? BE32(bss + 0x50A4) : 0;  // the counter means something only in a race
  }
  if (!s.started)
  {
    const int keep = s.mode;
    s = SlowSecond{};
    s.mode = keep;
    s.started = true;
    s.start_us = s.last_us = now;
    s.last_race_frame = race_frame;
    s.seen_trips = g_dispatches;
    s.seen_idles = g_idles;
    if (g_core_counters)
      g_core_counters(s.core);
    CpuTimes(&s.thread_100ns, &s.process_100ns, &s.pc_idle_100ns, &s.pc_all_100ns);
    return;
  }
  ++s.fields;
  s.longest_us = std::max(s.longest_us, now - s.last_us);
  s.last_us = now;
  if (race_frame != s.last_race_frame)
  {
    s.race_frames += race_frame > s.last_race_frame ? race_frame - s.last_race_frame : 0;
    if (s.first_advance == 0)
      s.first_advance = s.fields;
    s.last_advance = s.fields;
    s.last_race_frame = race_frame;
  }
  s.racing = s.racing && mode == 2;
  s.game_mode = mode;
  s.game_submode = submode;
  s.trips += g_dispatches - s.seen_trips;
  s.idles += g_idles - s.seen_idles;
  s.seen_trips = g_dispatches;
  s.seen_idles = g_idles;
  if (now - s.start_us < 1000000)
    return;

  // a second has passed
  const double seconds = (now - s.start_us) / 1e6;
  const double pictures = system.GetPerfMetrics().GetFPS();
  // a race counts when it ran through the whole second: it was moving at the second's start and at its end (a pause
  // or the finish inside the second is not a slow second)
  const bool race_ran = s.racing && s.first_advance != 0 && s.first_advance <= 3 && s.last_advance + 3 > s.fields;
  const bool slow = s.fields < 55.0 * seconds || (race_ran && (s.race_frames < 55.0 * seconds || pictures < 55.0));
  if (slow || s.mode == 2)
  {
    unsigned long long core[4] = {};
    if (g_core_counters)
      g_core_counters(core);
    u64 thread, process, pc_idle, pc_all;
    CpuTimes(&thread, &process, &pc_idle, &pc_all);
    const double pc_busy = pc_all > s.pc_all_100ns ?
        100.0 * (1.0 - double(pc_idle - s.pc_idle_100ns) / double(pc_all - s.pc_all_100ns)) : 0.0;
    char clock[16] = "";
#ifdef _WIN32
    SYSTEMTIME t;
    GetLocalTime(&t);
    std::snprintf(clock, sizeof(clock), "%02u:%02u:%02u", t.wHour, t.wMinute, t.wSecond);
#endif
    std::fprintf(stderr,
                 "[slow] %s %u fields in %.2f s (longest %.0f ms), pictures %.1f/s; game mode %u/%u, race frame %u "
                 "(+%u); CPU thread %.2f s, all threads %.2f s, PC %.0f%% busy; run-loop trips %llu (idle %llu)",
                 clock, s.fields, seconds, s.longest_us / 1000.0, pictures, s.game_mode, s.game_submode, race_frame,
                 s.race_frames, (thread - s.thread_100ns) / 1e7, (process - s.process_100ns) / 1e7, pc_busy,
                 (unsigned long long)s.trips, (unsigned long long)s.idles);
    if (g_core_counters)
      std::fprintf(stderr, ", fallback steps %llu, interpreted %llu, chunk checks %llu, chunks failing %llu",
                   core[0] - s.core[0], core[1] - s.core[1], core[3] - s.core[3], core[2]);
    std::fprintf(stderr, "%s\n", slow ? "" : " (not slow)");
    std::fflush(stderr);
  }
  s.started = false;  // the next field starts the next second (and reads the counters anew)
}
}  // namespace

// The pacing log: see FzgxProbe.h.
bool g_pace_on = false;
namespace
{
std::mutex s_pace_mutex;      // every record is appended under it; held for a copy only
std::string s_pace_records;
std::string s_pace_path;
u64 s_pace_cycles = 0, s_pace_field = 0;
unsigned long long s_pace_core[4] = {};
u64 s_pace_trips = 0, s_pace_idles = 0, s_pace_jit = 0;
struct PaceEvents
{
  u64 count = 0;
  unsigned int first[4] = {};
} s_pace_events[3];
GCPadStatus s_pace_pad = {};
u32 s_pace_pad_reads = 0;
std::atomic<int> s_pace_rumble[4] = {{-1}, {-1}, {-1}, {-1}};

void PaceRecord(const char* format, ...)
{
  char line[512];
  va_list args;
  va_start(args, format);
  const int n = std::vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  if (n <= 0)
    return;
  std::lock_guard<std::mutex> guard(s_pace_mutex);
  s_pace_records.append(line, std::min<std::size_t>(static_cast<std::size_t>(n), sizeof(line) - 1));
}

void PaceWrite()
{
  std::string records;
  {
    std::lock_guard<std::mutex> guard(s_pace_mutex);
    records.swap(s_pace_records);
  }
  if (records.empty())
    return;
  if (std::FILE* file = std::fopen(s_pace_path.c_str(), "ab"))
  {
    std::fwrite(records.data(), 1, records.size(), file);
    std::fclose(file);
  }
}

void PaceField(Core::System& system)
{
  if (s_pace_field == 0)
  {
    const char* reuse = RunSettings::Get("FZGX_REINTERPRET_REUSE");
    PaceRecord("# graphics backend %s; reinterpreted copies reused: %s\n",
               Config::Get(Config::MAIN_GFX_BACKEND).c_str(), reuse && reuse[0] == '0' ? "no" : "yes");
  }
  const u64 now = QpcUs();
  const u64 cycles = ThreadCpu100ns();
  u32 mode = 0, submode = 0, race_frame = 0, flags = 0, energy = 0;
  const u8* ram = system.GetMemory().GetRAM();
  const u32 ram_size = system.GetMemory().GetRamSizeReal();
  const u32 off = (ram ? BE32(ram + 0x30C8) : 0) - 0x80000000u;
  if (ram && off < 0x01800000u && off + 0x1BA040u + 0x8FEA4u <= ram_size)
  {
    const u8* bss = ram + off + 0x1BA040u;
    mode = (u32(bss[0x960]) << 8) | bss[0x961];
    submode = (u32(bss[0x962]) << 8) | bss[0x963];
    race_frame = BE32(bss + 0x50A4);
    const u32 rp = BE32(bss + 0x6D838) - 0x80000000u;
    if (rp < ram_size && rp + 0x620u <= ram_size)
    {
      flags = BE32(ram + rp);
      energy = BE32(ram + rp + 0x184);
    }
  }
  unsigned long long core[4] = {};
  if (g_core_counters)
    g_core_counters(core);
  char events[3][64];
  for (int k = 0; k < 3; ++k)
  {
    PaceEvents& e = s_pace_events[k];
    int n = 0;
    events[k][0] = 0;
    for (u64 i = 0; i < e.count && i < 4; ++i)
      n += std::snprintf(events[k] + n, sizeof(events[k]) - n, "%s%08x", i ? " " : "", e.first[i]);
    e = PaceEvents{};
  }
  // the probe's own file resets these three at every field; without it they count up
  const u64 trips = g_dispatches >= s_pace_trips ? g_dispatches - s_pace_trips : g_dispatches;
  const u64 idles = g_idles >= s_pace_idles ? g_idles - s_pace_idles : g_idles;
  const u64 jit = g_jit_tsc >= s_pace_jit ? g_jit_tsc - s_pace_jit : g_jit_tsc;
  s_pace_trips = s_file ? 0 : g_dispatches;
  s_pace_idles = s_file ? 0 : g_idles;
  s_pace_jit = s_file ? 0 : g_jit_tsc;
  PaceRecord("F,%llu,%llu,%llu,%u,%u,%u,%08x,%08x,%04x,%d,%d,%d,%d,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%s,%s,%s\n",
             (unsigned long long)now, (unsigned long long)s_pace_field, (unsigned long long)(cycles - s_pace_cycles),
             mode, submode, race_frame, flags, energy, s_pace_pad.button, s_pace_pad.stickX, s_pace_pad.stickY,
             s_pace_pad.substickX, s_pace_pad.substickY, s_pace_pad.triggerLeft, s_pace_pad.triggerRight,
             s_pace_pad_reads, core[0] - s_pace_core[0], core[1] - s_pace_core[1], core[3] - s_pace_core[3], core[2],
             (unsigned long long)trips, (unsigned long long)idles, (unsigned long long)jit, events[0], events[1],
             events[2]);
  for (int i = 0; i < 4; ++i)
    s_pace_core[i] = core[i];
  s_pace_pad_reads = 0;
  ++s_pace_field;
  s_pace_cycles = ThreadCpu100ns();
}
}  // namespace

void StartPaceLog(const char* path)
{
  if (g_pace_on || !path || !*path)
    return;
  s_pace_path = path;
  std::FILE* file = std::fopen(path, "wb");
  if (!file)
    return;
  SYSTEMTIME t;
  GetLocalTime(&t);
  std::fprintf(file,
               "# pacing log; started %04u-%02u-%02u %02u:%02u:%02u.%03u local time, when the clock of every record read %llu us\n"
               "# F,end_us,field,cpu_thread_cycles,mode,submode,race_frame,racer0_flags,racer0_energy,pad1_buttons,stick_x,stick_y,"
               "c_x,c_y,trigger_l,trigger_r,pad1_reads,fallback_steps,interpreted,chunk_checks,chunks_failing,run_loop_trips,"
               "idles,fallback_jit_cycles,fallback PCs,interpreted PCs,checked chunks\n"
               "#   (cycles: the processor's time stamp counter; the C records give its rate. pad1_buttons: 0001 left, 0002 "
               "right, 0004 down, 0008 up, 0010 Z, 0020 R, 0040 L, 0100 A, 0200 B, 0400 X, 0800 Y, 1000 Start)\n"
               "# P,before_wait_us,after_wait_us,after_present_us,thread_cycles_since_last_present,thread_id,draws,primitives,"
               "vertex_bytes,uniform_bytes,textures_created,textures_uploaded,textures_alive   (draws to uniform_bytes: the "
               "frame of the last copy to the display before the present, whole)\n"
               "# T,start_us,duration_us,what,thread_id   (a stretch of graphics work of 2 ms or more)\n"
               "# X,time_us,duration_us,what,width,height,layers,levels,format,samples,during,address   (a texture made or given up by the backend)\n"
               "# N,time_us,cache_entries,pooled_textures,invalidated_this_frame   (the texture cache at a frame's end)\n"
               "# E,start_us,duration_us,to_texture,to_memory,xfb,depth,width,height,format,address   (a copy out of the frame buffer)\n"
               "# S,start_us,duration_us,what,thread_id\n"
               "# R,time_us,port (0 = port 1),strength\n"
               "# C,time_us,time_stamp_counter\n"
               "# G,start_us,wait_ns,in_flight   (command-list reads turned off: the wait for the GPU thread's pass; "
               "in_flight 1 when one was running)\n",
               t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, (unsigned long long)QpcUs());
  std::fclose(file);
  g_pace_on = true;
  std::atexit(PaceWrite);
  std::thread([] {
    for (;;)
    {
      PaceRecord("C,%llu,%llu\n", (unsigned long long)QpcUs(), (unsigned long long)Tsc());
      PaceWrite();
      Sleep(3000);
    }
  }).detach();
}

unsigned long long PaceClock()
{
  return g_pace_on ? QpcUs() : 0;
}

namespace
{
// The drawing counts of whole frames (PaceFrameCountsEnd): what the statistics counted in frames they have started
// again since, and the running total at the last copy to the display (both on the thread that draws), and the
// whole frame's at that copy (read by the thread that presents).
struct DrawCounts
{
  long long draws = 0, prims = 0, vertex = 0, uniform = 0;
};
DrawCounts s_carried, s_at_copy, s_whole;
std::mutex s_whole_mutex;
std::atomic<unsigned long long> s_cpu_slept_ns{0};

DrawCounts Running()
{
  const auto& f = g_stats.this_frame;
  return {s_carried.draws + f.num_draw_calls, s_carried.prims + f.num_prims + f.num_dl_prims,
          s_carried.vertex + f.bytes_vertex_streamed, s_carried.uniform + f.bytes_uniform_streamed};
}
}  // namespace

void PaceFrameCountsEnd()
{
  if (g_pace_on)
    s_carried = Running();
}

void CountCpuThreadSleep(unsigned long long ns)
{
  s_cpu_slept_ns.fetch_add(ns, std::memory_order_relaxed);
}

void PaceReadsOff(unsigned long long start_us, unsigned long long wait_ns, bool in_flight)
{
  if (!g_pace_on)
    return;
  PaceRecord("G,%llu,%llu,%d\n", start_us, wait_ns, in_flight ? 1 : 0);
}

unsigned long long CpuThreadSleptNs()
{
  return s_cpu_slept_ns.load(std::memory_order_relaxed);
}

void PacePresent(unsigned long long before_wait_us, unsigned long long after_wait_us)
{
  if (!g_pace_on)
    return;
  static thread_local u64 last_cycles = 0;
  const u64 cycles = ThreadCpu100ns();
  static thread_local int last_created = 0, last_uploaded = 0;
  DrawCounts frame;
  {
    std::lock_guard<std::mutex> guard(s_whole_mutex);
    frame = s_whole;
  }
  PaceRecord("P,%llu,%llu,%llu,%llu,%lu,%lld,%lld,%lld,%lld,%d,%d,%d\n", before_wait_us, after_wait_us,
             (unsigned long long)QpcUs(), (unsigned long long)(cycles - last_cycles), GetCurrentThreadId(),
             frame.draws, frame.prims, frame.vertex, frame.uniform, g_stats.num_textures_created - last_created,
             g_stats.num_textures_uploaded - last_uploaded, g_stats.num_textures_alive);
  last_created = g_stats.num_textures_created;
  last_uploaded = g_stats.num_textures_uploaded;
  last_cycles = cycles;
}

void PaceCompiled(const char* what, unsigned long long start_us)
{
  if (!g_pace_on)
    return;
  PaceRecord("S,%llu,%llu,%s,%lu\n", start_us, (unsigned long long)QpcUs() - start_us, what, GetCurrentThreadId());
}

void PaceSpan(const char* what, unsigned long long start_us, unsigned long long min_us)
{
  if (!g_pace_on)
    return;
  const u64 duration = QpcUs() - start_us;
  if (duration >= min_us)
    PaceRecord("T,%llu,%llu,%s,%lu\n", start_us, (unsigned long long)duration, what, GetCurrentThreadId());
}

void PaceCopy(unsigned long long start_us, bool to_texture, bool to_memory, bool xfb, bool depth, unsigned int width,
              unsigned int height, unsigned int format, unsigned int address)
{
  if (!g_pace_on)
    return;
  if (xfb)
  {
    const DrawCounts now = Running();
    const DrawCounts whole{now.draws - s_at_copy.draws, now.prims - s_at_copy.prims, now.vertex - s_at_copy.vertex,
                           now.uniform - s_at_copy.uniform};
    s_at_copy = now;
    std::lock_guard<std::mutex> guard(s_whole_mutex);
    s_whole = whole;
  }
  PaceRecord("E,%llu,%llu,%d,%d,%d,%d,%u,%u,%u,%08x\n", start_us, (unsigned long long)QpcUs() - start_us, to_texture,
             to_memory, xfb, depth, width, height, format, address);
}

static thread_local const char* s_pace_ctx = "";
static thread_local unsigned int s_pace_ctx_address = 0;
PaceContext::PaceContext(const char* what, unsigned int address) : old_what(s_pace_ctx), old_address(s_pace_ctx_address)
{
  s_pace_ctx = what;
  s_pace_ctx_address = address;
}
PaceContext::~PaceContext()
{
  s_pace_ctx = old_what;
  s_pace_ctx_address = old_address;
}

void PaceTexture(const char* what, unsigned long long start_us, unsigned int width, unsigned int height,
                 unsigned int layers, unsigned int levels, unsigned int format, unsigned int samples)
{
  if (!g_pace_on)
    return;
  PaceRecord("X,%llu,%llu,%s,%u,%u,%u,%u,%u,%u,%s,%08x\n", start_us, (unsigned long long)QpcUs() - start_us, what, width,
             height, layers, levels, format, samples, s_pace_ctx, s_pace_ctx_address);
}

void PaceCacheSize(unsigned long long entries, unsigned long long pooled, unsigned int invalidated)
{
  if (g_pace_on)
    PaceRecord("N,%llu,%llu,%llu,%u\n", (unsigned long long)QpcUs(), entries, pooled, invalidated);
}

void PaceRumble(int port, double strength)
{
  if (!g_pace_on || port < 0 || port > 3)
    return;
  const int value = static_cast<int>(strength * 100.0);
  if (s_pace_rumble[port].exchange(value) != value)
    PaceRecord("R,%llu,%d,%d\n", (unsigned long long)QpcUs(), port, value);
}

void PacePad(const GCPadStatus& pad)
{
  s_pace_pad = pad;
  ++s_pace_pad_reads;
}

void PaceEvent(int kind, unsigned int address)
{
  if (!g_pace_on)
    return;
  PaceEvents& e = s_pace_events[kind];
  if (e.count < 4)
    e.first[e.count] = address;
  ++e.count;
}

void OnPadRead(Core::System& system)
{
  ++s_pad_reads;
  s_last_pad_tick = system.GetCoreTiming().GetTicks();
}

void OnNewField(Core::System& system)
{
  SlowSecondField(system);
  if (!s_checked)
    StartPaceLog(RunSettings::Get("FZGX_PACE_LOG"));
  if (g_pace_on)
    PaceField(system);
  ApplyPokes(system, RunSettings::Get("FZGX_POKE8"), s_poke_field);
  ++s_poke_field;
  if (!s_checked)
  {
    s_checked = true;
    const char* path = RunSettings::Get("FZGX_PROBE");
    if (path && *path)
      s_file = std::fopen(path, "w");
    const char* hash = RunSettings::Get("FZGX_PROBE_HASH");
    s_hash = !(hash && hash[0] == '0');
    if (s_file)
      std::fprintf(s_file, "field,ticks,module,mode,submode,race_frame,racers_hash,rel_hash,seat0_hash,"
                           "ext,sim_hash,busy,t83858,n83858,t850C4,n850C4,tC4ABC,nC4ABC,ttasks,ntasks,c83858,c850C4,cC4ABC,ctasks,mxcsr,cpu_cycles,pad_reads,last_pad_tick,tsc,qpc_us,native_tsc,jit_tsc,interp_tsc,dispatches,idles,e83858,e850C4,eC4ABC,etasks\n");
    s_last_end_cpu = ThreadCpu100ns();
  }
  if (!s_file)
    return;
  const u64 start_cpu = ThreadCpu100ns();
  const u64 busy = start_cpu - s_last_end_cpu;
  const u64 ticks = system.GetCoreTiming().GetTicks();
  const u8* ram = system.GetMemory().GetRAM();
  const u32 ram_size = system.GetMemory().GetRamSizeReal();
  u32 module = ram ? BE32(ram + 0x30C8) : 0;
  u32 mode = 0, submode = 0, race_frame = 0;
  u64 racers = 0, rel = 0, seat0 = 0, sim = 0;
  u32 busy_flag = 0;
  const u32 off = module - 0x80000000u;
  // main.rel: .data at +0x16D600 (0x4C9C1 bytes), .bss at +0x1BA040 (0x8FEA4 bytes).
  if (ram && module >= 0x80000000u && off + 0x1BA040u + 0x8FEA4u <= ram_size)
  {
    const u8* bss = ram + off + 0x1BA040u;
    mode = (u32(bss[0x960]) << 8) | bss[0x961];
    submode = (u32(bss[0x962]) << 8) | bss[0x963];
    race_frame = BE32(bss + 0x50A4);
    if (s_hash)
    {
      // main.rel .data+.bss with the game's timebase-derived profiling words zeroed
      static std::vector<u8> buf;
      const u32 len = 0x1BA040u + 0x8FEA4u - 0x16D600u;
      buf.assign(ram + off + 0x16D600u, ram + off + 0x16D600u + len);
      const u32 b0 = 0x1BA040u - 0x16D600u;
      for (u32 o : {0x4u, 0xD90u, 0xDA0u, 0x50D4u, 0x6C854u, 0x6C858u})
        std::memset(&buf[b0 + o], 0, 4);
      std::memset(&buf[b0 + 0x6EA00u], 0, 0x200);
      sim = Hash(buf.data(), len);
      busy_flag = BE32(ram + off + 0x16D600u + 0x2B64u);
      // .bss 0x6D838 holds the pointer to the racer array (30 x 0x620, heap)
      const u32 rp = BE32(bss + 0x6D838) - 0x80000000u;
      if (rp + 30u * 0x620u <= ram_size)
      {
        seat0 = Hash(ram + rp, 0x620);
        racers = Hash(ram + rp + 0x620, 29 * 0x620);
      }
      rel = Hash(ram + off + 0x16D600u, 0x1BA040u + 0x8FEA4u - 0x16D600u);
    }
  }
  {
    // FZGX_DUMP_FIELDS="N,M,...": write main.rel .data+.bss at those fields to <probe>.<N>.bin
    static std::vector<u64> dump_fields;
    static bool dump_checked = false;
    if (!dump_checked)
    {
      dump_checked = true;
      if (const char* d = RunSettings::Get("FZGX_DUMP_FIELDS"))
        for (const char* p = d; *p;)
        {
          char* e = nullptr;
          dump_fields.push_back(std::strtoull(p, &e, 10));
          if (e == p) break;
          p = e;
          while (*p == ',') ++p;
        }
    }
    for (u64 f : dump_fields)
      if (f == s_field && ram && module >= 0x80000000u && off + 0x1BA040u + 0x8FEA4u <= ram_size)
      {
        char name[600];
        std::snprintf(name, sizeof(name), "%s.%llu.bin", RunSettings::Get("FZGX_PROBE"), (unsigned long long)f);
        if (RunSettings::Get("FZGX_DUMP_RAM"))
        {
          char rname[620];
          std::snprintf(rname, sizeof(rname), "%s.%llu.ram", RunSettings::Get("FZGX_PROBE"), (unsigned long long)f);
          if (std::FILE* r = std::fopen(rname, "wb"))
          {
            std::fwrite(ram, 1, ram_size, r);
            std::fclose(r);
          }
        }
        if (std::FILE* out = std::fopen(name, "wb"))
        {
          std::fwrite(ram + off + 0x16D600u, 1, 0x1BA040u + 0x8FEA4u - 0x16D600u, out);
          const u32 rp = BE32(ram + off + 0x1BA040u + 0x6D838) - 0x80000000u;
          if (rp + 30u * 0x620u <= ram_size)
            std::fwrite(ram + rp, 1, 30 * 0x620, out);
          std::fclose(out);
        }
      }
  }
  char ext_str[400];
  {
    int n = 0;
    for (int i = 0; i < 32; ++i)
      if (g_ext[i])
        n += std::snprintf(ext_str + n, sizeof(ext_str) - n, "%c%X:%llu ", i < 16 ? 'r' : 'w', i & 15, g_ext[i]);
    ext_str[n] = 0;
    for (auto& v : g_ext) v = 0;
  }
  std::fprintf(s_file, "%llu,%llu,%08x,%u,%u,%u,%016llx,%016llx,%016llx,%s,%016llx,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
               (unsigned long long)s_field, (unsigned long long)ticks, module, mode, submode,
               race_frame, (unsigned long long)racers, (unsigned long long)rel, (unsigned long long)seat0, ext_str, (unsigned long long)sim, busy_flag, s_tt[0], s_tn[0], s_tt[1], s_tn[1], s_tt[2], s_tn[2], s_tt[3], s_tn[3], s_tc[0], s_tc[1], s_tc[2], s_tc[3], (unsigned long long)s_mxcsr,
               (unsigned long long)busy, (unsigned long long)s_pad_reads,
               (unsigned long long)s_last_pad_tick, (unsigned long long)Tsc(),
               (unsigned long long)QpcUs(), g_native_tsc, g_jit_tsc, g_interp_tsc, g_dispatches, g_idles, s_te[0], s_te[1], s_te[2], s_te[3]);
  g_native_tsc = g_jit_tsc = g_interp_tsc = g_dispatches = g_idles = 0;
  s_slow.seen_trips = s_slow.seen_idles = 0;  // the slow-second line counts on from the reset
  for (int k = 0; k < 4; ++k) s_tt[k] = s_tn[k] = s_tc[k] = s_te[k] = 0;
#ifdef _WIN32
  StartSampler();
  s_sampling = mode == 2 && race_frame > 0;
#endif
  if (module >= 0x80000000u && mode == 2 && !s_hooks && RunSettings::Get("FZGX_TASKPROF"))
  {
    s_text = module + TEXT_OFF;
    const u32 call = BE32(ram + s_text - 0x80000000u + 0x411C), ret = BE32(ram + s_text - 0x80000000u + 0x4120);
    if (call == 0x4E800421 && ret == 0x38600005)
    {
      HLE::Patch(system, s_text + 0x411C, "FzgxProfTaskStart");
      HLE::Patch(system, s_text + 0x4120, "FzgxProfTaskEnd");
      s_hooks = true;
    }
  }
  if ((s_field & 63) == 0)
    std::fflush(s_file);
  ++s_field;
  s_pad_reads = 0;
  s_last_end_cpu = ThreadCpu100ns();
}
}  // namespace FzgxProbe
