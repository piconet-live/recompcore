// Spike probe (not upstream). SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

namespace Core
{
class System;
}

struct GCPadStatus;
namespace Core { class CPUThreadGuard; }

namespace FzgxProbe
{
void OverridePad(int port, GCPadStatus* pad);
// This run's census, access trace and task list settings, read when the native core starts.
void ReadSettings();
void OnNewField(Core::System& system);
void OnPadRead(Core::System& system);
void HLE_TaskStart(const Core::CPUThreadGuard&);
void HLE_TaskEnd(const Core::CPUThreadGuard&);
// native core: called for every dispatch address (linked)
void OnNativeDispatch(unsigned int address, unsigned int lr);
extern unsigned long long g_ext[32];
// Time inside CoreTiming::Advance (events: GPU FIFO work, VI, audio) while a task is being timed.
extern unsigned long long g_task_emu_tsc;
extern bool g_task_active;
// Census of what leaves generated code while one of the three physics tasks runs (index 0-2; -1 outside):
// [0] task calls, [1] run-loop dispatches after the first, [2] hardware reads through the MMU, [3] locked-cache
// reads, [4] gather-pipe writes, [5] other hardware writes through the MMU, [6] locked-cache writes, [7] fallback
// instructions, [8] SPR reads, [9] SPR writes, [10] guest exceptions, [11] CoreTiming::Advance calls.
extern int g_census_task;
extern unsigned long long g_census[3][12];
extern bool g_census_on;  // FZGX_CENSUS=1; off by default so timing runs do not pay for it
inline void Census(int what) { if (g_census_task >= 0) ++g_census[g_census_task][what]; }
extern unsigned long long g_gather_flushes;  // pipe flushes asked for by generated code, whole run
extern unsigned long long g_gather_tsc, g_gather_writes, g_hwwrite_tsc, g_hwwrite_count;  // whole run, FZGX_CENSUS=1
extern unsigned long long g_census_read_tsc[3];  // time inside hardware reads through the MMU, per task
void CensusDispatch(unsigned int address);
void CensusAccess(unsigned int pc, unsigned int ea, bool write);
void PrintCensus();
// Access trace (a module built with DOLRECOMP_MEM_TRACE=1 and FZGX_TRACE=1): which 64-byte blocks of RAM each
// fn_1_83858 call of a frame reads and writes, and where two calls of the same frame touch the same block with at
// least one of them writing.
void TraceAccess(unsigned int address, unsigned int width_and_write);
void TraceCallBegin(unsigned int stack_pointer, unsigned int first_argument);
void TraceFrameEnd();
void PrintTrace();
// The run loop saw the guest at the external-interrupt vector: accesses until the interrupted code resumes (a
// dispatch at that address with that stack pointer) are the handler's, and FZGX_TRACE=noirq leaves them out.
void TraceInterrupt(unsigned int interrupted_pc, unsigned int stack_pointer);
extern bool g_trace_on;
extern unsigned int g_trace_r1, g_trace_r3;  // guest r1 and r3 at the dispatch being reported
extern unsigned long long g_native_tsc, g_jit_tsc, g_interp_tsc, g_dispatches, g_idles;
// The slow-second line (FzgxProbe.cpp, SlowSecond): the CPU core in use tells its whole-run counts when asked, once a
// second: [0] steps on the fallback JIT, [1] instructions interpreted through the instruction hook, [2] chunks
// failing their check now, [3] chunk checks made. Unset under the emulator's own JIT.
extern void (*g_core_counters)(unsigned long long out[4]);

// The pacing log, for a fault a person feels and no counter sees (a dropped or doubled picture). Off unless
// StartPaceLog names a file (the runner's --fzgx-pace-log, or FZGX_PACE_LOG in the environment). Every record is one
// line with times in microseconds of one clock (QueryPerformanceCounter); records are kept in memory and written by a
// thread of their own every three seconds and when the process ends, so that logging never waits for the disc.
//   F  a field ended (CPU thread)        P  a picture was presented (the thread that presents)
//   S  a shader or pipeline was made     R  a pad's rumble command changed
//   C  clock calibration (the writer thread)
// The file's first lines name every column.
extern bool g_pace_on;
void StartPaceLog(const char* path);
unsigned long long PaceClock();                         // microseconds; 0 when the log is off
void PacePresent(unsigned long long before_wait_us, unsigned long long after_wait_us);
// The statistics' counts of a frame are about to start again (Statistics::ResetFrame): what they hold is carried, so
// that a P record gives the drawing of a whole frame, from one copy out of the frame buffer to the display to the
// next (PaceCopy), and not from a frame's first draw to the present, which shows part of a frame when a picture is
// held for its moment.
void PaceFrameCountsEnd();
// The time the throttle slept on the CPU thread so far (CoreTiming::SleepUntil), in nanoseconds, counted whether or
// not a log is on: the port's adapter takes a frame's own work as its time less this (the replay scrubber's budget).
void CountCpuThreadSleep(unsigned long long ns);
// Reads of the command list turned off (CommandProcessor::SetCpControlRegister) and the wait there for the GPU
// thread's pass in flight: a G record, when the log is on.
void PaceReadsOff(unsigned long long start_us, unsigned long long wait_ns, bool in_flight);
unsigned long long CpuThreadSleptNs();
void PaceCompiled(const char* what, unsigned long long start_us);
void PaceRumble(int port, double strength);
void PacePad(const GCPadStatus& pad);                   // port 1's pad as the game is about to read it
// A stretch of work on any thread, written when it took at least min_us: T,start_us,duration_us,what,thread_id
void PaceSpan(const char* what, unsigned long long start_us, unsigned long long min_us = 2000);
// A copy out of the frame buffer: E,start_us,duration_us,to_texture,to_memory,xfb,depth,width,height,format,address
void PaceCopy(unsigned long long start_us, bool to_texture, bool to_memory, bool xfb, bool depth, unsigned int width,
              unsigned int height, unsigned int format, unsigned int address);
// A texture made or destroyed by the graphics backend: X,time_us,duration_us,what,width,height,layers,levels,format,samples
void PaceTexture(const char* what, unsigned long long start_us, unsigned int width, unsigned int height,
                 unsigned int layers, unsigned int levels, unsigned int format, unsigned int samples);
// What the texture cache is doing on this thread when a texture is made or given up (X records' last two fields),
// and its size at each frame's end: N,time_us,entries,pooled,invalidated_this_frame
struct PaceContext
{
  PaceContext(const char* what, unsigned int address);
  ~PaceContext();
  const char* old_what;
  unsigned int old_address;
};
void PaceCacheSize(unsigned long long entries, unsigned long long pooled, unsigned int invalidated);
void PaceEvent(int kind, unsigned int address);         // 0 fallback JIT entered, 1 instruction interpreted, 2 chunk checked
}  // namespace FzgxProbe
