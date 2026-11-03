// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_present (gxruntime/aurora_present.h): Aurora on a thread of its own, fed the retail command stream by an
// emulator that keeps running the game's GPU, its pictures read back for the emulator to show.

#include "../../graphics/aurora/lib/gfx/stagelog.hpp"
#include <aurora/gfx.hpp>
#include "gx/rt_capture.hpp"
#include "gx/redraw.hpp"
#include "gfx/render_worker.hpp"
#include "../../graphics/aurora/lib/thread.hpp"
#include "gxruntime/aurora_present.h"

#include "aurora_backend_private.h"
#include "aurora_overlay.h"
#include "model_pipelines.h"
#include "gxruntime/aurora_backend.h"
#include "gxruntime/gx_recomp.h"

#include <aurora/gfx.h>
#include <dolphin/gx.h>
#include <dolphin/gx/GXAurora.h>
#include <dolphin/vi.h>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <dxgi.h>
#pragma comment(lib, "dbghelp.lib")
#endif

extern "C" volatile int g_auroraDrawDump;  // command_processor.cpp (AURORA_DUMP_COPY)

namespace aurora::webgpu {
void set_host_window(void* window) noexcept;  // dawn/BackendBinding.cpp
void force_device_loss_for_test() noexcept;   // webgpu/gpu.cpp
}
namespace aurora::gx::fifo {
void set_drawing_off(bool off) noexcept;  // gx/command_processor.cpp
}
namespace aurora::window {
void set_host_size(int width, int height, int frame_buffer_width, int frame_buffer_height) noexcept;  // window.cpp
}
namespace aurora::gfx::timewarp {  // gfx/timewarp.cpp
void start();
void stop();
void true_frame_coming() noexcept;
void true_frame_done();
void frame_not_drawn() noexcept;
}

namespace {
using Clock = std::chrono::steady_clock;

#if defined(_WIN32)
// AURORA_MTX_CHECK=1 (a diagnostic, as dolgx_replay's): msvcp140's _Mtx_unlock is replaced, for the whole process,
// with a copy that first checks that the calling thread owns the mutex. An unlock of a mutex the thread does not own
// (a freed one among them: the stage-5 heap bug's shape) prints the mutex and the caller's stack, then breaks.
// Layout of the STL's _Mtx_internal_imp_t in this msvcp140: SRWLOCK at +0x10, owner thread id at +0x48, count +0x4C.
extern "C" int aurora_present_mtx_unlock_checked(void* mtx) {
  auto* p = static_cast<unsigned char*>(mtx);
  auto* tid = reinterpret_cast<volatile long*>(p + 0x48);
  auto* count = reinterpret_cast<volatile int*>(p + 0x4C);
  if (*tid != static_cast<long>(GetCurrentThreadId())) {
    std::fprintf(stderr, "[mtx-check] unlock of a mutex this thread does not own: %p owner %08lx count %08x thread %08lx\n",
                 mtx, static_cast<unsigned long>(*tid), static_cast<unsigned>(*count),
                 static_cast<unsigned long>(GetCurrentThreadId()));
    void* frames[32];
    const USHORT n = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
      HMODULE mod = nullptr;
      char name[MAX_PATH] = "?";
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             static_cast<LPCSTR>(frames[i]), &mod)) {
        GetModuleFileNameA(mod, name, MAX_PATH);
      }
      const char* base = std::strrchr(name, '\\');
      std::fprintf(stderr, "[mtx-check]   #%u %s+0x%llx\n", i, base ? base + 1 : name,
                   static_cast<unsigned long long>(static_cast<unsigned char*>(frames[i]) -
                                                   reinterpret_cast<unsigned char*>(mod)));
    }
    std::fflush(stderr);
    __debugbreak();
  }
  if (--*count == 0) {
    *tid = -1;
    ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(p + 0x10));
  }
  return 0;
}

bool install_mtx_check() {
  const char* v = std::getenv("AURORA_MTX_CHECK");
  if (v == nullptr || v[0] == '0')
    return false;
  HMODULE msvcp = GetModuleHandleA("msvcp140.dll");
  auto* target = msvcp ? reinterpret_cast<unsigned char*>(GetProcAddress(msvcp, "_Mtx_unlock")) : nullptr;
  if (target == nullptr) {
    std::fprintf(stderr, "[mtx-check] msvcp140 _Mtx_unlock not found\n");
    return false;
  }
  unsigned char patch[12] = {0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xE0};  // mov rax, imm64; jmp rax
  const auto addr = reinterpret_cast<unsigned long long>(&aurora_present_mtx_unlock_checked);
  std::memcpy(patch + 2, &addr, 8);
  DWORD old = 0;
  VirtualProtect(target, sizeof(patch), PAGE_EXECUTE_READWRITE, &old);
  std::memcpy(target, patch, sizeof(patch));
  VirtualProtect(target, sizeof(patch), old, &old);
  FlushInstructionCache(GetCurrentProcess(), target, sizeof(patch));
  std::fprintf(stderr, "[mtx-check] _Mtx_unlock checked (the whole process)\n");
  return true;
}
#else
bool install_mtx_check() { return false; }
#endif

// --- a small PNG writer (RGBA8, stored deflate), for the test pictures ---
std::uint32_t crc32_update(std::uint32_t crc, const std::uint8_t* data, std::size_t size) {
  static std::uint32_t table[256];
  static bool ready = false;
  if (!ready) {
    for (std::uint32_t n = 0; n < 256u; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k)
        c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[n] = c;
    }
    ready = true;
  }
  crc ^= 0xFFFFFFFFu;
  for (std::size_t i = 0; i < size; ++i)
    crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}
void put_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int s = 24; s >= 0; s -= 8)
    out.push_back(static_cast<std::uint8_t>(v >> s));
}
void put_chunk(std::vector<std::uint8_t>& out, const char type[4], const std::vector<std::uint8_t>& payload) {
  put_be32(out, static_cast<std::uint32_t>(payload.size()));
  const std::size_t begin = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), payload.begin(), payload.end());
  put_be32(out, crc32_update(0u, out.data() + begin, out.size() - begin));
}
bool write_png(const std::string& path, const std::uint8_t* rgba, std::uint32_t w, std::uint32_t h) {
  std::vector<std::uint8_t> raw;
  raw.reserve((static_cast<std::size_t>(w) * 4u + 1u) * h);
  for (std::uint32_t y = 0; y < h; ++y) {
    raw.push_back(0u);
    const std::uint8_t* row = rgba + static_cast<std::size_t>(y) * w * 4u;
    raw.insert(raw.end(), row, row + static_cast<std::size_t>(w) * 4u);
  }
  std::vector<std::uint8_t> z{0x78u, 0x01u};
  for (std::size_t off = 0; off < raw.size();) {
    const std::size_t n = std::min<std::size_t>(65535u, raw.size() - off);
    z.push_back(off + n == raw.size() ? 1u : 0u);
    z.push_back(static_cast<std::uint8_t>(n & 0xFFu));
    z.push_back(static_cast<std::uint8_t>(n >> 8));
    z.push_back(static_cast<std::uint8_t>(~n & 0xFFu));
    z.push_back(static_cast<std::uint8_t>(~(n >> 8) & 0xFFu));
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    off += n;
  }
  std::uint32_t a = 1u, b = 0u;
  for (const std::uint8_t byte : raw) {
    a = (a + byte) % 65521u;
    b = (b + a) % 65521u;
  }
  put_be32(z, (b << 16) | a);
  std::vector<std::uint8_t> png{0x89u, 'P', 'N', 'G', 0x0Du, 0x0Au, 0x1Au, 0x0Au};
  std::vector<std::uint8_t> ihdr;
  put_be32(ihdr, w);
  put_be32(ihdr, h);
  ihdr.insert(ihdr.end(), {8u, 6u, 0u, 0u, 0u});
  put_chunk(png, "IHDR", ihdr);
  put_chunk(png, "IDAT", z);
  put_chunk(png, "IEND", {});
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
  return static_cast<bool>(out);
}

struct Frame {
  std::uint64_t copy = 0;
  std::vector<std::uint8_t> bytes;
  Clock::time_point in;
  float between = 0;  // a picture between two steps (aurora_present_frame_between): its fraction of a field
  std::uint64_t position = 0;  // the stream's read position at its XFB copy (aurora_present_frame_at)
};

// The run ahead's pictures between two steps: each a fraction of a field after the true picture before it, or left out.
constexpr double kFieldMs = 1000.0 * 1001.0 / 60000.0;
constexpr std::uint64_t kRedrawMostBurstsAfterCopy = 4096;  // a request belongs to the copy just before it: tuned, 128 KB of stream
constexpr double kLateMs = 5.0;  // past its time by more than this, a picture between is not drawn at all

void wait_until(Clock::time_point due) {
#if defined(_WIN32)
  static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  const auto left = due - Clock::now() - std::chrono::microseconds(600);
  if (timer != nullptr && left > std::chrono::microseconds(100)) {
    LARGE_INTEGER when;
    when.QuadPart = -std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() / 100;
    if (SetWaitableTimer(timer, &when, 0, nullptr, nullptr, FALSE))
      WaitForSingleObject(timer, 100);
  }
#endif
  while (Clock::now() < due)
    std::this_thread::yield();
}

// AURORA_PACE=1 (off by default): each picture between is presented in a refresh of its own, at a fixed place in it, not as
// soon as it is drawn. Measured before (c37, 30 racers, DISPLAY1 240 Hz, ETW): of 217.5 presents/s 26.7/s fell in a refresh
// another one had (the compositor shows one a refresh), and the screen got 191. The true frame is never held. A picture
// between whose refresh already has a present waits for the next refresh's start; one whose refresh is free goes at once
// (the render worker has little room in a field: the true frame and three pictures fill most of it, so holding each to a
// fixed refresh after its true frame lost about one picture a field, measured). Held only on the render worker, just before
// its Present (its GPU work already submitted), at most one refresh, let go at once when a true frame comes to the worker.
// Paced pictures are drawn as soon as the worker is free, timed from the true frame's present as the worker made it (about
// 8 ms after Aurora's thread handed it over, c37 rays off). Nothing the race reads changes: only when pictures reach the screen.
// The refreshes: IDXGIOutput::WaitForVBlank on a thread of its own, for the monitor the window is on; without it, the
// measured refresh AURORA_PACE_HZ (240) from the true frame's present. AURORA_PACE_PHASE_US (300): how far into its refresh a
// held picture is presented. AURORA_PACE=2: the refreshes are timed and logged (stage log pace_vblank), nothing held.
namespace pace {
int g_mode = 0;
double g_phase_ms = 0.3, g_fallback_period_ms = 1000.0 / 240.0;
std::atomic<long long> g_vblank_ns{0};    // the last refresh seen (steady_clock ns)
std::atomic<long long> g_period_ns{0};    // its measured period
std::atomic<bool> g_true_coming{false};   // a true frame is on its way to the render worker
std::atomic<bool> g_quit{false};
// the true frame's present as the worker made it, for Aurora's thread to time the pictures after it by: how many true frames
// were handed to the worker and presented, and the last one's time and refresh
std::atomic<std::uint64_t> g_true_sent{0}, g_true_presented{0};
std::atomic<long long> g_true_ns{0};
// the render worker's own
float g_picture_fraction = 0;  // >0: the present coming is a picture between, this fraction of a field after its true frame
long long g_true_slot = 0, g_last_slot = 0;
Clock::time_point g_true_at{};
bool g_have_true = false;
std::uint64_t g_held = 0, g_let_go = 0, g_on_time = 0, g_shifted = 0;
double g_held_ms = 0;
double g_last_hold_ms = 0;  // the last picture's hold: not part of what drawing it cost

long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

#if defined(_WIN32)
using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
void vblank_thread(HWND window) {
  SetThreadDescription(GetCurrentThread(), L"Aurora pace vblank");
  HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
  auto create = dxgi ? reinterpret_cast<CreateFactory>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
  IDXGIFactory1* factory = nullptr;
  if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
    std::fprintf(stderr, "[aurora-present] pace: no DXGI factory, refresh %.3f ms from the true frame\n", g_fallback_period_ms);
    return;
  }
  IDXGIOutput* output = nullptr;
  HMONITOR on = nullptr;
  const auto find = [&] {
    HMONITOR want = MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY);
    if (output != nullptr && want == on)
      return;
    if (output)
      output->Release(), output = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; output == nullptr && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
      IDXGIOutput* o = nullptr;
      for (UINT i = 0; adapter->EnumOutputs(i, &o) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_OUTPUT_DESC d{};
        o->GetDesc(&d);
        if (d.Monitor == want) {
          output = o;
          std::fprintf(stderr, "[aurora-present] pace: refreshes of %ls\n", d.DeviceName);
          break;
        }
        o->Release();
      }
      adapter->Release();
    }
    on = want;
  };
  long long last = 0;
  std::vector<long long> first;
  for (std::uint64_t n = 0; !g_quit.load(std::memory_order_relaxed); ++n) {
    if (n % 512 == 0)
      find();
    if (output == nullptr || FAILED(output->WaitForVBlank())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    const long long t = now_ns();
    if (aurora::gfx::stagelog::on())
      aurora::gfx::stagelog::note("pace_vblank", Clock::now());
    if (last != 0) {
      const long long d = t - last;
      const long long p = g_period_ns.load(std::memory_order_relaxed);
      if (p == 0) {
        first.push_back(d);
        if (first.size() == 31) {
          std::nth_element(first.begin(), first.begin() + 15, first.end());
          g_period_ns.store(first[15], std::memory_order_relaxed);
          std::fprintf(stderr, "[aurora-present] pace: refresh %.3f ms\n", first[15] / 1e6);
        }
      } else if (d > p / 2 && d < p + p / 2) {
        g_period_ns.store(p + (d - p) / 64, std::memory_order_relaxed);  // (a missed wait's double gap left out)
      }
    }
    last = t;
    g_vblank_ns.store(t, std::memory_order_relaxed);
  }
  if (output)
    output->Release();
  factory->Release();
}
#endif

// the refresh a time is in, counted from a refresh seen (its boundary g_phase_ms after the vblank), and the time one begins
struct Grid {
  long long origin = 0, period = 0;
  long long slot(long long t) const { return (t - origin) >= 0 ? (t - origin) / period : -((origin - t + period - 1) / period); }
  long long start(long long k) const { return origin + k * period; }
};
Grid grid() {
  Grid g;
  g.period = g_period_ns.load(std::memory_order_relaxed);
  const long long v = g_vblank_ns.load(std::memory_order_relaxed);
  if (g.period > 0 && v != 0) {
    g.origin = v + static_cast<long long>(g_phase_ms * 1e6);
  } else {
    g.period = static_cast<long long>(g_fallback_period_ms * 1e6);
    g.origin = std::chrono::duration_cast<std::chrono::nanoseconds>(g_true_at.time_since_epoch()).count();
  }
  return g;
}

void before_present() {
  const auto now = Clock::now();
  const long long t = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  const float fraction = g_picture_fraction;
  if (fraction <= 0) {
    // a true frame (or anything not a picture between): never held
    g_true_at = now;
    g_have_true = true;
    g_true_coming.store(false, std::memory_order_relaxed);
    const Grid g = grid();
    g_true_slot = g_last_slot = g.slot(t);
    g_true_ns.store(t, std::memory_order_relaxed);
    g_true_presented.fetch_add(1, std::memory_order_release);
    return;
  }
  if (!g_have_true || g_mode != 1)
    return;
  // the refresh it is in, if no present has it yet; else the next one's start (the render worker has little room in a field:
  // a hold only when the picture would share a refresh, and never past the next one)
  const Grid g = grid();
  const long long period = g.period;
  const long long after = 1;
  const long long here = g.slot(t);
  if (here > g_last_slot) {
    ++g_on_time;
    g_last_slot = here;
    return;
  }
  const long long want = g_last_slot + 1;
  const long long begin = g.start(want);
  const long long until = std::min(begin, t + period);
  const auto due = now + std::chrono::nanoseconds(until - t);
  bool let_go = false;
#if defined(_WIN32)
  static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
#endif
  for (;;) {
    if (g_true_coming.load(std::memory_order_relaxed)) {
      let_go = true;
      break;
    }
    const auto left = due - Clock::now();
    if (left <= std::chrono::nanoseconds(0))
      break;
#if defined(_WIN32)
    if (timer != nullptr && left > std::chrono::microseconds(700)) {
      // in steps of at most half a millisecond, to see a true frame coming
      const auto step = std::min<Clock::duration>(left - std::chrono::microseconds(500), std::chrono::microseconds(500));
      LARGE_INTEGER when;
      when.QuadPart = -std::chrono::duration_cast<std::chrono::nanoseconds>(step).count() / 100;
      if (SetWaitableTimer(timer, &when, 0, nullptr, nullptr, FALSE))
        WaitForSingleObject(timer, 10);
      continue;
    }
#endif
    std::this_thread::yield();
  }
  const auto end = Clock::now();
  ++g_held;
  g_let_go += let_go ? 1 : 0;
  g_shifted += let_go ? 0 : 1;  // held to the next refresh's start
  g_last_hold_ms = std::chrono::duration<double, std::milli>(end - now).count();
  g_held_ms += g_last_hold_ms;
  g_last_slot = std::max(want, g.slot(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()));
  if (aurora::gfx::stagelog::on())
    aurora::gfx::stagelog::note("pace_hold", now, static_cast<std::uint64_t>(after));
}

Clock::time_point true_at() { return Clock::time_point(std::chrono::nanoseconds(g_true_ns.load(std::memory_order_relaxed))); }

void start(void* window) {
  const char* v = std::getenv("AURORA_PACE");
  g_mode = v != nullptr ? std::atoi(v) : 0;
  if (g_mode != 1 && g_mode != 2) {
    g_mode = 0;
    return;
  }
  if (const char* p = std::getenv("AURORA_PACE_PHASE_US"))
    g_phase_ms = std::atof(p) / 1000.0;
  if (const char* h = std::getenv("AURORA_PACE_HZ"); h && std::atof(h) > 1)
    g_fallback_period_ms = 1000.0 / std::atof(h);
  aurora::gfx::g_beforePresent = before_present;
#if defined(_WIN32)
  std::thread(vblank_thread, static_cast<HWND>(window)).detach();
#endif
  std::fprintf(stderr, "[aurora-present] pace: AURORA_PACE=%d, a held picture at %.2f ms into its refresh\n", g_mode, g_phase_ms);
}
} // namespace pace

struct Picture {
  std::vector<std::uint8_t> rgba;
  std::uint32_t width = 0, height = 0;
  std::uint64_t copy = 0;
};

struct State {
  // settings (fixed after start)
  const std::uint8_t* ram = nullptr;
  std::uint32_t ram_size = 0;
  std::string data_dir, shot_prefix, log_path;
  std::vector<std::uint64_t> shots;
  std::uint32_t width = 640, height = 528, most_behind = 4, scale = 1;
  void* window = nullptr;
  std::atomic<bool> fatal{false};  // a fatal error of Aurora's: its thread is parked, never joined, Aurora never shut down
  // the bytes waiting at most: the bound on what falling behind costs. A race frame is about 2.4 MB with its lists in
  // place (course 37, 30 racers). Into the host's window a stop leaves the player without a picture, so it holds more:
  // about 7 s of a thread of Aurora's stuck, against about 2 s.
  static constexpr std::size_t most_bytes_behind = 256u << 20;
  static constexpr std::size_t most_bytes_behind_window = 1024u << 20;
  std::size_t bytes_waiting = 0;  // (under lock)
  float next_between = 0;         // the GPU thread's: aurora_present_frame_between's, for the next frame
  std::uint64_t next_position = 0;  // the GPU thread's: aurora_present_frame_at's
  // AURORA_REDRAW: the requests not yet taken, oldest first, each with the stream's position when it came (under lock), and
  // what became of them. More than one: a request is made at its step's start, before that step's list is drawn, and a true
  // frame is shown about 20 ms after its tick, past the next step's request (c37, 30 racers, 240); one slot lost the older.
  struct RedrawWaiting {
    std::shared_ptr<aurora::gx::redraw::Request> request;
    std::uint64_t position = 0;
  };
  std::deque<RedrawWaiting> redraw_requests;
  static constexpr std::size_t kRedrawMostWaiting = 4;
  std::atomic<std::uint64_t> redraw_generation{0};
  std::atomic<std::uint64_t> redraw_drawn{0}, redraw_stale{0};
  std::uint64_t redraw_asked = 0, redraw_busy = 0, redraw_passed = 0, redraw_no_request = 0, redraw_far = 0, redraw_old = 0;
  std::atomic<double> redraw_cost_ms{2.0};
  double redraw_cost_sum = 0, redraw_cost_most = 0;
  std::uint64_t between_drawn = 0, between_late = 0, between_left_out = 0, between_out_true_waits = 0;  // Aurora's thread's
  double between_late_ms = 0;
  Clock::time_point t0;

  // the GPU thread's frame being built (its own; moved into the queue at each copy)
  std::vector<std::uint8_t> building;

  std::mutex lock;  // the queue, the pictures, the counters below it
  std::condition_variable wake;
  std::deque<Frame> queue;
  bool quit = false;
  Picture ready, front;
  bool fresh = false;
  AuroraPresentStats stats{};
  std::thread thread;
  std::atomic<bool> running{false};
  std::atomic<bool> stopped{false};
  bool up = false;  // (under lock) Aurora started and the frames that waited for it drained: the tight bound from then on
  std::vector<double> feed_ms, present_ms, readback_ms;
  std::uint64_t frames_not_drawn = 0;  // (Aurora's thread) into the host's window: decoded, not drawn, to catch up
};
State* s = nullptr;

double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(t - s->t0).count(); }

// The recent times the medians are of, not all of them: the stats are read every frame into the host's window, and a
// session lasts for hours.
void keep_time(std::vector<double>& times, double value) {
  constexpr std::size_t most = 8192;
  if (times.size() >= most)
    times.erase(times.begin(), times.begin() + most / 2);
  times.push_back(value);
}

double median(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// The guest's RAM as Aurora's seam asks for it: an address in any of RAM's segments, the bytes there and how many
// follow. Read only.
#ifdef _WIN32
DWORD s_aurora_thread_id = 0;
#endif
#ifdef _WIN32
// AURORA_CRASH_TRACE=1 (a test): the stack of an abort, a terminate or an access violation, as symbols, on stderr
void print_stack(const char* what) {
  std::fprintf(stderr, "[aurora-present] %s, thread %lu:\n", what, GetCurrentThreadId());
  HANDLE process = GetCurrentProcess();
  SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
  SymInitialize(process, nullptr, TRUE);
  void* frames[48];
  const USHORT n = CaptureStackBackTrace(1, 48, frames, nullptr);
  for (USHORT i = 0; i < n; ++i) {
    char buffer[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buffer);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 off = 0;
    IMAGEHLP_LINE64 line{};
    line.SizeOfStruct = sizeof(line);
    DWORD loff = 0;
    const bool has = SymFromAddr(process, reinterpret_cast<DWORD64>(frames[i]), &off, sym);
    const bool hasl = SymGetLineFromAddr64(process, reinterpret_cast<DWORD64>(frames[i]), &loff, &line);
    std::fprintf(stderr, "  %p %s+0x%llx %s:%lu\n", frames[i], has ? sym->Name : "?", static_cast<unsigned long long>(off),
                 hasl ? line.FileName : "", hasl ? line.LineNumber : 0ul);
  }
  std::fflush(stderr);
}
void install_crash_trace() {
  const char* v = std::getenv("AURORA_CRASH_TRACE");
  if (v == nullptr || v[0] != '1')
    return;
  std::signal(SIGABRT, [](int) { print_stack("abort"); });
  std::set_terminate([] { print_stack("terminate"); std::_Exit(3); });
  AddVectoredExceptionHandler(1, [](EXCEPTION_POINTERS* e) -> LONG {
    const DWORD code = e->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == 0xC0000409 || code == EXCEPTION_STACK_OVERFLOW) {
      // (the game's own code faults on purpose only on Dolphin's threads with fastmem: this reports the first few)
      static std::atomic<int> reports{0};
      if (GetCurrentThreadId() == s_aurora_thread_id && reports.fetch_add(1) < 3)
        print_stack("exception on Aurora's thread");
    }
    return EXCEPTION_CONTINUE_SEARCH;
  });
}
#else
void install_crash_trace() {}
#endif

bool ram_resolver(void*, u32 address, u32, DolGuestAddressSpace, DolGuestResourceKind, const void** data,
                  u32* available) {
  const u32 physical = dol_gx_recomp_guest_to_physical(address);
  if (physical >= s->ram_size)
    return false;
  *data = s->ram + physical;
  *available = s->ram_size - physical;
  return true;
}

// AURORA_REDRAW_SHOTS=a,b,c (a test, in order): a picture between from each of those copies on, to a file; each at the first picture
// from that copy on (the render worker's)
const std::vector<std::uint64_t>& redraw_shots() {
  static const std::vector<std::uint64_t> shots = [] {
    std::vector<std::uint64_t> out;
    if (const char* v = std::getenv("AURORA_REDRAW_SHOTS"))
      for (const char* at = v; *at != 0;) {
        char* end = nullptr;
        out.push_back(std::strtoull(at, &end, 10));
        at = *end != 0 ? end + 1 : end;
      }
    return out;
  }();
  return shots;
}
std::atomic<std::size_t> g_redraw_shots_taken{0};
bool redraw_shot_due(std::uint64_t copy) {
  const auto& shots = redraw_shots();
  const std::size_t taken = g_redraw_shots_taken.load();
  return s != nullptr && !s->shot_prefix.empty() && taken < shots.size() && copy >= shots[taken];
}

bool is_shot(std::uint64_t copy) {
  return std::find(s->shots.begin(), s->shots.end(), copy) != s->shots.end();
}

#ifdef _WIN32
// Into the host's window there is no other picture to go back to: the race goes on under a still picture, so the player
// is told. A window of its own, over the host's, on a thread of its own (Aurora's may be parked for good), which never
// takes the focus. It goes with the process.
void show_stopped_notice(HWND host) {
  std::thread([host] {
    RECT r{};
    if (!GetClientRect(host, &r))
      return;
    POINT at{(r.right - r.left) / 2, (r.bottom - r.top) / 2};
    ClientToScreen(host, &at);
    constexpr int w = 620, h = 76;
    HWND notice = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, L"STATIC",
                                  L"The picture has stopped, but your race is still running.\r\n"
                                  L"Close the game, then rejoin the race from the launcher.",
                                  WS_POPUP | WS_BORDER | SS_CENTER, at.x - w / 2, at.y - h / 2, w, h, host, nullptr,
                                  GetModuleHandleW(nullptr), nullptr);
    if (notice == nullptr)
      return;
    LOGFONTW font{};
    font.lfHeight = -20;
    wcscpy_s(font.lfFaceName, L"Segoe UI");
    SendMessageW(notice, WM_SETFONT, reinterpret_cast<WPARAM>(CreateFontIndirectW(&font)), TRUE);
    ShowWindow(notice, SW_SHOWNOACTIVATE);
    std::fprintf(stderr, "[aurora-present] the player is told the picture has stopped\n");
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0)
      DispatchMessageW(&m);
  }).detach();
}
#endif

void stop_for_run(const char* why, bool tell_the_player = true) {
  if (!s->stopped.exchange(true)) {
    std::fprintf(stderr, "[aurora-present] stopped for the run: %s\n", why);
#ifdef _WIN32
    if (s->window != nullptr && tell_the_player)
      show_stopped_notice(static_cast<HWND>(s->window));
#else
    (void)tell_the_player;
#endif
  }
  s->stats.stopped = 1;
}

// In place of Aurora's abort, on whichever thread of Aurora's it happened: an abort would end the game and its race,
// and nothing of the race depends on Aurora. The thread never returns (Aurora's state is not to be trusted from here),
// Aurora is stopped for the run and the emulator's own picture comes back.
// AURORA_TEST_DUMP_BAD=PATH: the frame being decoded goes to PATH at a fatal error
const std::vector<std::uint8_t>* s_test_frame = nullptr;
std::uint64_t s_test_frame_copy = 0;
[[noreturn]] void on_fatal(const char* message) {
  std::fprintf(stderr, "[aurora-present] fatal error, Aurora stopped for the run: %s\n", message);
  std::fflush(stderr);
  if (const char* dump = std::getenv("AURORA_TEST_DUMP_BAD"); dump && s_test_frame) {
    if (FILE* out = std::fopen(dump, "wb")) {
      std::fwrite(s_test_frame->data(), 1, s_test_frame->size(), out);
      std::fclose(out);
      std::fprintf(stderr, "[aurora-present] frame of copy %llu dumped, %zu bytes\n",
                   static_cast<unsigned long long>(s_test_frame_copy), s_test_frame->size());
    }
    s_test_frame = nullptr;
  }
  s->fatal = true;
  stop_for_run("a fatal error");
  for (;;)
    std::this_thread::sleep_for(std::chrono::hours(24));
}

#ifdef _WIN32
bool on_auroras_thread() {
  if (GetCurrentThreadId() == s_aurora_thread_id)
    return true;
  PWSTR name = nullptr;
  if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || name == nullptr)
    return false;
  const bool ours = std::wcsncmp(name, L"Aurora", 6) == 0;
  LocalFree(name);
  return ours;
}
// An access violation on a thread of Aurora's goes the way of its fatal errors: nothing of the race depends on Aurora,
// so Aurora stops for the run and the game goes on. No other thread is touched (Dolphin's own faults, fastmem above
// all, pass through). A wild write that faults nothing is not seen here.
void install_fault_stop() {
  AddVectoredExceptionHandler(0, [](EXCEPTION_POINTERS* e) -> LONG {
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && s != nullptr && on_auroras_thread())
      on_fatal("an access violation on a thread of Aurora's");
    return EXCEPTION_CONTINUE_SEARCH;
  });
}
#else
void install_fault_stop() {}
#endif

// ---- the picture follows the host's window -----------------------------------------------------------------------
// The picture's shape, width over height (aurora_present_picture_shape: the game's screen option); 0 fills the window.
std::atomic<float> s_picture_shape{16.0f / 9.0f};

// Not remade oftener than this while a window is being dragged: it costs a wait for the GPU and new targets. In between
// the swap chain's old pictures are stretched to the window by the system.
constexpr auto FOLLOW_AT_MOST_EVERY = std::chrono::milliseconds(100);
// The smallest frame buffer's picture lines: the window may be smaller, the game's copies and its 2D stay whole.
constexpr int FOLLOW_LEAST_LINES = 90;

struct FollowedWindow {
  int width = 0, height = 0;
  float shape = 0;
  bool on = false;
  std::uint32_t lines_each = 0;  // 2, 3, 4: this many picture lines for each of the game's; 0: the window's own pixels
  std::chrono::steady_clock::time_point last{};
};
FollowedWindow s_followed;

// Every frame, between frames, on Aurora's thread. The system's question takes no message and never waits for the host.
void follow_window(std::uint64_t copy) {
#ifdef _WIN32
  if (!s_followed.on)
    return;
  RECT client{};
  // (minimised, or going: the last size stays)
  if (!GetClientRect(static_cast<HWND>(s->window), &client) || client.right <= 0 || client.bottom <= 0)
    return;
  const int width = client.right, height = client.bottom;
  const float shape = s_picture_shape.load(std::memory_order_relaxed);
  if (width == s_followed.width && height == s_followed.height && shape == s_followed.shape)
    return;
  const auto now = std::chrono::steady_clock::now();
  if (now - s_followed.last < FOLLOW_AT_MOST_EVERY)
    return;
  s_followed.width = width;
  s_followed.height = height;
  s_followed.shape = shape;
  s_followed.last = now;
  std::fprintf(stderr, "[aurora-present] at copy %llu the window is %dx%d\n", static_cast<unsigned long long>(copy), width, height);
  // The largest picture of the game's shape in the window
  int picture_width = width, picture_height = height;
  if (shape > 0) {
    if (static_cast<float>(width) / static_cast<float>(height) > shape)
      picture_width = std::max(1, static_cast<int>(std::lround(static_cast<float>(height) * shape)));
    else
      picture_height = std::max(1, static_cast<int>(std::lround(static_cast<float>(width) / shape)));
  }
  if (s_followed.lines_each != 0) {
    const float wide = static_cast<float>(picture_width) / static_cast<float>(picture_height);
    picture_height = static_cast<int>(448u * s_followed.lines_each);
    picture_width = static_cast<int>(std::lround(static_cast<float>(picture_height) * wide));
  } else if (picture_height < FOLLOW_LEAST_LINES) {
    const float wide = static_cast<float>(picture_width) / static_cast<float>(picture_height);
    picture_height = FOLLOW_LEAST_LINES;
    picture_width = static_cast<int>(std::lround(static_cast<float>(picture_height) * wide));
  }
  // The game's display copy is 448 of the frame buffer's 528 lines (40 to 488), and the present letterboxes by the copy's
  // size: the frame buffer is taller than the picture by 528/448, so that the copy is the picture's pixels.
  aurora::window::set_host_size(width, height, picture_width,
                                static_cast<int>(std::lround(static_cast<double>(picture_height) * 528.0 / 448.0)));
#endif
}

void run() {
  aurora::webgpu::set_host_window(s->window);
  install_mtx_check();
  install_fault_stop();
  aurora_backend_set_fatal_handler(on_fatal);
  aurora_backend_set_guest_address_resolver(ram_resolver, nullptr);
  // Owner's picture switches, read once. AURORA_MSAA=<samples>; AURORA_ANISO=<max>, with texture.cpp's AURORA_ANISO_GX=0;
  // AURORA_NATIVE=1: into the host's window, the content frame buffer is the window's own pixels (follow_window).
  const auto env_number = [](const char* name) -> unsigned {
    const char* v = std::getenv(name);
    return v != nullptr ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : 0u;
  };
  const bool native_size = env_number("AURORA_NATIVE") != 0;
  const AuroraBackendConfig config = {
      .app_name = "Piconet Aurora",
      .window_width = s->width,
      .window_height = s->height,
      .vsync = false,
      .allow_texture_dumps = false,
      .info_logging = false,
      .graphics_logging = false,
      .force_untextured = false,
      .data_dir = s->data_dir.c_str(),
      .embedded = true,
      .msaa = env_number("AURORA_MSAA"),
      .max_anisotropy = env_number("AURORA_ANISO"),
  };
#ifdef _WIN32
  s_aurora_thread_id = GetCurrentThreadId();
#endif
  install_crash_trace();
  char arg0[] = "aurora-present";
  char* argv[] = {arg0, nullptr};
  if (!dol_aurora_initialize(1, argv, &config)) {
    std::lock_guard g{s->lock};
    s->stats.failed = 1;
    stop_for_run("Aurora could not start");
    s->queue.clear();
    return;
  }
  // The SDK's state as a booting game sets it (dolgx_replay's baseline): Aurora's own GX layer reads it before the
  // stream has rewritten everything. Draws do not wait for their pipelines (Aurora's default): a draw whose pipeline
  // still compiles is left out of that frame. Waiting would put Aurora seconds behind at the first race frames with
  // a cold cache, and Aurora must stay close behind the emulator: it reads the display lists, arrays and textures in
  // the guest's RAM when it decodes a frame, and the game rewrites them once the emulator's GPU said draw-done.
  // AURORA_DRAWS_BLOCKING=1: they wait (a test of pictures against pictures, where no draw may be missing).
  static std::vector<std::uint8_t> gx_fifo(64u * 1024u);
  GXInit(gx_fifo.data(), static_cast<u32>(gx_fifo.size()));
  const char* blocking = std::getenv("AURORA_DRAWS_BLOCKING");
  AuroraSetDrawsBlocking(blocking != nullptr && blocking[0] == '1');
  GXFlush();
  if (s->window != nullptr || s->scale > 1) {
    // The render scale, as Aurora's own games ask for it: the game's EFB (640x528) is the logical frame buffer, and
    // viewports, scissors and copies map from it to a larger content frame buffer. The (hidden) SDL window keeps its size.
    static GXRenderModeObj mode = GXNtsc480IntDf;
    mode.fbWidth = 640;
    mode.efbHeight = 528;
    mode.xfbHeight = 528;
    VIConfigure(&mode);
    if (s->window != nullptr) {
      s_followed.on = true;
      s_followed.lines_each = native_size ? 0 : s->scale;
      follow_window(0);
    } else {
      VISetFrameBufferScale(static_cast<float>(s->scale));
    }
    AuroraSetViewportPolicy(AURORA_VIEWPORT_STRETCH);
  } else {
    AuroraSetViewportPolicy(AURORA_VIEWPORT_NATIVE);  // the game's own viewport, unscaled
  }
  aurora_overlay_start(s->ram);
  model_pipelines_aurora_up();
  // AURORA_TEST_FATAL=<copy> (a test): a fatal error at this copy, the way any of Aurora's goes
  const char* test_fatal_text = std::getenv("AURORA_TEST_FATAL");
  const std::uint64_t test_fatal = test_fatal_text != nullptr ? std::strtoull(test_fatal_text, nullptr, 10) : 0;
  // AURORA_TEST_FAULT=<copy> (a test): an access violation at this copy. AURORA_TEST_BAD_XF=<copy> (a test): that
  // copy's stream ends in the largest XF load a garbled stream can ask for, to the position matrices.
  const auto test_copy = [](const char* name) -> std::uint64_t {
    const char* v = std::getenv(name);
    return v != nullptr ? std::strtoull(v, nullptr, 10) : 0;
  };
  const std::uint64_t test_fault = test_copy("AURORA_TEST_FAULT");
  const std::uint64_t test_bad_xf = test_copy("AURORA_TEST_BAD_XF");
  // AURORA_TEST_DEVICE_LOST=<copy> (a test): the graphics device is lost at this copy, the way a driver's reset loses it.
  // AURORA_TEST_STALL=<copy>:<ms> (a test): Aurora's thread stops for that long once at this copy.
  // AURORA_TEST_SLOW=<ms> (a test): every frame drawn takes at least that long, as a GPU too slow for the game.
  const std::uint64_t test_device_lost = test_copy("AURORA_TEST_DEVICE_LOST");
  std::uint64_t test_stall_copy = 0;
  unsigned test_stall_ms = 0;
  if (const char* v = std::getenv("AURORA_TEST_STALL")) {
    char* end = nullptr;
    test_stall_copy = std::strtoull(v, &end, 10);
    if (end != nullptr && *end == ':')
      test_stall_ms = static_cast<unsigned>(std::strtoul(end + 1, nullptr, 10));
  }
  const unsigned test_slow_ms = env_number("AURORA_TEST_SLOW");

  // Speed tests (stage 8), one at a time. AURORA_NO_READBACK=1: Aurora draws and presents into its hidden window and
  // no picture comes back (the emulator's own stays on screen), a stand-in for the later PresentTarget design.
  // AURORA_DISCARD=1: frames are copied to this thread and thrown away, nothing drawn: what the copying costs alone.
  const auto env_on = [](const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
  };
  const bool no_readback = env_on("AURORA_NO_READBACK") || s->window != nullptr;
  const bool discard = env_on("AURORA_DISCARD");

  aurora::gfx::timewarp::start();
  pace::start(s->window);
  FILE* log = s->log_path.empty() ? nullptr : std::fopen(s->log_path.c_str(), "w");
  if (log)
    std::fprintf(log, "copy,in_us,start_us,fed_us,presented_us,picture_us,waiting,bytes,between\n");

  Clock::time_point last_true{};
  bool redraw_frame = false;  // a true frame was just presented: pictures between may follow it
  std::uint64_t redraw_frame_position = 0, redraw_copy = 0;
  std::shared_ptr<aurora::gx::redraw::Request> redraw_for;
  std::size_t redraw_next = 0;
  double feed_mean_ms = 3.0;  // what a frame's decoding has cost lately
  Picture back;
  // A readback asked for and not yet taken, and the frame it is of. Aurora never waits for its pixels while a frame
  // waits behind it: a slow GPU copy (100+ ms seen with a busy GPU) would run it into the stop for being behind.
  bool asked = false;
  std::uint64_t asked_copy = 0;
  // Into the host's window nothing can be drawn late: with this many frames waiting behind one, it is decoded for its
  // state and not drawn, so a slow GPU shows fewer pictures and never older ones (and the RAM a draw reads belongs to a
  // frame no more than this far back). No effect on the race: nothing of it depends on Aurora.
  constexpr std::size_t most_waiting_to_draw = 3;
  const auto frame_waits = [] {
    std::lock_guard g{s->lock};
    return !s->queue.empty();
  };
  for (;;) {
    Frame f;
    std::size_t waiting = 0;
    bool true_waits = false;
    {
      std::unique_lock g{s->lock};
      while (!s->quit && s->queue.empty()) {
        if (!aurora::gx::redraw::g_on || !redraw_frame) {
          s->wake.wait(g, [&] { return s->quit || !s->queue.empty() || (aurora::gx::redraw::g_on && redraw_frame); });
          continue;
        }
        // A picture between when its moment comes; the next true frame always comes first.
        const auto now = Clock::now();
        if (!redraw_for) {
          // (the read position moves by whole passes of the FIFO, so a frame's can be a little past what was written at the tick)
          while (!s->redraw_requests.empty() && s->redraw_requests.front().position + kRedrawMostBurstsAfterCopy < redraw_frame_position) {
            s->redraw_requests.pop_front();  // a frame's before this one
            ++s->redraw_old;
          }
          if (!s->redraw_requests.empty()) {
            if (s->redraw_requests.front().position > redraw_frame_position + kRedrawMostBurstsAfterCopy) {
              // made after a list whose copy is still on its way: it waits for that frame
              ++s->redraw_far;
              redraw_frame = false;
              continue;
            }
            redraw_for = std::move(s->redraw_requests.front().request);
            s->redraw_requests.pop_front();
            redraw_next = 0;
            ++s->redraw_asked;
          } else {
            if (ms(now - last_true) > kFieldMs) {
              ++s->redraw_no_request;
              redraw_frame = false;
              continue;
            }
            s->wake.wait_until(g, now + std::chrono::microseconds(250));
            continue;
          }
        }
        // the first picture whose moment is still ahead by what a picture takes
        const double cost = s->redraw_cost_ms.load(std::memory_order_relaxed);
        const auto& pictures = redraw_for->pictures;
        // AURORA_PACE=1: the moments are the refreshes after the true frame's present as the worker made it (it comes some
        // ms after it was handed over), not after the hand-over
        const bool paced = pace::g_mode == 1 && !redraw_shot_due(redraw_copy);
        if (paced && pace::g_true_presented.load(std::memory_order_acquire) < pace::g_true_sent.load(std::memory_order_relaxed)) {
          if (ms(now - last_true) > kFieldMs) {
            ++s->redraw_no_request;
            redraw_frame = false;
            redraw_for.reset();
            continue;
          }
          s->wake.wait_until(g, now + std::chrono::microseconds(250));
          continue;
        }
        const Clock::time_point from = paced ? pace::true_at() : last_true;
        const auto moment = [&](std::size_t i) { return pictures[i].fraction * kFieldMs; };
        const double slack = kFieldMs / (pictures.size() + 1);
        const double since = ms(now - from);
        // (a test's shot due: its picture is drawn late rather than passed)
        while (!redraw_shot_due(redraw_copy) && redraw_next < pictures.size() && moment(redraw_next) < since + cost - slack) {
          ++s->redraw_passed;
          ++redraw_next;
        }
        if (redraw_next >= pictures.size()) {
          redraw_frame = false;
          redraw_for.reset();
          continue;
        }
        // (paced: drawn as soon as the worker is free, and held there to its refresh, so that a later picture is not caught
        // by the next true frame before it was drawn)
        const double at = paced ? 0.0 : moment(redraw_next) - cost;
        if (since < at) {
          s->wake.wait_until(g, from + std::chrono::microseconds(static_cast<long long>(at * 1000.0)));
          continue;
        }
        if (!aurora::gfx::render_worker::is_idle()) {
          ++s->redraw_busy;
          s->wake.wait_until(g, now + std::chrono::microseconds(100));
          continue;
        }
        const std::size_t index = redraw_next++;
        const std::uint64_t generation = s->redraw_generation.load(std::memory_order_relaxed);
        aurora::gfx::render_worker::enqueue_work([request = redraw_for, index, generation, copy = redraw_copy, log, t_due = last_true] {
          if (generation != s->redraw_generation.load(std::memory_order_relaxed)) {
            s->redraw_stale.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          const auto began = Clock::now();
          const bool shoot = redraw_shot_due(copy);
          if (shoot)
            ++g_redraw_shots_taken;
          const auto save_shot = [&](const char* kind) {
            const std::uint8_t* rgba = nullptr;
            u32 w = 0, h = 0;
            for (int attempt = 0; attempt < 8000 && !aurora_take_framebuffer_readback(&rgba, &w, &h); ++attempt) {
              aurora_pump_framebuffer_readback();
              std::this_thread::sleep_for(std::chrono::microseconds(250));
            }
            if (rgba != nullptr) {
              char path[64];
              std::snprintf(path, sizeof path, "-%s-%06llu-%zu.png", kind, static_cast<unsigned long long>(copy), index);
              write_png(s->shot_prefix + path, rgba, w, h);
            }
          };
          if (shoot) {
            // the frame again with no correction first: this picture against its own true frame, by the same way
            aurora::gx::redraw::Request still = *request;
            auto& p = still.pictures[index];
            static constexpr aurora::gx::redraw::Matrix one{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            p.camera = one;
            p.machine.fill(one);
            char table[64];
            std::snprintf(table, sizeof table, "-draws-%06llu.csv", static_cast<unsigned long long>(copy));
            aurora::gx::redraw::table_next(s->shot_prefix + table);
            aurora_request_framebuffer_readback();
            aurora::gx::redraw::draw_picture(still, index);
            save_shot("still");
            aurora_request_framebuffer_readback();
          }
          pace::g_picture_fraction = shoot ? 0.0f : request->pictures[index].fraction;  // (a test's shot is never held)
          pace::g_last_hold_ms = 0;
          const bool drew = aurora::gx::redraw::draw_picture(*request, index);
          pace::g_picture_fraction = 0;
          if (!drew)
            return;
          const auto done = Clock::now();
          if (shoot) {
            save_shot("between");
            // AURORA_REDRAW_BUNDLES_CHECK=1 (a test): the same picture again the other way (render bundles or each draw encoded), to its
            // own file: the two must be the same
            // (2: the same way again, a control: what differs between two draws of one picture by one way)
            if (const char* v = std::getenv("AURORA_REDRAW_BUNDLES_CHECK"); v != nullptr && (v[0] == '1' || v[0] == '2')) {
              const bool was = aurora::gx::redraw::bundles();
              aurora::gx::redraw::set_bundles(v[0] == '1' ? !was : was);
              aurora_request_framebuffer_readback();
              aurora::gx::redraw::draw_picture(*request, index);
              save_shot(v[0] == '2' ? "again" : was ? "direct" : "bundled");
              aurora::gx::redraw::set_bundles(was);
            }
            // AURORA_REDRAW_TEST_TEXTURES=hex,hex (a test): the picture again with the draws of each first texture uncorrected (1: no
            // vertex moved), each to a file of its own
            if (const char* v = std::getenv("AURORA_REDRAW_TEST_TEXTURES"))
              for (const char* at = v; *at != 0;) {
                char* end = nullptr;
                const auto texture = static_cast<u32>(std::strtoul(at, &end, 16));
                at = *end != 0 ? end + 1 : end;
                aurora::gx::redraw::test_uncorrected(texture);
                aurora_request_framebuffer_readback();
                aurora::gx::redraw::draw_picture(*request, index);
                char kind[32];
                std::snprintf(kind, sizeof kind, "test%08X", texture);
                save_shot(kind);
              }
            aurora::gx::redraw::test_uncorrected(0);
          }
          const double took = ms(done - began) - pace::g_last_hold_ms;
          if (!shoot)  // (a test's shot waits for its read-backs: not a picture's cost)
            s->redraw_cost_ms.store(s->redraw_cost_ms.load(std::memory_order_relaxed) * 0.9 + took * 0.1, std::memory_order_relaxed);
          s->redraw_cost_sum += took;
          s->redraw_cost_most = std::max(s->redraw_cost_most, took);
          s->redraw_drawn.fetch_add(1, std::memory_order_relaxed);
          if (log)
            std::fprintf(log, "%llu,%.0f,%.0f,%.0f,%.0f,-1,0,0,%.3f\n", static_cast<unsigned long long>(copy), us_since(t_due), us_since(began),
                         us_since(began), us_since(done), request->pictures[index].fraction);
        });
      }
      if (s->quit)
        break;
      s->redraw_generation.fetch_add(1, std::memory_order_relaxed);
      redraw_frame = false;
      redraw_for.reset();
      f = std::move(s->queue.front());
      s->queue.pop_front();
      s->bytes_waiting -= f.bytes.size();
      waiting = s->queue.size();
      for (const Frame& q : s->queue)
        true_waits = true_waits || q.between <= 0;
      // the tight bound from when the frames that waited for the start are gone: while they drain, a new frame found
      // 4 waiting and stopped Aurora for the run in its first half second (course 37, a window)
      if (waiting == 0)
        s->up = true;
    }
    const auto start = Clock::now();
    follow_window(f.copy);
    if (discard) {
      std::lock_guard g{s->lock};
      ++s->stats.frames_drawn;
      continue;
    }
    static const std::uint64_t dump_copy = [] {
      const char* v = std::getenv("AURORA_DUMP_COPY");
      return v != nullptr ? std::strtoull(v, nullptr, 10) : 0ull;
    }();
    g_auroraDrawDump = dump_copy != 0 && f.copy == dump_copy;
    if (test_fatal != 0 && f.copy == test_fatal) {
      on_fatal("AURORA_TEST_FATAL");
    }
    if (test_fault != 0 && f.copy == test_fault) {
      *static_cast<volatile int*>(nullptr) = 1;
    }
    if (test_device_lost != 0 && f.copy == test_device_lost) {
      aurora::webgpu::force_device_loss_for_test();
    }
    if (test_stall_copy != 0 && f.copy == test_stall_copy) {
      std::this_thread::sleep_for(std::chrono::milliseconds(test_stall_ms));
    }
    if (test_bad_xf != 0 && f.copy == test_bad_xf) {
      f.bytes.push_back(0x10);
      f.bytes.insert(f.bytes.end(), {0xFF, 0xFF, 0x00, 0x00});
      f.bytes.insert(f.bytes.end(), 65536u * 4u, 0x3F);
    }
    s_test_frame = &f.bytes;
    s_test_frame_copy = f.copy;
    aurora::gx::rt::begin_copy(f.copy);
    // A picture between that cannot be on time is left out: the true one never waits for it. (Decoded all the same: a
    // frame missed would leave Aurora's state wrong.)
    const bool between = f.between > 0;
    const Clock::time_point due =
        last_true + std::chrono::microseconds(static_cast<long long>(f.between * kFieldMs * 1000.0));
    const bool too_late = between && !is_shot(f.copy) &&
                          (last_true == Clock::time_point{} || true_waits ||
                           ms(start - due) + feed_mean_ms > kLateMs);
    if (too_late) {
      ++s->between_left_out;
      s->between_out_true_waits += true_waits ? 1 : 0;
    }
    const bool not_drawn = (s->window != nullptr && waiting >= most_waiting_to_draw && !is_shot(f.copy)) || too_late;
    aurora::gx::fifo::set_drawing_off(not_drawn);
    for (std::size_t off = 0; off < f.bytes.size(); off += 32768) {
      const std::size_t n = std::min<std::size_t>(32768, f.bytes.size() - off);
      aurora_backend_gx_write_burst(f.bytes.data() + off, static_cast<u32>(n));
    }
    if (not_drawn) {
      ++s->frames_not_drawn;
      aurora::gfx::timewarp::frame_not_drawn();
      continue;
    }
    const auto fed = Clock::now();
    feed_mean_ms += (ms(fed - start) - feed_mean_ms) / 8.0;
    if (between) {
      const double late = ms(fed - due);
      if (late > 0) {
        ++s->between_late;
        s->between_late_ms += late;
      } else {
        wait_until(due);
      }
      ++s->between_drawn;
    } else {
      last_true = Clock::now();
    }
    if (test_slow_ms != 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(test_slow_ms));
    // a picture when this is the newest frame (none waits behind it) or a test picture is wanted of it (which must be
    // of that frame: one still on its way comes first)
    const bool shot = is_shot(f.copy);
    if (shot && asked) {
      const std::uint8_t* old = nullptr;
      u32 oldw = 0, oldh = 0;
      for (int attempt = 0; attempt < 8000 && !aurora_take_framebuffer_readback(&old, &oldw, &oldh); ++attempt) {
        aurora_pump_framebuffer_readback();
        std::this_thread::sleep_for(std::chrono::microseconds(250));
      }
      asked = false;
    }
    if (!asked && (waiting == 0 || shot) && (!no_readback || shot)) {
      aurora_request_framebuffer_readback();
      asked = true;
      asked_copy = f.copy;
    }
    aurora::gfx::timewarp::true_frame_coming();
    pace::g_true_coming.store(true, std::memory_order_relaxed);
    pace::g_true_sent.fetch_add(1, std::memory_order_relaxed);
    aurora_backend_present();
    aurora::gfx::timewarp::true_frame_done();
    if (aurora::gx::redraw::g_on) {
      last_true = Clock::now();
      redraw_frame = true;
      redraw_frame_position = f.position;
      redraw_copy = f.copy;
    }
    const auto presented = Clock::now();
    if (aurora::gfx::stagelog::on()) {
      aurora::gfx::stagelog::note("copy_queue", f.in, f.copy);
      aurora::gfx::stagelog::note("copy_feed", start, f.copy);
      aurora::gfx::stagelog::note("copy_present", fed, f.copy);
    }
    auto pictured = presented;
    bool got = false;
    if (asked) {
      const std::uint8_t* rgba = nullptr;
      u32 w = 0, h = 0;
      for (int attempt = 0; attempt < 8000; ++attempt) {  // (2 s at most)
        if (aurora_take_framebuffer_readback(&rgba, &w, &h)) {
          got = true;
          break;
        }
        aurora_pump_framebuffer_readback();
        if (!shot && frame_waits())
          break;
        std::this_thread::sleep_for(std::chrono::microseconds(250));
      }
      pictured = Clock::now();
      if (got) {
        back.rgba.assign(rgba, rgba + static_cast<std::size_t>(w) * h * 4u);
        back.width = w;
        back.height = h;
        back.copy = asked_copy;
        asked = false;
        if (!s->shot_prefix.empty() && is_shot(back.copy)) {
          char path[64];
          std::snprintf(path, sizeof path, "-aurora-%06llu.png", static_cast<unsigned long long>(back.copy));
          write_png(s->shot_prefix + path, back.rgba.data(), w, h);
        }
      }
    }
    if (log)
      std::fprintf(log, "%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%zu,%zu,%.3f\n", static_cast<unsigned long long>(f.copy),
                   us_since(f.in), us_since(start), us_since(fed), us_since(presented),
                   got ? us_since(pictured) : -1.0, waiting, f.bytes.size(), f.between);
    {
      std::lock_guard g{s->lock};
      ++s->stats.frames_drawn;
      keep_time(s->feed_ms, ms(fed - start));
      keep_time(s->present_ms, ms(presented - fed));
      if (got) {
        keep_time(s->readback_ms, ms(pictured - presented));
        std::swap(back, s->ready);
        s->fresh = true;
        ++s->stats.pictures;
      }
    }
    if (aurora_backend_should_quit()) {
      std::lock_guard g{s->lock};
      stop_for_run("Aurora's window was closed", false);
      s->queue.clear();
      break;
    }
  }
  if (log)
    std::fclose(log);
  if (s->between_drawn != 0 || s->between_left_out != 0)
    std::fprintf(stderr, "[aurora-present] pictures between two steps: %llu drawn (%llu of them late, %.2f ms on average), %llu left out (%llu with a true one waiting)\n",
                 static_cast<unsigned long long>(s->between_drawn), static_cast<unsigned long long>(s->between_late),
                 s->between_late ? s->between_late_ms / static_cast<double>(s->between_late) : 0.0,
                 static_cast<unsigned long long>(s->between_left_out),
                 static_cast<unsigned long long>(s->between_out_true_waits));
  if (pace::g_mode != 0)
    std::fprintf(stderr, "[aurora-present] pace: pictures between held %llu (%.2f ms on average, %llu let go for a true frame), its refresh free %llu, held to the next refresh %llu\n",
                 static_cast<unsigned long long>(pace::g_held), pace::g_held ? pace::g_held_ms / static_cast<double>(pace::g_held) : 0.0,
                 static_cast<unsigned long long>(pace::g_let_go), static_cast<unsigned long long>(pace::g_on_time),
                 static_cast<unsigned long long>(pace::g_shifted));
  if (s->frames_not_drawn != 0)
    std::fprintf(stderr, "[aurora-present] %llu frames not drawn, to catch up\n",
                 static_cast<unsigned long long>(s->frames_not_drawn));
  if (aurora::gx::redraw::g_on) {
    aurora::gfx::render_worker::synchronize();
    const auto c = aurora::gx::redraw::counts();
    const auto drawn = s->redraw_drawn.load();
    std::fprintf(stderr, "[aurora-present] redraw: requests taken %llu, pictures drawn %llu (cost mean %.2f ms, most %.2f on the render worker), passed by the clock %llu, worker busy waits %llu, stale %llu, frames without a request %llu, request ahead of its frame %llu, older than its frame %llu; classes per frame: world %.0f, machine %.0f, whole %.0f, icon %.0f, flat %.0f, offscreen %.0f; uniform bytes %llu; draws with a split matrix %llu; view-space draws moved vertex by vertex %.0f a frame (%.0f vertices moved a picture, %.0f bytes written), with no float positions %.0f; flat draws over a machine (name tags) %.0f a frame\n",
                 (unsigned long long)s->redraw_asked, (unsigned long long)drawn, drawn ? s->redraw_cost_sum / drawn : 0.0, s->redraw_cost_most,
                 (unsigned long long)s->redraw_passed, (unsigned long long)s->redraw_busy, (unsigned long long)s->redraw_stale.load(),
                 (unsigned long long)s->redraw_no_request, (unsigned long long)s->redraw_far, (unsigned long long)s->redraw_old, c.frames ? double(c.world) / c.frames : 0.0,
                 c.frames ? double(c.machine) / c.frames : 0.0, c.frames ? double(c.whole) / c.frames : 0.0, c.frames ? double(c.icon) / c.frames : 0.0,
                 c.frames ? double(c.flat) / c.frames : 0.0, c.frames ? double(c.offscreen) / c.frames : 0.0, (unsigned long long)c.uniformBytes, (unsigned long long)c.splits,
                 c.frames ? double(c.byVertex) / c.frames : 0.0, c.pictures ? double(c.moved) / c.pictures : 0.0,
                 c.pictures ? double(c.written) / c.pictures : 0.0, c.frames ? double(c.unreadable) / c.frames : 0.0,
                 c.frames ? double(c.over) / c.frames : 0.0);
  }
  aurora::gfx::timewarp::stop();
  dol_aurora_shutdown();
}
}  // namespace

extern "C" {

void aurora_present_picture_shape(float width_over_height) {
  s_picture_shape.store(width_over_height, std::memory_order_relaxed);
}

int aurora_present_start(const AuroraPresentConfig* config) {
  if (s != nullptr || config == nullptr || config->abi != AURORA_PRESENT_ABI || config->ram == nullptr)
    return 0;
  s = new State();
  s->t0 = Clock::now();
  s->ram = config->ram;
  s->ram_size = config->ram_size;
  s->data_dir = config->data_dir ? config->data_dir : "";
  s->shot_prefix = config->shot_prefix ? config->shot_prefix : "";
  s->log_path = config->log_path ? config->log_path : "";
  if (config->shots)
    s->shots.assign(config->shots, config->shots + config->shot_count);
  if (config->width)
    s->width = config->width;
  if (config->height)
    s->height = config->height;
  if (config->most_behind)
    s->most_behind = config->most_behind;
  s->window = config->window;
  s->scale = std::clamp<std::uint32_t>(config->scale, 1u, 8u);
  // AURORA_TEST_SIZE=<w>x<h> (tests): the picture's size, whatever the window's
  if (const char* v = std::getenv("AURORA_TEST_SIZE")) {
    unsigned w = 0, h = 0;
    if (std::sscanf(v, "%ux%u", &w, &h) == 2 && w > 0 && h > 0)
      s->width = w, s->height = h;
  }
  if (const char* v = std::getenv("AURORA_MOST_BEHIND"); v != nullptr && std::atoi(v) > 0)  // (tests)
    s->most_behind = static_cast<std::uint32_t>(std::atoi(v));
  s->building.reserve(512u * 1024u);
  s->running = true;
  aurora::thread::raise_picture_threads_from_env();
  s->thread = std::thread(run);
  return 1;
}

void aurora_present_write(const uint8_t* bytes, uint32_t size) {
  if (s == nullptr || s->stopped.load(std::memory_order_relaxed))
    return;
  s->building.insert(s->building.end(), bytes, bytes + size);
}

void aurora_present_frame_at(uint64_t position) {
  if (s != nullptr)
    s->next_position = position;
}

void aurora_present_redraw(const AuroraRedrawRequest* request) {
  if (s == nullptr || request == nullptr || !aurora::gx::redraw::g_on || s->stopped.load(std::memory_order_relaxed))
    return;
  auto q = std::make_shared<aurora::gx::redraw::Request>();
  q->machines = request->machines;
  std::memcpy(q->at.data(), request->at, sizeof request->at);
  // a seat with no machine (its place not a number, or out past any course) has no place: nothing is near it
  for (std::uint32_t i = 0; i < AURORA_REDRAW_MACHINES; ++i) {
    const auto& at = q->at[i];
    if (!(std::fabs(at[0]) < 1e5f && std::fabs(at[1]) < 1e5f && std::fabs(at[2]) < 1e5f))
      q->machines &= ~(1u << i);
  }
  q->pictures.resize(std::min<uint32_t>(request->pictures, AURORA_REDRAW_MOST));
  for (std::size_t k = 0; k < q->pictures.size(); ++k) {
    auto& to = q->pictures[k];
    const auto& from = request->picture[k];
    to.fraction = from.fraction;
    std::memcpy(to.camera.data(), from.camera, sizeof from.camera);
    std::memcpy(to.machine.data(), from.machine, sizeof from.machine);
  }
  {
    std::lock_guard g{s->lock};
    if (s->redraw_requests.size() >= State::kRedrawMostWaiting) {
      s->redraw_requests.pop_front();
      ++s->redraw_old;
    }
    s->redraw_requests.push_back({std::move(q), request->position});
  }
  s->wake.notify_all();
}

void aurora_present_frame_between(uint64_t copy, float fraction) {
  (void)copy;
  if (s != nullptr)
    s->next_between = fraction;
}

void aurora_present_frame(uint64_t copy) {
  if (s == nullptr || s->stopped.load(std::memory_order_relaxed))
    return;
  Frame f;
  f.copy = copy;
  f.in = Clock::now();
  f.between = s->next_between;
  f.position = s->next_position;
  s->next_between = 0;
  f.bytes = std::move(s->building);
  s->building = {};
  s->building.reserve(std::max<std::size_t>(f.bytes.size() + f.bytes.size() / 4, 64u * 1024u));
  {
    std::lock_guard g{s->lock};
    ++s->stats.frames_in;
    s->stats.bytes_in += f.bytes.size();
    // Aurora has fallen too far behind (a frame missed would leave its state wrong, so none is skipped): it stops
    // for the run, and the emulator shows its own pictures again. The bound is tight (4 frames waiting): further
    // behind, the guest's RAM it reads may already hold a later frame's lists and vertices (garbage commands)
    // (while Aurora itself starts, half a second, the frames of the boot wait: 1800 of them at most)
    // In the host window there is no picture of the emulator's to go back to: a stop would freeze the window with the
    // race going on (course 36 under load, 1214 frames in), so only the byte bound stops it there.
    const std::size_t most = (s->up && s->window == nullptr) ? s->most_behind : 1800u;
    // AURORA_MOST_MB_BEHIND=<MB> (tests only: further behind, the guest's RAM may hold a later frame's lists); AURORA_QUEUE_LOG=<file>: a line a frame, copy, bytes, frames and bytes waiting
    static const std::size_t mostMb = [] {
      const char* v = std::getenv("AURORA_MOST_MB_BEHIND");
      return v != nullptr ? static_cast<std::size_t>(std::atoll(v)) : std::size_t{0};
    }();
    static FILE* queueLog = [] {
      const char* v = std::getenv("AURORA_QUEUE_LOG");
      return v != nullptr ? std::fopen(v, "wb") : nullptr;
    }();
    if (queueLog != nullptr)
      std::fprintf(queueLog, "%llu,%zu,%zu,%zu\n", static_cast<unsigned long long>(copy), f.bytes.size(), s->queue.size(), s->bytes_waiting);
    const std::size_t most_bytes = mostMb != 0 ? mostMb << 20 : s->window != nullptr ? State::most_bytes_behind_window : State::most_bytes_behind;
    if (s->queue.size() >= most || s->bytes_waiting + f.bytes.size() > most_bytes) {
      stop_for_run(s->queue.size() >= most ? "too many frames behind" : "too many bytes behind");
      s->queue.clear();
      s->bytes_waiting = 0;
      return;
    }
    s->bytes_waiting += f.bytes.size();
    s->queue.push_back(std::move(f));
    s->stats.most_behind = std::max<uint64_t>(s->stats.most_behind, s->queue.size());
  }
  s->wake.notify_one();
}

int aurora_present_take(AuroraPresentPicture* out) {
  if (s == nullptr || out == nullptr)
    return 0;
  if (s->stopped.load(std::memory_order_relaxed))
    return -1;
  std::lock_guard g{s->lock};
  if (!s->fresh)
    return 0;
  std::swap(s->front, s->ready);
  s->fresh = false;
  ++s->stats.pictures_taken;
  out->rgba = s->front.rgba.data();
  out->width = s->front.width;
  out->height = s->front.height;
  out->copy = s->front.copy;
  return 1;
}

void aurora_present_stats(AuroraPresentStats* out) {
  if (out == nullptr)
    return;
  *out = {};
  if (s == nullptr)
    return;
  std::lock_guard g{s->lock};
  *out = s->stats;
  out->stopped = s->stopped ? 1 : 0;
  out->feed_ms_median = median(s->feed_ms);
  out->present_ms_median = median(s->present_ms);
  out->readback_ms_median = median(s->readback_ms);
  out->overflow_drops = aurora_overflow_drops();
  out->fatal = s->fatal ? 1 : 0;
  out->scale = s_followed.on && s_followed.lines_each == 0 ? 0 : s->scale;
  out->pictures_between = s->redraw_drawn.load(std::memory_order_relaxed);
}

void aurora_present_stop(void) {
  model_pipelines_stop();
  pace::g_quit.store(true, std::memory_order_relaxed);
  if (s == nullptr)
    return;
  {
    std::lock_guard g{s->lock};
    s->quit = true;
  }
  s->wake.notify_one();
  if (s->fatal) {
    // (a thread of Aurora's is parked for good: nothing of Aurora can be waited for or shut down; the process ends soon)
    if (s->thread.joinable())
      s->thread.detach();
    return;
  }
  if (s->thread.joinable())
    s->thread.join();
  // (s is kept: a picture taken before the stop may still be read; the process ends soon after)
}
}
