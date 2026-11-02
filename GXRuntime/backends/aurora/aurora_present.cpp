// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_present (gxruntime/aurora_present.h): Aurora on a thread of its own, fed the retail command stream by an
// emulator that keeps running the game's GPU, its pictures read back for the emulator to show.

#include "../../graphics/aurora/lib/gfx/stagelog.hpp"
#include "gx/rt_capture.hpp"
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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
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
void set_frame_buffer_size(int width, int height) noexcept;  // window.cpp
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
};

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

void run() {
  aurora::webgpu::set_host_window(s->window);
  install_mtx_check();
  install_fault_stop();
  aurora_backend_set_fatal_handler(on_fatal);
  aurora_backend_set_guest_address_resolver(ram_resolver, nullptr);
  // Owner's picture switches, read once. AURORA_MSAA=<samples>; AURORA_ANISO=<max>, with texture.cpp's AURORA_ANISO_GX=0;
  // AURORA_NATIVE=1: the content frame buffer is the window's own pixels, no whole-number scale.
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
  if (s->scale > 1 || native_size) {
    // The render scale, as Aurora's own games ask for it: the game's EFB (640x528) is the logical frame buffer, and
    // viewports, scissors and copies map from it to a content frame buffer this many times its size. The (hidden)
    // window keeps its size.
    static GXRenderModeObj mode = GXNtsc480IntDf;
    mode.fbWidth = 640;
    mode.efbHeight = 528;
    mode.xfbHeight = 528;
    VIConfigure(&mode);
    // AURORA_NATIVE: the game's display copy is 448 of these 528 lines (40 to 488), and the present letterboxes by
    // the copy's size: the frame buffer is taller than the window by 528/448, so that the copy is the window's pixels.
    if (native_size)
      aurora::window::set_frame_buffer_size(static_cast<int>(s->width),
                                            static_cast<int>(s->height * 528u / 448u));
    else
      VISetFrameBufferScale(static_cast<float>(s->scale));
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
  FILE* log = s->log_path.empty() ? nullptr : std::fopen(s->log_path.c_str(), "w");
  if (log)
    std::fprintf(log, "copy,in_us,start_us,fed_us,presented_us,picture_us,waiting,bytes\n");

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
    {
      std::unique_lock g{s->lock};
      s->wake.wait(g, [] { return s->quit || !s->queue.empty(); });
      if (s->quit)
        break;
      f = std::move(s->queue.front());
      s->queue.pop_front();
      s->bytes_waiting -= f.bytes.size();
      waiting = s->queue.size();
      // the tight bound from when the frames that waited for the start are gone: while they drain, a new frame found
      // 4 waiting and stopped Aurora for the run in its first half second (course 37, a window)
      if (waiting == 0)
        s->up = true;
    }
    const auto start = Clock::now();
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
    const bool not_drawn = s->window != nullptr && waiting >= most_waiting_to_draw && !is_shot(f.copy);
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
    aurora_backend_present();
    aurora::gfx::timewarp::true_frame_done();
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
      std::fprintf(log, "%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%zu,%zu\n", static_cast<unsigned long long>(f.copy),
                   us_since(f.in), us_since(start), us_since(fed), us_since(presented),
                   got ? us_since(pictured) : -1.0, waiting, f.bytes.size());
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
  if (s->frames_not_drawn != 0)
    std::fprintf(stderr, "[aurora-present] %llu frames not drawn, to catch up\n",
                 static_cast<unsigned long long>(s->frames_not_drawn));
  aurora::gfx::timewarp::stop();
  dol_aurora_shutdown();
}
}  // namespace

extern "C" {

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
  s->thread = std::thread(run);
  return 1;
}

void aurora_present_write(const uint8_t* bytes, uint32_t size) {
  if (s == nullptr || s->stopped.load(std::memory_order_relaxed))
    return;
  s->building.insert(s->building.end(), bytes, bytes + size);
}

void aurora_present_frame(uint64_t copy) {
  if (s == nullptr || s->stopped.load(std::memory_order_relaxed))
    return;
  Frame f;
  f.copy = copy;
  f.in = Clock::now();
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
  out->scale = s->scale;
}

void aurora_present_stop(void) {
  model_pipelines_stop();
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
