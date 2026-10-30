// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_present (gxruntime/aurora_present.h): Aurora on a thread of its own, fed the retail command stream by an
// emulator that keeps running the game's GPU, its pictures read back for the emulator to show.

#include "gxruntime/aurora_present.h"

#include "aurora_backend_private.h"
#include "gxruntime/aurora_backend.h"
#include "gxruntime/gx_recomp.h"

#include <aurora/gfx.h>
#include <dolphin/gx.h>
#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#endif

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
  std::uint32_t width = 640, height = 528, most_behind = 120;
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
  std::vector<double> feed_ms, present_ms, readback_ms;
};
State* s = nullptr;

double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(t - s->t0).count(); }

double median(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// The guest's RAM as Aurora's seam asks for it: an address in any of RAM's segments, the bytes there and how many
// follow. Read only.
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

void stop_for_run(const char* why) {
  if (!s->stopped.exchange(true))
    std::fprintf(stderr, "[aurora-present] stopped for the run: %s\n", why);
  s->stats.stopped = 1;
}

void run() {
  install_mtx_check();
  aurora_backend_set_guest_address_resolver(ram_resolver, nullptr);
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
  };
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
  // stream has rewritten everything. Draws wait for their pipelines, so no draw is left out while one compiles (the
  // thread falls behind instead; the emulator is never held).
  static std::vector<std::uint8_t> gx_fifo(64u * 1024u);
  GXInit(gx_fifo.data(), static_cast<u32>(gx_fifo.size()));
  AuroraSetDrawsBlocking(true);
  GXFlush();
  AuroraSetViewportPolicy(static_cast<AuroraViewportPolicy>(2));  // native: the game's own viewport, unscaled

  FILE* log = s->log_path.empty() ? nullptr : std::fopen(s->log_path.c_str(), "w");
  if (log)
    std::fprintf(log, "copy,in_us,start_us,fed_us,presented_us,picture_us,waiting,bytes\n");

  Picture back;
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
      waiting = s->queue.size();
    }
    const auto start = Clock::now();
    for (std::size_t off = 0; off < f.bytes.size(); off += 32768) {
      const std::size_t n = std::min<std::size_t>(32768, f.bytes.size() - off);
      aurora_backend_gx_write_burst(f.bytes.data() + off, static_cast<u32>(n));
    }
    const auto fed = Clock::now();
    // a picture when this is the newest frame (none waits behind it) or a test picture is wanted of it
    const bool picture = waiting == 0 || is_shot(f.copy);
    if (picture)
      aurora_request_framebuffer_readback();
    aurora_backend_present();
    const auto presented = Clock::now();
    auto pictured = presented;
    bool got = false;
    if (picture) {
      const std::uint8_t* rgba = nullptr;
      u32 w = 0, h = 0;
      for (int attempt = 0; attempt < 8000; ++attempt) {  // (2 s at most)
        if (aurora_take_framebuffer_readback(&rgba, &w, &h)) {
          got = true;
          break;
        }
        aurora_pump_framebuffer_readback();
        std::this_thread::sleep_for(std::chrono::microseconds(250));
      }
      pictured = Clock::now();
      if (got) {
        back.rgba.assign(rgba, rgba + static_cast<std::size_t>(w) * h * 4u);
        back.width = w;
        back.height = h;
        back.copy = f.copy;
        if (!s->shot_prefix.empty() && is_shot(f.copy)) {
          char path[64];
          std::snprintf(path, sizeof path, "-aurora-%06llu.png", static_cast<unsigned long long>(f.copy));
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
      s->feed_ms.push_back(ms(fed - start));
      s->present_ms.push_back(ms(presented - fed));
      if (got) {
        s->readback_ms.push_back(ms(pictured - presented));
        std::swap(back, s->ready);
        s->fresh = true;
        ++s->stats.pictures;
      }
    }
    if (aurora_backend_should_quit()) {
      std::lock_guard g{s->lock};
      stop_for_run("Aurora's window was closed");
      s->queue.clear();
      break;
    }
  }
  if (log)
    std::fclose(log);
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
    // for the run, and the emulator shows its own pictures again
    if (s->queue.size() >= s->most_behind) {
      stop_for_run("too many frames behind");
      s->queue.clear();
      return;
    }
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
}

void aurora_present_stop(void) {
  if (s == nullptr)
    return;
  {
    std::lock_guard g{s->lock};
    s->quit = true;
  }
  s->wake.notify_one();
  if (s->thread.joinable())
    s->thread.join();
  // (s is kept: a picture taken before the stop may still be read; the process ends soon after)
}
}
