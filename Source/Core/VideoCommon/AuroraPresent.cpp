// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/AuroraPresent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include <fmt/format.h>

#include "Common/Logging/Log.h"
#include "Common/StringUtil.h"
#include "Core/HW/Memmap.h"
#include "Core/System.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/FrameRedrawFrame.h"
#include "VideoCommon/TextureConfig.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace VideoCommon::AuroraPresent
{
namespace
{
// gxruntime/aurora_present.h, as the library exports it (ABI 1)
struct LibConfig
{
  u32 abi;
  const u8* ram;
  u32 ram_size;
  const char* data_dir;
  const char* shot_prefix;
  const u64* shots;
  u32 shot_count;
  const char* log_path;
  u32 width, height;
  u32 most_behind;
};
struct LibPicture
{
  const u8* rgba;
  u32 width, height;
  u64 copy;
};
struct LibStats
{
  u64 frames_in, frames_drawn, pictures, pictures_taken, bytes_in, most_behind;
  u32 stopped, failed;
  double feed_ms_median, present_ms_median, readback_ms_median;
};
constexpr u32 LIB_ABI = 1;

struct Library
{
  int (*start)(const LibConfig*) = nullptr;
  void (*write)(const u8*, u32) = nullptr;
  void (*frame)(u64) = nullptr;
  int (*take)(LibPicture*) = nullptr;
  void (*stats)(LibStats*) = nullptr;
  void (*stop)() = nullptr;
};

Settings s_settings;
Library s_lib;
std::string s_load_problem;
std::atomic<bool> s_on{false};      // configured, the library loaded
std::atomic<bool> s_started{false}; // Aurora's thread started (the GPU thread's first burst)
bool s_ever = false;                // on at some time (for the report)
u64 s_copies = 0;                   // the XFB copies so far (the GPU thread's)
// the GPU thread's: the bytes read and not yet handed to Aurora (the frame so far, and after a cut the bytes of the
// next frame that came in the same bursts), the end of the last burst in the video buffer, the copy a cut waits for
std::vector<u8> s_pending;
const u8* s_fed_end = nullptr;
u64 s_cut_copy = 0;
u64 s_bad_cuts = 0;  // a command end outside the bytes handed in (never seen: the frame is cut at the burst's end)
bool s_cut = false;  // a cut waits for the copy's command to end
const u8* s_list = nullptr;  // the display list the command decoded now called (its bytes in RAM), and its size
u32 s_list_size = 0;
u64 s_lists = 0, s_list_bytes = 0;  // the lists put in place of their calls

// the presenter's (the GPU thread's)
std::unique_ptr<AbstractTexture> s_texture;
MathUtil::Rectangle<int> s_rect;
u64 s_shown_copy = 0;
bool s_stopped = false;  // Aurora stopped for the run: the emulator's own pictures from then on
u64 s_presents = 0, s_presents_own = 0, s_pictures_taken = 0;
std::array<u64, 10> s_lag{};  // presents by how many XFB copies Aurora's picture was behind (9: 9 or more)
FILE* s_shown_log = nullptr;
std::chrono::steady_clock::time_point s_t0;
std::string s_report;  // made at the stop

#ifdef _WIN32
HMODULE s_module = nullptr;
template <typename T>
bool Bind(T* out, const char* name)
{
  *out = reinterpret_cast<T>(GetProcAddress(s_module, name));
  return *out != nullptr;
}
#endif

bool Load(const std::string& path)
{
#ifdef _WIN32
  if (!s_module)
  {
    const std::wstring wide = UTF8ToWString(path);
    // its own folder first for its DLLs (Dawn, SDL, the shader compiler beside it)
    s_module = LoadLibraryExW(wide.c_str(), nullptr,
                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!s_module)
    {
      s_load_problem = fmt::format("cannot load {} (error {})", path, GetLastError());
      return false;
    }
  }
  if (!Bind(&s_lib.start, "aurora_present_start") || !Bind(&s_lib.write, "aurora_present_write") ||
      !Bind(&s_lib.frame, "aurora_present_frame") || !Bind(&s_lib.take, "aurora_present_take") ||
      !Bind(&s_lib.stats, "aurora_present_stats") || !Bind(&s_lib.stop, "aurora_present_stop"))
  {
    s_load_problem = fmt::format("{} lacks an aurora_present function", path);
    s_lib = {};
    return false;
  }
  return true;
#else
  s_load_problem = "Aurora is only loaded on Windows so far";
  return false;
#endif
}

bool IsShot(u64 copy)
{
  return std::find(s_settings.shots.begin(), s_settings.shots.end(), copy) != s_settings.shots.end();
}

void Start()
{
  auto& memory = Core::System::GetInstance().GetMemory();
  if (!memory.GetRAM())
    return;
  LibConfig c{};
  c.abi = LIB_ABI;
  c.ram = memory.GetRAM();
  c.ram_size = memory.GetRamSizeReal();
  c.data_dir = s_settings.data_dir.c_str();
  c.shot_prefix = s_settings.shot_prefix.empty() ? nullptr : s_settings.shot_prefix.c_str();
  c.shots = s_settings.shots.data();
  c.shot_count = static_cast<u32>(s_settings.shots.size());
  c.log_path = s_settings.log_path.empty() ? nullptr : s_settings.log_path.c_str();
  c.width = 640;
  c.height = 528;
  if (!s_lib.start(&c))
  {
    s_load_problem = "the library's interface is not this emulator's (ABI 1)";
    s_on = false;
    return;
  }
  s_started = true;
  s_t0 = std::chrono::steady_clock::now();
  if (!s_settings.log_path.empty())
  {
    s_shown_log = std::fopen((s_settings.log_path + "-shown.csv").c_str(), "w");
    if (s_shown_log)
      std::fprintf(s_shown_log, "present_us,dolphin_copy,aurora_copy,new\n");
  }
  INFO_LOG_FMT(VIDEO, "AuroraPresent: Aurora started, RAM {} bytes, data folder {}", c.ram_size,
               s_settings.data_dir);
}

std::string MakeReport()
{
  if (!s_ever)
    return {};
  if (!s_started)
    return fmt::format("aurora: not started ({})", s_load_problem.empty() ? "no command came" : s_load_problem);
  LibStats st{};
  s_lib.stats(&st);
  std::string lag;
  for (size_t i = 0; i < s_lag.size(); ++i)
    lag += fmt::format("{}{}", i ? " " : "", s_lag[i]);
  return fmt::format("aurora: {} frames in ({} KB), {} drawn, {} pictures, {} shown; presents {}: {} with Aurora's "
                     "picture, {} the emulator's own; picture behind by 0..9+ copies: {}; most frames waiting {}; "
                     "Aurora's thread a frame (medians): feed {:.2f} ms, present {:.2f} ms, readback {:.2f} ms{}{}{}{}",
                     st.frames_in, st.bytes_in / 1024, st.frames_drawn, st.pictures, s_pictures_taken, s_presents,
                     s_presents - s_presents_own, s_presents_own, lag, st.most_behind, st.feed_ms_median,
                     st.present_ms_median, st.readback_ms_median, st.stopped ? "; STOPPED (fell behind or failed)" : "",
                     st.failed ? "; Aurora could not start" : "",
                     s_bad_cuts ? fmt::format("; {} frame cuts outside the stream", s_bad_cuts) : "",
                     fmt::format("; {} display lists put in place of their calls ({} KB)", s_lists, s_list_bytes / 1024));
}

// Aurora's thread ended (it reads the guest's RAM: before the RAM goes), the report kept.
void StopAll()
{
  s_on = false;
  if (s_started)
  {
    s_report = MakeReport();
    s_lib.stop();
    s_started = false;
  }
  if (s_shown_log)
  {
    std::fclose(s_shown_log);
    s_shown_log = nullptr;
  }
}
}  // namespace

bool g_pending = false;

void Configure(const Settings& settings)
{
  if (!settings.on)
  {
    StopAll();
    return;
  }
  s_settings = settings;
  s_ever = true;
  s_report.clear();
  s_copies = 0;
  s_pending.clear();
  s_fed_end = nullptr;
  s_bad_cuts = 0;
  g_pending = false;
  s_cut = false;
  s_list = nullptr;
  s_lists = s_list_bytes = 0;
  if (!Load(settings.library))
  {
    ERROR_LOG_FMT(VIDEO, "AuroraPresent: {}; the emulator's own pictures are shown", s_load_problem);
    return;
  }
  s_on = true;
}

std::string Report()
{
  return s_report.empty() ? MakeReport() : s_report;
}

bool On()
{
  return s_on.load(std::memory_order_relaxed);
}

void OnBurstRead(const u8* bytes, u32 size)
{
  if (!s_on.load(std::memory_order_relaxed))
    return;
  if (!s_started)
  {
    Start();
    if (!s_started)
      return;
  }
  s_pending.insert(s_pending.end(), bytes, bytes + size);
  s_fed_end = bytes + size;
}

void OnXFBEntry(const RcTcacheEntry& entry)
{
  if (!s_started || s_settings.shot_prefix.empty() || !entry || !entry->texture || !IsShot(s_copies + 1))
    return;
  const AbstractTexture* t = entry->texture.get();
  FrameRedraw::QueueStill(t, MathUtil::Rectangle<int>(0, 0, t->GetWidth(), t->GetHeight()),
                          fmt::format("{}-dolphin-{:06}.png", s_settings.shot_prefix, s_copies + 1), s_copies + 1);
}

void OnXFBCopy()
{
  if (!s_started)
    return;
  ++s_copies;
  s_cut_copy = s_copies;
  s_cut = true;
  g_pending = true;
  if (!s_settings.shot_prefix.empty())
    FrameRedraw::TakeStills(s_copies, false);
}

void OnDisplayList(const u8* bytes, u32 size)
{
  if (!s_started || !bytes || !size)
    return;
  s_list = bytes;
  s_list_size = size;
  g_pending = true;
}

void OnCommandEnd(const u8* data, u32 size)
{
  g_pending = false;
  if (!s_started)
    return;
  const u8* end = data + size;
  // where the command is among the bytes handed in: `tail` bytes after it (the next commands' first)
  size_t tail = 0;
  const bool inside = s_fed_end && data && end <= s_fed_end &&
                      static_cast<size_t>(s_fed_end - data) <= s_pending.size();
  if (inside)
    tail = static_cast<size_t>(s_fed_end - end);
  if (s_list)
  {
    // the call (its 9 bytes) gives its place to the list's bytes
    if (inside && size == 9)
    {
      const size_t at = s_pending.size() - tail - size;
      s_pending.erase(s_pending.begin() + at, s_pending.begin() + at + size);
      s_pending.insert(s_pending.begin() + at, s_list, s_list + s_list_size);
      ++s_lists;
      s_list_bytes += s_list_size;
    }
    s_list = nullptr;
    s_list_size = 0;
  }
  if (!s_cut)
    return;
  s_cut = false;
  if (!inside)
  {
    ++s_bad_cuts;
    tail = 0;
  }
  const size_t frame = s_pending.size() - tail;
  if (frame)
    s_lib.write(s_pending.data(), static_cast<u32>(frame));
  s_lib.frame(s_cut_copy);
  s_pending.erase(s_pending.begin(), s_pending.begin() + frame);
}

const AbstractTexture* Picture(MathUtil::Rectangle<int>* rect)
{
  if (!s_started)
    return nullptr;
  ++s_presents;
  LibPicture p{};
  const int took = s_lib.take(&p);
  const bool fresh = took == 1;
  if (fresh && p.rgba && p.width && p.height)
  {
    if (!s_texture || s_texture->GetWidth() != p.width || s_texture->GetHeight() != p.height)
    {
      s_texture = g_gfx->CreateTexture(
          TextureConfig(p.width, p.height, 1, 1, 1, AbstractTextureFormat::RGBA8, 0,
                        AbstractTextureType::Texture_2DArray),
          "Aurora picture");
    }
    if (s_texture)
    {
      s_texture->Load(0, p.width, p.height, p.width, p.rgba, static_cast<size_t>(p.width) * p.height * 4);
      s_rect = MathUtil::Rectangle<int>(0, 0, static_cast<int>(p.width), static_cast<int>(p.height));
      s_shown_copy = p.copy;
      ++s_pictures_taken;
    }
  }
  if (took < 0)
    s_stopped = true;
  if (!s_texture || s_stopped)
  {
    ++s_presents_own;
    return nullptr;
  }
  const u64 behind = s_copies > s_shown_copy ? s_copies - s_shown_copy : 0;
  ++s_lag[std::min<u64>(behind, s_lag.size() - 1)];
  if (s_shown_log)
  {
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s_t0).count();
    std::fprintf(s_shown_log, "%.0f,%llu,%llu,%d\n", us, static_cast<unsigned long long>(s_copies),
                 static_cast<unsigned long long>(s_shown_copy), fresh ? 1 : 0);
  }
  *rect = s_rect;
  return s_texture.get();
}

void Shutdown()
{
  StopAll();
  s_texture.reset();
}
}  // namespace VideoCommon::AuroraPresent
