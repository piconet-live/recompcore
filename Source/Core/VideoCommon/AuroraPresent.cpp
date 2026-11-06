// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/AuroraPresent.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <fmt/format.h>

#include "Common/Logging/Log.h"
#include "Common/StringUtil.h"
#include "Core/Config/GraphicsSettings.h"
#include "VideoCommon/VideoConfig.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/Memmap.h"
#include "Core/System.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/FrameRedrawFrame.h"
#include "VideoCommon/FzgxFifoTest.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/PerformanceMetrics.h"
#include "VideoCommon/TextureConfig.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace VideoCommon::AuroraPresent
{
namespace
{
// gxruntime/aurora_present.h, as the library exports it (ABI 5)
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
  u32 scale;
  void* window;
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
  u64 overflow_drops;
  u32 fatal;
  u32 scale;
  u64 pictures_between;
};
constexpr u32 LIB_ABI = 5;
void* s_video_off_window = nullptr;  // SetVideoOffWindow
u64 s_presents_counted = 0;
// The hand-off's time on the thread that decodes (the report's last figure): the bursts kept, the calls swapped for
// their lists, the frame given to the library
u64 s_handoff_ns = 0;
struct HandoffTimer
{
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  ~HandoffTimer()
  {
    s_handoff_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  }
};

// (gxruntime/aurora_present.h's, as the rest: this build does not include it)
struct AuroraOverlayVertex
{
  float x, y, u, v;
  u32 rgba;
};
struct AuroraOverlayDraw
{
  u64 texture;
  float clip[4];
  u32 first_index, index_count;
  s32 base_vertex;
  u32 linear;
};
struct AuroraOverlayFrame
{
  u32 width, height;
  const AuroraOverlayVertex* vertices;
  u32 vertex_count;
  const u16* indices;
  u32 index_count;
  const AuroraOverlayDraw* draws;
  u32 draw_count;
};

struct Library
{
  int (*start)(const LibConfig*) = nullptr;
  void (*write)(const u8*, u32) = nullptr;
  void (*frame)(u64) = nullptr;
  void (*frame_between)(u64, float) = nullptr;
  void (*frame_at)(u64) = nullptr;
  void (*redraw)(const void*) = nullptr;
  int (*take)(LibPicture*) = nullptr;
  void (*stats)(LibStats*) = nullptr;
  void (*stop)() = nullptr;
  void (*overlay_texture)(u64, u32, u32, u32, u32, u32, u32, const u8*, u32) = nullptr;
  void (*overlay_forget)(u64) = nullptr;
  void (*overlay_frame)(const void*) = nullptr;
  int (*picture_in_window)(u32, u32, s32*) = nullptr;
  void (*picture_shape)(float) = nullptr;
  void (*model_file)(const char*, const u8*, u64, int) = nullptr;
  void (*scene)(int, const char*) = nullptr;
  void (*step)(u32) = nullptr;  // optional: older libraries have none
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
u64 s_cut_lists = 0;                // the lists of which Aurora got less than all (OnListRead)
// A test's look (AURORA_TEST_FIFO_LOG): the list's bytes as the emulator was about to read them, and each frame's lists
std::vector<u8> s_test_list;
u64 s_test_changed = 0, s_test_kept = 0, s_test_nested = 0;
FILE* s_test_table = nullptr;

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

// NoteBetween's, oldest first; the fraction of the copy being cut (0: a true picture)
struct Between
{
  u64 position;
  float fraction;
};
std::mutex s_between_lock;
std::deque<Between> s_between;
float s_cut_fraction = 0;
u64 s_cut_position = 0;

// NoteMark's, oldest first: the stream's byte position at the mark and its tag. The CPU's thread adds, the GPU thread places.
struct Mark
{
  u64 position;
  u32 tag;
};
constexpr size_t MOST_MARKS = 4096;  // a frame has about 4 a machine; more waiting means none is being placed
constexpr u16 SET_DRAW_TAG = 0x0060;  // GX_AURORA's subcommand (GXRuntime's GXAurora.h GX_AURORA_SET_DRAW_TAG), a u32 after it
constexpr u8 GX_AURORA_COMMAND = 0x50;
std::mutex s_marks_lock;
std::deque<Mark> s_marks;
std::atomic<u64> s_mark_first{~u64{0}};      // the oldest waiting mark's position: the GPU thread's cheap look
std::atomic<u64> s_linked_bursts{0};         // the CPU's: bursts written to the FIFO the GPU reads, since the boot
u64 s_read_bytes = 0;                        // the GPU thread's: bytes of the bursts read, since the boot
std::atomic<u64> s_marks_noted{0}, s_marks_merged{0}, s_marks_dropped{0};
u64 s_marks_exact = 0, s_marks_late = 0, s_marks_inside = 0;  // the GPU thread's: placed at a command's edge, before, inside

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
    // The shader compiler is loaded first and never freed, so it initialises before Dawn and detaches after it
    // (Windows detaches in reverse order of initialisation). Dawn would load it last, and at process exit a DXC object
    // it still held was released after dxcompiler.dll had detached: DXC's own assert in DxcSwapThreadMalloc, a
    // breakpoint at dxcompiler.dll+0x10750f, and WerFault at every Aurora run's end. Dawn's load finds this one.
    // AURORA_DXC_FIRST=0 leaves it to Dawn (the old order, for a comparison run only).
    const char* const dxc_first = std::getenv("AURORA_DXC_FIRST");
    if (!(dxc_first && dxc_first[0] == '0'))
    {
      const size_t slash = wide.find_last_of(L"\\/");
      const std::wstring dxc = (slash == std::wstring::npos ? std::wstring() : wide.substr(0, slash + 1)) + L"dxcompiler.dll";
      const HMODULE loaded = LoadLibraryExW(dxc.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      std::fprintf(stderr, "[aurora-present] dxcompiler.dll loaded first: %s\n", loaded ? "yes" : "no");
    }
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
      !Bind(&s_lib.stats, "aurora_present_stats") || !Bind(&s_lib.stop, "aurora_present_stop") ||
      !Bind(&s_lib.overlay_texture, "aurora_present_overlay_texture") ||
      !Bind(&s_lib.overlay_forget, "aurora_present_overlay_forget") ||
      !Bind(&s_lib.overlay_frame, "aurora_present_overlay_frame") ||
      !Bind(&s_lib.picture_in_window, "aurora_present_picture_in_window") ||
      !Bind(&s_lib.picture_shape, "aurora_present_picture_shape") ||
      !Bind(&s_lib.model_file, "aurora_present_model_file"))
  {
    s_load_problem = fmt::format("{} lacks an aurora_present function", path);
    s_lib = {};
    return false;
  }
  Bind(&s_lib.frame_between, "aurora_present_frame_between");
  Bind(&s_lib.frame_at, "aurora_present_frame_at");
  Bind(&s_lib.redraw, "aurora_present_redraw");
  Bind(&s_lib.scene, "aurora_present_scene");
  Bind(&s_lib.step, "aurora_present_step");
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
  if (s_video_off_window)
  {
    // Dolphin draws nothing: Aurora presents into the window, at its size when Aurora starts
    c.window = s_video_off_window;
#ifdef _WIN32
    RECT client{};
    if (GetClientRect(static_cast<HWND>(s_video_off_window), &client) && client.right > 0 && client.bottom > 0)
    {
      c.width = static_cast<u32>(client.right);
      c.height = static_cast<u32>(client.bottom);
    }
#endif
  }
  // the same internal resolution as the emulator's own, unless the host asks for another
  c.scale = s_settings.scale ? s_settings.scale :
                               (g_framebuffer_manager ? g_framebuffer_manager->GetEFBScale() : 1);
  if (!s_lib.start(&c))
  {
    s_load_problem = "the library's interface is not this emulator's (ABI 5)";
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
  return fmt::format("aurora: picture size {}, {} draws left out (buffers full){}; {} frames in ({} KB), {} drawn, {} pictures, {} shown; presents {}: {} with Aurora's "
                     "picture, {} the emulator's own; picture behind by 0..9+ copies: {}; most frames waiting {}; "
                     "Aurora's thread a frame (medians): feed {:.2f} ms, present {:.2f} ms, readback {:.2f} ms{}{}{}{}",
                     st.scale != 0 ? fmt::format("{} times the game's", st.scale) : std::string("the window's pixels"), st.overflow_drops, st.fatal ? ", STOPPED BY A FATAL ERROR" : "", st.frames_in, st.bytes_in / 1024, st.frames_drawn, st.pictures, s_pictures_taken, s_presents,
                     s_presents - s_presents_own, s_presents_own, lag, st.most_behind, st.feed_ms_median,
                     st.present_ms_median, st.readback_ms_median, st.stopped ? "; STOPPED (fell behind or failed)" : "",
                     st.failed ? "; Aurora could not start" : "",
                     s_bad_cuts ? fmt::format("; {} frame cuts outside the stream", s_bad_cuts) : "",
                     fmt::format("; {} display lists put in place of their calls ({} KB), {} cut short", s_lists,
                                 s_list_bytes / 1024, s_cut_lists)) +
         fmt::format("; draw marks: {} noted ({} at the place of the one before, {} dropped), {} placed at their command's edge, "
                     "{} later, {} inside a command",
                     s_marks_noted.load(), s_marks_merged.load(), s_marks_dropped.load(), s_marks_exact, s_marks_late,
                     s_marks_inside);
}

// Aurora's thread ended (it reads the guest's RAM: before the RAM goes), the report kept.
void StopAll()
{
  s_on = false;
  if (s_started)
  {
    s_report = MakeReport();
    s_report += fmt::format("; hand-off on the decoding thread {:.0f} ms over {} copies ({:.3f} ms a copy)",
                            s_handoff_ns / 1e6, s_copies, s_copies ? s_handoff_ns / 1e6 / s_copies : 0.0);
    s_lib.stop();
    s_started = false;
  }
  if (s_shown_log)
  {
    std::fclose(s_shown_log);
    s_shown_log = nullptr;
  }
  if (s_test_table)
  {
    std::fclose(s_test_table);
    s_test_table = nullptr;
  }
  if (FzgxFifoTest::On())
    std::fprintf(stderr, "fzgx11: Aurora's feed: lists changed under the read %llu, calls left in the stream %llu, nested %llu\n",
                 static_cast<unsigned long long>(s_test_changed), static_cast<unsigned long long>(s_test_kept),
                 static_cast<unsigned long long>(s_test_nested));
}
}  // namespace

bool g_pending = false;

bool WantsModelFiles()
{
  return s_on.load(std::memory_order_relaxed) && s_lib.model_file != nullptr;
}

bool WantsWholeDisc()
{
  static const bool wanted = [] {
    const char* rt = std::getenv("AURORA_RT");
    const char* asked = std::getenv("AURORA_WHOLE_DISC");
    return (rt != nullptr && (rt[0] == '1' || rt[0] == '2')) || (asked != nullptr && asked[0] == '1');
  }();
  return wanted && WantsModelFiles();
}

void ModelFile(const std::string& path, const u8* bytes, u64 size, bool later)
{
  if (WantsModelFiles())
    s_lib.model_file(path.c_str(), bytes, size, later ? 1 : 0);
}

void Scene(bool racing, const std::string& where)
{
  if (s_on.load(std::memory_order_relaxed) && s_lib.scene != nullptr)
    s_lib.scene(racing ? 1 : 0, where.c_str());
}

void Step(u32 step)
{
  if (s_on.load(std::memory_order_relaxed) && s_lib.step != nullptr)
    s_lib.step(step);
}

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
  s_lists = s_list_bytes = s_cut_lists = 0;
  if (!Load(settings.library))
  {
    ERROR_LOG_FMT(VIDEO, "AuroraPresent: {}; the emulator's own pictures are shown", s_load_problem);
    return;
  }
  s_on = true;
}

void Redraw(const RedrawRequest* request)
{
  if (s_started && s_lib.redraw)
    s_lib.redraw(request);
}

void OnBurstWritten(bool linked)
{
  if (linked)
    s_linked_bursts.fetch_add(1, std::memory_order_relaxed);
}

void NoteMark(u32 tag)
{
  if (!s_on.load(std::memory_order_relaxed))
    return;
  auto& system = Core::System::GetInstance();
  const u64 at = s_linked_bursts.load(std::memory_order_relaxed) * GPFifo::GATHER_PIPE_SIZE +
                 system.GetGPFifo().GetGatherPipeCount();
  s_marks_noted.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard guard(s_marks_lock);
  if (!s_marks.empty() && s_marks.back().position == at)
  {
    s_marks.back().tag = tag;  // nothing drawn between the two: the later holds
    s_marks_merged.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (s_marks.size() >= MOST_MARKS)
  {
    s_marks_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  s_marks.push_back({at, tag});
  if (s_marks.size() == 1)
    s_mark_first.store(at, std::memory_order_relaxed);
}

namespace
{
// OnCommandEnd: every mark up to the end of the command just read goes into Aurora's bytes: before the command when its
// place is at or before the command's start (it was noted before the command was written), else after it. `end`: the
// command's end in the stream; `end_at`: the same place in s_pending; `occupies`: the command's bytes there (a list's
// in place of its call).
void PlaceMarks(u64 end, size_t end_at, u32 size, size_t occupies)
{
  const u64 start = end - size;
  size_t start_at = end_at - occupies;
  std::lock_guard guard(s_marks_lock);
  while (!s_marks.empty() && s_marks.front().position <= end)
  {
    const Mark m = s_marks.front();
    s_marks.pop_front();
    const u8 bytes[7] = {GX_AURORA_COMMAND, static_cast<u8>(SET_DRAW_TAG >> 8), static_cast<u8>(SET_DRAW_TAG & 0xFF),
                         static_cast<u8>(m.tag >> 24), static_cast<u8>(m.tag >> 16), static_cast<u8>(m.tag >> 8),
                         static_cast<u8>(m.tag)};
    if (m.position <= start)
    {
      s_pending.insert(s_pending.begin() + start_at, bytes, bytes + sizeof(bytes));
      start_at += sizeof(bytes);
      end_at += sizeof(bytes);
      ++(m.position == start ? s_marks_exact : s_marks_late);
    }
    else
    {
      s_pending.insert(s_pending.begin() + end_at, bytes, bytes + sizeof(bytes));
      end_at += sizeof(bytes);
      ++(m.position == end ? s_marks_exact : s_marks_inside);
    }
  }
  s_mark_first.store(s_marks.empty() ? ~u64{0} : s_marks.front().position, std::memory_order_relaxed);
}
}  // namespace

void NoteBetween(u64 position, float fraction)
{
  if (!s_started)
    return;
  std::lock_guard guard(s_between_lock);
  if (s_between.size() < 64)
    s_between.push_back({position, fraction});
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
  s_read_bytes += size;  // (every burst, Aurora on or not: the marks' places count from the boot, as the CPU's bursts do)
  if (!s_on.load(std::memory_order_relaxed))
    return;
  if (!s_started)
  {
    Start();
    if (!s_started)
      return;
  }
  HandoffTimer timer;
  s_pending.insert(s_pending.end(), bytes, bytes + size);
  s_fed_end = bytes + size;
  if (s_read_bytes > s_mark_first.load(std::memory_order_relaxed))
    g_pending = true;  // a mark's place has been read: the commands from here on are looked at (OnCommandEnd)
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
  s_cut_position = FrameRedraw::ReadPosition();
  {
    std::lock_guard guard(s_between_lock);
    s_cut_fraction = 0;
    if (!s_between.empty() && FrameRedraw::ReadPosition() > s_between.front().position)
    {
      s_cut_fraction = s_between.front().fraction;
      s_between.pop_front();
    }
  }
  g_pending = true;
  if (!s_settings.shot_prefix.empty())
    FrameRedraw::TakeStills(s_copies, false);
}

void OnDisplayList(const u8* bytes, u32 size)
{
  if (!s_started || !bytes || !size)
    return;
  if (FzgxFifoTest::On())
  {
    if (s_list && s_test_nested++ < 20)
      std::fprintf(stderr, "fzgx11: a list called inside a list: size %u inside size %u\n", size, s_list_size);
    s_test_list.assign(bytes, bytes + size);
  }
  s_list = bytes;
  s_list_size = size;
  g_pending = true;
}

void OnListRead(u32 good)
{
  if (s_list && good < s_list_size)
  {
    if (s_cut_lists++ < 20)
      std::fprintf(stderr, "[aurora-present] list at RAM+%#zx: only %u of %u bytes are commands (copy %llu)\n",
                   static_cast<size_t>(s_list - Core::System::GetInstance().GetMemory().GetRAM()), good, s_list_size,
                   static_cast<unsigned long long>(s_copies));
    s_list_size = good;
  }
}

void OnCommandEnd(const u8* data, u32 size)
{
  g_pending = false;
  if (!s_started)
    return;
  HandoffTimer timer;
  const u8* end = data + size;
  // where the command is among the bytes handed in: `tail` bytes after it (the next commands' first)
  size_t tail = 0;
  const bool inside = s_fed_end && data && end <= s_fed_end &&
                      static_cast<size_t>(s_fed_end - data) <= s_pending.size();
  if (inside)
    tail = static_cast<size_t>(s_fed_end - end);
  // the marks' look goes on while a mark's place lies in what was read
  struct Rearm
  {
    ~Rearm()
    {
      if (s_read_bytes > s_mark_first.load(std::memory_order_relaxed))
        g_pending = true;
    }
  } rearm;
  size_t occupies = size;  // the command's bytes in s_pending: a list's in place of its call
  if (s_list)
  {
    if (FzgxFifoTest::On())
    {
      if (std::memcmp(s_test_list.data(), s_list, std::min<size_t>(s_test_list.size(), s_list_size)) != 0)
      {
        size_t off = 0;
        while (s_test_list[off] == s_list[off])
          ++off;
        if (s_test_changed++ < 60)
          std::fprintf(stderr, "fzgx11: list at RAM+%#zx size %#x changed while the emulator read it, first at %#zx; copy %llu\n",
                       static_cast<size_t>(s_list - Core::System::GetInstance().GetMemory().GetRAM()), s_list_size, off,
                       static_cast<unsigned long long>(s_copies));
      }
      if (!(inside && size == 9) && s_test_kept++ < 60)
        std::fprintf(stderr, "fzgx11: a call left in the stream (Aurora will read RAM late): inside %d size %u list size %#x copy %llu\n",
                     inside, size, s_list_size, static_cast<unsigned long long>(s_copies));
      static const char* table = std::getenv("AURORA_TEST_LISTS");
      if (table && !s_test_table)
        s_test_table = std::fopen(table, "wb");
      if (s_test_table && inside && size == 9)
      {
        const u64 rec[3] = {s_copies, s_pending.size() - tail - size,
                            (static_cast<u64>(s_list - Core::System::GetInstance().GetMemory().GetRAM()) << 32) | s_list_size};
        std::fwrite(rec, sizeof(rec), 1, s_test_table);
      }
    }
    // the call (its 9 bytes) gives its place to the list's bytes
    if (inside && size == 9)
    {
      const size_t at = s_pending.size() - tail - size;
      s_pending.erase(s_pending.begin() + at, s_pending.begin() + at + size);
      s_pending.insert(s_pending.begin() + at, s_list, s_list + s_list_size);
      occupies = s_list_size;
      ++s_lists;
      s_list_bytes += s_list_size;
    }
    s_list = nullptr;
    s_list_size = 0;
  }
  if (inside && s_read_bytes > s_mark_first.load(std::memory_order_relaxed))
    PlaceMarks(s_read_bytes - tail, s_pending.size() - tail, size, occupies);
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
  if (s_lib.frame_at)
    s_lib.frame_at(s_cut_position);
  if (s_cut_fraction > 0 && s_lib.frame_between)
    s_lib.frame_between(s_cut_copy, s_cut_fraction);
  s_lib.frame(s_cut_copy);
  // the picture's shape, as the emulator's own video would show it (the host sets it from the game's screen option)
  if (s_video_off_window)
  {
    const AspectMode mode = Config::Get(Config::GFX_ASPECT_RATIO);
    static AspectMode s_shape_sent = AspectMode::Custom;
    if (mode != s_shape_sent)
    {
      s_shape_sent = mode;
      s_lib.picture_shape(mode == AspectMode::ForceStandard ? 4.0f / 3.0f : mode == AspectMode::Stretch ? 0.0f : 16.0f / 9.0f);
    }
  }
  // Dolphin presents nothing and draws no statistics then: its frame rate counts Aurora's presents, the true frames'
  // and the pictures drawn between them from their draws (the window strip's FPS)
  if (s_video_off_window)
  {
    LibStats st{};
    s_lib.stats(&st);
    auto& metrics = Core::System::GetInstance().GetPerfMetrics();
    for (; s_presents_counted < st.frames_drawn + st.pictures_between; ++s_presents_counted)
      metrics.CountFrame();
    metrics.UpdateCounters();
  }
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

bool CountsFrames()
{
  return s_video_off_window && On();
}

bool VideoOffSize(u32* width, u32* height)
{
  if (!CountsFrames())
    return false;
#ifdef _WIN32
  RECT client{};
  if (!GetClientRect(static_cast<HWND>(s_video_off_window), &client) || client.right <= 0 || client.bottom <= 0)
    return false;
  *width = static_cast<u32>(client.right);
  *height = static_cast<u32>(client.bottom);
  return true;
#else
  return false;
#endif
}

bool PictureInWindow(MathUtil::Rectangle<int>* rect)
{
  s32 r[4];
  u32 width = 0, height = 0;
  if (!s_started || !s_lib.picture_in_window || !VideoOffSize(&width, &height) || !s_lib.picture_in_window(width, height, r) ||
      r[2] <= 0 || r[3] <= 0)
    return false;
  *rect = MathUtil::Rectangle<int>(r[0], r[1], r[0] + r[2], r[1] + r[3]);
  return true;
}

// The presenter's thread. Aurora starts at the GPU's first burst, after the overlay has made its fonts: the textures
// are sent whole the first frame after that, and only their changes from then on.
namespace
{
bool s_overlay_sent = false;
bool s_overlay_smooth = false;
std::vector<AuroraOverlayDraw> s_overlay_draws;
std::vector<ImDrawVert> s_overlay_vertices;
std::vector<ImDrawIdx> s_overlay_indices;

void SendWhole(ImTextureData* tex)
{
  s_lib.overlay_texture(static_cast<u64>(tex->GetTexID()), tex->Width, tex->Height, 0, 0, tex->Width, tex->Height,
                        static_cast<const u8*>(tex->GetPixels()), tex->GetPitch());
}
}  // namespace

void OverlayTexture(ImTextureData* tex, int status, u64 id)
{
  if (!s_started || !s_overlay_sent)
    return;
  if (tex->Status == ImTextureStatus_Destroyed)
    s_lib.overlay_forget(id);
  else if (status == ImTextureStatus_WantCreate)
    SendWhole(tex);
  else if (status == ImTextureStatus_WantUpdates)
    for (const ImTextureRect& r : tex->Updates)
      s_lib.overlay_texture(static_cast<u64>(tex->GetTexID()), tex->Width, tex->Height, r.x, r.y, r.w, r.h,
                            static_cast<const u8*>(tex->GetPixelsAt(r.x, r.y)), tex->GetPitch());
}

void OverlaySampler(bool smooth)
{
  s_overlay_smooth = smooth;
}

void OverlayFrame(const ImDrawData* data, u32 width, u32 height)
{
  if (!s_started)
    return;
  if (!s_overlay_sent && data->Textures != nullptr)
  {
    s_overlay_sent = true;
    for (ImTextureData* tex : *data->Textures)
      if (tex->Status == ImTextureStatus_OK && tex->GetTexID() != ImTextureID_Invalid)
        SendWhole(tex);
  }
  static_assert(sizeof(ImDrawVert) == sizeof(AuroraOverlayVertex) && sizeof(ImDrawIdx) == sizeof(u16));
  s_overlay_draws.clear();
  s_overlay_vertices.clear();
  s_overlay_indices.clear();
  s_overlay_smooth = false;  // (each frame starts nearest, as OnScreenUI::DrawImGui)
  for (int i = 0; i < data->CmdListsCount; ++i)
  {
    const ImDrawList* list = data->CmdLists[i];
    const s32 list_vertex = static_cast<s32>(s_overlay_vertices.size());
    const u32 list_index = static_cast<u32>(s_overlay_indices.size());
    s_overlay_vertices.insert(s_overlay_vertices.end(), list->VtxBuffer.begin(), list->VtxBuffer.end());
    s_overlay_indices.insert(s_overlay_indices.end(), list->IdxBuffer.begin(), list->IdxBuffer.end());
    for (const ImDrawCmd& cmd : list->CmdBuffer)
    {
      if (cmd.UserCallback != nullptr)
      {
        if (cmd.UserCallback != ImDrawCallback_ResetRenderState)
          cmd.UserCallback(list, &cmd);
        continue;
      }
      if (cmd.ElemCount == 0)
        continue;
      s_overlay_draws.push_back({static_cast<u64>(cmd.GetTexID()),
                                 {cmd.ClipRect.x, cmd.ClipRect.y, cmd.ClipRect.z, cmd.ClipRect.w},
                                 list_index + cmd.IdxOffset,
                                 cmd.ElemCount,
                                 list_vertex + static_cast<s32>(cmd.VtxOffset),
                                 s_overlay_smooth ? 1u : 0u});
    }
  }
  const AuroraOverlayFrame frame{width,
                                 height,
                                 reinterpret_cast<const AuroraOverlayVertex*>(s_overlay_vertices.data()),
                                 static_cast<u32>(s_overlay_vertices.size()),
                                 s_overlay_indices.data(),
                                 static_cast<u32>(s_overlay_indices.size()),
                                 s_overlay_draws.data(),
                                 static_cast<u32>(s_overlay_draws.size())};
  s_lib.overlay_frame(&frame);
}

u64 OverlayCopyTexture(u32 physical)
{
  return 0xA500000000000000ull | physical;
}

void SetVideoOffWindow(void* window)
{
  s_video_off_window = window;
  s_presents_counted = 0;
}

void Shutdown()
{
  StopAll();
  s_video_off_window = nullptr;
  s_texture.reset();
}
}  // namespace VideoCommon::AuroraPresent
