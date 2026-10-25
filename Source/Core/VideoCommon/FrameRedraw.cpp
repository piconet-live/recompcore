// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/FrameRedraw.h"
#include "VideoCommon/FrameRedrawFrame.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "Common/Logging/Log.h"
#include "Core/System.h"
#include "VideoCommon/AbstractFramebuffer.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractStagingTexture.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/ConstantManager.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/Present.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VertexShaderManager.h"
#include "VideoCommon/VideoBackendBase.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/CommandProcessor.h"
#include "VideoCommon/NativeVertexFormat.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/XFMemory.h"

namespace VideoCommon::FrameRedraw
{
namespace
{

std::atomic<bool> s_on{false};
std::mutex s_settings_mutex;
Settings s_settings, s_pending;
bool s_settings_changed = false;
FILE* s_pairs = nullptr;  // with a log: a line for every taken request (its pairing)
std::mutex s_stats_mutex;
Stats s_stats;
FILE* s_log = nullptr;
u32 s_dumps = 0;
u64 s_over_budget = 0;  // frames whose replays stopped at the budget
bool s_idle = false;  // the frame being counted was replayed while the FIFO loop had nothing to read
Frame s_closed;         // the frame the last XFB copy ended, until it is replayed
MathUtil::Rectangle<int> s_closed_rect;
bool s_closed_waits = false;  // s_closed is still to be replayed
std::optional<Common::Matrix44> s_turn;  // the spike's camera correction (Settings::turn_degrees)
// The high refresh rate's requests (Queue) and the command stream's position on both sides of the FIFO
std::mutex s_queue_mutex;
std::deque<Request> s_queue;
std::atomic<u64> s_written{0};
std::atomic<u64> s_read{0};
// The request the last XFB copy took for the frame it closed (s_closed), until its replays between two passes
std::optional<Request> s_redraw;
bool s_redraw_waits = false;  // (a plain flag: BetweenPasses is asked after every burst)
double s_replay_us = 1500;  // the last replay's video thread time: when the next picture would be ready
double s_replay_mean_us = 1500;  // the replays' video thread time, a running mean: how many fit the budget

// The pictures of a request that fit (`fit` of its `n` fractions), evenly spread across the step: every second, third
// ... of them when that divides the step's refreshes (n + 1) evenly (a display of 360 Hz, 5 between a step: 2, one at
// every second refresh, when 5 do not fit), else the nearest to an even spread. Never the first few and a gap: that
// shows as a picture held for several refreshes once a step.
std::vector<float> Even(const std::vector<float>& fractions, size_t fit)
{
  const size_t n = fractions.size();
  if (fit >= n)
    return fractions;
  std::vector<float> out;
  for (size_t m = fit; m >= 1 && out.empty(); --m)
  {
    if ((n + 1) % (m + 1) != 0)
      continue;
    const size_t every = (n + 1) / (m + 1);
    for (size_t j = 1; j <= m; ++j)
      out.push_back(fractions[j * every - 1]);
  }
  if (out.empty() && fit >= 1)
  {
    for (size_t j = 1; j <= fit; ++j)
    {
      const size_t at = static_cast<size_t>(std::lround(double(j) * double(n + 1) / double(fit + 1))) - 1;
      out.push_back(fractions[std::min(at, n - 1)]);
    }
  }
  return out;
}
u64 s_copies = 0;  // the XFB copies so far (Frame::copy)
FILE* s_stills = nullptr;  // with stills (Settings::stills): a line for every picture saved

// A test's stills (Settings::stills): whether the frame closed by this copy is one, and a picture of it saved.
bool StillOf(u64 copy)
{
  if (s_settings.stills == 0 || s_settings.dump_prefix.empty() || copy < s_settings.stills_at)
    return false;
  const u64 since = copy - s_settings.stills_at;
  return (s_settings.stills_every != 0 ? since % s_settings.stills_every : since) < s_settings.stills;
}

void Still(const Frame& f, const AbstractTexture* texture, u32 k, float planned, float drawn, double cpu_us)
{
  if (!texture || !StillOf(f.copy))
    return;
  MathUtil::Rectangle<int> rect = f.xfb.display;
  rect.ClampUL(0, 0, static_cast<int>(texture->GetWidth()), static_cast<int>(texture->GetHeight()));
  const std::string path = fmt::format("{}-still-{:06}-{}.png", s_settings.dump_prefix, f.copy, k);
  const bool saved = QueueStill(texture, rect, path, s_copies);
  if (!s_stills)
  {
    s_stills = std::fopen((s_settings.dump_prefix + "-stills.csv").c_str(), "w");
    if (s_stills)
      std::fprintf(s_stills, "copy,picture,planned,drawn,replay_us,asked,file\n");
  }
  if (s_stills)
  {
    std::fprintf(s_stills, "%llu,%u,%.4f,%.4f,%.1f,%d,%s\n", static_cast<unsigned long long>(f.copy), k, planned,
                 drawn, cpu_us, saved ? 1 : 0, path.c_str());
    std::fflush(s_stills);
  }
}

void TakeSettings()
{
  std::lock_guard<std::mutex> guard(s_settings_mutex);
  if (!s_settings_changed)
    return;
  s_settings_changed = false;
  if (s_log && s_pending.log_path != s_settings.log_path)
  {
    std::fclose(s_log);
    s_log = nullptr;
  }
  s_settings = s_pending;
  g_keep_xfb = s_settings.stills != 0;
  if (s_settings.hfr && s_settings.budget_ms == 0)
    s_settings.budget_ms = 10;  // (FrameRedraw.h: what 6 and 10 cost)
  s_turn.reset();
  if (s_settings.turn_degrees != 0)
  {
    s_turn = Common::Matrix44::FromMatrix33(
        Common::Matrix33::RotateY(s_settings.turn_degrees * 3.14159265f / 180.0f));
  }
  if (!s_pairs && !s_settings.log_path.empty() && s_settings.hfr)
  {
    s_pairs = std::fopen((s_settings.log_path + ".pairs.csv").c_str(), "w");
    if (s_pairs)
      std::fprintf(s_pairs, "frame,position,read,stale,before_last,before_draws,closed_first,closed_last,"
                            "closed_draws,pairing\n");
  }
  if (!s_log && !s_settings.log_path.empty())
  {
    s_log = std::fopen(s_settings.log_path.c_str(), "w");
    if (s_log)
      std::fprintf(s_log, "frame,draws,clears,copies,reinterprets,vertex_bytes,index_bytes,vs_blocks,"
                          "gs_blocks,ps_blocks,record_us,replays,replay_us,replay_us_most,gpu_us,"
                          "diff_rgb,diff_rgba,pixels,at_us,idle\n");
  }
}

}  // namespace

void Configure(const Settings& settings)
{
  std::lock_guard<std::mutex> guard(s_settings_mutex);
  s_pending = settings;
  s_settings_changed = true;
  s_on = settings.replays != 0 || settings.hfr;
}

bool Recording()
{
  return !g_replaying && s_on.load(std::memory_order_relaxed);
}

void Queue(Request request)
{
  std::lock_guard<std::mutex> guard(s_queue_mutex);
  if (s_queue.size() >= 8)  // (requests no draw-done came for: the oldest go)
    s_queue.pop_front();
  s_queue.push_back(std::move(request));
}

u64 WrittenPosition()
{
  return s_written.load(std::memory_order_relaxed);
}

void OnBurstWritten()
{
  s_written.fetch_add(1, std::memory_order_relaxed);
}

void OnBurstRead()
{
  s_read.fetch_add(1, std::memory_order_relaxed);
}

u64 ReadPosition()
{
  return s_read.load(std::memory_order_relaxed);
}

namespace
{
void Count(const Frame& f, const std::vector<Replayed>& done);

// The game has handed the GPU commands it has not read yet (its next command list, or a linked list being written).
// A replay then would put the game's draws behind where they would be without the high refresh rate, and the game
// reuses memory its list points to on that timing: it streams data from ARAM over display lists of a list about
// 0.7 ms after handing that list over (the high-refresh redraw notes, "Stage 5: the fault"). So the replays wait for
// the list to be read to its end and give way to the next at once.
bool GameHasWork()
{
  const auto& fifo = Core::System::GetInstance().GetCommandProcessor().GetFifo();
  return fifo.bFF_GPReadEnable.load(std::memory_order_relaxed) != 0 &&
         fifo.CPReadWriteDistance.load(std::memory_order_relaxed) != 0;
}

// The high refresh rate: the closed frame drawn again for each of the request's pictures, each a guess forward from
// the step for the time it will be shown at, each picture handed to the presenter; the rest left out as soon as the
// game has more for the GPU (GameHasWork).
void ReplayForHost(const Request& q, bool idle)
{
  s_closed_waits = false;
  const Frame& f = s_closed;
  std::vector<Replayed> done;
  if (!f.whole || f.draws.empty() || !f.xfb.valid || !q.predict)
  {
    Count(f, done);
    return;
  }
  Corrections c;
  std::vector<u32> by_vertex;
  const TagCounts counts = Tag(f, q, s_settings.radius, &c, &by_vertex);
  // (Vulkan binds its stream buffers whole, the vertex stride is the pipeline's: a draw's vertices may be drawn
  // again from where the first picture put them)
  const std::string backend = g_video_backend ? g_video_backend->GetConfigName() : std::string();
  c.reuse = backend == "Vulkan" || backend == "Null";
  {
    std::lock_guard<std::mutex> guard(s_stats_mutex);
    s_stats.tag_world += counts.world;
    s_stats.tag_machine += counts.machine;
    s_stats.tag_flat += counts.flat;
    s_stats.tag_offscreen += counts.offscreen;
    s_stats.tag_vertex += counts.vertex;
    s_stats.tag_over += counts.over_machine;
    s_stats.tag_whole += counts.whole;
  }
  u64 moved = 0;
  const u64 number = s_stats.frames;
  if (s_settings.draws_at != 0 && number + 1 == s_settings.draws_at && !s_settings.dump_prefix.empty())
    WriteDraws(f, q, c, fmt::format("{}-draws-{}.csv", s_settings.dump_prefix, number + 1));
  if (StillOf(f.copy) && !StillOf(f.copy - 1))  // (each group of stills' first frame: its draws too)
    WriteDraws(f, q, c, fmt::format("{}-draws-c{:06}.csv", s_settings.dump_prefix, f.copy));
  // Late latching: the step's own picture's moment (the presenter's intended time for it, when it holds pictures
  // for their time); a picture that would be ready after its planned time is guessed for the time it will be
  // ready at, and one past the next picture's planned time or the next step's is left out
  const bool holds = g_presenter && g_presenter->HoldsPictures();
  const TimePoint moment = holds ? g_presenter->LastTrueMoment() : TimePoint{};
  const double field_us = s_settings.field_us > 0 ? s_settings.field_us : 16683.0;
  const double budget_us = s_settings.budget_ms * 1000.0;
  double spent_us = 0;
  u64 pictures = 0, left = 0, latched = 0, late = 0, yielded = 0, stopped = 0;
  // as many as fit the budget at the replays' recent cost, and the cap, evenly spread (Even); the budget's own rule
  // below still holds every picture to the time left
  const size_t fit =
      std::min<size_t>(s_settings.replays,
                       budget_us > 0 ? std::max<size_t>(1, static_cast<size_t>(budget_us / std::max(1.0, s_replay_mean_us))) :
                                       q.fractions.size());
  const std::vector<float> fractions = Even(q.fractions, fit);
  const size_t n = fractions.size();
  for (size_t k = 0; k < n; ++k)
  {
    if (GameHasWork())
    {
      yielded = n - k;
      break;
    }
    if (!done.empty() && budget_us > 0 && spent_us + done.back().cpu_us > budget_us)
    {
      left = n - k;
      ++s_over_budget;
      break;
    }
    float fraction = fractions[k];
    if (holds)
    {
      const double ready = (Us(Clock::now() - moment) + s_replay_us) / field_us;
      if (ready > fraction)
      {
        if (ready > 0.98 || (k + 1 < n && ready >= fractions[k + 1]))
        {
          ++late;  // (the next picture, or the next step's own, is shown by then)
          continue;
        }
        fraction = static_cast<float>(ready);
        ++latched;
      }
    }
    InBetween b;
    q.predict(fraction, &b);
    b.fraction = fraction;
    Fill(&c, q, b);
    moved += Move(f, q, b, by_vertex, &c);
    RcTcacheEntry picture;
    Replayed r = Replay(f, nullptr, number, {}, nullptr, &c, &picture, &GameHasWork);
    if (r.stopped || !picture)
    {
      if (r.stopped || GameHasWork())
      {
        yielded = n - k;
        stopped += r.stopped;
        break;
      }
    }
    done.push_back(r);
    spent_us += done.back().cpu_us;
    s_replay_us = done.back().cpu_us;
    s_replay_mean_us += 0.1 * (s_replay_us - s_replay_mean_us);
    if (picture)
      Still(f, picture->texture.get(), static_cast<u32>(k + 1), fractions[k], fraction, r.cpu_us);
    if (picture && g_presenter)
    {
      g_presenter->PresentInBetween(std::move(picture), f.xfb.display, fraction);
      ++pictures;
    }
    if (g_presenter)
      g_presenter->PresentHeldSwap();  // (a picture whose time came while this one was drawn)
  }
  {
    std::lock_guard<std::mutex> guard(s_stats_mutex);
    ++s_stats.requests;
    s_stats.idle += idle;
    s_stats.pictures += pictures;
    s_stats.budget_left += left;
    s_stats.thinned += fractions.size() < q.fractions.size();
    s_stats.thinned_out += q.fractions.size() - fractions.size();
    s_stats.latched += latched;
    s_stats.late += late;
    s_stats.yielded += yielded;
    s_stats.stopped += stopped;
    s_stats.vertices_moved += moved;
  }
  s_idle = idle;
  Count(f, done);
  s_idle = false;
}

// Replays the closed frame (s_closed) and counts.
void ReplayClosed()
{
  s_closed_waits = false;
  const Frame& f = s_closed;
  const MathUtil::Rectangle<int>& rect = s_closed_rect;
  const u64 number = s_stats.frames;
  const u32 replays = s_settings.replays;
  // (at the draw-done the true EFB is no longer the frame's: compares only at the copy)
  const bool compare = s_settings.compare_every != 0 && !s_settings.at_draw_done && f.whole &&
                       number % s_settings.compare_every == 0;

  std::vector<Replayed> done;
  double spent_us = 0;
  Corrections turn;  // the spike's: every perspective draw turned
  if (s_turn)
  {
    turn.matrices.push_back(*s_turn);
    for (const Draw& d : f.draws)
      turn.tags.push_back(f.perspective[d.vs] ? 0 : -1);
  }
  if (f.whole && !f.draws.empty())
  {
    for (u32 k = 0; k < replays; ++k)
    {
      // the budget: one more replay only while it fits, at the cost of the last (the first is always made)
      const double budget_us = s_settings.budget_ms * 1000.0;
      if (k > 0 && budget_us > 0 && spent_us + done.back().cpu_us > budget_us)
      {
        ++s_over_budget;
        break;
      }
      done.push_back(Replay(f, compare && k == 0 && rect.GetWidth() > 0 ? &rect : nullptr, number,
                            s_settings.dump_prefix, &s_dumps, s_turn ? &turn : nullptr));
      spent_us += done.back().cpu_us;
    }
  }

  Count(f, done);
}

// The frame's line in the log and its counts.
void Count(const Frame& f, const std::vector<Replayed>& done)
{
  const u64 number = s_stats.frames;
  double sum = 0, most = 0;
  for (const Replayed& r : done)
  {
    sum += r.cpu_us;
    most = std::max(most, r.cpu_us);
  }
  const Replayed first = done.empty() ? Replayed{} : done[0];
  const u64 bytes = f.vertices.size() + f.indices.size() * 2;
  if (s_log)
  {
    // (at_us: the steady clock's microseconds when the frame's replays ended, the pacing log's clock on Windows)
    const long long at_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
    std::fprintf(s_log, "%llu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.1f,%zu,%.1f,%.1f,%.1f,%lld,%lld,%lld,%lld,%d\n",
                 static_cast<unsigned long long>(number), f.draws.size(), f.clears.size(),
                 f.copies.size(), f.reinterprets.size(), f.vertices.size(), f.indices.size() * 2,
                 f.vs.size(), f.gs.size(), f.ps.size(), f.record_us, done.size(),
                 done.empty() ? 0.0 : sum / done.size(), most, first.gpu_us,
                 static_cast<long long>(first.diff_rgb), static_cast<long long>(first.diff_rgba),
                 static_cast<long long>(first.pixels), at_us, s_idle ? 1 : 0);
  }
  {
    std::lock_guard<std::mutex> guard(s_stats_mutex);
    Stats& s = s_stats;
    ++s.frames;
    if (done.empty())
    {
      ++s.skipped;
    }
    else
    {
      ++s.replayed;
      s.replays += done.size();
      s.draws += f.draws.size();
      for (const Draw& d : f.draws)
        s.perspective += f.perspective[d.vs];
      s.most_draws = std::max<u64>(s.most_draws, f.draws.size());
      s.bytes += bytes;
      s.most_bytes = std::max(s.most_bytes, bytes);
      s.clears += f.clears.size();
      s.copies += f.copies.size();
      s.reinterprets += f.reinterprets.size();
      s.blocks += f.vs.size() + f.gs.size() + f.ps.size();
      s.record_us.push_back(static_cast<float>(f.record_us));
      for (const Replayed& r : done)
      {
        s.replay_us.push_back(static_cast<float>(r.cpu_us));
        s.uniforms_us += r.uniforms_us;
        s.vertices_us += r.vertices_us;
        s.draws_us += r.draws_us;
        s.copies_us += r.copies_us;
        s.picture_us += r.picture_us;
        s.uniform_uploads += r.uniform_uploads;
        s.uniform_bytes += r.uniform_bytes;
        s.vertex_bytes += r.vertex_bytes;
        s.vertex_reused += r.vertex_reused;
        ++s.parted;
      }
      if (first.gpu_us >= 0)
        s.gpu_us.push_back(static_cast<float>(first.gpu_us));
      if (first.diff_rgb >= 0)
      {
        ++s.compares;
        s.identical += first.diff_rgb == 0;
        if (first.diff_rgb > 0 && s.first_diff_frame == 0)
          s.first_diff_frame = number;
        s.most_diff = std::max<u64>(s.most_diff, static_cast<u64>(first.diff_rgb));
      }
    }
  }
}
}  // namespace

namespace
{
// The request for the frame the XFB copy just closed (s_closed): the one made after the frame before's draws
// (`before_last`, the stream's bursts read at its last draw) and before this frame's first. Older ones are stale
// (their frame is gone); among those made between the same two frames (a pass that drew nothing) the newest is
// taken; newer ones wait for their own frame's copy.
void Pair(u64 before_last, bool before_drawn)
{
  std::optional<Request> take;
  u64 stale = 0;
  const u64 read = s_read.load(std::memory_order_relaxed);
  const u64 first = s_closed.draws.empty() ? read : s_closed.first_read;
  {
    std::lock_guard<std::mutex> guard(s_queue_mutex);
    while (!s_queue.empty() && s_queue.front().position < first)
    {
      if (before_drawn && before_last > s_queue.front().position)
      {
        ++stale;
        s_queue.pop_front();
        continue;
      }
      stale += take.has_value();
      take = std::move(s_queue.front());
      s_queue.pop_front();
    }
  }
  std::lock_guard<std::mutex> guard(s_stats_mutex);
  s_stats.stale += stale;
  if (!take)
    return;
  // the check of the pairing: the frame's draws all after the request, the frame before's all before it
  const bool right = !s_closed.draws.empty() && (!before_drawn || before_last <= take->position);
  ++(right ? s_stats.pair_right : s_stats.unframed);
  if (s_pairs)
  {
    std::fprintf(s_pairs, "%llu,%llu,%llu,%llu,%llu,%d,%llu,%llu,%zu,%s\n",
                 static_cast<unsigned long long>(s_stats.frames), static_cast<unsigned long long>(take->position),
                 static_cast<unsigned long long>(read), static_cast<unsigned long long>(stale),
                 static_cast<unsigned long long>(before_last), before_drawn ? 1 : 0,
                 static_cast<unsigned long long>(s_closed.first_read),
                 static_cast<unsigned long long>(s_closed.last_read), s_closed.draws.size(),
                 right ? "right" : "unframed");
  }
  s_redraw = std::move(take);
  s_redraw_waits = true;
}
}  // namespace

void OnXFBCopy(const MathUtil::Rectangle<int>& src_rect)
{
  TakeSettings();
  if (s_closed_waits && !s_settings.hfr)
    ReplayClosed();  // (a frame whose draw-done never came)
  if (s_redraw_waits)
  {
    // (no pass ended between two copies: that frame is not drawn again)
    s_redraw.reset();
    s_redraw_waits = false;
    Count(s_closed, {});
    std::lock_guard<std::mutex> guard(s_stats_mutex);
    ++s_stats.unreplayed;
  }
  MathUtil::Rectangle<int> rect = g_framebuffer_manager->ConvertEFBRectangle(src_rect);
  rect = g_gfx->ConvertFramebufferRectangle(rect, g_framebuffer_manager->GetEFBFramebuffer());
  rect.ClampUL(0, 0, g_framebuffer_manager->GetEFBWidth(), g_framebuffer_manager->GetEFBHeight());
  const u64 before_last = s_closed.last_read;
  const bool before_drawn = !s_closed.draws.empty();
  std::swap(RecordedFrame(), s_closed);
  s_closed_rect = rect;
  s_closed.copy = ++s_copies;
  if (s_closed.xfb.entry)
  {
    Still(s_closed, s_closed.xfb.entry->texture.get(), 0, 0, 0, 0);
    s_closed.xfb.entry.reset();
  }
  if (g_keep_xfb)
    TakeStills(s_copies, false);
  s_closed_waits = true;
  if (s_settings.hfr)
  {
    MarkOffscreen(s_closed, rect);
    Pair(before_last, before_drawn);
    if (!s_redraw_waits)
      s_closed_waits = false;  // (no request: nothing to draw again)
  }
  else if (!s_settings.at_draw_done)
  {
    ReplayClosed();
  }
  // the next frame starts here, after this copy (its clear comes next)
  StartRecord(g_framebuffer_manager->GetPrevPixelFormat());
}

void OnDrawDone()
{
  if (!s_settings.hfr && s_closed_waits)
    ReplayClosed();
}

void BetweenPasses()
{
  // (only once the game's list is read to its end: GameHasWork)
  if (!s_redraw_waits || GameHasWork())
    return;
  s_redraw_waits = false;
  const Request q = std::move(*s_redraw);
  s_redraw.reset();
  // (the game's primitives batched so far are drawn first, as when the loop has nothing to read)
  const bool idle = s_read.load(std::memory_order_relaxed) == s_written.load(std::memory_order_relaxed);
  g_vertex_manager->Flush();
  ReplayForHost(q, idle);
}

void Shutdown()
{
  StartRecord({});
  RecordedFrame().whole = false;
  s_closed.Reset();
  s_closed_waits = false;
  s_redraw.reset();
  s_redraw_waits = false;
  {
    std::lock_guard<std::mutex> guard(s_queue_mutex);
    s_queue.clear();
  }
  ReleaseReplay();
  if (s_log)
  {
    std::fclose(s_log);
    s_log = nullptr;
  }
  if (s_pairs)
  {
    std::fclose(s_pairs);
    s_pairs = nullptr;
  }
  TakeStills(0, true);
  if (s_stills)
  {
    std::fclose(s_stills);
    s_stills = nullptr;
  }
  s_copies = 0;
  std::lock_guard<std::mutex> guard(s_settings_mutex);
  s_settings_changed = true;  // (a log reopened by the next start, as it was configured)
}

std::string Report()
{
  std::lock_guard<std::mutex> guard(s_stats_mutex);
  return Summary(s_stats, s_over_budget, g_video_backend ? g_video_backend->GetDisplayName() : "?");
}
}  // namespace VideoCommon::FrameRedraw