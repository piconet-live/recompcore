// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw's summary for the host's log (FrameRedraw.h, Report).

#include "VideoCommon/FrameRedrawFrame.h"

#include <algorithm>

#include <fmt/format.h>

namespace VideoCommon::FrameRedraw
{
namespace
{
float Percentile(std::vector<float> v, double p)
{
  if (v.empty())
    return 0;
  const size_t k = std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}
}  // namespace

std::string Summary(const Stats& s, u64 over_budget, const std::string& backend)
{
  if (s.frames == 0)
    return {};
  const double n = s.replayed ? static_cast<double>(s.replayed) : 1.0;
  return fmt::format(
      "frame redraw ({}): {} frames, {} replayed {} times, {} not (the first, empty); a frame: {:.0f} draws "
      "(most {}), {:.2f} MB of vertices and indices (most {:.2f}), {:.0f} constant blocks, {:.1f} clears, "
      "{:.1f} copies, {:.1f} reinterpretations, {:.0f} draws with a perspective projection; the record {:.3f} ms (median, 95%: {:.3f}); a replay on the video "
      "thread {:.3f} ms (median; 95%: {:.3f}, most {:.3f}); on the GPU {:.3f} ms (median of {}); compares {}, "
      "identical {}, most pixels differing {} (first at frame {}); {} frames stopped at the budget",
      backend, s.frames, s.replayed, s.replays,
      s.skipped, s.draws / n, s.most_draws, s.bytes / n / 1048576.0, s.most_bytes / 1048576.0,
      s.blocks / n, s.clears / n, s.copies / n, s.reinterprets / n, s.perspective / n,
      Percentile(s.record_us, 0.5) / 1000.0, Percentile(s.record_us, 0.95) / 1000.0,
      Percentile(s.replay_us, 0.5) / 1000.0, Percentile(s.replay_us, 0.95) / 1000.0,
      Percentile(s.replay_us, 1.0) / 1000.0, Percentile(s.gpu_us, 0.5) / 1000.0, s.gpu_us.size(),
      s.compares, s.identical, s.most_diff, s.first_diff_frame, over_budget) +
         (s.requests == 0 ?
              std::string() :
              fmt::format("; high refresh rate: {} requests drawn again ({} stale), {} pictures after the "
                          "step's own, {} left out for the budget ({} requests planned thinner to fit it, evenly, {} pictures fewer), {} for the game's next list ({} given up while drawn), {} guessed for a later time than planned (they "
                          "would be ready after theirs), {} left out for that; {} requests drawn while the loop had "
                          "nothing to read, {} frames whose next copy came first; a frame's draws: {:.0f} the "
                          "world's, {:.0f} a machine's ({:.1f} of them by their place: its rank icons, what the game put in the view all "
                          "near it), {:.0f} put in the view by the game ({:.0f} of their vertices "
                          "moved with a machine a picture), {:.0f} flat ({:.1f} of them over a machine, moved with "
                          "it), {:.0f} offscreen; requests paired with their own frame {}, not {}",
                          s.requests, s.stale, s.pictures, s.budget_left, s.thinned, s.thinned_out, s.yielded, s.stopped, s.latched, s.late, s.idle, s.unreplayed,
                          s.tag_world / double(s.requests), s.tag_machine / double(s.requests),
                          s.tag_whole / double(s.requests),
                          s.tag_vertex / double(s.requests),
                          s.pictures ? s.vertices_moved / double(s.pictures) : 0.0,
                          s.tag_flat / double(s.requests), s.tag_over / double(s.requests),
                          s.tag_offscreen / double(s.requests), s.pair_right, s.unframed)) +
         (s.parted == 0 ?
              std::string() :
              fmt::format("; a replay's video thread by part (mean): constants {:.3f} ms ({:.0f} uploads, {:.2f} "
                          "MB), vertices {:.3f} ms ({:.2f} MB uploaded, {:.2f} MB drawn again from the first "
                          "picture's), draws {:.3f} ms, EFB copies {:.3f} ms, the picture {:.3f} ms",
                          s.uniforms_us / s.parted / 1000.0, double(s.uniform_uploads) / s.parted,
                          s.uniform_bytes / double(s.parted) / 1048576.0, s.vertices_us / s.parted / 1000.0,
                          s.vertex_bytes / double(s.parted) / 1048576.0,
                          s.vertex_reused / double(s.parted) / 1048576.0, s.draws_us / s.parted / 1000.0,
                          s.copies_us / s.parted / 1000.0, s.picture_us / s.parted / 1000.0));
}
}  // namespace VideoCommon::FrameRedraw
