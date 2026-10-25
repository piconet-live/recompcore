// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw's record, shared by its two files: FrameRedraw.cpp records and keeps the settings and the counts,
// FrameRedrawReplay.cpp draws a record again and compares.
#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "Common/BitSet.h"
#include "Common/Matrix.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/ConstantManager.h"
#include "VideoCommon/FrameRedraw.h"
#include "VideoCommon/FramebufferManager.h"

namespace VideoCommon::FrameRedraw
{
using Clock = std::chrono::steady_clock;

enum class Kind : u8
{
  Draw,
  Clear,
  Copy,
  Reinterpret
};

struct Op
{
  Kind kind;
  u32 index;
};

struct Draw
{
  const AbstractPipeline* pipeline;
  u32 vertex_at, num_vertices, stride, index_at, num_indices;
  u32 vs, gs, ps;
  AbstractGfx::ViewportAndScissor view;
  BitSet32 textures;
  bool expand;
  bool offscreen;              // inside an offscreen pass (Close)
  std::array<float, 12> mtx;   // its position matrix (model to view, rows of 4): its translation is where its model is
  s32 position = -1;           // the offset of a vertex's position (three floats) in its record; -1: not three floats
  bool vertex_matrices;        // its vertices name their own position matrices (mtx is then the default's)
  bool ztest;                  // compared with the depth buffer (bpmem.zmode)
  std::array<const AbstractTexture*, 8> texture;
  std::array<SamplerState, 8> sampler;
  std::array<u32, 8> address;  // each texture's place in the game's memory
  u8 copied;                   // a bit for each texture that is an EFB copy (this frame's or older)
};

struct ClearOp
{
  MathUtil::Rectangle<int> rc;
  bool color, alpha, z;
  u32 color_value, z_value;
  PixelFormat format;
};

struct CopyOp
{
  RcTcacheEntry entry;
  const AbstractTexture* texture;
  bool depth;
  MathUtil::Rectangle<int> src;
  bool half, linear;
  EFBCopyFormat format;
  bool intensity;
  float gamma;
  bool top, bottom;
  std::array<u32, 3> coef;
  MathUtil::Rectangle<int> scaled;  // src in the EFB texture's pixels
};

// The XFB copy that ends a frame, as TextureCacheBase made it: a replay's picture is made the same way.
struct XFBCopy
{
  bool valid = false;
  TextureConfig config{};
  u32 native_width = 0, native_height = 0, stride = 0;
  bool depth = false, linear = false;
  MathUtil::Rectangle<int> src;
  float gamma = 1;
  bool top = false, bottom = false;
  std::array<u32, 3> coef{};
  MathUtil::Rectangle<int> display;  // the picture's rectangle in the texture (the presenter's)
  RcTcacheEntry entry;  // the true copy itself, kept only while stills are saved (g_keep_xfb)
};

struct Frame
{
  bool whole = false;  // began at an XFB copy
  PixelFormat start_format{};
  std::vector<Op> ops;
  std::vector<Draw> draws;
  std::vector<ClearOp> clears;
  std::vector<CopyOp> copies;
  std::vector<EFBReinterpretType> reinterprets;
  std::vector<u8> vertices;
  std::vector<u16> indices;
  std::vector<VertexShaderConstants> vs;
  std::vector<bool> perspective;  // each vs block's projection is a perspective one (a 3D draw's)
  std::vector<GeometryShaderConstants> gs;
  std::vector<PixelShaderConstants> ps;
  std::vector<RcTcacheEntry> keep;  // the bound textures' entries, alive until the frame is let go
  std::array<const TCacheEntry*, 8> kept{};
  XFBCopy xfb;
  double record_us = 0;
  u64 first_read = 0, last_read = 0;  // the stream's bursts read at its first and last draw (the request's pairing)
  u64 copy = 0;  // the XFB copy that closed it, counted from 1 (the stills' names)

  void Reset()
  {
    whole = false;
    ops.clear();
    draws.clear();
    clears.clear();
    copies.clear();
    reinterprets.clear();
    vertices.clear();
    indices.clear();
    vs.clear();
    perspective.clear();
    gs.clear();
    ps.clear();
    keep.clear();
    kept = {};
    xfb = {};
    record_us = 0;
    first_read = last_read = 0;
    copy = 0;
  }
};

struct Replayed
{
  double cpu_us = 0, gpu_us = -1;
  s64 diff_rgb = -1, diff_rgba = -1, pixels = 0;
  // the video thread's time by part: the constants' uploads, the vertices' and indices', the draws (pipeline,
  // viewport, textures, the draw call), the EFB copies, the picture's XFB copy; and the uploads' bytes
  double uniforms_us = 0, vertices_us = 0, draws_us = 0, copies_us = 0, picture_us = 0;
  u64 uniform_uploads = 0, uniform_bytes = 0, vertex_bytes = 0, vertex_reused = 0;
  bool stopped = false;  // given up part way (the `stop` of Replay said so): no picture
};

// The counts of a start (FrameRedraw.cpp keeps them; FrameRedrawReport.cpp says them).
struct Stats
{
  u64 frames = 0, replayed = 0, replays = 0, skipped = 0, compares = 0, identical = 0;
  u64 draws = 0, most_draws = 0, bytes = 0, most_bytes = 0, clears = 0, copies = 0, reinterprets = 0;
  u64 blocks = 0, most_diff = 0, first_diff_frame = 0, perspective = 0;
  // the high refresh rate: requests taken, stale (a newer one was made before the frame), pictures made, replays
  // left out for the budget, requests replayed while the FIFO loop had nothing to read, pictures guessed for a
  // later time than planned (late latching) and left out for that (the next one was due by then), frames whose
  // copy came before a pass ended (not drawn again); the draws of the replayed frames by their correction
  u64 requests = 0, stale = 0, pictures = 0, budget_left = 0, idle = 0, latched = 0, late = 0, unreplayed = 0;
  // pictures left out because the game handed the GPU more of its commands (the next list) first, and of those
  // the ones given up while being drawn
  u64 yielded = 0, stopped = 0;
  // requests planned thinner to fit the budget at the replays' recent cost (evenly spread), and the pictures so
  // left out
  u64 thinned = 0, thinned_out = 0;
  u64 tag_world = 0, tag_machine = 0, tag_flat = 0, tag_offscreen = 0, tag_vertex = 0, tag_over = 0;
  u64 tag_whole = 0;  // of the machine's draws, those by place: its rank icon, its draws in the view all near it
  u64 vertices_moved = 0;
  std::vector<float> replay_us, record_us, gpu_us;
  // the pairing of each taken request with the frame the copy closed: right (the frame's draws all after the
  // request, the frame before's all before it), or not (no draws, or the frame before's last after it)
  u64 pair_right = 0, unframed = 0;
  // the replays' time by part (sums over every replay), and their uploads
  double uniforms_us = 0, vertices_us = 0, draws_us = 0, copies_us = 0, picture_us = 0;
  u64 uniform_uploads = 0, uniform_bytes = 0, vertex_bytes = 0, vertex_reused = 0, parted = 0;
};
// The summary line (Report): the counts, the frames stopped at the budget, the video backend's name.
std::string Summary(const Stats& s, u64 over_budget, const std::string& backend);

// True while a replay runs: nothing is recorded then.
extern bool g_replaying;
// True while stills are saved (Settings::stills): the record keeps its frame's true XFB copy.
extern bool g_keep_xfb;
// A test's stills: the texture's rectangle copied for a read back (RGBA8 or BGRA8; no wait), at the XFB copy `now`;
// TakeStills reads those asked four copies ago or more (all: every one, and waits for their files) and writes them
// as PNG files on threads of their own.
bool QueueStill(const AbstractTexture* texture, const MathUtil::Rectangle<int>& rect, const std::string& path,
                u64 now);
void TakeStills(u64 now, bool all);

double Us(Clock::duration d);
// Draws the frame again into the second EFB. With compare, waits for the GPU before and after (its time), reads
// both EFBs back over *compare and counts the pixels that differ; the first differing ones are dumped to
// dump_prefix (at most three, counted in *dumps).
// corrections and tags: the 4x4s a draw's projection is multiplied by, and for each draw the index of its own in
// corrections (-1: none). Null: the replay as recorded.
// picture: the frame's XFB copy made again from the replay into a texture of its own (null: none).
struct Corrections
{
  std::vector<Common::Matrix44> matrices;
  std::vector<s8> tags;
  // draws whose vertices are moved one by one: their offset in `vertices` (~0u: the record's own)
  std::vector<u32> moved_at;
  std::vector<u8> vertices;
  // corrections from this index on are a shift on the screen: the projection is multiplied on their left (the
  // flat draws over a machine: its name tag, its rank icon, moved by the machine's projected move)
  size_t screen_from = ~size_t(0);
  // ... up to this one (from here on: a billboard over a machine, its rank icon: the camera's correction and the
  // machine's move, not its turn, so that it keeps facing the camera)
  size_t icon_from = ~size_t(0);
  // the world's projection (a perspective draw's), for the machines' places on the screen
  bool has_world = false;
  Common::Matrix44 world;
  // the vertices and indices uploaded by the request's first picture, drawn again from where they are by the
  // others (a backend whose stream buffers keep their place and whose vertex stride is the pipeline's): each
  // draw's base vertex and base index (~0u: not uploaded), and the stream's last places, a wrap ending the reuse
  bool reuse = false;
  std::vector<std::array<u32, 2>> uploaded;
  u64 last_vertex_byte = 0, last_index = 0;
};
// `stop` (may be null) is asked every few draws; true gives the replay up (Replayed::stopped, no picture), with
// every piece of state it touched put back as at its end.
Replayed Replay(const Frame& f, const MathUtil::Rectangle<int>* compare, u64 frame_number,
                const std::string& dump_prefix, u32* dumps, const Corrections* corrections,
                RcTcacheEntry* picture = nullptr, bool (*stop)() = nullptr);
// FrameRedraw.cpp: the command stream's bursts read so far (the GPU thread's side of WrittenPosition).
u64 ReadPosition();
// FrameRedrawRecord.cpp: the frame being recorded, and a new one started (after an XFB copy).
Frame& RecordedFrame();
void StartRecord(PixelFormat start_format);

// FrameRedrawTags.cpp, the high refresh rate's corrections of a closed frame's draws (FrameRedraw.h):
// the draws inside offscreen passes marked (picture: the XFB copy's rectangle in the EFB's pixels);
void MarkOffscreen(Frame& f, const MathUtil::Rectangle<int>& picture);
// each draw's correction for a request (-1 none, 0 the world's, 1 + i machine i's), and the draws whose vertices
// are moved one by one;
struct TagCounts
{
  u64 world = 0, machine = 0, flat = 0, offscreen = 0, vertex = 0, over_machine = 0, whole = 0;
};
TagCounts Tag(const Frame& f, const Request& q, float radius, Corrections* c, std::vector<u32>* by_vertex);
// the in-between's matrices into c, and those draws' vertices moved (the count moved);
void Fill(Corrections* c, const Request& q, const InBetween& b);
u64 Move(const Frame& f, const Request& q, const InBetween& b, const std::vector<u32>& by_vertex, Corrections* c);
// a test's table of the draws (Settings::draws_at).
void WriteDraws(const Frame& f, const Request& q, const Corrections& c, const std::string& path);

// Lets the replay's own GPU objects go (the video backend's shutdown).
void ReleaseReplay();
}  // namespace VideoCommon::FrameRedraw
