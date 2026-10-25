// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw's corrections for the high refresh rate (FrameRedraw.h): which draws of a recorded frame are
// offscreen, which correction each draw takes, the vertices moved one by one, and a test's table of the draws.

#include "VideoCommon/FrameRedrawFrame.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>

namespace VideoCommon::FrameRedraw
{
namespace
{
bool Inside(const AbstractGfx::ViewportAndScissor& v, const MathUtil::Rectangle<int>& rc)
{
  constexpr float e = 2;
  const float l = static_cast<float>(std::min(rc.left, rc.right)), r = static_cast<float>(std::max(rc.left, rc.right));
  const float t = static_cast<float>(std::min(rc.top, rc.bottom)), b = static_cast<float>(std::max(rc.top, rc.bottom));
  const float x0 = std::min(v.viewport_x, v.viewport_x + v.viewport_width);
  const float x1 = std::max(v.viewport_x, v.viewport_x + v.viewport_width);
  const float y0 = std::min(v.viewport_y, v.viewport_y + v.viewport_height);
  const float y1 = std::max(v.viewport_y, v.viewport_y + v.viewport_height);
  if (x0 >= l - e && x1 <= r + e && y0 >= t - e && y1 <= b + e)
    return true;
  const MathUtil::Rectangle<int>& s = v.scissor_rect;
  return std::min(s.left, s.right) >= l - e && std::max(s.left, s.right) <= r + e &&
         std::min(s.top, s.bottom) >= t - e && std::max(s.top, s.bottom) <= b + e;
}


bool Identity(const std::array<float, 12>& m)
{
  constexpr std::array<float, 12> one{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 12; ++i)
  {
    if (std::fabs(m[i] - one[i]) > 1e-5f)
      return false;
  }
  return true;
}

// A 4x4 of the shader constants' rows.
Common::Matrix44 Rows(const std::array<float4, 4>& rows)
{
  std::array<float, 16> a;
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      a[4 * r + k] = rows[r][k];
  return Common::Matrix44::FromArray(a);
}

// A point of the view on the screen (normalized device coordinates, y up); false behind the eye.
bool OnScreen(const Common::Matrix44& projection, float x, float y, float z, float* sx, float* sy)
{
  const Common::Vec4 clip = projection * Common::Vec4(x, y, z, 1.0f);
  if (!(clip.w > 1e-4f))
    return false;
  *sx = clip.x / clip.w;
  *sy = clip.y / clip.w;
  return true;
}

// A flat draw's box on the screen (normalized device coordinates): x0, x1, y0, y1. False without float positions.
bool FlatBox(const Frame& f, const Draw& d, float box[4])
{
  if (d.position < 0 || d.num_vertices == 0)
    return false;
  const Common::Matrix44 projection = Rows(f.vs[d.vs].projection);
  box[0] = box[2] = 1e30f;
  box[1] = box[3] = -1e30f;
  const auto& m = d.mtx;
  for (u32 v = 0; v < d.num_vertices; ++v)
  {
    float p[3];
    std::memcpy(p, &f.vertices[d.vertex_at + static_cast<size_t>(v) * d.stride + d.position], sizeof(p));
    float w[3], sx, sy;
    for (int r = 0; r < 3; ++r)
      w[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
    if (!OnScreen(projection, w[0], w[1], w[2], &sx, &sy))
      return false;
    box[0] = std::min(box[0], sx);
    box[1] = std::max(box[1], sx);
    box[2] = std::min(box[2], sy);
    box[3] = std::max(box[3], sy);
  }
  return true;
}

// A draw the game put in the view whose every vertex lies within this many units of one machine moves with it whole.
constexpr float WHOLE_NEAR = 10.0f;

// The machine a draw the game put in the view lies all near (its sparks, its burner's glow, its shadow on the ground):
// the one whose farthest vertex is nearest, if that is within WHOLE_NEAR; or -1. Vertices at the eye itself (the
// view's origin: unused corners of the burner's glow) are left out. A long trail is not all near any machine.
int WholeNear(const Frame& f, const Draw& d, const Request& q)
{
  if (d.position < 0 || d.num_vertices == 0)
    return -1;
  std::array<float, MOST_MACHINES> far2{};
  bool any = false;
  for (u32 v = 0; v < d.num_vertices; ++v)
  {
    float p[3];
    std::memcpy(p, &f.vertices[d.vertex_at + static_cast<size_t>(v) * d.stride + d.position], sizeof(p));
    if (std::fabs(p[0]) + std::fabs(p[1]) + std::fabs(p[2]) < 1e-4f)
      continue;
    any = true;
    for (u32 i = 0; i < MOST_MACHINES; ++i)
    {
      const float dx = p[0] - q.at[i][0], dy = p[1] - q.at[i][1], dz = p[2] - q.at[i][2];
      far2[i] = std::max(far2[i], dx * dx + dy * dy + dz * dz);
    }
  }
  int best = -1;
  float nearest = WHOLE_NEAR * WHOLE_NEAR;
  for (u32 i = 0; i < MOST_MACHINES && any; ++i)
  {
    if ((q.machines & (1u << i)) && far2[i] <= nearest)
    {
      nearest = far2[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

// The machine a rank icon stands over, or -1. The game queues its icons (fn_1_129D9C) in its depth-sorted list and
// draws them later, where no machine is known, so an icon is told by its place: a model scaled up 2 times or more
// (the icons' billboards are 5 to 8 times), standing straight up from machine i in the view: 4 to 40 units above it
// on the screen's vertical, within 1 unit and a 20th of the height of that line (m23 at 1700: 8 to 17 units up, 0.01
// off it).
int IconOver(const Draw& d, const Request& q)
{
  const auto& m = d.mtx;
  const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                    m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (!(std::fabs(det) >= 8.0f))
    return -1;
  for (u32 i = 0; i < MOST_MACHINES; ++i)
  {
    if (!(q.machines & (1u << i)))
      continue;
    const float x = m[3] - q.at[i][0], h = m[7] - q.at[i][1], z = m[11] - q.at[i][2];
    const float most = 1.0f + 0.05f * h;
    if (h >= 4.0f && h <= 40.0f && x * x + z * z <= most * most)
      return static_cast<int>(i);
  }
  return -1;
}

// The frame before's flat draws (their first texture, vertex count and box) and its machines' places on the screen:
// what a flat draw over a machine is told apart by, since the HUD also lies over machines at times.
struct FlatSeen
{
  const AbstractTexture* texture;
  u32 vertices, nth;  // nth: its place among the frame's flat draws of that texture
  float box[4];
};
std::vector<FlatSeen> s_flat_before, s_flat_now;
std::array<std::array<float, 2>, MOST_MACHINES> s_screen_before{}, s_screen_now{};
u32 s_screen_before_on = 0, s_screen_now_on = 0;

// Whether the frame before had this draw (its texture's nth flat draw then) where machine i's move on the screen
// since then puts it (the HUD does not move with a machine; a name tag or rank icon does). A machine that hardly
// moved on the screen: no (what would follow it would hardly move).
bool MovedWith(const Draw& d, u32 nth, const float box[4], u32 i)
{
  if (!(s_screen_before_on & s_screen_now_on & (1u << i)))
    return false;
  const float mx = s_screen_now[i][0] - s_screen_before[i][0], my = s_screen_now[i][1] - s_screen_before[i][1];
  if (std::fabs(mx) + std::fabs(my) < 0.01f)
    return false;
  const AbstractTexture* texture = d.textures[0] ? d.texture[0] : nullptr;
  for (const FlatSeen& s : s_flat_before)
  {
    if (s.texture != texture || s.nth != nth)
      continue;
    const float bx = (box[0] + box[1] - s.box[0] - s.box[1]) / 2, by = (box[2] + box[3] - s.box[2] - s.box[3]) / 2;
    return s.vertices == d.num_vertices && std::fabs(bx - mx) < 0.003f + 0.15f * std::fabs(mx) &&
           std::fabs(by - my) < 0.003f + 0.15f * std::fabs(my);
  }
  return false;
}

// Its place among the frame's flat draws of its texture (counted as they come).
u32 Nth(const Draw& d)
{
  const AbstractTexture* texture = d.textures[0] ? d.texture[0] : nullptr;
  u32 n = 0;
  for (const FlatSeen& s : s_flat_now)
    n += s.texture == texture;
  return n;
}

// The machine a flat draw is over (its name tag, its rank icon: drawn by the game at the machine's projected
// place, above it), or -1: a small box, its middle within 0.4 of the machine's place across, its foot from 0.05
// below it to 0.9 above (normalized device coordinates, the screen 2 high), that moved with the machine on the
// screen since the frame before. A name tag is a quad a letter, centred 2.5 units above its machine: its outer
// letters stand up to 0.35 to the side, and a machine closer than the tags' NEAR (15 units) has its tag more than
// 0.35 above it (the first rule, 0.15 and 0.35, took only the middle letters of a tag at 15 to 40 units).
int Over(const Draw& d, u32 nth, const float box[4], const Request& q, const Corrections& c, float* dx, float* dy)
{
  *dx = *dy = 1e30f;
  if (!c.has_world)
    return -1;
  const float middle = (box[0] + box[1]) / 2, foot = box[2];
  int best = -1;
  float nearest = 1e30f;
  for (u32 i = 0; i < MOST_MACHINES; ++i)
  {
    float mx, my;
    if (!(q.machines & (1u << i)) || !OnScreen(c.world, q.at[i][0], q.at[i][1], q.at[i][2], &mx, &my))
      continue;
    const float ax = middle - mx, ay = foot - my;
    if (std::fabs(ax) + std::fabs(ay) < std::fabs(*dx) + std::fabs(*dy))
      *dx = ax, *dy = ay;
    if (box[1] - box[0] > 0.5f || box[3] - box[2] > 0.3f || std::fabs(ax) > 0.4f || ay < -0.05f || ay > 0.9f ||
        !MovedWith(d, nth, box, i))
      continue;
    if (std::fabs(ax) + std::fabs(ay) < nearest)
    {
      nearest = std::fabs(ax) + std::fabs(ay);
      best = static_cast<int>(i);
    }
  }
  return best;
}
// printf into a string (a test's table, written to its file later)
void Put(std::string* out, const char* format, ...)
{
  char line[1024];
  va_list args;
  va_start(args, format);
  const int n = std::vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  if (n > 0)
    out->append(line, static_cast<size_t>(std::min(n, static_cast<int>(sizeof(line)) - 1)));
}
}  // namespace

// The offscreen passes of a closed frame: draws whose viewport or scissor lies inside the source of a later EFB copy
// smaller than half the picture, after the copy before it (shadow pictures, portraits, effect pictures). The
// screen's own copies (the screen effects') take the whole picture: their draws are the world's.
void MarkOffscreen(Frame& f, const MathUtil::Rectangle<int>& picture)
{
  const s64 whole = static_cast<s64>(picture.GetWidth()) * picture.GetHeight();
  u32 since = 0, drawn = 0;
  for (const Op& op : f.ops)
  {
    if (op.kind == Kind::Draw)
    {
      drawn = op.index + 1;
      continue;
    }
    if (op.kind != Kind::Copy)
      continue;
    const MathUtil::Rectangle<int>& rc = f.copies[op.index].scaled;
    if (2 * static_cast<s64>(rc.GetWidth()) * rc.GetHeight() < whole)
    {
      for (u32 i = since; i < drawn; ++i)
        f.draws[i].offscreen = f.draws[i].offscreen || Inside(f.draws[i].view, rc);
    }
    since = drawn;
  }
}


// Each draw's correction for a request: -1 none (orthographic, offscreen), 0 the world's, 1 + i machine i's (the
// nearest machine within the radius of where the draw's model is).
TagCounts Tag(const Frame& f, const Request& q, float radius, Corrections* c, std::vector<u32>* by_vertex)
{
  const float r2 = radius * radius;
  std::vector<s8>* tags = &c->tags;
  tags->assign(f.draws.size(), -1);
  by_vertex->clear();
  c->screen_from = 1 + MOST_MACHINES;
  c->icon_from = 1 + 2 * MOST_MACHINES;
  c->has_world = false;
  for (const Draw& d : f.draws)
  {
    if (f.perspective[d.vs] && !d.offscreen)
    {
      c->world = Rows(f.vs[d.vs].projection);
      c->has_world = true;
      break;
    }
  }
  // this frame's machines on the screen, and its flat draws, for the next frame's
  std::swap(s_flat_before, s_flat_now);
  s_flat_now.clear();
  s_screen_before = s_screen_now;
  s_screen_before_on = s_screen_now_on;
  s_screen_now_on = 0;
  for (u32 i = 0; i < MOST_MACHINES && c->has_world; ++i)
  {
    if ((q.machines & (1u << i)) &&
        OnScreen(c->world, q.at[i][0], q.at[i][1], q.at[i][2], &s_screen_now[i][0], &s_screen_now[i][1]))
      s_screen_now_on |= 1u << i;
  }
  u64 world = 0, machine = 0, flat = 0, offscreen = 0, over = 0, whole = 0;
  for (size_t k = 0; k < f.draws.size(); ++k)
  {
    const Draw& d = f.draws[k];
    if (!f.perspective[d.vs])
    {
      ++flat;
      float box[4], dx, dy;
      if (d.offscreen || !FlatBox(f, d, box))
        continue;
      const u32 nth = Nth(d);
      s_flat_now.push_back(
          {d.textures[0] ? d.texture[0] : nullptr, d.num_vertices, nth, {box[0], box[1], box[2], box[3]}});
      const int i = Over(d, nth, box, q, *c, &dx, &dy);
      if (i >= 0)
      {
        (*tags)[k] = static_cast<s8>(1 + MOST_MACHINES + i);
        ++over;
      }
      continue;
    }
    if (d.offscreen)
    {
      ++offscreen;
      continue;
    }
    // a machine's rank icon, and what the game put in the view all near one machine: with that machine whole
    const bool in_view = Identity(d.mtx) && d.position >= 0 && !d.vertex_matrices;
    const int icon = in_view || d.vertex_matrices ? -1 : IconOver(d, q);
    if (const int i = in_view ? WholeNear(f, d, q) : icon; i >= 0)
    {
      (*tags)[k] = static_cast<s8>((icon >= 0 ? 1 + 2 * MOST_MACHINES : 1) + i);
      ++machine;
      ++whole;
      continue;
    }
    if (in_view)
    {
      // the game put its vertices in the view itself: the world's correction, its vertices near machines moved
      (*tags)[k] = 0;
      by_vertex->push_back(static_cast<u32>(k));
      continue;
    }
    int best = -1;
    float nearest = r2;
    for (u32 i = 0; i < MOST_MACHINES; ++i)
    {
      if (!(q.machines & (1u << i)))
        continue;
      const float dx = d.mtx[3] - q.at[i][0], dy = d.mtx[7] - q.at[i][1], dz = d.mtx[11] - q.at[i][2];
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < nearest)
      {
        nearest = d2;
        best = static_cast<int>(i);
      }
    }
    (*tags)[k] = static_cast<s8>(best + 1);
    ++(best >= 0 ? machine : world);
  }
  return {world, machine, flat, offscreen, by_vertex->size(), over, whole};
}

// The vertices of the draws the game put in the view itself, each near a machine moved by that machine's own move
// (InBetween::motion): fully within 3 units of its place in the old view, less and less to none at 8.
u64 Move(const Frame& f, const Request& q, const InBetween& b, const std::vector<u32>& by_vertex, Corrections* c)
{
  constexpr float FULL = 3.0f, NONE = 8.0f;
  c->moved_at.assign(f.draws.size(), ~0u);
  c->vertices.clear();
  u64 moved = 0;
  for (const u32 k : by_vertex)
  {
    const Draw& d = f.draws[k];
    const size_t bytes = static_cast<size_t>(d.num_vertices) * d.stride;
    const u32 at = static_cast<u32>(c->vertices.size());
    c->vertices.insert(c->vertices.end(), f.vertices.begin() + d.vertex_at,
                       f.vertices.begin() + d.vertex_at + bytes);
    bool any = false;
    for (u32 v = 0; v < d.num_vertices; ++v)
    {
      u8* where = &c->vertices[at + static_cast<size_t>(v) * d.stride + d.position];
      float p[3];
      std::memcpy(p, where, sizeof(p));
      int best = -1;
      float nearest = NONE * NONE;
      for (u32 i = 0; i < MOST_MACHINES; ++i)
      {
        if (!(q.machines & (1u << i)))
          continue;
        const float dx = p[0] - q.at[i][0], dy = p[1] - q.at[i][1], dz = p[2] - q.at[i][2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < nearest)
        {
          nearest = d2;
          best = static_cast<int>(i);
        }
      }
      if (best < 0)
        continue;
      const float w = std::clamp((NONE - std::sqrt(nearest)) / (NONE - FULL), 0.0f, 1.0f);
      const Matrix& m = b.motion[best];
      float o[3];
      for (int r = 0; r < 3; ++r)
      {
        const float to = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
        o[r] = p[r] + w * (to - p[r]);
      }
      std::memcpy(where, o, sizeof(o));
      ++moved;
      any = true;
    }
    if (any)
      c->moved_at[k] = at;
  }
  return moved;
}

// A test's table of a frame's draws (Settings::draws_at): what each is, its correction, and where its vertices are in
// the view against the nearest machine: the nearest and farthest vertex from it, and the vertices within 6 units.
void WriteDraws(const Frame& f, const Request& q, const Corrections& c, const std::string& path)
{
  const std::vector<s8>& tags = c.tags;
  // (written to memory here and to the file on a thread of its own: the GPU thread must not wait, or the game's
  // next list would be read late)
  std::string out, copies;
  // each draw's and each copy's place among the frame's operations, and the first copy to each texture and place
  std::vector<u32> draw_at(f.draws.size(), 0);
  std::unordered_map<const AbstractTexture*, u32> copy_to;
  std::unordered_map<u32, u32> copy_place;
  for (u32 n = 0; n < f.ops.size(); ++n)
  {
    const Op& op = f.ops[n];
    if (op.kind == Kind::Draw)
      draw_at[op.index] = n;
    if (op.kind != Kind::Copy)
      continue;
    const CopyOp& copy = f.copies[op.index];
    copy_to.try_emplace(copy.texture, n);
    if (copy.entry)
      copy_place.try_emplace(copy.entry->addr, n);
  }
  Put(&out, "draw,tag,perspective,offscreen,vertices,indices,stride,vertex_matrices,tx,ty,tz,rotation_det,"
                    "texture0,machine,near,far,within6,minx,maxx,miny,maxy,minz,maxz,ztest,screen_x0,screen_x1,"
                    "screen_y0,screen_y1,over_dx,over_dy,address0,copied0,copy_kind0\n");
  for (size_t k = 0; k < f.draws.size(); ++k)
  {
    const Draw& d = f.draws[k];
    const auto& m = d.mtx;
    const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                      m[2] * (m[4] * m[9] - m[5] * m[8]);
    int machine = -1;
    float closest = 1e30f, farthest = 0, lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    u32 within = 0;
    if (d.position >= 0)
    {
      std::vector<std::array<float, 3>> seen;
      // (a big draw's every few vertices: the GPU thread must not wait long)
      const u32 every = std::max<u32>(1, d.num_vertices / 64);
      for (u32 v = 0; v < d.num_vertices; v += every)
      {
        float p[3];
        std::memcpy(p, &f.vertices[d.vertex_at + static_cast<size_t>(v) * d.stride + d.position], sizeof(p));
        std::array<float, 3> w;
        for (int r = 0; r < 3; ++r)
        {
          w[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
          lo[r] = std::min(lo[r], w[r]);
          hi[r] = std::max(hi[r], w[r]);
        }
        seen.push_back(w);
      }
      for (u32 i = 0; i < MOST_MACHINES; ++i)
      {
        if (!(q.machines & (1u << i)))
          continue;
        for (const auto& w : seen)
        {
          const float dx = w[0] - q.at[i][0], dy = w[1] - q.at[i][1], dz = w[2] - q.at[i][2];
          const float dd = std::sqrt(dx * dx + dy * dy + dz * dz);
          if (dd < closest)
          {
            closest = dd;
            machine = static_cast<int>(i);
          }
        }
      }
      if (machine >= 0)
      {
        for (const auto& w : seen)
        {
          const float dx = w[0] - q.at[machine][0], dy = w[1] - q.at[machine][1], dz = w[2] - q.at[machine][2];
          const float dd = std::sqrt(dx * dx + dy * dy + dz * dz);
          farthest = std::max(farthest, dd);
          within += dd < 6.0f;
        }
      }
    }
    float box[4] = {0, 0, 0, 0}, dx = 0, dy = 0;
    if (!f.perspective[d.vs] && FlatBox(f, d, box))
      Over(d, 0, box, q, c, &dx, &dy);
    // texture 0 if an EFB copy: "now" one of this frame's copies before the draw (the replay draws its own), "place"
    // a texture at the place of such a copy but not it (the replay draws the true one), "older" one from before
    const char* kind = "-";
    if (d.textures[0] && (d.copied & 1))
    {
      const auto to = copy_to.find(d.texture[0]);
      const auto place = copy_place.find(d.address[0]);
      kind = to != copy_to.end() && to->second < draw_at[k]        ? "now" :
             place != copy_place.end() && place->second < draw_at[k] ? "place" :
                                                                       "older";
    }
    Put(&out, "%zu,%d,%d,%d,%u,%u,%u,%d,%.2f,%.2f,%.2f,%.3f,%p,%d,%.2f,%.2f,%u,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                      "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%08X,%d,%s\n",
                 k, tags[k], f.perspective[d.vs] ? 1 : 0, d.offscreen ? 1 : 0, d.num_vertices, d.num_indices,
                 d.stride, d.vertex_matrices ? 1 : 0, m[3], m[7], m[11], det,
                 static_cast<const void*>(d.textures[0] ? d.texture[0] : nullptr), machine, closest, farthest, within,
                 lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], d.ztest ? 1 : 0, box[0], box[1], box[2], box[3], dx, dy,
                 d.textures[0] ? d.address[0] : 0u, (d.copied & 1) ? 1 : 0, kind);
  }
  // and its EFB copies, in order: after which draw, where to, from where
  Put(&copies, "copy,after_draw,address,src_left,src_top,src_right,src_bottom,scaled_width,scaled_height,half,"
                       "format,depth,intensity,texture\n");
  s32 last = -1;
  for (const Op& op : f.ops)
  {
    if (op.kind == Kind::Draw)
      last = static_cast<s32>(op.index);
    if (op.kind != Kind::Copy)
      continue;
    const CopyOp& copy = f.copies[op.index];
    Put(&copies, "%u,%d,%08X,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%p\n", op.index, last,
                 copy.entry ? copy.entry->addr : 0u, copy.src.left, copy.src.top, copy.src.right, copy.src.bottom,
                 copy.scaled.GetWidth(), copy.scaled.GetHeight(), copy.half ? 1 : 0,
                 static_cast<u32>(copy.format), copy.depth ? 1 : 0, copy.intensity ? 1 : 0,
                 static_cast<const void*>(copy.texture));
  }
  std::thread([out = std::move(out), copies = std::move(copies), path] {
    if (FILE* file = std::fopen(path.c_str(), "w"))
    {
      std::fwrite(out.data(), 1, out.size(), file);
      std::fclose(file);
    }
    if (FILE* file = std::fopen((path + ".copies.csv").c_str(), "w"))
    {
      std::fwrite(copies.data(), 1, copies.size(), file);
      std::fclose(file);
    }
  }).detach();
}

void Fill(Corrections* c, const Request& q, const InBetween& b)
{
  c->matrices.resize(1 + 3 * MOST_MACHINES);
  c->matrices[0] = Common::Matrix44::FromArray(b.camera);
  for (u32 i = 0; i < MOST_MACHINES; ++i)
  {
    c->matrices[1 + i] = Common::Matrix44::FromArray(b.machine[i]);
    // what is drawn flat over machine i: moved on the screen as its place is (seen through its correction)
    Common::Matrix44 shift = Common::Matrix44::Identity();
    float x0, y0, x1, y1;
    const Common::Vec4 at = c->matrices[1 + i] * Common::Vec4(q.at[i][0], q.at[i][1], q.at[i][2], 1.0f);
    if (c->has_world && (q.machines & (1u << i)) &&
        OnScreen(c->world, q.at[i][0], q.at[i][1], q.at[i][2], &x0, &y0) &&
        OnScreen(c->world, at.x / at.w, at.y / at.w, at.z / at.w, &x1, &y1))
    {
      shift.data[3] = x1 - x0;
      shift.data[7] = y1 - y0;
    }
    c->matrices[1 + MOST_MACHINES + i] = shift;
    // machine i's rank icon: where the camera's correction puts it, moved on as the machine's place is
    const Common::Vec4 seen = c->matrices[0] * Common::Vec4(q.at[i][0], q.at[i][1], q.at[i][2], 1.0f);
    Common::Matrix44 icon = c->matrices[0];
    icon.data[3] += at.x - seen.x;
    icon.data[7] += at.y - seen.y;
    icon.data[11] += at.z - seen.z;
    c->matrices[1 + 2 * MOST_MACHINES + i] = icon;
  }
}

}  // namespace VideoCommon::FrameRedraw
