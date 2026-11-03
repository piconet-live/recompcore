// Pictures between two steps by drawing the true frame again. The game bakes its view into every position matrix on the CPU, so each
// world draw reaches the GPU as P * (V * M): a correction C = V_new * V_old^-1 after the modelview, P' = P * C, moves it by the camera's
// change; a machine i's draws take C_i = V_new * W_i * V_old^-1 (research/study-redraw.md). The projection sits in each draw's own uniform
// block, so a picture is the frame's uniforms with those patched, then the frame's passes encoded again.
//
// Which correction a draw takes comes from the draw's own data (the Vulkan re-draw's rules, VideoCommon/FrameRedrawTags.cpp, on Aurora's
// data):
//  - in a pass that ends in a copy smaller than half the picture (shadow pictures, portraits): none;
//  - orthographic (the HUD): none, but one over a machine that moved with it on the screen since the frame before (its name tag) is
//    shifted on the screen as the machine's place is (P' = S * P);
//  - an identity position matrix (the game put its vertices in the view: trails, arcs, flames, sparks) with every vertex within WholeNear
//    of one machine: that machine's; otherwise the camera's, its vertices near a machine first moved by that machine's own move, fully
//    within MoveFull units, less and less to none at MoveNone (a trail's head stays on its machine, its older points where they were);
//  - a model scaled 2 times or more standing 4 to 40 units straight up from a machine (its rank icon): the camera's, moved on as the
//    machine's place is (not turned with it);
//  - a position matrix whose translation is within Radius of a machine: that machine's; else the camera's.
// A machine's own move in the true view (what moves the vertices) is the camera's correction undone from the machine's:
// C^-1 * C_i = V_old * W_i * V_old^-1. Tuned numbers, from the Vulkan re-draw work and not re-tuned here: Radius 6, WholeNear 10,
// MoveFull 3 and MoveNone 8, the icon's 4 to 40 and 8 (2 cubed), the name tag's box and moves (Over, MovedWith).
#include "redraw.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/stagelog.hpp"
#include "../gfx/upload_read.hpp"
#include "rt_capture.hpp"
#include "rt_live.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace aurora::gx::redraw {
bool g_on = false;

namespace {
constexpr float Radius = 6.f, WholeNearUnits = 10.f, MoveFull = 3.f, MoveNone = 8.f;
constexpr u32 MostWorldVertices = 1024;  // a world-space draw read for vertex-by-vertex moves: at most this many vertices
// a note's correction: None, World, ByVertex (the world's, its vertices moved), machine i (0..), its icon (Icon + i), over it (Over + i)
constexpr int None = -1, World = -2, ByVertex = -3, Icon = 100, Over = 200;

Notes s_building;
Notes s_held;
std::vector<int> s_class;  // the held frame's notes' corrections
std::vector<u32> s_blocks;  // the projections a picture patches (each 64 bytes at its note's projAt), sorted: AURORA_REDRAW_BUNDLES
bool s_classified = false;
const Request* s_classifiedFor = nullptr;
bool s_advanced = false;  // the flat draws' and the machines' places of the held frame taken for the next frame's
std::vector<u8> s_scratch;
std::vector<bool> s_dirty;  // a segment written moved in a picture of the held frame: written again (moved or not) in each after
std::string s_tablePath;
u32 s_testTexture = 0;
std::atomic<u64> s_pictures{0}, s_world{0}, s_machine{0}, s_whole{0}, s_icon{0}, s_flat{0}, s_offscreen{0}, s_splits{0}, s_frames{0},
    s_bytes{0}, s_byVertex{0}, s_unreadable{0}, s_moved{0}, s_over{0}, s_written{0};

float be_float(const u8* p) noexcept {
  u32 v;
  std::memcpy(&v, p, 4);
  v = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}
void put_be_float(u8* p, float f) noexcept {
  u32 v;
  std::memcpy(&v, &f, 4);
  v = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
  std::memcpy(p, &v, 4);
}

bool identity(const std::array<float, 12>& m) noexcept {
  static constexpr std::array<float, 12> one{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 12; ++i)
    if (std::fabs(m[i] - one[i]) > 1e-5f)
      return false;
  return true;
}

// a * b, 4x4 row-major
Matrix mul(const Matrix& a, const Matrix& b) noexcept {
  Matrix out{};
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      out[4 * r + k] = a[4 * r] * b[k] + a[4 * r + 1] * b[4 + k] + a[4 * r + 2] * b[8 + k] + a[4 * r + 3] * b[12 + k];
  return out;
}

// The inverse of an affine 4x4 (last row 0 0 0 1); the identity when its 3x3 has none
Matrix inverse(const Matrix& m) noexcept {
  const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
  const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const double det = a * A + b * B + c * C;
  if (!(std::fabs(det) > 1e-12))
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const double r[9] = {A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
                       B / det, (a * i - c * g) / det,  -(a * f - c * d) / det,
                       C / det, -(a * h - b * g) / det, (a * e - b * d) / det};
  Matrix out{};
  for (int row = 0; row < 3; ++row) {
    for (int k = 0; k < 3; ++k)
      out[4 * row + k] = static_cast<float>(r[3 * row + k]);
    out[4 * row + 3] = static_cast<float>(-(r[3 * row] * m[3] + r[3 * row + 1] * m[7] + r[3 * row + 2] * m[11]));
  }
  out[15] = 1;
  return out;
}

void apply(const Matrix& m, const float p[3], float out[3]) noexcept {
  for (int r = 0; r < 3; ++r)
    out[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
}

// A point of the view on the screen (normalized device coordinates, y up); false behind the eye
bool on_screen(const std::array<float, 16>& p, float x, float y, float z, float* sx, float* sy) noexcept {
  const float w = p[12] * x + p[13] * y + p[14] * z + p[15];
  if (!(w > 1e-4f))
    return false;
  *sx = (p[0] * x + p[1] * y + p[2] * z + p[3]) / w;
  *sy = (p[4] * x + p[5] * y + p[6] * z + p[7]) / w;
  return true;
}

// The machine all of a view-space draw's vertices lie near, or -1. Vertices at the eye itself (unused corners) are left out.
int whole_near(const Notes& n, const Note& d, const Request& q) noexcept {
  std::array<float, MostMachines> far2{};
  bool any = false;
  for (u32 v = 0; v < d.positions; ++v) {
    const float* p = &n.positions[(d.firstPosition + v) * 3];
    if (std::fabs(p[0]) + std::fabs(p[1]) + std::fabs(p[2]) < 1e-4f)
      continue;
    any = true;
    for (u32 i = 0; i < MostMachines; ++i) {
      const float dx = p[0] - q.at[i][0], dy = p[1] - q.at[i][1], dz = p[2] - q.at[i][2];
      far2[i] = std::max(far2[i], dx * dx + dy * dy + dz * dz);
    }
  }
  int best = -1;
  float nearest = WholeNearUnits * WholeNearUnits;
  for (u32 i = 0; i < MostMachines && any; ++i)
    if ((q.machines & (1u << i)) && far2[i] <= nearest) {
      nearest = far2[i];
      best = static_cast<int>(i);
    }
  return best;
}

int icon_over(const Note& d, const Request& q) noexcept {
  const auto& m = d.mtx;
  const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (!(std::fabs(det) >= 8.f))
    return -1;
  for (u32 i = 0; i < MostMachines; ++i) {
    if (!(q.machines & (1u << i)))
      continue;
    const float x = m[3] - q.at[i][0], h = m[7] - q.at[i][1], z = m[11] - q.at[i][2];
    const float most = 1.f + 0.05f * h;
    if (h >= 4.f && h <= 40.f && x * x + z * z <= most * most)
      return static_cast<int>(i);
  }
  return -1;
}

// The name tags (FrameRedrawTags.cpp's Over and MovedWith): the frame before's flat draws (their first texture, vertex count and box)
// and its machines' places on the screen, since the HUD also lies over machines at times
struct FlatSeen {
  u32 texture, vertices, nth;  // nth: its place among the frame's flat draws of that texture
  float box[4];
};
std::vector<FlatSeen> s_flatBefore, s_flatNow;
std::array<std::array<float, 2>, MostMachines> s_screenBefore{}, s_screenNow{};
u32 s_screenBeforeOn = 0, s_screenNowOn = 0;
bool s_hasWorld = false;
std::array<float, 16> s_worldProj{};  // the world's projection (the first perspective draw's not offscreen)

// A flat draw's box on the screen (normalized device coordinates): x0, x1, y0, y1. False without float positions.
bool flat_box(const Notes& n, const Note& d, float box[4]) noexcept {
  if (d.positions == 0)
    return false;
  box[0] = box[2] = 1e30f;
  box[1] = box[3] = -1e30f;
  for (u32 v = 0; v < d.positions; ++v) {
    const float* p = &n.positions[(d.firstPosition + v) * 3];
    float w[3], sx, sy;
    for (int r = 0; r < 3; ++r)
      w[r] = d.mtx[4 * r] * p[0] + d.mtx[4 * r + 1] * p[1] + d.mtx[4 * r + 2] * p[2] + d.mtx[4 * r + 3];
    if (!on_screen(d.proj, w[0], w[1], w[2], &sx, &sy))
      return false;
    box[0] = std::min(box[0], sx);
    box[1] = std::max(box[1], sx);
    box[2] = std::min(box[2], sy);
    box[3] = std::max(box[3], sy);
  }
  return true;
}

// Whether the frame before had this draw (its texture's nth flat draw then) where machine i's move on the screen since then puts it
// (the HUD does not move with a machine; a name tag does). A machine that hardly moved on the screen: no.
bool moved_with(const Note& d, u32 nth, const float box[4], u32 i) noexcept {
  if (!(s_screenBeforeOn & s_screenNowOn & (1u << i)))
    return false;
  const float mx = s_screenNow[i][0] - s_screenBefore[i][0], my = s_screenNow[i][1] - s_screenBefore[i][1];
  if (std::fabs(mx) + std::fabs(my) < 0.01f)
    return false;
  for (const FlatSeen& s : s_flatBefore) {
    if (s.texture != d.texture || s.nth != nth)
      continue;
    const float bx = (box[0] + box[1] - s.box[0] - s.box[1]) / 2, by = (box[2] + box[3] - s.box[2] - s.box[3]) / 2;
    return s.vertices == d.positions && std::fabs(bx - mx) < 0.003f + 0.15f * std::fabs(mx) &&
           std::fabs(by - my) < 0.003f + 0.15f * std::fabs(my);
  }
  return false;
}

u32 nth_of(const Note& d) noexcept {
  u32 n = 0;
  for (const FlatSeen& s : s_flatNow)
    n += s.texture == d.texture;
  return n;
}

// The machine a flat draw is over (its name tag: drawn by the game at the machine's projected place, above it), or -1: a small box, its
// middle within 0.4 of the machine's place across, its foot from 0.05 below it to 0.9 above, that moved with the machine on the screen
// since the frame before.
int over(const Note& d, u32 nth, const float box[4], const Request& q, bool moving) noexcept {
  if (!s_hasWorld)
    return -1;
  const float middle = (box[0] + box[1]) / 2, foot = box[2];
  int best = -1;
  float nearest = 1e30f;
  for (u32 i = 0; i < MostMachines; ++i) {
    float mx, my;
    if (!(q.machines & (1u << i)) || !on_screen(s_worldProj, q.at[i][0], q.at[i][1], q.at[i][2], &mx, &my))
      continue;
    const float ax = middle - mx, ay = foot - my;
    if (box[1] - box[0] > 0.5f || box[3] - box[2] > 0.3f || std::fabs(ax) > 0.4f || ay < -0.05f || ay > 0.9f || !moving ||
        !moved_with(d, nth, box, i))
      continue;
    if (std::fabs(ax) + std::fabs(ay) < nearest) {
      nearest = std::fabs(ax) + std::fabs(ay);
      best = static_cast<int>(i);
    }
  }
  return best;
}

// Whether any of a draw's vertices lies within MoveNone of a machine in the view
bool near_machine(const Notes& n, const Note& d, const Request& q) noexcept {
  for (u32 v = 0; v < d.positions; ++v) {
    const float* p = &n.positions[(d.firstPosition + v) * 3];
    float w[3];
    for (int r = 0; r < 3; ++r)
      w[r] = d.mtx[4 * r] * p[0] + d.mtx[4 * r + 1] * p[1] + d.mtx[4 * r + 2] * p[2] + d.mtx[4 * r + 3];
    for (u32 i = 0; i < MostMachines; ++i) {
      const float dx = w[0] - q.at[i][0], dy = w[1] - q.at[i][1], dz = w[2] - q.at[i][2];
      if ((q.machines & (1u << i)) && dx * dx + dy * dy + dz * dz < MoveNone * MoveNone)
        return true;
    }
  }
  return false;
}

// The held frame's draws, for a request. `advance`: the first for this frame (its flat draws and machines' places become the frame
// before's for the next); a request again for the same frame (a test's still) keeps the history as it is.
void classify(const Request& q, bool advance) {
  s_class.assign(s_held.notes.size(), World);
  s_hasWorld = false;
  for (const Note& d : s_held.notes)
    if (d.perspective && !gfx::redraw_pass_offscreen(d.pass)) {
      s_worldProj = d.proj;
      s_hasWorld = true;
      break;
    }
  if (advance) {
    std::swap(s_flatBefore, s_flatNow);
    s_flatNow.clear();
    s_screenBefore = s_screenNow;
    s_screenBeforeOn = s_screenNowOn;
    s_screenNowOn = 0;
    for (u32 i = 0; i < MostMachines && s_hasWorld; ++i)
      if ((q.machines & (1u << i)) && on_screen(s_worldProj, q.at[i][0], q.at[i][1], q.at[i][2], &s_screenNow[i][0], &s_screenNow[i][1]))
        s_screenNowOn |= 1u << i;
  }
  // (a classification again for the same frame finds its flat draws' places among those already taken)
  std::vector<FlatSeen> taken;
  if (!advance)
    taken.swap(s_flatNow);
  u64 world = 0, machine = 0, whole = 0, icon = 0, flat = 0, offscreen = 0, byVertex = 0, unreadable = 0, overs = 0;
  for (std::size_t k = 0; k < s_held.notes.size(); ++k) {
    const Note& d = s_held.notes[k];
    int c = World;
    if (gfx::redraw_pass_offscreen(d.pass)) {
      c = None;
      ++(d.perspective ? offscreen : flat);
    } else if (!d.perspective) {
      c = None;
      ++flat;
      float box[4];
      if (flat_box(s_held, d, box)) {
        const u32 nth = nth_of(d);
        s_flatNow.push_back({d.texture, d.positions, nth, {box[0], box[1], box[2], box[3]}});
        if (const int i = over(d, nth, box, q, true); i >= 0) {
          c = Over + i;
          ++overs;
        }
      }
    } else if (identity(d.mtx) && !d.vertexMatrices) {
      if (d.positions == 0) {
        ++unreadable;  // no float positions: the camera's
      } else if (const int i = whole_near(s_held, d, q); i >= 0) {
        c = i;
        ++whole;
      } else {
        c = ByVertex;
        ++byVertex;
      }
    } else if (const int i = d.vertexMatrices ? -1 : icon_over(d, q); i >= 0) {
      c = Icon + i;
      ++icon;
    } else {
      float nearest = Radius * Radius;
      for (u32 i = 0; i < MostMachines; ++i) {
        if (!(q.machines & (1u << i)))
          continue;
        const float dx = d.mtx[3] - q.at[i][0], dy = d.mtx[7] - q.at[i][1], dz = d.mtx[11] - q.at[i][2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < nearest) {
          nearest = d2;
          c = static_cast<int>(i);
        }
      }
      if (c >= 0)
        ++machine;
      else if (d.positions != 0 && !d.vertexMatrices && near_machine(s_held, d, q)) {
        c = ByVertex;  // a world-space draw near a machine (an arc between two): moved vertex by vertex as a view-space one
        ++byVertex;
      }
    }
    world += c == World || c == ByVertex;
    s_class[k] = c;
  }
  s_blocks.clear();
  for (std::size_t k = 0; k < s_held.notes.size(); ++k)
    if (s_class[k] != None)
      s_blocks.push_back(s_held.notes[k].projAt);
  std::sort(s_blocks.begin(), s_blocks.end());
  s_blocks.erase(std::unique(s_blocks.begin(), s_blocks.end()), s_blocks.end());
  if (!advance)
    s_flatNow.swap(taken);
  if (advance) {  // (the counts once a frame)
    s_world += world, s_machine += machine, s_whole += whole, s_icon += icon, s_flat += flat, s_offscreen += offscreen;
    s_byVertex += byVertex, s_unreadable += unreadable, s_over += overs;
    ++s_frames;
  }
}

// A test's table of the held frame's draws: what each is, its correction, and where its view-space vertices are against the nearest
// machine (written on a thread of its own)
void write_table(const Request& q, std::string path) {
  std::string out = "note,class,perspective,pass,offscreen,vertex_matrices,positions,segments,tx,ty,tz,det,texture,machine,near,far,"
                    "box_x0,box_x1,box_y0,box_y1,eye_vertices\n";
  std::vector<u32> segs(s_held.notes.size(), 0);
  for (const Segment& s : s_held.segments)
    ++segs[s.note];
  char line[512];
  for (std::size_t k = 0; k < s_held.notes.size(); ++k) {
    const Note& d = s_held.notes[k];
    const auto& m = d.mtx;
    const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
    int machine = -1;
    float closest = 1e30f, farthest = 0;
    for (u32 v = 0; v < d.positions && d.perspective; ++v) {
      const float* p = &s_held.positions[(d.firstPosition + v) * 3];
      float w[3];
      for (int r = 0; r < 3; ++r)
        w[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
      for (u32 i = 0; i < MostMachines; ++i) {
        if (!(q.machines & (1u << i)))
          continue;
        const float dx = w[0] - q.at[i][0], dy = w[1] - q.at[i][1], dz = w[2] - q.at[i][2];
        const float dd = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dd < closest)
          closest = dd, machine = static_cast<int>(i);
      }
    }
    u32 eye = 0;
    for (u32 v = 0; v < d.positions && machine >= 0; ++v) {
      const float* p = &s_held.positions[(d.firstPosition + v) * 3];
      if (std::fabs(p[0]) + std::fabs(p[1]) + std::fabs(p[2]) < 1e-4f) {
        ++eye;
        continue;
      }
      float w[3];
      for (int r = 0; r < 3; ++r)
        w[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
      const float dx = w[0] - q.at[machine][0], dy = w[1] - q.at[machine][1], dz = w[2] - q.at[machine][2];
      farthest = std::max(farthest, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    float box[4] = {0, 0, 0, 0};
    if (!d.perspective) {
      flat_box(s_held, d, box);
    } else if (d.positions != 0) {
      box[0] = box[2] = 1e30f;
      box[1] = box[3] = -1e30f;
      for (u32 v = 0; v < d.positions; ++v) {
        const float* p = &s_held.positions[(d.firstPosition + v) * 3];
        float w[3], sx, sy;
        for (int r = 0; r < 3; ++r)
          w[r] = m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3];
        if ((std::fabs(p[0]) + std::fabs(p[1]) + std::fabs(p[2]) < 1e-4f) || !on_screen(d.proj, w[0], w[1], w[2], &sx, &sy))
          continue;
        box[0] = std::min(box[0], sx), box[1] = std::max(box[1], sx), box[2] = std::min(box[2], sy), box[3] = std::max(box[3], sy);
      }
    }
    std::snprintf(line, sizeof line, "%zu,%d,%d,%u,%d,%d,%u,%u,%.2f,%.2f,%.2f,%.3f,%08X,%d,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%u\n", k,
                  s_class[k], d.perspective ? 1 : 0, d.pass, gfx::redraw_pass_offscreen(d.pass) ? 1 : 0, d.vertexMatrices ? 1 : 0,
                  d.positions, segs[k], m[3], m[7], m[11], det, d.texture, machine, closest, farthest, box[0], box[1], box[2], box[3], eye);
    out += line;
  }
  out += "machines";
  for (u32 i = 0; i < MostMachines; ++i)
    if (q.machines & (1u << i)) {
      std::snprintf(line, sizeof line, ",%u:%.2f:%.2f:%.2f", i, q.at[i][0], q.at[i][1], q.at[i][2]);
      out += line;
    }
  out += "\n";
  std::thread([out = std::move(out), path = std::move(path)] {
    if (FILE* f = std::fopen(path.c_str(), "w")) {
      std::fwrite(out.data(), 1, out.size(), f);
      std::fclose(f);
    }
  }).detach();
}

// The view-space draws' vertices of a picture, each near a machine moved by that machine's own move (`motion`): fully within MoveFull
// units of its place in the true view, less and less to none at MoveNone; written into the frame's vertex buffer.
void move_vertices(const Request& q, const std::array<Matrix, MostMachines>& motion) {
  if (s_dirty.size() != s_held.segments.size())
    s_dirty.assign(s_held.segments.size(), false);
  u64 moved = 0, written = 0;
  for (std::size_t n = 0; n < s_held.segments.size(); ++n) {
    const Segment& s = s_held.segments[n];
    if (s_class[s.note] != ByVertex || s.alignedBytes == 0)
      continue;
    s_scratch.assign(s_held.bytes.begin() + s.bytesAt, s_held.bytes.begin() + s.bytesAt + s.alignedBytes);
    // a world-space draw's vertices into the view and back (its position matrix and that one's inverse)
    const auto& m = s_held.notes[s.note].mtx;
    const bool model = !identity(m);
    const Matrix toView{m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], 0, 0, 0, 1};
    const Matrix toModel = model ? inverse(toView) : toView;
    // a line draw (an arc from a machine): the machine its nearest vertex is within WholeNear of, all of it moved with that one
    int whole = -1;
    if (s.lines) {
      float nearest = WholeNearUnits * WholeNearUnits;
      for (u32 v = 0; v < s.count; ++v) {
        const u8* at = s_scratch.data() + (s.vertexAt - s.alignedAt) + static_cast<std::size_t>(v) * s.stride + s.positionAt;
        const float raw[3] = {be_float(at), be_float(at + 4), be_float(at + 8)};
        float p[3] = {raw[0], raw[1], raw[2]};
        if (model)
          apply(toView, raw, p);
        for (u32 i = 0; i < MostMachines; ++i) {
          const float dx = p[0] - q.at[i][0], dy = p[1] - q.at[i][1], dz = p[2] - q.at[i][2];
          const float d2 = dx * dx + dy * dy + dz * dz;
          if ((q.machines & (1u << i)) && d2 < nearest)
            nearest = d2, whole = static_cast<int>(i);
        }
      }
    }
    bool any = false;
    for (u32 v = 0; v < s.count && s_testTexture != 1 && s_testTexture != s_held.notes[s.note].texture; ++v) {
      u8* where = s_scratch.data() + (s.vertexAt - s.alignedAt) + static_cast<std::size_t>(v) * s.stride + s.positionAt;
      const float raw[3] = {be_float(where), be_float(where + 4), be_float(where + 8)};
      float p[3] = {raw[0], raw[1], raw[2]};
      if (model)
        apply(toView, raw, p);
      int best = whole;
      float nearest = MoveNone * MoveNone;
      for (u32 i = 0; i < MostMachines && !s.lines; ++i) {
        if (!(q.machines & (1u << i)))
          continue;
        const float dx = p[0] - q.at[i][0], dy = p[1] - q.at[i][1], dz = p[2] - q.at[i][2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < nearest) {
          nearest = d2;
          best = static_cast<int>(i);
        }
      }
      if (best < 0)
        continue;
      const float w = s.lines ? 1.f : std::clamp((MoveNone - std::sqrt(nearest)) / (MoveNone - MoveFull), 0.f, 1.f);
      float to[3], o[3], back[3];
      apply(motion[best], p, to);
      for (int r = 0; r < 3; ++r)
        o[r] = p[r] + w * (to[r] - p[r]);
      if (s_testTexture == 3)
        o[1] += 3.f;  // (a test: every moved vertex 3 units up)
      if (o[0] == p[0] && o[1] == p[1] && o[2] == p[2])
        std::memcpy(back, raw, sizeof back);  // (not moved: the bytes as they were)
      else if (model)
        apply(toModel, o, back);
      else
        std::memcpy(back, o, sizeof back);
      for (int r = 0; r < 3; ++r)
        put_be_float(where + 4 * r, back[r]);
      ++moved;
      any = true;
    }
    if (any || s_dirty[n]) {
      gfx::redraw_write_vertices(s.alignedAt, s_scratch.data(), s.alignedBytes);
      written += s.alignedBytes;
      s_dirty[n] = any;
    }
  }
  s_moved += moved;
  s_written += written;
}

// p = p * c (a correction after the modelview)
void times(float* p, const Matrix& c) noexcept {
  float out[16];
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      out[4 * r + k] = p[4 * r] * c[k] + p[4 * r + 1] * c[4 + k] + p[4 * r + 2] * c[8 + k] + p[4 * r + 3] * c[12 + k];
  std::memcpy(p, out, sizeof out);
}
// p = (a move of dx, dy on the screen) * p
void shifted(float* p, float dx, float dy) noexcept {
  for (int k = 0; k < 4; ++k) {
    p[k] += dx * p[12 + k];
    p[4 + k] += dy * p[12 + k];
  }
}
} // namespace

void start() noexcept {
  const char* v = std::getenv("AURORA_REDRAW");
  g_on = v != nullptr && v[0] == '1';
}

void note_draw(u16 vtxCount, const u8* vertices, u32 positionAt, u32 stride, u32 vertexAt, u32 uniformOffset, u32 lineMode, u32 pass,
               u32 texture, bool vertexMatrices) noexcept {
  const auto& pn = g_gxState.pnMtx[g_gxState.currentPnMtx];
  std::array<float, 12> mtx;
  static_assert(sizeof(pn.pos) == sizeof(mtx));
  std::memcpy(mtx.data(), &pn.pos, sizeof mtx);
  const bool inView = identity(mtx) && !vertexMatrices;
  if (!s_building.notes.empty() && s_building.notes.back().uniformOffset == uniformOffset) {
    if (s_building.notes.back().mtx != mtx)
      ++s_building.split;
  } else {
    Note n;
    n.uniformOffset = uniformOffset;
    n.projAt = uniformOffset + (lineMode != 0 ? 32u : 16u);
    n.pass = pass;
    n.mtx = mtx;
    static_assert(sizeof(g_gxState.proj) == sizeof(n.proj));
    std::memcpy(n.proj.data(), &g_gxState.proj, sizeof n.proj);
    n.perspective = g_gxState.projType == GX_PERSPECTIVE;
    n.vertexMatrices = vertexMatrices;
    n.texture = texture;
    n.firstPosition = static_cast<u32>(s_building.positions.size() / 3);
    s_building.notes.push_back(n);
  }
  Note& n = s_building.notes.back();
  // the view-space draws' positions (whole-near, moved vertex by vertex) and the flat draws' (their box on the screen)
  // (and a world-space line draw's: the electric arcs the game builds in the world each frame from a machine)
  const bool smallWorld = lineMode != 0 && n.perspective && !inView && !vertexMatrices && vtxCount <= MostWorldVertices &&
                          n.positions + vtxCount <= MostWorldVertices;
  if (stride != 0 && ((inView && n.perspective) || !n.perspective || smallWorld)) {
    for (u32 v = 0; v < vtxCount; ++v) {
      const u8* p = vertices + v * stride + positionAt;
      s_building.positions.push_back(be_float(p));
      s_building.positions.push_back(be_float(p + 4));
      s_building.positions.push_back(be_float(p + 8));
    }
    n.positions += vtxCount;
    if (((inView && n.perspective) || smallWorld) && !n.vertexMatrices && vtxCount != 0) {
      Segment s;
      s.note = static_cast<u32>(s_building.notes.size() - 1);
      s.vertexAt = vertexAt;
      s.count = vtxCount;
      s.stride = stride;
      s.positionAt = positionAt;
      s.lines = lineMode != 0;
      s_building.segments.push_back(s);
    }
  }
}

Notes take_notes() noexcept {
  Notes out = std::move(s_building);
  s_building = {};
  s_building.notes.reserve(out.notes.size());
  s_building.positions.reserve(out.positions.size());
  s_building.segments.reserve(out.segments.size());
  return out;
}

void keep_vertices(Notes& notes, const u8* vertices, std::size_t size) noexcept {
  std::size_t total = 0;
  for (const Segment& s : notes.segments)
    total += static_cast<std::size_t>(s.count) * s.stride + 8;
  notes.bytes.clear();
  notes.bytes.reserve(total);
  for (Segment& s : notes.segments) {
    const std::size_t end = static_cast<std::size_t>(s.vertexAt) + static_cast<std::size_t>(s.count) * s.stride;
    const std::size_t from = s.vertexAt & ~3u, to = std::min<std::size_t>((end + 3) & ~std::size_t{3}, size);
    if (vertices == nullptr || end > size || to <= from) {
      s.alignedBytes = 0;
      continue;
    }
    s.alignedAt = static_cast<u32>(from);
    s.alignedBytes = static_cast<u32>(to - from);
    s.bytesAt = static_cast<u32>(notes.bytes.size());
    notes.bytes.resize(notes.bytes.size() + (to - from));
    gfx::copy_from_upload(notes.bytes.data() + s.bytesAt, vertices + from, to - from);
  }
}

void hold(Notes&& notes) noexcept {
  s_held = std::move(notes);
  s_classified = false;
  s_classifiedFor = nullptr;
  s_advanced = false;
  s_dirty.assign(s_held.segments.size(), false);
  s_splits += s_held.split;
}

void table_next(std::string path) noexcept { s_tablePath = std::move(path); }
void test_uncorrected(u32 texture) noexcept { s_testTexture = texture; }

bool draw_picture(const Request& request, std::size_t index) {
  if (index >= request.pictures.size())
    return false;
  gfx::stagelog::Scope pictureTime{"rd_picture"};
  if (!s_classified || s_classifiedFor != &request) {
    gfx::stagelog::Scope t{"rd_classify"};
    classify(request, !s_advanced);
    s_advanced = true;
    s_classified = true;
    s_classifiedFor = &request;
    if (!s_tablePath.empty()) {
      write_table(request, std::move(s_tablePath));
      s_tablePath.clear();
    }
  }
  const Picture& picture = request.pictures[index];
  if (rt::g_live)
    rt::live::between_set(request, index);
  // each machine's own move in the true view (the camera's correction undone from its own), and its rank icon's correction (the
  // camera's, moved on as the machine's place is) and its move on the screen (what is drawn flat over it)
  const Matrix unCamera = inverse(picture.camera);
  std::array<Matrix, MostMachines> motion, icon;
  std::array<std::array<float, 2>, MostMachines> shift{};
  for (u32 i = 0; i < MostMachines; ++i) {
    motion[i] = mul(unCamera, picture.machine[i]);
    float at[3], moved[3], seen[3];
    apply(picture.machine[i], request.at[i].data(), at);
    apply(picture.camera, request.at[i].data(), seen);
    icon[i] = picture.camera;
    icon[i][3] += at[0] - seen[0];
    icon[i][7] += at[1] - seen[1];
    icon[i][11] += at[2] - seen[2];
    float x0, y0, x1, y1;
    std::memcpy(moved, at, sizeof moved);
    if (s_hasWorld && (request.machines & (1u << i)) &&
        on_screen(s_worldProj, request.at[i][0], request.at[i][1], request.at[i][2], &x0, &y0) &&
        on_screen(s_worldProj, moved[0], moved[1], moved[2], &x1, &y1))
      shift[i] = {x1 - x0, y1 - y0};
  }
  {
    gfx::stagelog::Scope t{"rd_vertices"};
    move_vertices(request, motion);
  }
  const bool drawn = gfx::redraw_held([&](u8* uniforms, std::size_t bytes) {
    s_bytes = bytes;
    for (std::size_t k = 0; k < s_held.notes.size(); ++k) {
      const int c = s_class[k];
      if (c == None)
        continue;
      const Note& d = s_held.notes[k];
      if (d.projAt + 64 > bytes || (s_testTexture > 1 && d.texture == s_testTexture))
        continue;
      float* p = reinterpret_cast<float*>(uniforms + d.projAt);
      if (c >= Over)
        shifted(p, shift[c - Over][0], shift[c - Over][1]);
      else if (c >= Icon)
        times(p, icon[c - Icon]);
      else
        times(p, c == World || c == ByVertex ? picture.camera : picture.machine[static_cast<u32>(c)]);
    }
  }, gfx::detail::g_redrawBundles ? &s_blocks : nullptr);
  if (drawn)
    ++s_pictures;
  return drawn;
}

bool bundles() noexcept { return gfx::detail::g_redrawBundles; }
void set_bundles(bool on) noexcept { gfx::detail::g_redrawBundles = on; }

Counts counts() noexcept {
  Counts c;
  c.pictures = s_pictures.load(), c.world = s_world.load(), c.machine = s_machine.load(), c.whole = s_whole.load(), c.icon = s_icon.load();
  c.flat = s_flat.load(), c.offscreen = s_offscreen.load(), c.splits = s_splits.load(), c.frames = s_frames.load();
  c.uniformBytes = s_bytes.load(), c.byVertex = s_byVertex.load(), c.unreadable = s_unreadable.load(), c.moved = s_moved.load();
  c.over = s_over.load(), c.written = s_written.load();
  return c;
}
} // namespace aurora::gx::redraw
