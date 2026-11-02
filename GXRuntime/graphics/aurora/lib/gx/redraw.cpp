// Pictures between two steps by drawing the true frame again. The game bakes its view into every position matrix on the CPU, so each
// world draw reaches the GPU as P * (V * M): a correction C = V_new * V_old^-1 after the modelview, P' = P * C, moves it by the camera's
// change; a machine i's draws take C_i = V_new * W_i * V_old^-1 (research/study-redraw.md). The projection sits in each draw's own uniform
// block, so a picture is the frame's uniforms with those patched, then the frame's passes encoded again.
//
// Which correction a draw takes comes from the draw's own data (FrameRedrawTags.cpp's rules, on Aurora's data):
//  - orthographic, or in a pass that ends in a copy smaller than half the picture (shadow pictures, portraits): none;
//  - an identity position matrix (the game put its vertices in the view) with every vertex within WholeNear of one machine: that machine's;
//    otherwise the camera's (a long trail's head can lag its machine by the part of a step: Aurora's vertices are not moved one by one here);
//  - a model scaled 2 times or more standing 4 to 40 units straight up from a machine (its rank icon): that machine's;
//  - a position matrix whose translation is within Radius of a machine: that machine's; else the camera's.
// Tuned numbers, from the Vulkan re-draw work and not re-tuned here: Radius 6, WholeNear 10, the icon's 4 to 40 and 8 (2 cubed).
#include "redraw.hpp"

#include "../gfx/frame.hpp"
#include "rt_capture.hpp"
#include "rt_live.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace aurora::gx::redraw {
bool g_on = false;

namespace {
constexpr float Radius = 6.f, WholeNearUnits = 10.f;
constexpr int None = -1, World = -2;

Notes s_building;
Notes s_held;
std::vector<int> s_class;  // the held frame's notes' corrections: a machine, World or None
bool s_classified = false;
const Request* s_classifiedFor = nullptr;
std::atomic<u64> s_pictures{0}, s_world{0}, s_machine{0}, s_whole{0}, s_icon{0}, s_flat{0}, s_offscreen{0}, s_splits{0}, s_frames{0}, s_bytes{0};

float be_float(const u8* p) noexcept {
  u32 v;
  std::memcpy(&v, p, 4);
  v = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

bool identity(const std::array<float, 12>& m) noexcept {
  static constexpr std::array<float, 12> one{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 12; ++i)
    if (std::fabs(m[i] - one[i]) > 1e-5f)
      return false;
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

void classify(const Request& q) {
  s_class.resize(s_held.notes.size());
  u64 world = 0, machine = 0, whole = 0, icon = 0, flat = 0, offscreen = 0;
  for (std::size_t k = 0; k < s_held.notes.size(); ++k) {
    const Note& d = s_held.notes[k];
    int c = World;
    if (!d.perspective) {
      c = None;
      ++flat;
    } else if (gfx::redraw_pass_offscreen(d.pass)) {
      c = None;
      ++offscreen;
    } else if (identity(d.mtx)) {
      const int i = whole_near(s_held, d, q);
      if (i >= 0) {
        c = i;
        ++whole;
      }
    } else if (const int i = icon_over(d, q); i >= 0) {
      c = i;
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
    }
    world += c == World;
    s_class[k] = c;
  }
  s_world += world, s_machine += machine, s_whole += whole, s_icon += icon, s_flat += flat, s_offscreen += offscreen;
  ++s_frames;
}

void times(float* p, const Matrix& c) noexcept {
  float out[16];
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      out[4 * r + k] = p[4 * r] * c[k] + p[4 * r + 1] * c[4 + k] + p[4 * r + 2] * c[8 + k] + p[4 * r + 3] * c[12 + k];
  std::memcpy(p, out, sizeof out);
}
} // namespace

void start() noexcept {
  const char* v = std::getenv("AURORA_REDRAW");
  g_on = v != nullptr && v[0] == '1';
}

void note_draw(u16 vtxCount, const u8* vertices, u32 positionAt, u32 stride, u32 uniformOffset, u32 lineMode, u32 pass) noexcept {
  const auto& pn = g_gxState.pnMtx[g_gxState.currentPnMtx];
  std::array<float, 12> mtx;
  static_assert(sizeof(pn.pos) == sizeof(mtx));
  std::memcpy(mtx.data(), &pn.pos, sizeof mtx);
  const bool inView = identity(mtx);
  if (!s_building.notes.empty() && s_building.notes.back().uniformOffset == uniformOffset) {
    if (s_building.notes.back().mtx != mtx)
      ++s_building.split;
  } else {
    Note n;
    n.uniformOffset = uniformOffset;
    n.projAt = uniformOffset + (lineMode != 0 ? 32u : 16u);
    n.pass = pass;
    n.mtx = mtx;
    n.perspective = g_gxState.projType == GX_PERSPECTIVE;
    n.firstPosition = static_cast<u32>(s_building.positions.size() / 3);
    s_building.notes.push_back(n);
  }
  if (inView && stride != 0 && s_building.notes.back().perspective) {
    for (u32 v = 0; v < vtxCount; ++v) {
      const u8* p = vertices + v * stride + positionAt;
      s_building.positions.push_back(be_float(p));
      s_building.positions.push_back(be_float(p + 4));
      s_building.positions.push_back(be_float(p + 8));
    }
    s_building.notes.back().positions += vtxCount;
  }
}

Notes take_notes() noexcept {
  Notes out = std::move(s_building);
  s_building = {};
  s_building.notes.reserve(out.notes.size());
  s_building.positions.reserve(out.positions.size());
  return out;
}

void hold(Notes&& notes) noexcept {
  s_held = std::move(notes);
  s_classified = false;
  s_classifiedFor = nullptr;
  s_splits += s_held.split;
}

bool draw_picture(const Request& request, std::size_t index) {
  if (index >= request.pictures.size())
    return false;
  if (!s_classified || s_classifiedFor != &request) {
    classify(request);
    s_classified = true;
    s_classifiedFor = &request;
  }
  const Picture& picture = request.pictures[index];
  if (rt::g_live)
    rt::live::between_set(request, index);
  const bool drawn = gfx::redraw_held([&](u8* uniforms, std::size_t bytes) {
    s_bytes = bytes;
    for (std::size_t k = 0; k < s_held.notes.size(); ++k) {
      const int c = s_class[k];
      if (c == None)
        continue;
      const Note& d = s_held.notes[k];
      if (d.projAt + 64 > bytes)
        continue;
      times(reinterpret_cast<float*>(uniforms + d.projAt), c == World ? picture.camera : picture.machine[static_cast<u32>(c)]);
    }
  });
  if (drawn)
    ++s_pictures;
  return drawn;
}

Counts counts() noexcept {
  return {s_pictures.load(), s_world.load(), s_machine.load(), s_whole.load(), s_icon.load(), s_flat.load(), s_offscreen.load(), s_splits.load(), s_frames.load(), s_bytes.load()};
}
} // namespace aurora::gx::redraw
