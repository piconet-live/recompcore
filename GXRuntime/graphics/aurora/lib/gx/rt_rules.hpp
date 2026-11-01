#pragma once
// What the ray tracer derives from a frame's own draws, for any model: no tables, no names, no graphics API.
#include "rt_format.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace aurora::gx::rt {
struct EngineLight {
  float pos[3];
  float colour[3];  // the flame's own K colour
};

// Engine lights: a machine's flame is drawn as a plume, additive quads without depth writes, one stage, one texture, each
// a step behind the last with its K colour fading. The first of such a run sits at the nozzle and holds the flame's colour,
// so a custom flame colour carries through. Nothing here looks at where the flame points. Other additive draws (signs,
// beams, sparks, decals) have no fading run and cast no light. `from`/`to`: the draws of the main view.
inline std::vector<EngineLight> engine_lights(const DrawRec* draws, std::uint32_t count, const Vertex* verts,
                                              bool (*in_view)(const DrawRec&)) {
  constexpr std::uint32_t MinRun = 4, Look = 3;  // quads in a run; draws looked ahead for the next (degenerate twins between)
  constexpr float MaxStep = 3.f, MinBright = 0.25f;
  const auto bright = [](const DrawRec& d) { return std::max({d.stage[0].k[0], d.stage[0].k[1], d.stage[0].k[2]}); };
  const auto plume = [&](const DrawRec& d) {
    return in_view(d) && d.vertices > 0 && d.blendMode == 1 && d.blendDst == 1 && !d.depthUpdate && d.stages == 1;
  };
  const auto centre = [&](const DrawRec& d, float* c) {
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (std::uint32_t v = 0; v < d.vertices; ++v)
      for (int a = 0; a < 3; ++a) {
        mn[a] = std::min(mn[a], verts[d.firstVertex + v].pos[a]);
        mx[a] = std::max(mx[a], verts[d.firstVertex + v].pos[a]);
      }
    float size = 0;
    for (int a = 0; a < 3; ++a) {
      c[a] = (mn[a] + mx[a]) / 2;
      size = std::max(size, mx[a] - mn[a]);
    }
    return size;
  };
  std::vector<EngineLight> out;
  for (std::uint32_t i = 0; i < count; ++i) {
    const DrawRec& head = draws[i];
    float at[3];
    if (!plume(head) || bright(head) < MinBright || centre(head, at) <= 0.f)
      continue;
    std::uint32_t run = 1, last = i;
    float was[3] = {at[0], at[1], at[2]}, level = bright(head);
    for (std::uint32_t j = i + 1; j < count && j <= last + Look; ++j) {
      const DrawRec& d = draws[j];
      float c[3];
      if (!plume(d) || centre(d, c) <= 0.f)
        continue;
      if (d.stage[0].tex != head.stage[0].tex || bright(d) >= level)
        break;
      if (std::hypot(c[0] - was[0], c[1] - was[1], c[2] - was[2]) > MaxStep)
        break;
      level = bright(d);
      was[0] = c[0], was[1] = c[1], was[2] = c[2];
      last = j;
      ++run;
    }
    if (run < MinRun)
      continue;
    out.push_back({{at[0], at[1], at[2]}, {head.stage[0].k[0], head.stage[0].k[1], head.stage[0].k[2]}});
    i = last;
  }
  return out;
}

// A machine: a draw right beside an engine light. Only machines earn the boosted reflection; nothing else is told apart.
inline bool beside_engine_light(const DrawRec& d, const Vertex* verts, const std::vector<EngineLight>& lights, float reach = 5.f) {
  if (d.vertices == 0)
    return false;
  float c[3]{};
  for (std::uint32_t v = 0; v < d.vertices; ++v)
    for (int a = 0; a < 3; ++a)
      c[a] += verts[d.firstVertex + v].pos[a] / d.vertices;
  for (const auto& l : lights)
    if (std::hypot(c[0] - l.pos[0], c[1] - l.pos[1], c[2] - l.pos[2]) < reach)
      return true;
  return false;
}
} // namespace aurora::gx::rt
