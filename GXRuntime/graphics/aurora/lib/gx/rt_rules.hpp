#pragma once
#include <memory>
#include <new>
#include <utility>
// What the ray tracer derives from a frame's own draws, for any model: no tables, no names, no graphics API.
#include "rt_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <utility>
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

// What the rays read of a draw: small, the stages apart
struct GpuDraw {
  std::uint32_t stages, firstStage, cls;  // cls: 0 solid, 1 translucent, 2 additive, 3 ignored
  float strength;                         // reflection strength; machines boosted, scenery capped
  std::uint32_t highlight;                // has a stage looked up by the normal: the game's own shine
  std::uint32_t fogType;
  float fog[3];
  float fogColour[3];
  float regs[4][4];
  std::uint32_t swap[4];  // one table a word, a byte a channel
};
struct Tuning {
  float machineGain = 2.f, machineCap = 0.45f, sceneryCap = 0.25f;
  std::uint32_t maxLights = 16;
};
struct Frame {
  std::vector<Vertex> verts;        // scene triangles first, then the sky's
  std::uint32_t skyFirstVertex = 0; // the sky's first vertex
  std::vector<GpuDraw> draws;
  std::vector<StageRec> stages;
  std::vector<EngineLight> lights;
  const DrawRec* main = nullptr;
  std::uint32_t machineDraws = 0, reflective = 0;
};

// The main view is the perspective projection with the most triangles. Sky: the largest see-through shells around the view,
// by size against the largest in the frame, with no table of courses.
inline void build_frame(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const Tuning& t, Frame& out) {
  out.verts.clear(), out.draws.clear(), out.stages.clear(), out.main = nullptr, out.machineDraws = out.reflective = 0, out.skyFirstVertex = 0;
  const auto key = [](const DrawRec& d) { return std::array<int, 2>{static_cast<int>(d.proj[0] * 1000), static_cast<int>(d.proj[5] * 1000)}; };
  std::array<int, 2> best{};
  {
    // few distinct projections: a short list beats a map
    std::array<std::pair<std::array<int, 2>, std::uint64_t>, 16> seen{};
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      if (draws[i].projType != 0)
        continue;
      const auto k = key(draws[i]);
      std::uint32_t j = 0;
      while (j < n && seen[j].first != k)
        ++j;
      if (j == n) {
        if (n == seen.size())
          continue;
        seen[n++].first = k;
      }
      seen[j].second += draws[i].vertices;
    }
    std::uint64_t most = 0;
    for (std::uint32_t j = 0; j < n; ++j)
      if (seen[j].second > most)
        most = seen[j].second, best = seen[j].first;
  }
  static thread_local std::array<int, 2> s_best;
  s_best = best;
  out.lights = engine_lights(draws, count, verts, [](const DrawRec& d) {
    return d.projType == 0 && std::array<int, 2>{static_cast<int>(d.proj[0] * 1000), static_cast<int>(d.proj[5] * 1000)} == s_best;
  });
  // the nearest few: a packed grid must stay affordable (the camera is the origin)
  if (out.lights.size() > t.maxLights) {
    const auto dist = [](const EngineLight& l) { return l.pos[0] * l.pos[0] + l.pos[1] * l.pos[1] + l.pos[2] * l.pos[2]; };
    std::nth_element(out.lights.begin(), out.lights.begin() + t.maxLights, out.lights.end(),
                     [&](const EngineLight& a, const EngineLight& b) { return dist(a) < dist(b); });
    out.lights.resize(t.maxLights);
  }
  out.draws.resize(count);
  std::vector<float> extent(count, 0.f);
  float largest = 0.f;
  for (std::uint32_t i = 0; i < count; ++i) {
    const DrawRec& d = draws[i];
    GpuDraw& g = out.draws[i];
    g.stages = std::min<std::uint32_t>(d.stages, 16u);
    g.firstStage = static_cast<std::uint32_t>(out.stages.size());
    for (std::uint32_t s = 0; s < g.stages; ++s) {
      out.stages.push_back(d.stage[s]);
      if (d.stage[s].reflection == 2)
        g.highlight = 1;
    }
    float strength = 0;
    for (std::uint32_t s = 0; s < g.stages; ++s)
      if (d.stage[s].reflection == 1) {
        const auto& st = d.stage[s];
        strength = std::max(strength, st.cc == 14 ? std::max({st.k[0], st.k[1], st.k[2]}) : 1.f);
      }
    std::memcpy(g.regs, d.regs, sizeof g.regs);
    for (int a = 0; a < 4; ++a)
      g.swap[a] = static_cast<std::uint32_t>(d.swap[a][0]) | static_cast<std::uint32_t>(d.swap[a][1]) << 8 |
                  static_cast<std::uint32_t>(d.swap[a][2]) << 16 | static_cast<std::uint32_t>(d.swap[a][3]) << 24;
    g.fogType = d.fogType;
    g.fog[0] = d.fog[0], g.fog[1] = d.fog[1], g.fog[2] = d.fog[2];
    g.fogColour[0] = d.fogColour[0], g.fogColour[1] = d.fogColour[1], g.fogColour[2] = d.fogColour[2];
    // the blend the draw asks for: GX factors ZERO 0, ONE 1, SRCCLR 2, INVSRCCLR 3, SRCALPHA 4, INVSRCALPHA 5
    const bool additive = d.blendMode == 1 && d.blendDst == 1;
    const bool darkens = d.blendMode == 1 && (d.blendDst == 2 || d.blendSrc == 0 || d.blendSrc == 2);
    const bool opaqueBlend = d.blendMode == 0 || (d.blendSrc == 1 && d.blendDst == 0);
    bool translucent = false;
    if (!opaqueBlend && !additive && !darkens) {
      for (std::uint32_t v = 0; v < d.vertices && !translucent; v += 3)
        translucent = verts[d.firstVertex + v].ras[3] < 0.99f;
      for (std::uint32_t s = 0; s < g.stages && !translucent; ++s) {
        const auto& st = d.stage[s];
        translucent = st.aa == 4 || st.ab == 4 || st.ac == 4 || st.ad == 4 || st.aa == 6 || st.ab == 6 || st.ac == 6 || st.ad == 6;
      }
    }
    g.cls = !d.colorUpdate || darkens ? 3 : additive ? 2 : translucent ? 1 : 0;
    const bool inView = d.projType == 0 && key(d) == best && d.vertices > 0;
    if (strength > 0 && inView) {
      ++out.reflective;
      if (beside_engine_light(d, verts, out.lights)) {
        strength = std::max(strength, std::min(strength * t.machineGain, t.machineCap));
        ++out.machineDraws;
      } else {
        strength = std::min(strength, t.sceneryCap);
      }
    }
    g.strength = strength;
    if (!inView)
      continue;
    if (!out.main)
      out.main = &d;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (std::uint32_t v = 0; v < d.vertices; ++v)
      for (int a = 0; a < 3; ++a) {
        mn[a] = std::min(mn[a], verts[d.firstVertex + v].pos[a]);
        mx[a] = std::max(mx[a], verts[d.firstVertex + v].pos[a]);
      }
    extent[i] = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
    largest = std::max(largest, extent[i]);
  }
  const auto sky = [&](std::uint32_t i) { return extent[i] > largest * 0.3f && out.draws[i].cls != 0 && out.draws[i].cls != 3; };
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1)
      out.skyFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    for (std::uint32_t i = 0; i < count; ++i) {
      if (extent[i] <= 0.f || sky(i) != (pass == 1))
        continue;
      out.verts.insert(out.verts.end(), verts + draws[i].firstVertex, verts + draws[i].firstVertex + draws[i].vertices);
    }
  }
}
} // namespace aurora::gx::rt
