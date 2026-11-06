#pragma once
#include <cmath>
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
  float colour[3];  // the draw's own colour, by its strength
  float reach;      // 1 for a machine's flame, less for a small dim one
  std::uint32_t owner;  // 1 + the machine whose draws it came from (the game's draw marks: DrawRec::tag), 0 none (a course lamp, a mine's blast)
};

// The one rule for lights: anything additive, small and bright in the main view casts light. No labels: a flame, a spark or a
// mine's blast are the same to it. Colour from the draw, strength its brightness by its size. Lights follow what the game has:
// the draws the game marked as one burner of a machine (the draw marks: one Burner record, one nozzle of the machine's .fmi)
// become that burner's light, so a machine with three engines has three; a machine's other marked glows (its body's) one more;
// never merged across burners or machines. Together they are as strong as one light of all the machine's flames (build_frame).
// Unmarked draws (a course's lamps, sparks, blasts) within MergeR of one another become one.
struct LightPick {
  float pos[3], colour[3], weight, extent, bright;
  std::uint32_t owner;   // EngineLight::owner
  std::uint32_t burner;  // which of the owner's burners (tag_burner), 0 none
};
inline void light_picks(const DrawRec* draws, std::uint32_t count, bool (*in_view)(const DrawRec&), std::vector<LightPick>& out, float maxSize, float minBright) {
  out.clear();
  for (std::uint32_t i = 0; i < count; ++i) {
    const DrawRec& d = draws[i];
    if (d.lightExtent <= 0.f || d.lightExtent >= maxSize || d.lightBright <= minBright || !d.colorUpdate || !in_view(d))
      continue;
    out.push_back({{d.lightPos[0], d.lightPos[1], d.lightPos[2]},
                   {d.lightColour[0], d.lightColour[1], d.lightColour[2]},
                   d.lightBright * std::max(d.lightExtent, 0.5f), d.lightExtent, d.lightBright, tag_machine(d.tag), tag_burner(d.tag)});
  }
}
inline void merge_lights(std::vector<LightPick>& picks, float mergeR, std::vector<LightPick>& out) {
  std::sort(picks.begin(), picks.end(), [](const LightPick& a, const LightPick& b) { return a.weight > b.weight; });
  out.clear();
  for (const LightPick& p : picks) {
    bool merged = false;
    for (LightPick& q : out) {
      if (q.owner != p.owner || q.burner != p.burner)
        continue;
      const float dx = q.pos[0] / q.weight - p.pos[0], dy = q.pos[1] / q.weight - p.pos[1], dz = q.pos[2] / q.weight - p.pos[2];
      if (p.owner != 0 || dx * dx + dy * dy + dz * dz < mergeR * mergeR) {
        for (int a = 0; a < 3; ++a)
          q.pos[a] += p.pos[a] * p.weight, q.colour[a] += p.colour[a] * p.weight;
        q.weight += p.weight;
        merged = true;
        break;
      }
    }
    if (!merged) {
      LightPick q = p;
      for (int a = 0; a < 3; ++a)
        q.pos[a] *= p.weight, q.colour[a] *= p.weight;
      out.push_back(q);
    }
  }
  for (LightPick& q : out)
    for (int a = 0; a < 3; ++a)
      q.pos[a] /= q.weight, q.colour[a] /= q.weight;
}

// A big glowing surface as light: a piece of it, its power as colour by area
struct GlowLight {
  float pos[3], area, colour[3], radius;
  std::uint32_t owner;  // EngineLight::owner: the machine whose draws the piece is of, 0 none
};
// What a draw's stages give on its textures' mean colours and its rasterised colour: the surface's colour on the whole, no pixels read.
// Reflection layers left out, as the rays do.
inline void draw_colour(const DrawRec& d, const Vertex* verts, const float (*texMeans)[4], std::uint32_t texCount, float out[4]) {
  float ras[4]{};
  const std::uint32_t step = std::max<std::uint32_t>(d.vertices / 8, 1);
  std::uint32_t n = 0;
  for (std::uint32_t v = 0; v < d.vertices; v += step, ++n)
    for (int c = 0; c < 4; ++c)
      ras[c] += verts[d.firstVertex + v].ras[c];
  for (int c = 0; c < 4; ++c)
    ras[c] /= static_cast<float>(std::max(n, 1u));
  float r[4][4];
  std::memcpy(r, d.regs, sizeof r);
  for (std::uint32_t i = 0; i < std::min<std::uint32_t>(d.stages, 16u); ++i) {
    const StageRec& s = d.stage[i];
    if (s.reflection != 0)
      continue;
    float white[4] = {1.f, 1.f, 1.f, 1.f};
    const float* t = s.tex < texCount ? texMeans[s.tex] : white;
    const auto ci = [&](std::uint32_t a, int c) -> float {
      if (a < 8)
        return r[a / 2][a % 2 == 0 ? c : 3];
      switch (a) {
      case 8: return t[c];
      case 9: return t[3];
      case 10: return ras[c];
      case 11: return ras[3];
      case 12: return 1.f;
      case 13: return 0.5f;
      case 14: return s.k[c];
      default: return 0.f;
      }
    };
    const auto ai = [&](std::uint32_t a) -> float {
      switch (a) {
      case 0: case 1: case 2: case 3: return r[a][3];
      case 4: return t[3];
      case 5: return ras[3];
      case 6: return s.k[3];
      default: return 0.f;
      }
    };
    const auto bias = [](std::uint32_t b) { return b == 1 ? 0.5f : b == 2 ? -0.5f : 0.f; };
    const auto scale = [](std::uint32_t b) { return b == 1 ? 2.f : b == 2 ? 4.f : b == 3 ? 0.5f : 1.f; };
    float col[3];
    for (int c = 0; c < 3; ++c) {
      const float k = ci(s.cc, c), l = ci(s.ca, c) * (1 - k) + ci(s.cb, c) * k, e = ci(s.cd, c);
      col[c] = ((s.cop == 0 ? e + l : e - l) + bias(s.cbias)) * scale(s.cscale);
      if (s.cclamp != 0)
        col[c] = std::clamp(col[c], 0.f, 1.f);
    }
    const float k = ai(s.ac), l = ai(s.aa) * (1 - k) + ai(s.ab) * k, e = ai(s.ad);
    float al = ((s.aop == 0 ? e + l : e - l) + bias(s.abias)) * scale(s.ascale);
    if (s.aclamp != 0)
      al = std::clamp(al, 0.f, 1.f);
    for (int c = 0; c < 3; ++c)
      r[s.creg & 3][c] = col[c];
    r[s.areg & 3][3] = al;
  }
  std::memcpy(out, r[0], sizeof r[0]);
}

// What the rays read of a draw: small, the stages apart
struct GpuDraw {
  std::uint32_t stages, firstStage, cls;  // cls: 0 solid, 1 translucent, 2 additive, 3 ignored
  float strength;                         // reflection strength, at most Tuning::reflectCap
  std::uint32_t highlight;                // bit 0: has a stage looked up by the normal, the game's own shine; bit 1: glows, gives light; bit 2: outside the depth, drawn over the world, takes no light; bit 3: a blended layer writing no depth;
                                          // bit 4: a card turned to the picture (faces_picture, AURORA_RT_HUD_SKIP): the camera's rays alone meet it, it takes no light and casts no shadow
                                          // bit 5 (AURORA_RT_GLOW_WIDE): an additive sheet wider than a source (wide_glow): no hot core, no flare where it lies over the picture
                                          // bit 6 (AURORA_RT_SHEETS): a sheet, translucent with a soft alpha and no reflection layer (soft_alpha): the camera's rays pass it
                                          // bits 8-12: 1 + the machine whose draw it is, by the game's own mark (DrawRec::tag), 0 none: its lights' own body (rt.hlsl engine_light, glow_light)
  std::uint32_t fogType;
  float fog[3];
  float fogColour[3];
  float regs[4][4];
  std::uint32_t swap[4];  // one table a word, a byte a channel
};
// AURORA_RT_GLOW_WIDE: the glow (hot core, flare) is the character of small sources: a flame, a spark, a glow card. An additive draw with a card as
// wide as no such source is (a card: two triangles in a row, or a lone last one; its area over its length, so a long thin streak stays a source: at
// least size across, in world units) is a sheet of light (the pits' refill, a lit pad, a strip along the road): where it passes white it does so only
// over a bright surface, and takes no glow. Card by card, not the whole draw: a machine's flame is one draw of up to 13 small cards over each other,
// whose summed area over the draw's length reads as wide (measured 13.7 on Drift Highway's boost) though no card is.
// Measured on Drift Highway: flame cards 0.3-2.4 across, the boost's speed streaks 50 long and thin; the pit's cards 4-8. Lines stay sources.
inline float card_width(const Vertex* verts, std::uint32_t first, std::uint32_t count) {
  float area = 0.f, mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
  for (std::uint32_t v = 0; v + 2 < count; v += 3) {
    const float* p0 = verts[first + v].pos;
    const float* p1 = verts[first + v + 1].pos;
    const float* p2 = verts[first + v + 2].pos;
    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const float cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    area += 0.5f * std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
  }
  for (std::uint32_t v = 0; v < count; ++v)
    for (int a = 0; a < 3; ++a)
      mn[a] = std::min(mn[a], verts[first + v].pos[a]), mx[a] = std::max(mx[a], verts[first + v].pos[a]);
  const float extent = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
  return extent > 0.f ? area / extent : 0.f;
}
inline float widest_card(const DrawRec& d, const Vertex* verts) {
  float widest = 0.f;
  for (std::uint32_t v = 0; v + 2 < d.vertices; v += 6)
    widest = std::max(widest, card_width(verts, d.firstVertex + v, std::min<std::uint32_t>(6, d.vertices - v)));
  return widest;
}
inline bool wide_glow(const DrawRec& d, const Vertex* verts, const GpuDraw& g, float extent, float size) {
  constexpr std::uint32_t Lines = 0xA8, LineStrip = 0xB0;
  if (g.cls != 2 || extent < size || d.prim == Lines || d.prim == LineStrip)
    return false;
  return widest_card(d, verts) >= size;
}
// AURORA_RT_SHEETS: a sheet is a translucent draw (class 1) with no reflection layer whose alpha runs smoothly across it: a cloud layer, a waterfall.
// Its alpha's 0.5 line, where the camera's ray used to stop on it or pass it, cut such a sheet into sharp patches lit two ways. A cut-out (a grating,
// a mesh road, a fence, a machine's number) is solid or clear texel by texel: its 0.5 line is its own outline, and it stays as it was. Told from the
// draw's own data, the same on every course: some stage reads a soft alpha, from a texture (TEXA, through the stage's swap table) with at least
// SheetSoftShare of its texels partly clear, or from the vertices' colour (RASA) running over at least SheetVertexSpread across the draw; and no
// stage reads a cut-out's, a texture with at least SheetEdgeShare of its neighbouring texel pairs half the range apart (rt_capture.cpp s_texSoft,
// s_texEdge). Measured: the cloud sheets of Loop Cross and Half Pipe and the falls of Drift Highway 0.87 to 1.00 partly clear, no such pairs; the
// gratings and mesh roads 0.12 to 0.29 and 0.07 to 0.18; the machines' numbers 0.11 to 0.56 and 0.04. One texture is so for every draw that reads
// it, so a fall's parts with a vertex fade and those without are told alike.
constexpr float SheetSoftShare = 0.5f, SheetEdgeShare = 0.01f, SheetVertexSpread = 0.1f;
inline bool soft_alpha(const DrawRec& d, const Vertex* verts, const float (*texSoft)[4], const float (*texEdge)[4], std::uint32_t texCount) {
  const std::uint32_t stages = std::min<std::uint32_t>(d.stages, 16u);
  bool soft = false;
  for (std::uint32_t s = 0; s < stages; ++s) {
    const StageRec& st = d.stage[s];
    if (st.reflection != 0)
      continue;
    const auto reads = [&](std::uint32_t input) { return st.aa == input || st.ab == input || st.ac == input || st.ad == input; };
    if (reads(4) && st.tex < texCount) {
      const int ch = std::clamp(static_cast<int>(d.swap[st.swapTex & 3][3]), 0, 3);
      if (texEdge[st.tex][ch] >= SheetEdgeShare)
        return false;
      soft = soft || texSoft[st.tex][ch] >= SheetSoftShare;
    }
    if (!soft && reads(5) && d.vertices > 0) {
      float lo = 1.f, hi = 0.f;
      for (std::uint32_t v = 0; v < d.vertices; ++v)
        lo = std::min(lo, verts[d.firstVertex + v].ras[3]), hi = std::max(hi, verts[d.firstVertex + v].ras[3]);
      soft = hi - lo >= SheetVertexSpread;
    }
  }
  return soft;
}

struct Tuning {
  float reflectCap = 0.25f;  // the most reflection strength a draw takes (AURORA_RT_REFLECT_CAP; the look's)
  std::uint32_t maxLights = 16;  // unmarked lights (a machine's are all kept)
  float lightMaxSize = 40.f, lightMinBright = 0.07f, lightMerge = 8.f, lightFull = 2.f;
  // Glowing surfaces: on when texMeans is given. The size where small lights end is where these begin.
  const float (*texMeans)[4] = nullptr;
  std::uint32_t texCount = 0, maxGlows = 64, glowPerDraw = 0;  // glowPerDraw: most lights one draw may hold, 0 no limit
  float glowAddBright = 0.25f, glowBright = 0.75f, glowColourful = 0.4f;
  // Draws the stream left out (AURORA_RT_SCENE_STATIC=1) still count in what the frame's main view and largest size are
  std::uint32_t extraViews = 0;
  std::array<int, 2> extraKey[4]{};
  std::uint64_t extraVerts[4]{};
  float largestFloor = 0.f;
  bool overlayApart = false;  // AURORA_RT_HUD_SKIP=1: cards turned to the picture are the game's overlay in the world (faces_picture), not surfaces
  float glowWide = 0.f;       // AURORA_RT_GLOW_WIDE=1: an additive draw at least this wide across (world units, wide_glow) is a sheet of light, not a source: no hot core, no flare (bit 5)
  // AURORA_RT_SHEETS=1: each texture's soft and edge shares, by channel (rt_capture.cpp s_texSoft, s_texEdge); null: off (soft_alpha, bit 6)
  const float (*texSoft)[4] = nullptr;
  const float (*texEdge)[4] = nullptr;
  std::uint32_t texSoftCount = 0;
};
struct Frame {
  std::vector<float> prevPos;       // each of verts in the last frame's view space, with motion data on (rt_motion.hpp)
  std::vector<std::pair<std::uint32_t, std::uint32_t>> spans;  // the draw order's first vertex and count behind verts
  std::vector<Vertex> verts;        // solid surfaces first, then what light passes through (see-through, glows), then the sky
  std::uint32_t seeFirstVertex = 0, layerFirstVertex = 0, skyFirstVertex = 0;
  std::uint32_t primaryFirstVertex = 0;  // AURORA_RT_SCENE_STEADY: static placements' draws for primary rays alone, after the sky
  std::uint32_t passFirstVertex = 0;     // additive draws and draws that write no colour or darken (class 2, 3): last, a structure camera rays pass at once
  std::vector<GpuDraw> draws;
  std::vector<StageRec> stages;
  std::vector<LightPick> picks, merged;
  std::vector<EngineLight> lights;
  std::vector<GlowLight> glows;
  std::uint32_t glowDraws = 0, glowPieces = 0, glowTopDraw = 0;  // glowTopDraw: lights held by the draw holding most
  const DrawRec* main = nullptr;
  std::uint32_t machineDraws = 0, reflective = 0;  // machineDraws: draws the game marked as a machine's (DrawRec::tag)
  std::uint32_t burnerSolid = 0;  // of them, solid or see-through (not additive) with a burner's mark: drawn in a burner's span, not its flames
  std::uint32_t machineLights = 0, machineGlows = 0;  // lights and glow lights of a machine (owner)
  std::uint32_t machinesLit = 0;  // machines with a light (machineLights over it: lights a machine, its burners)
  std::vector<float> extent;  // each draw's size in the main view
  float largest = 0.f;
  bool is_sky(std::uint32_t i) const { return extent[i] > largest * 0.3f && draws[i].cls != 0 && draws[i].cls != 3; }
};

// The one rule for glowing surfaces, the big side of the one rule for lights: big (not a small light) in the main view, colour written,
// not the sky, and either additive and at all bright, or on the whole bright and colourful (a pale wall is bright but not colourful).
// No labels and no tables: lava, a neon strip and a lit pad are the same to it. A big triangle is cut into pieces fixed to it, smaller
// the nearer it is; pieces near one another become one; those that matter most from the view are kept.
template <class Sky>
inline void build_glows(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const Tuning& t, const std::vector<float>& extent, const Sky& sky, Frame& out) {
  out.glows.clear(), out.glowDraws = out.glowPieces = out.machineGlows = 0;
  if (t.texMeans == nullptr)
    return;
  struct Piece {
    float pos[3], area, colour[3], power;
    std::uint32_t draw, owner;
  };
  static thread_local std::vector<Piece> pieces;
  static thread_local std::vector<std::uint32_t> held;
  held.assign(count, 0);
  out.glowTopDraw = 0;
  pieces.clear();
  for (std::uint32_t i = 0; i < count; ++i) {
    const DrawRec& d = draws[i];
    GpuDraw& g = out.draws[i];
    if (extent[i] < t.lightMaxSize || !d.colorUpdate || g.cls == 3 || (g.highlight & 16) != 0 || sky(i))
      continue;
    float c[4];
    draw_colour(d, verts, t.texMeans, t.texCount, c);
    const float a = d.blendSrc == 4 ? std::clamp(c[3], 0.f, 1.f) : 1.f;
    const float most = std::max({c[0], c[1], c[2]}), least = std::min({c[0], c[1], c[2]});
    const bool glows = g.cls == 2 ? most * a > t.glowAddBright : most * a > t.glowBright && most > 0 && (most - least) / most > t.glowColourful;
    if (!glows)
      continue;
    g.highlight |= 2;
    ++out.glowDraws;
    const float col[3] = {std::clamp(c[0], 0.f, 1.5f) * a, std::clamp(c[1], 0.f, 1.5f) * a, std::clamp(c[2], 0.f, 1.5f) * a};
    const float lum = col[0] * 0.3f + col[1] * 0.59f + col[2] * 0.11f;
    for (std::uint32_t v = 0; v + 2 < d.vertices; v += 3) {
      const float* p0 = verts[d.firstVertex + v].pos;
      const float* p1 = verts[d.firstVertex + v + 1].pos;
      const float* p2 = verts[d.firstVertex + v + 2].pos;
      const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      const float cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
      const float area = 0.5f * std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
      if (area <= 0.f)
        continue;
      const float mid[3] = {(p0[0] + p1[0] + p2[0]) / 3, (p0[1] + p1[1] + p2[1]) / 3, (p0[2] + p1[2] + p2[2]) / 3};
      const float dist = std::sqrt(mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]);
      const float edge = std::max(std::hypot(e1[0], e1[1], e1[2]), std::hypot(e2[0], e2[1], e2[2]));
      const int n = std::clamp(static_cast<int>(edge / std::max(16.f, dist * 0.35f)) + 1, 1, 8);
      for (int a0 = 0; a0 < n; ++a0)
        for (int a1 = 0; a1 < n - a0; ++a1) {
          // a coarse grid of the triangle's barycentrics: fixed to it, so it travels with the world
          const float u = (a0 + 1.f / 3) / n, w = (a1 + 1.f / 3) / n;
          Piece q;
          q.draw = i, q.owner = tag_machine(d.tag);
          for (int k = 0; k < 3; ++k)
            q.pos[k] = p0[k] + e1[k] * u + e2[k] * w, q.colour[k] = col[k];
          q.area = area * 2 / static_cast<float>(n * n);  // pairs of small triangles of the grid, near enough
          const float dd = q.pos[0] * q.pos[0] + q.pos[1] * q.pos[1] + q.pos[2] * q.pos[2];
          q.power = lum * q.area / (3.14159f * dd + q.area + 1.f);
          pieces.push_back(q);
        }
    }
  }
  out.glowPieces = static_cast<std::uint32_t>(pieces.size());
  std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) { return a.power > b.power; });
  // greedy: the strongest first, each joining a light close by for its distance from the view
  static thread_local std::vector<Piece> sums;
  sums.clear();
  for (const Piece& p : pieces) {
    bool merged = false;
    const float dd = p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1] + p.pos[2] * p.pos[2];
    const float r2 = std::max(8.f * 8.f, dd * 0.09f);
    for (Piece& q : sums) {
      if (q.owner != p.owner)
        continue;
      const float x = q.pos[0] / q.area - p.pos[0], y = q.pos[1] / q.area - p.pos[1], z = q.pos[2] / q.area - p.pos[2];
      if (x * x + y * y + z * z < r2) {
        for (int k = 0; k < 3; ++k)
          q.pos[k] += p.pos[k] * p.area, q.colour[k] += p.colour[k] * p.area;
        q.area += p.area;
        merged = true;
        break;
      }
    }
    if (merged || sums.size() == t.maxGlows || (t.glowPerDraw != 0 && held[p.draw] >= t.glowPerDraw))
      continue;
    out.glowTopDraw = std::max(out.glowTopDraw, ++held[p.draw]);
    Piece q = p;
    for (int k = 0; k < 3; ++k)
      q.pos[k] *= p.area, q.colour[k] *= p.area;
    sums.push_back(q);
  }
  for (const Piece& q : sums) {
    GlowLight l;
    for (int k = 0; k < 3; ++k)
      l.pos[k] = q.pos[k] / q.area, l.colour[k] = q.colour[k] / q.area;
    l.area = q.area, l.radius = std::sqrt(q.area / 3.14159f), l.owner = q.owner;
    out.machineGlows += q.owner != 0 ? 1 : 0;
    out.glows.push_back(l);
  }
}

// A card turned to the picture: a draw that writes depth and every one of whose triangles lies in a plane parallel to the picture's. The game builds
// such a sprite in view space, so it faces the camera exactly, every frame, wherever the camera turns (the place markers over the machines). A
// surface of the world lies so only by chance, and never to this precision: a ten-thousandth of a radian. Cards that write no depth (dust, smoke,
// glows) are the blended layers' and additive draws' rules, not this one.
inline bool faces_picture(const DrawRec& d, const Vertex* verts) {
  static constexpr float MostTilt = 1e-4f;
  if (!d.depthUpdate || d.vertices < 3)
    return false;
  bool any = false;
  for (std::uint32_t v = 0; v + 2 < d.vertices; v += 3) {
    const float* p0 = verts[d.firstVertex + v].pos;
    const float* p1 = verts[d.firstVertex + v + 1].pos;
    const float* p2 = verts[d.firstVertex + v + 2].pos;
    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len <= 0.f)
      continue;
    if (std::fabs(n[0]) + std::fabs(n[1]) > MostTilt * len)
      return false;
    any = true;
  }
  return any;
}

// The main view is the perspective projection with the most triangles. Sky: the largest see-through shells around the view,
// by size against the largest in the frame, with no table of courses.
inline void build_frame(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const Tuning& t, Frame& out) {
  out.verts.clear(), out.prevPos.clear(), out.spans.clear(), out.draws.clear(), out.stages.clear(), out.main = nullptr, out.machineDraws = out.reflective = out.machineLights = out.machinesLit = out.burnerSolid = 0, out.seeFirstVertex = out.layerFirstVertex = out.skyFirstVertex = out.primaryFirstVertex = out.passFirstVertex = 0;
  const auto key = [](const DrawRec& d) { return std::array<int, 2>{static_cast<int>(d.proj[0] * 1000), static_cast<int>(d.proj[5] * 1000)}; };
  std::array<int, 2> best{};
  {
    // few distinct projections: a short list beats a map
    std::array<std::pair<std::array<int, 2>, std::uint64_t>, 16> seen{};
    std::uint32_t n = 0;
    for (std::uint32_t e = 0; e < t.extraViews && e < 4; ++e) {
      seen[n].first = t.extraKey[e], seen[n].second = t.extraVerts[e];
      ++n;
    }
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
  const auto inMain = [](const DrawRec& d) {
    return d.projType == 0 && std::array<int, 2>{static_cast<int>(d.proj[0] * 1000), static_cast<int>(d.proj[5] * 1000)} == s_best;
  };
  light_picks(draws, count, inMain, out.picks, t.lightMaxSize, t.lightMinBright);
  merge_lights(out.picks, t.lightMerge, out.merged);
  // with their strengths: colour by strength (a light, or a machine's together, at most 1.5), reach by the root of it, so a spark lights close by and a flame further.
  // Every machine's lights (its burners; at most 30 machines: 75 lights at the start of m33's 30), and the nearest maxLights unmarked ones.
  // (Build l kept the 16 nearest lights, each its 8-unit merge; with a light a burner 16 would leave 15% of the machines it lit dark.)
  std::sort(out.merged.begin(), out.merged.end(), [](const LightPick& a, const LightPick& b) {
    return a.pos[0] * a.pos[0] + a.pos[1] * a.pos[1] + a.pos[2] * a.pos[2] < b.pos[0] * b.pos[0] + b.pos[1] * b.pos[1] + b.pos[2] * b.pos[2];
  });
  out.lights.clear();
  // A machine's lights together are as strong as one light of all its flames (at most 1.5, as any light), shared among its burners by each
  // one's own flames (their weight); each reaches as that one light would
  std::array<float, 256> machineWeight{};
  for (const LightPick& q : out.merged)
    machineWeight[std::min(q.owner, 255u)] += q.owner != 0 ? q.weight : 0.f;
  std::uint64_t lit = 0;  // the machines lit (bit owner)
  std::uint32_t unmarked = 0;
  for (const LightPick& q : out.merged) {
    if (q.owner == 0) {
      if (unmarked == t.maxLights)
        continue;
      ++unmarked;
    }
    lit |= q.owner != 0 ? 1ull << std::min(q.owner, 63u) : 0;
    const float all = q.owner != 0 ? machineWeight[std::min(q.owner, 255u)] : q.weight;
    const float k = all / t.lightFull;
    const float strength = std::min(k, 1.5f) * q.weight / all;
    out.lights.push_back({{q.pos[0], q.pos[1], q.pos[2]}, {q.colour[0] * strength, q.colour[1] * strength, q.colour[2] * strength},
                          std::clamp(std::sqrt(k), 0.25f, 1.25f), q.owner});
    out.machineLights += q.owner != 0 ? 1 : 0;
  }
  for (std::uint64_t m = lit; m != 0; m &= m - 1)
    ++out.machinesLit;
  out.draws.resize(count);
  out.extent.assign(count, 0.f);
  std::vector<float>& extent = out.extent;
  float largest = t.largestFloor;
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
    if (!d.depthUpdate || !d.depthCompare)
      g.highlight |= 4;
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
    // a layer: blended over what is there by its alpha, writing no depth (dust, sand sheets, smoke cards). Rays may pass it (rt.hlsl AURORA_RT_LAYER_PASS)
    if (d.blendMode == 1 && !additive && !darkens && !opaqueBlend && !d.depthUpdate && g.cls < 2)
      g.highlight |= 8;
    const bool inView = d.projType == 0 && key(d) == best && d.vertices > 0;
    // the game's overlay in the world (AURORA_RT_HUD_SKIP): drawn as the game draws it; primary rays meet it (assemble_frame), so what lies behind
    // takes no light through it, and it takes none itself (bit 2); no other ray meets it: no shadow, no reflection, no light from it
    if (t.overlayApart && inView && g.cls < 2 && faces_picture(d, verts))
      g.highlight |= 16 | 4;
    // the machine whose draw this is, by the game's own mark: every light of that machine passes it (rt.hlsl engine_light, glow_light), its
    // reflections pass its own flames (trace_reflection)
    if (g.cls < 3 && (g.highlight & 16) == 0) {
      g.highlight |= std::min(tag_machine(d.tag), 31u) << 8;
      // bit 13: the mark is a body's (CarDisp's span): a surface of that machine (rt.hlsl body_machine)
      if (tag_machine(d.tag) != 0 && tag_kind(d.tag) == TagBody)
        g.highlight |= 1u << 13;
    }
    out.machineDraws += inView && tag_machine(d.tag) != 0 ? 1 : 0;
    out.burnerSolid += inView && g.cls < 2 && tag_machine(d.tag) != 0 && tag_kind(d.tag) == TagBurner ? 1 : 0;
    if (strength > 0 && inView) {
      ++out.reflective;
      strength = std::min(strength, t.reflectCap);
    }
    g.strength = strength;
    if (t.texSoft != nullptr && g.cls == 1 && strength <= 0.f && soft_alpha(d, verts, t.texSoft, t.texEdge, t.texSoftCount))
      g.highlight |= 64;
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
    if (t.glowWide > 0.f && wide_glow(d, verts, g, extent[i], t.glowWide))
      g.highlight |= 32;
  }
  out.largest = largest;
  const auto sky = [&](std::uint32_t i) { return out.is_sky(i); };
  build_glows(draws, count, verts, t, extent, sky, out);
}
// The triangles the rays get: solid, what light passes through, blended layers writing no depth (only with layersApart, AURORA_RT_LAYER_PASS; else they stay
// in the first two), sky, then what primary rays alone see. drop: a byte a draw, 1 where the static track already holds it, 2 where a static placement's
// draw is for primary rays alone (AURORA_RT_SCENE_STEADY; a layer or the sky stays in its own) (may be null). A card turned to the picture (highlight
// bit 4, AURORA_RT_HUD_SKIP) is for primary rays alone too.
// Last, the draws every camera ray passes: additive (engine fire, flames, glow cards) and those that write no colour or darken (class 2, 3). Camera rays
// stepped past them one look each, and a stack of flame cards used all of a ray's looks: the road under a machine's flames got no light of its own and the
// raster showed through as a patch. Their structure is marked non-opaque (rt_live.cpp) and camera rays cull non-opaque structures (rt.hlsl), so they pass
// all of them at no look at all; every other ray forces all to opaque and meets them as before (the see-through mask). A static placement's such draw kept
// for primary rays alone has nobody left to meet it and is left out.
inline void assemble_frame(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const std::uint8_t* drop, Frame& out, bool layersApart = false) {
  for (int pass = 0; pass < 6; ++pass) {
    if (pass == 1)
      out.seeFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    if (pass == 2)
      out.layerFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    if (pass == 3)
      out.skyFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    if (pass == 4)
      out.primaryFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    if (pass == 5)
      out.passFirstVertex = static_cast<std::uint32_t>(out.verts.size());
    for (std::uint32_t i = 0; i < count; ++i) {
      const bool passed = out.draws[i].cls >= 2;
      int group = out.is_sky(i) ? 3 : layersApart && (out.draws[i].highlight & 8) != 0 ? 2 : passed ? 5 : out.draws[i].cls == 0 ? 0 : 1;
      if (drop != nullptr && drop[i] == 2 && group == 5)
        continue;
      if ((drop != nullptr && drop[i] == 2 && group < 2) || ((out.draws[i].highlight & 16) != 0 && group < 3))
        group = 4;
      if (out.extent[i] <= 0.f || group != pass || (drop != nullptr && drop[i] == 1))
        continue;
      out.verts.insert(out.verts.end(), verts + draws[i].firstVertex, verts + draws[i].firstVertex + draws[i].vertices);
      out.spans.emplace_back(draws[i].firstVertex, draws[i].vertices);
    }
  }
}
} // namespace aurora::gx::rt
