#pragma once
// A captured GX frame as the ray-tracing rules read it: plain data, no Aurora or graphics API types.
#include <cstdint>

namespace aurora::gx::rt {
constexpr std::uint32_t MaxUv = 4;
struct Vertex {
  float pos[3];
  float nrm[3];
  float ras[4];  // colour channel 0 as the hardware computes it: lights in view space, material, ambient
  float uv[MaxUv][2];  // after the texture generators
  std::uint32_t draw;
};
struct StageRec {
  std::uint32_t tc, map, tex;  // map -1: none; tex: index of the captured texture, or -1
  std::uint32_t ca, cb, cc, cd, cop, cbias, cscale, cclamp, creg;
  std::uint32_t aa, ab, ac, ad, aop, abias, ascale, aclamp, areg;
  std::uint32_t kc, ka, swapTex;
  std::uint32_t reflection;  // 1 a reflection layer, 2 looked up by the normal but not a reflection (a highlight)
  float k[4];  // the stage's K colour and alpha selection resolved
};
struct DrawRec {
  float proj[16];
  float viewport[6];
  std::uint32_t projType, stages, firstVertex, vertices;
  std::uint32_t blendMode, blendSrc, blendDst, depthUpdate, depthCompare, cull, colorUpdate;
  std::uint32_t prim, vtxCount, hasNormal, hasColor, lit;
  float regs[4][4];
  float swap[4][4];  // for each swap table, the channel routed to red, green, blue, alpha
  std::uint32_t fogType;
  float fog[3];       // a, b, c
  float fogColour[4];
  // An additive draw's light, as the draw itself says: where (a line's first vertex, else the middle), how big, colour, brightness
  float lightPos[3], lightExtent, lightColour[3], lightBright;
  StageRec stage[16];
};
// A draw's lighting as the stream set it, kept beside the draw (AURORA_RT_LIGHTFACTS only)
struct LightNote {
  std::uint32_t enabled, vtxColour, ambVtx, matVtx, mask, diffFn, readsRas, readsReg;
  float amb[4], mat[4], pos[8][3], col[8][3];
};
} // namespace aurora::gx::rt
