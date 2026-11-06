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
  // How the raster picks the texture's level: the game's LOD bias and clamps, the sampler's anisotropy (rt.hlsl shade, asRaster)
  float lodBias, lodMin, lodMax, aniso;
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
  // What motion data knows a draw by, its own bytes: raw list and arrays (rt_motion.hpp); its one matrix, when it has just one
  std::uint64_t motionKey;
  float motionMtx[12];
  std::uint32_t motionSingle;
  std::uint64_t sceneKey;  // the draw's own bytes alone, the same as the model file's list (rt_scene.hpp); 0 unknown
  std::uint32_t samplesCopy;  // a stage reads a texture the frame copied from the picture
  std::uint64_t stateKey;     // what the draw's translation reads besides its bytes, as far as its own stages read it: stream's state_key
  float nrmMtx[9];            // its one normal matrix, when it has one
  std::uint32_t fading;       // its alpha is one material alpha below full, by the draw's own channel state: a whole list fading, however its bytes read
  std::uint32_t tag;          // the game's draw mark (aurora_present.h): bits 0-7 1 + the machine, 0 none; bits 8-15 the kind; 16-23 a burner's 1 + slot
  StageRec stage[16];
};
// A draw's mark, as the game's draw sites gave it (aurora_present.h AURORA_TAG_*)
constexpr std::uint32_t TagBody = 1, TagBurner = 2, TagStar = 3;  // TagStar: the burner's star (its emitter at +0xB8), with the burner's bits
inline std::uint32_t tag_machine(std::uint32_t tag) { return tag & 0xFFu; }  // 1 + the machine, 0 none
inline std::uint32_t tag_kind(std::uint32_t tag) { return (tag >> 8) & 0xFFu; }
inline std::uint32_t tag_burner(std::uint32_t tag) { return (tag >> 16) & 0xFFu; }  // which of the machine's burners (1 + its record's slot), 0 none
// A draw's lighting as the stream set it, kept beside the draw (AURORA_RT_LIGHTFACTS only)
struct LightNote {
  std::uint32_t enabled, vtxColour, ambVtx, matVtx, mask, diffFn, readsRas, readsReg;
  float amb[4], mat[4], pos[8][3], col[8][3];
};
} // namespace aurora::gx::rt
