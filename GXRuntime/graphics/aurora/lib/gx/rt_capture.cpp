#include "../gfx/stagelog.hpp"
#include <functional>
#include "rt_capture.hpp"
#include "rt_format.hpp"
#if defined(AURORA_RT_LIVE)
#include "rt_live.hpp"
#endif

#include "recomp.hpp"
#include "../gfx/texture_convert.hpp"
#include "../gfx/efb_readback.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <string>
#include <tuple>
#include <vector>

// The capture's file (read by rt-tool): all little-endian, floats 32-bit.
//   "RTC3", u32 counts: draws, vertices, textures; then the frame's XFB source rect (x, y, w, h as f32)
//   DrawRec[draws], Vertex[vertices] (three to a triangle), then each texture: u32 w, h, wrapS, wrapT, RGBA8 w*h*4
namespace aurora::gx::rt {
bool g_neutral = false;
bool g_gameLightOff = false;
bool g_noVertexColour = false;
bool g_gameLit[2]{};
float g_ambientWarmth = 0.f;
bool g_capturing = false;
bool g_noHud = false;
bool g_live = false;
bool g_hook = false;
static bool s_seenPerspective = false;
bool g_noGameShadow = false;
static bool s_motionOn = false;  // AURORA_RT_DENOISE=1: motion data from the draws (rt_motion.hpp)

bool skip_draw() noexcept {
  if (g_gxState.projType == GX_PERSPECTIVE) {
    s_seenPerspective = true;
    return false;
  }
  return s_seenPerspective;
}

namespace {
std::string s_path;
u64 s_at = 0;
u64 s_frame = 0;
u64 s_atCopy = 0;
u32 s_liveTextures = 0;
bool s_rayed = false;
double s_captureMs = 0;
float s_road = 0.9f, s_light = 1.f, s_glow = 0.f, s_calm = 1.f, s_wet = 0.f, s_machineCap = 0.45f, s_sceneryCap = 0.25f, s_machineGain = 2.f;
u32 s_liveFirstNew = 0;
u64 s_copyNow = 0;
std::vector<DrawRec> s_draws;
std::vector<Vertex> s_verts;
struct Tex {
  u32 w, h, wrapS, wrapT, index;
  std::vector<u8> rgba;
};
std::vector<Tex> s_texs;
std::map<std::tuple<const void*, u32, u32, u32, const void*>, u32> s_texIndex;

u32 rd_be(const u8* p, u32 n) {
  u32 v = 0;
  for (u32 i = 0; i < n; ++i)
    v = v << 8 | p[i];
  return v;
}
float rd_comp(const u8* p, GXCompType type, u32 frac, bool le) {
  const float scale = 1.f / static_cast<float>(1u << frac);
  u32 raw;
  switch (type) {
  case GX_U8:
    return static_cast<float>(p[0]) * scale;
  case GX_S8:
    return static_cast<float>(static_cast<s8>(p[0])) * scale;
  case GX_U16:
    raw = le ? (p[1] << 8 | p[0]) : rd_be(p, 2);
    return static_cast<float>(static_cast<u16>(raw)) * scale;
  case GX_S16:
    raw = le ? (p[1] << 8 | p[0]) : rd_be(p, 2);
    return static_cast<float>(static_cast<s16>(raw)) * scale;
  default: {
    raw = le ? (p[3] << 24 | p[2] << 16 | p[1] << 8 | p[0]) : rd_be(p, 4);
    float f;
    std::memcpy(&f, &raw, 4);
    return f;
  }
  }
}
void rd_color(const u8* p, GXCompType type, float out[4]) {
  const u32 v2 = rd_be(p, 2), v3 = rd_be(p, 3), v4 = rd_be(p, 4);
  switch (static_cast<u32>(type)) {
  case GX_RGB565:
    out[0] = ((v2 >> 11) & 31) / 31.f, out[1] = ((v2 >> 5) & 63) / 63.f, out[2] = (v2 & 31) / 31.f, out[3] = 1.f;
    break;
  case GX_RGB8:
  case GX_RGBX8:
    out[0] = p[0] / 255.f, out[1] = p[1] / 255.f, out[2] = p[2] / 255.f, out[3] = 1.f;
    break;
  case GX_RGBA4:
    out[0] = ((v2 >> 12) & 15) / 15.f, out[1] = ((v2 >> 8) & 15) / 15.f, out[2] = ((v2 >> 4) & 15) / 15.f,
    out[3] = (v2 & 15) / 15.f;
    break;
  case GX_RGBA6:
    out[0] = ((v3 >> 18) & 63) / 63.f, out[1] = ((v3 >> 12) & 63) / 63.f, out[2] = ((v3 >> 6) & 63) / 63.f,
    out[3] = (v3 & 63) / 63.f;
    break;
  default:
    out[0] = p[0] / 255.f, out[1] = p[1] / 255.f, out[2] = p[2] / 255.f, out[3] = p[3] / 255.f;
    (void)v4;
    break;
  }
}
void k_select(u32 sel, const std::array<Vec4<float>, GX_MAX_KCOLOR>& k, float* out, bool alpha) {
  if (sel <= 7) {  // 8/8 .. 1/8
    const float v = (8 - sel) / 8.f;
    out[0] = out[1] = out[2] = out[3] = v;
    return;
  }
  if (!alpha && sel >= GX_TEV_KCSEL_K0 && sel <= GX_TEV_KCSEL_K0 + 3) {
    const auto& c = k[sel - GX_TEV_KCSEL_K0];
    out[0] = c[0], out[1] = c[1], out[2] = c[2];
    return;
  }
  if (sel >= 0x10) {
    const auto& c = k[(sel - 0x10) & 3];
    const float v = c[(sel - 0x10) >> 2];
    out[0] = out[1] = out[2] = out[3] = v;
  }
}
// What one stage's texture needs after the stream has moved on
struct TexSnap {
  GXTexObj_ obj;
  GXTlutObj_ tlut;
  bool palette, tlutOk;
};
constexpr u32 MostTexAvg = 8192;
float s_texAvg[MostTexAvg][4];  // a texture's mean colour, for a draw's brightness; written before its index is known to anyone
using TexKey = std::tuple<const void*, u32, u32, u32, const void*>;
std::mutex s_texLock;
u32 add_texture(const TexSnap& snap) {
  auto obj = snap.obj;
  if (!recomp::resolve_texture_data(obj) || obj.data == nullptr)
    return UINT32_MAX;
  const u32 w = obj.width(), h = obj.height(), fmt = obj.format();
  const GXTlutObj_& tlut = snap.tlut;
  const bool palette = snap.palette;
  if (palette && !snap.tlutOk)
    return UINT32_MAX;
  const TexKey key = std::make_tuple(obj.data, w, h, fmt, palette ? tlut.data : nullptr);
  // indices never change: each worker keeps its own memory of them and takes the lock only for one it has not seen
  static thread_local std::map<TexKey, u32> s_seen;
  if (const auto it = s_seen.find(key); it != s_seen.end())
    return it->second;
  std::lock_guard lock{s_texLock};
  if (const auto it = s_texIndex.find(key); it != s_texIndex.end())
    return s_seen[key] = it->second;
  if (g_live && s_liveTextures >= 8000)
    return UINT32_MAX;
  const size_t bytes = GXGetTexBufferSize(static_cast<u16>(w), static_cast<u16>(h), fmt, GX_FALSE, 0);
  gfx::ConvertedTexture conv;
  if (palette) {
    const size_t tlutBytes = tlut.numEntries * 2u;
    conv = gfx::convert_texture_palette(fmt, w, h, 1, {static_cast<const u8*>(obj.data), bytes}, tlut.format, tlut.numEntries,
                                        {static_cast<const u8*>(tlut.data), tlutBytes});
  } else {
    conv = gfx::convert_texture(fmt, w, h, 1, {static_cast<const u8*>(obj.data), bytes});
  }
  if (conv.data.size() < static_cast<size_t>(w) * h * 4u) {
    std::fprintf(stderr, "[rt] texture %ux%u format %u not decoded\n", w, h, fmt);
    s_texIndex[key] = UINT32_MAX;
    return UINT32_MAX;
  }
  // live: an index for good, the pixels leave with the job that first needs them
  const u32 index = g_live ? s_liveTextures++ : static_cast<u32>(s_texs.size());
  Tex t{w, h, static_cast<u32>(obj.wrap_s()), static_cast<u32>(obj.wrap_t()), index, {}};
  t.rgba.assign(conv.data.data(), conv.data.data() + static_cast<size_t>(w) * h * 4u);
  if (index < MostTexAvg) {
    u64 sum[4]{};
    for (size_t i = 0; i < t.rgba.size(); i += 4)
      for (int c = 0; c < 4; ++c)
        sum[c] += t.rgba[i + c];
    const double n = 255.0 * w * h;
    for (int c = 0; c < 4; ++c)
      s_texAvg[index][c] = static_cast<float>(sum[c] / n);
  }
  s_texs.push_back(std::move(t));
  s_texIndex[key] = index;
  return index;
}
float dot4(const Vec4<float>& m, float x, float y, float z, float w) { return m[0] * x + m[1] * y + m[2] * z + m[3] * w; }
} // namespace

void start() noexcept {
  const char* on = std::getenv("AURORA_RT");
  // 1: the live pass; 2: only the game's fake reflections off, for the still tool's pictures
  g_neutral = on != nullptr && (on[0] == '1' || on[0] == '2');
#if defined(AURORA_RT_LIVE)
  g_live = on != nullptr && on[0] == '1';
#endif
  if (const char* path = std::getenv("AURORA_RT_CAPTURE"))
    s_path = path;
  if (const char* at = std::getenv("AURORA_RT_CAPTURE_AT"))
    s_at = std::strtoull(at, nullptr, 10);
  if (const char* v = std::getenv("AURORA_RT_GAME_LIGHT"); v != nullptr && v[0] == '0')
    g_gameLightOff = true;
  if (const char* v = std::getenv("AURORA_RT_AMBIENT_WARMTH"))
    g_ambientWarmth = static_cast<float>(std::atof(v));
  if (const char* v = std::getenv("AURORA_RT_DENOISE"); v != nullptr && v[0] == '1')
    s_motionOn = true;
  // with our own light: the rays' shadows stand in for the game's. AURORA_RT_GAME_SHADOW=1 keeps the game's
  if (g_gameLightOff && std::getenv("AURORA_RT_LIGHTING") != nullptr) {
    const char* v = std::getenv("AURORA_RT_GAME_SHADOW");
    g_noGameShadow = v == nullptr || v[0] != '1';
  }
  // with our own light: the star the game puts on far machines' engines, to keep them readable, looks out of place. Off by default
  if (g_live && std::getenv("AURORA_RT_LIGHTING") != nullptr) {
    const char* v = std::getenv("AURORA_RT_NO_ENGINE_STAR");
    g_noEngineStar = v != nullptr && v[0] == '1';
  }
  if (std::getenv("AURORA_RT_MOTION_ONLY") != nullptr)  // the cost of the motion data alone
    s_motionOn = true;
  if (const char* v = std::getenv("AURORA_RT_VERTEX_COLOUR"); v != nullptr && v[0] == '0')
    g_noVertexColour = true;
  const char* noHud = std::getenv("AURORA_RT_NO_HUD");
  g_noHud = noHud != nullptr && noHud[0] == '1';
  if (const char* at = std::getenv("AURORA_RT_CAPTURE_COPY"))
    s_atCopy = std::strtoull(at, nullptr, 10);
  // the live pass captures from the first frame after its compile is done, see frame_end
  g_capturing = !g_live && !s_path.empty() && s_at == 0 && s_atCopy == 0;
#if defined(AURORA_RT_LIVE)
  g_hook = g_live || live::look_on();
  if (g_hook)
    live::start_compile();
#endif
  // AURORA_RT_LOOK=1, the overhaul look: reflections you can't miss. Each number can still be set by its own switch below. All tuned by eye.
  if (const char* v = std::getenv("AURORA_RT_LOOK"); v != nullptr && v[0] == '1') {
    s_road = 1.4f, s_glow = 0.5f, s_calm = 0.3f, s_wet = 0.35f, s_machineCap = 0.9f, s_sceneryCap = 0.7f, s_machineGain = 3.f;
  }
  // AURORA_RT_LOOK=2, the material rule: the game's own reflection layers stay in the raster, the rays only add (rt.hlsl, MatGapLow and on). The picture stays the rays-off picture, better lit.
  if (const char* v = std::getenv("AURORA_RT_LOOK"); v != nullptr && v[0] == '2') {
    g_neutral = false;
    s_road = 0.f, s_glow = 0.5f, s_wet = -1.f, s_machineCap = 1.f, s_sceneryCap = 1.f, s_machineGain = 1.f;
  }
  const auto setting = [](const char* name, float& to) {
    if (const char* v = std::getenv(name))
      to = static_cast<float>(std::atof(v));
  };
  setting("AURORA_RT_BODY_CALM", s_calm), setting("AURORA_RT_WET", s_wet), setting("AURORA_RT_MACHINE_CAP", s_machineCap), setting("AURORA_RT_SCENERY_CAP", s_sceneryCap), setting("AURORA_RT_MACHINE_GAIN", s_machineGain);
  if (const char* v = std::getenv("AURORA_RT_ROAD"))
    s_road = static_cast<float>(std::atof(v));
  if (const char* v = std::getenv("AURORA_RT_LIGHT"))
    s_light = static_cast<float>(std::atof(v));
  if (const char* v = std::getenv("AURORA_RT_GLOW"))
    s_glow = static_cast<float>(std::atof(v));
  if (g_neutral || !s_path.empty())
    std::fprintf(stderr, "[rt] reflection stages %s; capture %s at display copy %llu\n", g_neutral ? "off" : "on",
                 s_path.c_str(), static_cast<unsigned long long>(s_at));
}

bool is_reflection_stage(const TevStage& stage, const std::array<TcgConfig, MaxTexCoord>& tcgs) noexcept {
  if (stage.texCoordId == GX_TEXCOORD_NULL || stage.texMapId == GX_TEXMAP_NULL || stage.texCoordId >= MaxTexCoord)
    return false;
  const auto& tcg = tcgs[stage.texCoordId];
  if (tcg.src != GX_TG_NRM || (tcg.type != GX_TG_MTX2x4 && tcg.type != GX_TG_MTX3x4))
    return false;
  if (stage.colorOp.outReg != GX_TEVPREV)
    return false;
  const auto& c = stage.colorPass;
  const bool adds = c.d == GX_CC_CPREV && c.a == GX_CC_ZERO && c.b == GX_CC_TEXC &&
                    (c.c == GX_CC_KONST || c.c == GX_CC_ONE || c.c == GX_CC_RASC);
  const bool multiplies = c.a == GX_CC_ZERO && c.d == GX_CC_ZERO && ((c.b == GX_CC_TEXC && c.c == GX_CC_CPREV) || (c.b == GX_CC_CPREV && c.c == GX_CC_TEXC));
  return adds || multiplies;
}

void neutralise(ShaderConfig& config) noexcept {
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    auto& s = config.tevStages[i];
    if (!is_reflection_stage(s, config.tcgs))
      continue;
    s.colorPass.a = s.colorPass.b = s.colorPass.c = GX_CC_ZERO;
    s.colorPass.d = GX_CC_CPREV;
    s.colorOp.op = GX_TEV_ADD;
    s.colorOp.bias = GX_TB_ZERO;
    s.colorOp.scale = GX_CS_SCALE_1;
    // the alpha part reads nothing of the texture in these layers (it passes through), so the sampling can go
    if (s.alphaPass.a != GX_CA_TEXA && s.alphaPass.b != GX_CA_TEXA && s.alphaPass.c != GX_CA_TEXA && s.alphaPass.d != GX_CA_TEXA) {
      s.texCoordId = GX_TEXCOORD_NULL;
      s.texMapId = GX_TEXMAP_NULL;
    }
  }
}

// All a draw's translation reads, copied where the stream stands: the same names as GXState, the parts only this draw uses
struct Packet {
  u32 size;        // bytes to the next packet
  u32 epoch;       // live: draws before the last EFB clear went to a texture, not the picture
  u32 drawSlot, vertSlot, drawIndex, firstVertex;
  GXPrimitive prim;
  u16 vtxCount;
  u32 samplesCopy;
  std::uint64_t stateKey;
  u32 rawOffset, rawBytes, stagesOffset, mtxOffset, mtxCount, lightsOffset;
  decltype(GXState::proj) proj;
  decltype(GXState::logicalViewport) logicalViewport;
  decltype(GXState::projType) projType;
  decltype(GXState::numTevStages) numTevStages;
  decltype(GXState::blendMode) blendMode;
  decltype(GXState::blendFacSrc) blendFacSrc;
  decltype(GXState::blendFacDst) blendFacDst;
  decltype(GXState::depthUpdate) depthUpdate;
  decltype(GXState::depthCompare) depthCompare;
  decltype(GXState::cullMode) cullMode;
  decltype(GXState::colorUpdate) colorUpdate;
  decltype(GXState::fog) fog;
  decltype(GXState::colorRegs) colorRegs;
  decltype(GXState::tevSwapTable) tevSwapTable;
  decltype(GXState::kcolors) kcolors;
  decltype(GXState::tcgs) tcgs;
  decltype(GXState::colorChannelConfig) colorChannelConfig;
  decltype(GXState::colorChannelState) colorChannelState;
  decltype(GXState::vtxDesc) vtxDesc;
  decltype(GXState::arrays) arrays;
  decltype(GXState::numTexGens) numTexGens;
  VtxFmt vf;
  Mat3x4<float> texMtx[MaxUv], ptTexMtx[MaxUv];
  u32 currentPnMtx;
};

u32 triangle_vertices(GXPrimitive prim, u32 n) {
  switch (prim) {
  case GX_TRIANGLES:
    return n / 3 * 3;
  case GX_TRIANGLESTRIP:
  case GX_TRIANGLEFAN:
    return (n - 2) * 3;
  case GX_QUADS:
    return n / 4 * 6;
  default:
    return 0;
  }
}

// AURORA_RT_SCENE_FADE=0: a fading draw counts against its list as before, to compare
static const bool s_fadeAsDraw = [] {
  const char* v = std::getenv("AURORA_RT_SCENE_FADE");
  return v == nullptr || v[0] != '0';
}();
// What the translation of a proved list reads besides its bytes, read only as the draw's own stages read it: registers no stage reads go stale without the surface changing.
// Texture matrices, the texture by its data, the colours a stage or channel takes from a register. No vertex data, no per-frame table: some dozens of words a draw, and most draws have no matrix.
static std::uint64_t state_key() noexcept {
  const auto& st = g_gxState;
  std::uint64_t h = 0xcbf29ce484222325ull;
  const auto mix = [&h](std::uint64_t w) {
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  };
  const auto words = [&mix](const void* p, size_t bytes) {
    const auto* u = static_cast<const std::uint32_t*>(p);
    for (size_t i = 0; i < bytes / 4; ++i)
      mix(u[i]);
  };
  const u32 gens = std::min<u32>(st.numTexGens, MaxUv);
  mix(gens);
  for (u32 t = 0; t < gens; ++t) {
    const auto& g = st.tcgs[t];
    if (g.mtx != GX_IDENTITY && g.mtx >= GX_TEXMTX0 && (g.mtx - GX_TEXMTX0) / 3 < MaxTexMtx && g.src != GX_TG_NRM)
      words(&st.texMtxs[(g.mtx - GX_TEXMTX0) / 3], sizeof(st.texMtxs[0]));
    if (g.postMtx != GX_PTIDENTITY && g.postMtx >= GX_PTTEXMTX0 && (g.postMtx - GX_PTTEXMTX0) / 3 < MaxPTTexMtx && g.src != GX_TG_NRM)
      words(&st.ptTexMtxs[(g.postMtx - GX_PTTEXMTX0) / 3], sizeof(st.ptTexMtxs[0]));
  }
  const u32 stages = std::min<u32>(st.numTevStages, 16u);
  bool reads[4]{}, readsRas = false;
  u32 chan = 0;
  for (u32 i = 0; i < stages; ++i) {
    const TevStage& s = st.tevStages[i];
    readsRas = readsRas || s.colorPass.a == GX_CC_RASC || s.colorPass.b == GX_CC_RASC || s.colorPass.c == GX_CC_RASC || s.colorPass.d == GX_CC_RASC || s.colorPass.a == GX_CC_RASA || s.colorPass.b == GX_CC_RASA || s.colorPass.c == GX_CC_RASA || s.colorPass.d == GX_CC_RASA;
    if (s.channelId == GX_COLOR1A1 || s.channelId == GX_COLOR1 || s.channelId == GX_ALPHA1)
      chan = 1;
    // a reflection layer's own fields, as material_of leaves them out
    if (is_reflection_stage(s, st.tcgs))
      continue;
    if (s.texMapId != GX_TEXMAP_NULL && s.texMapId < MaxTextures)
      mix(reinterpret_cast<std::uintptr_t>(st.textures[s.texMapId].texObj.data));
    for (const auto a : {s.colorPass.a, s.colorPass.b, s.colorPass.c, s.colorPass.d})
      if (a < GX_CC_TEXC)
        reads[a / 2] = true;
    for (const auto a : {s.alphaPass.a, s.alphaPass.b, s.alphaPass.c, s.alphaPass.d})
      if (a < GX_CA_TEXA)
        reads[a] = true;
    const auto& pc = s.colorPass;
    if (pc.a == GX_CC_KONST || pc.b == GX_CC_KONST || pc.c == GX_CC_KONST || pc.d == GX_CC_KONST) {
      mix(s.kcSel);
      if (s.kcSel >= GX_TEV_KCSEL_K0)
        words(&st.kcolors[(s.kcSel - GX_TEV_KCSEL_K0) & 3], 12);
    }
  }
  for (int r = 0; r < 4; ++r)
    if (reads[r])
      words(&st.colorRegs[r], 12);
  const auto& cc = st.colorChannelConfig[GX_COLOR0 + chan];
  if (readsRas && cc.matSrc != GX_SRC_VTX)
    words(&st.colorChannelState[GX_COLOR0 + chan].matColor, 12);
  return h | 1;
}
// Anywhere, in any order: the packet says where its draw and triangles go
void translate(const Packet& st, DrawRec& d, Vertex* outVerts) noexcept {
  const auto* base = reinterpret_cast<const u8*>(&st);
  const auto* tevStages = reinterpret_cast<const TevStage*>(base + st.stagesOffset);
  const auto* texSnaps = reinterpret_cast<const TexSnap*>(tevStages + std::min<u32>(st.numTevStages, 16u));
  const auto* pnMtx = reinterpret_cast<const PnMtx*>(base + st.mtxOffset);
  const auto* lights = reinterpret_cast<const Light*>(base + st.lightsOffset);
  const u8* vertices = base + st.rawOffset;
  const GXPrimitive prim = st.prim;
  const u16 vtxCount = st.vtxCount;
  const u32 drawIndex = st.drawIndex;
  std::memset(static_cast<void*>(&d), 0, g_live ? offsetof(DrawRec, stage) : sizeof(DrawRec));
  std::memcpy(d.proj, &st.proj, sizeof d.proj);
  const auto& vp = st.logicalViewport;
  d.viewport[0] = vp.left, d.viewport[1] = vp.top, d.viewport[2] = vp.width, d.viewport[3] = vp.height,
  d.viewport[4] = vp.znear, d.viewport[5] = vp.zfar;
  d.projType = st.projType;
  d.stages = st.numTevStages;
  d.samplesCopy = st.samplesCopy;
  d.stateKey = st.stateKey;
  d.blendMode = st.blendMode, d.blendSrc = st.blendFacSrc, d.blendDst = st.blendFacDst;
  d.depthUpdate = st.depthUpdate, d.depthCompare = st.depthCompare, d.cull = st.cullMode, d.colorUpdate = st.colorUpdate;
  d.prim = prim, d.vtxCount = vtxCount;
  d.fogType = st.fog.type, d.fog[0] = st.fog.a, d.fog[1] = st.fog.b, d.fog[2] = st.fog.c;
  for (int i = 0; i < 4; ++i)
    d.fogColour[i] = st.fog.color[i];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      d.regs[i][j] = st.colorRegs[i][j];
  for (int i = 0; i < 4; ++i) {
    d.swap[i][0] = st.tevSwapTable[i].red, d.swap[i][1] = st.tevSwapTable[i].green, d.swap[i][2] = st.tevSwapTable[i].blue,
    d.swap[i][3] = st.tevSwapTable[i].alpha;
  }
  for (u32 i = 0; i < d.stages && i < 16; ++i) {
    const auto& g = tevStages[i];
    auto& s = d.stage[i];
    s.tc = g.texCoordId, s.map = g.texMapId;
    s.tex = g.texMapId != GX_TEXMAP_NULL ? add_texture(texSnaps[i]) : UINT32_MAX;
    s.ca = g.colorPass.a, s.cb = g.colorPass.b, s.cc = g.colorPass.c, s.cd = g.colorPass.d;
    s.cop = g.colorOp.op, s.cbias = g.colorOp.bias, s.cscale = g.colorOp.scale, s.cclamp = g.colorOp.clamp, s.creg = g.colorOp.outReg;
    s.aa = g.alphaPass.a, s.ab = g.alphaPass.b, s.ac = g.alphaPass.c, s.ad = g.alphaPass.d;
    s.aop = g.alphaOp.op, s.abias = g.alphaOp.bias, s.ascale = g.alphaOp.scale, s.aclamp = g.alphaOp.clamp, s.areg = g.alphaOp.outReg;
    s.kc = g.kcSel, s.ka = g.kaSel, s.swapTex = g.tevSwapTex;
    s.reflection = is_reflection_stage(g, st.tcgs) ? 1 : 0;
    if (s.reflection == 0 && g.texCoordId < MaxTexCoord && st.tcgs[g.texCoordId].src == GX_TG_NRM)
      s.reflection = 2;  // looked up by the normal, not a reflection layer (the world-specular highlight)
    k_select(g.kcSel, st.kcolors, s.k, false);
    float ka[4]{};
    k_select(g.kaSel, st.kcolors, ka, true);
    s.k[3] = ka[0];
  }

  // The layout of one vertex in the stream
  const auto& vf = st.vf;
  struct Field {
    GXAttrType type = GX_NONE;
    u32 offset = 0, size = 0;
  };
  std::array<Field, GX_VA_TEX7 + 1> fields{};
  u32 stride = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = st.vtxDesc[i];
    if (type == GX_NONE)
      continue;
    fields[i].type = type;
    fields[i].offset = stride;
    u32 size;
    if (type == GX_DIRECT) {
      size = i <= GX_VA_TEX7MTXIDX ? 1 : comp_type_size(attr, vf.attrs[i].type) * comp_cnt_count(attr, vf.attrs[i].cnt);
    } else {
      size = type == GX_INDEX8 ? 1 : 2;
      if (i == GX_VA_NRM && vf.attrs[i].cnt == GX_NRM_NBT3)
        size *= 3;
    }
    fields[i].size = size;
    stride += size;
  }
  const auto source = [&](int attr, const u8* v) -> const u8* {
    const auto& f = fields[attr];
    const u8* p = v + f.offset;
    if (f.type == GX_DIRECT)
      return p;
    const u32 index = f.type == GX_INDEX8 ? p[0] : rd_be(p, 2);
    const auto& a = st.arrays[attr];
    if (a.data == nullptr || index * a.stride >= a.size)
      return nullptr;
    return static_cast<const u8*>(a.data) + index * a.stride;
  };
  d.hasNormal = fields[GX_VA_NRM].type != GX_NONE;
  d.hasColor = fields[GX_VA_CLR0].type != GX_NONE;
  // one rasterised colour to a vertex: the channel the draw's stages read
  u32 chan = 0;
  for (u32 i = 0; i < d.stages && i < 16; ++i)
    if (tevStages[i].channelId == GX_COLOR1A1 || tevStages[i].channelId == GX_COLOR1 || tevStages[i].channelId == GX_ALPHA1) {
      chan = 1;
      break;
    }
  if (s_motionOn || scene::on()) {
    // the draw's own bytes and where its arrays are: the same list of the same model next frame has the same key
    std::uint64_t h = 0;
    const size_t used = std::min<size_t>(st.rawBytes, size_t(stride) * vtxCount);
    d.sceneKey = fields[GX_VA_POS].type == GX_DIRECT ? scene::list_key(prim, vtxCount, vertices, used, &h) : (scene::list_key(prim, vtxCount, vertices, used, &h), 0);
    const auto mix = [&h](std::uint64_t w) {
      h = (h ^ w) * 0x9E3779B97F4A7C15ull;
      h ^= h >> 29;
    };
    mix(reinterpret_cast<std::uintptr_t>(st.arrays[GX_VA_POS].data));
    d.motionKey = h | 1;
    d.motionSingle = st.mtxCount == 1 ? 1 : 0;
    if (st.mtxCount == 1) {
      const auto& m = pnMtx[0].pos;
      const Vec4<float>* rows[3] = {&m.m0, &m.m1, &m.m2};
      for (int r = 0; r < 3; ++r)
        d.motionMtx[r * 4] = rows[r]->x(), d.motionMtx[r * 4 + 1] = rows[r]->y(), d.motionMtx[r * 4 + 2] = rows[r]->z(), d.motionMtx[r * 4 + 3] = rows[r]->w();
      const auto& n = pnMtx[0].nrm;
      const Vec4<float>* nrows[3] = {&n.m0, &n.m1, &n.m2};
      for (int r = 0; r < 3; ++r)
        d.nrmMtx[r * 3] = nrows[r]->x(), d.nrmMtx[r * 3 + 1] = nrows[r]->y(), d.nrmMtx[r * 3 + 2] = nrows[r]->z();
    }
  }
  d.lit = st.colorChannelConfig[GX_COLOR0 + chan].lightingEnabled;
  {
    const auto& cc = st.colorChannelConfig[GX_ALPHA0 + chan];
    d.fading = s_fadeAsDraw && cc.matSrc != GX_SRC_VTX && !cc.lightingEnabled && st.colorChannelState[GX_ALPHA0 + chan].matColor[3] < 0.99f ? 1 : 0;
  }

  const auto read_vertex = [&](const u8* v) {
    Vertex out{};
    out.draw = drawIndex;
    const u32 mi = fields[GX_VA_PNMTXIDX].type != GX_NONE ? v[fields[GX_VA_PNMTXIDX].offset] / 3u : st.currentPnMtx;
    const auto& mtx = st.mtxCount == 1 ? pnMtx[0] : pnMtx[std::min<u32>(mi, MaxPnMtx - 1)];
    float p[3]{};
    if (const u8* s = source(GX_VA_POS, v)) {
      const auto& a = vf.attrs[GX_VA_POS];
      const u32 sz = comp_type_size(GX_VA_POS, a.type);
      const bool le = fields[GX_VA_POS].type != GX_DIRECT && st.arrays[GX_VA_POS].le;
      const u32 n = comp_cnt_count(GX_VA_POS, a.cnt);
      for (u32 c = 0; c < n && c < 3; ++c)
        p[c] = rd_comp(s + c * sz, a.type, a.frac, le);
    }
    out.pos[0] = dot4(mtx.pos.m0, p[0], p[1], p[2], 1.f);
    out.pos[1] = dot4(mtx.pos.m1, p[0], p[1], p[2], 1.f);
    out.pos[2] = dot4(mtx.pos.m2, p[0], p[1], p[2], 1.f);
    float n[3]{0.f, 0.f, 1.f};
    if (d.hasNormal) {
      if (const u8* s = source(GX_VA_NRM, v)) {
        const auto& a = vf.attrs[GX_VA_NRM];
        const u32 sz = comp_type_size(GX_VA_NRM, a.type);
        const bool le = fields[GX_VA_NRM].type != GX_DIRECT && st.arrays[GX_VA_NRM].le;
        const u32 frac = a.type == GX_S8 ? 6 : a.type == GX_S16 ? 14 : 0;
        for (u32 c = 0; c < 3; ++c)
          n[c] = rd_comp(s + c * sz, a.type, frac, le);
      }
      const float x = dot4(mtx.nrm.m0, n[0], n[1], n[2], 0.f), y = dot4(mtx.nrm.m1, n[0], n[1], n[2], 0.f),
                  z = dot4(mtx.nrm.m2, n[0], n[1], n[2], 0.f);
      const float len = std::sqrt(x * x + y * y + z * z);
      out.nrm[0] = len > 0 ? x / len : 0, out.nrm[1] = len > 0 ? y / len : 0, out.nrm[2] = len > 0 ? z / len : 1;
    }
    // The channel the stages read, computed as the hardware does: colour and alpha each by their own control
    float vc[4]{1.f, 1.f, 1.f, 1.f};
    const int clrAttr = GX_VA_CLR0 + chan;
    if (fields[clrAttr].type != GX_NONE)
      if (const u8* s = source(clrAttr, v))
        rd_color(s, vf.attrs[clrAttr].type, vc);
    if (g_noVertexColour)
      vc[0] = vc[1] = vc[2] = 1.f;
    const auto lit = [&](u32 ctrl, int first, int last) {
      const auto& cc = st.colorChannelConfig[ctrl];
      const auto& cs = st.colorChannelState[ctrl];
      float sum[4]{};
      for (int c = first; c < last; ++c)
        sum[c] = cc.ambSrc == GX_SRC_VTX ? vc[c] : cs.ambColor[c];
      if (cc.lightingEnabled) {
        for (u32 l = 0; l < GX::MaxLights; ++l) {
          if (!cs.lightMask.test(l))
            continue;
          const auto& light = lights[l];
          float lx = light.pos[0] - out.pos[0], ly = light.pos[1] - out.pos[1], lz = light.pos[2] - out.pos[2];
          const float ll = std::sqrt(lx * lx + ly * ly + lz * lz);
          if (ll > 0)
            lx /= ll, ly /= ll, lz /= ll;
          float ndl = lx * out.nrm[0] + ly * out.nrm[1] + lz * out.nrm[2];
          ndl = cc.diffFn == GX_DF_NONE ? 1.f : cc.diffFn == GX_DF_SIGN ? ndl : std::max(ndl, 0.f);
          for (int c = first; c < last; ++c)
            sum[c] += light.color[c] * ndl;
        }
      }
      for (int c = first; c < last; ++c) {
        const float mat = cc.matSrc == GX_SRC_VTX ? vc[c] : cs.matColor[c];
        out.ras[c] = cc.lightingEnabled ? mat * std::clamp(sum[c], 0.f, 1.f) : mat;
      }
    };
    lit(GX_COLOR0 + chan, 0, 3);
    lit(GX_ALPHA0 + chan, 3, 4);
    for (u32 t = 0; t < MaxUv && t < st.numTexGens; ++t) {
      const auto& g = st.tcgs[t];
      float in[4] = {0.f, 0.f, 1.f, 1.f};
      if (g.src >= GX_TG_TEX0 && g.src <= GX_TG_TEX7) {
        const int attr = GX_VA_TEX0 + (g.src - GX_TG_TEX0);
        if (fields[attr].type != GX_NONE)
          if (const u8* s = source(attr, v)) {
            const auto& a = vf.attrs[attr];
            const u32 sz = comp_type_size(static_cast<GXAttr>(attr), a.type);
            const bool le = fields[attr].type != GX_DIRECT && st.arrays[attr].le;
            in[0] = rd_comp(s, a.type, a.frac, le);
            if (comp_cnt_count(static_cast<GXAttr>(attr), a.cnt) > 1)
              in[1] = rd_comp(s + sz, a.type, a.frac, le);
          }
      } else if (g.src == GX_TG_POS) {
        in[0] = p[0], in[1] = p[1], in[2] = p[2];
      } else if (g.src == GX_TG_NRM) {
        in[0] = n[0], in[1] = n[1], in[2] = n[2];
      }
      float o[3] = {in[0], in[1], in[2]};
      if (g.mtx != GX_IDENTITY && g.mtx >= GX_TEXMTX0 && (g.mtx - GX_TEXMTX0) / 3 < MaxTexMtx) {
        const auto& m = st.texMtx[t];
        o[0] = dot4(m.m0, in[0], in[1], in[2], 1.f), o[1] = dot4(m.m1, in[0], in[1], in[2], 1.f);
        o[2] = g.type == GX_TG_MTX3x4 ? dot4(m.m2, in[0], in[1], in[2], 1.f) : 1.f;
      }
      if (g.postMtx != GX_PTIDENTITY && g.postMtx >= GX_PTTEXMTX0 && (g.postMtx - GX_PTTEXMTX0) / 3 < MaxPTTexMtx) {
        const auto& m = st.ptTexMtx[t];
        const float a = o[0], b = o[1], c = o[2];
        o[0] = dot4(m.m0, a, b, c, 1.f), o[1] = dot4(m.m1, a, b, c, 1.f), o[2] = dot4(m.m2, a, b, c, 1.f);
      }
      const float q = g.type == GX_TG_MTX3x4 && o[2] != 0.f ? o[2] : 1.f;
      out.uv[t][0] = o[0] / q, out.uv[t][1] = o[1] / q;
    }
    return out;
  };

  d.firstVertex = st.firstVertex;
  static thread_local std::vector<Vertex> strip;
  strip.resize(vtxCount);
  for (u32 i = 0; i < vtxCount; ++i)
    strip[i] = read_vertex(vertices + i * stride);
  Vertex* out = outVerts;
  const auto tri = [&](u32 a, u32 b, u32 c) {
    out[0] = strip[a], out[1] = strip[b], out[2] = strip[c];
    out += 3;
  };
  switch (prim) {
  case GX_TRIANGLES:
    for (u32 i = 0; i + 2 < vtxCount; i += 3)
      tri(i, i + 1, i + 2);
    break;
  case GX_TRIANGLESTRIP:
    for (u32 i = 0; i + 2 < vtxCount; ++i)
      (i & 1) ? tri(i + 1, i, i + 2) : tri(i, i + 1, i + 2);
    break;
  case GX_TRIANGLEFAN:
    for (u32 i = 1; i + 1 < vtxCount; ++i)
      tri(0, i, i + 1);
    break;
  case GX_QUADS:
    for (u32 i = 0; i + 3 < vtxCount; i += 4) {
      tri(i, i + 1, i + 2);
      tri(i, i + 2, i + 3);
    }
    break;
  default:
    break;
  }
  d.vertices = static_cast<u32>(out - outVerts);
  // an additive draw's light, as its first stage and first vertex say
  if (d.projType == 0 && d.blendMode == 1 && d.blendDst == 1 && d.stages > 0) {
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (u32 i = 0; i < vtxCount; ++i)
      for (int a = 0; a < 3; ++a)
        mn[a] = std::min(mn[a], strip[i].pos[a]), mx[a] = std::max(mx[a], strip[i].pos[a]);
    const bool line = prim == GX_LINES || prim == GX_LINESTRIP;
    for (int a = 0; a < 3; ++a) {
      d.lightPos[a] = line ? strip[0].pos[a] : (mn[a] + mx[a]) / 2;
      d.lightExtent = std::max(d.lightExtent, mx[a] - mn[a]);
    }
    const StageRec& s0 = d.stage[0];
    float tex[4] = {1.f, 1.f, 1.f, 1.f};
    if (s0.tex != UINT32_MAX && s0.map != GX_TEXMAP_NULL && s0.tex < MostTexAvg)
      std::memcpy(tex, s_texAvg[s0.tex], sizeof tex);
    const float* ras = strip[0].ras;
    const auto arg = [&](u32 code, int c) -> float {
      switch (code) {
      case 8:
        return tex[c];
      case 9:
        return tex[3];
      case 10:
        return ras[c];
      case 11:
        return ras[3];
      case 12:
        return 1.f;
      case 14:
        return s0.k[c];
      default:
        return 0.f;
      }
    };
    float most = 0.f;
    for (int c = 0; c < 3; ++c) {
      const float a = arg(s0.ca, c), b = arg(s0.cb, c), k = arg(s0.cc, c), e = arg(s0.cd, c);
      d.lightColour[c] = std::clamp(e + (1 - k) * a + k * b, 0.f, 1.f);
      most = std::max(most, d.lightColour[c]);
    }
    const float alpha = d.blendSrc == 4 ? tex[3] * (d.hasColor ? ras[3] : s0.k[3]) : 1.f;
    d.lightBright = most * alpha;
  }
}

#if defined(AURORA_RT_LIVE)
// AURORA_RT_LIGHTFACTS=<file>: what each draw's light is made of, measured: GX lights, vertex colours or neither. Reads, writes nothing the picture or the game uses.
void light_note(const Packet& st, LightNote& n) noexcept {
  const auto* base = reinterpret_cast<const u8*>(&st);
  const auto* stages = reinterpret_cast<const TevStage*>(base + st.stagesOffset);
  const u32 count = std::min<u32>(st.numTevStages, 16u);
  u32 chan = 0;
  n.readsRas = n.readsReg = 0;
  for (u32 i = 0; i < count; ++i) {
    const auto& g = stages[i];
    if (g.channelId == GX_COLOR1A1 || g.channelId == GX_COLOR1 || g.channelId == GX_ALPHA1)
      chan = 1;
    const u32 c[4] = {g.colorPass.a, g.colorPass.b, g.colorPass.c, g.colorPass.d};
    for (const u32 a : c)
      n.readsRas |= a == GX_CC_RASC ? 1 : 0, n.readsReg |= (a <= GX_CC_A2 && a >= GX_CC_C0) ? 1 : 0;
  }
  const auto& cc = st.colorChannelConfig[GX_COLOR0 + chan];
  const auto& cs = st.colorChannelState[GX_COLOR0 + chan];
  n.enabled = cc.lightingEnabled ? 1 : 0, n.ambVtx = cc.ambSrc == GX_SRC_VTX ? 1 : 0, n.matVtx = cc.matSrc == GX_SRC_VTX ? 1 : 0, n.diffFn = cc.diffFn;
  n.vtxColour = st.vtxDesc[GX_VA_CLR0 + chan] != GX_NONE ? 1 : 0;
  n.mask = 0;
  for (int c = 0; c < 4; ++c)
    n.amb[c] = cs.ambColor[c], n.mat[c] = cs.matColor[c];
  if (!cc.lightingEnabled)
    return;
  const auto* lights = reinterpret_cast<const Light*>(base + st.lightsOffset);
  for (u32 l = 0; l < 8; ++l) {
    if (!cs.lightMask.test(l))
      continue;
    n.mask |= 1u << l;
    for (int c = 0; c < 3; ++c)
      n.pos[l][c] = lights[l].pos[c], n.col[l][c] = lights[l].color[c];
  }
}
// One frame in s_factsEvery: a line a draw of the main view, a line a light in use, the road's normal beside the player's machine
FILE* s_facts = nullptr;
u32 s_factsEvery = 30;
void log_facts(const live::Job& j, const Frame& f) {
  static u32 n = 0;
  if (++n % s_factsEvery != 0 || !j.notes || f.main == nullptr)
    return;
  const DrawRec* draws = j.draws.get() + j.baseDraw;
  const Vertex* verts = j.verts.get() + j.baseVert;
  const LightNote* notes = j.notes.get() + j.baseDraw;
  const float px = f.main->proj[0], py = f.main->proj[5], ox = f.main->proj[2], oy = f.main->proj[6];
  std::fprintf(s_facts, "F,%llu,%u,%zu\n", static_cast<unsigned long long>(j.copy), j.drawCount, f.machineLights.size());
  std::map<std::tuple<u32, int, int, int, int, int, int>, u32> seen;
  double nearN[3]{};
  for (u32 i = 0; i < j.drawCount; ++i) {
    const DrawRec& d = draws[i];
    // what is drawn in the frame's list by another projection (a marker over the world, say)
    if (d.vertices >= 3 && d.projType == 0 && (d.proj[0] != f.main->proj[0] || d.proj[5] != f.main->proj[5])) {
      const float* m0 = verts[d.firstVertex].pos;
      std::fprintf(s_facts, "X,%u,%g,%g,%u,%u,%u,%u,%u,%u,%g,%g,%g\n", i, d.proj[0], d.proj[5], d.blendSrc, d.blendDst, d.depthUpdate, d.depthCompare, d.vertices / 3, d.samplesCopy, m0[0], m0[1], m0[2]);
    }
    if (d.vertices < 3 || d.projType != 0 || d.proj[0] != f.main->proj[0] || d.proj[5] != f.main->proj[5])
      continue;
    const LightNote& o = notes[i];
    double solid = 0, nx = 0, ny = 0, nz = 0, lumSum = 0;
    float lo = 1e9f, hi = -1e9f;
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    double rgb[3]{};
    for (u32 v = 0; v < d.vertices; ++v) {
      const Vertex& a = verts[d.firstVertex + v];
      const float l = a.ras[0] * 0.3f + a.ras[1] * 0.59f + a.ras[2] * 0.11f;
      lo = std::min(lo, l), hi = std::max(hi, l), lumSum += l;
      for (int c = 0; c < 3; ++c)
        mn[c] = std::min(mn[c], a.pos[c]), mx[c] = std::max(mx[c], a.pos[c]), rgb[c] += a.ras[c];
    }
    for (u32 v = 0; v + 2 < d.vertices; v += 3) {
      const float* p0 = verts[d.firstVertex + v].pos;
      const float* p1 = verts[d.firstVertex + v + 1].pos;
      const float* p2 = verts[d.firstVertex + v + 2].pos;
      const double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
      const double m[3] = {(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0};
      if (m[2] >= -0.1)
        continue;
      const double dist = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
      // towards the view, whichever way the triangle was wound
      const double side = cr[0] * m[0] + cr[1] * m[1] + cr[2] * m[2];
      if (side > 0)
        cr[0] = -cr[0], cr[1] = -cr[1], cr[2] = -cr[2];
      nx += cr[0], ny += cr[1], nz += cr[2];
      const double sx = m[0] / -m[2] * px + ox, sy = m[1] / -m[2] * py + oy;
      if (std::abs(sx) < 1.2 && std::abs(sy) < 1.2)
        solid += std::abs(side) / 2 / (dist * dist * dist);
      if (d.depthUpdate && d.blendMode != 1 && dist < 30) {
        // the track close by, its normal by area
        nearN[0] += cr[0], nearN[1] += cr[1], nearN[2] += cr[2];
      }
    }
    const double len = std::max(std::sqrt(nx * nx + ny * ny + nz * nz), 1e-9);
    const double dn = static_cast<double>(d.vertices);
    const bool machine = beside_engine_light(d, verts, f.machineLights);
    const float ext = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
    u32 tex = 0;
    for (u32 s = 0; s < std::min<u32>(d.stages, 16u); ++s)
      tex += d.stage[s].tex != UINT32_MAX && d.stage[s].map != GX_TEXMAP_NULL ? 1 : 0;
    std::fprintf(s_facts, "D,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%.1f,%.1f,%.1f,%.1f,%u,%.5f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,%u\n", i, o.enabled, o.vtxColour, o.ambVtx, o.matVtx, o.mask, o.diffFn, o.readsRas, o.readsReg, d.blendMode, d.blendSrc, d.blendDst,
                 d.depthUpdate, d.colorUpdate, machine ? 1 : 0, ext, (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2, d.vertices / 3, solid, lumSum / dn, lo, hi, rgb[0] / dn, rgb[1] / dn, rgb[2] / dn, nx / len, ny / len,
                 nz / len, tex, d.stages, d.depthCompare, d.samplesCopy);
    for (u32 l = 0; l < 8; ++l)
      if ((o.mask >> l) & 1) {
        const auto key = std::make_tuple(l, static_cast<int>(o.pos[l][0]), static_cast<int>(o.pos[l][1]), static_cast<int>(o.pos[l][2]), static_cast<int>(o.col[l][0] * 255), static_cast<int>(o.col[l][1] * 255), static_cast<int>(o.col[l][2] * 255));
        if (seen[key]++ == 0)
          std::fprintf(s_facts, "L,%u,%g,%g,%g,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", l, o.pos[l][0], o.pos[l][1], o.pos[l][2], o.col[l][0], o.col[l][1], o.col[l][2], o.amb[0], o.amb[1], o.amb[2]);
      }
  }
  const double len = std::max(std::sqrt(nearN[0] * nearN[0] + nearN[1] * nearN[1] + nearN[2] * nearN[2]), 1e-9);
  std::fprintf(s_facts, "G,%.3f,%.3f,%.3f\n", nearN[0] / len, nearN[1] / len, nearN[2] / len);
  std::fflush(s_facts);
}

// Translation beside the stream: chunks of packets to a couple of workers, each draw into the slot the stream gave it
constexpr size_t ChunkSize = 256 * 1024;
constexpr u32 MostDraws = 65536, MostVerts = 4000000;
struct Chunk {
  live::Job* job = nullptr;
  std::unique_ptr<u8[]> data;
  size_t capacity = 0, used = 0;
};
std::mutex s_queueLock;
std::condition_variable s_queueWake;
std::deque<std::unique_ptr<Chunk>> s_queue;
std::vector<std::unique_ptr<Chunk>> s_freeChunks;
std::deque<live::Job*> s_rulesToRun;
live::Job* s_job = nullptr;
std::unique_ptr<live::Job> s_jobOwned;
std::unique_ptr<Chunk> s_chunk;
u32 s_viewDraws = 0;
float s_sunPower = 0.f, s_sunPos[3]{}, s_sunCol[3]{};
u32 s_rulesTick = 0;
u32 s_epoch = 0, s_baseDraw = 0, s_baseVert = 0, s_nextDraw = 0, s_nextVert = 0;

// A light's middle hops as the draws in it change from frame to frame: half a step towards where it was, when it was near
void smooth_lights(Frame& f) {
  static std::mutex lock;
  static std::vector<EngineLight> last;
  std::lock_guard l{lock};
  for (EngineLight& n : f.lights) {
    const EngineLight* best = nullptr;
    float bestD = 6.f * 6.f;
    for (const EngineLight& o : last) {
      const float dx = o.pos[0] - n.pos[0], dy = o.pos[1] - n.pos[1], dz = o.pos[2] - n.pos[2], dd = dx * dx + dy * dy + dz * dz;
      if (dd < bestD)
        bestD = dd, best = &o;
    }
    if (best != nullptr)
      for (int a = 0; a < 3; ++a)
        n.pos[a] = (n.pos[a] + best->pos[a]) / 2;
  }
  last = f.lights;
}

static std::vector<std::vector<u8>> make_mips(u32 w, u32 h, std::vector<u8> rgba) {
  std::vector<std::vector<u8>> mips;
  mips.push_back(std::move(rgba));
  while (w > 1 || h > 1) {
    const u32 nw = std::max(w / 2, 1u), nh = std::max(h / 2, 1u);
    std::vector<u8> m(size_t(nw) * nh * 4);
    const auto& a = mips.back();
    for (u32 y = 0; y < nh; ++y)
      for (u32 x = 0; x < nw; ++x)
        for (u32 c = 0; c < 4; ++c) {
          const u32 xa = std::min(x * 2, w - 1), ya = std::min(y * 2, h - 1), xb = std::min(x * 2 + 1, w - 1), yb = std::min(y * 2 + 1, h - 1);
          m[(size_t(y) * nw + x) * 4 + c] = static_cast<u8>((a[(size_t(ya) * w + xa) * 4 + c] + a[(size_t(ya) * w + xb) * 4 + c] +
                                                             a[(size_t(yb) * w + xa) * 4 + c] + a[(size_t(yb) * w + xb) * 4 + c] + 2) / 4);
        }
    mips.push_back(std::move(m));
    w = nw, h = nh;
  }
  return mips;
}

FILE* s_lightLog = nullptr;
bool s_lateCheck = false;
struct LateRec {
  const live::Job* job;
  u32 draw, verts;
  bool additive, indexed;
  std::vector<Vertex> v;
};
std::mutex s_lateLock;
std::vector<LateRec> s_late;  // the stream's own reading of each draw, to compare with the workers'
u64 s_lateDraws[2]{}, s_lateMoved[2]{};
double s_lateMost[2]{};

void log_lights(const live::Job& j, const Frame& f) {
  std::fprintf(s_lightLog, "F,%llu,%g,%g,%g,%g,%g,%g,%g,%g\n", static_cast<unsigned long long>(j.copy), j.proj[0], j.proj[5], j.proj[2], j.proj[6], j.viewport[0],
               j.viewport[1], j.viewport[2], j.viewport[3]);
  for (const auto& p : f.merged)
    std::fprintf(s_lightLog, "M,%g,%g,%g,%g,%g,%g,%g\n", p.pos[0], p.pos[1], p.pos[2], p.colour[0], p.colour[1], p.colour[2], p.weight);
  for (u32 i = 0; i < j.drawCount; ++i) {
    const DrawRec& d = j.draws[j.baseDraw + i];
    if (d.lightExtent > 0.f)
      std::fprintf(s_lightLog, "C,%u,%u,%g,%g,%g,%g,%g,%g,%g,%g,%u,%u,%u,%u\n", static_cast<u32>(d.prim), d.vtxCount, d.lightPos[0], d.lightPos[1], d.lightPos[2], d.lightColour[0], d.lightColour[1],
                   d.lightColour[2], d.lightExtent, d.lightBright, d.stage[0].tex, d.stage[0].ca, d.stage[0].cc, d.blendSrc);
  }
}

// How far the workers' later reading of guest memory strays from what the stream saw
void late_compare(const live::Job& j) {
  std::lock_guard lock{s_lateLock};
  for (const LateRec& r : s_late) {
    if (r.job != &j || r.draw < j.baseDraw)
      continue;
    const DrawRec& d = j.draws[r.draw];
    double most = 0;
    for (u32 v = 0; v < r.verts; ++v) {
      const Vertex& a = j.verts[d.firstVertex + j.baseVert + v];
      const Vertex& b = r.v[v];
      most = std::max<double>(most, std::hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1], a.pos[2] - b.pos[2]));
    }
    const int k = r.additive ? 1 : 0;
    s_lateDraws[k] += 1;
    s_lateMoved[k] += most > 1e-4 ? 1 : 0;
    s_lateMost[k] = std::max(s_lateMost[k], most);
  }
  s_late.erase(std::remove_if(s_late.begin(), s_late.end(), [&](const LateRec& r) { return r.job == &j; }), s_late.end());
  static u32 n = 0;
  if (++n % 300 == 0)
    std::fprintf(stderr, "[rt] late read: draws moved by reading later: others %llu of %llu (most %.3f), additive %llu of %llu (most %.3f)\n",
                 static_cast<unsigned long long>(s_lateMoved[0]), static_cast<unsigned long long>(s_lateDraws[0]), s_lateMost[0],
                 static_cast<unsigned long long>(s_lateMoved[1]), static_cast<unsigned long long>(s_lateDraws[1]), s_lateMost[1]);
}

bool motion_thread_on() {
  static const bool v = [] {
    const char* e = std::getenv("AURORA_RT_MOTION_THREAD");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}
// One job at a time: the rules hold the motion lock from before they post until after they wait
std::mutex s_motionWorkLock;
std::condition_variable s_motionWorkWake;
std::function<void()> s_motionWork;
bool s_motionWorkDone = false;
void motion_post(std::function<void()> work) {
  static const bool started = [] {
    std::thread([] {
      live::name_thread();
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock lock{s_motionWorkLock};
          s_motionWorkWake.wait(lock, [] { return static_cast<bool>(s_motionWork); });
          job = std::move(s_motionWork);
          s_motionWork = nullptr;
        }
        job();
        {
          std::lock_guard lock{s_motionWorkLock};
          s_motionWorkDone = true;
        }
        s_motionWorkWake.notify_all();
      }
    }).detach();
    return true;
  }();
  (void)started;
  {
    std::lock_guard lock{s_motionWorkLock};
    s_motionWork = std::move(work);
    s_motionWorkDone = false;
  }
  s_motionWorkWake.notify_all();
}
void motion_wait() {
  std::unique_lock lock{s_motionWorkLock};
  s_motionWorkWake.wait(lock, [] { return s_motionWorkDone; });
}

// AURORA_RT_LAYER_PASS=1: blended layers that write no depth get a structure of their own (rt.hlsl MaskLayer), so rays pass them without looking at each
static bool layers_apart() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_LAYER_PASS");
    return v != nullptr && std::atof(v) != 0;
  }();
  return on;
}
void run_rules(live::Job& j) {
  const auto t0 = std::chrono::steady_clock::now();
  Tuning tuning;
  if (const char* v = std::getenv("AURORA_RT_LIGHT_FULL"))
    tuning.lightFull = static_cast<float>(std::atof(v));
  {
    static const char* perDraw = std::getenv("AURORA_RT_GLOW_PERDRAW");
    static const char* most = std::getenv("AURORA_RT_GLOW_MAX");
    if (perDraw != nullptr)
      tuning.glowPerDraw = static_cast<u32>(std::atoi(perDraw));
    if (most != nullptr)
      tuning.maxGlows = static_cast<u32>(std::atoi(most));
  }
  tuning.machineCap = s_machineCap, tuning.sceneryCap = s_sceneryCap, tuning.machineGain = s_machineGain;
  if (j.glow > 0.f) {
    tuning.texMeans = s_texAvg;
    tuning.texCount = MostTexAvg;
  }
  // AURORA_RT_SCENE_STATIC=1: the main view counts what the stream left out, and the camera is known already
  const bool staticScene = scene::static_on();
  const scene::StreamView* staticView = nullptr;
  if (staticScene) {
    for (u32 i = 0; i < j.world.streamViews && i < 4; ++i) {
      const scene::StreamView& v = j.world.stream[i];
      tuning.extraKey[tuning.extraViews] = v.key, tuning.extraVerts[tuning.extraViews] = v.vertices;
      ++tuning.extraViews;
      tuning.largestFloor = std::max(tuning.largestFloor, v.largest);
      if (v.known && (staticView == nullptr || v.vertices > staticView->vertices))
        staticView = &v;
    }
  }
  // Motion data beside the rules, not after them: the rules alone are near a frame's whole time on a busy course, and the picture waits for them
  static std::vector<float> prev;
  static std::mutex motionLock;  // two frames' rules can overlap
  std::unique_lock motionHeld{motionLock, std::defer_lock};
  std::thread motion;
  bool motionPosted = false;
  static const bool hashOnly = [] {
    const char* v = std::getenv("AURORA_RT_MOTION_ONLY");
    return v != nullptr && v[0] == '2';
  }();
  if (s_motionOn) {
    gfx::stagelog::Scope t{"motion_lock", j.copy};
    motionHeld.lock();
  }
  // the camera's move from the two frames' views, when the stream knew both
  static float lastView[12];
  static u64 lastViewCopy = 0;
  static bool lastViewKnown = false;
  Rigid staticMove;
  bool staticMoveKnown = false;
  if (staticScene && s_motionOn) {
    if (staticView != nullptr && lastViewKnown && j.copy > lastViewCopy)
      staticMoveKnown = rigid_a_over_b(lastView, staticView->view, staticMove);
    if (staticView != nullptr && j.copy > lastViewCopy)
      std::memcpy(lastView, staticView->view, sizeof lastView), lastViewCopy = j.copy;
    lastViewKnown = staticView != nullptr;
  }
  if (s_motionOn && !hashOnly)
  {
    const auto asked = std::chrono::steady_clock::now();
    const auto work = [&j, staticMove, staticMoveKnown, asked] {
      if (gfx::stagelog::on())
        gfx::stagelog::note("motion_start", asked, j.copy);
      static MotionTracker tracker;
      const auto ts0 = std::chrono::steady_clock::now();
      tracker.step(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, j.copy, prev, j.motion, staticMoveKnown ? &staticMove : nullptr);
      if (gfx::stagelog::on())
        gfx::stagelog::note("motion_step", ts0, j.copy);
      static double stepMs = 0;
      static u32 stepN = 0;
      stepMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
      if (++stepN == 300) {
        std::fprintf(stderr, "[rt] motion step %.2f ms, %u draws, %zu verts\n", stepMs / stepN, j.drawCount, prev.size() / 3);
        stepN = 0, stepMs = 0;
      }
    };
    // AURORA_RT_MOTION_THREAD=1: one thread for this for the whole run, not a new one each frame
    if (motion_thread_on()) {
      motion_post(work);
      motionPosted = true;
    } else {
      motion = std::thread(work);
    }
  }
  {
    gfx::stagelog::Scope t{"build_frame", j.copy};
    build_frame(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, tuning, j.frame);
  }
  if (staticScene && j.frame.main != nullptr) {
    {
      gfx::stagelog::Scope t{"match_static", j.copy};
      scene::match_static(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, s_texAvg, MostTexAvg, j.frame.main->proj[0], j.frame.main->proj[5], j.frame, j.world);
    }
    gfx::stagelog::Scope t{"assemble", j.copy};
    assemble_frame(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, j.world.drop.data(), j.frame, layers_apart());
  } else {
    assemble_frame(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, nullptr, j.frame, layers_apart());
    if (scene::on() && !staticScene && j.frame.main != nullptr)
      scene::match(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, s_texAvg, MostTexAvg, j.frame.main->proj[0], j.frame.main->proj[5], j.world);
  }
  if (motion.joinable() || motionPosted) {
    {
      gfx::stagelog::Scope t{"motion_join", j.copy};
      if (motionPosted)
        motion_wait();
      else
        motion.join();
    }
    j.frame.prevPos.clear();
    j.frame.prevPos.reserve(j.frame.verts.size() * 3);
    for (const auto& [first, count] : j.frame.spans)
      j.frame.prevPos.insert(j.frame.prevPos.end(), prev.data() + size_t(first) * 3, prev.data() + size_t(first + count) * 3);
  }
  if (motionHeld.owns_lock())
    motionHeld.unlock();
  smooth_lights(j.frame);
  j.drawable = j.frame.main != nullptr && !j.frame.verts.empty();
  j.textures.clear();
  if (j.drawable) {
    // only a job that will run carries textures away: their indices are for good
    // In index order, a bounded amount a frame: a scene's first frame brings dozens, and they must not stall the picture. Until a
    // texture is up, rays read it as plain white for the few frames that takes.
    {
      constexpr size_t Budget = 1u << 20;
      size_t bytes = 0;
      std::unique_lock lock{s_texLock};
      size_t n = 0;
      gfx::stagelog::Scope mipsTime{"tex_mips", j.copy};
      static thread_local std::vector<Tex> take;
      take.clear();
      while (n < s_texs.size() && (n == 0 || bytes + s_texs[n].rgba.size() <= Budget)) {
        bytes += s_texs[n].rgba.size();
        take.push_back(std::move(s_texs[n]));
        ++n;
      }
      s_texs.erase(s_texs.begin(), s_texs.begin() + static_cast<std::ptrdiff_t>(n));
      lock.unlock();
      for (Tex& t : take)
        j.textures.push_back({t.index, t.w, t.h, make_mips(t.w, t.h, std::move(t.rgba))});
    }
    std::memcpy(j.proj, j.frame.main->proj, sizeof j.proj);
    std::memcpy(j.viewport, j.frame.main->viewport, sizeof j.viewport);
  }
  j.rulesMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (gfx::stagelog::on())
    gfx::stagelog::note("rules", t0, j.copy);
  j.mergedLights = static_cast<u32>(j.frame.merged.size()), j.lightPicks = static_cast<u32>(j.frame.picks.size());
  if (s_lightLog != nullptr && j.drawable)
    log_lights(j, j.frame);
  if (s_lateCheck)
    late_compare(j);
  if (s_facts != nullptr && j.drawable)
    log_facts(j, j.frame);
  j.done.set_value();
}

void worker() {
  for (;;) {
    std::unique_ptr<Chunk> c;
    live::Job* rules = nullptr;
    {
      std::unique_lock lock{s_queueLock};
      s_queueWake.wait(lock, [] { return !s_queue.empty() || !s_rulesToRun.empty(); });
      if (!s_rulesToRun.empty()) {
        rules = s_rulesToRun.front();
        s_rulesToRun.pop_front();
      } else {
        c = std::move(s_queue.front());
        s_queue.pop_front();
      }
    }
    if (rules != nullptr) {
      run_rules(*rules);
      continue;
    }
    live::Job& j = *c->job;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t at = 0; at < c->used;) {
      const auto& p = *reinterpret_cast<const Packet*>(c->data.get() + at);
      if (p.epoch == j.epoch.load(std::memory_order_relaxed))
      {
        translate(p, j.draws[p.drawSlot], j.verts.get() + p.vertSlot);
        if (s_facts != nullptr)
          light_note(p, j.notes[p.drawSlot]);
      }
      at += p.size;
    }
    j.translateUs.fetch_add(static_cast<u64>(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()),
                            std::memory_order_relaxed);
    c->used = 0;
    if (c->capacity == ChunkSize) {
      std::lock_guard lock{s_queueLock};
      s_freeChunks.push_back(std::move(c));
    }
    if (j.pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
      run_rules(j);
  }
}

void publish_chunk() {
  if (!s_chunk)
    return;
  if (s_chunk->used == 0)
    return;
  s_job->pending.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard lock{s_queueLock};
    s_queue.push_back(std::move(s_chunk));
  }
  s_queueWake.notify_one();
}

void start_workers() {
  static const bool started = [] {
    const char* v = std::getenv("AURORA_RT_WORKERS");
    const int n = v != nullptr ? std::atoi(v) : 2;
    if (const char* path = std::getenv("AURORA_RT_LIGHTLOG")) {
      s_lightLog = std::fopen(path, "wb");
      if (s_lightLog != nullptr)
        std::setvbuf(s_lightLog, nullptr, _IOFBF, 1 << 20);
    }
    s_lateCheck = std::getenv("AURORA_RT_LATECHECK") != nullptr;
    if (const char* path = std::getenv("AURORA_RT_LIGHTFACTS"))
      s_facts = std::fopen(path, "wb");
    for (int i = 0; i < std::max(n, 1); ++i)
      std::thread([] {
        live::name_thread();
        worker();
      }).detach();
    return true;
  }();
  (void)started;
}

void ensure_job() {
  if (!s_job) {
    start_workers();
    s_jobOwned = live::take_job();
    s_job = s_jobOwned.get();
    if (!s_job->draws) {
      s_job->draws.reset(new DrawRec[MostDraws]);
      s_job->verts.reset(new Vertex[MostVerts]);
      if (s_facts != nullptr)
        s_job->notes.reset(new LightNote[MostDraws]);
    }
    s_job->pending.store(1, std::memory_order_relaxed);
    s_job->epoch.store(0, std::memory_order_relaxed);
    s_job->translateUs.store(0, std::memory_order_relaxed);
    s_job->done = std::promise<void>();
    s_job->rules = s_job->done.get_future();
    s_epoch = s_baseDraw = s_baseVert = s_nextDraw = s_nextVert = 0;
    if (scene::static_on()) {
      s_job->world.scene = scene::current();
      scene::stream_reset(s_job->world);
    }
  }
}
u8* packet_room(size_t need) {
  ensure_job();
  if (s_chunk && s_chunk->used + need > s_chunk->capacity)
    publish_chunk();
  if (!s_chunk) {
    if (need <= ChunkSize) {
      std::lock_guard lock{s_queueLock};
      if (!s_freeChunks.empty()) {
        s_chunk = std::move(s_freeChunks.back());
        s_freeChunks.pop_back();
      }
    }
    if (!s_chunk) {
      s_chunk = std::make_unique<Chunk>();
      s_chunk->capacity = std::max(need, ChunkSize);
      s_chunk->data.reset(new u8[s_chunk->capacity]);
    }
    s_chunk->job = s_job;
    s_chunk->used = 0;
  }
  s_chunk->job = s_job;
  u8* room = s_chunk->data.get() + s_chunk->used;
  s_chunk->used += need;
  return room;
}
#endif

// The stream's part: copy what the draw reads, in order; the reading itself happens later
// a stage reads a texture this frame's stream copied out of the EFB: its own data, nothing named
static bool samples_copy() noexcept {
  auto& st = g_gxState;
  for (u32 i = 0; i < std::min<u32>(st.numTevStages, 16u); ++i) {
    const u32 map = st.tevStages[i].texMapId;
    if (map == GX_TEXMAP_NULL || map >= MaxTextures)
      continue;
    auto obj = st.textures[map].texObj;
    if (st.copyTextures.find(obj.data) != st.copyTextures.end())
      return true;
  }
  return false;
}
// The game's own machine shadow, from the draw alone: its blend takes the source by one minus the picture's alpha and the picture by its alpha, and it writes no depth. No rule by model or course. Rays stand in for it.
// Not asked: whether its texture is a known picture copy. The game's shadow samples an address only sometimes known as one (the rule missed 15 in 16 draws on course 15).
bool is_game_shadow() noexcept {
  const auto& st = g_gxState;
  return st.blendMode == GX_BM_BLEND && st.blendFacSrc == GX_BL_INVDSTALPHA && st.blendFacDst == GX_BL_DSTALPHA && !st.depthUpdate;
}
// A draw's corners in view space, read from its own bytes before it is drawn: where its middle is, how big it is, and how squarely
// its first triangle lies in the view plane (1: square to the view's axis, as a sprite the game turns to the screen)
struct DrawShape {
  float mid[3], extent, facing;
};
static bool draw_shape(const GXVtxFmt fmt, const u16 vtxCount, const u8* data, DrawShape& out) noexcept {
  const auto& st = g_gxState;
  const auto& vf = st.vtxFmts[fmt];
  u32 stride = 0, posOff = 0, mtxOff = UINT32_MAX;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = st.vtxDesc[i];
    if (type == GX_NONE)
      continue;
    if (i == GX_VA_POS)
      posOff = stride;
    if (i == GX_VA_PNMTXIDX)
      mtxOff = stride;
    if (type == GX_DIRECT)
      stride += i <= GX_VA_TEX7MTXIDX ? 1 : comp_type_size(attr, vf.attrs[i].type) * comp_cnt_count(attr, vf.attrs[i].cnt);
    else
      stride += (type == GX_INDEX8 ? 1 : 2) * (i == GX_VA_NRM && vf.attrs[i].cnt == GX_NRM_NBT3 ? 3 : 1);
  }
  if (st.vtxDesc[GX_VA_POS] == GX_NONE || vtxCount < 3)
    return false;
  const auto& a = vf.attrs[GX_VA_POS];
  const u32 sz = comp_type_size(GX_VA_POS, a.type), n = std::min<u32>(comp_cnt_count(GX_VA_POS, a.cnt), 3);
  float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f}, sum[3]{}, tri[3][3]{};
  for (u32 v = 0; v < vtxCount; ++v) {
    const u8* vb = data + v * stride;
    const u8* s = vb + posOff;
    bool le = false;
    if (st.vtxDesc[GX_VA_POS] != GX_DIRECT) {
      const u32 index = st.vtxDesc[GX_VA_POS] == GX_INDEX8 ? s[0] : rd_be(s, 2);
      const auto& arr = st.arrays[GX_VA_POS];
      if (arr.data == nullptr || index * arr.stride >= arr.size)
        return false;
      s = static_cast<const u8*>(arr.data) + index * arr.stride, le = arr.le;
    }
    float p[3]{};
    for (u32 c = 0; c < n; ++c)
      p[c] = rd_comp(s + c * sz, a.type, a.frac, le);
    const u32 mi = mtxOff != UINT32_MAX ? vb[mtxOff] / 3u : st.currentPnMtx / 3u;
    const auto& m = st.pnMtx[std::min<u32>(mi, MaxPnMtx - 1)].pos;
    const float q[3] = {dot4(m.m0, p[0], p[1], p[2], 1.f), dot4(m.m1, p[0], p[1], p[2], 1.f), dot4(m.m2, p[0], p[1], p[2], 1.f)};
    for (int c = 0; c < 3; ++c) {
      mn[c] = std::min(mn[c], q[c]), mx[c] = std::max(mx[c], q[c]), sum[c] += q[c];
      if (v < 3)
        tri[v][c] = q[c];
    }
  }
  for (int c = 0; c < 3; ++c)
    out.mid[c] = sum[c] / vtxCount;
  out.extent = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
  const float e1[3] = {tri[1][0] - tri[0][0], tri[1][1] - tri[0][1], tri[1][2] - tri[0][2]}, e2[3] = {tri[2][0] - tri[0][0], tri[2][1] - tri[0][1], tri[2][2] - tri[0][2]};
  const float nn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  const float ln = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
  out.facing = ln > 0 ? std::fabs(nn[2]) / ln : 0.f;
  return true;
}
// A texture whose alpha is a star: thin bright arms along its middle row and column, dark between them. Read on the ring at a quarter
// to nearly half its size: the arms' mean against the diagonals' mean at the same radius. A round glow is the same both ways; smoke
// and flame have no arms. Worked out once a texture, from its own pixels.
static bool star_alpha(const GXTexObj_& in) noexcept {
  static std::map<std::tuple<const void*, u32, u32, u32>, bool> known;  // what each texture's pixels said, not a list of textures
  GXTexObj_ obj = in;
  if (!recomp::resolve_texture_data(obj) || obj.data == nullptr)
    return false;
  const u32 w = obj.width(), h = obj.height(), f = obj.format();
  const auto key = std::make_tuple(obj.data, w, h, f);
  if (const auto it = known.find(key); it != known.end())
    return it->second;
  bool star = false;
  // square, small, and not a palette texture (whose colours need a table load to read)
  if (w == h && w >= 16 && w <= 128 && f != GX_TF_C4 && f != GX_TF_C8 && f != GX_TF_C14X2) {
    const size_t bytes = GXGetTexBufferSize(static_cast<u16>(w), static_cast<u16>(h), f, GX_FALSE, 0);
    const auto conv = gfx::convert_texture(f, w, h, 1, {static_cast<const u8*>(obj.data), bytes});
    if (conv.data.size() >= size_t(w) * h * 4) {
      const u8* px = conv.data.data();
      const auto alpha = [&](float x, float y) {
        const int xi = std::clamp(static_cast<int>(x), 0, static_cast<int>(w) - 1), yi = std::clamp(static_cast<int>(y), 0, static_cast<int>(h) - 1);
        return px[(size_t(yi) * w + xi) * 4 + 3] / 255.f;
      };
      const float c = (w - 1) * 0.5f;
      float arms = 0, diag = 0;
      int k = 0;
      for (float r = w * 0.25f; r <= w * 0.45f; r += 1.f, ++k) {
        // the two pixels either side of the middle line, as the middle of an even texture falls between them
        arms += (alpha(c + r, c - 0.5f) + alpha(c + r, c + 0.5f) + alpha(c - r, c - 0.5f) + alpha(c - r, c + 0.5f) + alpha(c - 0.5f, c + r) +
                 alpha(c + 0.5f, c + r) + alpha(c - 0.5f, c - r) + alpha(c + 0.5f, c - r)) / 8.f;
        const float d = r * 0.7071f;
        diag += (alpha(c + d, c + d) + alpha(c - d, c + d) + alpha(c + d, c - d) + alpha(c - d, c - d)) / 4.f;
      }
      arms /= std::max(k, 1), diag /= std::max(k, 1);
      star = arms > 0.15f && arms > diag * 4.f;
    }
  }
  known.emplace(key, star);
  return star;
}
bool g_noEngineStar = false;
std::atomic<u64> g_starsSkipped{0};
// The game's star on a far machine's engine, from the draw alone: an additive sprite (one quad turned square to the view, writing no
// depth) whose texture's alpha is a star (star_alpha), and that the game has grown with distance to keep it readable: at least
// StarGrown across in the world (twice the largest engine glow, which keeps its own size) and a set share of its distance. Near the
// camera the star keeps its own size and stays, as do glows, flames, smoke and sparks (round, cloudy or lines). Picture only.
constexpr float StarGrown = 4.f, StarShareOfDistance = 0.05f;
bool is_engine_star(const GXPrimitive prim, const GXVtxFmt fmt, const u16 vtxCount, const u8* data) noexcept {
  const auto& st = g_gxState;
  if (vtxCount != 4 || (prim != GX_QUADS && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN) || st.projType != GX_PERSPECTIVE ||
      st.blendMode != GX_BM_BLEND || st.blendFacDst != GX_BL_ONE || st.depthUpdate || !st.colorUpdate || st.numTevStages == 0)
    return false;
  const u32 map = st.tevStages[0].texMapId;
  if (map >= MaxTextures)
    return false;
  DrawShape sh{};
  if (!draw_shape(fmt, vtxCount, data, sh) || sh.facing < 0.95f || sh.mid[2] >= 0.f)
    return false;
  if (sh.extent < StarGrown || sh.extent < -sh.mid[2] * StarShareOfDistance)
    return false;
  if (!star_alpha(st.textures[map].texObj))
    return false;
  g_starsSkipped.fetch_add(1, std::memory_order_relaxed);
  return true;
}
// AURORA_RT_DRAWLOG=first,last: on those copies, each draw that writes no depth or samples a picture copy: its state, whether the rule took it, where it is.
// AURORA_RT_STARLOG=first,last: on those copies, each additive or depth-less sprite of up to 8 corners: its shape and texture, and whether is_engine_star took it
void note_draw(const u16 vtxCount, const GXPrimitive prim, const GXVtxFmt fmt, const u8* data) noexcept {
  static u64 first = 0, last = 0;
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_DRAWLOG");
    if (v != nullptr && std::strchr(v, ',') != nullptr)
      first = std::strtoull(v, nullptr, 10), last = std::strtoull(std::strchr(v, ',') + 1, nullptr, 10);
    return v != nullptr;
  }();
  static u64 sfirst = 0, slast = 0;
  static const bool starOn = [] {
    const char* v = std::getenv("AURORA_RT_STARLOG");
    if (v != nullptr && std::strchr(v, ',') != nullptr)
      sfirst = std::strtoull(v, nullptr, 10), slast = std::strtoull(std::strchr(v, ',') + 1, nullptr, 10);
    return v != nullptr;
  }();
  if (starOn && s_copyNow >= sfirst && s_copyNow <= slast) {
    const auto& st = g_gxState;
    const bool additive = st.blendMode == GX_BM_BLEND && st.blendFacDst == GX_BL_ONE;
    DrawShape sh{};
    if (st.projType == GX_PERSPECTIVE && (additive || !st.depthUpdate) && vtxCount <= 8 && draw_shape(fmt, vtxCount, data, sh)) {
      const auto& p = st.proj;
      const float cx = dot4(p.m0, sh.mid[0], sh.mid[1], sh.mid[2], 1.f), cy = dot4(p.m1, sh.mid[0], sh.mid[1], sh.mid[2], 1.f), cw = dot4(p.m3, sh.mid[0], sh.mid[1], sh.mid[2], 1.f);
      const GXTexObj_* t = st.numTevStages > 0 && st.tevStages[0].texMapId < MaxTextures ? &st.textures[st.tevStages[0].texMapId].texObj : nullptr;
      std::fprintf(stderr, "[rt] star copy %llu: blend %d %d %d z %d%d verts %u prim %d mid %.1f %.1f %.1f ext %.2f face %.3f e/z %.4f screen %.3f %.3f tex %ux%u f%u %p star %d rule %d\n",
                   static_cast<unsigned long long>(s_copyNow), int(st.blendMode), int(st.blendFacSrc), int(st.blendFacDst), int(st.depthUpdate), int(st.depthCompare), vtxCount, int(prim),
                   sh.mid[0], sh.mid[1], sh.mid[2], sh.extent, sh.facing, sh.extent / std::max(std::fabs(sh.mid[2]), 1e-3f), cw != 0 ? cx / cw * 0.5f + 0.5f : -1.f,
                   cw != 0 ? 0.5f - cy / cw * 0.5f : -1.f, t ? t->width() : 0, t ? t->height() : 0, t ? t->format() : 0, t ? t->data : nullptr, t ? int(star_alpha(*t)) : 0,
                   int(is_engine_star(prim, fmt, vtxCount, data)));
    }
  }
  if (!on || s_copyNow < first || s_copyNow > last)
    return;
  const auto& st = g_gxState;
  if (st.depthUpdate && !samples_copy())
    return;
  const auto& m = st.pnMtx[st.currentPnMtx / 3];
  std::fprintf(stderr, "[rt] draw copy %llu: proj %d blend %d src %d dst %d zwrite %d zcmp %d zfunc %d colour %d alpha %d copy %d stages %u gens %u verts %u cull %d rule %d tex %p at %.1f %.1f %.1f\n",
               static_cast<unsigned long long>(s_copyNow), int(st.projType), int(st.blendMode), int(st.blendFacSrc), int(st.blendFacDst), int(st.depthUpdate), int(st.depthCompare), int(st.depthFunc),
               int(st.colorUpdate), int(st.alphaUpdate), int(samples_copy()), st.numTevStages, st.numTexGens, vtxCount, int(st.cullMode), int(is_game_shadow()), st.numTevStages > 0 && st.tevStages[0].texMapId < MaxTextures ? st.textures[st.tevStages[0].texMapId].texObj.data : nullptr, m.pos.m0.w(), m.pos.m1.w(), m.pos.m2.w());
}
static std::uint64_t s_stateKeyNow = 0;  // this draw's, while it's captured: on the stream's thread
static void make_packet(u8* room, const size_t size, const u32 stages, const u32 mtxCount, const bool lit, GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount,
                        const u8* vertices, const u32 rawBytes, u32 drawSlot, u32 vertSlot, u32 drawIndex, u32 firstVertex, u32 epoch) noexcept {
  auto& st = g_gxState;
  auto& p = *new (room) Packet;
  p.size = static_cast<u32>(size), p.epoch = epoch, p.drawSlot = drawSlot, p.vertSlot = vertSlot, p.drawIndex = drawIndex, p.firstVertex = firstVertex;
  p.prim = prim, p.vtxCount = vtxCount;
  p.proj = st.proj, p.logicalViewport = st.logicalViewport, p.projType = st.projType, p.numTevStages = st.numTevStages;
  p.blendMode = st.blendMode, p.blendFacSrc = st.blendFacSrc, p.blendFacDst = st.blendFacDst, p.depthUpdate = st.depthUpdate;
  p.depthCompare = st.depthCompare, p.cullMode = st.cullMode, p.colorUpdate = st.colorUpdate;
  p.fog = st.fog, p.colorRegs = st.colorRegs, p.tevSwapTable = st.tevSwapTable, p.kcolors = st.kcolors, p.tcgs = st.tcgs;
  p.colorChannelConfig = st.colorChannelConfig, p.colorChannelState = st.colorChannelState;
  p.vtxDesc = st.vtxDesc, p.arrays = st.arrays, p.numTexGens = st.numTexGens, p.vf = st.vtxFmts[fmt], p.currentPnMtx = st.currentPnMtx;
  for (u32 t = 0; t < MaxUv && t < st.numTexGens; ++t) {
    const auto& g = st.tcgs[t];
    if (g.mtx != GX_IDENTITY && g.mtx >= GX_TEXMTX0 && (g.mtx - GX_TEXMTX0) / 3 < MaxTexMtx)
      p.texMtx[t] = st.texMtxs[(g.mtx - GX_TEXMTX0) / 3];
    if (g.postMtx != GX_PTIDENTITY && g.postMtx >= GX_PTTEXMTX0 && (g.postMtx - GX_PTTEXMTX0) / 3 < MaxPTTexMtx)
      p.ptTexMtx[t] = st.ptTexMtxs[(g.postMtx - GX_PTTEXMTX0) / 3];
  }
  size_t at = (sizeof(Packet) + 15) & ~size_t(15);
  p.stagesOffset = static_cast<u32>(at);
  auto* stage = reinterpret_cast<TevStage*>(room + at);
  auto* snap = reinterpret_cast<TexSnap*>(stage + stages);
  for (u32 i = 0; i < stages; ++i) {
    new (stage + i) TevStage(st.tevStages[i]);
    const u32 map = st.tevStages[i].texMapId;
    auto* ts = new (snap + i) TexSnap;
    ts->palette = false, ts->tlutOk = false;
    if (map != GX_TEXMAP_NULL && map < MaxTextures) {
      ts->obj = st.textures[map].texObj;
      const u32 f = ts->obj.format();
      ts->palette = f == GX_TF_C4 || f == GX_TF_C8 || f == GX_TF_C14X2;
      if (ts->palette)
        ts->tlutOk = recomp::resolve_texture_tlut(map, &ts->tlut);
    }
  }
  at = reinterpret_cast<u8*>(snap + stages) - room;
  p.mtxOffset = static_cast<u32>(at), p.mtxCount = mtxCount;
  if (mtxCount == 1)
    new (room + at) PnMtx(st.pnMtx[std::min<u32>(st.currentPnMtx, MaxPnMtx - 1)]);
  else
    std::memcpy(room + at, st.pnMtx.data(), sizeof(PnMtx) * MaxPnMtx);
  at += sizeof(PnMtx) * mtxCount;
  p.lightsOffset = static_cast<u32>(at);
  if (lit)
    std::memcpy(room + at, st.lights.data(), sizeof(Light) * GX::MaxLights);
  at += lit ? sizeof(Light) * GX::MaxLights : 0;
  p.rawOffset = static_cast<u32>(at), p.rawBytes = rawBytes;
  p.samplesCopy = samples_copy() ? 1 : 0;
  p.stateKey = s_stateKeyNow;
  std::memcpy(room + at, vertices, rawBytes);
}

// As translate's d.fading, from the stream's state: the channel the stages read is the first, or the second if a stage asks for it
static bool draw_fading() noexcept {
  const auto& st = g_gxState;
  u32 chan = 0;
  for (u32 i = 0; i < std::min<u32>(st.numTevStages, 16u); ++i)
    if (st.tevStages[i].channelId == GX_COLOR1A1 || st.tevStages[i].channelId == GX_COLOR1 || st.tevStages[i].channelId == GX_ALPHA1)
      chan = 1;
  const auto& cc = st.colorChannelConfig[GX_ALPHA0 + chan];
  return s_fadeAsDraw && cc.matSrc != GX_SRC_VTX && !cc.lightingEnabled && st.colorChannelState[GX_ALPHA0 + chan].matColor[3] < 0.99f;
}
void capture_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* vertices, size_t rawBytes) noexcept {
  auto& st = g_gxState;
  // lines only for what they light: sparks are additive lines and strips, and no ray ever hits a line
  const bool line = prim == GX_LINES || prim == GX_LINESTRIP;
  if (prim == GX_POINTS || (!line && vtxCount < 3) || (line && (vtxCount < 2 || st.blendMode != 1 || st.blendFacDst != 1 || !g_live)))
    return;
  if (g_live && st.projType != GX_PERSPECTIVE)
    return;
  const auto liveStart = g_live ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
#if defined(AURORA_RT_LIVE)
  // The sun: the strongest light any draw of this frame asked for, as the stream has it
  for (u32 c = 0; c < 2; ++c) {
    const bool asked = g_gameLightOff ? g_gameLit[c] : st.colorChannelConfig[c].lightingEnabled;
    if (!asked)
      continue;
    for (u32 l = 0; l < 8; ++l)
      if (st.colorChannelState[c].lightMask.test(l)) {
        const auto& li = st.lights[l];
        const float power = li.color[0] + li.color[1] + li.color[2];
        if (power > s_sunPower) {
          s_sunPower = power;
          for (int a = 0; a < 3; ++a)
            s_sunPos[a] = li.pos[a], s_sunCol[a] = li.color[a];
        }
      }
  }
#endif
#if defined(AURORA_RT_LIVE)
  s_stateKeyNow = g_live && scene::static_on() ? state_key() : 0;
  // AURORA_RT_SCENE_STATIC=1: a draw the static scene holds, proved the same, costs its bytes' hash here and nothing further on. Additive and no-depth passes always go on: a second pass over the same list isn't the surface.
  if (g_live && scene::static_on() && (prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP || prim == GX_TRIANGLEFAN) && vtxCount >= 3 && st.vtxDesc[GX_VA_PNMTXIDX] == GX_NONE && st.vtxDesc[GX_VA_POS] == GX_DIRECT && st.depthUpdate && st.depthCompare && st.colorUpdate && !(st.blendMode == 1 && st.blendFacDst == 1) && !draw_fading()) {
    ensure_job();
    if (scene::Scene* sc = s_job->world.scene.get()) {
      const auto& mtx = st.pnMtx[std::min<u32>(st.currentPnMtx, MaxPnMtx - 1)].pos;
      float m[12] = {mtx.m0.x(), mtx.m0.y(), mtx.m0.z(), mtx.m0.w(), mtx.m1.x(), mtx.m1.y(), mtx.m1.z(), mtx.m1.w(), mtx.m2.x(), mtx.m2.y(), mtx.m2.z(), mtx.m2.w()};
      float proj[16];
      std::memcpy(proj, &st.proj, sizeof proj);
      if (scene::stream_skip(s_job->world, *sc, prim, vtxCount, vertices, rawBytes, m, proj, triangle_vertices(prim, vtxCount), s_rulesTick, s_stateKeyNow)) {
        s_captureMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - liveStart).count();
        return;
      }
    }
  }
#endif
  const u32 stages = std::min<u32>(st.numTevStages, 16u);
  const u32 mtxCount = st.vtxDesc[GX_VA_PNMTXIDX] != GX_NONE ? MaxPnMtx : 1;
  bool lit = false;
  for (const auto& c : st.colorChannelConfig)
    lit = lit || c.lightingEnabled;
  const size_t size = (((sizeof(Packet) + 15) & ~size_t(15)) + (sizeof(TevStage) + sizeof(TexSnap)) * stages + sizeof(PnMtx) * mtxCount +
                       (lit ? sizeof(Light) * GX::MaxLights : 0) + rawBytes + 15) & ~size_t(15);
  const u32 tv = triangle_vertices(prim, vtxCount);
#if defined(AURORA_RT_LIVE)
  if (g_live) {
    u8* room = packet_room(size);
    if (s_nextDraw >= MostDraws || s_nextVert + tv > MostVerts) {
      // beyond the buffers: not traced, the frame goes on
      s_chunk->used -= size;
      return;
    }
    make_packet(room, size, stages, mtxCount, lit, prim, fmt, vtxCount, vertices, static_cast<u32>(rawBytes), s_nextDraw, s_nextVert,
                s_nextDraw - s_baseDraw, s_nextVert - s_baseVert, s_epoch);
    if (s_lateCheck && tv > 0) {
      LateRec r{s_job, s_nextDraw, tv, st.blendMode == 1 && st.blendFacDst == 1, st.vtxDesc[GX_VA_POS] != GX_DIRECT, std::vector<Vertex>(tv)};
      DrawRec tmp;
      translate(*reinterpret_cast<const Packet*>(room), tmp, r.v.data());
      std::lock_guard lock{s_lateLock};
      s_late.push_back(std::move(r));
    }
    s_nextDraw += 1, s_nextVert += tv;
    s_captureMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - liveStart).count();
    return;
  }
#endif
  // the still tool's capture file: all of it here, in order
  static std::vector<u8> one;
  one.resize(size);
  const u32 drawIndex = static_cast<u32>(s_draws.size()), first = static_cast<u32>(s_verts.size());
  make_packet(one.data(), size, stages, mtxCount, lit, prim, fmt, vtxCount, vertices, static_cast<u32>(rawBytes), drawIndex, first, drawIndex, first, 0);
  s_draws.emplace_back();
  s_verts.resize(first + tv);
  translate(*reinterpret_cast<const Packet*>(one.data()), s_draws.back(), s_verts.data() + first);
}

#if defined(AURORA_RT_LIVE)
// What was drawn so far went elsewhere: its packets are skipped, the slots after them used next. The buffers fill only if a
// frame does this without ever reaching the rays; then wait for the workers and start again.
static void forget_draws() noexcept {
  if (!s_job)
    return;
  publish_chunk();
  s_baseDraw = s_nextDraw, s_baseVert = s_nextVert;
  s_job->epoch.store(++s_epoch, std::memory_order_relaxed);
  s_captureMs = 0;
  if (scene::static_on())
    scene::stream_reset(s_job->world);
  if (s_nextDraw > MostDraws / 2 || s_nextVert > MostVerts / 2) {
    while (s_job->pending.load(std::memory_order_acquire) > 1)
      std::this_thread::yield();
    s_baseDraw = s_baseVert = s_nextDraw = s_nextVert = 0;
  }
}

static void send_rays() noexcept {
  if (s_rayed)
    return;
  s_rayed = true;
  g_capturing = false;
  if (!s_job) {
    // the look alone, or a frame with nothing for the rays
    if (g_live || !g_hook)
      return;
    // a real view is many draws (a logo or a menu piece is not): those frames stay as they are
    const u32 viewDraws = s_viewDraws;
    s_viewDraws = 0;
    if (viewDraws < 100)
      return;
    auto look = live::take_job();
    look->postOnly = true, look->drawable = true;
    look->sunColour[0] = look->sunColour[1] = look->sunColour[2] = 0.f;
    look->captureMs = 0, look->rulesMs = 0, look->road = look->light = look->glow = look->calm = look->wet = 0;
    look->copy = s_copyNow;
    look->done = std::promise<void>();
    look->rules = look->done.get_future();
    look->done.set_value();
    live::submit_job(std::move(look));
    return;
  }
  const u32 draws = s_nextDraw - s_baseDraw;
  ++s_rulesTick;
  publish_chunk();
  auto job = std::move(s_jobOwned);
  s_job = nullptr;
  job->baseDraw = s_baseDraw, job->baseVert = s_baseVert, job->drawCount = draws;
  job->road = s_road, job->light = s_light, job->glow = s_glow, job->calm = s_calm, job->wet = s_wet;
  job->captureMs = s_captureMs;
  const float sunLen = std::sqrt(s_sunPos[0] * s_sunPos[0] + s_sunPos[1] * s_sunPos[1] + s_sunPos[2] * s_sunPos[2]);
  for (int a = 0; a < 3; ++a) {
    job->sunDir[a] = sunLen > 0.f ? s_sunPos[a] / sunLen : 0.f;
    job->sunColour[a] = sunLen > 0.f ? s_sunCol[a] : 0.f;
  }
  s_sunPower = 0.f;
  job->copy = s_copyNow;
  s_captureMs = 0;
  // the guard the job began with: if the workers are already through, one of them runs the rules
  live::Job* j = job.get();
  if (j->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    {
      std::lock_guard lock{s_queueLock};
      s_rulesToRun.push_back(j);
    }
    s_queueWake.notify_one();
  }
  live::submit_job(std::move(job));
}
#endif

static u32 s_zNow = 0, s_zLast = 0;
static bool s_beforeTaken = false;
void before_draw() noexcept {
#if defined(AURORA_RT_LIVE)
  // AURORA_RT_LIGHTFACTS: where the frame's orthographic draws fall against the scene
  if (s_facts != nullptr && g_gxState.projType != GX_PERSPECTIVE) {
    static u64 before = 0, beforeDepth = 0, after = 0, afterDepth = 0, n = 0;
    (s_seenPerspective ? after : before) += 1;
    (s_seenPerspective ? afterDepth : beforeDepth) += g_gxState.depthUpdate ? 1 : 0;
    if (++n % 20000 == 0)
      std::fprintf(stderr, "[rt] orthographic draws: before the scene %llu (write depth %llu), after %llu (write depth %llu)\n", static_cast<unsigned long long>(before), static_cast<unsigned long long>(beforeDepth), static_cast<unsigned long long>(after), static_cast<unsigned long long>(afterDepth));
  }
  if (s_rayed)
    return;
  if (g_gxState.projType == GX_PERSPECTIVE) {
    s_seenPerspective = true;
    ++s_viewDraws;
    // AURORA_RT_BEFORE_BLENDS=1: the frame's last blended draws come after its last depth-writing draw; last frame's count of those says when
    static const bool beforeOn = live::before_on();
    if (beforeOn && g_capturing) {
      if (g_gxState.depthUpdate)
        ++s_zNow;
      else if (!s_beforeTaken && s_zLast > 0 && s_zNow >= s_zLast) {
        s_beforeTaken = true;
        live::snapshot_before();
      }
    }
  } else if (s_seenPerspective)
    send_rays();
#endif
}

void efb_cleared() noexcept {
  if (!g_hook || s_rayed)
    return;
  s_zNow = 0, s_beforeTaken = false;
#if defined(AURORA_RT_LIVE)
  forget_draws();
#endif
  s_seenPerspective = false;
}

void begin_copy(std::uint64_t copy) noexcept {
#if defined(AURORA_RT_LIVE)
  s_copyNow = copy;
#endif
  if (s_atCopy != 0 && copy == s_atCopy) {
    g_capturing = true;
    s_atCopy = 0;
  }
}

void frame_end() noexcept {
  // AURORA_LOOK_PEAKLOG=<file>: the raw picture's range, measured a few times a second's worth of frames
  static const bool peakLog = std::getenv("AURORA_LOOK_PEAKLOG") != nullptr;
  static u32 peakAt = 0;
  if (peakLog && ++peakAt % 120 == 0)
    gfx::efb_readback::request();
#if defined(AURORA_RT_LIVE)
  if (g_hook) {
    if (s_seenPerspective)
      send_rays();
    s_zLast = s_zNow, s_zNow = 0, s_beforeTaken = false;
    s_rayed = false;
    s_seenPerspective = false;
    forget_draws();
    s_viewDraws = 0;
    g_capturing = g_live && !live::failed() && live::ready();
    return;
  }
#endif
  s_seenPerspective = false;
  if (g_capturing) {
    g_capturing = false;
    if (FILE* f = std::fopen(s_path.c_str(), "wb")) {
      const u32 head[4] = {0x34435452u, static_cast<u32>(s_draws.size()), static_cast<u32>(s_verts.size()), static_cast<u32>(s_texs.size())};
      std::fwrite(head, 4, 4, f);
      const float rect[4] = {static_cast<float>(g_gxState.texCopySrc.x), static_cast<float>(g_gxState.texCopySrc.y),
                             static_cast<float>(g_gxState.texCopySrc.width), static_cast<float>(g_gxState.texCopySrc.height)};
      std::fwrite(rect, 4, 4, f);
      std::fwrite(s_draws.data(), sizeof(DrawRec), s_draws.size(), f);
      std::fwrite(s_verts.data(), sizeof(Vertex), s_verts.size(), f);
      for (const auto& t : s_texs) {
        const u32 h[4] = {t.w, t.h, t.wrapS, t.wrapT};
        std::fwrite(h, 4, 4, f);
        std::fwrite(t.rgba.data(), 1, t.rgba.size(), f);
      }
      std::fclose(f);
    }
    std::fprintf(stderr, "[rt] captured display copy %llu: %zu draws, %zu triangles, %zu textures, draw record %zu bytes, vertex %zu\n",
                 static_cast<unsigned long long>(s_frame), s_draws.size(), s_verts.size() / 3, s_texs.size(), sizeof(DrawRec), sizeof(Vertex));
    s_draws.clear();
    s_verts.clear();
    s_texs.clear();
  }
  ++s_frame;
  if (!s_path.empty() && s_atCopy == 0 && s_at != 0 && s_frame == s_at)
    g_capturing = true;
}
} // namespace aurora::gx::rt
