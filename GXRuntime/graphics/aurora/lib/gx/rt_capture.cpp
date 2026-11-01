#include "rt_capture.hpp"
#include "rt_format.hpp"
#if defined(AURORA_RT_LIVE)
#include "rt_live.hpp"
#endif

#include "recomp.hpp"
#include "../gfx/texture_convert.hpp"

#include <algorithm>
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
bool g_capturing = false;
bool g_noHud = false;
bool g_live = false;
bool g_hook = false;
static bool s_seenPerspective = false;

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
  const char* noHud = std::getenv("AURORA_RT_NO_HUD");
  g_noHud = noHud != nullptr && noHud[0] == '1';
  if (const char* at = std::getenv("AURORA_RT_CAPTURE_COPY"))
    s_atCopy = std::strtoull(at, nullptr, 10);
  g_capturing = g_live || (!s_path.empty() && s_at == 0 && s_atCopy == 0);
#if defined(AURORA_RT_LIVE)
  g_hook = g_live || live::look_on();
#endif
  // AURORA_RT_LOOK=1, the overhaul look: reflections you can't miss. Each number can still be set by its own switch below. All tuned by eye.
  if (const char* v = std::getenv("AURORA_RT_LOOK"); v != nullptr && v[0] == '1') {
    s_road = 1.4f, s_glow = 0.5f, s_calm = 0.3f, s_wet = 0.35f, s_machineCap = 0.9f, s_sceneryCap = 0.7f, s_machineGain = 3.f;
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
  u32 rawOffset, stagesOffset, mtxOffset, mtxCount, lightsOffset;
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
  d.lit = st.colorChannelConfig[GX_COLOR0 + chan].lightingEnabled;

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
// Translation beside the stream: chunks of packets to a couple of workers, each draw into the slot the stream gave it
constexpr size_t ChunkSize = 256 * 1024;
constexpr u32 MostDraws = 32768, MostVerts = 1500000;
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

void run_rules(live::Job& j) {
  const auto t0 = std::chrono::steady_clock::now();
  Tuning tuning;
  if (const char* v = std::getenv("AURORA_RT_LIGHT_FULL"))
    tuning.lightFull = static_cast<float>(std::atof(v));
  tuning.machineCap = s_machineCap, tuning.sceneryCap = s_sceneryCap, tuning.machineGain = s_machineGain;
  if (j.glow > 0.f) {
    tuning.texMeans = s_texAvg;
    tuning.texCount = MostTexAvg;
  }
  build_frame(j.draws.get() + j.baseDraw, j.drawCount, j.verts.get() + j.baseVert, tuning, j.frame);
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
  j.mergedLights = static_cast<u32>(j.frame.merged.size()), j.lightPicks = static_cast<u32>(j.frame.picks.size());
  if (s_lightLog != nullptr && j.drawable)
    log_lights(j, j.frame);
  if (s_lateCheck)
    late_compare(j);
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
        translate(p, j.draws[p.drawSlot], j.verts.get() + p.vertSlot);
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
    for (int i = 0; i < std::max(n, 1); ++i)
      std::thread([] {
        live::name_thread();
        worker();
      }).detach();
    return true;
  }();
  (void)started;
}

u8* packet_room(size_t need) {
  if (!s_job) {
    start_workers();
    s_jobOwned = live::take_job();
    s_job = s_jobOwned.get();
    if (!s_job->draws) {
      s_job->draws.reset(new DrawRec[MostDraws]);
      s_job->verts.reset(new Vertex[MostVerts]);
    }
    s_job->pending.store(1, std::memory_order_relaxed);
    s_job->epoch.store(0, std::memory_order_relaxed);
    s_job->translateUs.store(0, std::memory_order_relaxed);
    s_job->done = std::promise<void>();
    s_job->rules = s_job->done.get_future();
    s_epoch = s_baseDraw = s_baseVert = s_nextDraw = s_nextVert = 0;
  }
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
  p.rawOffset = static_cast<u32>(at);
  std::memcpy(room + at, vertices, rawBytes);
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
    look->captureMs = 0, look->rulesMs = 0, look->road = look->light = look->glow = look->calm = look->wet = 0;
    look->copy = s_copyNow;
    look->done = std::promise<void>();
    look->rules = look->done.get_future();
    look->done.set_value();
    live::submit_job(std::move(look));
    return;
  }
  const u32 draws = s_nextDraw - s_baseDraw;
  publish_chunk();
  auto job = std::move(s_jobOwned);
  s_job = nullptr;
  job->baseDraw = s_baseDraw, job->baseVert = s_baseVert, job->drawCount = draws;
  job->road = s_road, job->light = s_light, job->glow = s_glow, job->calm = s_calm, job->wet = s_wet;
  job->captureMs = s_captureMs;
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

void before_draw() noexcept {
#if defined(AURORA_RT_LIVE)
  if (s_rayed)
    return;
  if (g_gxState.projType == GX_PERSPECTIVE) {
    s_seenPerspective = true;
    ++s_viewDraws;
  } else if (s_seenPerspective)
    send_rays();
#endif
}

void efb_cleared() noexcept {
  if (!g_hook || s_rayed)
    return;
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
#if defined(AURORA_RT_LIVE)
  if (g_hook) {
    if (s_seenPerspective)
      send_rays();
    s_rayed = false;
    s_seenPerspective = false;
    forget_draws();
    s_viewDraws = 0;
    g_capturing = g_live && !live::failed();
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
