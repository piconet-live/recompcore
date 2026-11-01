#include "rt_capture.hpp"
#include "rt_format.hpp"

#include "recomp.hpp"
#include "../gfx/texture_convert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
std::vector<DrawRec> s_draws;
std::vector<Vertex> s_verts;
struct Tex {
  u32 w, h, wrapS, wrapT;
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
u32 add_texture(u32 slot) {
  if (slot >= MaxTextures)
    return UINT32_MAX;
  auto obj = g_gxState.textures[slot].texObj;
  if (!recomp::resolve_texture_data(obj) || obj.data == nullptr)
    return UINT32_MAX;
  const u32 w = obj.width(), h = obj.height(), fmt = obj.format();
  GXTlutObj_ tlut{};
  const bool palette = fmt == GX_TF_C4 || fmt == GX_TF_C8 || fmt == GX_TF_C14X2;
  if (palette && !recomp::resolve_texture_tlut(slot, &tlut))
    return UINT32_MAX;
  const auto key = std::make_tuple(obj.data, w, h, fmt, palette ? tlut.data : nullptr);
  if (const auto it = s_texIndex.find(key); it != s_texIndex.end())
    return it->second;
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
  Tex t{w, h, static_cast<u32>(obj.wrap_s()), static_cast<u32>(obj.wrap_t()), {}};
  t.rgba.assign(conv.data.data(), conv.data.data() + static_cast<size_t>(w) * h * 4u);
  s_texs.push_back(std::move(t));
  const u32 index = static_cast<u32>(s_texs.size() - 1);
  s_texIndex[key] = index;
  return index;
}
float dot4(const Vec4<float>& m, float x, float y, float z, float w) { return m[0] * x + m[1] * y + m[2] * z + m[3] * w; }
} // namespace

void start() noexcept {
  const char* on = std::getenv("AURORA_RT");
  g_neutral = on != nullptr && on[0] == '1';
  if (const char* path = std::getenv("AURORA_RT_CAPTURE"))
    s_path = path;
  if (const char* at = std::getenv("AURORA_RT_CAPTURE_AT"))
    s_at = std::strtoull(at, nullptr, 10);
  const char* noHud = std::getenv("AURORA_RT_NO_HUD");
  g_noHud = noHud != nullptr && noHud[0] == '1';
  if (const char* at = std::getenv("AURORA_RT_CAPTURE_COPY"))
    s_atCopy = std::strtoull(at, nullptr, 10);
  g_capturing = !s_path.empty() && s_at == 0 && s_atCopy == 0;
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

void capture_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* vertices) noexcept {
  if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS || vtxCount < 3)
    return;
  auto& st = g_gxState;
  DrawRec d{};
  std::memcpy(d.proj, &st.proj, sizeof d.proj);
  const auto& vp = st.logicalViewport;
  d.viewport[0] = vp.left, d.viewport[1] = vp.top, d.viewport[2] = vp.width, d.viewport[3] = vp.height,
  d.viewport[4] = vp.znear, d.viewport[5] = vp.zfar;
  d.projType = st.projType;
  d.stages = st.numTevStages;
  d.blendMode = st.blendMode, d.blendSrc = st.blendFacSrc, d.blendDst = st.blendFacDst;
  d.depthUpdate = st.depthUpdate, d.depthCompare = st.depthCompare, d.cull = st.cullMode, d.colorUpdate = st.colorUpdate;
  d.prim = prim, d.vtxCount = vtxCount;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      d.regs[i][j] = st.colorRegs[i][j];
  for (int i = 0; i < 4; ++i) {
    d.swap[i][0] = st.tevSwapTable[i].red, d.swap[i][1] = st.tevSwapTable[i].green, d.swap[i][2] = st.tevSwapTable[i].blue,
    d.swap[i][3] = st.tevSwapTable[i].alpha;
  }
  for (u32 i = 0; i < d.stages && i < 16; ++i) {
    const auto& g = st.tevStages[i];
    auto& s = d.stage[i];
    s.tc = g.texCoordId, s.map = g.texMapId;
    s.tex = g.texMapId != GX_TEXMAP_NULL ? add_texture(g.texMapId) : UINT32_MAX;
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
  const auto& vf = st.vtxFmts[fmt];
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
    if (st.tevStages[i].channelId == GX_COLOR1A1 || st.tevStages[i].channelId == GX_COLOR1 || st.tevStages[i].channelId == GX_ALPHA1) {
      chan = 1;
      break;
    }
  d.lit = st.colorChannelConfig[GX_COLOR0 + chan].lightingEnabled;

  const auto read_vertex = [&](const u8* v) {
    Vertex out{};
    out.draw = static_cast<u32>(s_draws.size());
    const u32 mi = fields[GX_VA_PNMTXIDX].type != GX_NONE ? v[fields[GX_VA_PNMTXIDX].offset] / 3u : st.currentPnMtx;
    const auto& mtx = st.pnMtx[std::min<u32>(mi, MaxPnMtx - 1)];
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
          const auto& light = st.lights[l];
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
        const auto& m = st.texMtxs[(g.mtx - GX_TEXMTX0) / 3];
        o[0] = dot4(m.m0, in[0], in[1], in[2], 1.f), o[1] = dot4(m.m1, in[0], in[1], in[2], 1.f);
        o[2] = g.type == GX_TG_MTX3x4 ? dot4(m.m2, in[0], in[1], in[2], 1.f) : 1.f;
      }
      if (g.postMtx != GX_PTIDENTITY && g.postMtx >= GX_PTTEXMTX0 && (g.postMtx - GX_PTTEXMTX0) / 3 < MaxPTTexMtx) {
        const auto& m = st.ptTexMtxs[(g.postMtx - GX_PTTEXMTX0) / 3];
        const float a = o[0], b = o[1], c = o[2];
        o[0] = dot4(m.m0, a, b, c, 1.f), o[1] = dot4(m.m1, a, b, c, 1.f), o[2] = dot4(m.m2, a, b, c, 1.f);
      }
      const float q = g.type == GX_TG_MTX3x4 && o[2] != 0.f ? o[2] : 1.f;
      out.uv[t][0] = o[0] / q, out.uv[t][1] = o[1] / q;
    }
    return out;
  };

  d.firstVertex = static_cast<u32>(s_verts.size());
  std::vector<Vertex> strip(vtxCount);
  for (u32 i = 0; i < vtxCount; ++i)
    strip[i] = read_vertex(vertices + i * stride);
  const auto tri = [&](u32 a, u32 b, u32 c) {
    s_verts.push_back(strip[a]);
    s_verts.push_back(strip[b]);
    s_verts.push_back(strip[c]);
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
  d.vertices = static_cast<u32>(s_verts.size()) - d.firstVertex;
  s_draws.push_back(d);
}

void begin_copy(std::uint64_t copy) noexcept {
  if (s_atCopy != 0 && copy == s_atCopy) {
    g_capturing = true;
    s_atCopy = 0;
  }
}

void frame_end() noexcept {
  s_seenPerspective = false;
  if (g_capturing) {
    g_capturing = false;
    if (FILE* f = std::fopen(s_path.c_str(), "wb")) {
      const u32 head[4] = {0x33435452u, static_cast<u32>(s_draws.size()), static_cast<u32>(s_verts.size()), static_cast<u32>(s_texs.size())};
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
