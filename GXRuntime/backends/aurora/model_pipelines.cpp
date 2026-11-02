// SPDX-License-Identifier: GPL-3.0-or-later
// The pipelines a game's models will want, compiled when their files are read from the disc rather than at their first
// draw (a draw whose pipeline is still compiling is left out of the picture: a piece "loads in").
//
// The source is the files themselves: an Amusement Vision model file (.gma, .gma.lz) says for each mesh its render flags,
// vertex format, blend factors and texture layers, and the game's renderer turns exactly those into TEV state
// (F-Zero GX: mesh setup fn_80075908, layer loop fn_80076CD4, plain layer fn_80075D48, reflection fn_80076238,
// world specular fn_8007698C, alpha mask and bump in the loop, lit and unlit channels fn_80077384 and fn_80077488, blend
// fn_80075C78, post stages fn_80077654 and fn_80077714). This walks the same steps on the file's bytes and asks Aurora to
// compile what comes out. A course file (COLI_COURSE) gives the fog type at its header's +0x84.
//
// What the file cannot say is tried in every form the renderer has: cull mode, depth update, destination alpha, scene
// alpha. A form no draw ever wants costs a little compile time and nothing else.
// The post stages (a tint the game sets around a draw: fn_80074A8C and its kin, a machine's flash among them) are not in any
// file, and nothing in a file says whose model it is, so every model gets those forms.
//
// Order is what makes this work in a load's time: nearly all the cost of a pipeline is its shader (about 200 ms uncached,
// against about 1.5 ms for one more pipeline of a shader already made). So the first form of each shader goes first, then the
// rest (Tier below). The tiers only say when: no form is left out for any model. Only new shaders get every compile thread;
// the rest runs on Behind threads, since it runs during the race and a draw's own pipeline wants a free thread.
//
// Files are told apart by their contents, not their names: LZ, a U8 archive (a machine's models come in one), a model file.
// Only the course file goes by its name, the name the game itself opens: it has no mark of its own.
// Order of work: files the game is reading now, each tier, then the rest of the disc (aurora_present_model_file's later: only
// when ray tracing is on or a test asks), each only while the compile queue is short.
// Nothing here reads or writes the guest: the bytes are the disc's, copied on the disc thread. Effects and the HUD are not models,
// so they stay as they were: their pipelines compile at their first draw.
#include "gxruntime/aurora_present.h"

#include "gfx/pipeline_cache.hpp"
#include "gx/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "model_pipelines.h"

namespace {
using namespace aurora::gx;
using Clock = std::chrono::steady_clock;

struct File {
  std::string path;
  std::vector<uint8_t> bytes;
};
std::mutex s_lock;
std::condition_variable s_wake;
std::deque<File> s_files, s_disc_files;
uint64_t s_disc_bytes = 0;
// Tier: First a new shader, plain form, the scene's target. Same: more pipelines of known shaders. PostSmall: post stages of
// models in the 16-bit vertex form (machines are; only an order, a guess that costs nothing if wrong). Rest: all else.
// DiscFirst: a new shader from the disc walk, not from a file the game reads.
enum Tier { First, Same, PostSmall, DiscFirst, Rest, Tiers };
bool s_from_disc = false;  // the file being read comes from the disc walk
uint64_t s_disc_read = 0;
struct Form {
  PipelineConfig config;
  bool scene, other;
};
std::deque<Form> s_tiers[Tiers];
std::set<std::vector<uint8_t>> s_shaders;
bool s_quit = false, s_aurora_up = false;
std::thread s_thread;
struct Base {
  PipelineConfig config;  // fog left at none
  int post;
  bool small;
};
std::vector<Base> s_bases;
std::set<std::vector<uint8_t>> s_seen;
std::set<uint8_t> s_fogs{0};
uint64_t s_meshes = 0, s_meshes_left_out = 0, s_configs = 0;
FILE* s_dump = nullptr;

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint16_t be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t le32(const uint8_t* p) { return uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0]; }

// Amusement Vision's LZSS: sizes, then flag bytes; a 4096-byte ring from 4078
std::vector<uint8_t> unpack(const std::vector<uint8_t>& in) {
  std::vector<uint8_t> out;
  if (in.size() < 8) return out;
  const uint32_t size = le32(in.data() + 4);
  if (size > 256u * 1024 * 1024) return out;
  out.reserve(size);
  uint8_t ring[4096] = {};
  uint32_t r = 4078, flags = 0;
  size_t p = 8;
  while (p < in.size() && out.size() < size) {
    flags >>= 1;
    if (!(flags & 0x100)) flags = in[p++] | 0xFF00;
    if (p >= in.size()) break;
    if (flags & 1) {
      out.push_back(ring[r] = in[p++]);
      r = (r + 1) & 4095;
    } else {
      if (p + 1 >= in.size()) break;
      const uint32_t at = in[p] | (in[p + 1] & 0xF0u) << 4, n = (in[p + 1] & 0xFu) + 3;
      p += 2;
      for (uint32_t k = 0; k < n; ++k) {
        const uint8_t b = ring[(at + k) & 4095];
        out.push_back(b);
        ring[r] = b;
        r = (r + 1) & 4095;
      }
    }
  }
  return out;
}

using CArg = GXTevColorArg;
using AArg = GXTevAlphaArg;
struct Build {
  ShaderConfig& c;
  void gen(u32 i, GXTexGenType type, GXTexGenSrc src, GXTexMtx mtx, bool norm, GXPTTexMtx post) {
    c.tcgs[i] = TcgConfig{.type = type, .src = src, .mtx = mtx, .postMtx = post, .normalize = norm};
  }
  TevStage& stage(CArg a, CArg b, CArg cc, CArg d, AArg e, AArg f, AArg g, AArg h, int coord, int map) {
    auto& s = c.tevStages[c.tevStageCount++];
    s = {};
    s.colorPass = {a, b, cc, d};
    s.alphaPass = {e, f, g, h};
    s.texCoordId = map == 255 ? GX_TEXCOORD0 : static_cast<GXTexCoordID>(coord);
    s.texMapId = map == 255 ? GX_TEXMAP_NULL : static_cast<GXTexMapID>(map);
    const auto reads = [](auto x, auto y) { return x == y; };
    const bool raster = reads(a, GX_CC_RASC) || reads(b, GX_CC_RASC) || reads(cc, GX_CC_RASC) || reads(d, GX_CC_RASC) ||
                        reads(a, GX_CC_RASA) || reads(b, GX_CC_RASA) || reads(cc, GX_CC_RASA) || reads(d, GX_CC_RASA) ||
                        reads(e, GX_CA_RASA) || reads(f, GX_CA_RASA) || reads(g, GX_CA_RASA) || reads(h, GX_CA_RASA);
    s.channelId = raster ? GX_COLOR0A0 : GX_COLOR_NULL;
    return s;
  }
};

struct Mesh {
  uint32_t flags, attributes, blend;
  const uint8_t* header;
  const uint8_t* layers;
  uint32_t layer_count;
  uint32_t model;
};

// The TEV stages of fn_80075908 for one mesh. plain: lbl_801A6D58, the renderer's "no textures" switch. post: bit 0
// fn_80077654, bit 1 fn_80077714.
void stages(const Mesh& m, ShaderConfig& c, bool plain, int post) {
  Build b{c};
  const uint32_t flags = m.flags;
  CArg cin = GX_CC_RASC;
  AArg ain = GX_CA_RASA;
  if ((flags & 1) && !(flags & 0x100)) {
    cin = GX_CC_C0;
    ain = GX_CA_A0;
  }
  u32 t = 0, map = 1;
  u32 made = 0;
  const uint32_t n = m.header[0x12];
  if (!(flags & 0x80) && !plain) {
    for (u32 i = 0; i < n && i < 3; ++i) {
      const uint32_t index = be16(m.header + 0x16 + i * 2);
      if (index >= m.layer_count) continue;
      const uint8_t* L = m.layers + index * 0x20;
      const uint32_t w = be32(L), word = be32(L + 0x10);
      const GXTexGenSrc src = static_cast<GXTexGenSrc>(GX_TG_TEX0 + i);
      if ((w & 0x50A003) == 0) {
        const uint32_t scroll = word >> 8 & 3;
        const auto mtx = !(w & 0x20000) ? GX_IDENTITY : scroll ? static_cast<GXTexMtx>(scroll * 3 + 0x30) : GX_TEXMTX2;
        b.gen(t, GX_TG_MTX2x4, src, mtx, false, GX_PTIDENTITY);
        const uint32_t cop = word & 0xF, aop = word >> 4 & 0xF;
        TevStage* s;
        if (cop == 4) {
          s = &b.stage(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
          s->colorOp.outReg = GX_TEVREG2;
          s = &b.stage(cin, GX_CC_C2, GX_CC_TEXA, GX_CC_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        } else if (cop == 1 || cop == 2) {
          s = &b.stage(GX_CC_ZERO, GX_CC_TEXC, GX_CC_ONE, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
          if (cop == 2) s->colorOp.op = GX_TEV_SUB;
        } else if (cop == 3) {
          s = &b.stage(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        } else {
          s = &b.stage(GX_CC_ZERO, GX_CC_TEXC, cin, GX_CC_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        }
        switch (aop) {
        case 1: case 2: s->alphaPass = {GX_CA_TEXA, GX_CA_ZERO, GX_CA_ZERO, ain}; if (aop == 2) s->alphaOp.op = GX_TEV_SUB; break;
        case 0: s->alphaPass = {GX_CA_ZERO, GX_CA_TEXA, ain, GX_CA_ZERO}; break;
        case 4: s->alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA}; break;
        case 5: s->alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST}; break;
        default: break;
        }
        ++t;
      } else if (w & 0x2000) {
        b.gen(t, GX_TG_MTX2x4, src, (w & 0x20000) ? GX_TEXMTX2 : GX_IDENTITY, false, GX_PTIDENTITY);
        auto& s = b.stage(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, cin, GX_CA_ZERO, GX_CA_TEXA, ain, GX_CA_ZERO, t, map);
        s.tevSwapTex = (w & 0xC0000) ? GX_TEV_SWAP2 : GX_TEV_SWAP1;
        ++t;
      } else if (w & 2) {
        b.gen(t, GX_TG_MTX2x4, src, GX_IDENTITY, false, GX_PTIDENTITY);
        auto& s0 = b.stage(GX_CC_TEXC, GX_CC_ZERO, GX_CC_ZERO, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        s0.colorOp.clamp = false;
        const u32 bump_from = t;
        ++t;
        b.gen(t, GX_TG_BUMP0, static_cast<GXTexGenSrc>(GX_TG_TEXCOORD0 + bump_from), GX_IDENTITY, false, GX_PTIDENTITY);
        auto& s1 = b.stage(GX_CC_TEXC, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV, t, map);
        s1.colorOp.op = GX_TEV_SUB;
        ++t;
      } else if (w & 1) {
        b.gen(t, GX_TG_MTX3x4, GX_TG_NRM, GX_TEXMTX0, true, static_cast<GXPTTexMtx>(64));
        TevStage* s;
        if (w & 0x400000) {
          s = &b.stage(GX_CC_ZERO, GX_CC_TEXC, cin, GX_CC_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        } else {
          s = &b.stage(GX_CC_ZERO, GX_CC_TEXC, GX_CC_KONST, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
          s->kcSel = GX_TEV_KCSEL_K0;
        }
        ++t;
      } else {
        b.gen(t, GX_TG_MTX3x4, GX_TG_NRM, GX_TEXMTX0, true, static_cast<GXPTTexMtx>(70));
        auto& s0 = b.stage(GX_CC_ZERO, GX_CC_TEXC, GX_CC_KONST, GX_CC_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, 0);
        s0.kcSel = GX_TEV_KCSEL_K1;
        s0.colorOp.outReg = GX_TEVREG2;
        s0.alphaOp.outReg = GX_TEVREG2;
        ++t;
        b.gen(t, GX_TG_MTX3x4, GX_TG_NRM, GX_TEXMTX0, true, static_cast<GXPTTexMtx>(67));
        b.stage(GX_CC_ZERO, GX_CC_TEXC, GX_CC_C2, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, t, map);
        ++t;
      }
      ++map;
      ++made;
      cin = GX_CC_CPREV;
      ain = GX_CA_APREV;
    }
  }
  if (made == 0) {
    b.stage(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, cin, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, ain, 0, 255);
    cin = GX_CC_CPREV;
    ain = GX_CA_APREV;
  }
  if (post & 1) {
    auto& s = b.stage(GX_CC_ZERO, GX_CC_CPREV, GX_CC_KONST, GX_CC_ZERO, GX_CA_ZERO, GX_CA_APREV, GX_CA_KONST, GX_CA_ZERO, 0, 255);
    s.kcSel = GX_TEV_KCSEL_K2;
    s.kaSel = GX_TEV_KASEL_K2_A;
  }
  if (post & 2) {
    auto& s = b.stage(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST, GX_CA_APREV, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST, 0, 255);
    s.kcSel = GX_TEV_KCSEL_K3;
    s.kaSel = GX_TEV_KASEL_K3_A;
  }
}

void attrs(const Mesh& m, ShaderConfig& c) {
  const uint32_t va = be32(m.header + 0x1C);
  const bool small = m.model & 1;
  u8 off = 0;
  const auto add = [&](int i, u8 cnt, u8 comp, u8 frac, u8 size, bool nbt3 = false) {
    c.attrs[i] = AttrConfig{.attrType = GX_DIRECT, .cnt = cnt, .compType = comp, .offset = off, .stride = 0, .frac = frac, .le = false, .nbt3 = nbt3};
    off = static_cast<u8>(off + size);
  };
  for (auto& a : c.attrs) a = {};
  if ((va & 1) || (m.model & 4)) add(GX_VA_PNMTXIDX, 1, 0, 0, 1);  // (a stitched model: a matrix index on every vertex)
  add(GX_VA_POS, 3, small ? GX_S16 : GX_F32, small ? 13 : 0, small ? 6 : 12);
  if (va & 0x02000000) add(GX_VA_NRM, 9, small ? GX_S16 : GX_F32, small ? 14 : 0, small ? 18 : 36);
  else if (va & 0x400) add(GX_VA_NRM, 3, small ? GX_S16 : GX_F32, small ? 14 : 0, small ? 6 : 12);
  if (va & 0x800) add(GX_VA_CLR0, 1, GX_RGBA8, 0, 4);
  if (va & 0x2000) add(GX_VA_TEX0, 2, small ? GX_S16 : GX_F32, small ? 13 : 0, small ? 4 : 8);
  if (va & 0x4000) add(GX_VA_TEX1, 2, small ? GX_S16 : GX_F32, small ? 13 : 0, small ? 4 : 8);
  if (va & 0x8000) add(GX_VA_TEX2, 2, small ? GX_S16 : GX_F32, small ? 13 : 0, small ? 4 : 8);
  c.vtxStride = off;
}

void ask(const Form& f) {
  static const bool count_only = std::getenv("AURORA_TEST_COUNT_ONLY") != nullptr;  // a test: how many forms, none compiled
  if (!count_only) aurora::gfx::precompile_pipeline(f.config, f.scene, f.other);
  ++s_configs;
  if (s_dump) std::fwrite(&f.config, sizeof(f.config), 1, s_dump);
}

// A form with this fog goes in its tier: its first of a shader (destination alpha off or on makes a different one) for the
// scene's target at the front, the other target's copy last
void plan(const PipelineConfig& base, uint8_t fog, int post, bool small) {
  PipelineConfig c = base;
  c.shaderConfig.fogType = fog;
  std::vector<uint8_t> shader(sizeof(c.shaderConfig) + 1);
  std::memcpy(shader.data(), &c.shaderConfig, sizeof(c.shaderConfig));
  shader.back() = c.dstAlpha == UINT32_MAX;
  const bool first = s_shaders.insert(std::move(shader)).second;
  const Tier tier = post != 0 ? (small && c.dstAlpha != UINT32_MAX ? PostSmall : Rest)
                    : first && c.dstAlpha != UINT32_MAX ? (s_from_disc ? DiscFirst : First)
                    : first || c.dstAlpha == 255 ? Same
                                                 : Rest;
  s_tiers[tier].push_back(Form{c, true, false});
  s_tiers[Rest].push_back(Form{c, false, true});
}

void forms(const Mesh& m) {
  ++s_meshes;
  const uint32_t va = be32(m.header + 0x1C);
  if (!(va & 0x200) || (va & ~0x0200EE01u)) {
    ++s_meshes_left_out;
    return;
  }
  const uint32_t flags = m.flags;
  // (seen: a mesh with the no-fog flag drawn with fog on, and one without it drawn with fog off: both forms for all)
  ShaderConfig shader{};
  shader.tevSwapTable = {TevSwap{GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE, GX_CH_ALPHA},
                         TevSwap{GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE, GX_CH_RED},
                         TevSwap{GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE, GX_CH_GREEN},
                         TevSwap{GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE, GX_CH_BLUE}};
  attrs(m, shader);
  const auto blend_src = (flags & 0x20) ? static_cast<GXBlendFactor>(m.blend & 0xF) : GX_BL_SRCALPHA;
  const auto blend_dst = (flags & 0x40) ? static_cast<GXBlendFactor>(m.blend >> 4 & 0xF) : GX_BL_INVSRCALPHA;
  {
    const bool plain = false;
    for (int post = 0; post < 4; ++post) {
      ShaderConfig c = shader;
      stages(m, c, plain, post);
      // the texture coordinates in use; what stands beyond them is not in a real key either
      const bool lit = !(flags & 1), vertex = flags & 0x100;
      bool raster = false;
      for (u32 i = 0; i < c.tevStageCount; ++i) raster |= c.tevStages[i].channelId != GX_COLOR_NULL;
      for (const int scene_alpha : {0, 1}) {  // fn_80077384's and fn_80077488's scene alpha test: the vertex alpha or none
        if (scene_alpha && !vertex) continue;
        if (raster) {
          c.colorChannels[0] = lit ? ColorChannelConfig{vertex ? GX_SRC_VTX : GX_SRC_REG, GX_SRC_REG, GX_DF_CLAMP, GX_AF_SPOT, true}
                                   : ColorChannelConfig{.matSrc = vertex ? GX_SRC_VTX : GX_SRC_REG};
          c.colorChannels[2] = ColorChannelConfig{.matSrc = (vertex && scene_alpha) ? GX_SRC_VTX : GX_SRC_REG};
        }
        for (const auto cull : {GX_CULL_NONE, GX_CULL_FRONT, GX_CULL_BACK}) {
          for (const bool depth : {true, false}) {
            for (const uint32_t dst : {255u, 128u, UINT32_MAX}) {
              PipelineConfig p{};
              p.shaderConfig = c;
              p.depthFunc = GX_LEQUAL;
              p.cullMode = cull;
              p.blendMode = GX_BM_BLEND;
              p.blendFacSrc = blend_src;
              p.blendFacDst = blend_dst;
              p.blendOp = GX_LO_CLEAR;
              p.dstAlpha = dst;
              p.polygonOffsetBits = p.polygonOffsetScaleBits = p.polygonOffsetClampBits = 0;
              p.depthCompare = true;
              p.depthUpdate = depth;
              p.alphaUpdate = true;
              p.colorUpdate = true;
              std::vector<uint8_t> key(sizeof(p));
              std::memcpy(key.data(), &p, sizeof(p));
              if (s_seen.insert(std::move(key)).second) {
                s_bases.push_back(Base{p, post, (m.model & 1) != 0});
                for (const uint8_t fog : s_fogs) plan(p, fog, post, (m.model & 1) != 0);
              }
            }
          }
        }
      }
    }
  }
}

void read_models(const std::vector<uint8_t>& d) {
  if (d.size() < 8) return;
  const uint32_t count = be32(d.data()), base = be32(d.data() + 4);
  for (uint32_t i = 0; i < count && 16ull + i * 8 <= d.size(); ++i) {
    const uint32_t at = be32(d.data() + 8 + i * 8);
    if (at == UINT32_MAX) continue;
    const size_t s = size_t(base) + at;
    if (s + 0x40 > d.size() || std::memcmp(d.data() + s, "GCMF", 4) != 0) continue;
    Mesh m{};
    m.model = be32(d.data() + s + 4);
    m.layer_count = be16(d.data() + s + 0x18);
    const uint32_t meshes = uint32_t(be16(d.data() + s + 0x1A)) + be16(d.data() + s + 0x1C);
    size_t p = s + be32(d.data() + s + 0x20);
    m.layers = d.data() + s + 0x40;
    if (s + 0x40 + size_t(m.layer_count) * 0x20 > d.size()) continue;
    for (uint32_t k = 0; k < meshes && p + 0x60 <= d.size(); ++k) {
      m.header = d.data() + p;
      m.flags = be32(m.header);
      m.blend = be32(m.header + 0x40);
      const uint32_t va = be32(m.header + 0x1C);
      if (va & 0x800) m.flags |= 0x100;
      if (m.header[0x12] == 0) m.flags |= 0x80;
      forms(m);
      const uint8_t groups = m.header[0x13];
      p += 0x60;
      if (groups & 1) p += be32(m.header + 0x28);
      if (groups & 2) p += be32(m.header + 0x2C);
      if ((groups & 0xC) && p + 0x20 <= d.size()) {
        const uint32_t a = be32(d.data() + p + 8), b = be32(d.data() + p + 12);
        p += 0x20;
        if (groups & 4) p += a;
        if (groups & 8) p += b;
      }
    }
  }
}

bool packed(const std::vector<uint8_t>& d) {
  // LZ: the packed size, the whole size, then the stream to the file's end (an archive may pad what it holds)
  if (d.size() < 16) return false;
  const uint64_t stream = uint64_t(le32(d.data())) + 8;
  return stream <= d.size() && d.size() - stream < 32 && le32(d.data() + 4) >= stream / 16;
}

void read_contents(const std::vector<uint8_t>& d, int depth) {
  if (depth > 4 || d.size() < 16) return;
  if (packed(d)) {
    read_contents(unpack(d), depth + 1);
    return;
  }
  if (be32(d.data()) == 0x55AA382D) {  // U8: nodes of 12 bytes from the root's; the root's size is how many
    const uint64_t root = be32(d.data() + 4);
    if (root + 12 > d.size()) return;
    const uint64_t nodes = be32(d.data() + root + 8);
    if (nodes == 0 || root + nodes * 12 > d.size()) return;
    for (uint64_t i = 1; i < nodes; ++i) {
      const uint8_t* n = d.data() + root + i * 12;
      if (n[0] != 0) continue;  // a directory
      const uint64_t at = be32(n + 4), size = be32(n + 8);
      if (at + size > d.size()) continue;
      read_contents(std::vector<uint8_t>(d.begin() + at, d.begin() + at + size), depth + 1);
    }
    return;
  }
  read_models(d);
}

void read_file(const File& f) {
  if (f.path.find("COLI_COURSE") != std::string::npos) {
    const std::vector<uint8_t> whole = packed(f.bytes) ? unpack(f.bytes) : f.bytes;
    if (whole.size() < 0x88) return;
    const uint32_t at = be32(whole.data() + 0x84);
    if (at >= whole.size()) return;
    const uint8_t fog = whole[at];
    if (s_fogs.insert(fog).second) {
      std::fprintf(stderr, "[pipelines] fog type %u from %s\n", fog, f.path.c_str());
      for (const Base& b : s_bases) plan(b.config, fog, b.post, b.small);
    }
    return;
  }
  const uint64_t meshes = s_meshes;
  const size_t first = s_tiers[First].size();
  read_contents(f.bytes, 0);
  if (s_meshes != meshes)
    std::fprintf(stderr, "[pipelines] file %s: %llu meshes, %zu new shaders\n", f.path.c_str(),
                 (unsigned long long)(s_meshes - meshes), s_tiers[First].size() - first);
}

bool tiers_empty() {
  return std::all_of(std::begin(s_tiers), std::end(s_tiers), [](const auto& t) { return t.empty(); });
}

bool near_tiers_empty() {
  return s_tiers[First].empty() && s_tiers[Same].empty() && s_tiers[PostSmall].empty() && s_tiers[DiscFirst].empty();
}

// The nearest tier's next few; false when none is left
bool ask_next(size_t count) {
  for (auto& tier : s_tiers) {
    if (tier.empty()) continue;
    for (; count != 0 && !tier.empty(); --count) {
      ask(tier.front());
      tier.pop_front();
    }
    return true;
  }
  return false;
}

void run() {
#ifdef _WIN32
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
  {
    std::unique_lock g{s_lock};
    s_wake.wait(g, [] { return s_aurora_up || s_quit; });
  }
  if (const char* dump = std::getenv("AURORA_TEST_MODEL_DUMP")) s_dump = std::fopen(dump, "wb");
  const auto t0 = Clock::now();
  static const unsigned behind = [] {
    const char* v = std::getenv("AURORA_PIPELINE_BEHIND");
    return v != nullptr && std::atoi(v) > 0 ? unsigned(std::atoi(v)) : 3u;
  }();
  constexpr size_t Short = 32;  // the compile queue counts as short: later work never stands in front of a file being read
  bool told = true, told_first = true, walk_told = false;
  for (;;) {
    File f;
    bool have = false;
    {
      std::unique_lock g{s_lock};
      if (s_quit) break;
      if (!s_files.empty()) {
        aurora::gfx::set_background_compile_threads(UINT_MAX);
        f = std::move(s_files.front());
        s_files.pop_front();
        have = true;
      } else {
        if (s_tiers[First].empty() && aurora::gfx::pipelines_waiting() == 0 && !told_first) {
          told_first = true;
          std::fprintf(stderr, "[pipelines] every shader so far made: %zu shaders, %.0f ms after the first file\n", s_shaders.size(),
                       std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        aurora::gfx::set_background_compile_threads(s_tiers[First].empty() ? behind : UINT_MAX);
        const bool more = !tiers_empty() || !s_disc_files.empty();
        const size_t waiting = aurora::gfx::pipelines_waiting();
        if (waiting >= Short || (waiting != 0 && !more)) {
          s_wake.wait_for(g, std::chrono::milliseconds(20), [] { return s_quit || !s_files.empty(); });
          continue;
        }
        if (!s_disc_files.empty() && near_tiers_empty()) {
          f = std::move(s_disc_files.front());
          s_disc_files.pop_front();
          s_disc_bytes -= f.bytes.size();
          g.unlock();
          s_wake.notify_all();  // (the disc walk waits for room)
          if (f.path.empty()) {  // (the walk's end)
            std::fprintf(stderr, "[pipelines] whole disc read: %llu files, %zu shaders asked, %zu fog types, %llu meshes, %.0f ms after the first file\n",
                         (unsigned long long)s_disc_read, s_shaders.size(), s_fogs.size(), (unsigned long long)s_meshes,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            walk_told = true;
          } else {
            s_from_disc = true;
            read_file(f);
            s_from_disc = false;
            ++s_disc_read;
          }
          told = false;
          continue;
        }
        if (walk_told && s_tiers[DiscFirst].empty() && waiting == 0) {
          walk_told = false;
          std::fprintf(stderr, "[pipelines] whole disc: every shader made, %.0f ms after the first file\n",
                       std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        if (!tiers_empty()) {
          g.unlock();
          ask_next(Short);
          told = false;
          continue;
        }
        {
          if (!told) {
            told = true;
            std::fprintf(stderr, "[pipelines] models: %llu meshes (%llu left out), %llu forms, %zu pipelines compiled, nothing waiting, %.0f ms after the first file\n",
                         (unsigned long long)s_meshes, (unsigned long long)s_meshes_left_out, (unsigned long long)s_configs,
                         aurora::gfx::pipelines_compiled(),
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            if (s_dump) std::fflush(s_dump);
          }
          s_wake.wait(g, [] { return s_quit || !s_files.empty() || !s_disc_files.empty(); });
          continue;
        }
      }
    }
    if (have) {
      read_file(f);
      told = false;
      told_first = false;
    }
  }
  if (s_dump) std::fclose(s_dump);
  s_dump = nullptr;
}
}  // namespace

extern "C" void aurora_present_model_file(const char* path, const uint8_t* bytes, uint64_t size, int later) {
  static const bool off = std::getenv("AURORA_TEST_NO_MODELS") != nullptr;  // a test: today's behaviour, to compare
  if (off) return;
  File f;
  f.path = path;
  f.bytes.assign(bytes, bytes + size);
  {
    std::unique_lock g{s_lock};
    if (later) {
      // The disc walk's thread waits here, so a whole disc never stands in memory
      s_wake.wait(g, [] { return s_quit || s_disc_bytes < 64u * 1024 * 1024; });
      if (s_quit) return;
      s_disc_bytes += f.bytes.size();
      s_disc_files.push_back(std::move(f));
    } else {
      s_files.push_back(std::move(f));
    }
    if (!s_thread.joinable()) s_thread = std::thread(run);
  }
  s_wake.notify_all();
}

void model_pipelines_aurora_up() {
  {
    std::lock_guard g{s_lock};
    s_aurora_up = true;
  }
  s_wake.notify_all();
}

void model_pipelines_stop() {
  {
    std::lock_guard g{s_lock};
    s_quit = true;
  }
  s_wake.notify_all();
  if (s_thread.joinable()) s_thread.join();
}
