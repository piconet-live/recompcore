#pragma once
// What Aurora's own looks put into a pipeline's key beyond the game's state, as functions of that state alone, so the draw path
// (gx.cpp populate_pipeline_config, regs.cpp, soft_cards.cpp) and the load-time planner (backends/aurora/model_pipelines.cpp)
// work out the same key. A key the planner compiles in the wrong look is never drawn: with the ray-traced look on, every key
// carries a card range and the game's lights are off, so a planner blind to these compiled nothing a race could use.
//
// Header only: the offline pipeline census (backends/aurora/tools) builds the planner without the rest of Aurora.

#include "pipeline.hpp"

namespace aurora::gx {
namespace rt {
// A stage is a reflection layer when it adds or multiplies a texture looked up by the normal (a sphere map), with a
// constant strength: fn_80075908's reflection layers, for any model.
inline bool is_reflection_stage(const TevStage& stage, const std::array<TcgConfig, MaxTexCoord>& tcgs) noexcept {
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

// AURORA_RT=1/2: the game's fake reflection stages pass through (rays stand in for them)
inline void neutralise(ShaderConfig& config) noexcept {
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
} // namespace rt

// Config version 15: a swap table entry no stage selects is left out of the key (as 14 left out what no stage input reads). The game
// leaves in the table whatever the last draw that used it set (a menu's RRRB), and only a draw could say which: the same material
// made another pipeline after such a draw (688 of 2,356 recorded keys of the ray-traced look, 2026-11-05). The shader reads the table
// only through the stages' selections (shader.cpp).
inline void unread_swaps_out(ShaderConfig& c) noexcept {
  bool read[MaxTevSwap] = {};
  for (u32 i = 0; i < c.tevStageCount && i < MaxTevStages; ++i) {
    read[c.tevStages[i].tevSwapRas] = true;
    read[c.tevStages[i].tevSwapTex] = true;
  }
  for (u32 i = 0; i < MaxTevSwap; ++i)
    if (!read[i])
      c.tevSwapTable[i] = TevSwap{};
}

namespace look {
// The switches that shape keys, read once by Aurora (current(), gx.cpp)
struct Look {
  bool neutral = false;       // AURORA_RT=1/2 (rt::g_neutral)
  bool gameLightOff = false;  // AURORA_RT_GAME_LIGHT=0 (rt::g_gameLightOff): the colour channels' lighting off in the state
  bool soft = false;          // AURORA_RT_SOFT_CARDS (soft::on())
  bool cards = false;         // AURORA_RT_CARDS with the raw picture (gx.cpp cards_on())
  int cardLevel = 0;          // AURORA_RT_CARDS's number (2: the alpha alone, a check)
  bool operator==(const Look&) const = default;
};
Look current() noexcept;

// gx.cpp card_range: 1 the draw's alpha in the game's range, 2 its colour too (a blended card)
inline u8 card_range(GXBlendMode mode, GXBlendFactor src, GXBlendFactor dst, int level) noexcept {
  const bool blended = mode == GX_BM_BLEND && !(src == GX_BL_ONE && dst == GX_BL_ZERO);
  return blended && level != 2 ? 2 : 1;
}

// soft_cards.cpp kind: 0 none, 1 the alpha, 2 the colour. keepsDstAlpha: the EFB has alpha, alpha is written and a destination
// alpha constant is set (in a key: alphaUpdate && dstAlpha != UINT32_MAX)
inline u8 soft_kind(GXBlendMode mode, bool depthCompare, bool depthUpdate, bool colorUpdate, bool keepsDstAlpha, GXBlendFactor src,
                    GXBlendFactor dst, GXPrimitive primitive) noexcept {
  if (mode != GX_BM_BLEND || !depthCompare || depthUpdate || !colorUpdate || primitive == GX_LINES || primitive == GX_LINESTRIP ||
      primitive == GX_POINTS)
    return 0;
  if (keepsDstAlpha)
    return 0;
  if (src == GX_BL_SRCALPHA && (dst == GX_BL_INVSRCALPHA || dst == GX_BL_ONE))
    return 1;
  if (src == GX_BL_ONE && dst == GX_BL_ONE)
    return 2;
  return 0;
}

// A key as the draw path leaves it under the look, from the key of the game's state alone (the planner's): what
// populate_pipeline_config does after the state is read (neutralise, the soft card, the card range) and what regs.cpp does to
// the colour channels' lighting (a channel with lighting off keeps only its material source). Triangles.
inline void apply(PipelineConfig& c, const Look& l) noexcept {
  if (l.gameLightOff) {
    for (int i = 0; i < 2; ++i)
      if (c.shaderConfig.colorChannels[i].lightingEnabled)
        c.shaderConfig.colorChannels[i] = ColorChannelConfig{.matSrc = c.shaderConfig.colorChannels[i].matSrc};
  }
  if (l.neutral)
    rt::neutralise(c.shaderConfig);
  if (l.soft)
    c.shaderConfig.softCard = soft_kind(c.blendMode, c.depthCompare, c.depthUpdate, c.colorUpdate, c.alphaUpdate && c.dstAlpha != UINT32_MAX,
                                        c.blendFacSrc, c.blendFacDst, GX_TRIANGLES);
  if (l.cards)
    c.shaderConfig.cardRange = card_range(c.blendMode, c.blendFacSrc, c.blendFacDst, l.cardLevel);
}

// Whether a recorded key belongs to the look: every key drawn with cards on carries a card range, none without
inline bool fits(const PipelineConfig& c, const Look& l) noexcept { return (c.shaderConfig.cardRange != 0) == l.cards; }
} // namespace look
} // namespace aurora::gx
