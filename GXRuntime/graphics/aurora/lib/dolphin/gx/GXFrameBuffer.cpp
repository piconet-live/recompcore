#include "gx.hpp"
#include "__gx.h"

#include "../../gfx/tex_copy_conv.hpp"
#include "../../gfx/texture.hpp"
#include "../../window.hpp"
#include "../../gfx/clear.hpp"
#include "../../webgpu/gpu.hpp"
#include "../vi/vi_internal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace {
aurora::Vec2<uint32_t> scale_copy_dst(u32 logicalWidth, u32 logicalHeight) {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return {logicalWidth, logicalHeight};
  }

  const auto [logicalFbWidth, logicalFbHeight] = aurora::gx::logical_fb_size();
  const auto [targetWidth, targetHeight] = aurora::gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return {logicalWidth, logicalHeight};
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  const auto scaledWidth = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalWidth) * scaleX)), 1);
  const auto scaledHeight = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalHeight) * scaleY)), 1);
  return {scaledWidth, scaledHeight};
}
} // namespace

namespace aurora::gx {
void copy_tex(const void* dest, GXBool clear) noexcept {
  const auto rect = map_logical_scissor(g_gxState.texCopySrc);
  const auto [dstWidth, dstHeight] = scale_copy_dst(g_gxState.texCopyDstWidth, g_gxState.texCopyDstHeight);
  const auto texCopyFmt = g_gxState.texCopyFmt;

  const GXState::CopyTextureKey key{
      .dest = dest,
      .width = dstWidth,
      .height = dstHeight,
      .format = texCopyFmt,
  };
  auto it = g_gxState.copyTextureCache.find(key);
  if (it == g_gxState.copyTextureCache.end()) {
    gfx::TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(texCopyFmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, texCopyFmt, "Copy Conv Texture");
    } else {
      // Configure the texture swizzle to use alpha 1.0 if targeting RGB565 or EFB doesn't have alpha
      const auto fmt = texCopyFmt == GX_TF_RGB565 || g_gxState.pixelFmt == GX_PF_RGB8_Z24 ||
                               g_gxState.pixelFmt == GX_PF_RGB565_Z16
                           ? GX_TF_RGB565
                           : GX_TF_RGBA8;
      handle = gfx::new_render_texture(dstWidth, dstHeight, fmt, "Resolved Texture");
    }
    it = g_gxState.copyTextureCache.emplace(key, GXState::CopyTextureRef{.handle = handle, .revision = 0}).first;
  }
  auto& handle = it->second;

  if (g_gxState.alphaUpdate && g_gxState.dstAlpha != UINT32_MAX) {
    if (!clear) {
      // TODO: figure out the right behavior here.
      // should the copy have a specific alpha value but the EFB remains untouched?
    }
    // Overwrite alpha before resolving
    gfx::push_draw_command(gfx::clear::DrawData{
        .pipeline = gfx::pipeline_ref(gfx::clear::PipelineConfig{
            .clearColor = false,
            .clearAlpha = true,
            .clearDepth = false,
        }),
        .color = wgpu::Color{0.f, 0.f, 0.f, g_gxState.dstAlpha / 255.f},
    });
  }
  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate;
  const auto clearDepth = clear && g_gxState.depthUpdate;
  static const bool copyLog = std::getenv("AURORA_COPY_LOG") != nullptr;
  if (copyLog) {
    std::fprintf(stderr,
                 "[copy] phys %08X dest %p src %d,%d %dx%d mapped %d,%d %dx%d dst %ux%u fmt %u pixfmt %u clear %d "
                 "colorUpd %d alphaUpd %d new %d\n",
                 g_gxState.texCopyDestPhysical, dest, g_gxState.texCopySrc.x, g_gxState.texCopySrc.y,
                 g_gxState.texCopySrc.width, g_gxState.texCopySrc.height, rect.x, rect.y, rect.width, rect.height,
                 dstWidth, dstHeight, static_cast<unsigned>(texCopyFmt), static_cast<unsigned>(g_gxState.pixelFmt),
                 clear ? 1 : 0, g_gxState.colorUpdate ? 1 : 0, g_gxState.alphaUpdate ? 1 : 0,
                 handle.revision == 0 ? 1 : 0);
  }
  gfx::resolve_pass(handle.handle, rect, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value(),
                    texCopyFmt);
  ++handle.revision;
  g_gxState.copyTextures[dest] = handle;
}

// The display copy's texture: released with the copy textures (size change, shutdown), never by a static destructor
static gfx::TextureHandle g_xfbCopy;
void release_xfb_copy() noexcept { g_xfbCopy.reset(); }

void copy_xfb(u32 copyValue) noexcept {
  // Bits of BP 0x52 (Dolphin's UPE_Copy): 0 clamp top, 1 clamp bottom, 7-8 gamma, 10 y-scale invert, 11 clear
  static const bool enabled = [] {
    const char* v = std::getenv("AURORA_XFB");
    return v == nullptr || v[0] != '0';
  }();
  // The vertical (deflicker) filter as the hardware applies it: AURORA_XFB_FILTER=1. By default the coefficients are
  // summed into the middle row, as Dolphin's default "Disable Copy Filter" (brightness kept, no blur).
  static const bool hwFilter = [] {
    const char* v = std::getenv("AURORA_XFB_FILTER");
    return v != nullptr && v[0] == '1';
  }();
  const bool clear = (copyValue >> 11 & 1) != 0;
  const auto rect = map_logical_scissor(g_gxState.texCopySrc);
  const u32 yScaleReg = g_gxState.bpRegCache[0x4E] & 0x1FF;
  float yScale = 1.f;
  if (yScaleReg != 0) {
    yScale = (copyValue >> 10 & 1) != 0 ? 256.f / static_cast<float>(yScaleReg) : static_cast<float>(yScaleReg) / 256.f;
  }
  // As the hardware: 1 + (height - 1) * scale lines
  const u32 lines = std::max<u32>(
      static_cast<u32>(1.f + static_cast<float>(g_gxState.texCopySrc.height - 1) * yScale), 1);
  const u32 width = static_cast<u32>(std::max(rect.width, 1));
  static constexpr float GammaRcp[4] = {1.f, 1.f / 1.7f, 1.f / 2.2f, 1.f / 2.2f};
  const u32 c0 = g_gxState.bpRegCache[0x53], c1 = g_gxState.bpRegCache[0x54];
  // Seven 6-bit coefficients (0x53: 0-3, 0x54: 4-6), summed into rows above, at and below (Dolphin's RAM sums)
  const auto k = [&](u32 i) { return static_cast<float>((i < 4 ? c0 >> (6 * i) : c1 >> (6 * (i - 4))) & 0x3F); };
  const bool clampTop = (copyValue & 1) != 0, clampBottom = (copyValue >> 1 & 1) != 0;
  const auto [efbW, efbH] = gfx::get_render_target_size();
  const float above = k(0) + k(1), middle = k(2) + k(3) + k(4), below = k(5) + k(6);
  const std::array<float, 8> params{
      hwFilter ? above : 0.f,
      hwFilter ? middle : above + middle + below,
      hwFilter ? below : 0.f,
      GammaRcp[copyValue >> 7 & 3],
      static_cast<float>(clampTop ? rect.y : 0),
      static_cast<float>((clampBottom ? rect.y + rect.height : static_cast<int32_t>(efbH)) - 1),
      0.f,
      0.f,
  };
  static const bool copyLog = std::getenv("AURORA_COPY_LOG") != nullptr;
  if (copyLog) {
    std::fprintf(stderr,
                 "[xfb] value %06X src %d,%d %dx%d -> %ux%u yscale %u (%.4f) coef %g %g %g gamma-rcp %.4f clamp %d%d "
                 "clear %d efb %ux%u\n",
                 copyValue, rect.x, rect.y, rect.width, rect.height, width, lines, yScaleReg, yScale, params[0],
                 params[1], params[2], params[3], clampTop ? 1 : 0, clampBottom ? 1 : 0, clear ? 1 : 0, efbW, efbH);
  }
  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate;
  const auto clearDepth = clear && g_gxState.depthUpdate;
  if (!enabled) {
    return; // stage 3's behaviour: the EFB is presented, and not cleared here
  }
  auto& s_xfb = g_xfbCopy;
  if (!s_xfb || s_xfb->size.width != width || s_xfb->size.height != lines) {
    s_xfb = gfx::new_render_texture(width, lines, GX_TF_RGBA8, "XFB copy");
  }
  gfx::resolve_pass(s_xfb, rect, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value(),
                    gfx::tex_copy_conv::XfbCopyFormat, &params);
}
} // namespace aurora::gx

extern "C" {
GXRenderModeObj GXNtsc480IntDf = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXNtsc480Int = {
    VI_TVMODE_NTSC_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {0, 0, 21, 22, 21, 0, 0},
};
GXRenderModeObj GXPal528IntDf = {
    VI_TVMODE_PAL_INT,
    704,
    528,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};
GXRenderModeObj GXMpal480IntDf = {
    VI_TVMODE_PAL_INT,
    640,
    480,
    480,
    40,
    0,
    640,
    480,
    VI_XFBMODE_DF,
    0,
    0,
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
    {8, 8, 10, 12, 10, 8, 8},
};

void GXAdjustForOverscan(GXRenderModeObj* rmin, GXRenderModeObj* rmout, u16 hor, u16 ver) {
  *rmout = *rmin;
  const auto size = aurora::window::get_window_size();
  rmout->fbWidth = size.fb_width;
  rmout->efbHeight = size.fb_height;
  rmout->xfbHeight = size.fb_height;
}

void GXSetDispCopySrc(u16 left, u16 top, u16 wd, u16 ht) {}

void GXSetTexCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_SRC);
  GX_WRITE_U32(left);
  GX_WRITE_U32(top);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
}

void GXSetDispCopyDst(u16 wd, u16 ht) {}

void GXSetTexCopyDst(u16 wd, u16 ht, GXTexFmt fmt, GXBool mipmap) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DST);
  GX_WRITE_U32(wd);
  GX_WRITE_U32(ht);
  GX_WRITE_U32(fmt);
  GX_WRITE_U8(mipmap != GX_FALSE);
}

// TODO GXSetDispCopyFrame2Field
// TODO GXSetCopyClamp

u32 GXSetDispCopyYScale(f32 vscale) { return 0; }

void GXSetCopyClear(GXColor color, u32 depth) {
  // BP 0x4F: clear color R + A
  u32 reg0 = 0;
  SET_REG_FIELD(0, reg0, 8, 0, color.r);
  SET_REG_FIELD(0, reg0, 8, 8, color.a);
  SET_REG_FIELD(0, reg0, 8, 24, 0x4F);
  GX_WRITE_RAS_REG(reg0);

  // BP 0x50: clear color B + G
  u32 reg1 = 0;
  SET_REG_FIELD(0, reg1, 8, 0, color.b);
  SET_REG_FIELD(0, reg1, 8, 8, color.g);
  SET_REG_FIELD(0, reg1, 8, 24, 0x50);
  GX_WRITE_RAS_REG(reg1);

  // BP 0x51: clear Z (24-bit)
  u32 reg2 = 0;
  SET_REG_FIELD(0, reg2, 24, 0, depth);
  SET_REG_FIELD(0, reg2, 8, 24, 0x51);
  GX_WRITE_RAS_REG(reg2);
  __gx->bpSent = 1;
}

void GXSetCopyFilter(GXBool aa, u8 sample_pattern[12][2], GXBool vf, u8 vfilter[7]) {}

void GXSetDispCopyGamma(GXGamma gamma) {}

void GXCopyDisp(void* dest, GXBool clear) {}

void GXCopyTex(void* dest, GXBool clear) {
  GX_WRITE_AURORA(GX_AURORA_LOAD_COPY_DEST);
  GX_WRITE_U64(reinterpret_cast<u64>(dest));

  SET_REG_FIELD(0, __gx->cpTex, 1, 11, clear != GX_FALSE);
  SET_REG_FIELD(0, __gx->cpTex, 1, 14, 0);
  SET_REG_FIELD(0, __gx->cpTex, 8, 24, 0x52);
  GX_WRITE_RAS_REG(__gx->cpTex);
}

// TODO GXGetYScaleFactor
// TODO GXGetNumXfbLines
// TODO GXClearBoundingBox
// TODO GXReadBoundingBox
}
