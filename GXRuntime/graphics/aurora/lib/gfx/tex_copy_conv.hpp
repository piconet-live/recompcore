#pragma once

#include "common.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct ConvRequest {
  GXTexFmt fmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // UV transform uniform (offset + scale)
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

// Not a GX format: the display (XFB) copy's conversion (copy filter, gamma). Its uniform is XfbUniform.
inline constexpr GXTexFmt XfbCopyFormat = static_cast<GXTexFmt>(0x7D);
struct XfbUniform {
  float offset[2]; // as the UV transform of the other conversions
  float scale[2];
  float coef[4];  // the three summed filter coefficients (rows above, this, below), then 1/gamma
  float clamp[4]; // top and bottom rows (UV of the row centres), one EFB row in UV, unused
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);

} // namespace aurora::gfx::tex_copy_conv
