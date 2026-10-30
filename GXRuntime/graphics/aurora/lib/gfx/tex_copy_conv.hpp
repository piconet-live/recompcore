#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <array>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct alignas(16) Uniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  std::array<uint32_t, 3> _pad{};
};
static_assert(sizeof(Uniforms) == 32);

struct ConvRequest {
  GXTexFmt fmt;
  GXPixelFmt srcFmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // Uniforms
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

// Not a GX format: the display (XFB) copy's conversion (copy filter, gamma). Its uniform is XfbUniform.
inline constexpr GXTexFmt XfbCopyFormat = static_cast<GXTexFmt>(0x7D);
struct alignas(16) XfbUniform {
  float offset[2]; // as the UV transform of the other conversions
  float scale[2];
  float coef[4];  // the three summed filter coefficients (rows above, this, below), then 1/gamma
  float clamp[4]; // top and bottom rows (UV of the row centres), one EFB row in UV, unused
};
static_assert(sizeof(XfbUniform) == 48);

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);

bool snapshot_depth_supported() noexcept;
void snapshot_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t msaaSamples,
                    const wgpu::TextureView& dst);

} // namespace aurora::gfx::tex_copy_conv
