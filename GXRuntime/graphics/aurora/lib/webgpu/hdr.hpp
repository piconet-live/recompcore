#pragma once

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

// AURORA_LOOK_HDR=1: the picture to an HDR display in scRGB (16-bit float surface, 1.0 = 80 nits), range above white kept.
// The surface only: nothing the game reads changes. Off, or on a display not in HDR mode, nothing here is used.
namespace aurora::webgpu::hdr {
// Before the first pipeline: asks the window's display and the surface, says what it found on stderr, decides.
void choose(const wgpu::SurfaceCapabilities& capabilities);
bool on() noexcept;
// The display copy keeps its float range: HDR on, or a range log (AURORA_LOOK_PEAKLOG) of the raw picture.
bool float_copy() noexcept;
// The present and everything over it are drawn into this, as an 8-bit surface would hold them; final_pass then turns it into scRGB on the surface.
wgpu::TextureView target(uint32_t width, uint32_t height);
void final_pass(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& surface);
void shutdown() noexcept;
} // namespace aurora::webgpu::hdr
