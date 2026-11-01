#pragma once
// Positional (depth) timewarp, AURORA_TIMEWARP=1: pictures between the true frames, each the last true frame's colour
// and depth lifted to view space and reprojected for where the camera is guessed to be by then. See timewarp.cpp.

#include <cstddef>
#include <cstdint>

#include "../webgpu/gpu.hpp"

#include <functional>
#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx::timewarp {
extern bool g_on;   // fixed at start
extern bool g_pin;  // AURORA_TIMEWARP_PIN (default on): the followed machine, what clings to it and the HUD stay as in the true frame

// A host's overlay on every warped picture too, after the warp and before its present, on the render worker
using AfterWarp = std::function<void(const wgpu::RenderPassEncoder& pass, const webgpu::Viewport& viewport, uint32_t width,
                                     uint32_t height)>;
void set_after_warp(AfterWarp hook);

void start();  // once, after Aurora is up: reads the switches; a thread of its own paces the warps when on
void stop();

// The Aurora FIFO processor, before each draw (only when g_on): a static object's matrix, drawn in two frames, gives
// the camera's move between them.
void note_draw(uint8_t cmd, uint16_t vtxCount, const uint8_t* vertices, size_t available) noexcept;
uint32_t draw_class() noexcept;  // after note_draw: the stencil reference

// The render worker, right after the display copy: depth is kept here, before the next pass clears it.
void capture(const wgpu::CommandEncoder& encoder, const wgpu::Texture& colorTexture, const wgpu::TextureView& colorView,
             wgpu::Extent3D colorSize, const wgpu::TextureView& depthView, uint32_t msaaSamples, wgpu::Extent3D fbSize,
             int32_t rectX, int32_t rectY, int32_t rectWidth, int32_t rectHeight);

// Aurora's own thread, around aurora_backend_present: the true frame is on its way (warps give way), then it is out.
void true_frame_coming() noexcept;
void true_frame_done();
// Aurora's own thread, for a frame decoded and not drawn (Patch 25)
void frame_not_drawn() noexcept;
} // namespace aurora::gfx::timewarp
