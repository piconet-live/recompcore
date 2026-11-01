#pragma once

#include <aurora/aurora.h>
#include <aurora/math.hpp>

#include "wgpu.hpp"

#include <array>
#include <functional>
#include <cstdint>

struct SDL_Window;

namespace aurora::webgpu {
inline constexpr wgpu::TextureFormat NormalBufferFormat = wgpu::TextureFormat::RGB10A2Unorm;

struct GraphicsConfig {
  wgpu::SurfaceConfiguration surfaceConfiguration;
  wgpu::TextureFormat depthFormat;
  uint32_t msaaSamples;
  uint16_t textureAnisotropy;
  bool normalBuffer = false;
};
struct TextureWithSampler {
  wgpu::Texture texture;
  wgpu::TextureView view;
  wgpu::Extent3D size;
  wgpu::TextureFormat format;
  wgpu::Sampler sampler;
};
struct Viewport {
  float left;
  float top;
  float width;
  float height;
  float znear;
  float zfar;

  bool operator==(const Viewport& rhs) const {
    return left == rhs.left && top == rhs.top && width == rhs.width && height == rhs.height && znear == rhs.znear &&
           zfar == rhs.zfar;
  }
  bool operator!=(const Viewport& rhs) const { return !(*this == rhs); }
};

extern wgpu::Device g_device;
extern wgpu::Queue g_queue;
extern wgpu::Surface g_surface;
extern wgpu::BackendType g_backendType;
extern GraphicsConfig g_graphicsConfig;
extern TextureWithSampler g_frameBuffer;
extern TextureWithSampler g_frameBufferResolved;
extern TextureWithSampler g_depthBuffer;
extern TextureWithSampler g_normalBuffer;
extern wgpu::RenderPipeline g_CopyPipeline;
extern wgpu::RenderPipeline g_CopyPremultipliedAlphaPipeline;
extern wgpu::BindGroup g_CopyBindGroup;
extern wgpu::Instance g_instance;
extern wgpu::AdapterInfo g_adapterInfo;
extern bool g_hasCoreFeatures;
extern bool g_bcTexturesSupported;
extern bool g_astcTexturesSupported;
extern bool g_textureComponentSwizzleSupported;
extern bool g_dualSourceBlendingSupported;

bool initialize(AuroraBackend backend, bool allowCpu);
void shutdown();
void release_surface() noexcept;
bool refresh_surface(bool recreate = true);
void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force = false);
TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled);
bool enable_normal_buffer();
const TextureWithSampler& present_source() noexcept;
// The display (XFB) copy's texture as the present source, from the frame it was made on (render worker only).
void set_present_override(const wgpu::Texture& texture, const wgpu::TextureView& view, wgpu::Extent3D size,
                          wgpu::TextureFormat format) noexcept;
wgpu::BindGroup create_copy_bind_group(const TextureWithSampler& source);
void set_resampler(AuroraSampler sampler) noexcept;
AuroraSampler get_resampler() noexcept;
Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept;
const TextureWithSampler& resample_present_source(const wgpu::CommandEncoder& encoder, const Viewport& viewport);
void draw_clear(const wgpu::RenderPassEncoder& pass, bool clearColor, bool clearAlpha, bool clearDepth,
                const Vec4<float>& clearColorValue, float clearDepthValue);

size_t load_from_cache(void const* key, size_t keySize, void* value, size_t valueSize, void* userdata);
void store_to_cache(void const* key, size_t keySize, void const* value, size_t valueSize, void* userdata);
void cache_prune();
void cache_shutdown();

// Patch: what an embedding program draws at the present, on top of everything Aurora drew (a frame's picture, however it
// got to the surface) and never into it; it is not in the present source and not in a readback.
struct PresentHook {
  using Draw = std::function<void(const wgpu::RenderPassEncoder& pass, const Viewport& picture, uint32_t surfaceWidth,
                                  uint32_t surfaceHeight)>;
  // Aurora's thread at the frame's end, the FIFO drained (the frame's copies can be looked up): what the frame's present
  // draws, on the render worker. It may be called again for another present of the same frame.
  Draw (*capture)() = nullptr;
  // The render worker: the view the present draws into instead of the surface (null: the surface itself), and then
  // the last pass from that view to the surface. For a filter of the whole picture.
  wgpu::TextureView (*finalTarget)(uint32_t width, uint32_t height) = nullptr;
  void (*finalPass)(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& from,
                    const wgpu::TextureView& surface) = nullptr;
};
void set_present_hook(const PresentHook& hook) noexcept;
const PresentHook& present_hook() noexcept;

} // namespace aurora::webgpu
