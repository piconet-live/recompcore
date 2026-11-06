#include "soft_cards.hpp"
#include "look_keys.hpp"

#include "fifo.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resource_cache.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace aurora::gx::soft {
namespace {
wgpu::BindGroupLayout s_layout;
wgpu::PipelineLayout s_pipelineLayout;
wgpu::Texture s_farTexture;
wgpu::TextureView s_farView;
// the frame's latest depth snapshot (null: none since the EFB was cleared), and the depth writes then and now
wgpu::TextureView s_view;
bool s_hasView = false;
uint64_t s_writes = 0, s_writesAtView = 0;
uint32_t s_taken = 0;
// AURORA_RT_SOFT_CARDS_LOG=1: per 600 cleared frames, the cards, the snapshots taken and the cards that took an older one
bool s_log = false;
uint64_t s_logCards = 0, s_logTaken = 0, s_logOlder = 0, s_logOffscreen = 0, s_logFrames = 0;

float env_number(const char* name, float otherwise) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' ? static_cast<float>(std::atof(v)) : otherwise;
}
// Tuned by eye: the fade's length as a share of the card's own distance
constexpr float ShareDefault = 0.03f;
// The most snapshots a frame takes (each a pass break and a full-size depth copy); past that, cards take the latest. Tuned against the cost.
constexpr uint32_t MostDefault = 6;
uint32_t most() noexcept {
  static const uint32_t n = static_cast<uint32_t>(env_number("AURORA_RT_SOFT_CARDS_MOST", static_cast<float>(MostDefault)));
  return n;
}
bool writes_depth() noexcept { return g_gxState.depthCompare && g_gxState.depthUpdate; }
} // namespace

bool on() noexcept {
  static const bool v = env_number("AURORA_RT_SOFT_CARDS", 0.f) != 0.f;
  return v;
}

float share() noexcept {
  static const float v = env_number("AURORA_RT_SOFT_CARDS_SHARE", ShareDefault);
  return v;
}

u8 kind(GXPrimitive primitive) noexcept {
  const auto& s = g_gxState;
  // blended over what is there, depth tested, writing no depth, colour written, no destination alpha constant (its alpha is the EFB's); whole triangles
  return look::soft_kind(s.blendMode, s.depthCompare, s.depthUpdate, s.colorUpdate,
                         efb_has_alpha(s.pixelFmt) && s.alphaUpdate && s.dstAlpha != UINT32_MAX, s.blendFacSrc, s.blendFacDst, primitive);
}

void before_draw(GXPrimitive primitive) noexcept {
  if (writes_depth()) {
    ++s_writes;
    return;
  }
  if (g_gxState.projType != GX_PERSPECTIVE || kind(primitive) == 0)
    return;
  if (s_log)
    ++s_logCards;
  if (s_hasView && (s_writes == s_writesAtView || s_taken >= most())) {
    if (s_log && s_writes != s_writesAtView)
      ++s_logOlder;
    return;
  }
  // an offscreen pass (GXCreateFrameBuffer) has its own depth, and a pass break would end it: its cards stay as they are
  const auto at = gfx::readback_capture();
  if (!at.active || at.offscreen) {
    if (s_log)
      ++s_logOffscreen;
    return;
  }
  // from inside the command stream: the pass break must not wait for the stream
  const fifo::HoldDrain hold;
  gfx::ResolvedTargets targets;
  if (!gfx::resolve_pass({.color = false, .depth = true}, targets) || !targets.depth)
    return;
  s_view = targets.depth;
  s_hasView = true;
  s_writesAtView = s_writes;
  ++s_taken;
  if (s_log)
    ++s_logTaken;
}

void efb_cleared() noexcept {
  s_view = {};
  s_hasView = false;
  s_writes = s_writesAtView = 0;
  s_taken = 0;
  if (s_log && ++s_logFrames % 600 == 0) {
    std::fprintf(stderr, "[soft] per frame: %.1f cards in perspective, %.2f depth snapshots, %.1f cards on an older one, %.1f offscreen\n",
                 s_logCards / 600.0, s_logTaken / 600.0, s_logOlder / 600.0, s_logOffscreen / 600.0);
    s_logCards = s_logTaken = s_logOlder = s_logOffscreen = 0;
  }
}

gfx::BindGroupRef bind_group() noexcept {
  const WGPUBindGroupEntry entry{
      .binding = 0,
      .textureView = s_hasView ? s_view.Get() : s_farView.Get(),
  };
  const WGPUBindGroupDescriptor descriptor{
      .label = {"GX Soft Card Depth", WGPU_STRLEN},
      .layout = s_layout.Get(),
      .entryCount = 1,
      .entries = &entry,
  };
  return gfx::bind_group_ref(descriptor);
}

const wgpu::PipelineLayout& pipeline_layout() noexcept { return s_pipelineLayout; }

void initialize(const wgpu::BindGroupLayout& staticLayout, const wgpu::BindGroupLayout& uniformLayout, const wgpu::BindGroupLayout& textureLayout,
                uint32_t immediateSize) noexcept {
  s_log = env_number("AURORA_RT_SOFT_CARDS_LOG", 0.f) != 0.f;
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture =
          {
              .sampleType = wgpu::TextureSampleType::UnfilterableFloat,
              .viewDimension = wgpu::TextureViewDimension::e2D,
          },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "GX Soft Card Depth Layout",
      .entryCount = 1,
      .entries = &entry,
  };
  s_layout = webgpu::g_device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array layouts{staticLayout, uniformLayout, textureLayout, s_layout};
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "GX Soft Card Pipeline Layout",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
      .immediateSize = immediateSize,
  };
  s_pipelineLayout = webgpu::g_device.CreatePipelineLayout(&pipelineDescriptor);
  // no snapshot: the depth of nothing (the far end; reversed depth: 0)
  const wgpu::TextureDescriptor farDescriptor{
      .label = "GX Soft Card No Depth",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {1, 1},
      .format = wgpu::TextureFormat::R32Float,
  };
  s_farTexture = webgpu::g_device.CreateTexture(&farDescriptor);
  s_farView = s_farTexture.CreateView();
  const float far = UseReversedZ ? 0.f : 1.f;
  const wgpu::TexelCopyTextureInfo dst{.texture = s_farTexture};
  const wgpu::TexelCopyBufferLayout bufferLayout{.bytesPerRow = 4, .rowsPerImage = 1};
  const wgpu::Extent3D size{1, 1, 1};
  webgpu::g_queue.WriteTexture(&dst, &far, sizeof far, &bufferLayout, &size);
}

void shutdown() noexcept {
  s_view = {};
  s_hasView = false;
  s_farView = {};
  s_farTexture = {};
  s_pipelineLayout = {};
  s_layout = {};
}
} // namespace aurora::gx::soft
