// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_overlay.h. Three threads: the host's (the frame and textures, into the mailbox below), Aurora's (the frame's
// capture, with the FIFO drained: only then may the game's copies be looked up) and the render worker's (all of the GPU
// work: the textures, the buffers, the drawing).
//
// The drawing is its own small pipeline rather than Aurora's ImGui: the host's ImGui is another version, its font atlas
// changes as glyphs are baked, and its drawing switches between nearest and smooth sampling mid-list.

#include "aurora_overlay.h"

#include "gxruntime/aurora_present.h"

#include <dolphin/gx.h>

#include "gfx/render_worker.hpp"
#include "gfx/texture.hpp"
#include "gfx/timewarp.hpp"
#include "gx/gx.hpp"
#include "webgpu/gpu.hpp"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
namespace webgpu = aurora::webgpu;
namespace gfx = aurora::gfx;

constexpr uint64_t COPY_TAG = AURORA_OVERLAY_COPY_TEXTURE(0);
constexpr uint64_t COPY_TAG_MASK = 0xFFFFFFFF00000000ull;

struct Frame {
  uint32_t width = 0, height = 0;
  std::vector<AuroraOverlayVertex> vertices;
  std::vector<uint16_t> indices;
  std::vector<AuroraOverlayDraw> draws;
};

struct TextureOp {
  uint64_t id = 0;
  uint32_t width = 0, height = 0, x = 0, y = 0, part_width = 0, part_height = 0;
  std::vector<uint8_t> rgba;  // tightly packed; empty: forget
};

// The host's mailbox
std::mutex s_lock;
std::shared_ptr<const Frame> s_frame;
std::vector<TextureOp> s_ops;

const uint8_t* s_ram = nullptr;
std::atomic<int32_t> s_rect[4];
std::atomic<uint32_t> s_rect_surface[2];  // the surface s_rect is in
std::atomic<bool> s_has_rect{false};

// ---- the render worker's ------------------------------------------------------------------------------------------
struct Texture {
  wgpu::Texture texture;
  std::array<wgpu::BindGroup, 2> bind;  // nearest, smooth
};

wgpu::RenderPipeline s_pipeline;
wgpu::BindGroupLayout s_layout, s_size_layout;
std::array<wgpu::Sampler, 2> s_samplers;
wgpu::Buffer s_vertex_buffer, s_index_buffer, s_size_buffer;
wgpu::BindGroup s_size_bind;
uint64_t s_vertex_room = 0, s_index_room = 0;
std::unordered_map<uint64_t, Texture> s_textures;
// A copy's bind groups, by the texture's identity; the handle is kept so the identity stays that texture's
struct CopyBind {
  gfx::TextureHandle handle;
  std::array<wgpu::BindGroup, 2> bind;
};
std::unordered_map<const void*, CopyBind> s_copy_binds;

constexpr const char* SHADER = R"(
struct Size { size: vec2f };
@group(0) @binding(0) var<uniform> u: Size;
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var tex: texture_2d<f32>;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) col: vec4f };
@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f, @location(2) col: vec4f) -> V {
  var v: V;
  v.pos = vec4f(pos.x / u.size.x * 2.0 - 1.0, 1.0 - pos.y / u.size.y * 2.0, 0.0, 1.0);
  v.uv = uv;
  v.col = col;
  return v;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return v.col * textureSample(tex, samp, v.uv); }
)";

wgpu::BindGroup make_bind(const wgpu::TextureView& view, int smooth) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = s_samplers[smooth]},
      wgpu::BindGroupEntry{.binding = 1, .textureView = view},
  };
  const wgpu::BindGroupDescriptor desc{
      .label = "overlay texture",
      .layout = s_layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return webgpu::g_device.CreateBindGroup(&desc);
}

void make_pipeline() {
  if (s_pipeline)
    return;
  wgpu::ShaderSourceWGSL source{};
  source.code = SHADER;
  const wgpu::ShaderModuleDescriptor module_desc{.nextInChain = &source, .label = "overlay"};
  const auto module = webgpu::g_device.CreateShaderModule(&module_desc);

  const std::array texture_entries{
      wgpu::BindGroupLayoutEntry{.binding = 0,
                                 .visibility = wgpu::ShaderStage::Fragment,
                                 .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering}},
      wgpu::BindGroupLayoutEntry{.binding = 1,
                                 .visibility = wgpu::ShaderStage::Fragment,
                                 .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::Float,
                                                                       .viewDimension = wgpu::TextureViewDimension::e2D}},
  };
  const wgpu::BindGroupLayoutDescriptor texture_layout{.entryCount = texture_entries.size(),
                                                       .entries = texture_entries.data()};
  s_layout = webgpu::g_device.CreateBindGroupLayout(&texture_layout);
  const std::array size_entries{
      wgpu::BindGroupLayoutEntry{.binding = 0,
                                 .visibility = wgpu::ShaderStage::Vertex,
                                 .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform}},
  };
  const wgpu::BindGroupLayoutDescriptor size_layout{.entryCount = size_entries.size(), .entries = size_entries.data()};
  s_size_layout = webgpu::g_device.CreateBindGroupLayout(&size_layout);
  const std::array layouts{s_size_layout, s_layout};
  const wgpu::PipelineLayoutDescriptor layout_desc{.bindGroupLayoutCount = layouts.size(),
                                                   .bindGroupLayouts = layouts.data()};
  const auto layout = webgpu::g_device.CreatePipelineLayout(&layout_desc);

  const wgpu::BufferDescriptor size_buffer{
      .label = "overlay size", .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, .size = 16};
  s_size_buffer = webgpu::g_device.CreateBuffer(&size_buffer);
  const wgpu::BindGroupEntry size_entry{.binding = 0, .buffer = s_size_buffer, .size = 16};
  const wgpu::BindGroupDescriptor size_bind{.layout = s_size_layout, .entryCount = 1, .entries = &size_entry};
  s_size_bind = webgpu::g_device.CreateBindGroup(&size_bind);

  const wgpu::SamplerDescriptor nearest{.label = "overlay nearest",
                                        .addressModeU = wgpu::AddressMode::ClampToEdge,
                                        .addressModeV = wgpu::AddressMode::ClampToEdge};
  const wgpu::SamplerDescriptor smooth{.label = "overlay smooth",
                                       .addressModeU = wgpu::AddressMode::ClampToEdge,
                                       .addressModeV = wgpu::AddressMode::ClampToEdge,
                                       .magFilter = wgpu::FilterMode::Linear,
                                       .minFilter = wgpu::FilterMode::Linear};
  s_samplers = {webgpu::g_device.CreateSampler(&nearest), webgpu::g_device.CreateSampler(&smooth)};

  const std::array attributes{
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x2, .offset = 0, .shaderLocation = 0},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x2, .offset = 8, .shaderLocation = 1},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Unorm8x4, .offset = 16, .shaderLocation = 2},
  };
  const wgpu::VertexBufferLayout vertex_layout{.arrayStride = sizeof(AuroraOverlayVertex),
                                               .attributeCount = attributes.size(),
                                               .attributes = attributes.data()};
  // as the emulator's on-screen drawing: straight alpha, the target's alpha left
  const wgpu::BlendState blend{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::SrcAlpha,
                .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::ColorTargetState target{
      .format = webgpu::present_format(), .blend = &blend, .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor pipeline{
      .label = "overlay",
      .layout = layout,
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs", .bufferCount = 1, .buffers = &vertex_layout},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  s_pipeline = webgpu::g_device.CreateRenderPipeline(&pipeline);
}

void apply(const TextureOp& op) {
  make_pipeline();
  if (op.rgba.empty()) {
    s_textures.erase(op.id);
    return;
  }
  Texture& t = s_textures[op.id];
  if (!t.texture || t.texture.GetWidth() != op.width || t.texture.GetHeight() != op.height) {
    const wgpu::TextureDescriptor desc{
        .label = "overlay texture",
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {op.width, op.height, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    t.texture = webgpu::g_device.CreateTexture(&desc);
    const auto view = t.texture.CreateView();
    t.bind = {make_bind(view, 0), make_bind(view, 1)};
  }
  const wgpu::TexelCopyTextureInfo dst{.texture = t.texture, .origin = {op.x, op.y, 0}};
  const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = op.part_width * 4, .rowsPerImage = op.part_height};
  const wgpu::Extent3D part{op.part_width, op.part_height, 1};
  webgpu::g_queue.WriteTexture(&dst, op.rgba.data(), op.rgba.size(), &layout, &part);
}

void grow(wgpu::Buffer& buffer, uint64_t& room, uint64_t want, wgpu::BufferUsage usage, const char* label) {
  if (room >= want && buffer)
    return;
  room = std::max<uint64_t>(want + want / 2, 64u * 1024u);
  const wgpu::BufferDescriptor desc{.label = label, .usage = usage | wgpu::BufferUsage::CopyDst, .size = room};
  buffer = webgpu::g_device.CreateBuffer(&desc);
}

using Copies = std::vector<std::pair<uint64_t, gfx::TextureHandle>>;

const wgpu::BindGroup* copy_bind(const gfx::TextureHandle& handle, int smooth) {
  auto it = s_copy_binds.find(handle.get());
  if (it == s_copy_binds.end()) {
    if (s_copy_binds.size() >= 64)
      s_copy_binds.clear();
    it = s_copy_binds
             .emplace(handle.get(), CopyBind{handle, {make_bind(handle->sampleTextureView, 0),
                                                      make_bind(handle->sampleTextureView, 1)}})
             .first;
  }
  return &it->second.bind[smooth];
}

void draw_frame(const Frame& f, const Copies& copies, const wgpu::RenderPassEncoder& pass, uint32_t surface_w,
                uint32_t surface_h) {
  if (f.draws.empty() || f.width == 0 || f.height == 0 || surface_w == 0 || surface_h == 0)
    return;
  make_pipeline();
  const uint64_t vertex_bytes = f.vertices.size() * sizeof(AuroraOverlayVertex);
  const uint64_t index_bytes = (f.indices.size() * sizeof(uint16_t) + 3) & ~uint64_t{3};
  grow(s_vertex_buffer, s_vertex_room, vertex_bytes, wgpu::BufferUsage::Vertex, "overlay vertices");
  grow(s_index_buffer, s_index_room, index_bytes, wgpu::BufferUsage::Index, "overlay indices");
  webgpu::g_queue.WriteBuffer(s_vertex_buffer, 0, f.vertices.data(), vertex_bytes);
  webgpu::g_queue.WriteBuffer(s_index_buffer, 0, f.indices.data(), f.indices.size() * sizeof(uint16_t));
  const float size[4] = {static_cast<float>(f.width), static_cast<float>(f.height), 0, 0};
  webgpu::g_queue.WriteBuffer(s_size_buffer, 0, size, sizeof(size));

  pass.SetPipeline(s_pipeline);
  pass.SetBindGroup(0, s_size_bind, 0, nullptr);
  pass.SetVertexBuffer(0, s_vertex_buffer, 0, vertex_bytes);
  pass.SetIndexBuffer(s_index_buffer, wgpu::IndexFormat::Uint16, 0, f.indices.size() * sizeof(uint16_t));
  const float sx = static_cast<float>(surface_w) / static_cast<float>(f.width);
  const float sy = static_cast<float>(surface_h) / static_cast<float>(f.height);
  for (const AuroraOverlayDraw& d : f.draws) {
    const int smooth = d.linear != 0 ? 1 : 0;
    const wgpu::BindGroup* bind = nullptr;
    if ((d.texture & COPY_TAG_MASK) == COPY_TAG) {
      for (const auto& [id, handle] : copies)
        if (id == d.texture && handle)
          bind = copy_bind(handle, smooth);
    } else if (const auto it = s_textures.find(d.texture); it != s_textures.end()) {
      bind = &it->second.bind[smooth];
    }
    if (bind == nullptr)
      continue;  // (a copy the game has not made yet)
    const uint32_t x0 = static_cast<uint32_t>(std::clamp(d.clip[0] * sx, 0.f, static_cast<float>(surface_w)));
    const uint32_t y0 = static_cast<uint32_t>(std::clamp(d.clip[1] * sy, 0.f, static_cast<float>(surface_h)));
    const uint32_t x1 = static_cast<uint32_t>(std::clamp(d.clip[2] * sx, 0.f, static_cast<float>(surface_w)));
    const uint32_t y1 = static_cast<uint32_t>(std::clamp(d.clip[3] * sy, 0.f, static_cast<float>(surface_h)));
    if (x1 <= x0 || y1 <= y0)
      continue;
    pass.SetScissorRect(x0, y0, x1 - x0, y1 - y0);
    pass.SetBindGroup(1, *bind, 0, nullptr);
    pass.DrawIndexed(d.index_count, 1, d.first_index, d.base_vertex, 0);
  }
  pass.SetScissorRect(0, 0, surface_w, surface_h);
}

// The render worker's: the last true frame's drawing, again over each warped picture until the next true frame
webgpu::PresentHook::Draw s_last_draw;

// ---- Aurora's thread at a frame's end ------------------------------------------------------------------------------
webgpu::PresentHook::Draw capture() {
  std::shared_ptr<const Frame> frame;
  std::vector<TextureOp> ops;
  {
    std::lock_guard g{s_lock};
    frame = s_frame;
    ops.swap(s_ops);
  }
  if (!ops.empty()) {
    gfx::render_worker::enqueue_work([ops = std::move(ops)] {
      for (const TextureOp& op : ops)
        apply(op);
    });
  }
  // the game's copies this frame's drawing uses: only here is the table Aurora's FIFO thread writes at rest
  Copies copies;
  if (frame) {
    for (const AuroraOverlayDraw& d : frame->draws) {
      if ((d.texture & COPY_TAG_MASK) != COPY_TAG)
        continue;
      const bool have = std::any_of(copies.begin(), copies.end(), [&](const auto& c) { return c.first == d.texture; });
      if (have || s_ram == nullptr)
        continue;
      const void* at = s_ram + static_cast<uint32_t>(d.texture);
      if (const auto it = aurora::gx::g_gxState.copyTextures.find(at); it != aurora::gx::g_gxState.copyTextures.end())
        copies.emplace_back(d.texture, it->second.handle);
    }
  }
  webgpu::PresentHook::Draw draw = [frame = std::move(frame), copies = std::move(copies)](const wgpu::RenderPassEncoder& pass,
                                                                                          const webgpu::Viewport& picture, uint32_t surface_w,
                                                                                          uint32_t surface_h) {
    s_rect[0].store(static_cast<int32_t>(picture.left), std::memory_order_relaxed);
    s_rect[1].store(static_cast<int32_t>(picture.top), std::memory_order_relaxed);
    s_rect[2].store(static_cast<int32_t>(picture.width), std::memory_order_relaxed);
    s_rect[3].store(static_cast<int32_t>(picture.height), std::memory_order_relaxed);
    s_rect_surface[0].store(surface_w, std::memory_order_relaxed);
    s_rect_surface[1].store(surface_h, std::memory_order_relaxed);
    s_has_rect.store(true, std::memory_order_release);
    if (frame)
      draw_frame(*frame, copies, pass, surface_w, surface_h);
  };
  if (!gfx::timewarp::g_on)
    return draw;
  return [draw = std::move(draw)](const wgpu::RenderPassEncoder& pass, const webgpu::Viewport& picture, uint32_t surface_w,
                                  uint32_t surface_h) {
    s_last_draw = draw;
    draw(pass, picture, surface_w, surface_h);
  };
}

// ---- the colour filter: AURORA_COLOURBLIND=deuteranopia|protanopia|tritanopia ------------------------------------------
// The whole window, the overlay included, in a last pass. It corrects, not simulates: what the deficiency loses is moved
// to what it keeps (Fidaner's daltonization, on Machado, Oliveira and Fernandes 2009's matrices at severity 1, in linear light).
struct Mat3 {
  float m[3][3];
};
Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r.m[i][j] += a.m[i][k] * b.m[k][j];
  return r;
}
constexpr Mat3 PROTAN = {{{0.152286f, 1.052583f, -0.204868f}, {0.114503f, 0.786281f, 0.099216f}, {-0.003882f, -0.048116f, 1.051998f}}};
constexpr Mat3 DEUTAN = {{{0.367322f, 0.860646f, -0.227968f}, {0.280085f, 0.672501f, 0.047413f}, {-0.011820f, 0.042940f, 0.968881f}}};
constexpr Mat3 TRITAN = {{{1.255528f, -0.076749f, -0.178779f}, {-0.078411f, 0.930809f, 0.147602f}, {0.004733f, 0.691367f, 0.303900f}}};
// What was lost in red moves to green and blue; what was lost in blue, to red and green.
constexpr Mat3 SHIFT_RED = {{{0, 0, 0}, {0.7f, 1, 0}, {0.7f, 0, 1}}};
constexpr Mat3 SHIFT_BLUE = {{{1, 0, 0.7f}, {0, 1, 0.7f}, {0, 0, 0}}};

// corrected = rgb + shift * (rgb - simulated)
Mat3 correction(const Mat3& simulated, const Mat3& shift) {
  Mat3 lost{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      lost.m[i][j] = (i == j ? 1.0f : 0.0f) - simulated.m[i][j];
  Mat3 r = mul(shift, lost);
  for (int i = 0; i < 3; ++i)
    r.m[i][i] += 1;
  return r;
}

bool s_filter_on = false;
Mat3 s_filter{};

void read_filter_switch() {
  const char* v = std::getenv("AURORA_COLOURBLIND");
  if (v == nullptr || v[0] == 0 || std::strcmp(v, "off") == 0)
    return;
  if (std::strcmp(v, "deuteranopia") == 0)
    s_filter = correction(DEUTAN, SHIFT_RED);
  else if (std::strcmp(v, "protanopia") == 0)
    s_filter = correction(PROTAN, SHIFT_RED);
  else if (std::strcmp(v, "tritanopia") == 0)
    s_filter = correction(TRITAN, SHIFT_BLUE);
  else {
    std::fprintf(stderr, "[aurora-present] AURORA_COLOURBLIND=%s is none of deuteranopia, protanopia, tritanopia, off: off\n", v);
    return;
  }
  s_filter_on = true;
  std::fprintf(stderr, "[aurora-present] colour filter: %s\n", v);
}

wgpu::Texture s_filter_texture;
wgpu::RenderPipeline s_filter_pipeline;
wgpu::BindGroupLayout s_filter_layout;
wgpu::TextureView s_filter_view;
wgpu::BindGroup s_filter_bind;

void make_filter_pipeline() {
  if (s_filter_pipeline)
    return;
  const auto row = [](int i) {
    char text[80];
    std::snprintf(text, sizeof text, "vec3f(%.6f, %.6f, %.6f)", s_filter.m[i][0], s_filter.m[i][1], s_filter.m[i][2]);
    return std::string(text);
  };
  // (the rows go in as the columns of a transposed matrix: dot products by row)
  const std::string code = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
fn to_linear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }
fn to_srgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }
@fragment fn fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let c = to_linear(textureLoad(src, vec2i(at.xy), 0).rgb);
  let r = clamp(vec3f(dot()" + row(0) + R"(, c), dot()" + row(1) + R"(, c), dot()" + row(2) + R"(, c)), vec3f(0.0), vec3f(1.0));
  return vec4f(to_srgb(r), 1.0);
}
)";
  wgpu::ShaderSourceWGSL source{};
  source.code = code.c_str();
  const wgpu::ShaderModuleDescriptor module_desc{.nextInChain = &source, .label = "colour filter"};
  const auto module = webgpu::g_device.CreateShaderModule(&module_desc);
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                            .viewDimension = wgpu::TextureViewDimension::e2D}};
  const wgpu::BindGroupLayoutDescriptor layout_desc{.entryCount = 1, .entries = &entry};
  s_filter_layout = webgpu::g_device.CreateBindGroupLayout(&layout_desc);
  const wgpu::PipelineLayoutDescriptor pipeline_layout{.bindGroupLayoutCount = 1, .bindGroupLayouts = &s_filter_layout};
  const auto layout = webgpu::g_device.CreatePipelineLayout(&pipeline_layout);
  const wgpu::ColorTargetState target{.format = webgpu::present_format(),
                                      .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor pipeline{
      .label = "colour filter",
      .layout = layout,
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs"},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  s_filter_pipeline = webgpu::g_device.CreateRenderPipeline(&pipeline);
}

wgpu::TextureView filter_target(uint32_t width, uint32_t height) {
  if (!s_filter_on)
    return {};
  make_filter_pipeline();
  if (!s_filter_texture || s_filter_texture.GetWidth() != width || s_filter_texture.GetHeight() != height) {
    const wgpu::TextureDescriptor desc{
        .label = "colour filter's picture",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {width, height, 1},
        .format = webgpu::present_format(),
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    s_filter_texture = webgpu::g_device.CreateTexture(&desc);
    s_filter_view = s_filter_texture.CreateView();
    const wgpu::BindGroupEntry entry{.binding = 0, .textureView = s_filter_view};
    const wgpu::BindGroupDescriptor bind{.layout = s_filter_layout, .entryCount = 1, .entries = &entry};
    s_filter_bind = webgpu::g_device.CreateBindGroup(&bind);
  }
  return s_filter_view;
}

void filter_pass(const wgpu::CommandEncoder& encoder, const wgpu::TextureView&, const wgpu::TextureView& surface) {
  const wgpu::RenderPassColorAttachment attachment{
      .view = surface, .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store};
  const wgpu::RenderPassDescriptor desc{.label = "colour filter", .colorAttachmentCount = 1, .colorAttachments = &attachment};
  const auto pass = encoder.BeginRenderPass(&desc);
  pass.SetPipeline(s_filter_pipeline);
  pass.SetBindGroup(0, s_filter_bind, 0, nullptr);
  pass.Draw(3);
  pass.End();
}
}  // namespace

void aurora_overlay_start(const uint8_t* ram) {
  s_ram = ram;
  read_filter_switch();
  webgpu::PresentHook hook;
  hook.capture = capture;
  hook.finalTarget = filter_target;
  hook.finalPass = filter_pass;
  webgpu::set_present_hook(hook);
  gfx::timewarp::set_after_warp([](const wgpu::RenderPassEncoder& pass, const webgpu::Viewport& picture, uint32_t surface_w,
                                   uint32_t surface_h) {
    if (s_last_draw)
      s_last_draw(pass, picture, surface_w, surface_h);
  });
}

extern "C" {
void aurora_present_overlay_texture(uint64_t id, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t part_width,
                                    uint32_t part_height, const uint8_t* rgba, uint32_t row_bytes) {
  if (part_width == 0 || part_height == 0 || rgba == nullptr)
    return;
  TextureOp op;
  op.id = id;
  op.width = width;
  op.height = height;
  op.x = x;
  op.y = y;
  op.part_width = part_width;
  op.part_height = part_height;
  op.rgba.resize(static_cast<std::size_t>(part_width) * part_height * 4u);
  for (uint32_t row = 0; row < part_height; ++row)
    std::memcpy(op.rgba.data() + static_cast<std::size_t>(row) * part_width * 4u, rgba + static_cast<std::size_t>(row) * row_bytes,
                part_width * 4u);
  std::lock_guard g{s_lock};
  s_ops.push_back(std::move(op));
}

void aurora_present_overlay_forget(uint64_t id) {
  TextureOp op;
  op.id = id;
  std::lock_guard g{s_lock};
  s_ops.push_back(std::move(op));
}

void aurora_present_overlay_frame(const AuroraOverlayFrame* frame) {
  if (frame == nullptr)
    return;
  auto f = std::make_shared<Frame>();
  f->width = frame->width;
  f->height = frame->height;
  f->vertices.assign(frame->vertices, frame->vertices + frame->vertex_count);
  f->indices.assign(frame->indices, frame->indices + frame->index_count);
  f->draws.assign(frame->draws, frame->draws + frame->draw_count);
  std::lock_guard g{s_lock};
  s_frame = std::move(f);
}

int aurora_present_picture_in_window(uint32_t window_width, uint32_t window_height, int32_t out[4]) {
  if (!s_has_rect.load(std::memory_order_acquire))
    return 0;
  // (the overlay is in the window's pixels; while a window is dragged the surface is a size behind it, stretched to it)
  const float sx = static_cast<float>(window_width) / static_cast<float>(std::max<uint32_t>(s_rect_surface[0].load(std::memory_order_relaxed), 1u));
  const float sy = static_cast<float>(window_height) / static_cast<float>(std::max<uint32_t>(s_rect_surface[1].load(std::memory_order_relaxed), 1u));
  out[0] = static_cast<int32_t>(std::lround(static_cast<float>(s_rect[0].load(std::memory_order_relaxed)) * sx));
  out[1] = static_cast<int32_t>(std::lround(static_cast<float>(s_rect[1].load(std::memory_order_relaxed)) * sy));
  out[2] = static_cast<int32_t>(std::lround(static_cast<float>(s_rect[2].load(std::memory_order_relaxed)) * sx));
  out[3] = static_cast<int32_t>(std::lround(static_cast<float>(s_rect[3].load(std::memory_order_relaxed)) * sy));
  return 1;
}
}
