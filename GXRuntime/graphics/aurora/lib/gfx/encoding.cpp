#include "encoding.hpp"
#include "stagelog.hpp"
#include "timewarp.hpp"

#include "frame.hpp"

#include "clear.hpp"
#include "pipeline_cache.hpp"
#include "readback/gpu.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;
using webgpu::g_device;
using webgpu::g_queue;

void (*g_beforeFrameSubmit)() = nullptr;
void (*g_betweenTask)(wgpu::CommandEncoder& cmd, EncoderTaskId type) = nullptr;
void (*g_afterFrameSubmit)() = nullptr;
void (*g_beforePresent)() = nullptr;

namespace {
constexpr Module Log{"aurora::gfx"};
WGPURenderPipeline g_currentPipeline = nullptr;

void apply_viewport(const wgpu::RenderPassEncoder& pass, const Viewport& vp) {
  const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
  const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
  pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
}

void apply_scissor(const wgpu::RenderPassEncoder& pass, const ClipRect& sc, const wgpu::Extent3D& size) {
  const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
  const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
  const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
  const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
  pass.SetScissorRect(x, y, w, h);
}

DrawContext make_draw_context(const RenderPass& passInfo) {
  auto& res = resources();
  return {
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
      .layout = passInfo.target_layout(),
  };
}

void render_custom_draw(const CustomDrawCommand& draw, const wgpu::RenderPassEncoder& pass,
                        const RenderPass& passInfo) {
  const auto drawType = find_runtime_draw_type(draw.type);
  if (!drawType) {
    // Unregistered between record and replay; the command is a no-op.
    return;
  }

  const auto context = make_draw_context(passInfo);
  drawType->draw(context, pass, draw.payload.data(), draw.payloadSize, drawType->userdata);
}

void execute_encoder_task(wgpu::CommandEncoder& cmd, FramePacket& frame, const EncoderTask& task) {
  const auto taskType = find_runtime_encoder_task_type(task.type);
  if (!taskType) {
    // Unregistered between record and encode; the task is a no-op.
    return;
  }

  auto& res = resources();
  const EncoderTaskContext context{
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
      .frameEncoder = &cmd,
  };
  taskType->callback(context, cmd, task.payload.data(), task.payloadSize, taskType->userdata);
  if (taskType->afterSubmit != nullptr) {
    const auto payload = task.payload;
    const auto payloadSize = task.payloadSize;
    const auto callback = taskType->afterSubmit;
    const auto userdata = taskType->userdata;
    frame.afterSubmitCallbacks.emplace_back([payload, payloadSize, callback, userdata] {
      const EncoderTaskCompletionContext completionContext{
          .device = g_device,
          .queue = g_queue,
      };
      callback(completionContext, payload.data(), payloadSize, userdata);
    });
  }
}

// AURORA_REDRAW_BUNDLES: a held frame's pass, its runs of two or more GX draws in a row (nothing between them but more of the same,
// each with its pipeline ready and no blend constant) recorded into render bundles at its first picture between and replayed in each
// after. A bundle reads the uniform buffer as it is when the picture runs, so the patched projections reach it as they reach a draw
// encoded directly. The rest of the pass (viewports, scissors, stencil references, other draws) is encoded as before; a bundle leaves
// the viewport and scissor as they were and the pipeline and bind groups unset.
struct BundleRun {
  uint32_t first = 0, end = 0;  // the pass's commands [first, end)
  uint32_t dstAlpha = UINT32_MAX;  // the blend constant its draws use (the pass's, set before the bundle), or none
  wgpu::RenderBundle bundle;
};
struct PassBundles {
  bool built = false;
  std::vector<BundleRun> runs;
};
std::vector<PassBundles> g_passBundles;  // by the pass's index (the render worker's)
constexpr uint32_t BundleLeast = 2;

bool bundle_draw(const Command& cmd) noexcept {
  return cmd.type == CommandType::Draw && cmd.data.draw.encoder == gx_draw_encoder();
}
const gx::DrawData& gx_data(const Command& cmd) noexcept {
  return *reinterpret_cast<const gx::DrawData*>(cmd.data.draw.payload.data());
}

void build_bundles(PassBundles& out, RenderPass& passInfo) {
  const auto began = std::chrono::steady_clock::now();
  uint64_t bundled = 0, direct = 0, runs = 0;
  out.runs.clear();
  out.built = true;
  const auto layout = passInfo.target_layout();
  std::array<wgpu::TextureFormat, MaxColorAttachments> formats{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    formats[i] = layout.colorAttachments[i].format;
  }
  const wgpu::RenderBundleEncoderDescriptor descriptor{
      .label = "Picture between bundle",
      .colorFormatCount = layout.colorAttachmentCount,
      .colorFormats = formats.data(),
      .depthStencilFormat = layout.depthStencilFormat,
      .sampleCount = layout.sampleCount,
  };
  const auto& commands = passInfo.commands;
  for (uint32_t i = 0; i < commands.size();) {
    if (!bundle_draw(commands[i])) {
      direct += commands[i].type == CommandType::Draw || commands[i].type == CommandType::CustomDraw;
      ++i;
      continue;
    }
    // GX draws in a row whose blend constants agree (draws without one go with any; a bundle cannot set it)
    uint32_t end = i, constant = UINT32_MAX;
    while (end < commands.size() && bundle_draw(commands[end])) {
      const uint32_t a = gx_data(commands[end]).dstAlpha;
      if (a != UINT32_MAX && constant != UINT32_MAX && a != constant) {
        break;
      }
      if (a != UINT32_MAX) {
        constant = a;
      }
      ++end;
    }
    // a draw whose pipeline is not ready ends the run before it (it is drawn directly, as before, and tries its pipeline each picture)
    while (i < end) {
      uint32_t last = i;
      if (end - i >= BundleLeast) {
        auto encoder = g_device.CreateRenderBundleEncoder(&descriptor);
        encoder.SetBindGroup(0, resources().staticBindGroup);
        encoder.SetBindGroup(2, gx::g_emptyTextureBindGroup);
        WGPURenderPipeline current = nullptr;
        uint32_t runConstant = UINT32_MAX;
        while (last < end && gx::render_bundle(gx_data(commands[last]), encoder, current)) {
          if (gx_data(commands[last]).dstAlpha != UINT32_MAX) {
            runConstant = gx_data(commands[last]).dstAlpha;
          }
          ++last;
        }
        if (last - i >= BundleLeast) {
          out.runs.push_back({i, last, runConstant, encoder.Finish()});
          bundled += last - i;
          ++runs;
        } else {
          last = i;  // (too short: drawn directly)
        }
      }
      direct += last == i;
      i = std::max(last, i + 1);
    }
  }
  if (stagelog::on()) {  // (the draws bundled; then the draws left direct, as the count of a stretch of its own)
    stagelog::note("rd_bundles", began, bundled);
    stagelog::note("rd_bundles_direct", began, direct);
    stagelog::note("rd_bundles_runs", began, runs);
  }
}

void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, RenderPass& passInfo, PassBundles* bundles = nullptr) {
  ZoneScoped;
  if (bundles != nullptr && !bundles->built) {
    build_bundles(*bundles, passInfo);
  }
  size_t nextRun = 0;
  g_currentPipeline = nullptr;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> lastDebugGroupStack;
#endif
  Viewport currentViewport{};
  ClipRect currentScissor{};
  bool hasViewport = false;
  bool hasScissor = false;

  // Bind bind group for the whole pass
  pass.SetBindGroup(0, resources().staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);

  for (uint32_t at = 0; at < passInfo.commands.size(); ++at) {
    if (bundles != nullptr && nextRun < bundles->runs.size() && bundles->runs[nextRun].first == at) {
      const auto& run = bundles->runs[nextRun++];
      if (run.dstAlpha != UINT32_MAX) {
        const wgpu::Color color{0.f, 0.f, 0.f, run.dstAlpha / 255.f};
        pass.SetBlendConstant(&color);
      }
      pass.ExecuteBundles(1, &run.bundle);
      at = run.end - 1;
      g_currentPipeline = nullptr;
      pass.SetBindGroup(0, resources().staticBindGroup);
      pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);
      continue;
    }
    auto& cmd = passInfo.commands[at];
#ifdef AURORA_GFX_DEBUG_GROUPS
    {
      size_t firstDiff = lastDebugGroupStack.size();
      for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
        if (i >= cmd.debugGroupStack.size() || cmd.debugGroupStack[i] != lastDebugGroupStack[i]) {
          firstDiff = i;
          break;
        }
      }
      for (size_t i = firstDiff; i < lastDebugGroupStack.size(); ++i) {
        pass.PopDebugGroup();
      }
      for (size_t i = firstDiff; i < cmd.debugGroupStack.size(); ++i) {
        pass.PushDebugGroup(cmd.debugGroupStack[i].c_str());
      }
      lastDebugGroupStack = cmd.debugGroupStack;
    }
#endif
    switch (cmd.type) {
    case CommandType::SetViewport: {
      const auto& vp = cmd.data.setViewport;
      apply_viewport(pass, vp);
      currentViewport = vp;
      hasViewport = true;
    } break;
    case CommandType::SetScissor: {
      const auto& sc = cmd.data.setScissor;
      apply_scissor(pass, sc, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      currentScissor = sc;
      hasScissor = true;
    } break;
    case CommandType::Draw: {
      auto& draw = cmd.data.draw;
      if (draw.encoder != nullptr) {
        draw.encoder(draw.payload.data(), pass, passInfo);
      }
    } break;
    case CommandType::CustomDraw: {
      render_custom_draw(cmd.data.customDraw, pass, passInfo);
      g_currentPipeline = nullptr;
      pass.SetBindGroup(0, resources().staticBindGroup);
      pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);
      if (hasViewport) {
        apply_viewport(pass, currentViewport);
      }
      if (hasScissor) {
        apply_scissor(pass, currentScissor, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      }
    } break;
    case CommandType::SetStencilRef:
      pass.SetStencilReference(cmd.data.setStencilRef);
      break;
    case CommandType::DebugMarker: {
#if defined(AURORA_GFX_DEBUG_GROUPS)
      pass.InsertDebugMarker(wgpu::StringView(frame.debugMarkers[cmd.data.debugMarkerIndex]));
#endif
    } break;
    }
  }

#ifdef AURORA_GFX_DEBUG_GROUPS
  for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
    pass.PopDebugGroup();
  }
#endif
}

void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex, PassBundles* bundles = nullptr) {
  ZoneScoped;
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(cmd, conv);
  }
  if (passInfo.discardable) {
    // This pass has no effect and can be safely discarded (e.g. an empty EFB segment between two back-to-back pass
    // breaks, or an unresolved offscreen pass).
    return;
  }

  std::array<wgpu::RenderPassColorAttachment, MaxColorAttachments> attachments{};
  for (uint32_t i = 0; i < passInfo.colorAttachmentCount; ++i) {
    const auto& source = passInfo.colorAttachments[i];
    attachments[i] = {
        .view = source.view,
        .resolveTarget = source.resolveView,
        .loadOp = source.loadOp != wgpu::LoadOp::Undefined ? source.loadOp
                                                           : (source.clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load),
        .storeOp = source.storeOp,
        .clearValue =
            {
                .r = source.clearValue.x(),
                .g = source.clearValue.y(),
                .b = source.clearValue.z(),
                .a = source.clearValue.w(),
            },
    };
  }
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{};
  const wgpu::RenderPassDepthStencilAttachment* depthStencilAttachmentPtr = nullptr;
  if (passInfo.depthStencilView) {
    depthStencilAttachment = {
        .view = passInfo.depthStencilView,
        .depthLoadOp = passInfo.hasDepth ? (passInfo.depthLoadOp != wgpu::LoadOp::Undefined
                                                ? passInfo.depthLoadOp
                                                : (passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load))
                                         : wgpu::LoadOp::Undefined,
        .depthStoreOp = passInfo.hasDepth ? passInfo.depthStoreOp : wgpu::StoreOp::Undefined,
        .depthClearValue = passInfo.clearDepthValue,
        .stencilLoadOp = passInfo.hasStencil ? passInfo.stencilLoadOp : wgpu::LoadOp::Undefined,
        .stencilStoreOp = passInfo.hasStencil ? passInfo.stencilStoreOp : wgpu::StoreOp::Undefined,
        .stencilClearValue = passInfo.stencilClearValue,
    };
    if (passInfo.depthStencilFormat == wgpu::TextureFormat::Depth32FloatStencil8) {
      // the warp's pinning: the stencil lives and clears with the depth
      depthStencilAttachment.stencilLoadOp = depthStencilAttachment.depthLoadOp;
      depthStencilAttachment.stencilStoreOp = wgpu::StoreOp::Store;
      depthStencilAttachment.stencilClearValue = 0;
    }
    depthStencilAttachmentPtr = &depthStencilAttachment;
  }
  const auto label = passInfo.label.empty() ? fmt::format("Render pass {}", passIndex)
                                            : fmt::format("{} {}", passInfo.label, passIndex);
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = label.c_str(),
      .colorAttachmentCount = passInfo.colorAttachmentCount,
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };

  auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  render_pass(pass, frame, passInfo, bundles);
  pass.End();

  if (passInfo.resolveTarget) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    if (isDepth && passInfo.msaaSamples > 1) {
      Log.fatal("Depth tex copies from multisampled EFB targets are not supported");
    }
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcFmt = passInfo.resolveSourceFormat,
        .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
    };
    const bool needsOpaqueAlpha = !isDepth && !gx::efb_has_alpha(convReq.srcFmt);
    const bool sameFormat =
        passInfo.resolveTarget->format == passInfo.colorAttachments[SceneColorAttachmentIndex].format;
    if (needsConversion) {
      tex_copy_conv::run(cmd, convReq);
      if (passInfo.resolveFormat == tex_copy_conv::XfbCopyFormat) {
        // The display copy is what the hardware shows: present it (render worker, as the present and readback)
        webgpu::set_present_override(passInfo.resolveTarget->texture, passInfo.resolveTarget->sampleTextureView,
                                     dstSize, passInfo.resolveTarget->format);
        if (timewarp::g_on) {
          timewarp::capture(cmd, passInfo.resolveTarget->texture, passInfo.resolveTarget->sampleTextureView, dstSize,
                            passInfo.copySourceDepthView, passInfo.msaaSamples,
                            passInfo.colorAttachments[SceneColorAttachmentIndex].size, passInfo.resolveRect.x,
                            passInfo.resolveRect.y, passInfo.resolveRect.width, passInfo.resolveRect.height);
        }
      }
    } else if (needsScaling || needsOpaqueAlpha || !sameFormat) {
      tex_copy_conv::blit(cmd, convReq);
    } else {
      const webgpu::gpu_prof::Zone zone{cmd, "EFB copy"};
      const wgpu::TexelCopyTextureInfo src{
          .texture = passInfo.copySourceTexture,
          .origin =
              wgpu::Origin3D{
                  .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                  .y = static_cast<uint32_t>(passInfo.resolveRect.y),
              },
      };
      const wgpu::TexelCopyTextureInfo dst{
          .texture = passInfo.resolveTarget->texture,
      };
      const wgpu::Extent3D size{
          .width = static_cast<uint32_t>(passInfo.resolveRect.width),
          .height = static_cast<uint32_t>(passInfo.resolveRect.height),
          .depthOrArrayLayers = 1,
      };
      cmd.CopyTextureToTexture(&src, &dst, &size);
    }
  }

  if (passInfo.snapshotColorDst) {
    const webgpu::gpu_prof::Zone zone{cmd, "Pass snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = passInfo.copySourceTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotColorDst,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
  if (passInfo.snapshotDepthDst) {
    tex_copy_conv::snapshot_depth(cmd, passInfo.copySourceDepthView, passInfo.msaaSamples, passInfo.snapshotDepthDst);
  }
  if (passInfo.snapshotNormalDst) {
    const webgpu::gpu_prof::Zone zone{cmd, "Normal snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = passInfo.copySourceNormalTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotNormalDst,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
}

constexpr uint64_t VertexStagingOffset = 0;
constexpr uint64_t UniformStagingOffset = VertexStagingOffset + VertexBufferSize;
constexpr uint64_t IndexStagingOffset = UniformStagingOffset + UniformBufferSize;
constexpr uint64_t StorageStagingOffset = IndexStagingOffset + IndexBufferSize;
constexpr uint64_t TextureUploadStagingOffset = StorageStagingOffset + StorageBufferSize;

constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~3u; }

void copy_staging_buffer_range(wgpu::CommandEncoder& cmd, const FramePacket& frame, uint32_t& copied,
                               uint32_t highWater, uint64_t stagingOffset, const wgpu::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd = AURORA_ALIGN(highWater, 4);
  cmd.CopyBufferToBuffer(staging_buffer(frame.stagingBuffer), stagingOffset + copyStart, dst, copyStart,
                         copyEnd - copyStart);
  copied = highWater;
}

bool needs_staging_copy(const FramePacket& frame, const FrameOp& op) {
  const auto& highWater = op.highWater;
  if (highWater.verts > frame.copied.verts || highWater.uniforms > frame.copied.uniforms ||
      highWater.indices > frame.copied.indices || highWater.storage > frame.copied.storage) {
    return true;
  }
  if constexpr (UseTextureBuffer) {
    return op.textureUploads.size() > frame.copied.textureUploadCount;
  }
  return false;
}

void copy_staging_to_high_water(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "Staging copies"};
  const auto& highWater = op.highWater;
  auto& res = resources();
  copy_staging_buffer_range(cmd, frame, frame.copied.verts, highWater.verts, VertexStagingOffset, res.vertexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.uniforms, highWater.uniforms, UniformStagingOffset,
                            res.uniformBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.indices, highWater.indices, IndexStagingOffset, res.indexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.storage, highWater.storage, StorageStagingOffset,
                            res.storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      const wgpu::TexelCopyBufferInfo buf{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .offset = item.buffer ? item.layout.offset : item.layout.offset + TextureUploadStagingOffset,
                  .bytesPerRow = AURORA_ALIGN(item.layout.bytesPerRow, 256),
                  .rowsPerImage = item.layout.rowsPerImage,
              },
          .buffer = item.buffer ? item.buffer : staging_buffer(frame.stagingBuffer),
      };
      cmd.CopyBufferToTexture(&buf, &item.tex, &item.size);
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}
} // namespace

namespace detail {
bool g_redrawBundles = [] {
  const char* v = std::getenv("AURORA_REDRAW_BUNDLES");
  return v != nullptr && v[0] == '1';
}();
void forget_redraw_bundles() noexcept { g_passBundles.clear(); }

void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  copy_staging_to_high_water(cmd, frame, op);
  switch (op.type) {
  case FrameOpType::RenderPass:
    if (op.renderPass != nullptr) {
      render(cmd, frame, *op.renderPass, op.index);
    }
    break;
  case FrameOpType::TextureCopy:
    if (op.textureCopy != nullptr) {
      const webgpu::gpu_prof::Zone zone{cmd, "Texture copy"};
      cmd.CopyTextureToTexture(&op.textureCopy->src, &op.textureCopy->dst, &op.textureCopy->size);
    }
    break;
  case FrameOpType::EncoderTask:
    if (op.encoderTask != nullptr) {
      execute_encoder_task(cmd, frame, *op.encoderTask);
    }
    break;
  case FrameOpType::Readback:
    if (op.readbackTask != nullptr) {
      readback::gpu::encode_job(cmd, frame, op.readbackTask->job);
    }
    break;
  }
}
// The same again for a picture between: nothing uploaded, nothing read back; a ray tracing task's job is gone
void encode_op_again(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  switch (op.type) {
  case FrameOpType::RenderPass:
    if (op.renderPass != nullptr) {
      PassBundles* bundles = nullptr;
      if (g_redrawBundles) {
        if (g_passBundles.size() <= op.index) {
          g_passBundles.resize(op.index + 1);
        }
        bundles = &g_passBundles[op.index];
      }
      render(cmd, frame, *op.renderPass, op.index, bundles);
    }
    break;
  case FrameOpType::TextureCopy:
    if (op.textureCopy != nullptr) {
      cmd.CopyTextureToTexture(&op.textureCopy->src, &op.textureCopy->dst, &op.textureCopy->size);
    }
    break;
  case FrameOpType::EncoderTask:
    if (g_betweenTask != nullptr) {
      stagelog::Scope t{"rd_task"};
      g_betweenTask(cmd, op.encoderTask != nullptr ? op.encoderTask->type : InvalidEncoderTask);
    }
    break;
  default:
    break;
  }
}
} // namespace detail

void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass) {
  if (pipeline.Get() == g_currentPipeline) {
    return;
  }
  pass.SetPipeline(pipeline);
  g_currentPipeline = pipeline.Get();
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) {
  CompiledPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    return false;
  }
  AURORA_ASSERT(!pipeline.prepass, "Prepass pipelines require individual binding");
  bind_pipeline(pipeline.main, pass);
  return true;
}
} // namespace aurora::gfx
