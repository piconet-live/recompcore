#pragma once

#include "frame_packet.hpp"

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);
// The encoder of a recorded GX draw (gx::render's): the draws a picture between's render bundles take (recording.cpp)
detail::DrawEncoder gx_draw_encoder() noexcept;

namespace detail {
void encode_op(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
void encode_op_again(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
// AURORA_REDRAW_BUNDLES=1 (off by default): a picture between's runs of GX draws from render bundles recorded once for the held frame
extern bool g_redrawBundles;
// A new frame is held: its bundles are recorded again at its first picture
void forget_redraw_bundles() noexcept;
// AURORA_REDRAW_SURVEY: the held frame's ops to stderr (after its bundles were built)
void survey_held(const FramePacket& frame);
}

} // namespace aurora::gfx
