#pragma once

#include "../gfx/types.hpp"
#include "gx.hpp"

#include <webgpu/webgpu_cpp.h>

// AURORA_RT_SOFT_CARDS=1 (off by default): soft effect cards. An effect card (blended, depth tested, writing no depth: flames, exhaust, glows,
// smoke) that cuts into the road, a machine or any other surface shows a hard straight seam where it meets it. With this its contribution fades
// by how near it lies to the opaque depth behind it, over a small distance that grows with its own distance (the standard soft-particle fade).
// Only such cards, chosen by draw state alone; opaque draws and anything drawn orthographic (the HUD) never change. A look change, the picture only.
//
// The depth behind: a snapshot of the frame's depth, taken by a pass break before the first card after depth was written (gfx::resolve_pass).
// A card drawn after more depth was written takes a new one, up to a few a frame; past that, the last one, which lies behind or at what is there
// now (depth only comes nearer), so a card can only fade less than it should, never wrongly.
namespace aurora::gx::soft {
// the switch, read once
bool on() noexcept;
// What the fade scales for the draw state now (any projection; the shader leaves orthographic draws alone): 0 none, 1 the alpha (source factor
// SRCALPHA: alpha blends and additive by alpha), 2 the colour (source factor ONE, destination ONE: plain additive). Kept in ShaderConfig::softCard.
u8 kind(GXPrimitive primitive) noexcept;
// From the command stream, before each draw is recorded: counts depth writes; before a card drawn in perspective, takes a new snapshot when one is due
void before_draw(GXPrimitive primitive) noexcept;
// The EFB was cleared (a copy with clear, the display copy): the snapshot is gone
void efb_cleared() noexcept;
// group 3 of a card's pipeline: the depth behind it (the latest snapshot, or one that is nothing but far)
gfx::BindGroupRef bind_group() noexcept;
// the pipeline layout of a card: the GX layout's three groups and the depth's
const wgpu::PipelineLayout& pipeline_layout() noexcept;
// the fade's share of the card's own distance, for its uniform
float share() noexcept;
void initialize(const wgpu::BindGroupLayout& staticLayout, const wgpu::BindGroupLayout& uniformLayout, const wgpu::BindGroupLayout& textureLayout,
                uint32_t immediateSize) noexcept;
void shutdown() noexcept;
} // namespace aurora::gx::soft
