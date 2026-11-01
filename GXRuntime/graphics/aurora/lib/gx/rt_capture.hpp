#pragma once
// Ray tracing, first stage (AURORA_RT, default off). Materials come from what the draw itself asks for: the TEV stages,
// the texture generators and the blend in the stream, never a table of models. See rt_capture.cpp.

#include "gx.hpp"

namespace aurora::gx::rt {
extern bool g_neutral;    // AURORA_RT=1: the game's fake reflection stages pass through; rays stand in for them
extern bool g_capturing;  // this frame's draws are being written (AURORA_RT_CAPTURE=<file>, AURORA_RT_CAPTURE_AT=<display copy>)

// AURORA_RT_NO_HUD=1 (for the composite's mask): orthographic draws after the frame's first perspective draw are left out
extern bool g_noHud;
bool skip_draw() noexcept;
void start() noexcept;  // reads the switches, once

// A stage is a reflection layer when it adds or multiplies a texture looked up by the normal (a sphere map), with a
// constant strength: fn_80075908's reflection layers, for any model.
bool is_reflection_stage(const TevStage& stage, const std::array<TcgConfig, MaxTexCoord>& tcgs) noexcept;
void neutralise(ShaderConfig& config) noexcept;

// The live pass: before each draw (the first orthographic draw after the view is the HUD: the rays go first), and at a copy that
// clears the EFB (what was drawn so far went to a texture, not the picture)
void before_draw() noexcept;
void efb_cleared() noexcept;
extern bool g_live;
extern bool g_hook;  // the rays or the look (AURORA_LOOK_POST=1) want to know where the HUD begins

// The command processor, once a draw's pipeline and textures are settled: `vertices` are the draw's raw bytes
void capture_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* vertices, size_t rawBytes) noexcept;
// copy_xfb, before the copy: a frame ends here
void frame_end() noexcept;
// The host's display copy about to be drawn. AURORA_RT_CAPTURE_COPY=<copy>: capture exactly that frame, the one its test picture shows
void begin_copy(std::uint64_t copy) noexcept;
} // namespace aurora::gx::rt
