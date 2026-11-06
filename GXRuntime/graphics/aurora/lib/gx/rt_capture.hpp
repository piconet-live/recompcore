#pragma once
// Ray tracing, first stage (AURORA_RT, default off). Materials come from what the draw itself asks for: the TEV stages,
// the texture generators and the blend in the stream, never a table of models. See rt_capture.cpp.

#include "gx.hpp"
#include "look_keys.hpp"

#include <atomic>

namespace aurora::gx::rt {
extern bool g_neutral;    // AURORA_RT=1: the game's fake reflection stages pass through; rays stand in for them
extern bool g_gameLightOff;  // AURORA_RT_GAME_LIGHT=0: no game lights in the picture (never anything the game reads)
extern bool g_noVertexColour;  // AURORA_RT_VERTEX_COLOUR=0: colour channels ignore vertex colour, to see how much light is baked there
extern float g_ambientWarmth;
extern bool g_gameLit[2];     // with g_gameLightOff: whether the game asked for lights on this colour channel
extern bool g_capturing;  // this frame's draws are being written (AURORA_RT_CAPTURE=<file>, AURORA_RT_CAPTURE_AT=<display copy>)

// AURORA_RT_NO_HUD=1 (for the composite's mask): orthographic draws after the frame's first perspective draw are left out
extern bool g_noHud;
bool skip_draw() noexcept;
// AURORA_RT_NO_ENGINE_STAR=1, with own light (AURORA_RT=1, AURORA_RT_LIGHTING): the game's star sprite on far machines' engines not drawn; AURORA_RT_ENGINE_STAR_FROM=<size>: from what size (4 by default)
extern bool g_noEngineStar;
extern std::atomic<u64> g_starsSkipped;  // stars is_engine_star took, for the report
bool is_engine_star(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data) noexcept;
void note_draw(u16 vtxCount, GXPrimitive prim, GXVtxFmt fmt, const u8* data) noexcept;
void start() noexcept;  // reads the switches, once

// is_reflection_stage and neutralise: look_keys.hpp (the planner works out the same keys)

// The live pass: before each draw (the first orthographic draw after the view is the HUD: the rays go first), and at a copy that
// clears the EFB (what was drawn so far went to a texture, not the picture)
void before_draw() noexcept;
void efb_cleared() noexcept;
extern bool g_live;
extern bool g_hook;  // the rays or the look (AURORA_LOOK_POST=1) want to know where the HUD begins

// The command processor, once a draw's pipeline and textures are settled: `vertices` are the draw's raw bytes
void capture_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* vertices, size_t rawBytes) noexcept;
void capture_sum_rows() noexcept;  // AURORA_STAGELOG: the stream thread's capture parts a frame (st_cap_*), written and cleared
// copy_xfb, before the copy: a frame ends here
void frame_end() noexcept;
// The host's display copy about to be drawn. AURORA_RT_CAPTURE_COPY=<copy>: capture exactly that frame, the one its test picture shows
void begin_copy(std::uint64_t copy) noexcept;
// The game's step a display copy belongs to (the host's aurora_present_step, told on the game's thread), for the step-stamped event
// lines (a slow true frame of the rays', a catch-up, a slow picture between): kept for the last 512 copies. 0xFFFFFFFF: not known.
void copy_step(std::uint64_t copy, std::uint32_t step) noexcept;
std::uint32_t step_of(std::uint64_t copy) noexcept;
} // namespace aurora::gx::rt
