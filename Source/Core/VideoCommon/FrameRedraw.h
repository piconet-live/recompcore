// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw: a frame's converted draws, recorded as the GPU thread makes them, and drawn again into a second EFB.
//
// A host that shows pictures between two of the game's (a high refresh rate) can draw the last frame again with
// corrected matrices instead of running the game's draw code again. This is the record and the replay, in
// VideoCommon so that every backend has it:
//  - the record: every GX draw as VertexManagerBase submits it (its pipeline, its vertices and indices, the
//    vertex, geometry and pixel shader constants when they changed, the bound textures and samplers, the viewport
//    and scissor), every EFB clear, every EFB copy to a texture and every EFB format reinterpretation, in order,
//    from one copy to the XFB to the next. The textures stay alive by their cache entries;
//  - the replay: the same calls into a second EFB of the same formats and size (FramebufferManager swaps it in
//    for the replay's time), the copies into scratch textures that later draws of the replay sample instead of
//    the true copy's, nothing written to the texture cache, the emulated EFB or the game's memory;
//  - the compare (a test): the true EFB and the replay's read back over the XFB copy's rectangle, and the pixels
//    that differ counted.
// Nothing the emulated machine can see changes: the replay runs on the GPU thread at the XFB copy, after the true
// frame, or at the next draw-done (the spike's settings), or between two passes of the FIFO loop (the high refresh
// rate), and puts every piece of state it touched back.
//
// The high refresh rate (Settings::hfr): the host asks for the pictures after each of the game's with a request a
// pass (Queue), made on the emulated CPU's thread once the pass's step is known and before its frame's draws, and
// keyed by the command stream's position then (WrittenPosition: the gather pipe's bursts written). The XFB copy that
// closes that frame takes it (the request made after the frame before's draws and before this frame's), and the
// frame is drawn again once the GPU has read the game's list to its end, between two passes of the FIFO loop or at its
// idle point (BetweenPasses: no read pointer held, so the game's switch of its command lists neither waits for the
// replays nor races with them, and no draw-done the game waits at covers them), and the replays give way the moment
// the game hands the GPU more (its next list): the game's own draws are never later than without them (the game
// reuses memory its list points to on that timing: the high-refresh redraw notes, "Stage 5: the fault"); once for each of the request's fractions, each a guess forward from the step (the host's
// Request::predict, asked just before each picture is drawn with the fraction its present time gives: a picture that
// would be ready after its planned moment is a guess for when it will be shown, or left out), each draw with its
// correction (a 4x4 the projection is multiplied by, as Free Look
// does): the world's perspective draws by the camera's, a draw at a machine's place (its position matrix's
// translation within Settings::radius of the machine's in the step's view) by that machine's; a draw whose vertices
// the game put in the view itself (an identity position matrix: engine trails and flames, sparks) by the camera's,
// its vertices near a machine first moved by that machine's own move, fully within 3 units, none from 8 (so a
// trail's head stays on its machine and its older points where they were), but by that machine's whole when all of
// them lie within 10 units of one machine (its burner's glow, its sparks, its shadow on the ground); a model scaled 2
// times or more standing 4 to 40 units straight up from a machine (its rank icon, which the game draws later from its
// depth-sorted list) by that machine's; orthographic draws (the HUD) by none,
// but those over a machine that moved with it on the screen since the frame before (its name tag, its rank icon) by
// its projected move; draws inside an offscreen pass (a viewport inside a smaller EFB copy's source that follows it:
// shadow pictures, portraits) by none. Each replay ends with the XFB copy the true frame made, into a texture of its
// own that the presenter shows at the host's time for its fraction (Presenter::PresentInBetween).
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "Common/CommonTypes.h"
#include "Common/MathUtil.h"
#include "VideoCommon/RenderState.h"
#include "VideoCommon/TextureCacheBase.h"

class AbstractPipeline;
enum class EFBReinterpretType;
enum class PixelFormat : u32;

namespace VideoCommon::FrameRedraw
{
struct Settings
{
  u32 replays = 0;        // replays of every frame, at its XFB copy (0: no record)
  u32 compare_every = 0;  // every this many frames the first replay is compared with the true EFB (0: never)
  std::string log_path;   // a CSV line for every frame (empty: none)
  std::string dump_prefix;  // the first differing compares' pictures, as PPM files (empty: none)
  // A camera correction for the spike: every replay's perspective draws seen from a camera turned by this many
  // degrees about its own up axis (the projection times one 4x4, as Free Look does; orthographic draws, the HUD,
  // untouched). 0: none, the replay as recorded.
  float turn_degrees = 0;
  // The video thread's time the replays of one frame may take: no further replay once the next, at the cost of
  // the last, would pass it (0: no limit; the first replay is always made).
  // (The high refresh rate's: 10 ms when 0. Measured 2026-10-18, course 37, 30 racers, Vulkan at 240 Hz: 6 ms left
  // out 10% of the pictures between, 10 ms 0.2%; fields over 17.5 ms 242 and 296-303 of 3,210, 94 at 60 Hz.)
  float budget_ms = 0;
  // The replays at the first draw-done after the XFB copy (before it is signalled, so that the game's wait for it
  // covers them) instead of at the copy; no compares then.
  bool at_draw_done = false;
  // The high refresh rate: replays only for the host's requests (Queue), after the XFB copy once the game's list is
  // read to its end, presented; `replays` is then the most a pass (no compares).
  bool hfr = false;
  // A field of the game's, in microseconds: a fraction's time after the step's own picture.
  float field_us = 16683.0f;
  // How near a draw's position matrix must be to a machine's place (in the old view) to be that machine's.
  float radius = 6.0f;
  // A test: the frame replayed for this request (1 the first) has its draws written to
  // dump_prefix-draws-N.csv: each draw's correction, position matrix, and its vertices' place in the view against
  // the nearest machine (0: none).
  u64 draws_at = 0;
  // A test: the pictures of `stills` frames from the XFB copy numbered `stills_at` (1 the first copy) saved as PNG
  // files beside dump_prefix: each copy's true picture as `<dump>-still-<copy>-0.png` and the pictures drawn again
  // after it as `-1.png` on, with a line each in `<dump>-stills.csv` (its fraction planned and drawn, its video
  // thread time), and again every `stills_every` copies from there (0: once). Nothing waits for the GPU (each
  // picture is read back four copies after it was drawn, its file written on a thread of its own) (0: none).
  u64 stills_at = 0;
  u32 stills = 0, stills_every = 0;
};

// The high refresh rate's request for one pass (Settings::hfr).
constexpr u32 MOST_MACHINES = 30;
using Matrix = std::array<float, 16>;  // row-major, column vectors (Common::Matrix44's order)
struct InBetween
{
  float fraction = 0;  // of a field after the step's own picture
  Matrix camera{};     // the world's correction
  std::array<Matrix, MOST_MACHINES> machine{};  // machine i's (the camera's when it does not move apart)
  // machine i's own move in the old view: what the game puts in the view itself (an identity position matrix: its
  // engine trails and flames, sparks) is moved vertex by vertex, a vertex near machine i by its move, fading to
  // none at a distance (the world's correction then applies, as to every perspective draw)
  std::array<Matrix, MOST_MACHINES> motion{};
};
struct Request
{
  u64 position = 0;  // WrittenPosition() when it was made, on the emulated CPU's thread
  u32 machines = 0;  // a bit for each machine with a place
  std::array<std::array<float, 3>, MOST_MACHINES> at{};  // machine i's place in the step's own view
  std::vector<float> fractions;  // the pictures' planned times, in fractions of a field after the step's own, in order
  // The corrections of a picture `fraction` of a field after the step's own: a guess forward from the step. Asked on
  // the GPU thread just before the picture is drawn (late: with the fraction of the time it will be shown at).
  std::function<void(float fraction, InBetween* out)> predict;
};
// The emulated CPU's thread: a pass's request, after its step and before its frame's draws.
void Queue(Request request);
// The command stream's position: the gather pipe's bursts written so far (the emulated CPU's thread).
u64 WrittenPosition();
// CommandProcessor (a burst written) and the FIFO loop (a burst read, before it is decoded).
void OnBurstWritten();
void OnBurstRead();

// From the host, before the video thread draws (and at any time after: the next frame takes it).
void Configure(const Settings& settings);
// The summary for the host's log at the end (empty when it was never on).
std::string Report();
// At the video backend's shutdown, before the texture cache goes: lets the record's textures go.
void Shutdown();

// Recording, from VideoCommon (GPU thread). Each is a no-op unless a record is kept, and during a replay.
bool Recording();
void OnBindTextures(BitSet32 used, const std::array<RcTcacheEntry, 8>& bound,
                    const std::array<SamplerState, 8>& samplers);
// The vertex, geometry and pixel shader constants the next draw uploads (those that changed, and all of them at a
// frame's first draw).
void OnConstants();
void OnDraw(const AbstractPipeline* pipeline, const u8* vertices, u32 num_vertices, u32 stride,
            const u16* indices, u32 num_indices, bool expand_base_vertex);
void OnClear(const MathUtil::Rectangle<int>& rc, bool color, bool alpha, bool z, u32 color_value,
             u32 z_value, PixelFormat format);
// The XFB copy's own (TextureCacheBase, before OnXFBCopy): what a replay's picture copies the same way.
void OnXFBEntry(const RcTcacheEntry& entry, bool is_depth_copy, const MathUtil::Rectangle<int>& src_rect,
                bool linear_filter, float gamma, bool clamp_top, bool clamp_bottom,
                const std::array<u32, 3>& filter_coefficients, u32 fb_width, u32 fb_height);
void OnCopy(const RcTcacheEntry& entry, bool is_depth_copy, const MathUtil::Rectangle<int>& src_rect,
            bool scale_by_half, bool linear_filter, EFBCopyFormat dst_format, bool is_intensity,
            float gamma, bool clamp_top, bool clamp_bottom,
            const std::array<u32, 3>& filter_coefficients);
void OnReinterpret(EFBReinterpretType type);
// The copy to the XFB: the frame ends here (src_rect in EFB coordinates). Replays it, compares, starts the next.
void OnXFBCopy(const MathUtil::Rectangle<int>& src_rect);
// The game's draw-done (BPMEM_SETDRAWDONE), before it is signalled.
void OnDrawDone();
// The FIFO loop between two passes (its read pointer stored, no pass in flight) and when it has nothing to read: the
// high refresh rate's replays of the frame the last XFB copy closed, once, when the game's list has nothing left for
// the GPU to read. Cheap when there are none.
void BetweenPasses();
}  // namespace VideoCommon::FrameRedraw
