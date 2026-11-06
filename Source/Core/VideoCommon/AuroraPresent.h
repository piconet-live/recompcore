// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// AuroraPresent: the game's pictures drawn a second time by Aurora (GXRuntime's aurora-present library), on a
// thread of its own, and shown in place of the emulator's own. Presentation only: the emulated GPU stays this
// emulator's (the command processor, the EFB and its peeks, the draw-done and tokens, the performance counters, the
// copies to RAM): Aurora is handed a copy of the command stream the GPU thread reads, reads the guest's RAM and
// writes nothing the emulated machine can see, and its picture only replaces the XFB's texture when the presenter
// draws the window. Whatever Aurora does, the race is the same.
//  - the feed: every burst the FIFO loop reads, in order (OnBurstRead), cut into frames at the end of each XFB copy's
//    command (OnXFBCopy, OnCommandEnd). The library copies each frame and returns; nothing waits for Aurora;
//  - the picture: the newest one Aurora finished, uploaded into a texture of its own at each present (Picture). A
//    present before Aurora has a picture, or after it stopped (it fell too far behind), shows the emulator's own;
//  - the library is loaded at run time (Windows: LoadLibrary), so this emulator's build never links Dawn or
//    Aurora. With it missing nothing changes but a line in the report.
#pragma once

#include <string>
#include <vector>

#include "Common/CommonTypes.h"
#include "Common/MathUtil.h"
#include "VideoCommon/TextureCacheBase.h"

class AbstractTexture;
struct ImDrawData;
struct ImTextureData;

namespace VideoCommon::AuroraPresent
{
struct Settings
{
  bool on = false;
  std::string library;   // the aurora-present library's path
  std::string data_dir;  // Aurora's own folder (its pipeline and Dawn caches): made by the host first
  // A test: the pictures of these XFB copies (1 the first) saved as <shot_prefix>-aurora-<copy>.png (Aurora's)
  // and <shot_prefix>-dolphin-<copy>.png (the emulator's own XFB, read back without a wait)
  std::string shot_prefix;
  std::vector<u64> shots;
  u32 scale = 0;         // the render scale (0: the emulator's own internal resolution)
  std::string log_path;  // Aurora's thread: a CSV line a frame (empty: none); the presents' beside it (-shown.csv)
};

// The host, before the video thread draws: loads the library (Aurora itself starts at the GPU thread's first burst,
// once the guest's RAM is there). Configure({}) at the end stops it.
void Configure(const Settings& settings);
// True while the library wants the model files the game reads (the disc thread asks).
bool WantsModelFiles();
// Every model on the disc compiled ahead: ray tracing is on (AURORA_RT 1 or 2), or AURORA_WHOLE_DISC=1 asks
bool WantsWholeDisc();
// A model or course file the game is reading: its materials' pipelines compile ahead. Disc bytes only; any thread.
void ModelFile(const std::string& path, const u8* bytes, u64 size, bool later);
// Where the game stands, on any change (the host's: the runner's frame tick): racing from the race's own phase to its end, and a
// short text for Aurora's log. While racing Aurora compiles nothing ahead. Any thread; nothing without the library.
void Scene(bool racing, const std::string& where);
// The Null video backend, at its start: the emulator will draw nothing, so Aurora presents straight into this window
// (a window system's handle; none in a headless run) and no picture comes back.
void SetVideoOffWindow(void* window);
// The frame rate is Aurora's presents' (a video-off run with Aurora on), not the emulator's swaps.
bool CountsFrames();
// The overlay of a video-off run (OnScreenUI's ImGui with nothing of the emulator's to draw on): the window's pixels now,
// false when it has none
bool VideoOffSize(u32* width, u32* height);
// Where Aurora's last present laid the game's picture in the window (false: no present yet).
bool PictureInWindow(MathUtil::Rectangle<int>* rect);
// An ImGui texture's change: `status` is what it was before the emulator's own update (OnScreenUI::Finalize), `id` its id then.
void OverlayTexture(ImTextureData* tex, int status, u64 id);
// The null backend's sampler, set by an ImGui list's callback while OverlayFrame walks the lists
void OverlaySampler(bool smooth);
// The frame's lists, handed to Aurora. Callbacks run here, as in OnScreenUI::DrawImGui.
void OverlayFrame(const ImDrawData* data, u32 width, u32 height);
// The ImGui id of a copy of the game's to the guest's RAM at this physical address: a head of the top 6, say. A frame
// that uses one before the game has made it draws nothing there.
u64 OverlayCopyTexture(u32 physical);
// The emulated CPU's thread: the next XFB copy after the stream position `position` (FrameRedraw::WrittenPosition) is a
// picture between two steps, `fraction` of a field after the true picture before it (the run ahead: fzgx/runahead.h).
// AURORA_REDRAW: the pictures between (the library's AuroraRedrawRequest, the same shape: this build does not include its header). The
// game's thread.
constexpr u32 REDRAW_MOST = 7;
struct RedrawPicture
{
  float fraction;
  float camera[16];
  float machine[30][16];
};
struct RedrawRequest
{
  u64 position;
  u32 machines;
  float at[30][3];
  u32 pictures;
  RedrawPicture picture[REDRAW_MOST];
};
void Redraw(const RedrawRequest* request);
void NoteBetween(u64 position, float fraction);
// The summary for the host's log (empty when it was never on).
std::string Report();
bool On();

// The GPU thread.
void OnBurstRead(const u8* bytes, u32 size);
// TextureCacheBase, the XFB copy's own entry (before OnXFBCopy): the emulator's picture of a test shot.
void OnXFBEntry(const RcTcacheEntry& entry);
// The XFB copy's BP write: Aurora's frame ends at the end of this command (OnCommandEnd), not at the end of the burst
// it came in: a frame handed to Aurora holds whole commands only (it decodes each frame to its last byte).
void OnXFBCopy();
// OpcodeDecoder, a display list called from the FIFO stream: its bytes, as the emulator reads them now. Aurora
// decodes the frame later, when the game may have rewritten the list: the list's bytes take the call's place in
// the stream Aurora is handed (OnCommandEnd).
void OnDisplayList(const u8* bytes, u32 size);
// OpcodeDecoder, the list read: `good` of its bytes were whole commands. The emulator stops a list at a command that
// does not fit it and takes a byte that is no command as one byte; Aurora, with the list in the stream, has no list's
// end to stop at and would read on through whatever follows. A list the game overwrote while the GPU was late (a heap
// block's header, new data from ARAM) is such a list. Aurora gets only the good bytes.
void OnListRead(u32 good);
// OpcodeDecoder, after each command of the FIFO stream itself (not of a display list) while a cut or a list is
// pending: [data, data + size) is the command in the video buffer, whose bytes up to the last burst read were
// handed in already.
extern bool g_pending;
inline bool Pending()
{
  return g_pending;
}
void OnCommandEnd(const u8* data, u32 size);

// The presenter (the GPU thread): Aurora's newest picture, or null (show the emulator's own). `rect` is the
// picture's: the display copy's size.
const AbstractTexture* Picture(MathUtil::Rectangle<int>* rect);
// At the video backend's shutdown: the texture goes.
void Shutdown();
}  // namespace VideoCommon::AuroraPresent
