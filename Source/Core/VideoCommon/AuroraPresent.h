// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// AuroraPresent: the game's pictures drawn a second time by Aurora (GXRuntime's aurora-present library), on a
// thread of its own, and shown in place of the emulator's own. Presentation only: the emulated GPU stays this
// emulator's (the command processor, the EFB and its peeks, the draw-done and tokens, the performance counters, the
// copies to RAM): Aurora is handed a copy of the command stream the GPU thread reads, reads the guest's RAM and
// writes nothing the emulated machine can see, and its picture only replaces the XFB's texture when the presenter
// draws the window. Whatever Aurora does, the race is the same.
//  - the feed: every burst the FIFO loop reads, in order (OnBurstRead), and each copy to the XFB (OnXFBCopy). The
//    library copies them and returns; nothing waits for Aurora;
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
  std::string log_path;  // Aurora's thread: a CSV line a frame (empty: none); the presents' beside it (-shown.csv)
};

// The host, before the video thread draws: loads the library (Aurora itself starts at the GPU thread's first burst,
// once the guest's RAM is there). Configure({}) at the end stops it.
void Configure(const Settings& settings);
// The summary for the host's log (empty when it was never on).
std::string Report();
bool On();

// The GPU thread.
void OnBurstRead(const u8* bytes, u32 size);
// TextureCacheBase, the XFB copy's own entry (before OnXFBCopy): the emulator's picture of a test shot.
void OnXFBEntry(const RcTcacheEntry& entry);
void OnXFBCopy();

// The presenter (the GPU thread): Aurora's newest picture, or null (show the emulator's own). `rect` is the
// picture's: the display copy's size.
const AbstractTexture* Picture(MathUtil::Rectangle<int>* rect);
// At the video backend's shutdown: the texture goes.
void Shutdown();
}  // namespace VideoCommon::AuroraPresent
