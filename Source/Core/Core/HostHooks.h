// What a host program (a game's port on ModernGekko) adds to the emulator core at fixed points. Everything is unset by
// default, and then nothing in the core changes. Set before the core boots; read on the CPU thread.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <vector>

#include "Common/CommonTypes.h"
#include "Core/Cheats/GeckoCode.h"

namespace Core
{
class CPUThreadGuard;
}

namespace HostHooks
{
struct Hooks
{
  // Boot: the size of main memory the game is told (the word at 0x80000028). 0: the emulated size.
  u32 reported_ram_size = 0;
  // MMU: main memory from reported_ram_size up to the emulated size stays mapped at its own addresses, for data and
  // for code, whatever the game does with its BATs. (A game told it has retail memory maps only that much.)
  bool map_ram_above_reported = false;
  // PatchEngine::LoadPatches: more Gecko codes, after the ones of the game's settings.
  std::function<std::vector<Gecko::GeckoCode>()> gecko_codes;
  // PatchEngine::ApplyFramePatches: every frame, before the Gecko code handler runs.
  std::function<void(const Core::CPUThreadGuard&)> frame;
  // HLE: a function of the host's at one guest address; it replaces the code there and returns to the caller.
  // Under the StaticRecomp core a host call at the same address is taken first (StaticRecompModuleSource).
  u32 call_address = 0;
  std::function<void(const Core::CPUThreadGuard&)> call;
  // A window of guest addresses [window_start, window_end) that has no guest memory behind it and is the host
  // program's own: every read and write of it, and every instruction fetched from it, is answered by these two
  // (value and size as the MMU's: big-endian values of 1, 2, 4 or 8 bytes) before any translation. Code there
  // exists only as native code of a StaticRecomp module, whose pieces inside the window count as checked: nothing
  // else could run them. The bytes behind the window must hold that code's words all the same: an instruction the
  // module hands to the interpreter (after an exception that returns into the middle of a piece) is fetched here.
  u32 window_start = 0;
  u32 window_end = 0;
  std::function<u64(u32 address, u32 size)> window_read;
  std::function<void(u32 address, u64 value, u32 size)> window_write;
  bool InWindow(u32 address) const { return address - window_start < window_end - window_start; }
  // VideoCommon/TextureCacheBase.cpp, on the thread that draws (so: safe to call from any thread): a copy out of the
  // frame buffer is about to write its destination in the game's memory, `size` bytes from the physical address
  // `address`. Asked for every such write: the clear of a copy kept only as a texture (a copy for a texture is
  // zeroed, a copy for the display filled with its key colour), and the pixels of a copy written to memory
  // (WriteEFBCopyToRAM, at once or put off to FlushEFBCopy; the "store copies to RAM" settings, or a backend that
  // keeps no copy as a texture). False: the memory is left as it is, and the copy is kept under the hash of what
  // stands there (a copy with no texture is then read from that memory, as it stands). The write lands when the
  // drawing thread comes to the copy, not when the game asked for it: a host that puts the game's memory back to
  // an earlier moment meanwhile (a rollback) has other things at those bytes by then.
  std::function<bool(u32 address, u32 size)> clear_copy_memory;
  // HW/SI (CSIDevice_GCController::Rumble, every rumble command of a standard controller or the adapter), on the CPU
  // thread: whether pad `pad`'s motor may run now. False: a command that runs a motor is passed on as one that stops
  // it, so no motor runs (a stop goes through as it is). Unset: every command goes through.
  std::function<bool(int pad)> rumble_allowed;
};

inline Hooks& Get()
{
  static Hooks hooks;
  return hooks;
}
}  // namespace HostHooks
