// The game window's frame, when the host program draws it. A host program (a game's port on ModernGekko) that wants
// the window in its own look, with no title bar and no menu of the system's, gives two functions here before the
// window is made. Unset, which is the default, nothing changes: the window is the system's own, with its menu.
// Only the Windows window asks (DolphinNoGUI/PlatformWin32.cpp). What it keeps for itself whoever draws the frame:
// the keys (full screen on F11 and Alt+Enter, Escape), the commands, closing, and going to full screen and back,
// where the window has no frame at all and the picture is the whole screen.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace WindowFrame
{
// The number of the window's own command "full screen and back" (a WM_COMMAND to the game's window): what a
// full-screen button of the host's frame sends.
constexpr unsigned TOGGLE_FULLSCREEN = 41004;

struct Host
{
  // Every message of the game's window, before the window's own handling. True: `result` is the answer and nothing
  // else looks at the message. False: the window handles it as it always did. The arguments are the window
  // procedure's (HWND, UINT, WPARAM, LPARAM, LRESULT), in plain types so that this file needs no system header.
  bool (*message)(void* window, unsigned message, std::uintptr_t wparam, std::intptr_t lparam,
                  std::intptr_t* result) = nullptr;
  // The surface the picture is drawn on, asked when the video backend is given the window: a child window of the
  // game's window that the host made (the frame's own parts are then not the picture's). Null: the window itself.
  void* (*surface)(void* window) = nullptr;
};

inline Host& Get()
{
  static Host host;
  return host;
}

// The host program draws the frame: the window makes no menu bar of its own.
inline bool Given()
{
  return Get().message != nullptr;
}
}  // namespace WindowFrame
