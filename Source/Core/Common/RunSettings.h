// The settings of one run that the host program hands to the emulator core: the test tools of the F-Zero GX port (the
// probe, the pad script, the pokes, the logs and profilers, the CPU core's comparison switches) and the two things
// its native module is built for. Each was an environment variable and keeps that name here.
// A host program that starts the game from a launch file calls RuleByLaunchFile() before anything else, then gives
// each setting its file names (Set). From that call on nothing is taken from the environment: a variable left in a
// player's environment changes nothing, and a test setting is on only when the launch file says so.
// A start that no launch file rules (the emulator alone; a test harness's reference run) is answered from the
// environment under the same names, as it was.
// Every such setting is read through Get() and nowhere else: a new one is a name given to Get(), never a getenv of
// its own. Written on the main thread before the core boots (no value is read before main(): none is a static
// initialiser's); read on any thread afterwards.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <functional>
#include <map>
#include <string>

namespace RunSettings
{
struct State
{
  bool ruled = false;
  std::map<std::string, std::string, std::less<>> given;
};

inline State& TheState()
{
  static State state;
  return state;
}

// A launch file rules this start: only what Set() gives is answered from here on.
inline void RuleByLaunchFile()
{
  TheState().ruled = true;
}

inline bool Ruled()
{
  return TheState().ruled;
}

inline void Set(const std::string& name, const std::string& value)
{
  TheState().given[name] = value;
}

// The setting's value, or nullptr when this run has none.
inline const char* Get(const char* name)
{
  const State& state = TheState();
  if (!state.ruled)
    return std::getenv(name);
  const auto it = state.given.find(name);
  return it == state.given.end() ? nullptr : it->second.c_str();
}

// The user folder's own settings file for the game (GameSettings/<game id>.ini: settings that win over the main
// ones, Gecko and Action Replay codes, memory patches applied every frame whether cheats are on or not) is read:
// always in a start no launch file rules; under a launch file only when the host program gave this name.
inline bool UserGameSettings()
{
  return !Ruled() || Get("FZGX_USER_GAME_SETTINGS") != nullptr;
}

// A file of the emulator's folders may change the game's memory or code: the every-frame patches and the codes of a
// settings file for the game, the system folder's (Sys/GameSettings) as much as the user folder's, and the user
// folder's map file, by which a game function can be given the name of one the emulator replaces. On the same
// word as the user's game settings: always in a start no launch file rules, under a launch file only with it.
inline bool FilesMayPatch()
{
  return UserGameSettings();
}
}  // namespace RunSettings
