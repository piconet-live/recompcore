// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// Mode B1 pixel replay: replays a .dolt trace through the LIVE
// Aurora path (window + GPU) and emits one pixel-digest line per frame from
// EFB readbacks. Compiled into dolgx_replay only when the tool is built in
// an Aurora-enabled tree (DOLGX_REPLAY_HAS_PIXELS); the headless build keeps
// Mode A only.

#include <cstdint>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
// POSIX setenv/unsetenv for the replay mains (Windows has _putenv_s).
inline int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr)
    return 0;
  return _putenv_s(name, value);
}
inline int unsetenv(const char* name) { return _putenv_s(name, ""); }
#endif

struct PixelReplayOptions {
  const char* golden_path = nullptr;  // exact line-compare against a golden
  const char* write_path = nullptr;   // write digest lines here
  const char* png_dir = nullptr;      // dump PNGs here
  unsigned png_every = 1;             // dump every Nth frame (and the last)
  bool quiet = false;
  unsigned window_width = 1024;       // --window WxH (framebuffer size)
  unsigned window_height = 768;
  int viewport_policy = -1;           // --viewport fit|stretch|native (-1: Aurora default)
  unsigned repeat = 1;                // --repeat N: replay the whole trace N times
  bool timing = false;                // --timing: per-frame CPU/submit/readback times
  unsigned burst = 0;                 // --burst N: GX writes handed on N bytes at a time, as the
                                      // gather pipe does (32); 0: one write at a time
};

// --burst: GX_WRITE records gathered into runs of N bytes (big-endian, as the
// gather pipe writes them), handed on when full and before any other record.
struct BurstGather {
  unsigned cap = 0;
  unsigned n = 0;
  std::uint8_t bytes[256] = {};
  template <typename Sink>
  bool add(std::uint64_t value, unsigned size, Sink&& sink) {
    for (unsigned i = 0; i < size; ++i) {
      bytes[n++] = static_cast<std::uint8_t>(value >> ((size - 1u - i) * 8u));
      if (n == cap) {
        if (!sink(bytes, n))
          return false;
        n = 0;
      }
    }
    return true;
  }
  template <typename Sink>
  bool flush(Sink&& sink) {
    if (n == 0)
      return true;
    const unsigned had = n;
    n = 0;
    return sink(bytes, had);
  }
};

int dolgx_replay_pixels_main(const char* trace_path,
                             const PixelReplayOptions& options);

// Shared by the pixel/core replay mains (defined in dolgx_replay_pixels.cpp):
// deterministic stored-deflate PNG writer.
bool dolgx_replay_write_png(const std::string& path, const std::uint8_t* rgba,
                            std::uint32_t width, std::uint32_t height);
