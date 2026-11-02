#pragma once
// AURORA_STAGELOG=<file> (off by default, a measuring tool): where a picture's time goes. Named stretches of the render
// threads with their time, and a sampler that reads every thread's scheduler state (running, ready and not given a CPU,
// waiting and why) a few hundred times a second. Reads nothing the race reads and writes nothing it reads.
#include <chrono>
#include <cstdint>

namespace aurora::gfx::stagelog {
bool on() noexcept;
void note(const char* what, std::chrono::steady_clock::time_point start, std::uint64_t copy = 0) noexcept;
struct Scope {
  const char* what;
  std::uint64_t copy;
  bool live;
  std::chrono::steady_clock::time_point start;
  explicit Scope(const char* w, std::uint64_t c = 0) noexcept : what(w), copy(c), live(on()) {
    if (live)
      start = std::chrono::steady_clock::now();
  }
  ~Scope() {
    if (live)
      note(what, start, copy);
  }
};
} // namespace aurora::gfx::stagelog
