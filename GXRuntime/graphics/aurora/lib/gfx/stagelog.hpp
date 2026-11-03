#pragma once
#include <cstdlib>
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
// Many short stretches of one thread summed (each far under the 100 us a row of its own needs): one row a frame,
// S,<start_us>,<summed_us>,<what>,<count>, written by sum_row (the summing thread's own, or after it is synchronised with)
struct Sum {
  std::uint64_t ns = 0;
  std::uint32_t n = 0;
};
void sum_row(const char* what, Sum& sum) noexcept;  // writes the row and clears the sum
// The per-draw sums cost the stream thread several ms a frame: only with AURORA_STAGELOG_STREAM=1 (as well as AURORA_STAGELOG)
inline bool stream_on() noexcept {
  static const bool v = std::getenv("AURORA_STAGELOG_STREAM") != nullptr;
  return v;
}
struct SumScope {
  Sum* sum;
  bool live;
  std::chrono::steady_clock::time_point start;
  explicit SumScope(Sum& s) noexcept : sum(&s), live(stream_on() && on()) {
    if (live)
      start = std::chrono::steady_clock::now();
  }
  ~SumScope() {
    if (live) {
      sum->ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
      ++sum->n;
    }
  }
};
} // namespace aurora::gfx::stagelog
