// RecompCore: the code tables' generation, which a second CPU context reads (StaticRecompCore::CodeGeneration).
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace StaticRecomp
{
// What another thread running the module (the fzgx runner's deep-rollback worker) may have used of one chunk, as
// bits: the run loop's dispatch (the chunk state is "verified"), a function body's entry test (the entry byte's bit
// 0) and the indirect-call table (its bit 1). The worker only runs a chunk's native code while one of them says yes:
// its run loop gives a job up at a chunk not verified, and a body whose entry test says no leaves to that loop.
//
// The generation moves only when a chunk loses a bit, or at a whole clear: then code the worker may have run, or may
// still run from a stale read, is no longer what the tables say, and its job gives up. A chunk that gains a bit (its
// first check in a race, or a check after an invalidation, which moved the generation when it took the bits away)
// does not move it: before, the worker could not run that code (it gave up at it); after, it runs the module's code,
// the same in both memories. So a chunk checked anywhere while a job runs no longer ends the job.
//
// Written only on the thread that owns the core (the game's); Value is read on any thread.
class CodeGeneration
{
public:
  static constexpr std::uint8_t DISPATCH = 1;  // the run loop's chunk state is "verified"
  static constexpr std::uint8_t ENTRY = 2;     // the module's entry byte, bit 0
  static constexpr std::uint8_t CALLABLE = 4;  // the module's entry byte, bit 1 (the indirect-call table)

  // A module of `chunks` chunks, none usable: moves.
  void Reset(std::size_t chunks)
  {
    m_bits.assign(chunks, 0);
    Move();
  }
  // The bits `mask` of one chunk become `bits`; moves when one that was set is cleared.
  void Set(std::size_t chunk, std::uint8_t mask, std::uint8_t bits)
  {
    if (chunk >= m_bits.size())
      return Move();
    const std::uint8_t was = m_bits[chunk];
    const std::uint8_t now = static_cast<std::uint8_t>((was & ~mask) | (bits & mask));
    m_bits[chunk] = now;
    if (was & ~now)
      Move();
  }
  // Every chunk's bits `mask` cleared at once (the module's entry bytes cleared whole): moves.
  void ClearAll(std::uint8_t mask)
  {
    for (std::uint8_t& b : m_bits)
      b = static_cast<std::uint8_t>(b & ~mask);
    Move();
  }
  std::uint64_t Value() const { return m_value.load(std::memory_order_acquire); }
  std::uint8_t Bits(std::size_t chunk) const { return chunk < m_bits.size() ? m_bits[chunk] : 0; }

private:
  void Move() { m_value.fetch_add(1, std::memory_order_release); }

  std::vector<std::uint8_t> m_bits;
  std::atomic<std::uint64_t> m_value{0};
};
}  // namespace StaticRecomp
