// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// A test build's look at where a garbled command stream comes from. Nothing here runs unless an environment switch
// is set, and nothing it does writes anything the game can see:
//  AURORA_TEST_FIFO_LOG=1            at an unknown opcode: where the byte is, the pointers, the bytes, and what came
//                                  before it (the list hand-overs, display lists, ARAM copies, commands)
//  AURORA_TEST_GPU_STALL=W:U:N:S:C   the GPU thread stops for U microseconds (it spins: no sleep's granularity), at
//                                  every Nth place W, from S seconds after its first burst, C times at most (0: no
//                                  limit). W: head (the first pass after the game turned reads on), pass (inside a
//                                  pass, before it copies its burst), between (after a pass's stores), dl (a display
//                                  list called from the stream, before it is read). Implies the log.
//  AURORA_TEST_WATCH=A:S[,A:S]...  physical address:size in hex: a thread looks at each range at about 20 us
//                                  intervals and says when it changes; the GPU's calls of a list starting in a range,
//                                  the game's list hand-overs, DMA writes over a range go in the same time line
//  AURORA_TEST_TIMELINE=PATH       the time line's file
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "Common/CommonTypes.h"

namespace FzgxFifoTest
{
enum class Where
{
  None,
  Head,
  Pass,
  Between,
  DisplayList
};

struct Config
{
  bool on = false;
  Where where = Where::None;
  u64 micros = 0;
  u64 every = 1;
  u64 from_seconds = 0;
  u64 most = 0;
};

inline Config ReadConfig()
{
  Config config;
  if (std::getenv("AURORA_TEST_FIFO_LOG") || std::getenv("AURORA_TEST_WATCH"))
    config.on = true;
  if (const char* stall = std::getenv("AURORA_TEST_GPU_STALL"))
  {
    char where[16] = {};
    unsigned long long micros = 0, every = 1, from = 0, most = 0;
    if (std::sscanf(stall, "%15[a-z]:%llu:%llu:%llu:%llu", where, &micros, &every, &from, &most) >= 2)
    {
      config.on = true;
      config.micros = micros;
      config.every = every ? every : 1;
      config.from_seconds = from;
      config.most = most;
      config.where = !std::strcmp(where, "head")    ? Where::Head :
                     !std::strcmp(where, "pass")    ? Where::Pass :
                     !std::strcmp(where, "between") ? Where::Between :
                     !std::strcmp(where, "dl")      ? Where::DisplayList :
                                                      Where::None;
    }
  }
  return config;
}

inline const Config& GetConfig()
{
  static const Config config = ReadConfig();
  return config;
}

inline bool On()
{
  return GetConfig().on;
}

inline u64 NowMicros()
{
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count());
}

// A ring of the last N of something; the threads' races only blur the record of a test.
template <size_t N>
struct Ring
{
  struct Entry
  {
    u64 us;
    std::array<u32, 8> v;
  };
  std::array<Entry, N> entries{};
  std::atomic<u64> next{0};

  void Note(std::array<u32, 8> v)
  {
    const u64 i = next.fetch_add(1, std::memory_order_relaxed);
    entries[i % N] = {NowMicros(), v};
  }
  template <typename F>
  void Each(F f) const
  {
    const u64 end = next.load(std::memory_order_relaxed);
    for (u64 i = end > N ? end - N : 0; i < end; ++i)
      f(entries[i % N]);
  }
};

inline Ring<12> g_handovers;   // reads on/off: on, read, write, distance, base, end, cpu write, cpu base
inline Ring<24> g_lists;       // called display lists: address, size
inline Ring<32> g_aram;        // ARAM copies: main RAM address, ARAM address, bytes, to main RAM
inline Ring<32> g_commands;    // the commands decoded: opcode, size, in a list

// the burst the GPU thread is reading: its guest address and where it is in the video buffer
inline std::atomic<u32> g_burst_guest{0};
inline std::atomic<const u8*> g_burst_host{nullptr};
inline std::atomic<bool> g_new_list{false};
inline std::atomic<u64> g_list_on_us{0};
inline std::atomic<u64> g_stalls{0};
inline std::atomic<u64> g_stalled_us{0};

inline void Stall(Where where)
{
  const Config& config = GetConfig();
  if (config.where != where)
    return;
  static u64 first = 0;
  static u64 seen = 0;
  const u64 now = NowMicros();
  if (!first)
    first = now;
  if (now - first < config.from_seconds * 1000000)
    return;
  if (++seen % config.every)
    return;
  const u64 done = g_stalls.load(std::memory_order_relaxed);
  if (config.most && done >= config.most)
    return;
  g_stalls.store(done + 1, std::memory_order_relaxed);
  g_stalled_us.fetch_add(config.micros, std::memory_order_relaxed);
  while (NowMicros() - now < config.micros)
  {
  }
}

struct WatchRange
{
  u32 address;
  u32 size;
  std::vector<u8> last;
};
inline std::vector<WatchRange> g_watches;
inline std::FILE* g_timeline = nullptr;
inline std::mutex g_timeline_lock;

inline bool Watching()
{
  return g_timeline != nullptr;
}

template <typename... Args>
void Line(const char* format, Args... args)
{
  if (!g_timeline)
    return;
  std::lock_guard lock(g_timeline_lock);
  std::fprintf(g_timeline, "%llu ", static_cast<unsigned long long>(NowMicros()));
  std::fprintf(g_timeline, format, args...);
  std::fputc('\n', g_timeline);
}

inline bool InWatch(u32 address, u32 size)
{
  address &= 0x3fffffff;
  for (const auto& w : g_watches)
  {
    if (address < w.address + w.size && w.address < address + size)
      return true;
  }
  return false;
}

// The watcher: BASE is guest RAM's start; PC, LR are read as they are (approximate). Runs until the process ends.
template <typename Registers>
void WatchLoop(const u8* base, Registers registers)
{
  for (auto& w : g_watches)
    w.last.assign(base + w.address, base + w.address + w.size);
  for (;;)
  {
    for (auto& w : g_watches)
    {
      if (std::memcmp(w.last.data(), base + w.address, w.size))
      {
        u32 first = 0;
        u32 changed = 0;
        bool any = false;
        for (u32 i = 0; i < w.size; ++i)
        {
          if (w.last[i] != base[w.address + i])
          {
            if (!any)
              first = i;
            any = true;
            ++changed;
          }
        }
        const auto [pc, lr] = registers();
        Line("changed %08x first +%#x, %u bytes; %02x%02x%02x%02x -> %02x%02x%02x%02x; PC %08x LR %08x", w.address, first,
             changed, w.last[first], w.last[first + 1 < w.size ? first + 1 : first], w.last[first + 2 < w.size ? first + 2 : first],
             w.last[first + 3 < w.size ? first + 3 : first], base[w.address + first],
             base[w.address + (first + 1 < w.size ? first + 1 : first)], base[w.address + (first + 2 < w.size ? first + 2 : first)],
             base[w.address + (first + 3 < w.size ? first + 3 : first)], pc, lr);
        std::memcpy(w.last.data(), base + w.address, w.size);
        std::fflush(g_timeline);
      }
    }
    const u64 start = NowMicros();
    while (NowMicros() - start < 20)
    {
    }
  }
}

inline std::string Hex(const u8* bytes, size_t size)
{
  std::string out;
  char buffer[4];
  for (size_t i = 0; i < size; ++i)
  {
    std::snprintf(buffer, sizeof(buffer), "%02x ", bytes[i]);
    out += buffer;
  }
  return out;
}
}  // namespace FzgxFifoTest
