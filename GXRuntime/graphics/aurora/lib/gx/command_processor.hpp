#pragma once

#include "../internal.hpp"

#include <cstdint>

#include "../gfx/stagelog.hpp"

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// AURORA_STAGELOG: the stream thread's summed stretches (rows st_*), written and cleared; the whole of process() summed
void stream_sum_rows() noexcept;
gfx::stagelog::Sum& stream_process_sum() noexcept;

} // namespace aurora::gx::fifo
