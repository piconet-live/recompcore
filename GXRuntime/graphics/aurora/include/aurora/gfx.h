#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include "stdint.h"
#endif

#if !defined(NDEBUG) && !defined(AURORA_GFX_DEBUG_GROUPS)
#define AURORA_GFX_DEBUG_GROUPS
#endif

void push_debug_group(const char* label);
void pop_debug_group();

typedef struct {
  uint32_t queuedPipelines;
  uint32_t createdPipelines;
  uint32_t drawCallCount;
  uint32_t mergedDrawCallCount;
  uint32_t lastVertSize;
  uint32_t lastUniformSize;
  uint32_t lastIndexSize;
  uint32_t lastStorageSize;
  uint32_t lastTextureUploadSize;
} AuroraStats;

const AuroraStats* aurora_get_stats();
// Draws and pushes left out of frames because a fixed buffer was full, since the start (patch 19)
uint64_t aurora_overflow_drops();
float aurora_get_fps();

void aurora_enable_vsync(bool enabled);

/*
 * Present-source readback (GXRuntime patch): arm a one-shot copy of the next
 * submitted frame's present source; take returns tightly packed RGBA8 rows
 * (top-left origin) once the copy completed and consumes the result. The
 * returned pointer stays valid until the next readback completes.
 */
void aurora_request_framebuffer_readback();
bool aurora_take_framebuffer_readback(const uint8_t** rgba, uint32_t* width, uint32_t* height);
/* Drive GPU event processing while waiting for take (no frame, no present). */
void aurora_pump_framebuffer_readback();
/* Frame timing for offline tools (steady clock, ns): when the render worker
 * submitted the last frame, and when the GPU reported that work done. */
uint64_t aurora_timing_last_submit_ns();
uint64_t aurora_timing_last_gpu_done_ns();
/* AURORA_PRESENT_OVERLAP: a true frame's submit to its GPU done, smoothed (ns; 0 before the first). */
uint64_t aurora_timing_true_gpu_lag_ns();

#ifdef __cplusplus
}
#endif

#endif
