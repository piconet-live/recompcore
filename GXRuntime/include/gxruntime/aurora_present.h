// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_present: Aurora as a presenter beside an emulator that keeps running the game's GPU (Dolphin's video
// thread). A shared library with a C interface, loaded by the emulator at run time, so that the emulator's own
// build never links Dawn or Aurora.
//
// The emulator's GPU thread hands over the retail command stream as it reads it (aurora_present_write, every burst
// in order, display lists by their guest addresses) and marks each copy to the XFB (aurora_present_frame). The
// library copies the bytes and returns at once: its own thread decodes them with GXRuntime's live Aurora path,
// resolving display lists, arrays and textures in the guest's RAM (read, never written), presents each frame and
// reads its display copy back. The emulator takes the newest finished picture when it presents
// (aurora_present_take). Nothing here waits on the emulator, and the emulator never waits on Aurora: a frame Aurora
// has not finished is simply not shown yet, and if Aurora falls too far behind it stops (aurora_present_stats says
// so) and the emulator shows its own picture again.
//
// The seam for a later Aurora with external present targets: the picture's way back (take/the readback) is the only
// part that changes; write, frame and the thread stay.
#ifndef GXRUNTIME_AURORA_PRESENT_H
#define GXRUNTIME_AURORA_PRESENT_H

#include <stdint.h>

#ifdef _WIN32
#define AURORA_PRESENT_API __declspec(dllexport)
#else
#define AURORA_PRESENT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define AURORA_PRESENT_ABI 4

typedef struct AuroraPresentConfig {
  uint32_t abi;            // AURORA_PRESENT_ABI
  const uint8_t* ram;      // the guest's main RAM (big-endian, as the console has it), read by Aurora's thread
  uint32_t ram_size;       // its bytes
  const char* data_dir;    // Aurora's own folder (pipeline and Dawn caches), made by the caller first; UTF-8
  const char* shot_prefix; // test pictures: <prefix>-aurora-<copy>.png for each copy in `shots` (NULL: none)
  const uint64_t* shots;   // XFB copy numbers (1 the first)
  uint32_t shot_count;
  const char* log_path;    // a CSV line a frame: copy, times (NULL: none)
  uint32_t width, height;  // the hidden window's size (Aurora's EFB is the game's own 640x528 whatever it is)
  uint32_t most_behind;    // frames Aurora may be behind before it stops for the run (0: 4; and 256 MB at most)
  uint32_t scale;          // the render scale: the game's 640x528 EFB drawn at this many times its size (0: 1)
  void* window;            // the host's window, when the emulator draws nothing: Aurora presents straight into it and
                           // no picture comes back (Windows: its HWND; NULL: the hidden window and the readback)
} AuroraPresentConfig;

typedef struct AuroraPresentPicture {
  const uint8_t* rgba;  // tightly packed RGBA8 rows, top-left first; valid until the next take
  uint32_t width, height;  // the display copy's size
  uint64_t copy;           // the XFB copy it shows
} AuroraPresentPicture;

typedef struct AuroraPresentStats {
  uint64_t frames_in;       // frames handed over
  uint64_t frames_drawn;    // frames Aurora decoded and presented
  uint64_t pictures;        // pictures read back and offered
  uint64_t pictures_taken;  // ... taken by the emulator
  uint64_t bytes_in;        // command bytes handed over
  uint64_t most_behind;     // the most frames ever waiting
  uint32_t stopped;         // 1: Aurora fell behind or failed and stopped for the run
  uint32_t failed;          // 1: Aurora could not start
  double feed_ms_median, present_ms_median, readback_ms_median;  // Aurora's thread, per frame
  uint64_t overflow_drops;  // draws left out because a frame's buffer was full
  uint32_t fatal;           // 1: a fatal error of Aurora's stopped it (the thread is parked; the game goes on)
  uint32_t scale;           // the render scale in use
} AuroraPresentStats;

// The host's overlay, with the emulator drawing nothing: its own drawing (Dear ImGui's lists, flattened; no ImGui type
// crosses, the two sides' ImGui versions differ) over Aurora's picture, at the present, never into the picture.
// Any one thread, never blocking on Aurora.
typedef struct AuroraOverlayVertex {
  float x, y, u, v;  // the window's pixels; the texture's 0 to 1
  uint32_t rgba;     // bytes r, g, b, a
} AuroraOverlayVertex;

// A texture of the frame: an id given with aurora_present_overlay_texture, or a copy of the game's by address
#define AURORA_OVERLAY_COPY_TEXTURE(physical) (0xA500000000000000ull | (uint64_t)(physical))

typedef struct AuroraOverlayDraw {
  uint64_t texture;
  float clip[4];  // left, top, right, bottom, the window's pixels
  uint32_t first_index, index_count;
  int32_t base_vertex;
  uint32_t linear;  // 1: smooth sampling; 0: nearest
} AuroraOverlayDraw;

typedef struct AuroraOverlayFrame {
  uint32_t width, height;  // the window's pixels the frame was laid in
  const AuroraOverlayVertex* vertices;
  uint32_t vertex_count;
  const uint16_t* indices;
  uint32_t index_count;
  const AuroraOverlayDraw* draws;
  uint32_t draw_count;
} AuroraOverlayFrame;

// A texture made (its whole size given) or part of it written, RGBA8, straight alpha. Copied.
AURORA_PRESENT_API void aurora_present_overlay_texture(uint64_t id, uint32_t width, uint32_t height, uint32_t x, uint32_t y,
                                                       uint32_t part_width, uint32_t part_height, const uint8_t* rgba,
                                                       uint32_t row_bytes);
AURORA_PRESENT_API void aurora_present_overlay_forget(uint64_t id);
// The newest frame replaces the one before. Copied. Each present draws the newest it has.
AURORA_PRESENT_API void aurora_present_overlay_frame(const AuroraOverlayFrame* frame);
// Where the last present laid the game's picture in the window, in pixels: left, top, width, height. 0: no present yet.
AURORA_PRESENT_API int aurora_present_picture_rect(int32_t out[4]);

// Starts Aurora's thread. Returns 1, or 0 when the ABI differs (nothing started). Aurora's own start (the window,
// Dawn) happens on its thread; a failure there shows in the stats.
AURORA_PRESENT_API int aurora_present_start(const AuroraPresentConfig* config);
// The GPU thread: `size` bytes of the command stream, in order. Copied; never blocks.
// A model or course file the game is reading from the disc (its path, its whole bytes, copied here). Its materials' pipelines
// compile on Aurora's compile threads. Any thread, at any time, before or after the start.
AURORA_PRESENT_API void aurora_present_model_file(const char* path, const uint8_t* bytes, uint64_t size);
AURORA_PRESENT_API void aurora_present_write(const uint8_t* bytes, uint32_t size);
// The GPU thread: the stream so far ends frame `copy` (its XFB copy, numbered from 1). Never blocks.
AURORA_PRESENT_API void aurora_present_frame(uint64_t copy);
// Any one thread: the newest picture finished since the last take (1), none (0), or -1 once Aurora has stopped for
// the run (show the emulator's own pictures from then on). Cheap: no wait, one lock.
AURORA_PRESENT_API int aurora_present_take(AuroraPresentPicture* out);
AURORA_PRESENT_API void aurora_present_stats(AuroraPresentStats* out);
// Ends Aurora's thread (it finishes the frame it is on, drops the rest) and shuts Aurora down.
AURORA_PRESENT_API void aurora_present_stop(void);

#ifdef __cplusplus
}
#endif

#endif
