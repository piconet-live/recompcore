#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"

#include <optional>

namespace aurora::gfx::detail {

inline constexpr size_t FrameSlotCount = 2;
inline constexpr size_t StagingBufferCount = FrameSlotCount + 3;
inline constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize +
                                              StorageBufferSize + (UseTextureBuffer ? TextureUploadSize : 0);

const wgpu::Buffer& staging_buffer(size_t slot);

struct RegisteredDrawType {
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
};

struct RegisteredEncoderTaskType {
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
};

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id);
std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id);

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void initialize();
void shutdown();
bool begin_frame();
void end_frame(EndFrameCallback callback);
// AURORA_REDRAW: the render worker, one more picture of the frame last ended: its uniforms given to `patch`, its passes encoded again,
// presented as it was. False when no frame is held.
// `blocks` (AURORA_REDRAW_BUNDLES; sorted, each the start of 64 bytes): only those are patched and written; the rest of the uniform
// buffer keeps the held frame's as its own true frame left it. nullptr: the whole of the frame's uniforms written.
bool redraw_held(const std::function<void(uint8_t* uniforms, size_t bytes)>& patch, const std::vector<uint32_t>* blocks = nullptr);
bool redraw_pass_offscreen(uint32_t pass) noexcept;
// AURORA_REDRAW: bytes into the frame's vertex buffer at `at` (both whole words), before a picture between is encoded
void redraw_write_vertices(uint32_t at, const uint8_t* data, size_t bytes);
uint32_t current_frame() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx
