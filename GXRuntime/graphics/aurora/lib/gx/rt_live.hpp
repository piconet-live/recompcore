#pragma once
// The live pass's face (AURORA_RT=1). See rt_live.cpp. Recording thread: submit_job at the frame's HUD; render worker: the two hooks
// around the frame's submit.
#include "rt_rules.hpp"

#include <atomic>
#include <future>
#include <memory>
#include <vector>

namespace aurora::gx::rt::live {
struct LiveTexture {
  std::uint32_t index, w, h;
  std::vector<std::vector<std::uint8_t>> mips;  // made beside the render worker; the first is the texture
};
struct Job {
  // filled by the translation workers, each draw in the slot its packet names; these are the only frame buffers, recycled
  std::unique_ptr<DrawRec[]> draws;
  std::unique_ptr<Vertex[]> verts;
  std::uint32_t baseDraw = 0, baseVert = 0, drawCount = 0;
  std::atomic<std::uint32_t> pending{0}, epoch{0};
  std::atomic<std::uint64_t> translateUs{0};
  std::promise<void> done;
  std::future<void> rules;
  Frame frame;
  std::vector<LiveTexture> textures;
  float proj[16], viewport[4];
  float road, light;
  double captureMs, rulesMs;
  std::uint64_t copy = 0;
  std::uint32_t mergedLights = 0, lightPicks = 0;
  bool drawable;
};
std::unique_ptr<Job> take_job();
bool submit_job(std::unique_ptr<Job> job);
void before_frame_submit();
void after_frame_submit();
bool failed() noexcept;
// The name Aurora's fault handler knows its own threads by
void name_thread() noexcept;
} // namespace aurora::gx::rt::live
