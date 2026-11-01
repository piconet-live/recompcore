#pragma once
// The live pass's face (AURORA_RT=1). See rt_live.cpp. Recording thread: submit_job at the frame's HUD; render worker: the two hooks
// around the frame's submit.
#include "rt_rules.hpp"

#include <future>
#include <memory>
#include <vector>

namespace aurora::gx::rt::live {
struct LiveTexture {
  std::uint32_t index, w, h;
  std::vector<std::uint8_t> rgba;
};
struct Job {
  DrawList draws;
  std::vector<Vertex> verts;
  std::future<void> rules;
  Frame frame;
  std::vector<LiveTexture> textures;
  float proj[16], viewport[4];
  float road, light;
  double captureMs, rulesMs;
  bool drawable;
};
std::unique_ptr<Job> take_job();
bool submit_job(std::unique_ptr<Job> job);
void before_frame_submit();
void after_frame_submit();
bool failed() noexcept;
} // namespace aurora::gx::rt::live
