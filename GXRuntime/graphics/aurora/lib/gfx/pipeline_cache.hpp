#pragma once

#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

struct CompiledPipeline {
  wgpu::RenderPipeline main;
  wgpu::RenderPipeline prepass;

  [[nodiscard]] uint32_t pipeline_count() const {
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout, bool blocking = false);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout, bool blocking = false);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);
// A GX pipeline a draw will want, compiled ahead on the compile threads. Any thread.
void precompile_pipeline(const gx::PipelineConfig& config, bool scene_target = true, bool other_target = true);
size_t pipelines_waiting();
void set_background_compile_threads(unsigned count);
size_t pipelines_compiled();
// Where the game stands, from the host (aurora_present_scene): `racing` from the race's own phase to its end. While racing nothing
// is compiled ahead (the background queue stands; a draw's own pipeline still compiles). `where` names the moment in the miss
// log. Any thread.
void set_scene(bool racing, const char* where);
bool racing();
// The pipelines still queued ahead of any draw (the background queue)
size_t pipelines_ahead_waiting();
// The pipeline build (aurora_present_build_start), before initialize_pipeline_cache: a pipeline is let go as soon as it is
// compiled (only Dawn's cache is wanted: a whole disc's pipelines would not fit in memory, about 300,000), its key kept so it
// is not compiled twice; no line a compile; and Dawn's cache is never pruned.
void set_build_mode(bool on);
// The mark the build leaves in Aurora's folder when it is done: a start that finds it never prunes Dawn's cache (the default
// prune keeps only the entries one session used, which would undo the build)
inline constexpr const char* PipelinesBuiltMark = "pipelines_built";

bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);

} // namespace aurora::gfx
