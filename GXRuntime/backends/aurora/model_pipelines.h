// SPDX-License-Identifier: GPL-3.0-or-later
// model_pipelines.cpp: the thread that reads model files and compiles their pipelines ahead.
#pragma once
void model_pipelines_aurora_up();  // Aurora's device and pipeline cache are there (Aurora's thread)
void model_pipelines_stop();
// The pipeline build (aurora_present_build_start), before any file: each file is taken as the disc walk's (one form of every
// shader), whether or not the walk is asked for, and compiles take every compile thread
void model_pipelines_build_mode();
struct ModelPipelinesBuild {
  unsigned long long bytes_read = 0;  // of the files handed over
  unsigned long long shaders = 0;     // planned
  bool walk_read = false;             // the walk's end read
  bool planned_all = false;           // and every form asked for (what is asked may still be compiling)
};
ModelPipelinesBuild model_pipelines_build_state();
