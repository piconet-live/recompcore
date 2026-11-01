// SPDX-License-Identifier: GPL-3.0-or-later
// model_pipelines.cpp: the thread that reads model files and compiles their pipelines ahead.
#pragma once
void model_pipelines_aurora_up();  // Aurora's device and pipeline cache are there (Aurora's thread)
void model_pipelines_stop();
