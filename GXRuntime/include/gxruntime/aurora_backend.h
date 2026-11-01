// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_AURORA_BACKEND_H
#define GXRUNTIME_AURORA_BACKEND_H

#include "gxruntime/platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraBackendConfig {
    const char* app_name;
    unsigned window_width;
    unsigned window_height;
    bool vsync;
    bool allow_texture_dumps;
    bool info_logging;
    bool graphics_logging;
    bool force_untextured;
    // Aurora's user and cache folder (pipeline and Dawn caches), made by the caller; NULL: DOL_AURORA_DATA_DIR, or
    // SDL's per-user folder
    const char* data_dir;
    // inside another program (aurora_present): a hidden window, no gamepads, no audio device
    bool embedded;
    // 0: Aurora's defaults (no MSAA; 16)
    unsigned msaa;
    unsigned max_anisotropy;
} AuroraBackendConfig;

bool dol_aurora_initialize(int argc, char** argv,
                           const AuroraBackendConfig* config);
void dol_aurora_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
