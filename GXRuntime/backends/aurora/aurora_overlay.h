// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_overlay: what the host draws over Aurora's picture (gxruntime/aurora_present.h, the overlay's functions) and the
// colour filter of the whole window. Both through Aurora's present hook, after everything else of the frame.
#pragma once

#include <stdint.h>

// Aurora's thread, Aurora up: installs the hook. `ram`: the guest's RAM, where a copy's address leads (Aurora knows a copy
// by the pointer its address resolved to).
void aurora_overlay_start(const uint8_t* ram);
