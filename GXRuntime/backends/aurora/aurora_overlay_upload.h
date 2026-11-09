// SPDX-License-Identifier: GPL-3.0-or-later
// aurora_overlay_upload.h: the overlay's lists as they go up to the GPU. WebGPU's Queue.WriteBuffer takes a size that is
// a multiple of 4 bytes and stops the device at any other (a validation error, which is fatal to Aurora: the picture
// stops for the run). A vertex is 20 bytes, always whole; a 16-bit index list of odd length is not (ImGui's
// anti-aliased fill of a polygon with an odd number of corners gives one), so it gets one more index. The extra index
// is never drawn: every draw names its own first index and count.
#pragma once

#include "gxruntime/aurora_present.h"

#include <cstdint>
#include <vector>

namespace aurora_overlay_upload {
static_assert(sizeof(AuroraOverlayVertex) % 4 == 0, "a vertex list's bytes are a multiple of 4");

inline void pad_indices(std::vector<uint16_t>& indices) {
  if (indices.size() % 2 != 0)
    indices.push_back(0);
}

inline uint64_t index_bytes(const std::vector<uint16_t>& indices) { return indices.size() * sizeof(uint16_t); }
}  // namespace aurora_overlay_upload
