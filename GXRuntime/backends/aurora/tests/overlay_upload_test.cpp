// SPDX-License-Identifier: GPL-3.0-or-later
// The overlay's index list goes up in a multiple of 4 bytes whatever its length (aurora_overlay_upload.h), and the
// indices drawn stay what the host gave.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "aurora_overlay_upload.h"

#include <cassert>
#include <cstdio>

int main() {
  for (uint16_t n = 0; n < 200; ++n) {
    std::vector<uint16_t> indices;
    for (uint16_t i = 0; i < n; ++i)
      indices.push_back(static_cast<uint16_t>(i * 3 + 1));
    aurora_overlay_upload::pad_indices(indices);
    assert(aurora_overlay_upload::index_bytes(indices) % 4 == 0);
    assert(indices.size() == n || indices.size() == n + 1u);
    for (uint16_t i = 0; i < n; ++i)
      assert(indices[i] == static_cast<uint16_t>(i * 3 + 1));
  }
  // ImGui's anti-aliased fill of a convex polygon of N corners: (N - 2) * 3 + N * 6 indices; 7 corners give 57
  std::vector<uint16_t> seven(57, 1);
  aurora_overlay_upload::pad_indices(seven);
  assert(seven.size() == 58 && aurora_overlay_upload::index_bytes(seven) == 116);
  std::puts("overlay_upload_test: ok");
  return 0;
}
