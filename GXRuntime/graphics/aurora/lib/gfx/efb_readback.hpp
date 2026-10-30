#pragma once

#include <cstdint>

// One-shot CPU readback of the frame's present source (the EFB, or the XFB copy), for offline pixel comparison
// (GXRuntime's dolgx_replay --pixels). Upstream's aurora::gfx::readback serves EFB peeks and GX copies, not the
// presented picture.
namespace aurora::gfx::efb_readback {

// Arm a readback of the next submitted frame. Any thread.
void request() noexcept;

// Copy the frame's present source into a readback buffer and start the map.
// Render worker only, after the frame's queue submit.
void after_submit() noexcept;

// True once a readback completed; returns tightly packed RGBA8 rows
// (top-left origin). Consumes the result; the pointer stays valid until the
// next readback completes. Any thread.
bool take(const std::uint8_t** rgba, std::uint32_t* width,
          std::uint32_t* height) noexcept;

} // namespace aurora::gfx::efb_readback
