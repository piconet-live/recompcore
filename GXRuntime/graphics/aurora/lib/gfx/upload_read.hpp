#pragma once
// Reading back what was written into a mapped upload buffer (AURORA_REDRAW keeps a frame's uniforms and vertices): that memory
// is write-combined on D3D12, where ordinary loads go one uncached line at a time (c37, 30 racers: 7 ms a frame on the render
// worker for a few MB). Streaming loads read it whole lines at once. The same bytes either way.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace aurora::gfx {
inline void copy_from_upload(void* to, const void* from, std::size_t bytes) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  auto* d = static_cast<std::uint8_t*>(to);
  auto* s = static_cast<const std::uint8_t*>(from);
  const std::size_t head = std::min<std::size_t>(bytes, (16 - (reinterpret_cast<std::uintptr_t>(s) & 15)) & 15);
  std::memcpy(d, s, head);
  d += head, s += head, bytes -= head;
  _mm_mfence();
  for (; bytes >= 64; bytes -= 64, s += 64, d += 64) {
    auto* p = reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(s));
    const __m128i a = _mm_stream_load_si128(p), b = _mm_stream_load_si128(p + 1), c = _mm_stream_load_si128(p + 2),
                  e = _mm_stream_load_si128(p + 3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d), a);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d) + 1, b);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d) + 2, c);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d) + 3, e);
  }
  for (; bytes >= 16; bytes -= 16, s += 16, d += 16)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(s))));
  std::memcpy(d, s, bytes);
#else
  std::memcpy(to, from, bytes);
#endif
}
} // namespace aurora::gfx
