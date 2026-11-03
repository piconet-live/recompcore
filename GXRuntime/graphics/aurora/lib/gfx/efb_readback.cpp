#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "efb_readback.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::gfx::efb_readback {
namespace {
Module Log("aurora::gfx::efb_readback");

std::mutex g_mutex;
std::atomic_bool g_requested{false};
bool g_inFlight = false;
bool g_ready = false;
// In-flight copy geometry (worker-written, callback-read under g_mutex).
wgpu::Buffer g_buffer;
// The buffer of the last completed map. complete_map runs inside Dawn's map callback, where dropping the last
// reference destroys the buffer while Dawn still holds a mutex inside it (it unlocks freed memory afterwards: the
// heap use-after-free behind the occasional access violation, exit code 5). It is released on the next
// after_submit instead, outside any callback.
wgpu::Buffer g_retiredBuffer;
uint32_t g_copyWidth = 0;
uint32_t g_copyHeight = 0;
uint32_t g_copyBytesPerRow = 0;
wgpu::TextureFormat g_copyFormat = wgpu::TextureFormat::Undefined;
// Completed result: tightly packed RGBA8.
std::vector<uint8_t> g_pixels;
uint32_t g_readyWidth = 0;
uint32_t g_readyHeight = 0;

void complete_map(wgpu::MapAsyncStatus status, wgpu::StringView message) {
  std::lock_guard lock{g_mutex};
  if (status == wgpu::MapAsyncStatus::Success) {
    const size_t byteSize =
        static_cast<size_t>(g_copyBytesPerRow) * g_copyHeight;
    const auto* mapped =
        static_cast<const uint8_t*>(g_buffer.GetConstMappedRange(0, byteSize));
    if (mapped != nullptr) {
      const bool bgra = g_copyFormat == wgpu::TextureFormat::BGRA8Unorm ||
                        g_copyFormat == wgpu::TextureFormat::BGRA8UnormSrgb;
      g_pixels.resize(static_cast<size_t>(g_copyWidth) * g_copyHeight * 4u);
      if (g_copyFormat == wgpu::TextureFormat::RGBA16Float) {
        // the raw picture: the range is reported (AURORA_LOOK_PEAKLOG=<file>), the picture shown as the display would, cut at white
        const auto half = [](uint16_t h) {
          const uint32_t e = (h >> 10) & 31, m = h & 1023;
          const float v = e == 0 ? static_cast<float>(m) / (1024.f * 16384.f) : e == 31 ? 65535.f : (1.f + static_cast<float>(m) / 1024.f) * static_cast<float>(1u << e) / 32768.f;
          return (h & 0x8000) != 0 ? -v : v;
        };
        float peak = 0.f, least = 0.f;
        uint64_t over1 = 0, over2 = 0, over4 = 0;
        for (uint32_t y = 0; y < g_copyHeight; ++y) {
          const auto* srcRow = reinterpret_cast<const uint16_t*>(mapped + static_cast<size_t>(y) * g_copyBytesPerRow);
          uint8_t* dstRow = g_pixels.data() + static_cast<size_t>(y) * g_copyWidth * 4u;
          for (uint32_t x = 0; x < g_copyWidth; ++x) {
            float most = 0.f;
            for (uint32_t c = 0; c < 4u; ++c) {
              const float v = half(srcRow[x * 4u + c]);
              if (c < 3u)
                most = std::max(most, v), least = std::min(least, v);
              dstRow[x * 4u + c] = static_cast<uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
            }
            peak = std::max(peak, most);
            over1 += most > 1.01f, over2 += most > 2.f, over4 += most > 4.f;
          }
        }
        if (const char* path = std::getenv("AURORA_LOOK_PEAKLOG"))
          if (FILE* f = std::fopen(path, "ab")) {
            const double all = static_cast<double>(g_copyWidth) * g_copyHeight / 100.0;
            std::fprintf(f, "peak %.2f least %.2f, share over 1: %.2f%%, over 2: %.2f%%, over 4: %.2f%%\n", peak, least, over1 / all, over2 / all, over4 / all);
            std::fclose(f);
          }
        g_readyWidth = g_copyWidth;
        g_readyHeight = g_copyHeight;
        g_ready = true;
        g_buffer.Unmap();
        g_retiredBuffer = std::move(g_buffer);
        g_inFlight = false;
        return;
      }
      for (uint32_t y = 0; y < g_copyHeight; ++y) {
        const uint8_t* srcRow = mapped + static_cast<size_t>(y) * g_copyBytesPerRow;
        uint8_t* dstRow = g_pixels.data() + static_cast<size_t>(y) * g_copyWidth * 4u;
        if (!bgra) {
          std::memcpy(dstRow, srcRow, static_cast<size_t>(g_copyWidth) * 4u);
        } else {
          for (uint32_t x = 0; x < g_copyWidth; ++x) {
            dstRow[x * 4u + 0u] = srcRow[x * 4u + 2u];
            dstRow[x * 4u + 1u] = srcRow[x * 4u + 1u];
            dstRow[x * 4u + 2u] = srcRow[x * 4u + 0u];
            dstRow[x * 4u + 3u] = srcRow[x * 4u + 3u];
          }
        }
      }
      g_readyWidth = g_copyWidth;
      g_readyHeight = g_copyHeight;
      g_ready = true;
    }
    g_buffer.Unmap();
  } else if (status != wgpu::MapAsyncStatus::CallbackCancelled &&
             status != wgpu::MapAsyncStatus::Aborted) {
    Log.warn("EFB readback mapping failed: {}", message);
  }
  g_retiredBuffer = std::move(g_buffer);
  g_inFlight = false;
}
} // namespace

void request() noexcept { g_requested.store(true, std::memory_order_release); }

void after_submit() noexcept {
  if (!g_requested.load(std::memory_order_acquire)) {
    return;
  }
  // MapAsync is issued after g_mutex is released: with AllowSpontaneous the
  // callback may run inside MapAsync (D3D12 does this when the copy is already
  // done), and complete_map takes g_mutex (std::mutex throws on the re-lock).
  wgpu::Buffer mapBuffer;
  uint64_t mapSize = 0;
  wgpu::Buffer retired;
  {
    std::lock_guard lock{g_mutex};
    // Dropped after g_mutex is released, outside any Dawn callback
    retired = std::move(g_retiredBuffer);
    if (g_inFlight) {
      // Previous copy still mapping; keep the request armed for next frame.
      return;
    }
    const auto& src = webgpu::present_source();
    if (!src.texture) {
      return;
    }
    const wgpu::TextureFormat format = src.format;
    if (format != wgpu::TextureFormat::RGBA8Unorm &&
        format != wgpu::TextureFormat::RGBA8UnormSrgb &&
        format != wgpu::TextureFormat::BGRA8Unorm &&
        format != wgpu::TextureFormat::BGRA8UnormSrgb &&
        format != wgpu::TextureFormat::RGBA16Float) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        Log.warn("EFB readback: unsupported present source format");
      }
      g_requested.store(false, std::memory_order_release);
      return;
    }
    const uint32_t width = src.size.width;
    const uint32_t height = src.size.height;
    // WebGPU requires 256-byte bytesPerRow alignment for texture->buffer.
    const uint32_t bytesPerRow = (width * (format == wgpu::TextureFormat::RGBA16Float ? 8u : 4u) + 255u) & ~255u;
    const uint64_t byteSize = static_cast<uint64_t>(bytesPerRow) * height;
    const wgpu::BufferDescriptor bufferDescriptor{
        .label = "EFB Readback Buffer",
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
        .size = byteSize,
    };
    g_buffer = webgpu::g_device.CreateBuffer(&bufferDescriptor);
    if (!g_buffer) {
      return;
    }
    g_copyWidth = width;
    g_copyHeight = height;
    g_copyBytesPerRow = bytesPerRow;
    g_copyFormat = format;
    g_inFlight = true;

    static constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{
        .label = "EFB readback encoder",
    };
    auto encoder = webgpu::g_device.CreateCommandEncoder(&EncoderDescriptor);
    const wgpu::TexelCopyTextureInfo srcInfo{
        .texture = src.texture,
    };
    const wgpu::TexelCopyBufferInfo dstInfo{
        .layout =
            {
                .offset = 0,
                .bytesPerRow = bytesPerRow,
                .rowsPerImage = height,
            },
        .buffer = g_buffer,
    };
    const wgpu::Extent3D extent{
        .width = width,
        .height = height,
        .depthOrArrayLayers = 1,
    };
    encoder.CopyTextureToBuffer(&srcInfo, &dstInfo, &extent);
    static constexpr wgpu::CommandBufferDescriptor CommandBufferDescriptor{
        .label = "EFB readback command buffer",
    };
    const auto buffer = encoder.Finish(&CommandBufferDescriptor);
    webgpu::g_queue.Submit(1, &buffer);
    mapBuffer = g_buffer;
    mapSize = byteSize;
  }
  g_requested.store(false, std::memory_order_release);
  if (mapBuffer) {
    mapBuffer.MapAsync(wgpu::MapMode::Read, 0, mapSize, wgpu::CallbackMode::AllowSpontaneous,
                       [](wgpu::MapAsyncStatus status, wgpu::StringView message) { complete_map(status, message); });
  }
}

bool take(const std::uint8_t** rgba, std::uint32_t* width,
          std::uint32_t* height) noexcept {
  std::lock_guard lock{g_mutex};
  if (!g_ready) {
    return false;
  }
  if (rgba != nullptr) {
    *rgba = g_pixels.data();
  }
  if (width != nullptr) {
    *width = g_readyWidth;
  }
  if (height != nullptr) {
    *height = g_readyHeight;
  }
  g_ready = false;
  return true;
}

} // namespace aurora::gfx::efb_readback
