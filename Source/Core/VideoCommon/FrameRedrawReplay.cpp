// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw's replay: a record drawn again into the second EFB, and the compare (FrameRedraw.h).

#include "VideoCommon/FrameRedrawFrame.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <unordered_map>

#include <fmt/format.h>

#include "Common/Image.h"
#include "Core/System.h"
#include "VideoCommon/AbstractStagingTexture.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/TextureCacheBase.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VertexShaderManager.h"

// The replay's way into the vertex manager's and the texture cache's own upload and copy paths.
struct FrameRedrawAccess
{
  static void UploadUniforms(VertexManagerBase& vm) { vm.UploadUniforms(); }
  static RcTcacheEntry Allocate(TextureCacheBase& tc, const TextureConfig& config)
  {
    return tc.AllocateCacheEntry(config);
  }
  static void Copy(TextureCacheBase& tc, RcTcacheEntry& entry, bool depth,
                   const MathUtil::Rectangle<int>& src, bool half, bool linear, EFBCopyFormat format,
                   bool intensity, float gamma, bool top, bool bottom, const std::array<u32, 3>& coef)
  {
    tc.CopyEFBToCacheEntry(entry, depth, src, half, linear, format, intensity, gamma, top, bottom,
                           coef);
  }
};

namespace VideoCommon::FrameRedraw
{
bool g_replaying = false;
bool g_keep_xfb = false;

double Us(Clock::duration d)
{
  return std::chrono::duration<double, std::micro>(d).count();
}

namespace
{
std::unique_ptr<AbstractStagingTexture> s_staging;

// The EFB's color over rect (EFB texture coordinates), read back; one u32 a texel (RGBA8 formats).
bool ReadColor(const MathUtil::Rectangle<int>& rect, std::vector<u32>* out)
{
  AbstractTexture* texture = g_framebuffer_manager->ResolveEFBColorTexture(rect);
  if (!texture || AbstractTexture::GetTexelSizeForFormat(texture->GetFormat()) != 4)
    return false;
  const TextureConfig config(rect.GetWidth(), rect.GetHeight(), 1, 1, 1, texture->GetFormat(), 0,
                             AbstractTextureType::Texture_2DArray);
  if (!s_staging || !(s_staging->GetConfig() == config))
    s_staging = g_gfx->CreateStagingTexture(StagingTextureType::Readback, config);
  if (!s_staging)
    return false;
  const MathUtil::Rectangle<int> to(0, 0, rect.GetWidth(), rect.GetHeight());
  s_staging->CopyFromTexture(texture, rect, 0, 0, to);
  out->assign(static_cast<size_t>(rect.GetWidth()) * rect.GetHeight(), 0);
  s_staging->ReadTexels(to, out->data(), rect.GetWidth() * 4);
  return true;
}

void DumpPpm(const std::string& path, const std::vector<u32>& pixels, int width, int height)
{
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f)
    return;
  std::fprintf(f, "P6\n%d %d\n255\n", width, height);
  std::vector<u8> row(static_cast<size_t>(width) * 3);
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const u32 p = pixels[static_cast<size_t>(y) * width + x];
      row[x * 3] = p & 0xFF;
      row[x * 3 + 1] = (p >> 8) & 0xFF;
      row[x * 3 + 2] = (p >> 16) & 0xFF;
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
}

}  // namespace

namespace
{
// The stills (QueueStill): each texture copied into a readback texture of its own when asked (a GPU command, no
// wait), read a few frames later (TakeStills: its commands long done, so no wait then either) and written as PNG
// on a thread of its own, so a stills run's pictures keep their timing.
struct PendingStill
{
  std::unique_ptr<AbstractStagingTexture> staging;
  int width, height;
  bool bgra;
  std::string path;
  u64 asked;
};
std::deque<PendingStill> s_stills;
std::vector<std::future<void>> s_writes;

void Write(PendingStill& p)
{
  std::vector<u8> pixels(static_cast<size_t>(p.width) * p.height * 4);
  p.staging->ReadTexels(MathUtil::Rectangle<int>(0, 0, p.width, p.height), pixels.data(), p.width * 4);
  s_writes.push_back(std::async(std::launch::async, [pixels = std::move(pixels), w = p.width, h = p.height,
                                                     bgra = p.bgra, path = p.path]() mutable {
    for (size_t i = 0; i < pixels.size(); i += 4)
    {
      if (bgra)
        std::swap(pixels[i], pixels[i + 2]);
      pixels[i + 3] = 255;  // (the copy's alpha is the EFB's, not a picture's)
    }
    Common::SavePNG(path, pixels.data(), Common::ImageByteFormat::RGBA, w, h, w * 4, 1);
  }));
}
}  // namespace

bool QueueStill(const AbstractTexture* texture, const MathUtil::Rectangle<int>& rect, const std::string& path,
                u64 now)
{
  if (!texture || rect.GetWidth() <= 0 || rect.GetHeight() <= 0)
    return false;
  const AbstractTextureFormat format = texture->GetFormat();
  if (format != AbstractTextureFormat::RGBA8 && format != AbstractTextureFormat::BGRA8)
    return false;
  const int width = rect.GetWidth(), height = rect.GetHeight();
  const TextureConfig config(width, height, 1, 1, 1, format, 0, AbstractTextureType::Texture_2DArray);
  std::unique_ptr<AbstractStagingTexture> staging =
      g_gfx->CreateStagingTexture(StagingTextureType::Readback, config);
  if (!staging)
    return false;
  staging->CopyFromTexture(texture, rect, 0, 0, MathUtil::Rectangle<int>(0, 0, width, height));
  s_stills.push_back({std::move(staging), width, height, format == AbstractTextureFormat::BGRA8, path, now});
  return true;
}

void TakeStills(u64 now, bool all)
{
  while (!s_stills.empty() && (all || s_stills.front().asked + 4 <= now))
  {
    Write(s_stills.front());
    s_stills.pop_front();
  }
  if (all)
  {
    for (std::future<void>& w : s_writes)
      w.wait();
    s_writes.clear();
  }
  std::erase_if(s_writes, [](std::future<void>& w) {
    return w.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  });
}

namespace
{
// A draw's vertices and indices into the stream buffers, or where the request's first picture put them.
void Upload(const Frame& f, const Draw& d, u32 k, const Corrections* c, bool moved, u32* base_vertex,
            u32* base_index, Replayed* r)
{
  Corrections* reuse = c && c->reuse && !moved ? const_cast<Corrections*>(c) : nullptr;
  if (reuse && reuse->uploaded.size() == f.draws.size() && reuse->uploaded[k][0] != ~0u)
  {
    *base_vertex = reuse->uploaded[k][0];
    *base_index = reuse->uploaded[k][1];
    r->vertex_reused += static_cast<u64>(d.stride) * d.num_vertices + 2ull * d.num_indices;
    return;
  }
  g_vertex_manager->UploadUtilityVertices(moved ? &c->vertices[c->moved_at[k]] : &f.vertices[d.vertex_at], d.stride,
                                          d.num_vertices, &f.indices[d.index_at], d.num_indices, base_vertex,
                                          base_index);
  r->vertex_bytes += static_cast<u64>(d.stride) * d.num_vertices + 2ull * d.num_indices;
  if (!c || !c->reuse)
    return;
  Corrections* w = const_cast<Corrections*>(c);
  const u64 byte = static_cast<u64>(*base_vertex) * d.stride;
  if (byte < w->last_vertex_byte || *base_index < w->last_index)
    w->uploaded.assign(f.draws.size(), {~0u, ~0u});  // the stream went round: what was put before may be gone
  w->last_vertex_byte = byte;
  w->last_index = *base_index;
  if (w->uploaded.size() != f.draws.size())
    w->uploaded.assign(f.draws.size(), {~0u, ~0u});
  if (!moved)
    w->uploaded[k] = {*base_vertex, *base_index};
}
}  // namespace

Replayed Replay(const Frame& f, const MathUtil::Rectangle<int>* compare, u64 frame_number,
                const std::string& dump_prefix, u32* dumps, const Corrections* corrections,
                RcTcacheEntry* picture, bool (*stop)())
{
  Replayed r;
  auto& system = Core::System::GetInstance();
  auto& vsm = system.GetVertexShaderManager();
  auto& gsm = system.GetGeometryShaderManager();
  auto& psm = system.GetPixelShaderManager();
  if (compare)
    g_gfx->WaitForGPUIdle();
  const Clock::time_point start = Clock::now();
  const VertexShaderConstants saved_vs = vsm.constants;
  const GeometryShaderConstants saved_gs = gsm.constants;
  const PixelShaderConstants saved_ps = psm.constants;
  const PixelFormat saved_format = g_framebuffer_manager->GetPrevPixelFormat();
  if (!g_framebuffer_manager->SwapReplayEFB())
    return r;
  g_replaying = true;
  g_framebuffer_manager->StorePixelFormat(f.start_format);
  g_framebuffer_manager->BindEFBFramebuffer();

  std::unordered_map<const AbstractTexture*, const AbstractTexture*> redirect;
  std::vector<RcTcacheEntry> scratch;
  u32 vs = ~0u, gs = ~0u, ps = ~0u;
  int tag = -2;  // the correction the uploaded vertex constants have (-1: none)
  const AbstractGfx::ViewportAndScissor* view = nullptr;
  u32 since_asked = 0;
  for (const Op& op : f.ops)
  {
    if (stop && ++since_asked == 16)
    {
      since_asked = 0;
      if (stop())
      {
        r.stopped = true;
        break;
      }
    }
    switch (op.kind)
    {
    case Kind::Draw:
    {
      const Draw& d = f.draws[op.index];
      const int want = corrections && op.index < corrections->tags.size() ? corrections->tags[op.index] : -1;
      if (d.vs != vs || want != tag)
      {
        vsm.constants = f.vs[vs = d.vs];
        vsm.dirty = true;
        tag = want;
        if (tag >= 0 && static_cast<size_t>(tag) < corrections->matrices.size())
        {
          std::array<float, 16> p;
          std::memcpy(p.data(), vsm.constants.projection.data(), sizeof(p));
          Common::Matrix44 m = Common::Matrix44::FromArray(p);
          if (static_cast<size_t>(tag) >= corrections->screen_from &&
              static_cast<size_t>(tag) < corrections->icon_from)
            m = corrections->matrices[tag] * m;
          else
            m *= corrections->matrices[tag];
          std::memcpy(vsm.constants.projection.data(), m.data.data(), sizeof(p));
        }
      }
      if (d.gs != gs)
      {
        gsm.constants = f.gs[gs = d.gs];
        gsm.dirty = true;
      }
      if (d.ps != ps)
      {
        psm.constants = f.ps[ps = d.ps];
        psm.dirty = true;
      }
      Clock::time_point at = Clock::now();
      {
        const u64 bytes = (vsm.dirty ? sizeof(VertexShaderConstants) : 0) +
                          (gsm.dirty ? sizeof(GeometryShaderConstants) : 0) +
                          (psm.dirty ? sizeof(PixelShaderConstants) : 0);
        r.uniform_uploads += bytes != 0;
        r.uniform_bytes += bytes;
      }
      FrameRedrawAccess::UploadUniforms(*g_vertex_manager);
      Clock::time_point then = Clock::now();
      r.uniforms_us += Us(then - at);
      at = then;
      g_gfx->SetPipeline(d.pipeline);
      if (!view || std::memcmp(view, &d.view, sizeof(d.view)) != 0)
      {
        view = &d.view;
        g_gfx->SetViewport(d.view.viewport_x, d.view.viewport_y, d.view.viewport_width,
                           d.view.viewport_height, d.view.viewport_near_depth,
                           d.view.viewport_far_depth);
        g_gfx->SetScissorRect(d.view.scissor_rect);
      }
      for (const u32 i : d.textures)
      {
        const auto it = redirect.find(d.texture[i]);
        g_gfx->SetTexture(i, it != redirect.end() ? it->second : d.texture[i]);
        g_gfx->SetSamplerState(i, d.sampler[i]);
      }
      u32 base_vertex, base_index;
      const bool moved = corrections && op.index < corrections->moved_at.size() &&
                         corrections->moved_at[op.index] != ~0u;
      then = Clock::now();
      r.draws_us += Us(then - at);
      at = then;
      Upload(f, d, op.index, corrections, moved, &base_vertex, &base_index, &r);
      then = Clock::now();
      r.vertices_us += Us(then - at);
      at = then;
      if (d.expand)
        base_vertex <<= 2;
      g_gfx->DrawIndexed(base_index, d.num_indices, base_vertex);
      r.draws_us += Us(Clock::now() - at);
      break;
    }
    case Kind::Clear:
    {
      const ClearOp& c = f.clears[op.index];
      g_framebuffer_manager->ClearEFB(c.rc, c.color, c.alpha, c.z, c.color_value, c.z_value, c.format);
      break;
    }
    case Kind::Copy:
    {
      const CopyOp& c = f.copies[op.index];
      if (!c.entry->texture)
        break;
      const Clock::time_point at = Clock::now();
      RcTcacheEntry e = FrameRedrawAccess::Allocate(*g_texture_cache, c.entry->texture->GetConfig());
      if (!e)
        break;
      FrameRedrawAccess::Copy(*g_texture_cache, e, c.depth, c.src, c.half, c.linear, c.format,
                              c.intensity, c.gamma, c.top, c.bottom, c.coef);
      redirect[c.texture] = e->texture.get();
      scratch.push_back(std::move(e));
      r.copies_us += Us(Clock::now() - at);
      break;
    }
    case Kind::Reinterpret:
      g_framebuffer_manager->ReinterpretPixelData(f.reinterprets[op.index]);
      g_framebuffer_manager->BindEFBFramebuffer();
      break;
    }
    if (op.kind != Kind::Draw)
    {
      // utility draws replace the GX uniforms and the viewport
      vs = gs = ps = ~0u;
      tag = -2;
      view = nullptr;
    }
  }
  // the picture: the true frame's XFB copy, from the replay's EFB into a texture of its own
  const Clock::time_point picture_at = Clock::now();
  if (picture && f.xfb.valid && !r.stopped && !(stop && stop()))
  {
    const XFBCopy& x = f.xfb;
    *picture = FrameRedrawAccess::Allocate(*g_texture_cache, x.config);
    if (*picture)
    {
      (*picture)->SetDimensions(x.native_width, x.native_height, 1);
      (*picture)->SetXfbCopy(x.stride);
      (*picture)->frameCount = FRAMECOUNT_INVALID;
      FrameRedrawAccess::Copy(*g_texture_cache, *picture, x.depth, x.src, false, x.linear, EFBCopyFormat::XFB,
                              false, x.gamma, x.top, x.bottom, x.coef);
    }
  }
  r.picture_us = Us(Clock::now() - picture_at);
  r.cpu_us = Us(Clock::now() - start);

  std::vector<u32> replayed, truth;
  bool read = false;
  if (compare)
  {
    const Clock::time_point submitted = Clock::now();
    g_gfx->WaitForGPUIdle();
    r.gpu_us = Us(Clock::now() - submitted);
    read = ReadColor(*compare, &replayed);
  }
  g_replaying = false;
  g_framebuffer_manager->SwapReplayEFB();
  g_framebuffer_manager->StorePixelFormat(saved_format);
  g_framebuffer_manager->BindEFBFramebuffer();
  const AbstractGfx::ViewportAndScissor& gx = g_gfx->GetViewportAndScissor();
  g_gfx->SetViewport(gx.viewport_x, gx.viewport_y, gx.viewport_width, gx.viewport_height,
                     gx.viewport_near_depth, gx.viewport_far_depth);
  g_gfx->SetScissorRect(gx.scissor_rect);
  vsm.constants = saved_vs;
  gsm.constants = saved_gs;
  psm.constants = saved_ps;
  vsm.dirty = gsm.dirty = psm.dirty = true;

  if (read && ReadColor(*compare, &truth) && truth.size() == replayed.size())
  {
    r.pixels = static_cast<s64>(truth.size());
    r.diff_rgb = r.diff_rgba = 0;
    for (size_t i = 0; i < truth.size(); ++i)
    {
      r.diff_rgba += truth[i] != replayed[i];
      r.diff_rgb += ((truth[i] ^ replayed[i]) & 0x00FFFFFFu) != 0;
    }
    if (r.diff_rgb != 0 && *dumps < 3 && !dump_prefix.empty())
    {
      ++*dumps;
      const std::string at = fmt::format("{}-{}", dump_prefix, frame_number);
      DumpPpm(at + "-true.ppm", truth, compare->GetWidth(), compare->GetHeight());
      DumpPpm(at + "-replay.ppm", replayed, compare->GetWidth(), compare->GetHeight());
    }
  }
  return r;
}

void ReleaseReplay()
{
  s_staging.reset();
}
}  // namespace VideoCommon::FrameRedraw
