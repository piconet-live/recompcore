// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FrameRedraw's record (FrameRedraw.h): a frame's draws, clears, copies and reinterpretations as VideoCommon makes
// them, from one XFB copy to the next.

#include "VideoCommon/FrameRedrawFrame.h"

#include <algorithm>
#include <iterator>

#include "Core/System.h"
#include "VideoCommon/AbstractTexture.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/NativeVertexFormat.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexShaderManager.h"
#include "VideoCommon/XFMemory.h"

namespace VideoCommon::FrameRedraw
{
namespace
{
Frame s_frame;
SamplerState s_bind_sampler[8];
std::array<RcTcacheEntry, 8> s_bind_entry;
BitSet32 s_bind_used;
}  // namespace

Frame& RecordedFrame()
{
  return s_frame;
}

void StartRecord(PixelFormat start_format)
{
  s_frame.Reset();
  s_bind_entry = {};
  s_frame.whole = true;
  s_frame.start_format = start_format;
}

void OnBindTextures(BitSet32 used, const std::array<RcTcacheEntry, 8>& bound,
                    const std::array<SamplerState, 8>& samplers)
{
  s_bind_used = {};
  for (u32 i = 0; i < 8; ++i)
  {
    if (!used[i] || !bound[i] || !bound[i]->texture)
      continue;
    s_bind_used[i] = true;
    s_bind_entry[i] = bound[i];
    s_bind_sampler[i] = samplers[i];
  }
}

void OnConstants()
{
  const Clock::time_point start = Clock::now();
  auto& system = Core::System::GetInstance();
  const bool first = s_frame.vs.empty();
  if (first || system.GetVertexShaderManager().dirty)
  {
    s_frame.vs.push_back(system.GetVertexShaderManager().constants);
    s_frame.perspective.push_back(xfmem.projection.type == ProjectionType::Perspective);
  }
  if (first || system.GetGeometryShaderManager().dirty)
    s_frame.gs.push_back(system.GetGeometryShaderManager().constants);
  if (first || system.GetPixelShaderManager().dirty)
    s_frame.ps.push_back(system.GetPixelShaderManager().constants);
  s_frame.record_us += Us(Clock::now() - start);
}

void OnDraw(const AbstractPipeline* pipeline, const u8* vertices, u32 num_vertices, u32 stride,
            const u16* indices, u32 num_indices, bool expand_base_vertex)
{
  if (s_frame.vs.empty() || s_frame.gs.empty() || s_frame.ps.empty())
    return;
  const Clock::time_point start = Clock::now();
  Frame& f = s_frame;
  Draw d{};
  d.pipeline = pipeline;
  d.vertex_at = static_cast<u32>(f.vertices.size());
  d.num_vertices = num_vertices;
  d.stride = stride;
  d.index_at = static_cast<u32>(f.indices.size());
  d.num_indices = num_indices;
  d.vs = static_cast<u32>(f.vs.size() - 1);
  d.gs = static_cast<u32>(f.gs.size() - 1);
  d.ps = static_cast<u32>(f.ps.size() - 1);
  d.view = g_gfx->GetViewportAndScissor();
  d.textures = s_bind_used;
  d.expand = expand_base_vertex;
  d.offscreen = false;
  d.address = {};
  d.copied = 0;
  {
    // where the draw's model is in the view: its position matrix's translation (XF rows of 4)
    const u32 m = g_main_cp_state.matrix_index_a.PosNormalMtxIdx * 4;
    d.mtx = {};
    if (m + 11 < std::size(xfmem.posMatrices))
      std::copy_n(&xfmem.posMatrices[m], 12, d.mtx.begin());
    const PortableVertexDeclaration& decl = VertexLoaderManager::GetCurrentVertexFormat()->GetVertexDeclaration();
    d.position = decl.position.enable && decl.position.type == ComponentFormat::Float &&
                         decl.position.components >= 3 ?
                     decl.position.offset :
                     -1;
    d.vertex_matrices = decl.posmtx.enable;
    d.ztest = bpmem.zmode.test_enable != 0;
  }
  for (const u32 i : s_bind_used)
  {
    d.texture[i] = s_bind_entry[i]->texture.get();
    d.sampler[i] = s_bind_sampler[i];
    d.address[i] = s_bind_entry[i]->addr;
    if (s_bind_entry[i]->IsEfbCopy())
      d.copied |= static_cast<u8>(1u << i);
    if (f.kept[i] != s_bind_entry[i].get())
    {
      f.kept[i] = s_bind_entry[i].get();
      f.keep.push_back(s_bind_entry[i]);
    }
  }
  f.vertices.insert(f.vertices.end(), vertices, vertices + static_cast<size_t>(num_vertices) * stride);
  f.indices.insert(f.indices.end(), indices, indices + num_indices);
  f.last_read = ReadPosition();
  if (f.draws.empty())
    f.first_read = f.last_read;
  f.ops.push_back({Kind::Draw, static_cast<u32>(f.draws.size())});
  f.draws.push_back(d);
  f.record_us += Us(Clock::now() - start);
}

void OnClear(const MathUtil::Rectangle<int>& rc, bool color, bool alpha, bool z, u32 color_value,
             u32 z_value, PixelFormat format)
{
  s_frame.ops.push_back({Kind::Clear, static_cast<u32>(s_frame.clears.size())});
  s_frame.clears.push_back({rc, color, alpha, z, color_value, z_value, format});
}

void OnCopy(const RcTcacheEntry& entry, bool is_depth_copy, const MathUtil::Rectangle<int>& src_rect,
            bool scale_by_half, bool linear_filter, EFBCopyFormat dst_format, bool is_intensity,
            float gamma, bool clamp_top, bool clamp_bottom,
            const std::array<u32, 3>& filter_coefficients)
{
  if (!entry || !entry->texture)
    return;
  s_frame.ops.push_back({Kind::Copy, static_cast<u32>(s_frame.copies.size())});
  MathUtil::Rectangle<int> scaled = g_framebuffer_manager->ConvertEFBRectangle(src_rect);
  scaled = g_gfx->ConvertFramebufferRectangle(scaled, g_framebuffer_manager->GetEFBFramebuffer());
  s_frame.copies.push_back({entry, entry->texture.get(), is_depth_copy, src_rect, scale_by_half,
                            linear_filter, dst_format, is_intensity, gamma, clamp_top, clamp_bottom,
                            filter_coefficients, scaled});
}

void OnXFBEntry(const RcTcacheEntry& entry, bool is_depth_copy, const MathUtil::Rectangle<int>& src_rect,
                bool linear_filter, float gamma, bool clamp_top, bool clamp_bottom,
                const std::array<u32, 3>& filter_coefficients, u32 fb_width, u32 fb_height)
{
  if (!entry || !entry->texture || entry->native_width == 0 || entry->native_height == 0)
    return;
  XFBCopy& x = s_frame.xfb;
  x.valid = true;
  x.config = entry->texture->GetConfig();
  x.native_width = entry->native_width;
  x.native_height = entry->native_height;
  x.stride = entry->memory_stride;
  x.depth = is_depth_copy;
  x.linear = linear_filter;
  x.src = src_rect;
  x.gamma = gamma;
  x.top = clamp_top;
  x.bottom = clamp_bottom;
  x.coef = filter_coefficients;
  if (g_keep_xfb)
    x.entry = entry;
  // (TextureCacheBase's GetDisplayRectForXFBEntry: the presenter's rectangle of a fetched XFB)
  x.display = MathUtil::Rectangle<int>(0, 0, static_cast<int>(fb_width * entry->GetWidth() / entry->native_width),
                                       static_cast<int>(fb_height * entry->GetHeight() / entry->native_height));
}

void OnReinterpret(EFBReinterpretType type)
{
  s_frame.ops.push_back({Kind::Reinterpret, static_cast<u32>(s_frame.reinterprets.size())});
  s_frame.reinterprets.push_back(type);
}

}  // namespace VideoCommon::FrameRedraw
