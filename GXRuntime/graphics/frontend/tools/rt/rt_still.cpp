// rt_still CAPTURE RASTER.raw OUT.raw [mode] [spp] [gain] [roadShine] [rough] [missR missG missB]
// One still: a frame captured by AURORA_RT_CAPTURE, its raster picture, and reflections traced by DXR 1.1 inline rays.
// The compute pass and its shader are what the live pass uses; only the frame's source differs.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
using u32 = uint32_t; using u8 = uint8_t;

#include "rt_rules.hpp"
using namespace aurora::gx::rt;
struct DrawGpu { u32 stages, cls; float strength; u32 highlight; float regs[4][4]; float swap[4][4]; StageRec stage[16]; };
struct Params {
  float size[2], rectXY[2], rectWH[2], vpXY[2], vpWH[2], pScale[2], pOffset[2];
  float pad0[2]; float miss[3]; u32 spp; u32 mode; float gain, roadShine, rough; u32 frame, textures; float conserve; float lightGain; u32 lightCount; float lightRange; u32 lightShadow; float bodyCalm;
};
struct Tex { u32 w, h, wrapS, wrapT; std::vector<u8> rgba; };

#define HR(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::fprintf(stderr, "failed %s: %08lx (line %d)\n", #x, hr_, __LINE__); std::exit(1);} } while (0)

static ComPtr<ID3D12Device5> dev;
static ComPtr<ID3D12Resource> buffer(u32 heapType, UINT64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES hp{}; hp.Type = (D3D12_HEAP_TYPE)heapType;
  D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = std::max<UINT64>(size, 4); d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = flags;
  ComPtr<ID3D12Resource> r; HR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))); return r;
}
static std::vector<u8> read_file(const char* p) { std::ifstream f(p, std::ios::binary); if (!f) { std::fprintf(stderr, "cannot read %s\n", p); std::exit(1);} return std::vector<u8>(std::istreambuf_iterator<char>(f), {}); }

int main(int argc, char** argv) {
  if (argc < 4) { std::puts("rt_still CAPTURE RASTER.raw OUT.raw [mode spp gain roadShine rough missR missG missB]"); return 2; }
  const u32 mode = argc > 4 ? atoi(argv[4]) : 0, spp = argc > 5 ? atoi(argv[5]) : 16;
  const float gain = argc > 6 ? (float)atof(argv[6]) : 1.f, road = argc > 7 ? (float)atof(argv[7]) : 0.f, rough = argc > 8 ? (float)atof(argv[8]) : 0.004f;
  float miss[3] = {0.03f, 0.03f, 0.06f};
  if (argc > 11) for (int i = 0; i < 3; ++i) miss[i] = (float)atof(argv[9 + i]);

  auto cap = read_file(argv[1]);
  const u32* head = (const u32*)cap.data();
  if (head[0] != 0x33435452u) { std::fprintf(stderr, "not a capture\n"); return 2; }
  const u32 nDraws = head[1], nVerts = head[2], nTex = head[3];
  const float* rect = (const float*)(head + 4);
  const u8* p = (const u8*)(rect + 4);
  const DrawRec* draws = (const DrawRec*)p; p += sizeof(DrawRec) * nDraws;
  const Vertex* verts = (const Vertex*)p; p += sizeof(Vertex) * nVerts;
  std::vector<Tex> texs(nTex);
  for (auto& t : texs) { const u32* h = (const u32*)p; t.w = h[0]; t.h = h[1]; t.wrapS = h[2]; t.wrapT = h[3]; p += 16; t.rgba.assign(p, p + size_t(t.w) * t.h * 4); p += size_t(t.w) * t.h * 4; }

  // The main view: the perspective projection with the most triangles
  std::map<std::pair<int,int>, u32> groups;
  auto key = [](const DrawRec& d) { return std::make_pair((int)(d.proj[0] * 1000), (int)(d.proj[5] * 1000)); };
  for (u32 i = 0; i < nDraws; ++i) if (draws[i].projType == 0) groups[key(draws[i])] += draws[i].vertices;
  std::pair<int,int> best{}; u32 bestN = 0;
  for (auto& g : groups) { std::printf("view p00 %.3f p11 %.3f: %u triangles\n", g.first.first / 1000.f, g.first.second / 1000.f, g.second / 3); if (g.second > bestN) bestN = g.second, best = g.first; }
  const DrawRec* mainDraw = nullptr;
  std::vector<DrawGpu> gpu(nDraws);
  std::vector<Vertex> kept; kept.reserve(nVerts);
  u32 cls[4] = {}, reflective = 0;
  for (u32 i = 0; i < nDraws; ++i) {
    const DrawRec& d = draws[i];
    auto& g = gpu[i];
    g.stages = std::min(d.stages, 16u);
    std::memcpy(g.regs, d.regs, sizeof g.regs); std::memcpy(g.swap, d.swap, sizeof g.swap); std::memcpy(g.stage, d.stage, sizeof g.stage);
    float strength = 0;
    for (u32 s = 0; s < g.stages; ++s) if (d.stage[s].reflection == 1) {
      const auto& st = d.stage[s];
      float k = st.cc == 14 ? std::max({st.k[0], st.k[1], st.k[2]}) : 1.f;
      strength = std::max(strength, k);
    }
    g.strength = strength;
    for (u32 s = 0; s < g.stages; ++s) if (d.stage[s].reflection == 2) g.highlight = 1;
    // the blend the draw asks for: GX factors ZERO 0, ONE 1, SRCCLR 2, INVSRCCLR 3, SRCALPHA 4, INVSRCALPHA 5
    const bool additive = d.blendMode == 1 && d.blendDst == 1;
    const bool darkens = d.blendMode == 1 && (d.blendDst == 2 || d.blendSrc == 0 || d.blendSrc == 2);
    const bool opaqueBlend = d.blendMode == 0 || (d.blendSrc == 1 && d.blendDst == 0);
    bool translucent = false;
    if (!opaqueBlend && !additive && !darkens) {
      // normal blend: translucent when the alpha can be below one: raster alpha, or a texture's alpha in the alpha stages
      for (u32 v = 0; v < d.vertices && !translucent; v += 3) if (verts[d.firstVertex + v].ras[3] < 0.99f) translucent = true;
      for (u32 s = 0; s < g.stages && !translucent; ++s) { const auto& st = d.stage[s]; if (st.aa == 4 || st.ab == 4 || st.ac == 4 || st.ad == 4) translucent = true; if (st.aa==6||st.ab==6||st.ac==6||st.ad==6) translucent = true; }
    }
    g.cls = !d.colorUpdate || darkens ? 3 : additive ? 2 : translucent ? 1 : 0;
    bool inView = d.projType == 0 && key(d) == best;
    if (!inView || d.vertices == 0) continue;
    if (!mainDraw) mainDraw = &d;
    cls[g.cls] += d.vertices / 3; if (strength > 0) reflective += d.vertices / 3;
    kept.insert(kept.end(), verts + d.firstVertex, verts + d.firstVertex + d.vertices);
  }
  if (!mainDraw) { std::fprintf(stderr, "no perspective view\n"); return 2; }
  static std::pair<int,int> s_best; s_best = best;
  auto lights = engine_lights(draws, nDraws, verts, [](const DrawRec& d) { return d.projType == 0 && std::make_pair((int)(d.proj[0] * 1000), (int)(d.proj[5] * 1000)) == s_best; });
  std::printf("engine lights %zu:", lights.size()); for (auto& l : lights) std::printf(" (%.1f %.1f %.1f; %.2f %.2f %.2f)", l.pos[0], l.pos[1], l.pos[2], l.colour[0], l.colour[1], l.colour[2]); std::puts("");
  if (lights.empty()) lights.push_back({{0,0,1e6f},{0,0,0}});
  float maxz = -1e30f, minz = 1e30f; u32 behind = 0;
  for (auto& v : kept) { maxz = std::max(maxz, v.pos[2]); minz = std::min(minz, v.pos[2]); if (v.pos[2] > 0) ++behind; }
  std::printf("kept %zu triangles: solid %u translucent %u additive %u ignored %u; reflective %u; textures %u; behind the camera %.1f%%; z %.0f .. %.0f\n",
              kept.size() / 3, cls[0], cls[1], cls[2], cls[3], reflective, nTex, 100.f * behind / kept.size(), minz, maxz);
  std::printf("proj:"); for (int i = 0; i < 16; ++i) std::printf(" %.4g", mainDraw->proj[i]); std::printf("\nviewport %g %g %g %g; rect %g %g %g %g\n", mainDraw->viewport[0], mainDraw->viewport[1], mainDraw->viewport[2], mainDraw->viewport[3], rect[0], rect[1], rect[2], rect[3]);

  auto ras = read_file(argv[2]);
  const u32 W = ((u32*)ras.data())[0], H = ((u32*)ras.data())[1];
  std::printf("raster %ux%u\n", W, H);

  if (std::getenv("RT_DEBUG")) { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer(); }
  ComPtr<IDXGIFactory6> fac; HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)));
  ComPtr<IDXGIAdapter1> ad; HR(fac->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad)));
  DXGI_ADAPTER_DESC1 desc; ad->GetDesc1(&desc); std::printf("adapter %ls\n", desc.Description);
  HR(D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&dev)));
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{}; HR(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof o5));
  std::printf("raytracing tier %d\n", (int)o5.RaytracingTier);
  if (o5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1) { std::fprintf(stderr, "needs DXR 1.1\n"); return 3; }

  D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ComPtr<ID3D12CommandQueue> queue; HR(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
  ComPtr<ID3D12CommandAllocator> alloc; HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
  ComPtr<ID3D12GraphicsCommandList4> cl; HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)));
  ComPtr<ID3D12Fence> fence; HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  UINT64 fv = 0; HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  auto flush = [&]() { HR(cl->Close()); ID3D12CommandList* l[] = {cl.Get()}; queue->ExecuteCommandLists(1, l); ++fv; HR(queue->Signal(fence.Get(), fv)); if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, INFINITE);} HR(alloc->Reset()); HR(cl->Reset(alloc.Get(), nullptr)); };
  std::vector<ComPtr<ID3D12Resource>> keep;
  auto upload = [&](const void* src, UINT64 size, D3D12_RESOURCE_FLAGS fl = D3D12_RESOURCE_FLAG_NONE) {
    auto u = buffer(D3D12_HEAP_TYPE_UPLOAD, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* m; HR(u->Map(0, nullptr, &m)); std::memcpy(m, src, size); u->Unmap(0, nullptr);
    auto b = buffer(D3D12_HEAP_TYPE_DEFAULT, size, fl, D3D12_RESOURCE_STATE_COMMON);
    cl->CopyResource(b.Get(), u.Get()); keep.push_back(u); return b;
  };

  auto vb = upload(kept.data(), kept.size() * sizeof(Vertex));
  auto db = upload(gpu.data(), gpu.size() * sizeof(DrawGpu));
  auto lb = upload(lights.data(), lights.size() * sizeof(EngineLight));

  setvbuf(stdout,nullptr,_IONBF,0);
  std::fputs("[A] ",stdout); std::fflush(stdout);
  // The textures, with mips by box filter
  D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 8 + std::max(nTex, 1u); hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ComPtr<ID3D12DescriptorHeap> heap; HR(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
  const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto cpuh = [&](u32 i) { auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * inc; return h; };
  auto gpuh = [&](u32 i) { auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * inc; return h; };
  std::vector<ComPtr<ID3D12Resource>> texRes;
  D3D12_HEAP_PROPERTIES dh{}; dh.Type = D3D12_HEAP_TYPE_DEFAULT;
  for (u32 i = 0; i < nTex; ++i) {
    auto& t = texs[i];
    std::vector<std::vector<u8>> mips; mips.push_back(t.rgba);
    u32 w = t.w, h = t.h;
    while (w > 1 || h > 1) {
      u32 nw = std::max(w / 2, 1u), nh = std::max(h / 2, 1u); std::vector<u8> m(size_t(nw) * nh * 4); auto& s = mips.back();
      for (u32 y = 0; y < nh; ++y) for (u32 x = 0; x < nw; ++x) for (u32 c = 0; c < 4; ++c) { u32 x1 = std::min(x * 2 + 1, w - 1), y1 = std::min(y * 2 + 1, h - 1), xa = std::min(x*2,w-1), ya=std::min(y*2,h-1);
        m[(size_t(y) * nw + x) * 4 + c] = (u8)((s[(size_t(ya) * w + xa) * 4 + c] + s[(size_t(ya) * w + x1) * 4 + c] + s[(size_t(y1) * w + xa) * 4 + c] + s[(size_t(y1) * w + x1) * 4 + c] + 2) / 4); }
      mips.push_back(std::move(m)); w = nw; h = nh;
    }
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = t.w; d.Height = t.h; d.DepthOrArraySize = 1; d.MipLevels = (UINT16)mips.size(); d.Format = DXGI_FORMAT_R8G8B8A8_UNORM; d.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> r; HR(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r)));
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(mips.size()); UINT64 total; dev->GetCopyableFootprints(&d, 0, (UINT)mips.size(), 0, fp.data(), nullptr, nullptr, &total);
    auto u = buffer(D3D12_HEAP_TYPE_UPLOAD, total, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    u8* m; HR(u->Map(0, nullptr, (void**)&m));
    for (u32 l = 0; l < mips.size(); ++l) { u32 mw = fp[l].Footprint.Width, mh = fp[l].Footprint.Height; for (u32 y = 0; y < mh; ++y) std::memcpy(m + fp[l].Offset + size_t(y) * fp[l].Footprint.RowPitch, mips[l].data() + size_t(y) * mw * 4, size_t(mw) * 4);
      D3D12_TEXTURE_COPY_LOCATION dst{r.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; dst.SubresourceIndex = l; D3D12_TEXTURE_COPY_LOCATION src{u.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}; src.PlacedFootprint = fp[l]; cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr); }
    u->Unmap(0, nullptr); keep.push_back(u);
    D3D12_RESOURCE_BARRIER b{}; b.Transition.pResource = r.Get(); b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; cl->ResourceBarrier(1, &b);
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{}; sv.Format = d.Format; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = d.MipLevels;
    dev->CreateShaderResourceView(r.Get(), &sv, cpuh(8 + i)); texRes.push_back(r);
    if (i % 32 == 31) flush();
  }
  std::fputs("[B] ",stdout); std::fflush(stdout);
  // The raster picture and the output
  D3D12_RESOURCE_DESC td{}; td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = W; td.Height = H; td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
  ComPtr<ID3D12Resource> rast; HR(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rast)));
  {
    const UINT pitch = (W * 4 + 255) & ~255u; auto u = buffer(D3D12_HEAP_TYPE_UPLOAD, UINT64(pitch) * H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    u8* m; HR(u->Map(0, nullptr, (void**)&m)); for (u32 y = 0; y < H; ++y) std::memcpy(m + size_t(y) * pitch, ras.data() + 8 + size_t(y) * W * 4, W * 4); u->Unmap(0, nullptr); keep.push_back(u);
    D3D12_TEXTURE_COPY_LOCATION dst{rast.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; D3D12_TEXTURE_COPY_LOCATION src{u.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM; src.PlacedFootprint.Footprint.Width = W; src.PlacedFootprint.Footprint.Height = H; src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = pitch; cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER b{}; b.Transition.pResource = rast.Get(); b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; cl->ResourceBarrier(1, &b);
  }
  td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> out; HR(dev->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out)));

  HR(cl->Close()); { ID3D12CommandList* l[] = {cl.Get()}; queue->ExecuteCommandLists(1, l); ++fv; HR(queue->Signal(fence.Get(), fv)); fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, INFINITE);} HR(alloc->Reset()); HR(cl->Reset(alloc.Get(), nullptr)); keep.clear();

  std::fputs("[C] ",stdout); std::fflush(stdout);
  // One BLAS of the frame's triangles, in view space, and a TLAS of one instance
  D3D12_RAYTRACING_GEOMETRY_DESC geo{}; geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES; geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
  geo.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress(); geo.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex); geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; geo.Triangles.VertexCount = (UINT)kept.size();
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{}; in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL; in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; in.NumDescs = 1; in.pGeometryDescs = &geo; in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{}; dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &pre);
  auto blas = buffer(D3D12_HEAP_TYPE_DEFAULT, pre.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  auto scr = buffer(D3D12_HEAP_TYPE_DEFAULT, pre.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{}; bd.Inputs = in; bd.DestAccelerationStructureData = blas->GetGPUVirtualAddress(); bd.ScratchAccelerationStructureData = scr->GetGPUVirtualAddress();
  cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
  D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = blas.Get(); cl->ResourceBarrier(1, &uav);
  D3D12_RAYTRACING_INSTANCE_DESC inst{}; inst.Transform[0][0] = inst.Transform[1][1] = inst.Transform[2][2] = 1; inst.InstanceMask = 0xFF; inst.AccelerationStructure = blas->GetGPUVirtualAddress();
  auto instU = buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof inst, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ); { void* m; HR(instU->Map(0, nullptr, &m)); std::memcpy(m, &inst, sizeof inst); instU->Unmap(0, nullptr);} 
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tin{}; tin.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL; tin.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY; tin.NumDescs = 1; tin.InstanceDescs = instU->GetGPUVirtualAddress();
  dev->GetRaytracingAccelerationStructurePrebuildInfo(&tin, &pre);
  auto tlas = buffer(D3D12_HEAP_TYPE_DEFAULT, pre.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  auto scratchTop = buffer(D3D12_HEAP_TYPE_DEFAULT, pre.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  bd = {}; bd.Inputs = tin; bd.DestAccelerationStructureData = tlas->GetGPUVirtualAddress(); bd.ScratchAccelerationStructureData = scratchTop->GetGPUVirtualAddress();
  cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
  uav.UAV.pResource = tlas.Get(); cl->ResourceBarrier(1, &uav);

  std::fputs("[D] ",stdout); std::fflush(stdout);
  // The shader
  auto src = read_file(argc > 12 ? argv[12] : "rt.hlsl");
  ComPtr<IDxcUtils> utils; ComPtr<IDxcCompiler3> comp; HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))); HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&comp)));
  DxcBuffer sb{src.data(), src.size(), DXC_CP_UTF8};
  LPCWSTR args[] = {L"-T", L"cs_6_5", L"-E", L"main", L"-O3", L"-HV", L"2021"};
  ComPtr<IDxcResult> res; HR(comp->Compile(&sb, args, _countof(args), nullptr, IID_PPV_ARGS(&res)));
  ComPtr<IDxcBlobUtf8> errs; res->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr); if (errs && errs->GetStringLength()) std::fprintf(stderr, "%s\n", errs->GetStringPointer());
  HRESULT status; res->GetStatus(&status); if (FAILED(status)) return 4;
  ComPtr<IDxcBlob> cs; HR(res->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&cs), nullptr));

  std::fputs("[E] ",stdout); std::fflush(stdout);
  // Root: constants, then t0-t3/u0 by a table, the texture array by a table
  D3D12_DESCRIPTOR_RANGE1 r1[2] = {}; r1[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; r1[0].NumDescriptors = 5; r1[0].OffsetInDescriptorsFromTableStart = 0; r1[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  r1[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; r1[1].NumDescriptors = 1; r1[1].OffsetInDescriptorsFromTableStart = 5; r1[1].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  D3D12_DESCRIPTOR_RANGE1 r2{}; r2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; r2.NumDescriptors = UINT_MAX; r2.RegisterSpace = 1; r2.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  D3D12_ROOT_PARAMETER1 rp[3] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[0].Constants.Num32BitValues = sizeof(Params) / 4; 
  rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[1].DescriptorTable.NumDescriptorRanges = 2; rp[1].DescriptorTable.pDescriptorRanges = r1;
  rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[2].DescriptorTable.NumDescriptorRanges = 1; rp[2].DescriptorTable.pDescriptorRanges = &r2;
  D3D12_STATIC_SAMPLER_DESC ss{}; ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR; ss.AddressU = ss.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP; ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP; ss.MaxLOD = D3D12_FLOAT32_MAX; ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{}; rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_1; rs.Desc_1_1.NumParameters = 3; rs.Desc_1_1.pParameters = rp; rs.Desc_1_1.NumStaticSamplers = 1; rs.Desc_1_1.pStaticSamplers = &ss;
  ComPtr<ID3DBlob> rsb, rse; HRESULT rh = D3D12SerializeVersionedRootSignature(&rs, &rsb, &rse); if (FAILED(rh)) { std::fprintf(stderr, "%s\n", rse ? (char*)rse->GetBufferPointer() : "root"); return 4; }
  ComPtr<ID3D12RootSignature> root; HR(dev->CreateRootSignature(0, rsb->GetBufferPointer(), rsb->GetBufferSize(), IID_PPV_ARGS(&root)));
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = root.Get(); pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
  ComPtr<ID3D12PipelineState> pso; HR(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));

  D3D12_SHADER_RESOURCE_VIEW_DESC sv{}; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE; sv.RaytracingAccelerationStructure.Location = tlas->GetGPUVirtualAddress(); dev->CreateShaderResourceView(nullptr, &sv, cpuh(0));
  sv = {}; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sv.Buffer.NumElements = (UINT)kept.size(); sv.Buffer.StructureByteStride = sizeof(Vertex); dev->CreateShaderResourceView(vb.Get(), &sv, cpuh(1));
  sv.Buffer.NumElements = (UINT)gpu.size(); sv.Buffer.StructureByteStride = sizeof(DrawGpu); dev->CreateShaderResourceView(db.Get(), &sv, cpuh(2));
  sv = {}; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Texture2D.MipLevels = 1; dev->CreateShaderResourceView(rast.Get(), &sv, cpuh(3));
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{}; uv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; dev->CreateUnorderedAccessView(out.Get(), nullptr, &uv, cpuh(5));
  sv = {}; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sv.Buffer.NumElements = (UINT)lights.size(); sv.Buffer.StructureByteStride = sizeof(EngineLight); dev->CreateShaderResourceView(lb.Get(), &sv, cpuh(4));

  std::fputs("[F] ",stdout); std::fflush(stdout);
  Params par{}; par.size[0] = (float)W; par.size[1] = (float)H; par.rectXY[0] = rect[0]; par.rectXY[1] = rect[1]; par.rectWH[0] = rect[2]; par.rectWH[1] = rect[3];
  par.vpXY[0] = mainDraw->viewport[0]; par.vpXY[1] = mainDraw->viewport[1]; par.vpWH[0] = mainDraw->viewport[2]; par.vpWH[1] = mainDraw->viewport[3];
  par.pScale[0] = mainDraw->proj[0]; par.pScale[1] = mainDraw->proj[5]; par.pOffset[0] = mainDraw->proj[2]; par.pOffset[1] = mainDraw->proj[6];
  std::memcpy(par.miss, miss, sizeof miss); par.spp = spp; par.mode = mode; par.gain = gain; par.roadShine = road; par.rough = rough; par.textures = nTex; par.conserve = std::getenv("RT_CONSERVE") ? (float)atof(std::getenv("RT_CONSERVE")) : 0.7f;
  auto envf = [](const char* n, float v) { const char* e = std::getenv(n); return e ? (float)atof(e) : v; };
  par.lightGain = envf("RT_LIGHT", 0.f); par.lightCount = (u32)lights.size(); par.lightRange = envf("RT_LIGHT_RANGE", 4.f); par.lightShadow = (u32)envf("RT_LIGHT_SHADOW", 0.f); par.bodyCalm = envf("RT_BODY_CALM", 1.f);
  ID3D12DescriptorHeap* hs[] = {heap.Get()}; cl->SetDescriptorHeaps(1, hs); cl->SetComputeRootSignature(root.Get()); cl->SetPipelineState(pso.Get());
  cl->SetComputeRoot32BitConstants(0, sizeof(Params) / 4, &par, 0); cl->SetComputeRootDescriptorTable(1, gpuh(0)); cl->SetComputeRootDescriptorTable(2, gpuh(8));
  ComPtr<ID3D12QueryHeap> qh; D3D12_QUERY_HEAP_DESC qhd{}; qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qhd.Count = 2; HR(dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh)));
  auto qr = buffer(D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
  cl->Dispatch((W + 7) / 8, (H + 7) / 8, 1);
  cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1); cl->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qr.Get(), 0);
  D3D12_RESOURCE_BARRIER b{}; b.Transition.pResource = out.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; cl->ResourceBarrier(1, &b);
  const UINT pitch = (W * 8 + 255) & ~255u; auto rb = buffer(D3D12_HEAP_TYPE_READBACK, UINT64(pitch) * H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION dst{rb.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}; dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; dst.PlacedFootprint.Footprint.Width = W; dst.PlacedFootprint.Footprint.Height = H; dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = pitch; D3D12_TEXTURE_COPY_LOCATION srcL{out.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; cl->CopyTextureRegion(&dst, 0, 0, 0, &srcL, nullptr);
  std::fputs("[G] ",stdout); std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now(); flush(); const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  { HRESULT rr = dev->GetDeviceRemovedReason(); if (rr != S_OK) { std::fprintf(stderr, "device removed: %08lx\n", rr); return 5; } }
  UINT64 freq; queue->GetTimestampFrequency(&freq); UINT64* ts; HR(qr->Map(0, nullptr, (void**)&ts)); std::printf("trace %.2f ms GPU at %ux%u, %u samples; build and trace together %.2f ms wall\n", double(ts[1] - ts[0]) * 1000.0 / freq, W, H, spp, ms); qr->Unmap(0, nullptr);
  std::vector<u8> outPx(size_t(W) * H * 4); uint16_t* hf; HR(rb->Map(0, nullptr, (void**)&hf));
  auto half = [](uint16_t h) { u32 s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v = e == 0 ? m * 5.9604645e-8f : e == 31 ? 1e9f : (1.f + m / 1024.f) * std::ldexp(1.f, (int)e - 15); return s ? -v : v; };
  for (u32 y = 0; y < H; ++y) { auto* row = (uint16_t*)((u8*)hf + size_t(y) * pitch); for (u32 x = 0; x < W; ++x) for (u32 c = 0; c < 4; ++c) outPx[(size_t(y) * W + x) * 4 + c] = (u8)(std::clamp(half(row[x * 4 + c]), 0.f, 1.f) * 255.f + 0.5f); }
  rb->Unmap(0, nullptr);
  std::ofstream of(argv[3], std::ios::binary); of.write((const char*)&W, 4); of.write((const char*)&H, 4); of.write((const char*)outPx.data(), outPx.size());
  std::puts("done");
  return 0;
}
