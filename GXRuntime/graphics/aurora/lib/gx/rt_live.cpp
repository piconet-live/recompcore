// The live pass (AURORA_RT=1): the rules and the shader are API-neutral (rt_rules.hpp, rt.hlsl); this file holds what is D3D12:
// inline ray queries on Dawn's own device and queue. A frame's encoder is cut in two where the HUD begins. The first part, the
// scene, is submitted, the rays run on the same queue, then the second part draws their picture back into the EFB and goes on
// with the HUD. Presentation only: nothing here reads or writes anything the game can see.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rt_live.hpp"

#include "rt_format.hpp"
#include "rt_rules.hpp"

#include <aurora/gfx.hpp>
#include <dawn/native/D3D12Backend.h>
#include <dxcapi.h>
#include <webgpu/webgpu_cpp.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../webgpu/gpu.hpp"
#include "fifo.hpp"

namespace aurora::gx::rt::live {
namespace {
using Microsoft::WRL::ComPtr;
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

const unsigned char kShader[] = {
#include "rt_shader.inc"
};

// As rt.hlsl's Params
struct Params {
  float size[2], rectXY[2], rectWH[2], vpXY[2], vpWH[2], pScale[2], pOffset[2];
  float pad0[2];
  float miss[3];
  u32 spp;
  u32 mode;
  float gain, roadShine, rough;
  u32 frame, textures;
  float conserve, lightGain;
  u32 lightCount;
  float lightRange;
  u32 lightShadow;
  float bodyCalm;
  float projZ[2];
  u32 skyFirst, pad2;
};

constexpr u32 Frames = 3, SlotDescriptors = 12, MaxTextures = 8192;

struct Slot {
  ComPtr<ID3D12CommandAllocator> alloc;
  ComPtr<ID3D12GraphicsCommandList4> cl;
  u64 fence = 0;
  struct Buf {
    ComPtr<ID3D12Resource> upload, gpu;
    u64 size = 0;
    void* map = nullptr;
  } verts, draws, stages, lights;
  ComPtr<ID3D12Resource> blas[2], tlas, scratch, instances;
  u64 blasSize[2]{}, tlasSize = 0, scratchSize = 0;
  std::vector<ComPtr<ID3D12Resource>> keep;
};
struct State {
  bool failed = false, ready = false;
  ComPtr<ID3D12Device5> dev;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  u64 fenceValue = 0;
  ComPtr<ID3D12DescriptorHeap> heap;
  UINT inc = 0;
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3D12PipelineState> trace, compose;
  ComPtr<ID3D12QueryHeap> queries;
  ComPtr<ID3D12Resource> queryResult;
  u64 gpuTicks = 0;
  Slot slots[Frames];
  u32 slot = 0, frame = 0, textures = 0;
  std::vector<ComPtr<ID3D12Resource>> textureRes;

  u32 w = 0, h = 0;
  ComPtr<ID3D12Resource> inputRes, outputRes, termRes;
  wgpu::SharedTextureMemory inputMem, outputMem;
  wgpu::Texture input, output;
  wgpu::BindGroup bind;
  wgpu::RenderPipeline pipeline;
  u64 pipelineKey = 0;
  wgpu::BindGroupLayout bgl;
  gfx::EncoderTaskId task = gfx::InvalidEncoderTask;
  gfx::DrawTypeId draw = gfx::InvalidDrawType;

  std::unique_ptr<Job> pending;
  bool drawThis = false;
  wgpu::CommandBuffer first;
  bool outputOpen = false;

  double msTrace = 0, msGpu = 0, msComposeWait = 0, msBuild = 0, msRules = 0, msTranslate = 0;
  u64 tris = 0, lightsSeen = 0, lightsMost = 0, reports = 0;
  u32 count = 0;
} g;

std::mutex s_poolLock;
std::vector<std::unique_ptr<Job>> s_pool;
void give_back(std::unique_ptr<Job> job) {
  std::lock_guard l{s_poolLock};
  s_pool.push_back(std::move(job));
}

void fail(const char* why) {
  std::fprintf(stderr, "[rt] live pass off: %s\n", why);
  std::fflush(stderr);
  g.failed = true;
}
#define RT_HR(x)                                                                                                       \
  do {                                                                                                                 \
    const HRESULT hr_ = (x);                                                                                           \
    if (FAILED(hr_)) {                                                                                                 \
      std::fprintf(stderr, "[rt] failed %s: %08lx\n", #x, static_cast<unsigned long>(hr_));                            \
      g.failed = true;                                                                                                 \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

ComPtr<ID3D12Resource> make_buffer(D3D12_HEAP_TYPE type, u64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = type;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  d.Width = std::max<u64>(size, 16);
  d.Height = 1;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.SampleDesc.Count = 1;
  d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  d.Flags = flags;
  ComPtr<ID3D12Resource> r;
  if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))))
    g.failed = true;
  return r;
}
D3D12_CPU_DESCRIPTOR_HANDLE cpuh(u32 i) {
  auto h = g.heap->GetCPUDescriptorHandleForHeapStart();
  h.ptr += SIZE_T(i) * g.inc;
  return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE gpuh(u32 i) {
  auto h = g.heap->GetGPUDescriptorHandleForHeapStart();
  h.ptr += u64(i) * g.inc;
  return h;
}

ComPtr<IDxcBlob> compile(IDxcCompiler3* comp, const wchar_t* entry) {
  DxcBuffer src{kShader, sizeof kShader, DXC_CP_UTF8};
  LPCWSTR args[] = {L"-T", L"cs_6_5", L"-E", entry, L"-O3", L"-HV", L"2021"};
  ComPtr<IDxcResult> res;
  if (FAILED(comp->Compile(&src, args, _countof(args), nullptr, IID_PPV_ARGS(&res))))
    return nullptr;
  ComPtr<IDxcBlobUtf8> errs;
  res->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
  if (errs && errs->GetStringLength())
    std::fprintf(stderr, "[rt] shader: %s\n", errs->GetStringPointer());
  HRESULT status = E_FAIL;
  res->GetStatus(&status);
  if (FAILED(status))
    return nullptr;
  ComPtr<IDxcBlob> out;
  res->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&out), nullptr);
  return out;
}

// On the render worker, once
void init_device() {
  const auto t0 = std::chrono::steady_clock::now();
  if (webgpu::g_backendType != wgpu::BackendType::D3D12)
    return fail("the backend is not D3D12");
  auto base = dawn::native::d3d12::GetD3D12Device(webgpu::g_device.Get());
  g.queue = dawn::native::d3d12::GetD3D12CommandQueue(webgpu::g_device.Get());
  if (!base || !g.queue || FAILED(base.As(&g.dev)))
    return fail("no D3D12 device");
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
  RT_HR(g.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof o5));
  if (o5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1)
    return fail("the GPU has no DXR 1.1");
  RT_HR(g.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence)));
  g.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  D3D12_DESCRIPTOR_HEAP_DESC hd{};
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.NumDescriptors = Frames * SlotDescriptors + MaxTextures;
  hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  RT_HR(g.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)));
  g.inc = g.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  for (auto& s : g.slots) {
    RT_HR(g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.alloc)));
    RT_HR(g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.alloc.Get(), nullptr, IID_PPV_ARGS(&s.cl)));
    RT_HR(s.cl->Close());
  }
  D3D12_QUERY_HEAP_DESC qd{};
  qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  qd.Count = 4 * Frames;
  RT_HR(g.dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&g.queries)));
  g.queryResult = make_buffer(D3D12_HEAP_TYPE_READBACK, 8 * qd.Count, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  RT_HR(g.queue->GetTimestampFrequency(&g.gpuTicks));

  HMODULE dxc = LoadLibraryW(L"dxcompiler.dll");
  if (!dxc)
    return fail("no dxcompiler.dll");
  const auto create = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(dxc, "DxcCreateInstance"));
  ComPtr<IDxcCompiler3> comp;
  if (!create || FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&comp))))
    return fail("no shader compiler");
  auto traceCs = compile(comp.Get(), L"main"), composeCs = compile(comp.Get(), L"compose");
  if (!traceCs || !composeCs)
    return fail("the shader does not compile");

  D3D12_DESCRIPTOR_RANGE1 r1[2]{};
  r1[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  r1[0].NumDescriptors = 7;
  r1[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  r1[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  r1[1].NumDescriptors = 2;
  r1[1].OffsetInDescriptorsFromTableStart = 7;
  r1[1].Flags = r1[0].Flags;
  D3D12_DESCRIPTOR_RANGE1 r2{};
  r2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  r2.NumDescriptors = UINT_MAX;
  r2.RegisterSpace = 1;
  r2.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  D3D12_ROOT_PARAMETER1 rp[3]{};
  rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rp[0].Constants.Num32BitValues = sizeof(Params) / 4;
  rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[1].DescriptorTable.NumDescriptorRanges = 2;
  rp[1].DescriptorTable.pDescriptorRanges = r1;
  rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[2].DescriptorTable.NumDescriptorRanges = 1;
  rp[2].DescriptorTable.pDescriptorRanges = &r2;
  D3D12_STATIC_SAMPLER_DESC ss{};
  ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
  ss.MaxLOD = D3D12_FLOAT32_MAX;
  ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{};
  rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
  rs.Desc_1_1.NumParameters = 3;
  rs.Desc_1_1.pParameters = rp;
  rs.Desc_1_1.NumStaticSamplers = 1;
  rs.Desc_1_1.pStaticSamplers = &ss;
  ComPtr<ID3DBlob> rsb, rse;
  // the runtime's serializer, by name: Aurora's link has no d3d12.lib of its own to rely on here
  const auto ser = reinterpret_cast<PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE>(
      GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12SerializeVersionedRootSignature"));
  if (!ser)
    return fail("no root signature serializer");
  RT_HR(ser(&rs, &rsb, &rse));
  RT_HR(g.dev->CreateRootSignature(0, rsb->GetBufferPointer(), rsb->GetBufferSize(), IID_PPV_ARGS(&g.root)));
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = g.root.Get();
  pd.CS = {traceCs->GetBufferPointer(), traceCs->GetBufferSize()};
  RT_HR(g.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.trace)));
  pd.CS = {composeCs->GetBufferPointer(), composeCs->GetBufferSize()};
  RT_HR(g.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.compose)));
  g.ready = true;
  std::fprintf(stderr, "[rt] live pass ready on Dawn's device, %.0f ms\n",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

ComPtr<ID3D12Resource> shared_texture(u32 w, u32 h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags, wgpu::SharedTextureMemory& mem, wgpu::Texture& tex) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w;
  d.Height = h;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = fmt;
  d.SampleDesc.Count = 1;
  d.Flags = flags | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  ComPtr<ID3D12Resource> r;
  if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) {
    fail("no shared texture");
    return nullptr;
  }
  dawn::native::d3d12::SharedTextureMemoryD3D12ResourceDescriptor native;
  native.resource = r;
  wgpu::SharedTextureMemoryDescriptor sd{};
  sd.nextInChain = &native;
  mem = webgpu::g_device.ImportSharedTextureMemory(&sd);
  tex = mem ? mem.CreateTexture() : nullptr;
  if (!tex)
    fail("Dawn did not take the shared texture");
  return r;
}

void make_targets(u32 w, u32 h, wgpu::TextureFormat format) {
  if (w == g.w && h == g.h && g.input)
    return;
  // a new size: the old textures may still be in flight
  if (g.fenceValue != 0 && g.fence->GetCompletedValue() < g.fenceValue) {
    g.fence->SetEventOnCompletion(g.fenceValue, g.event);
    WaitForSingleObject(g.event, INFINITE);
  }
  g.w = w, g.h = h;
  const DXGI_FORMAT dxgi = format == wgpu::TextureFormat::BGRA8Unorm ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
  g.inputRes = shared_texture(w, h, dxgi, D3D12_RESOURCE_FLAG_NONE, g.inputMem, g.input);
  g.outputRes = shared_texture(w, h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, g.outputMem, g.output);
  if (g.failed)
    return;
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w;
  d.Height = h;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  d.SampleDesc.Count = 1;
  d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  RT_HR(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g.termRes)));
  const wgpu::BindGroupLayoutEntry e{.binding = 0,
                                     .visibility = wgpu::ShaderStage::Fragment,
                                     .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat}};
  if (!g.bgl) {
    const wgpu::BindGroupLayoutDescriptor ld{.entryCount = 1, .entries = &e};
    g.bgl = webgpu::g_device.CreateBindGroupLayout(&ld);
  }
  const wgpu::TextureView view = g.output.CreateView();
  const wgpu::BindGroupEntry be{.binding = 0, .textureView = view};
  const wgpu::BindGroupDescriptor bd{.layout = g.bgl, .entryCount = 1, .entries = &be};
  g.bind = webgpu::g_device.CreateBindGroup(&bd);
}

void ensure(ComPtr<ID3D12Resource>& res, u64& have, u64 need, D3D12_RESOURCE_STATES state) {
  if (have >= need && res)
    return;
  have = need + need / 4;
  res = make_buffer(D3D12_HEAP_TYPE_DEFAULT, have, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, state);
}
void stage(Slot& s, Slot::Buf& b, const void* data, u64 size) {
  size = std::max<u64>(size, 16);
  if (b.size < size) {
    b.size = size + size / 4;
    b.upload = make_buffer(D3D12_HEAP_TYPE_UPLOAD, b.size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    b.gpu = make_buffer(D3D12_HEAP_TYPE_DEFAULT, b.size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    b.map = nullptr;
    if (g.failed)
      return;
    b.upload->Map(0, nullptr, &b.map);
  }
  std::memcpy(b.map, data, size);
  s.cl->CopyBufferRegion(b.gpu.Get(), 0, b.upload.Get(), 0, size);
}
void srv_buffer(ID3D12Resource* r, u32 count, u32 stride, u32 at) {
  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  sv.Buffer.NumElements = std::max(count, 1u);
  sv.Buffer.StructureByteStride = stride;
  g.dev->CreateShaderResourceView(r, &sv, cpuh(at));
}
void upload_texture(Slot& s, const LiveTexture& t) {
  std::vector<std::vector<u8>> mips;
  mips.push_back(t.rgba);
  u32 w = t.w, h = t.h;
  while (w > 1 || h > 1) {
    const u32 nw = std::max(w / 2, 1u), nh = std::max(h / 2, 1u);
    std::vector<u8> m(size_t(nw) * nh * 4);
    const auto& a = mips.back();
    for (u32 y = 0; y < nh; ++y)
      for (u32 x = 0; x < nw; ++x)
        for (u32 c = 0; c < 4; ++c) {
          const u32 xa = std::min(x * 2, w - 1), ya = std::min(y * 2, h - 1), xb = std::min(x * 2 + 1, w - 1), yb = std::min(y * 2 + 1, h - 1);
          m[(size_t(y) * nw + x) * 4 + c] = static_cast<u8>((a[(size_t(ya) * w + xa) * 4 + c] + a[(size_t(ya) * w + xb) * 4 + c] +
                                                             a[(size_t(yb) * w + xa) * 4 + c] + a[(size_t(yb) * w + xb) * 4 + c] + 2) / 4);
        }
    mips.push_back(std::move(m));
    w = nw, h = nh;
  }
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = t.w;
  d.Height = t.h;
  d.DepthOrArraySize = 1;
  d.MipLevels = static_cast<UINT16>(mips.size());
  d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  d.SampleDesc.Count = 1;
  ComPtr<ID3D12Resource> r;
  RT_HR(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r)));
  std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(mips.size());
  UINT64 total = 0;
  g.dev->GetCopyableFootprints(&d, 0, static_cast<UINT>(mips.size()), 0, fp.data(), nullptr, nullptr, &total);
  auto u = make_buffer(D3D12_HEAP_TYPE_UPLOAD, total, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  if (g.failed)
    return;
  u8* m = nullptr;
  RT_HR(u->Map(0, nullptr, reinterpret_cast<void**>(&m)));
  for (u32 l = 0; l < mips.size(); ++l) {
    const u32 mw = fp[l].Footprint.Width, mh = fp[l].Footprint.Height;
    for (u32 y = 0; y < mh; ++y)
      std::memcpy(m + fp[l].Offset + size_t(y) * fp[l].Footprint.RowPitch, mips[l].data() + size_t(y) * mw * 4, size_t(mw) * 4);
    D3D12_TEXTURE_COPY_LOCATION dst{r.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    dst.SubresourceIndex = l;
    D3D12_TEXTURE_COPY_LOCATION src{u.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = fp[l];
    s.cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  }
  u->Unmap(0, nullptr);
  s.keep.push_back(u);
  D3D12_RESOURCE_BARRIER b{};
  b.Transition.pResource = r.Get();
  b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
  b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  s.cl->ResourceBarrier(1, &b);
  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Format = d.Format;
  sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.Texture2D.MipLevels = d.MipLevels;
  g.dev->CreateShaderResourceView(r.Get(), &sv, cpuh(Frames * SlotDescriptors + t.index));
  g.textureRes.push_back(r);
  g.textures = std::max(g.textures, t.index + 1);
}

void build_blas(Slot& s, u32 which, D3D12_GPU_VIRTUAL_ADDRESS addr, u32 count, u64& scratchAt) {
  D3D12_RAYTRACING_GEOMETRY_DESC geo{};
  geo.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
  geo.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
  geo.Triangles.VertexBuffer.StartAddress = addr;
  geo.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
  geo.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
  geo.Triangles.VertexCount = count;
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
  in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
  in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  in.NumDescs = 1;
  in.pGeometryDescs = &geo;
  in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
  g.dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &pre);
  ensure(s.blas[which], s.blasSize[which], pre.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{};
  bd.Inputs = in;
  bd.DestAccelerationStructureData = s.blas[which]->GetGPUVirtualAddress();
  bd.ScratchAccelerationStructureData = s.scratch->GetGPUVirtualAddress() + scratchAt;
  scratchAt += (pre.ScratchDataSizeInBytes + 255) & ~u64(255);
  s.cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
}

// What the rays do for one frame, on Dawn's queue, between the frame's two command buffers
void run_rays(Job& job) {
  const auto t0 = std::chrono::steady_clock::now();
  g.slot = (g.slot + 1) % Frames;
  Slot& s = g.slots[g.slot];
  if (g.fence->GetCompletedValue() < s.fence) {
    RT_HR(g.fence->SetEventOnCompletion(s.fence, g.event));
    WaitForSingleObject(g.event, INFINITE);
  }
  // this slot's last timings, now done
  if (s.fence != 0) {
    u64* ts = nullptr;
    const D3D12_RANGE range{8 * 4 * g.slot, 8 * 4 * (g.slot + 1)};
    if (SUCCEEDED(g.queryResult->Map(0, &range, reinterpret_cast<void**>(&ts)))) {
      g.msGpu += double(ts[4 * g.slot + 3] - ts[4 * g.slot]) * 1000.0 / g.gpuTicks;
      g.msTrace += double(ts[4 * g.slot + 2] - ts[4 * g.slot + 1]) * 1000.0 / g.gpuTicks;
      g.queryResult->Unmap(0, nullptr);
    }
  }
  s.keep.clear();
  RT_HR(s.alloc->Reset());
  RT_HR(s.cl->Reset(s.alloc.Get(), nullptr));
  ID3D12GraphicsCommandList4* cl = s.cl.Get();
  ID3D12DescriptorHeap* heaps[] = {g.heap.Get()};
  cl->SetDescriptorHeaps(1, heaps);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4 * g.slot);
  for (const auto& t : job.textures)
    upload_texture(s, t);
  auto& fr = job.frame;
  if (fr.lights.empty())
    fr.lights.push_back({{0, 0, 1e6f}, {0, 0, 0}});
  if (fr.stages.empty())
    fr.stages.push_back({});
  stage(s, s.verts, fr.verts.data(), fr.verts.size() * sizeof(Vertex));
  stage(s, s.draws, fr.draws.data(), fr.draws.size() * sizeof(GpuDraw));
  stage(s, s.stages, fr.stages.data(), fr.stages.size() * sizeof(StageRec));
  stage(s, s.lights, fr.lights.data(), fr.lights.size() * sizeof(EngineLight));
  if (g.failed)
    return;
  const u32 skyVerts = static_cast<u32>(fr.verts.size()) - fr.skyFirstVertex;
  ensure(s.scratch, s.scratchSize, 64u << 20, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  u64 scratchAt = 0;
  {
    D3D12_RESOURCE_BARRIER bs[1]{};
    bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bs[0].UAV.pResource = nullptr;
    cl->ResourceBarrier(1, bs);
  }
  build_blas(s, 0, s.verts.gpu->GetGPUVirtualAddress(), fr.skyFirstVertex, scratchAt);
  if (skyVerts)
    build_blas(s, 1, s.verts.gpu->GetGPUVirtualAddress() + u64(fr.skyFirstVertex) * sizeof(Vertex), skyVerts, scratchAt);
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  cl->ResourceBarrier(1, &uav);
  D3D12_RAYTRACING_INSTANCE_DESC inst[2]{};
  for (auto& i : inst)
    i.Transform[0][0] = i.Transform[1][1] = i.Transform[2][2] = 1;
  inst[0].InstanceMask = 1;
  inst[0].AccelerationStructure = s.blas[0]->GetGPUVirtualAddress();
  if (skyVerts) {
    inst[1].InstanceMask = 2;
    inst[1].AccelerationStructure = s.blas[1]->GetGPUVirtualAddress();
  }
  if (!s.instances)
    s.instances = make_buffer(D3D12_HEAP_TYPE_UPLOAD, sizeof inst, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  void* im = nullptr;
  RT_HR(s.instances->Map(0, nullptr, &im));
  std::memcpy(im, inst, sizeof inst);
  s.instances->Unmap(0, nullptr);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tin{};
  tin.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  tin.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  tin.NumDescs = skyVerts ? 2 : 1;
  tin.InstanceDescs = s.instances->GetGPUVirtualAddress();
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
  g.dev->GetRaytracingAccelerationStructurePrebuildInfo(&tin, &pre);
  ensure(s.tlas, s.tlasSize, pre.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{};
  bd.Inputs = tin;
  bd.DestAccelerationStructureData = s.tlas->GetGPUVirtualAddress();
  bd.ScratchAccelerationStructureData = s.scratch->GetGPUVirtualAddress() + scratchAt;
  cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
  cl->ResourceBarrier(1, &uav);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4 * g.slot + 1);

  const u32 base = g.slot * SlotDescriptors;
  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
  sv.RaytracingAccelerationStructure.Location = s.tlas->GetGPUVirtualAddress();
  g.dev->CreateShaderResourceView(nullptr, &sv, cpuh(base));
  srv_buffer(s.verts.gpu.Get(), static_cast<u32>(fr.verts.size()), sizeof(Vertex), base + 1);
  srv_buffer(s.draws.gpu.Get(), static_cast<u32>(fr.draws.size()), sizeof(GpuDraw), base + 2);
  srv_buffer(s.lights.gpu.Get(), static_cast<u32>(fr.lights.size()), sizeof(EngineLight), base + 4);
  srv_buffer(s.stages.gpu.Get(), static_cast<u32>(fr.stages.size()), sizeof(StageRec), base + 5);
  sv = {};
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sv.Format = g.inputRes->GetDesc().Format;
  sv.Texture2D.MipLevels = 1;
  g.dev->CreateShaderResourceView(g.inputRes.Get(), &sv, cpuh(base + 3));
  sv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  g.dev->CreateShaderResourceView(g.termRes.Get(), &sv, cpuh(base + 6));
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
  uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  uv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  g.dev->CreateUnorderedAccessView(g.termRes.Get(), nullptr, &uv, cpuh(base + 7));
  uv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  g.dev->CreateUnorderedAccessView(g.outputRes.Get(), nullptr, &uv, cpuh(base + 8));

  Params p{};
  p.size[0] = static_cast<float>(g.w), p.size[1] = static_cast<float>(g.h);
  p.rectWH[0] = 640.f, p.rectWH[1] = 528.f;  // the EFB as the snapshot holds it, all of it
  p.vpXY[0] = job.viewport[0], p.vpXY[1] = job.viewport[1], p.vpWH[0] = job.viewport[2], p.vpWH[1] = job.viewport[3];
  p.pScale[0] = job.proj[0], p.pScale[1] = job.proj[5], p.pOffset[0] = job.proj[2], p.pOffset[1] = job.proj[6];
  p.projZ[0] = job.proj[10], p.projZ[1] = job.proj[11];
  p.miss[0] = p.miss[1] = p.miss[2] = 0.f;
  p.spp = 1;
  p.mode = 8;
  p.gain = 1;
  p.roadShine = job.road;
  p.rough = 0.002f;
  p.frame = g.frame++;
  p.textures = g.textures;
  p.conserve = 0.7f;
  p.lightGain = job.light;
  p.lightCount = static_cast<u32>(fr.lights.size());
  p.lightRange = 4.f;
  p.bodyCalm = 1.f;
  p.skyFirst = fr.skyFirstVertex;
  cl->SetComputeRootSignature(g.root.Get());
  cl->SetComputeRoot32BitConstants(0, sizeof(Params) / 4, &p, 0);
  cl->SetComputeRootDescriptorTable(1, gpuh(base));
  cl->SetComputeRootDescriptorTable(2, gpuh(Frames * SlotDescriptors));
  cl->SetPipelineState(g.trace.Get());
  cl->Dispatch((g.w + 7) / 8, (g.h + 7) / 8, 1);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4 * g.slot + 2);
  uav.UAV.pResource = g.termRes.Get();
  cl->ResourceBarrier(1, &uav);
  D3D12_RESOURCE_BARRIER tb{};
  tb.Transition.pResource = g.termRes.Get();
  tb.Transition.Subresource = 0;
  tb.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  tb.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  cl->ResourceBarrier(1, &tb);
  cl->SetPipelineState(g.compose.Get());
  cl->Dispatch((g.w + 7) / 8, (g.h + 7) / 8, 1);
  std::swap(tb.Transition.StateBefore, tb.Transition.StateAfter);
  cl->ResourceBarrier(1, &tb);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4 * g.slot + 3);
  cl->ResolveQueryData(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4 * g.slot, 4, g.queryResult.Get(), 8 * 4 * g.slot);
  RT_HR(cl->Close());
  ID3D12CommandList* lists[] = {cl};
  g.queue->ExecuteCommandLists(1, lists);
  s.fence = ++g.fenceValue;
  RT_HR(g.queue->Signal(g.fence.Get(), s.fence));

  g.msComposeWait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g.msBuild += job.captureMs;
  g.msRules += job.rulesMs;
  g.msTranslate += job.translateUs.load() / 1000.0;
  g.tris += fr.verts.size() / 3;
  g.lightsSeen += fr.lights.size();
  g.lightsMost = std::max<u64>(g.lightsMost, fr.lights.size());
  static const u32 every = [] {
    const char* v = std::getenv("AURORA_RT_REPORT");
    return v != nullptr ? static_cast<u32>(std::atoi(v)) : 300u;
  }();
  if (++g.count == every) {
    const double n = g.count;
    std::fprintf(stderr, "[rt] per frame, %u frames at %ux%u: GPU build %.2f ms and rays+cleanup %.2f ms (all %.2f); CPU in the stream %.2f ms, translation beside it %.2f ms, rules %.2f ms, worker %.2f ms; %.0f triangles, lights %.1f (most %llu)\n",
                 g.count, g.w, g.h, (g.msGpu - g.msTrace) / n, g.msTrace / n, g.msGpu / n, g.msBuild / n, g.msTranslate / n, g.msRules / n, g.msComposeWait / n, double(g.tris) / n, double(g.lightsSeen) / n,
                 static_cast<unsigned long long>(g.lightsMost));
    g.count = 0, g.msGpu = g.msTrace = g.msBuild = g.msRules = g.msTranslate = g.msComposeWait = 0, g.tris = 0, g.lightsSeen = 0, g.lightsMost = 0;
  }
}

void begin_access(wgpu::SharedTextureMemory& mem, const wgpu::Texture& tex) {
  wgpu::SharedTextureMemoryBeginAccessDescriptor d{};
  d.concurrentRead = false;
  d.initialized = true;
  d.fenceCount = 0;
  if (mem.BeginAccess(tex, &d) != wgpu::Status::Success)
    fail("Dawn refused the shared texture");
}
void end_access(wgpu::SharedTextureMemory& mem, const wgpu::Texture& tex) {
  wgpu::SharedTextureMemoryEndAccessState st{};
  mem.EndAccess(tex, &st);
}

struct TaskPayload {
  Job* job;
  WGPUTexture color;
  u32 w, h, format;
};

void task(const gfx::EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t, void*) {
  TaskPayload p;
  std::memcpy(&p, payload, sizeof p);
  std::unique_ptr<Job> job{p.job};
  const wgpu::Texture color = wgpu::Texture::Acquire(p.color);
  job->rules.wait();
  g.drawThis = job->drawable;
  if (!g.drawThis) {
    give_back(std::move(job));
    return;
  }
  if (!g.ready && !g.failed)
    init_device();
  if (g.failed)
    return;
  make_targets(p.w, p.h, static_cast<wgpu::TextureFormat>(p.format));
  if (g.failed)
    return;
  const wgpu::TexelCopyTextureInfo src{.texture = color}, dst{.texture = g.input};
  const wgpu::Extent3D size{.width = p.w, .height = p.h, .depthOrArrayLayers = 1};
  cmd.CopyTextureToTexture(&src, &dst, &size);
  g.first = ctx.frameEncoder->Finish();
  *ctx.frameEncoder = webgpu::g_device.CreateCommandEncoder();
  g.pending = std::move(job);
}

void draw(const gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void*, size_t, void*) {
  if (g.failed || !g.bind || !g.drawThis)
    return;
  if (!g.pipeline || g.pipelineKey != ctx.layout.key) {
    static const char* source = R"(
@group(0) @binding(0) var pic: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
}
@fragment fn fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(pic, vec2i(at.xy), 0).rgb, 1.0);
}
@fragment fn fs2(@builtin(position) at: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(pic, vec2i(at.xy), 0).rgb, 1.0);
}
)";
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = source;
    const wgpu::ShaderModuleDescriptor md{.nextInChain = &wgsl};
    const wgpu::ShaderModule module = webgpu::g_device.CreateShaderModule(&md);
    std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> targets{};
    for (u32 i = 0; i < ctx.layout.colorAttachmentCount; ++i) {
      targets[i].format = ctx.layout.colorAttachments[i].format;
      targets[i].writeMask = i == 0 ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None;
    }
    const wgpu::FragmentState fs{.module = module, .entryPoint = "fs", .targetCount = ctx.layout.colorAttachmentCount, .targets = targets.data()};
    wgpu::DepthStencilState ds{};
    ds.format = ctx.layout.depthStencilFormat;
    ds.depthWriteEnabled = wgpu::OptionalBool::False;
    ds.depthCompare = wgpu::CompareFunction::Always;
    const wgpu::PipelineLayoutDescriptor pl{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g.bgl};
    wgpu::RenderPipelineDescriptor rp{};
    rp.layout = webgpu::g_device.CreatePipelineLayout(&pl);
    rp.vertex.module = module;
    rp.vertex.entryPoint = "vs";
    rp.fragment = &fs;
    rp.depthStencil = ctx.layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &ds : nullptr;
    rp.multisample.count = ctx.layout.sampleCount;
    g.pipeline = webgpu::g_device.CreateRenderPipeline(&rp);
    g.pipelineKey = ctx.layout.key;
  }
  pass.SetPipeline(g.pipeline);
  pass.SetBindGroup(0, g.bind);
  pass.SetViewport(0.f, 0.f, static_cast<float>(g.w), static_cast<float>(g.h), 0.f, 1.f);
  pass.SetScissorRect(0, 0, g.w, g.h);
  pass.Draw(3);
}
} // namespace

void before_frame_submit();
void after_frame_submit();

std::unique_ptr<Job> take_job() {
  std::lock_guard l{s_poolLock};
  if (s_pool.empty())
    return std::make_unique<Job>();
  auto job = std::move(s_pool.back());
  s_pool.pop_back();
  return job;
}

bool submit_job(std::unique_ptr<Job> job) {
  if (g.failed) {
    job->rules.wait();
    give_back(std::move(job));
    return false;
  }
  if (g.task == gfx::InvalidEncoderTask) {
    g.task = gfx::register_encoder_task_type({.label = "ray tracing", .callback = task});
    g.draw = gfx::register_draw_type({.label = "ray traced picture", .draw = draw});
    gfx::g_beforeFrameSubmit = before_frame_submit;
    gfx::g_afterFrameSubmit = after_frame_submit;
  }
  // from inside the command stream: these calls must not drain it
  const gx::fifo::HoldDrain hold;
  gfx::ResolvedTargets targets;
  if (!gfx::resolve_pass({.color = true}, targets) || !targets.colorTexture) {
    job->rules.wait();
    give_back(std::move(job));
    return false;
  }
  TaskPayload p{job.release(), targets.colorTexture.MoveToCHandle(), targets.width, targets.height, static_cast<u32>(targets.colorFormat)};
  gfx::push_encoder_task(g.task, &p, sizeof p);
  gfx::push_custom_draw(g.draw, nullptr, 0);
  return true;
}

void before_frame_submit() {
  if (!g.pending || g.failed)
    return;
  begin_access(g.inputMem, g.input);
  webgpu::g_queue.Submit(1, &g.first);
  g.first = {};
  end_access(g.inputMem, g.input);
  run_rays(*g.pending);
  give_back(std::move(g.pending));
  begin_access(g.outputMem, g.output);
  g.outputOpen = true;
}
void after_frame_submit() {
  if (g.outputOpen) {
    end_access(g.outputMem, g.output);
    g.outputOpen = false;
  }
}
bool failed() noexcept { return g.failed; }
} // namespace aurora::gx::rt::live
