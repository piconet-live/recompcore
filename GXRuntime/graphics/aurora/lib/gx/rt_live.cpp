// The live pass (AURORA_RT=1): the rules and the shader are API-neutral (rt_rules.hpp, rt.hlsl); this file holds what is D3D12:
// inline ray queries on Dawn's own device and queue. A frame's encoder is cut in two where the HUD begins. The first part, the
// scene, is submitted, the rays run on the same queue, then the second part draws their picture back into the EFB and goes on
// with the HUD. Presentation only: nothing here reads or writes anything the game can see.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../gfx/stagelog.hpp"
#include "rt_live.hpp"
#include "rt_capture.hpp"

#include "rt_format.hpp"
#include "rt_rules.hpp"

#include <aurora/gfx.hpp>
#include <dawn/native/D3D12Backend.h>
#include <dxcapi.h>
#include <webgpu/webgpu_cpp.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../webgpu/gpu.hpp"
#include "../webgpu/hdr.hpp"
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
  u32 skyFirst, glowCount;
  float glowGain;
  u32 glowShow;
  float wet;
  // AURORA_RT_LIGHTING=1: the surface's light from rays (rt.hlsl lighting()). rays 0: off
  u32 lightRays;
  float skyGain, lightReach, bounce, visFloor;
  u32 stableShine;  // bit 0: AURORA_RT_SHINE_STABLE=1; bit 1: t13 holds the picture from before the last blends; bit 2: AURORA_RT_PAIR_GAP=1; bit 3: AURORA_RT_LAYER_PASS=1. The root signature has no room for more words
};

constexpr u32 Frames = 3, SlotDescriptors = 13, MaxTextures = 8192, PostLevels = 6;
// As rt.hlsl's PostParams
struct PostParams {
  float texel[2], range, bloom, dt, grade;
  u32 rayed, autoExposure, frame, levels;
};
// The look's descriptors, after the textures': for good until the size changes. Each texture a read view, then a write view
constexpr u32 BetweenDescriptors = 3;  // the warped term's two views, the true term's read
constexpr u32 PostDescriptors = 2 * (1 + PostLevels + PostLevels - 1) + 1 + BetweenDescriptors;
constexpr u32 PostFirst = Frames * SlotDescriptors + MaxTextures;
// As rt.hlsl's DenoiseParams, one for each step of a frame
struct DenoiseParams {
  float prevScale[2], prevOffset[2], prevVpXY[2], prevVpWH[2];
  float cam[3][4];
  float alphaLeast, phiNormal, phiDepth, phiLight;
  u32 step, reset, denoised, show;
  float litScale;  // full pixels to a light pixel: 2 with AURORA_RT_HALF=1
  u32 half;
};
constexpr u32 DenoiseSteps = 8, DenoiseViews = 13, DenoiseIterations = 4;
constexpr u32 ConstantStride = 256;
constexpr u32 ViewBetweenTrue = PostFirst + PostDescriptors - BetweenDescriptors, ViewBetween = ViewBetweenTrue + 1;
constexpr u32 ViewHdr = PostFirst, ViewDown = ViewHdr + 2, ViewUp = ViewDown + 2 * PostLevels, ViewExpo = ViewUp + 2 * (PostLevels - 1);
// The glow's two quarter-size textures (AURORA_RT_FLARE=1), each a read view then a write view, after the look's
constexpr u32 FlareDescriptors = 4, ViewFlare = PostFirst + PostDescriptors;

struct Slot {
  ComPtr<ID3D12CommandAllocator> alloc;
  ComPtr<ID3D12GraphicsCommandList4> cl;
  u64 fence = 0;
  struct Buf {
    ComPtr<ID3D12Resource> upload, gpu;
    u64 size = 0;
    void* map = nullptr;
  } verts, draws, stages, lights, glows, worldMeshes;
  ComPtr<ID3D12Resource> blas[6], tlas, scratch, instances, constants, attrUpload;
  u64 attrUploadSize = 0;
  void* attrUploadMap = nullptr;
  void* constantsMap = nullptr;
  Buf prevPos;
  u64 blasSize[6]{}, tlasSize = 0, scratchSize = 0;
  std::vector<ComPtr<ID3D12Resource>> keep;
  // what this slot's timings belong to, for the time log
  u64 copy = 0;
  u32 tris = 0, lightCount = 0, uploads = 0;
  bool between = false;  // its timings are a picture between's
  ComPtr<ID3D12Resource> betweenConstants;
  void* betweenMap = nullptr;
};
// rt.hlsl's BetweenParams
struct BetweenMachine {
  float m[3][4];
  float at[4];
};
struct BetweenParams {
  float size[2], rectXY[2], rectWH[2], vpXY[2], vpWH[2], pScale[2], pOffset[2];
  u32 count, iterations;
  float radius2, far_, pad_[2];
  float camera[3][4];
  BetweenMachine machine[30];
};
struct State {
  // ready: the compile thread has finished, everything below it is the render worker's from then on
  std::atomic<bool> failed{false}, ready{false};
  ComPtr<ID3D12Device5> dev;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  u64 fenceValue = 0;
  ComPtr<ID3D12DescriptorHeap> heap;
  UINT inc = 0;
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3D12PipelineState> trace, compose, lightPass, temporal, atrous, betweenWarp, betweenHdr, betweenCompose;
  // The glow (AURORA_RT_FLARE=1): the light above white at a quarter of the size, then its arms and halo; compose adds them
  ComPtr<ID3D12PipelineState> flareSrc, flareStar;
  ComPtr<ID3D12Resource> flareRes[2];
  u32 flareW = 0, flareH = 0;
  // pictures between: the true frame's colour texture and view for its rays; this picture's corrections; what they cost
  wgpu::Texture lastColor;
  bool haveTerm = false, pendingBetween = false;
  float lastView[10]{};  // viewport's and projection's numbers of the true frame: rect wh, vp xy, vp wh, scale, offset
  BetweenParams betweenParams{};
  ComPtr<ID3D12Resource> termBetween;
  u32 rw = 0, rh = 0;  // the rays' size
  double msBetweenGpu = 0, msBetweenCpu = 0;
  u64 betweenCount = 0, betweenGpuCount = 0;
  // Light across frames (AURORA_RT_DENOISE=1): textures for this frame and the last; a texture with its descriptors written per step
  ComPtr<ID3D12Resource> refl, guide[2], motion, noisyOpen, noisySky, histOpen[2], histSky[2], moments[2], tmpOpen[2], tmpSky[2];
  u32 flip = 0;
  float prevProj[16]{}, prevViewport[4]{};
  u64 lastLitCopy = 0;
  bool historyGood = false;
  double msLight = 0, msDenoise = 0, motionAgree = 0, motionOwn = 0;
  // The picture's look (AURORA_LOOK_POST=1): high range, bloom chain down and up, exposure, tone curve
  ComPtr<ID3D12PipelineState> hdrIn, bloomDown, bloomUp, exposure, tone;
  ComPtr<ID3D12Resource> hdrRes, downRes[PostLevels], upRes[PostLevels - 1], expoRes;
  u32 levelW[PostLevels]{}, levelH[PostLevels]{};
  std::chrono::steady_clock::time_point lastPost{};
  ComPtr<ID3D12QueryHeap> queries;
  ComPtr<ID3D12Resource> queryResult;
  u64 gpuTicks = 0;
  Slot slots[Frames];
  u32 slot = 0, frame = 0, textures = 0;
  std::vector<ComPtr<ID3D12Resource>> textureRes;

  u32 w = 0, h = 0;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  ComPtr<ID3D12Resource> inputRes, outputRes, termRes, beforeRes;
  wgpu::SharedTextureMemory inputMem, outputMem, beforeMem;
  wgpu::Texture input, output, before;
  bool beforeThis = false;
  gfx::EncoderTaskId beforeTask = gfx::InvalidEncoderTask;
  wgpu::BindGroup bind;
  wgpu::RenderPipeline pipeline;
  u64 pipelineKey = 0;
  wgpu::BindGroupLayout bgl;
  gfx::EncoderTaskId task = gfx::InvalidEncoderTask;
  gfx::DrawTypeId draw = gfx::InvalidDrawType;

  // The whole track (AURORA_RT_SCENE=1): one structure for each model, made once
  std::shared_ptr<scene::Scene> worldScene;
  ComPtr<ID3D12Resource> worldBlas, worldAttrs;
  double worldSkipped = 0, worldDropped = 0;
  std::vector<u64> worldAt;
  u64 worldMergedAt[2]{};
  bool worldMergedOn[2]{};
  u64 worldBytes = 0;
  double worldKnown = 0, worldDrawn = 0, worldInstances = 0, worldDraws = 0, worldMatched = 0, worldMoved = 0, worldMs = 0, worldPartial = 0;
  std::unique_ptr<Job> pending;
  bool drawThis = false;
  wgpu::CommandBuffer first;
  bool outputOpen = false;

  double msTrace = 0, msGpu = 0, msComposeWait = 0, msBuild = 0, msRules = 0, msTranslate = 0;
  double picks = 0, merged = 0, glowsSeen = 0, glowPieces = 0, glowDraws = 0, glowTop = 0;
  double msGpuMost = 0;
  u64 tris = 0, lightsSeen = 0, lightsMost = 0, reports = 0;
  FILE* timeLog = nullptr;
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

// AURORA_RT_SHADER=<file>: the shader read from there instead of the build's own copy, to try a change without a build
std::string s_shaderFile;
// AURORA_RT_FLARE=1, AURORA_RT_HOT_CORE=1 (off by default): the glow (rt.hlsl, above flare_src). Its numbers, AURORA_RT_FLARE_<NAME> and
// AURORA_RT_HOT_CORE_HALF, go to the shader as it compiles (AURORA_RT_GLOW_GAIN: the owner's brightness for both, 1 as the game): no room is left in the root signature's words.
bool env_one(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1';
}
bool flare_on() {
  static const bool on = env_one("AURORA_RT_FLARE");
  return on;
}
bool hot_core_on() {
  static const bool on = env_one("AURORA_RT_HOT_CORE");
  return on;
}
const std::vector<std::wstring>& shader_defines() {
  static const std::vector<std::wstring> defines = [] {
    // KEEP_RANGE: the rays' picture keeps what is above white, for the float display copy (hdr.hpp); otherwise cut as ever
    std::vector<std::wstring> d{webgpu::hdr::float_copy() ? L"KEEP_RANGE=1" : L"KEEP_RANGE=0", flare_on() ? L"FLARE=1" : L"FLARE=0",
                                hot_core_on() ? L"HOT_CORE=1" : L"HOT_CORE=0",
                                // AURORA_RT_GATE_LEVEL=1 (off by default): the glow gate reads the surface at the raster's texture level (rt.hlsl GATE_LEVEL)
                                env_one("AURORA_RT_GATE_LEVEL") ? L"GATE_LEVEL=1" : L"GATE_LEVEL=0"};
    // tuned by eye; what each means is at its use in rt.hlsl
    struct Number {
      const char* env;
      const wchar_t* name;
      float otherwise;
    };
    static constexpr Number numbers[] = {{"AURORA_RT_FLARE_KNEE", L"FLARE_KNEE", 1.f},   {"AURORA_RT_FLARE_REACH", L"FLARE_REACH", 0.03f},
                                         {"AURORA_RT_FLARE_HALF", L"FLARE_HALF", 1.f},   {"AURORA_RT_FLARE_ARMS", L"FLARE_ARMS", 0.5f},
                                         {"AURORA_RT_FLARE_HALO", L"FLARE_HALO", 0.6f},  {"AURORA_RT_FLARE_SAT", L"FLARE_SAT", 1.3f},
                                         {"AURORA_RT_FLARE_GAIN", L"FLARE_GAIN", 1.f},   {"AURORA_RT_FLARE_TAPS", L"FLARE_TAPS", 24.f},
                                         {"AURORA_RT_FLARE_SHOW", L"FLARE_SHOW", 0.f},   {"AURORA_RT_HOT_CORE_HALF", L"HOT_CORE_HALF", 1.f},
                                         {"AURORA_RT_GLOW_GAIN", L"GLOW_GAIN", 1.f},
                                         // the glow gate compares with the darkest raster pixel within this many (0: the one pixel; rt.hlsl GATE_MIN)
                                         {"AURORA_RT_GATE_MIN", L"GATE_MIN", 0.f}};
    for (const auto& n : numbers) {
      const char* v = std::getenv(n.env);
      const float f = v != nullptr && v[0] != '\0' ? static_cast<float>(std::atof(v)) : n.otherwise;
      d.push_back(std::wstring(n.name) + L"=" + std::to_wstring(f));
    }
    return d;
  }();
  return defines;
}

ComPtr<IDxcBlob> compile(const DxcCreateInstanceProc create, const wchar_t* entry) {
  // a compiler for each entry: they run side by side, and one compiler is not for two threads. Below the game's threads.
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
  ComPtr<IDxcCompiler3> comp;
  if (FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&comp))))
    return nullptr;
  DxcBuffer src{kShader, sizeof kShader, DXC_CP_UTF8};
  if (!s_shaderFile.empty())
    src.Ptr = s_shaderFile.data(), src.Size = s_shaderFile.size();
  std::vector<LPCWSTR> args{L"-T", L"cs_6_5", L"-E", entry, L"-O3", L"-HV", L"2021"};
  for (const auto& d : shader_defines())
    args.push_back(L"-D"), args.push_back(d.c_str());
  ComPtr<IDxcResult> res;
  if (FAILED(comp->Compile(&src, args.data(), static_cast<UINT32>(args.size()), nullptr, IID_PPV_ARGS(&res))))
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

// On its own thread, once, from the start: the compile takes seconds, and nothing else waits for it. The frames show the game's own picture until g.ready.
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
  hd.NumDescriptors = Frames * SlotDescriptors + MaxTextures + PostDescriptors + FlareDescriptors + Frames * DenoiseSteps * DenoiseViews;
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
  qd.Count = 6 * Frames;
  RT_HR(g.dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&g.queries)));
  g.queryResult = make_buffer(D3D12_HEAP_TYPE_READBACK, 8 * qd.Count, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  RT_HR(g.queue->GetTimestampFrequency(&g.gpuTicks));

  HMODULE dxc = LoadLibraryW(L"dxcompiler.dll");
  if (!dxc)
    return fail("no dxcompiler.dll");
  const auto create = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(dxc, "DxcCreateInstance"));
  if (!create)
    return fail("no shader compiler");
  if (const char* path = std::getenv("AURORA_RT_SHADER"))
    if (FILE* f = std::fopen(path, "rb")) {
      char buf[65536];
      for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
        s_shaderFile.append(buf, n);
      std::fclose(f);
    }
  const auto start = [&](const wchar_t* entry) { return std::async(std::launch::async, compile, create, entry); };
  auto traceF = start(L"main"), composeF = start(L"compose"), hdrF = start(L"hdr_in"), downF = start(L"bloom_down"), upF = start(L"bloom_up"), expoF = start(L"exposure"), toneF = start(L"tone"),
       warpF = start(L"between_warp"), betweenHdrF = start(L"between_hdr"), betweenComposeF = start(L"between_compose"),
       lightF = start(L"light_pass"), temporalF = start(L"temporal"), atrousF = start(L"atrous");
  // the glow's passes only when it is on: a shader file from before them still serves without it
  std::future<ComPtr<IDxcBlob>> flareSrcF, flareStarF;
  if (flare_on())
    flareSrcF = start(L"flare_src"), flareStarF = start(L"flare_star");
  auto traceCs = traceF.get(), composeCs = composeF.get(), hdrCs = hdrF.get(), downCs = downF.get(), upCs = upF.get(), expoCs = expoF.get(), toneCs = toneF.get(), lightCs = lightF.get(), temporalCs = temporalF.get(), atrousCs = atrousF.get();
  auto warpCs = warpF.get(), betweenHdrCs = betweenHdrF.get(), betweenComposeCs = betweenComposeF.get();
  if (!warpCs || !betweenHdrCs || !betweenComposeCs || !lightCs || !temporalCs || !atrousCs ||
      !traceCs || !composeCs || !hdrCs || !downCs || !upCs || !expoCs || !toneCs)
    return fail("the shader does not compile");

  D3D12_DESCRIPTOR_RANGE1 r1[3]{};
  r1[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  r1[0].NumDescriptors = 8;
  r1[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  r1[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  r1[1].NumDescriptors = 2;
  r1[1].OffsetInDescriptorsFromTableStart = 8;
  r1[1].Flags = r1[0].Flags;
  r1[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // t11, t12: the whole track's meshes and their attributes (rt_scene_live.inl)
  r1[2].NumDescriptors = 3;
  r1[2].BaseShaderRegister = 11;
  r1[2].OffsetInDescriptorsFromTableStart = 10;
  r1[2].Flags = r1[0].Flags;
  D3D12_DESCRIPTOR_RANGE1 rd[2]{};
  rd[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  rd[0].NumDescriptors = 1;
  rd[0].BaseShaderRegister = 10;
  rd[0].Flags = r1[0].Flags;
  rd[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  rd[1].NumDescriptors = DenoiseViews - 1;
  rd[1].BaseShaderRegister = 4;
  rd[1].OffsetInDescriptorsFromTableStart = 1;
  rd[1].Flags = r1[0].Flags;
  D3D12_DESCRIPTOR_RANGE1 r2{};
  r2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  r2.NumDescriptors = UINT_MAX;
  r2.RegisterSpace = 1;
  r2.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
  D3D12_DESCRIPTOR_RANGE1 one[4]{};
  for (u32 i = 0; i < 4; ++i) {
    one[i].RangeType = i < 2 ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    one[i].NumDescriptors = 1;
    one[i].BaseShaderRegister = i < 2 ? 8 + i : i;  // t8, t9, u2, u3
    one[i].Flags = r1[0].Flags;
  }
  D3D12_ROOT_PARAMETER1 rp[10]{};
  rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rp[0].Constants.Num32BitValues = sizeof(Params) / 4;
  rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[1].DescriptorTable.NumDescriptorRanges = 3;
  rp[1].DescriptorTable.pDescriptorRanges = r1;
  rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[2].DescriptorTable.NumDescriptorRanges = 1;
  rp[2].DescriptorTable.pDescriptorRanges = &r2;
  rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rp[3].Constants.ShaderRegister = 1;
  rp[3].Constants.Num32BitValues = sizeof(PostParams) / 4;
  for (u32 i = 0; i < 4; ++i) {
    rp[4 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[4 + i].DescriptorTable.NumDescriptorRanges = 1;
    rp[4 + i].DescriptorTable.pDescriptorRanges = &one[i];
  }
  rp[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[8].DescriptorTable.NumDescriptorRanges = 2;
  rp[8].DescriptorTable.pDescriptorRanges = rd;
  rp[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
  rp[9].Descriptor.ShaderRegister = 2;
  rp[9].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
  D3D12_STATIC_SAMPLER_DESC ss[2]{};
  ss[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  ss[0].AddressU = ss[0].AddressV = ss[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
  ss[0].MaxLOD = D3D12_FLOAT32_MAX;
  ss[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  ss[1] = ss[0];
  ss[1].ShaderRegister = 1;
  ss[1].AddressU = ss[1].AddressV = ss[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{};
  rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
  rs.Desc_1_1.NumParameters = 10;
  rs.Desc_1_1.pParameters = rp;
  rs.Desc_1_1.NumStaticSamplers = 2;
  rs.Desc_1_1.pStaticSamplers = ss;
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
  const std::pair<ComPtr<ID3D12PipelineState>*, IDxcBlob*> posts[] = {{&g.hdrIn, hdrCs.Get()}, {&g.bloomDown, downCs.Get()}, {&g.bloomUp, upCs.Get()}, {&g.exposure, expoCs.Get()}, {&g.tone, toneCs.Get()}, {&g.lightPass, lightCs.Get()}, {&g.temporal, temporalCs.Get()}, {&g.atrous, atrousCs.Get()}, {&g.betweenWarp, warpCs.Get()}, {&g.betweenHdr, betweenHdrCs.Get()}, {&g.betweenCompose, betweenComposeCs.Get()}};
  for (const auto& [pso, cs] : posts) {
    pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    RT_HR(g.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(pso->GetAddressOf())));
  }
  if (flare_on()) {
    const auto flareSrcCs = flareSrcF.get(), flareStarCs = flareStarF.get();
    if (!flareSrcCs || !flareStarCs)
      return fail("the shader's glow does not compile");
    pd.CS = {flareSrcCs->GetBufferPointer(), flareSrcCs->GetBufferSize()};
    RT_HR(g.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.flareSrc)));
    pd.CS = {flareStarCs->GetBufferPointer(), flareStarCs->GetBufferSize()};
    RT_HR(g.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.flareStar)));
  }
  if (const char* path = std::getenv("AURORA_RT_TIMELOG")) {
    g.timeLog = std::fopen(path, "wb");
    if (g.timeLog != nullptr) {
      std::setvbuf(g.timeLog, nullptr, _IOLBF, 4096);
      std::fprintf(g.timeLog, "copy,build_ms,light_ms,denoise_ms,reflect_ms,compose_ms,tris,lights,uploads\n");
    }
  }
  std::fprintf(stderr, "[rt] live pass ready on Dawn's device, %.0f ms, beside the game\n",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  g.ready = true;
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

bool before_setting() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_BEFORE_BLENDS");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
// AURORA_RT_RULES_LATE=1 (off by default): the render worker waits for a frame's rules just before its rays run
// (before_frame_submit), not at the ray task in the stream, so the rest of the frame is encoded while they finish. A frame the
// rules then find has nothing for the rays shows the look alone over the game's picture, as a look-only frame does.
bool rules_late() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_RULES_LATE");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
bool denoise_on() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_DENOISE");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
bool half_on() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_RT_HALF");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
bool post_on() {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_LOOK_POST");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
float post_setting(const char* name, float otherwise) {
  const char* v = std::getenv(name);
  return v != nullptr ? static_cast<float>(std::atof(v)) : otherwise;
}
ComPtr<ID3D12Resource> post_texture(u32 w, u32 h, DXGI_FORMAT fmt, u32 at) {
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
  d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> r;
  if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&r)))) {
    fail("no texture for the look");
    return nullptr;
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Format = fmt;
  sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.Texture2D.MipLevels = 1;
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
  uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  uv.Format = fmt;
  if (fmt == DXGI_FORMAT_R32_FLOAT) {
    g.dev->CreateUnorderedAccessView(r.Get(), nullptr, &uv, cpuh(at));
    return r;
  }
  g.dev->CreateShaderResourceView(r.Get(), &sv, cpuh(at));
  g.dev->CreateUnorderedAccessView(r.Get(), nullptr, &uv, cpuh(at + 1));
  return r;
}
void make_post_targets(u32 w, u32 h) {
  const DXGI_FORMAT f = DXGI_FORMAT_R16G16B16A16_FLOAT;
  g.hdrRes = post_texture(w, h, f, ViewHdr);
  for (u32 i = 0; i < PostLevels; ++i) {
    w = std::max(w / 2, 1u), h = std::max(h / 2, 1u);
    g.levelW[i] = w, g.levelH[i] = h;
    g.downRes[i] = post_texture(w, h, f, ViewDown + 2 * i);
    if (i + 1 < PostLevels)
      g.upRes[i] = post_texture(w, h, f, ViewUp + 2 * i);
  }
  if (!g.expoRes)
    g.expoRes = post_texture(1, 1, DXGI_FORMAT_R32_FLOAT, ViewExpo);
}

DXGI_FORMAT output_format() {
  return webgpu::hdr::float_copy() ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
}
void make_targets(u32 w, u32 h, wgpu::TextureFormat format) {
  if (w == g.w && h == g.h && format == g.format && g.input)
    return;
  // a new size: the old textures may still be in flight
  if (g.fenceValue != 0 && g.fence->GetCompletedValue() < g.fenceValue) {
    g.fence->SetEventOnCompletion(g.fenceValue, g.event);
    WaitForSingleObject(g.event, INFINITE);
  }
  g.w = w, g.h = h, g.format = format;
  g.rw = w, g.rh = h;
  const DXGI_FORMAT dxgi = format == wgpu::TextureFormat::RGBA16Float ? DXGI_FORMAT_R16G16B16A16_FLOAT : format == wgpu::TextureFormat::BGRA8Unorm ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
  g.inputRes = shared_texture(w, h, dxgi, D3D12_RESOURCE_FLAG_NONE, g.inputMem, g.input);
  g.outputRes = shared_texture(w, h, output_format(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, g.outputMem, g.output);
  if (before_setting())
    g.beforeRes = shared_texture(w, h, dxgi, D3D12_RESOURCE_FLAG_NONE, g.beforeMem, g.before);
  if (g.failed)
    return;
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = g.rw;
  d.Height = g.rh;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  d.SampleDesc.Count = 1;
  d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  RT_HR(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g.termRes)));
  if (denoise_on()) {
    const auto make = [&](ComPtr<ID3D12Resource>& r, DXGI_FORMAT fmt) {
      d.Format = fmt;
      if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&r))))
        fail("no texture for light across frames");
    };
    constexpr DXGI_FORMAT F16 = DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (half_on())
      d.Width = (w + 1) / 2, d.Height = (h + 1) / 2;
    make(g.refl, F16);
    make(g.motion, F16), make(g.noisyOpen, F16), make(g.noisySky, F16);
    for (u32 i = 0; i < 2; ++i) {
      make(g.guide[i], DXGI_FORMAT_R32G32B32A32_FLOAT);
      make(g.histOpen[i], F16), make(g.histSky[i], F16), make(g.moments[i], F16), make(g.tmpOpen[i], F16), make(g.tmpSky[i], F16);
    }
    g.historyGood = false;
  }
  if (post_on())
    make_post_targets(w, h);
  if (flare_on()) {
    g.flareW = (w + 3) / 4, g.flareH = (h + 3) / 4;
    for (u32 i = 0; i < 2; ++i)
      g.flareRes[i] = post_texture(g.flareW, g.flareH, DXGI_FORMAT_R16G16B16A16_FLOAT, ViewFlare + 2 * i);
  }
  if (redraw::g_on) {
    g.termBetween = post_texture(g.rw, g.rh, DXGI_FORMAT_R16G16B16A16_FLOAT, ViewBetween);
    if (g.termRes) {
      D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
      sv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sv.Texture2D.MipLevels = 1;
      g.dev->CreateShaderResourceView(g.termRes.Get(), &sv, cpuh(ViewBetweenTrue));
    }
  }
  if (g.failed)
    return;
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
  const auto& mips = t.mips;
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

#include "rt_scene_live.inl"

// The game's sun, d in view space as the frame's draws asked for it. AURORA_RT_SUN_FIXED=1 (off by default): the sun stays fixed in the
// world for the whole course. Its world direction is the game's own, taken through this frame's camera (the static track's, rt_scene)
// over the first frames of the course that have both, averaged; every frame after gets that direction back through its own camera.
// A course whose game light is already fixed in the world is unchanged. Without the camera the frame keeps the game's direction.
// AURORA_RT_SUN_LOG=1: a line a frame on stderr, the game's direction in view and world space, the camera's heading, what the rays get.
void world_sun(const Job& job, float d[3]) {
  static const bool on = post_setting("AURORA_RT_SUN_FIXED", 0.f) != 0.f, log = std::getenv("AURORA_RT_SUN_LOG") != nullptr;
  static constexpr u32 TakeFrames = 8;  // the first frames with a camera that the course's direction is the average of
  static const scene::Scene* course = nullptr;
  static float sum[3]{};
  static u32 taken = 0;
  if (!on && !log)
    return;
  const bool known = job.world.known && job.world.scene != nullptr;
  if (known && job.world.scene.get() != course)
    course = job.world.scene.get(), taken = 0, sum[0] = sum[1] = sum[2] = 0.f;
  const float* v = job.world.view;  // world to view, rows of 4
  const auto normal = [](float* a) {
    const float len = std::max(std::hypot(a[0], a[1], a[2]), 1e-20f);
    for (int i = 0; i < 3; ++i)
      a[i] /= len;
  };
  float w[3]{}, out[3] = {d[0], d[1], d[2]};
  if (known) {
    for (int c = 0; c < 3; ++c)
      w[c] = v[c] * d[0] + v[4 + c] * d[1] + v[8 + c] * d[2];
    normal(w);
    if (taken < TakeFrames) {
      for (int c = 0; c < 3; ++c)
        sum[c] += w[c];
      ++taken;
    }
    if (on) {
      float k[3] = {sum[0], sum[1], sum[2]};
      normal(k);
      for (int r = 0; r < 3; ++r)
        out[r] = v[r * 4] * k[0] + v[r * 4 + 1] * k[1] + v[r * 4 + 2] * k[2];
      normal(out);
    }
  }
  if (log) {
    // the camera looks along view -z: in the world, minus the third row
    const float fx = -v[8], fy = -v[9], fz = -v[10];
    std::fprintf(stderr, "[rt] sun copy %llu known %d view %.4f %.4f %.4f world %.4f %.4f %.4f heading %.1f pitch %.1f taken %u rays %.4f %.4f %.4f\n",
                 static_cast<unsigned long long>(job.copy), int(known), d[0], d[1], d[2], w[0], w[1], w[2],
                 known ? std::atan2(fx, fz) * 57.29578f : 0.f, known ? std::atan2(fy, std::hypot(fx, fz)) * 57.29578f : 0.f, taken, out[0], out[1], out[2]);
  }
  if (on)
    for (int a = 0; a < 3; ++a)
      d[a] = out[a];
}

// The rays' part: this frame's structures and the trace itself
void trace_part(Slot& s, Job& job, u32 base, u32& glowCount) {
  for (const auto& t : job.textures)
    upload_texture(s, t);
  auto& fr = job.frame;
  ID3D12GraphicsCommandList4* cl = s.cl.Get();
  if (fr.lights.empty())
    fr.lights.push_back({{0, 0, 1e6f}, {0, 0, 0}, 1.f});
  const u32 lightCount = static_cast<u32>(fr.lights.size());
  // one element past the engine lights: the sun, its direction in view space; AURORA_RT_SUN=x,y,z sets a direction fixed to the view, for the test where the game has no light
  {
    static const float gain = post_setting("AURORA_RT_SUN_GAIN", 0.f), white = post_setting("AURORA_RT_SUN_WHITE", 0.f);
    static float fixedDir[3]{};
    static const bool fixed = [] {
      const char* v = std::getenv("AURORA_RT_SUN");
      return v != nullptr && std::sscanf(v, "%f,%f,%f", &fixedDir[0], &fixedDir[1], &fixedDir[2]) == 3;
    }();
    const float power = job.sunColour[0] + job.sunColour[1] + job.sunColour[2];
    const bool own = !fixed && power > 0.f;
    float d[3] = {own ? job.sunDir[0] : fixedDir[0], own ? job.sunDir[1] : fixedDir[1], own ? job.sunDir[2] : fixedDir[2]};
    if (!own && !fixed)
      d[0] = 0.35f, d[1] = 0.8f, d[2] = 0.45f;
    if (own)
      world_sun(job, d);
    const float len = std::max(std::hypot(d[0], d[1], d[2]), 1e-5f);
    EngineLight sun{{d[0] / len, d[1] / len, d[2] / len}, {gain, gain, gain}, 1.f};
    // the colour as the game's light has it, its brightest channel at 1; AURORA_RT_SUN_WHITE=1 leaves it white
    if (own && white == 0.f) {
      const float most = std::max({job.sunColour[0], job.sunColour[1], job.sunColour[2]});
      for (int a = 0; a < 3; ++a)
        sun.colour[a] = gain * job.sunColour[a] / most;
    }
    fr.lights.push_back(sun);
  }
  if (fr.stages.empty())
    fr.stages.push_back({});
  stage(s, s.verts, fr.verts.data(), fr.verts.size() * sizeof(Vertex));
  const bool lit = denoise_on() && fr.prevPos.size() == fr.verts.size() * 3;
  if (lit)
    stage(s, s.prevPos, fr.prevPos.data(), fr.prevPos.size() * sizeof(float));
  stage(s, s.draws, fr.draws.data(), fr.draws.size() * sizeof(GpuDraw));
  stage(s, s.stages, fr.stages.data(), fr.stages.size() * sizeof(StageRec));
  stage(s, s.lights, fr.lights.data(), fr.lights.size() * sizeof(EngineLight));
  if (fr.glows.empty())
    fr.glows.push_back({});
  glowCount = job.glow > 0 ? static_cast<u32>(fr.glows.size()) : 0;
  stage(s, s.glows, fr.glows.data(), fr.glows.size() * sizeof(GlowLight));
  world_stage(s, job);
  if (g.failed)
    return;
  // solid, see-through and glows, blended layers (AURORA_RT_LAYER_PASS; else none), sky: a structure each, so a shadow ray can leave out what light passes through.
  // Then static placements' draws for primary rays alone (AURORA_RT_SCENE_STEADY; else none), by mask 64 (rt.hlsl MaskStaticDraws). Last the draws camera
  // rays pass (additive, no colour, darkening: rt_rules.hpp assemble_frame): seen through as before by every other ray (the see-through mask), but marked
  // non-opaque, which camera rays cull (rt.hlsl), so they cost those no look
  const u32 firsts[6] = {0, fr.seeFirstVertex, fr.layerFirstVertex, fr.skyFirstVertex, fr.primaryFirstVertex, fr.passFirstVertex};
  const u32 counts[6] = {fr.seeFirstVertex, fr.layerFirstVertex - fr.seeFirstVertex, fr.skyFirstVertex - fr.layerFirstVertex, fr.primaryFirstVertex - fr.skyFirstVertex,
                         fr.passFirstVertex - fr.primaryFirstVertex, static_cast<u32>(fr.verts.size()) - fr.passFirstVertex};
  static constexpr u32 masks[6] = {1, 4, 128, 2, 64, 4};
  constexpr u32 Passed = 5;
  ensure(s.scratch, s.scratchSize, 256u << 20, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  u64 scratchAt = 0;
  {
    D3D12_RESOURCE_BARRIER bs[1]{};
    bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    bs[0].UAV.pResource = nullptr;
    cl->ResourceBarrier(1, bs);
  }
  for (u32 i = 0; i < 6; ++i)
    if (counts[i])
      build_blas(s, i, s.verts.gpu->GetGPUVirtualAddress() + u64(firsts[i]) * sizeof(Vertex), counts[i], scratchAt);
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  cl->ResourceBarrier(1, &uav);
  static thread_local std::vector<D3D12_RAYTRACING_INSTANCE_DESC> inst;
  inst.assign(6, D3D12_RAYTRACING_INSTANCE_DESC{});
  u32 instances = 0;
  for (u32 i = 0; i < 6; ++i) {
    if (!counts[i])
      continue;
    auto& d = inst[instances++];
    d.Transform[0][0] = d.Transform[1][1] = d.Transform[2][2] = 1;
    d.InstanceMask = masks[i];
    if (i == Passed)
      d.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
    d.InstanceID = firsts[i] / 3;  // a hit's triangle: this plus its primitive index
    d.AccelerationStructure = s.blas[i]->GetGPUVirtualAddress();
  }
  inst.resize(instances);
  world_instances(s, job, inst);
  instances = static_cast<u32>(inst.size());
  const u64 instanceBytes = std::max<size_t>(instances, 4) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
  if (!s.instances || s.instances->GetDesc().Width < instanceBytes) {
    if (s.instances)
      s.keep.push_back(s.instances);
    s.instances = make_buffer(D3D12_HEAP_TYPE_UPLOAD, instanceBytes + instanceBytes / 4, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  }
  void* im = nullptr;
  RT_HR(s.instances->Map(0, nullptr, &im));
  std::memcpy(im, inst.data(), instanceBytes > inst.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC) ? inst.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC) : instanceBytes);
  s.instances->Unmap(0, nullptr);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tin{};
  tin.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  tin.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  tin.NumDescs = instances;
  tin.InstanceDescs = s.instances->GetGPUVirtualAddress();
  if (scene::static_on())
    tin.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;  // hundreds of overlapping instances: rays pay far more than the build
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
  g.dev->GetRaytracingAccelerationStructurePrebuildInfo(&tin, &pre);
  ensure(s.tlas, s.tlasSize, pre.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{};
  bd.Inputs = tin;
  bd.DestAccelerationStructureData = s.tlas->GetGPUVirtualAddress();
  bd.ScratchAccelerationStructureData = s.scratch->GetGPUVirtualAddress() + scratchAt;
  cl->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);
  cl->ResourceBarrier(1, &uav);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 1);

  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
  sv.RaytracingAccelerationStructure.Location = s.tlas->GetGPUVirtualAddress();
  g.dev->CreateShaderResourceView(nullptr, &sv, cpuh(base));
  srv_buffer(s.verts.gpu.Get(), static_cast<u32>(fr.verts.size()), sizeof(Vertex), base + 1);
  srv_buffer(s.draws.gpu.Get(), static_cast<u32>(fr.draws.size()), sizeof(GpuDraw), base + 2);
  srv_buffer(s.lights.gpu.Get(), static_cast<u32>(fr.lights.size()), sizeof(EngineLight), base + 4);
  srv_buffer(s.stages.gpu.Get(), static_cast<u32>(fr.stages.size()), sizeof(StageRec), base + 5);
  srv_buffer(s.glows.gpu.Get(), static_cast<u32>(fr.glows.size()), sizeof(GlowLight), base + 7);
  srv_buffer(s.worldMeshes.gpu.Get(), static_cast<u32>(std::max<size_t>(job.world.colours.size(), 1)), sizeof(scene::GpuMesh), base + 10);
  srv_buffer(scene::static_on() && job.world.scene == g.worldScene ? g.worldAttrs.Get() : nullptr, g.worldScene ? static_cast<u32>(g.worldScene->attrs.size()) : 1, sizeof(scene::WorldAttr), base + 11);
  Params p{};
  p.size[0] = static_cast<float>(g.rw), p.size[1] = static_cast<float>(g.rh);
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
  p.lightCount = lightCount;
  static const u32 lightRays = static_cast<u32>(post_setting("AURORA_RT_LIGHTING", 0.f));
  static const float skyGain = post_setting("AURORA_RT_SKY", 1.f), reach = post_setting("AURORA_RT_REACH", 150.f), bounce = post_setting("AURORA_RT_BOUNCE", 0.f), floorV = post_setting("AURORA_RT_VIS_FLOOR", 0.3f);
  // AURORA_RT_LIGHTING_SHOTS=rays,copy,copy..: that many rays on those test pictures only, so a run reaches them at speed
  static std::vector<u64> shotCopies;
  static const u32 shotRays = [] {
    u32 rays = 0;
    if (const char* v = std::getenv("AURORA_RT_LIGHTING_SHOTS")) {
      rays = static_cast<u32>(std::atoi(v));
      for (const char* c = std::strchr(v, ','); c != nullptr; c = std::strchr(c + 1, ','))
        shotCopies.push_back(std::strtoull(c + 1, nullptr, 10));
    }
    return rays;
  }();
  bool atShot = false;
  for (const u64 c : shotCopies)
    atShot = atShot || (job.copy + 1 >= c && job.copy <= c + 1);
  p.lightRays = lightRays > 0 && atShot ? shotRays : lightRays, p.skyGain = skyGain, p.lightReach = reach, p.bounce = bounce, p.visFloor = floorV;
  static const float range = [] {
    const char* v = std::getenv("AURORA_RT_LIGHT_RANGE");
    return v != nullptr ? static_cast<float>(std::atof(v)) : 4.f;
  }();
  // on unless switched off: AURORA_RT_LIGHT_SHADOW=0
  static const u32 shadow = [] {
    const char* v = std::getenv("AURORA_RT_LIGHT_SHADOW");
    return v != nullptr && v[0] == '0' ? 0u : 1u;
  }();
  p.lightRange = range;
  p.lightShadow = shadow;
  p.bodyCalm = job.calm;
  p.wet = job.wet;
  static const u32 stableShine = post_setting("AURORA_RT_SHINE_STABLE", 0.f) != 0.f ? 1 : 0;
  // AURORA_RT_PAIR_GAP=1 (rt.hlsl pair_gap_on): a primary ray past a drawn placement's triangle with no draw of it looks for that draw just around it
  static const u32 pairGap = post_setting("AURORA_RT_PAIR_GAP", 0.f) != 0.f ? 4 : 0;
  // AURORA_RT_LAYER_PASS=1 (rt.hlsl layer_pass_on): primary rays pass a blended draw that tests depth but writes none, as they pass additive draws
  static const u32 layerPass = post_setting("AURORA_RT_LAYER_PASS", 0.f) != 0.f ? 8 : 0;
  // AURORA_RT_SCENE_STEADY=1 (rt.hlsl steady_on): primary rays' look for draws around a drawn placement's hit takes the static draws kept for them
  const u32 steady = scene::steady_on() ? 16 : 0;
  p.stableShine = stableShine | (g.beforeThis ? 2u : 0u) | pairGap | layerPass | steady;
  p.skyFirst = fr.skyFirstVertex;
  p.glowCount = glowCount;
  p.glowGain = job.glow;
  static const u32 glowShow = std::getenv("AURORA_RT_GLOW_SHOW") != nullptr ? 1 : 0;
  p.glowShow = glowShow;
  std::memcpy(g.lastView, p.rectWH, sizeof g.lastView);
  cl->SetComputeRoot32BitConstants(0, sizeof(Params) / 4, &p, 0);
  // Each step's constants and textures: a slot's own descriptors, written as it goes
  if (!s.constants) {
    s.constants = make_buffer(D3D12_HEAP_TYPE_UPLOAD, ConstantStride * DenoiseSteps, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    RT_HR(s.constants->Map(0, nullptr, &s.constantsMap));
  }
  DenoiseParams dp{};
  const u32 denoiseFirst = PostFirst + PostDescriptors + FlareDescriptors + g.slot * DenoiseSteps * DenoiseViews;
  u32 stepAt = 0;
  // A step's textures by the order of rt.hlsl's registers u4 on; a null where it reads nothing
  const auto bind = [&](std::initializer_list<ID3D12Resource*> uavs) {
    const u32 at = denoiseFirst + stepAt * DenoiseViews;
    D3D12_SHADER_RESOURCE_VIEW_DESC pv{};
    pv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    pv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    pv.Format = DXGI_FORMAT_UNKNOWN;
    pv.Buffer.NumElements = lit ? static_cast<u32>(fr.verts.size()) : 1;
    pv.Buffer.StructureByteStride = 12;
    g.dev->CreateShaderResourceView(lit ? s.prevPos.gpu.Get() : nullptr, &pv, cpuh(at));
    u32 k = 1;
    for (ID3D12Resource* r : uavs) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
      uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      uv.Format = r != nullptr ? r->GetDesc().Format : DXGI_FORMAT_R16G16B16A16_FLOAT;
      g.dev->CreateUnorderedAccessView(r, nullptr, &uv, cpuh(at + k++));
    }
    for (; k < DenoiseViews; ++k) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
      uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      uv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      g.dev->CreateUnorderedAccessView(nullptr, nullptr, &uv, cpuh(at + k));
    }
    std::memcpy(static_cast<u8*>(s.constantsMap) + size_t(stepAt) * ConstantStride, &dp, sizeof dp);
    cl->SetComputeRootDescriptorTable(8, gpuh(at));
    cl->SetComputeRootConstantBufferView(9, s.constants->GetGPUVirtualAddress() + u64(stepAt) * ConstantStride);
    ++stepAt;
  };
  const auto barrier = [&] {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    cl->ResourceBarrier(1, &b);
  };
  static const float alphaLeast = post_setting("AURORA_RT_DENOISE_ALPHA", 0.1f), phiNormal = post_setting("AURORA_RT_DENOISE_PHI_N", 32.f),
                     phiDepth = post_setting("AURORA_RT_DENOISE_PHI_Z", 1.f), phiLight = post_setting("AURORA_RT_DENOISE_PHI_L", 4.f);
  static const u32 show = static_cast<u32>(post_setting("AURORA_RT_SHOW", 0.f));
  dp.alphaLeast = alphaLeast, dp.phiNormal = phiNormal, dp.phiDepth = phiDepth, dp.phiLight = phiLight;
  dp.prevScale[0] = g.prevProj[0], dp.prevScale[1] = g.prevProj[5], dp.prevOffset[0] = g.prevProj[2], dp.prevOffset[1] = g.prevProj[6];
  for (u32 i = 0; i < 2; ++i)
    dp.prevVpXY[i] = g.prevViewport[i], dp.prevVpWH[i] = g.prevViewport[2 + i];
  std::memcpy(dp.cam, job.motion.prevFromCur.m, sizeof dp.cam);
  dp.reset = !g.historyGood || !job.motion.known || job.copy > g.lastLitCopy + 4 ? 1 : 0;
  ID3D12Resource* litOpen = nullptr;
  ID3D12Resource* litSky = nullptr;
  const u32 now = g.flip, old = g.flip ^ 1;
  const u32 lightW = half_on() ? (g.w + 1) / 2 : g.w, lightH = half_on() ? (g.h + 1) / 2 : g.h;
  dp.litScale = half_on() ? 2.f : 1.f, dp.half = half_on() ? 1 : 0;
  if (lit && p.lightRays > 0) {
    dp.denoised = 1;
    bind({g.guide[now].Get(), g.motion.Get(), g.noisyOpen.Get(), g.noisySky.Get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, g.refl.Get()});
    cl->SetPipelineState(g.lightPass.Get());
    cl->Dispatch((lightW + 7) / 8, (lightH + 7) / 8, 1);
    barrier();
    cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 2);
    bind({g.guide[now].Get(), g.motion.Get(), g.noisyOpen.Get(), g.noisySky.Get(), g.guide[old].Get(), g.histOpen[old].Get(), g.histSky[old].Get(), g.moments[old].Get(),
          g.tmpOpen[0].Get(), g.tmpSky[0].Get(), g.moments[now].Get()});
    cl->SetPipelineState(g.temporal.Get());
    cl->Dispatch((lightW + 7) / 8, (lightH + 7) / 8, 1);
    barrier();
    // the first step's result is the next frame's history; then back and forth
    ID3D12Resource* fromO = g.tmpOpen[0].Get();
    ID3D12Resource* fromS = g.tmpSky[0].Get();
    cl->SetPipelineState(g.atrous.Get());
    for (u32 i = 0; i < DenoiseIterations; ++i) {
      ID3D12Resource* toO = i == 0 ? g.histOpen[now].Get() : g.tmpOpen[i & 1].Get();
      ID3D12Resource* toS = i == 0 ? g.histSky[now].Get() : g.tmpSky[i & 1].Get();
      dp.step = 1u << i;
      bind({g.guide[now].Get(), nullptr, fromO, fromS, nullptr, nullptr, nullptr, nullptr, toO, toS});
      cl->Dispatch((lightW + 7) / 8, (lightH + 7) / 8, 1);
      barrier();
      fromO = toO, fromS = toS;
    }
    litOpen = fromO, litSky = fromS;
    g.historyGood = true;
    g.lastLitCopy = job.copy;
    g.flip = old;
  } else {
    cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 2);
  }
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 3);
  std::memcpy(g.prevProj, job.proj, sizeof g.prevProj);
  std::memcpy(g.prevViewport, job.viewport, sizeof g.prevViewport);
  dp.show = lit ? show : 0;
  // AURORA_RT_SHOW=4: the raw rays in the place of the smoothed
  if (show == 4 && lit) litOpen = g.noisyOpen.Get(), litSky = g.noisySky.Get();
  bind({lit ? g.guide[now].Get() : nullptr, lit ? g.motion.Get() : nullptr, litOpen, litSky, nullptr, nullptr, nullptr, lit ? g.moments[now].Get() : nullptr, nullptr, nullptr, nullptr, lit ? g.refl.Get() : nullptr});
  cl->SetPipelineState(g.trace.Get());
  cl->Dispatch((g.rw + 7) / 8, (g.rh + 7) / 8, 1);
  barrier();
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 4);
}

// The look: the frame's high range, a bloom chain, exposure, the tone curve. On the frame's own queue: no frame waits for it.
// The glow's two passes from this picture's raster (t3), true or between; leaves the arms and halo readable at table 4 (t8) for compose,
// which adds them. flare_after puts both textures back as they were.
void run_flare(ID3D12GraphicsCommandList4* cl) {
  PostParams pp{};
  pp.texel[0] = 1.f / g.flareW, pp.texel[1] = 1.f / g.flareH;
  const auto barrier = [&](ID3D12Resource* r, bool toRead) {
    D3D12_RESOURCE_BARRIER b{};
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = toRead ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter = toRead ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cl->ResourceBarrier(1, &b);
  };
  cl->SetComputeRoot32BitConstants(3, sizeof(PostParams) / 4, &pp, 0);
  cl->SetPipelineState(g.flareSrc.Get());
  cl->SetComputeRootDescriptorTable(6, gpuh(ViewFlare + 1));
  cl->Dispatch((g.flareW + 7) / 8, (g.flareH + 7) / 8, 1);
  barrier(g.flareRes[0].Get(), true);
  cl->SetPipelineState(g.flareStar.Get());
  cl->SetComputeRootDescriptorTable(4, gpuh(ViewFlare));
  cl->SetComputeRootDescriptorTable(6, gpuh(ViewFlare + 3));
  cl->Dispatch((g.flareW + 7) / 8, (g.flareH + 7) / 8, 1);
  barrier(g.flareRes[0].Get(), false);
  barrier(g.flareRes[1].Get(), true);
  cl->SetComputeRootDescriptorTable(4, gpuh(ViewFlare + 2));
}
void flare_after(ID3D12GraphicsCommandList4* cl) {
  D3D12_RESOURCE_BARRIER b{};
  b.Transition.pResource = g.flareRes[1].Get();
  b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  b.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  cl->ResourceBarrier(1, &b);
}

void run_post(Slot& s, Job& job) {
  ID3D12GraphicsCommandList4* cl = s.cl.Get();
  static const float range = post_setting("AURORA_LOOK_RANGE", 12.f), bloom = post_setting("AURORA_LOOK_BLOOM", 0.18f), grade = post_setting("AURORA_LOOK_GRADE", 1.f);
  static const u32 autoExposure = post_setting("AURORA_LOOK_EXPOSURE", 1.f) != 0.f ? 1 : 0;
  const auto now = std::chrono::steady_clock::now();
  const float dt = g.lastPost == std::chrono::steady_clock::time_point{} ? 0.f : std::min(std::chrono::duration<float>(now - g.lastPost).count(), 0.25f);
  g.lastPost = now;
  PostParams pp{{0.f, 0.f}, range, bloom, dt, grade, job.postOnly ? 0u : 1u, autoExposure, g.frame++, PostLevels};
  const auto transition = [&](ID3D12Resource* r, bool toRead) {
    D3D12_RESOURCE_BARRIER b{};
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = toRead ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter = toRead ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cl->ResourceBarrier(1, &b);
  };
  const auto pass = [&](ID3D12PipelineState* pso, u32 readA, u32 readB, u32 write, u32 w, u32 h, float tx, float ty) {
    cl->SetPipelineState(pso);
    pp.texel[0] = tx, pp.texel[1] = ty;
    cl->SetComputeRoot32BitConstants(3, sizeof(PostParams) / 4, &pp, 0);
    cl->SetComputeRootDescriptorTable(4, gpuh(readA));
    cl->SetComputeRootDescriptorTable(5, gpuh(readB));
    cl->SetComputeRootDescriptorTable(6, gpuh(write));
    cl->SetComputeRootDescriptorTable(7, gpuh(ViewExpo));
    cl->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
  };
  pass(g.hdrIn.Get(), ViewHdr, ViewHdr, ViewHdr + 1, g.w, g.h, 0.f, 0.f);
  transition(g.hdrRes.Get(), true);
  for (u32 i = 0; i < PostLevels; ++i) {
    const u32 from = i == 0 ? ViewHdr : ViewDown + 2 * (i - 1);
    const u32 fw = i == 0 ? g.w : g.levelW[i - 1], fh = i == 0 ? g.h : g.levelH[i - 1];
    pp.rayed = (job.postOnly ? 0u : 1u) | (i == 0 ? 2u : 0u);  // bit 1: the first step down keeps a lone bright pixel from flickering
    pass(g.bloomDown.Get(), from, from, ViewDown + 2 * i + 1, g.levelW[i], g.levelH[i], 1.f / fw, 1.f / fh);
    transition(g.downRes[i].Get(), true);
  }
  pp.rayed = job.postOnly ? 0u : 1u;
  for (u32 i = PostLevels - 1; i-- > 0;) {
    const u32 lower = i == PostLevels - 2 ? ViewDown + 2 * (PostLevels - 1) : ViewUp + 2 * (i + 1);
    const u32 lw = g.levelW[i + 1], lh = g.levelH[i + 1];
    pass(g.bloomUp.Get(), lower, ViewDown + 2 * i, ViewUp + 2 * i + 1, g.levelW[i], g.levelH[i], 1.f / lw, 1.f / lh);
    transition(g.upRes[i].Get(), true);
  }
  // exposure from the frame's small copy, kept on the GPU
  cl->SetPipelineState(g.exposure.Get());
  cl->SetComputeRoot32BitConstants(3, sizeof(PostParams) / 4, &pp, 0);
  cl->SetComputeRootDescriptorTable(4, gpuh(ViewDown + 2 * (PostLevels - 2)));
  cl->SetComputeRootDescriptorTable(7, gpuh(ViewExpo));
  cl->Dispatch(1, 1, 1);
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav.UAV.pResource = g.expoRes.Get();
  cl->ResourceBarrier(1, &uav);
  pass(g.tone.Get(), ViewHdr, ViewUp, ViewHdr + 1, g.w, g.h, 1.f / g.w, 1.f / g.h);
  transition(g.hdrRes.Get(), false);
  for (u32 i = 0; i < PostLevels; ++i)
    transition(g.downRes[i].Get(), false);
  for (u32 i = 0; i + 1 < PostLevels; ++i)
    transition(g.upRes[i].Get(), false);
}

// What the rays do for one frame, on Dawn's queue, between the frame's two command buffers
// One picture between: no rays. The true frame's reflections and light (its term) carried by the corrections, then the picture's own raster
// beneath them; bloom and exposure as the true frame left them.
void run_between() {
  const auto t0 = std::chrono::steady_clock::now();
  g.slot = (g.slot + 1) % Frames;
  Slot& s = g.slots[g.slot];
  if (g.fence->GetCompletedValue() < s.fence) {
    RT_HR(g.fence->SetEventOnCompletion(s.fence, g.event));
    WaitForSingleObject(g.event, INFINITE);
  }
  if (s.between && s.fence != 0) {
    u64* ts = nullptr;
    const D3D12_RANGE range{8 * 6 * g.slot, 8 * 6 * (g.slot + 1)};
    if (SUCCEEDED(g.queryResult->Map(0, &range, reinterpret_cast<void**>(&ts)))) {
      const u64* t6 = ts + 6 * g.slot;
      g.msBetweenGpu += double(t6[5] - t6[0]) * 1000.0 / g.gpuTicks;
      ++g.betweenGpuCount;
      g.queryResult->Unmap(0, nullptr);
    }
  }
  s.between = true;
  s.keep.clear();
  if (!s.betweenConstants) {
    s.betweenConstants = make_buffer(D3D12_HEAP_TYPE_UPLOAD, (sizeof(BetweenParams) + 255) & ~255ull, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    RT_HR(s.betweenConstants->Map(0, nullptr, &s.betweenMap));
  }
  BetweenParams& b = g.betweenParams;
  b.size[0] = static_cast<float>(g.rw), b.size[1] = static_cast<float>(g.rh);
  b.rectXY[0] = b.rectXY[1] = 0;
  b.rectWH[0] = g.lastView[0], b.rectWH[1] = g.lastView[1];
  b.vpXY[0] = g.lastView[2], b.vpXY[1] = g.lastView[3], b.vpWH[0] = g.lastView[4], b.vpWH[1] = g.lastView[5];
  b.pScale[0] = g.lastView[6], b.pScale[1] = g.lastView[7], b.pOffset[0] = g.lastView[8], b.pOffset[1] = g.lastView[9];
  std::memcpy(s.betweenMap, &b, sizeof b);
  RT_HR(s.alloc->Reset());
  RT_HR(s.cl->Reset(s.alloc.Get(), nullptr));
  ID3D12GraphicsCommandList4* cl = s.cl.Get();
  ID3D12DescriptorHeap* heaps[] = {g.heap.Get()};
  cl->SetDescriptorHeaps(1, heaps);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot);
  const u32 base = g.slot * SlotDescriptors;
  D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
  sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sv.Format = g.inputRes->GetDesc().Format;
  sv.Texture2D.MipLevels = 1;
  g.dev->CreateShaderResourceView(g.inputRes.Get(), &sv, cpuh(base + 3));
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
  uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  uv.Format = output_format();
  g.dev->CreateUnorderedAccessView(g.outputRes.Get(), nullptr, &uv, cpuh(base + 9));
  cl->SetComputeRootSignature(g.root.Get());
  cl->SetComputeRootDescriptorTable(1, gpuh(base));
  cl->SetComputeRootDescriptorTable(2, gpuh(Frames * SlotDescriptors));
  const auto transition = [&](ID3D12Resource* r, bool toRead) {
    D3D12_RESOURCE_BARRIER t{};
    t.Transition.pResource = r;
    t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    t.Transition.StateBefore = toRead ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    t.Transition.StateAfter = toRead ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cl->ResourceBarrier(1, &t);
  };
  const bool rayed = g.haveTerm;
  transition(g.termRes.Get(), true);
  if (rayed) {
    cl->SetComputeRootConstantBufferView(9, s.betweenConstants->GetGPUVirtualAddress());
    cl->SetPipelineState(g.betweenWarp.Get());
    cl->SetComputeRootDescriptorTable(4, gpuh(ViewBetweenTrue));
    cl->SetComputeRootDescriptorTable(5, gpuh(ViewBetweenTrue));
    cl->SetComputeRootDescriptorTable(6, gpuh(ViewBetween + 1));
    cl->Dispatch((g.rw + 7) / 8, (g.rh + 7) / 8, 1);
  }
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 1);
  for (u32 q = 2; q < 5; ++q)
    cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + q);
  transition(g.termBetween.Get(), true);
  if (post_on()) {
    static const float range = post_setting("AURORA_LOOK_RANGE", 12.f), bloom = post_setting("AURORA_LOOK_BLOOM", 0.18f), grade = post_setting("AURORA_LOOK_GRADE", 1.f);
    static const u32 autoExposure = post_setting("AURORA_LOOK_EXPOSURE", 1.f) != 0.f ? 1 : 0;
    PostParams pp{{0.f, 0.f}, range, bloom, 0.f, grade, 1u, autoExposure, g.frame, PostLevels};
    const auto pass = [&](ID3D12PipelineState* pso, u32 readA, u32 readB, u32 write, float tx, float ty) {
      cl->SetPipelineState(pso);
      pp.texel[0] = tx, pp.texel[1] = ty;
      cl->SetComputeRoot32BitConstants(3, sizeof(PostParams) / 4, &pp, 0);
      cl->SetComputeRootDescriptorTable(4, gpuh(readA));
      cl->SetComputeRootDescriptorTable(5, gpuh(readB));
      cl->SetComputeRootDescriptorTable(6, gpuh(write));
      cl->SetComputeRootDescriptorTable(7, gpuh(ViewExpo));
      cl->Dispatch((g.w + 7) / 8, (g.h + 7) / 8, 1);
    };
    pass(g.betweenHdr.Get(), ViewBetween, ViewBetween, ViewHdr + 1, 0.f, 0.f);
    transition(g.hdrRes.Get(), true);
    transition(g.upRes[0].Get(), true);
    pass(g.tone.Get(), ViewHdr, ViewUp, ViewHdr + 1, 1.f / g.w, 1.f / g.h);
    transition(g.hdrRes.Get(), false);
    transition(g.upRes[0].Get(), false);
  } else {
    const bool flare = flare_on() && g.flareSrc;
    if (flare)
      run_flare(cl);
    cl->SetPipelineState(g.betweenCompose.Get());
    cl->SetComputeRootDescriptorTable(5, gpuh(ViewBetween));
    cl->Dispatch((g.w + 7) / 8, (g.h + 7) / 8, 1);
    if (flare)
      flare_after(cl);
  }
  transition(g.termBetween.Get(), false);
  transition(g.termRes.Get(), false);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 5);
  cl->ResolveQueryData(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot, 6, g.queryResult.Get(), 8 * 6 * g.slot);
  RT_HR(cl->Close());
  ID3D12CommandList* lists[] = {cl};
  g.queue->ExecuteCommandLists(1, lists);
  s.fence = ++g.fenceValue;
  RT_HR(g.queue->Signal(g.fence.Get(), s.fence));
  g.msBetweenCpu += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (++g.betweenCount % 1200 == 0)
    std::fprintf(stderr, "[rt] between: %llu pictures, CPU %.2f ms, GPU warp and compose %.2f ms each\n", static_cast<unsigned long long>(g.betweenCount), g.msBetweenCpu / g.betweenCount,
                 g.betweenGpuCount ? g.msBetweenGpu / g.betweenGpuCount : 0.0);
}

void run_rays(Job& job) {
  const auto t0 = std::chrono::steady_clock::now();
  g.slot = (g.slot + 1) % Frames;
  Slot& s = g.slots[g.slot];
  if (g.fence->GetCompletedValue() < s.fence) {
    gfx::stagelog::Scope t{"rt_fence_wait", job.copy};
    RT_HR(g.fence->SetEventOnCompletion(s.fence, g.event));
    WaitForSingleObject(g.event, INFINITE);
  }
  // this slot's last timings, now done
  if (s.fence != 0) {
    u64* ts = nullptr;
    const D3D12_RANGE range{8 * 6 * g.slot, 8 * 6 * (g.slot + 1)};
    if (SUCCEEDED(g.queryResult->Map(0, &range, reinterpret_cast<void**>(&ts)))) {
      const u64* t6 = ts + 6 * g.slot;
      const double k = 1000.0 / g.gpuTicks, all = double(t6[5] - t6[0]) * k;
      if (s.between) {
        g.msBetweenGpu += all;
        ++g.betweenGpuCount;
        g.queryResult->Unmap(0, nullptr);
        s.fence = 0;  // (this slot's timing taken)
      } else
      {
      const double build = double(t6[1] - t6[0]) * k, lightMs = double(t6[2] - t6[1]) * k, denoiseMs = double(t6[3] - t6[2]) * k, reflectMs = double(t6[4] - t6[3]) * k, composeMs = double(t6[5] - t6[4]) * k;
      g.msGpu += all;
      g.msGpuMost = std::max(g.msGpuMost, all);
      g.msTrace += reflectMs;
      g.msLight += lightMs, g.msDenoise += denoiseMs;
      // AURORA_RT_TIMELOG=<file>: a line a frame, to find one slow frame by its copy and what it carried
      if (g.timeLog != nullptr)
        std::fprintf(g.timeLog, "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u\n", static_cast<unsigned long long>(s.copy), build, lightMs, denoiseMs, reflectMs, composeMs, s.tris, s.lightCount, s.uploads);
      g.queryResult->Unmap(0, nullptr);
      }
    }
  }
  s.keep.clear();
  s.between = false;
  s.copy = job.copy, s.tris = static_cast<u32>(job.frame.verts.size() / 3), s.lightCount = static_cast<u32>(job.frame.lights.size()), s.uploads = static_cast<u32>(job.textures.size());
  RT_HR(s.alloc->Reset());
  RT_HR(s.cl->Reset(s.alloc.Get(), nullptr));
  ID3D12GraphicsCommandList4* cl = s.cl.Get();
  ID3D12DescriptorHeap* heaps[] = {g.heap.Get()};
  cl->SetDescriptorHeaps(1, heaps);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot);
  auto& fr = job.frame;
  const u32 base = g.slot * SlotDescriptors;
  u32 glowCount = 0;
  {
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Format = g.inputRes->GetDesc().Format;
    sv.Texture2D.MipLevels = 1;
    g.dev->CreateShaderResourceView(g.inputRes.Get(), &sv, cpuh(base + 3));
    g.dev->CreateShaderResourceView(g.beforeThis ? g.beforeRes.Get() : g.inputRes.Get(), &sv, cpuh(base + 12));
    sv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g.dev->CreateShaderResourceView(g.termRes.Get(), &sv, cpuh(base + 6));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
    uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g.dev->CreateUnorderedAccessView(g.termRes.Get(), nullptr, &uv, cpuh(base + 8));
    uv.Format = output_format();
    g.dev->CreateUnorderedAccessView(g.outputRes.Get(), nullptr, &uv, cpuh(base + 9));
  }
  cl->SetComputeRootSignature(g.root.Get());
  cl->SetComputeRootDescriptorTable(1, gpuh(base));
  cl->SetComputeRootDescriptorTable(2, gpuh(Frames * SlotDescriptors));
  if (job.postOnly) {
    cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 1);
    for (u32 q = 2; q < 5; ++q)
      cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + q);
  } else {
    trace_part(s, job, base, glowCount);
    if (g.failed)
      return;
  }
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav.UAV.pResource = g.termRes.Get();
  cl->ResourceBarrier(1, &uav);
  D3D12_RESOURCE_BARRIER tb{};
  tb.Transition.pResource = g.termRes.Get();
  tb.Transition.Subresource = 0;
  tb.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  tb.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  cl->ResourceBarrier(1, &tb);
  if (post_on()) {
    run_post(s, job);
  } else {
    const bool flare = flare_on() && g.flareSrc;
    if (flare)
      run_flare(cl);
    cl->SetPipelineState(g.compose.Get());
    cl->Dispatch((g.w + 7) / 8, (g.h + 7) / 8, 1);
    if (flare)
      flare_after(cl);
  }
  std::swap(tb.Transition.StateBefore, tb.Transition.StateAfter);
  cl->ResourceBarrier(1, &tb);
  cl->EndQuery(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot + 5);
  cl->ResolveQueryData(g.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6 * g.slot, 6, g.queryResult.Get(), 8 * 6 * g.slot);
  RT_HR(cl->Close());
  ID3D12CommandList* lists[] = {cl};
  g.queue->ExecuteCommandLists(1, lists);
  s.fence = ++g.fenceValue;
  RT_HR(g.queue->Signal(g.fence.Get(), s.fence));

  g.msComposeWait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  g.msBuild += job.captureMs;
  g.motionAgree += job.motion.agreeing, g.motionOwn += job.motion.matched;
  g.msRules += job.rulesMs;
  g.glowPieces += fr.glowPieces, g.glowDraws += fr.glowDraws, g.glowTop += fr.glowTopDraw;
  g.picks += job.lightPicks, g.merged += job.mergedLights, g.glowsSeen += glowCount;
  g.msTranslate += job.translateUs.load() / 1000.0;
  g.tris += fr.verts.size() / 3;
  g.lightsSeen += fr.lights.size();
  g.lightsMost = std::max<u64>(g.lightsMost, fr.lights.size());
  world_count(job);
  static const u32 every = [] {
    const char* v = std::getenv("AURORA_RT_REPORT");
    return v != nullptr ? static_cast<u32>(std::atoi(v)) : 300u;
  }();
  if (++g.count == every) {
    const double n = g.count;
    std::fprintf(stderr, "[rt] lit: light rays %.2f ms, smoothing %.2f ms, reflections %.2f ms; motion: camera agreed %.2f of what was compared, %.2f of vertices known by their own bytes\n", g.msLight / n, g.msDenoise / n, g.msTrace / n, g.motionAgree / n, g.motionOwn / n);
    std::fprintf(stderr, "[rt] per frame, %u frames at %ux%u: GPU build %.2f ms and rays+cleanup %.2f ms (all %.2f, worst %.2f); CPU in the stream %.2f ms, translation beside it %.2f ms, rules %.2f ms, worker %.2f ms; %.0f triangles, lights %.1f (most %llu), picks %.1f, merged %.1f, glows %.1f from %.0f pieces of %.1f draws, one draw held most %.1f\n",
                 g.count, g.w, g.h, (g.msGpu - g.msTrace - g.msLight - g.msDenoise) / n, g.msTrace / n, g.msGpu / n, g.msGpuMost, g.msBuild / n, g.msTranslate / n, g.msRules / n, g.msComposeWait / n, double(g.tris) / n, double(g.lightsSeen) / n,
                 static_cast<unsigned long long>(g.lightsMost), g.picks / n, g.merged / n, g.glowsSeen / n, g.glowPieces / n, g.glowDraws / n, g.glowTop / n);
    world_report(n);
    if (g_noEngineStar)
      std::fprintf(stderr, "[rt] engine stars not drawn: %llu so far\n", static_cast<unsigned long long>(g_starsSkipped.load(std::memory_order_relaxed)));
    g.msLight = g.msDenoise = g.motionAgree = g.motionOwn = 0;
    g.count = 0, g.msGpuMost = 0, g.msGpu = g.msTrace = g.msBuild = g.msRules = g.msTranslate = g.msComposeWait = 0, g.tris = 0, g.lightsSeen = 0, g.lightsMost = 0, g.picks = g.merged = g.glowsSeen = g.glowPieces = g.glowDraws = g.glowTop = 0;
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
  g.lastColor = color;
  g.haveTerm = false;
  const bool late = rules_late() && !job->postOnly;  // (a look-only job's rules are already done)
  if (!late) {
    gfx::stagelog::Scope t{"rules_wait", job->copy};
    job->rules.wait();
  }
  // not compiled yet: this frame keeps the game's own picture
  g.drawThis = (late || job->drawable) && g.ready && !g.failed;
  // a job not kept for its rays is given back or dropped here: its rules must be done with it first
  if (!g.drawThis) {
    if (late)
      job->rules.wait();
    give_back(std::move(job));
    return;
  }
  make_targets(p.w, p.h, static_cast<wgpu::TextureFormat>(p.format));
  if (g.failed) {
    if (late)
      job->rules.wait();
    return;
  }
  const wgpu::TexelCopyTextureInfo src{.texture = color}, dst{.texture = g.input};
  const wgpu::Extent3D size{.width = p.w, .height = p.h, .depthOrArrayLayers = 1};
  cmd.CopyTextureToTexture(&src, &dst, &size);
  g.first = ctx.frameEncoder->Finish();
  *ctx.frameEncoder = webgpu::g_device.CreateCommandEncoder();
  g.haveTerm = !job->postOnly;
  if (g.pending)  // (one its rays never ran for, with AURORA_RT_RULES_LATE perhaps still in its rules)
    g.pending->rules.wait();
  g.pending = std::move(job);
}

// A picture between: the corrected raster goes to the same place the true frame's went
void between_task(wgpu::CommandEncoder& cmd, gfx::EncoderTaskId type) {
  if (type != g.task)  // (the before-blends snapshot is the true frame's alone)
    return;
  if (g.failed || !g.ready || !g.drawThis || !g.lastColor || !g.input || !g.termBetween)
    return;
  const wgpu::TexelCopyTextureInfo src{.texture = g.lastColor}, dst{.texture = g.input};
  const wgpu::Extent3D size{.width = g.w, .height = g.h, .depthOrArrayLayers = 1};
  cmd.CopyTextureToTexture(&src, &dst, &size);
  g.first = cmd.Finish();
  cmd = webgpu::g_device.CreateCommandEncoder();
  g.pendingBetween = true;
}

void before_task(const gfx::EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void* payload, size_t, void*) {
  TaskPayload p;
  std::memcpy(&p, payload, sizeof p);
  const wgpu::Texture color = wgpu::Texture::Acquire(p.color);
  if (!g.ready || g.failed)
    return;
  make_targets(p.w, p.h, static_cast<wgpu::TextureFormat>(p.format));
  if (g.failed || !g.before)
    return;
  const wgpu::TexelCopyTextureInfo src{.texture = color}, dst{.texture = g.before};
  const wgpu::Extent3D size{.width = p.w, .height = p.h, .depthOrArrayLayers = 1};
  cmd.CopyTextureToTexture(&src, &dst, &size);
  g.beforeThis = true;
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

bool before_on() noexcept { return before_setting(); }
void snapshot_before() {
  if (g.failed || !g.ready)
    return;
  if (g.beforeTask == gfx::InvalidEncoderTask)
    g.beforeTask = gfx::register_encoder_task_type({.label = "ray tracing: before blends", .callback = before_task});
  const gx::fifo::HoldDrain hold;
  gfx::ResolvedTargets targets;
  if (!gfx::resolve_pass({.color = true}, targets) || !targets.colorTexture)
    return;
  TaskPayload p{nullptr, targets.colorTexture.MoveToCHandle(), targets.width, targets.height, static_cast<u32>(targets.colorFormat)};
  gfx::push_encoder_task(g.beforeTask, &p, sizeof p);
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
    gfx::g_betweenTask = between_task;
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

// This picture's corrections for the rays' pass: a machine that shows on screen and is near enough to matter (tuned: 200 units; radius 6, as
// the draws'), for how far away the machine's own pixels take this rule.
void between_set(const redraw::Request& q, std::size_t index) {
  constexpr float Reach = 200.f;
  static const u32 iterations = static_cast<u32>(post_setting("AURORA_RT_BETWEEN_ITERATIONS", 3.f));
  BetweenParams& b = g.betweenParams;
  const redraw::Picture& p = q.pictures[index];
  for (u32 r = 0; r < 3; ++r)
    for (u32 k = 0; k < 4; ++k)
      b.camera[r][k] = p.camera[4 * r + k];
  b.iterations = iterations;
  b.radius2 = 6.f * 6.f;
  b.far_ = 20000.f;
  b.count = 0;
  for (u32 i = 0; i < redraw::MostMachines; ++i) {
    if (!(q.machines & (1u << i)))
      continue;
    const auto& a = q.at[i];
    if (a[2] > 3.f || a[0] * a[0] + a[1] * a[1] + a[2] * a[2] > Reach * Reach)
      continue;
    BetweenMachine& m = b.machine[b.count++];
    for (u32 r = 0; r < 3; ++r)
      for (u32 k = 0; k < 4; ++k)
        m.m[r][k] = p.machine[i][4 * r + k];
    m.at[0] = a[0], m.at[1] = a[1], m.at[2] = a[2], m.at[3] = 0;
  }
}

void before_frame_submit() {
  if (g.pendingBetween && !g.failed) {
    g.pendingBetween = false;
    begin_access(g.inputMem, g.input);
    webgpu::g_queue.Submit(1, &g.first);
    g.first = {};
    end_access(g.inputMem, g.input);
    run_between();
    begin_access(g.outputMem, g.output);
    g.outputOpen = true;
    return;
  }
  if (!g.pending || g.failed)
    return;
  if (rules_late()) {
    {
      gfx::stagelog::Scope t{"rules_wait", g.pending->copy};
      g.pending->rules.wait();
    }
    if (!g.pending->drawable && !g.pending->postOnly) {
      Job& j = *g.pending;
      j.postOnly = true;
      j.sunColour[0] = j.sunColour[1] = j.sunColour[2] = 0.f;
      j.road = j.light = j.glow = j.calm = j.wet = 0;
      g.haveTerm = false;
    }
  }
  begin_access(g.inputMem, g.input);
  if (g.before)
    begin_access(g.beforeMem, g.before);
  {
    gfx::stagelog::Scope t{"first_submit"};
    webgpu::g_queue.Submit(1, &g.first);
  }
  g.first = {};
  end_access(g.inputMem, g.input);
  if (g.before)
    end_access(g.beforeMem, g.before);
  {
    gfx::stagelog::Scope t{"run_rays"};
    run_rays(*g.pending);
  }
  g.beforeThis = false;
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
bool ready() noexcept { return g.ready; }
void start_compile() noexcept {
  std::thread(init_device).detach();
}
bool look_on() noexcept { return post_on(); }
void name_thread() noexcept { SetThreadDescription(GetCurrentThread(), L"Aurora RT"); }
} // namespace aurora::gx::rt::live
