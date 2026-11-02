#include "hdr.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu.hpp"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <SDL3/SDL.h>
#include "../window.hpp"
#endif

namespace aurora::webgpu::hdr {
namespace {
constexpr float ScrgbWhiteNits = 80.f;  // scRGB's definition
// Used only if the display doesn't report them: tuned, not measured
constexpr float PaperNitsIfUnknown = 200.f;
constexpr float PeakNitsIfUnknown = 1000.f;

bool g_on = false;
float g_paperNits = 0.f;
float g_peakNits = 0.f;
wgpu::RenderPipeline g_pipeline;
wgpu::BindGroupLayout g_layout;
wgpu::Texture g_texture;
wgpu::TextureView g_view;
wgpu::BindGroup g_bind;

bool env_on(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1';
}
float env_float(const char* name, float otherwise) {
  const char* v = std::getenv(name);
  const float f = v != nullptr ? static_cast<float>(std::atof(v)) : 0.f;
  return f > 0.f ? f : otherwise;
}

struct Display {
  bool found = false;
  bool hdr = false;
  float peakNits = 0.f;
  float sdrWhiteNits = 0.f;
};

#ifdef _WIN32
// Windows's own "SDR content brightness" for this display, the level the desktop's white has
float sdr_white_nits(const wchar_t* gdiName) {
  UINT32 pathCount = 0, modeCount = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &pathCount, &modeCount) != ERROR_SUCCESS)
    return 0.f;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(pathCount);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(modeCount);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &pathCount, paths.data(), &modeCount, modes.data(), nullptr) != ERROR_SUCCESS)
    return 0.f;
  for (const auto& path : paths) {
    DISPLAYCONFIG_SOURCE_DEVICE_NAME name{};
    name.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof name, path.sourceInfo.adapterId, path.sourceInfo.id};
    if (DisplayConfigGetDeviceInfo(&name.header) != ERROR_SUCCESS || wcscmp(name.viewGdiDeviceName, gdiName) != 0)
      continue;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof white, path.targetInfo.adapterId, path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS)
      return static_cast<float>(white.SDRWhiteLevel) * ScrgbWhiteNits / 1000.f;
  }
  return 0.f;
}

// Every display is printed; the window's is returned
Display window_display() {
  Display out;
  using Microsoft::WRL::ComPtr;
  ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    return out;
  HWND hwnd = nullptr;
  if (SDL_Window* window = window::get_sdl_window())
    hwnd = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
  const HMONITOR monitor = hwnd != nullptr ? MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY) : nullptr;
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
    ComPtr<IDXGIOutput> output;
    for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
      ComPtr<IDXGIOutput6> output6;
      DXGI_OUTPUT_DESC1 desc{};
      if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)))
        continue;
      const bool hdr = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
      const float white = sdr_white_nits(desc.DeviceName);
      const bool mine = desc.Monitor == monitor;
      std::fprintf(stderr, "[hdr] display %ls: HDR %s, peak %.0f nits, full frame %.0f nits, Windows SDR white %.0f nits%s\n", desc.DeviceName,
                   hdr ? "on" : "off", desc.MaxLuminance, desc.MaxFullFrameLuminance, white, mine ? " (the window's)" : "");
      if (mine)
        out = {true, hdr, desc.MaxLuminance, white};
    }
  }
  return out;
}
#else
Display window_display() { return {}; }
#endif

void make_pipeline() {
  // Below the paper level exactly the SDR picture at that brightness; above, up to the display's peak on a curve with no corner, by the brightest channel so colour holds.
  const float scale = g_paperNits / ScrgbWhiteNits;
  const float room = std::max(g_peakNits / g_paperNits, 1.001f);
  std::string code = "const scale = " + std::to_string(scale) + "; const room = " + std::to_string(room) + ";\n" + R"(
@group(0) @binding(0) var from_: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}
fn linear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  var c = linear(max(textureLoad(from_, vec2i(p.xy), 0).rgb, vec3f(0.0)));
  let m = max(c.r, max(c.g, c.b));
  if (m > 1.0) {
    let over = room - 1.0;
    c *= (1.0 + over * (1.0 - exp(-(m - 1.0) / over))) / m;
  }
  return vec4f(c * scale, 1.0);
}
)";
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDesc{.nextInChain = &wgsl, .label = "HDR output"};
  const auto module = g_device.CreateShaderModule(&moduleDesc);
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                            .viewDimension = wgpu::TextureViewDimension::e2D}};
  const wgpu::BindGroupLayoutDescriptor layoutDesc{.entryCount = 1, .entries = &entry};
  g_layout = g_device.CreateBindGroupLayout(&layoutDesc);
  const wgpu::PipelineLayoutDescriptor pipelineLayout{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_layout};
  const wgpu::ColorTargetState target{.format = wgpu::TextureFormat::RGBA16Float, .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor pipeline{
      .label = "HDR output",
      .layout = g_device.CreatePipelineLayout(&pipelineLayout),
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs"},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  g_pipeline = g_device.CreateRenderPipeline(&pipeline);
}
} // namespace

void choose(const wgpu::SurfaceCapabilities& capabilities) {
  g_on = false;
  if (!env_on("AURORA_LOOK_HDR"))
    return;
  bool canFloat = false;
  std::string formats;
  for (size_t i = 0; i < capabilities.formatCount; ++i) {
    formats += std::to_string(static_cast<uint32_t>(capabilities.formats[i])) + " ";
    canFloat = canFloat || capabilities.formats[i] == wgpu::TextureFormat::RGBA16Float;
  }
  std::fprintf(stderr, "[hdr] surface formats (Dawn's numbers): %s; RGBA16Float (%u) %s\n", formats.c_str(),
               static_cast<uint32_t>(wgpu::TextureFormat::RGBA16Float), canFloat ? "offered" : "NOT offered");
  const Display display = window_display();
  const char* no = nullptr;
  const char* warp = std::getenv("AURORA_TIMEWARP");
  if (env_on("AURORA_HDR_TEST_SDR"))
    no = "a test (AURORA_HDR_TEST_SDR=1) says the display is not in HDR mode";
  else if (!display.found)
    no = "the window's display not found";
  else if (!display.hdr)
    no = "the window's display is not in HDR mode";
  else if (!canFloat)
    no = "the surface has no float format";
  else if (warp != nullptr && warp[0] == '1')
    no = "the warp (AURORA_TIMEWARP) only draws to the 8-bit surface";
  if (no != nullptr) {
    std::fprintf(stderr, "[hdr] AURORA_LOOK_HDR=1 asked, picture stays SDR: %s\n", no);
    return;
  }
  g_paperNits = env_float("AURORA_HDR_PAPER_NITS", display.sdrWhiteNits > 0.f ? display.sdrWhiteNits : PaperNitsIfUnknown);
  g_peakNits = env_float("AURORA_HDR_PEAK_NITS", display.peakNits > 0.f ? display.peakNits : PeakNitsIfUnknown);
  g_on = true;
  std::fprintf(stderr, "[hdr] on: scRGB float surface, paper white %.0f nits, highlights to %.0f nits%s\n", g_paperNits, g_peakNits,
               raw_picture() ? "" : "; no AURORA_LOOK_RAW=1, so no range above white to show");
}

bool on() noexcept { return g_on; }
bool float_copy() noexcept { return g_on || (raw_picture() && std::getenv("AURORA_LOOK_PEAKLOG") != nullptr); }

wgpu::TextureView target(uint32_t width, uint32_t height) {
  if (!g_pipeline)
    make_pipeline();
  if (!g_texture || g_texture.GetWidth() != width || g_texture.GetHeight() != height) {
    const wgpu::TextureDescriptor desc{
        .label = "HDR output's picture",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {width, height, 1},
        .format = wgpu::TextureFormat::RGBA16Float,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_texture = g_device.CreateTexture(&desc);
    g_view = g_texture.CreateView();
    const wgpu::BindGroupEntry entry{.binding = 0, .textureView = g_view};
    const wgpu::BindGroupDescriptor bind{.layout = g_layout, .entryCount = 1, .entries = &entry};
    g_bind = g_device.CreateBindGroup(&bind);
  }
  return g_view;
}

void final_pass(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& surface) {
  const wgpu::RenderPassColorAttachment attachment{.view = surface, .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store};
  const wgpu::RenderPassDescriptor desc{.label = "HDR output", .colorAttachmentCount = 1, .colorAttachments = &attachment};
  const auto pass = encoder.BeginRenderPass(&desc);
  pass.SetPipeline(g_pipeline);
  pass.SetBindGroup(0, g_bind, 0, nullptr);
  pass.Draw(3);
  pass.End();
}

void shutdown() noexcept {
  g_bind = {};
  g_view = {};
  g_texture = {};
  g_pipeline = {};
  g_layout = {};
}
} // namespace aurora::webgpu::hdr
