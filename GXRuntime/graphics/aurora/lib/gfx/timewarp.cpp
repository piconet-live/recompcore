// Positional (depth) timewarp. True frames are presented as ever, at once; a warped picture only fills the time after
// one and gives way to the next. The game thread does nothing for it.
//
// The camera. The game bakes its view into every modelview (P * (V * M)), so Aurora never sees V. It does not need to:
// a static object drawn in frames N-1 and N has MV(N) * MV(N-1)^-1 = V(N) * V(N-1)^-1, the camera's move in view space,
// exactly the picture's own (stage 5's C = V_new * V_old^-1, from the same stream as the frame instead of from game
// memory at another moment). Objects are recognised by their draw's first bytes and arrays; moving ones (machines) give
// other answers, so the answers vote, weighted by vertices. The guess: the same move again, a fraction of it.
//
// The depth. GX perspective clip z is p4 z + p5 over w = -z, and Aurora negates it for reversed Z, so the depth test's
// value is p4 - p5 / -z: the inverse distance is linear in depth, iz = (p4 - d) / p5, with d undone from the viewport's
// range. Under MSAA sample 0 is taken (Aurora's own snapshot); the resolved colour on an edge is a mix anyway.
//
// The warp. For each shown pixel a ray in the guessed view maps to source view linearly in its inverse distance s:
// h = R dir + s T. The search walks s from near to far in steps of at most 8 source pixels, to the first place the ray
// is behind the stored surface, and bisects. A crossing where the ray is far behind the surface it reached is no
// surface: the ray passed behind a near edge, a gap, filled from the farther side.

#include "timewarp.hpp"

#include "../webgpu/gpu.hpp"
#include "../window.hpp"
#include "../gx/gx.hpp"
#include "recording.hpp"
#include "render_worker.hpp"
#include "tex_copy_conv.hpp"

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace aurora::gfx::timewarp {
bool g_on = false;

namespace {
using Clock = std::chrono::steady_clock;
using webgpu::g_device;
using webgpu::g_queue;

constexpr double TrueFrameMs = 1000.0 / 59.94;
// A camera that moved further than this in a step is a cut: no warps from that frame.
constexpr double MostTurn = 0.35, MostMove = 60.0;

struct Rigid {
  double r[3][3];
  double t[3];
};
struct Sample {
  u64 key;
  std::array<float, 12> mv;
  float weight;
  std::array<float, 6> proj;
  gfx::Viewport viewport;
};

// --- the camera, from the stream: the FIFO processor's during a frame, Aurora's thread's once it has drained
std::vector<Sample> s_now, s_before;
std::array<float, 12> s_lastMv{};
int s_rejected = 0;

Rigid inverse_affine(const std::array<float, 12>& m, bool& ok) {
  // rows: x' = m[0] x + m[1] y + m[2] z + m[3] ...; the 3x3 may hold a scale
  const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
  const double A = e * i - f * h, B = c * h - b * i, C = b * f - c * e, D = f * g - d * i, E = a * i - c * g,
               F = c * d - a * f, G = d * h - e * g, H = b * g - a * h, I = a * e - b * d;
  const double det = a * A + b * D + c * G;
  Rigid out{};
  ok = std::abs(det) > 1e-9;
  if (!ok)
    return out;
  const double in[3][3] = {{A, B, C}, {D, E, F}, {G, H, I}};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k)
      out.r[r][k] = in[r][k] / det;
    out.t[r] = -(out.r[r][0] * m[3] + out.r[r][1] * m[7] + out.r[r][2] * m[11]);
  }
  return out;
}

Rigid correction(const std::array<float, 12>& now, const std::array<float, 12>& before, bool& ok) {
  const Rigid inv = inverse_affine(before, ok);
  Rigid c{};
  if (!ok)
    return c;
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k)
      c.r[r][k] = now[r * 4] * inv.r[0][k] + now[r * 4 + 1] * inv.r[1][k] + now[r * 4 + 2] * inv.r[2][k];
    c.t[r] = now[r * 4] * inv.t[0] + now[r * 4 + 1] * inv.t[1] + now[r * 4 + 2] * inv.t[2] + now[r * 4 + 3];
  }
  // a camera's move is rigid; anything else is an object that changed scale or shape
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      const double dot = c.r[0][a] * c.r[0][b] + c.r[1][a] * c.r[1][b] + c.r[2][a] * c.r[2][b];
      if (std::abs(dot - (a == b ? 1.0 : 0.0)) > 0.01)
        ok = false;
    }
  return c;
}

double turn_of(const Rigid& c) {
  return std::acos(std::clamp((c.r[0][0] + c.r[1][1] + c.r[2][2] - 1.0) * 0.5, -1.0, 1.0));
}

struct Solved {
  bool valid = false;
  Rigid c{};
  std::array<float, 6> proj{};
  gfx::Viewport viewport{};
  int agree = 0, candidates = 0;
  double share = 0;
};

Solved solve_camera() {
  struct Candidate {
    Rigid c;
    float weight;
    const Sample* sample;
  };
  static absl::flat_hash_map<u64, int> before;
  static std::vector<Candidate> cand;
  before.clear();
  cand.clear();
  for (size_t i = 0; i < s_before.size(); ++i) {
    auto [it, fresh] = before.try_emplace(s_before[i].key, static_cast<int>(i));
    if (!fresh && it->second >= 0 && s_before[it->second].mv != s_before[i].mv)
      it->second = -1;  // the same model at two places: no telling which is which
  }
  static absl::flat_hash_map<u64, int> now;
  now.clear();
  for (size_t i = 0; i < s_now.size(); ++i) {
    auto [it, fresh] = now.try_emplace(s_now[i].key, static_cast<int>(i));
    if (!fresh && it->second >= 0 && s_now[it->second].mv != s_now[i].mv)
      it->second = -1;
  }
  for (const auto& [key, at] : now) {
    if (at < 0)
      continue;
    const auto was = before.find(key);
    if (was == before.end() || was->second < 0)
      continue;
    bool ok = false;
    const Rigid c = correction(s_now[at].mv, s_before[was->second].mv, ok);
    if (ok)
      cand.push_back({c, s_now[at].weight, &s_now[at]});
  }
  Solved out;
  out.candidates = static_cast<int>(cand.size());
  if (cand.size() < 3)
    return out;
  std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.weight > b.weight; });
  if (cand.size() > 96)
    cand.resize(96);
  const auto apply = [](const Rigid& c, const double p[3], double o[3]) {
    for (int r = 0; r < 3; ++r)
      o[r] = c.r[r][0] * p[0] + c.r[r][1] * p[1] + c.r[r][2] * p[2] + c.t[r];
  };
  static constexpr double Points[3][3] = {{0, 0, -20}, {20, 0, -60}, {0, 20, -100}};
  double total = 0, best = -1;
  int bestAt = 0, bestCount = 0;
  for (size_t i = 0; i < cand.size(); ++i) {
    total += cand[i].weight;
    double vote = 0;
    int count = 0;
    for (const auto& other : cand) {
      bool same = true;
      for (const auto& p : Points) {
        double a[3], b[3];
        apply(cand[i].c, p, a);
        apply(other.c, p, b);
        same = same && std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2]) < 0.1;
      }
      if (same) {
        vote += other.weight;
        ++count;
      }
    }
    if (vote > best) {
      best = vote;
      bestAt = static_cast<int>(i);
      bestCount = count;
    }
  }
  out.agree = bestCount;
  out.share = total > 0 ? best / total : 0;
  const auto& winner = cand[bestAt];
  const double move = std::sqrt(winner.c.t[0] * winner.c.t[0] + winner.c.t[1] * winner.c.t[1] + winner.c.t[2] * winner.c.t[2]);
  out.c = winner.c;
  out.proj = winner.sample->proj;
  out.viewport = winner.sample->viewport;
  out.valid = bestCount >= 3 && out.share >= 0.25 && turn_of(winner.c) <= MostTurn && move <= MostMove;
  return out;
}

// --- the warp, all on the render worker
constexpr std::string_view Shader = R"(
struct U {
  r0: vec4f, r1: vec4f, r2: vec4f,  // source view from shown view: R in xyz, T in w
  proj: vec4f,                      // p0 p1 p2 p3
  z: vec4f,                         // p4 p5, the viewport's lowest depth and its range
  vp: vec4f,                        // the game's viewport in frame buffer pixels: x y w h
  rect: vec4f,                      // the display copy's source in frame buffer pixels: x y w h
  present: vec4f,                   // where the picture lies on the surface
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var color: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));
  return vec4f(p[i], 0.0, 1.0);
}

fn source_pixel(h0: vec3f) -> vec2f {
  let z = min(h0.z, -1e-5);
  let n = vec2f((u.proj.x * h0.x + u.proj.y * z) / -z, (u.proj.z * h0.y + u.proj.w * z) / -z);
  let q = u.vp.xy + vec2f(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5) * u.vp.zw;
  return clamp(q, u.rect.xy, u.rect.xy + u.rect.zw - vec2f(1.0));
}
fn surface_iz(q: vec2f) -> f32 {
  let d = textureLoad(depth, vec2i(q), 0).r;
  return max((u.z.x - (d - u.z.z) / u.z.w) / u.z.y, 0.0);
}
fn show(q: vec2f) -> vec4f {
  return vec4f(textureSampleLevel(color, samp, (q - u.rect.xy) / u.rect.zw, 0.0).rgb, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = (pos.xy - u.present.xy) / u.present.zw;
  let fb = u.rect.xy + uv * u.rect.zw;
  let n = (fb - u.vp.xy) / u.vp.zw * 2.0 - vec2f(1.0);
  let dir = vec3f((n.x + u.proj.y) / u.proj.x, (-n.y + u.proj.w) / u.proj.z, -1.0);
  let a = vec3f(dot(u.r0.xyz, dir), dot(u.r1.xyz, dir), dot(u.r2.xyz, dir));
  let b = vec3f(u.r0.w, u.r1.w, u.r2.w);
  let far_q = source_pixel(a);
  let near_s = max((u.z.x - 1.0) / u.z.y, 1e-4);
  // the source pixel per unit of s at far distances: the search's reach is 160 pixels at that rate (the true curve
  // bends towards the point of expansion close to the camera, so it cannot be measured at the near plane)
  let rate = length(source_pixel(a + b * 1e-3) - far_q) * 1e3;
  let s_most = min(near_s, 160.0 / max(rate, 1.0));
  let steps = i32(clamp(ceil(s_most * rate / 8.0), 2.0, 20.0));
  var lo = s_most;
  var hi = s_most;
  var found = false;
  for (var i = 0; i <= steps; i++) {
    let s = s_most * (1.0 - f32(i) / f32(steps));
    let h = a + b * s;
    if (s / -min(h.z, -1e-5) <= surface_iz(source_pixel(h))) {
      hi = s;
      found = true;
      break;
    }
    lo = s;
  }
  if (!found) { return show(far_q); }
  if (lo == hi) { return show(source_pixel(a + b * hi)); }  // nearer than the search reaches
  for (var i = 0; i < 4; i++) {
    let s = (lo + hi) * 0.5;
    let h = a + b * s;
    if (s / -min(h.z, -1e-5) <= surface_iz(source_pixel(h))) { hi = s; } else { lo = s; }
  }
  let h = a + b * hi;
  let q = source_pixel(h);
  let surface = surface_iz(q);
  let ray = hi / -min(h.z, -1e-5);
  if (surface - ray <= 0.15 * surface + 1e-5) { return show(q); }
  // a gap: the ray went behind a near edge; the farther surface fills it, a little away from the blended edge
  let back = source_pixel(a + b * lo);
  let away = back - q;
  let l = length(away);
  return show(back + select(vec2f(0.0), away / l * 1.5, l > 0.001));
}
)";

struct Uniform {
  float r[3][4];
  float proj[4];
  float z[4];
  float vp[4];
  float rect[4];
  float present[4];
};

wgpu::RenderPipeline s_pipeline;
wgpu::Buffer s_uniform;
wgpu::Sampler s_sampler;
wgpu::Texture s_depth;
wgpu::TextureView s_depthView;
wgpu::Extent3D s_depthSize{};
wgpu::TextureFormat s_targetFormat = wgpu::TextureFormat::Undefined;
wgpu::QuerySet s_queries;
wgpu::Buffer s_queryResolve, s_queryRead;
std::atomic<int> s_queryState{0};  // 0 free, 1 mapping, 2 mapped

// the frame on its way, then the frame shown
struct Source {
  wgpu::Texture colorTexture;
  wgpu::TextureView color;
  wgpu::Extent3D colorSize{};
  float rect[4]{};
  bool valid = false;
  Rigid c{};
  std::array<float, 6> proj{};
  gfx::Viewport viewport{};
};
Source s_coming, s_shown;
wgpu::BindGroup s_bind;
wgpu::TextureView s_bindColor, s_bindDepth;

std::atomic<bool> s_running{false};
std::thread s_pacer;
std::atomic<bool> s_trueComing{false};
std::atomic<int64_t> s_trueNs{0};
std::atomic<int> s_nextWarp{1};
int s_warpsAFrame = 3;
int64_t ns_now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

// the switches for tests: AURORA_TIMEWARP_SHOT=<prefix>, AURORA_TIMEWARP_SHOT_AT=<true frame>
std::string s_shotPrefix;
uint64_t s_shotAt = 0;
uint64_t s_trueFrames = 0;

// what the log says
std::atomic<uint64_t> s_warps{0}, s_warpsDropped{0};
uint64_t s_validFrames = 0, s_lastLogFrame = 0, s_warpsAtLog = 0, s_droppedAtLog = 0;
int64_t s_lastTrueNs = 0;
int64_t s_logStartNs = 0;
std::vector<double> s_trueGaps;
double s_warpGpuUs = 0, s_depthGpuUs = 0;
double s_warpCpuUs = 0;
double s_doneUs = 0;
Solved s_lastSolved;
struct TableRow {
  std::array<float, 6> p;
  float zn, zf;
  int count;
};
std::vector<TableRow> s_shotTable;

void make_resources() {
  if (s_pipeline)
    return;
  s_targetFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const std::string source{Shader};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = source.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDesc{.nextInChain = &wgsl, .label = "Timewarp"};
  const auto module = g_device.CreateShaderModule(&moduleDesc);
  const std::array targets{wgpu::ColorTargetState{.format = s_targetFormat}};
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor desc{
      .label = "Timewarp",
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  s_pipeline = g_device.CreateRenderPipeline(&desc);
  const wgpu::BufferDescriptor uniformDesc{
      .label = "Timewarp uniform",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
      .size = sizeof(Uniform),
  };
  s_uniform = g_device.CreateBuffer(&uniformDesc);
  const wgpu::SamplerDescriptor samplerDesc{
      .label = "Timewarp sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  s_sampler = g_device.CreateSampler(&samplerDesc);
  if (g_device.HasFeature(wgpu::FeatureName::TimestampQuery)) {
    const wgpu::QuerySetDescriptor queryDesc{.type = wgpu::QueryType::Timestamp, .count = 4};
    s_queries = g_device.CreateQuerySet(&queryDesc);
    std::fprintf(stderr, "[timewarp] timestamp queries available\n");
    const wgpu::BufferDescriptor resolveDesc{
        .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
        .size = 32,
    };
    s_queryResolve = g_device.CreateBuffer(&resolveDesc);
    const wgpu::BufferDescriptor readDesc{
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
        .size = 32,
    };
    s_queryRead = g_device.CreateBuffer(&readDesc);
  }
}

// Rotation by a fraction of the move's angle (Rodrigues), the same fraction of its translation; then the inverse:
// from the shown (guessed) view to the source view.
void fill_transform(const Rigid& c, double f, float out[3][4]) {
  const double angle = turn_of(c);
  double axis[3] = {c.r[2][1] - c.r[1][2], c.r[0][2] - c.r[2][0], c.r[1][0] - c.r[0][1]};
  const double len = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  double r[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  if (len > 1e-9) {
    for (double& v : axis)
      v /= len;
    const double th = angle * f, s = std::sin(th), k = 1.0 - std::cos(th);
    const double x = axis[0], y = axis[1], z = axis[2];
    const double m[3][3] = {{1 - k * (y * y + z * z), -s * z + k * x * y, s * y + k * x * z},
                            {s * z + k * x * y, 1 - k * (x * x + z * z), -s * x + k * y * z},
                            {-s * y + k * x * z, s * x + k * y * z, 1 - k * (x * x + y * y)}};
    std::memcpy(r, m, sizeof r);
  }
  const double t[3] = {c.t[0] * f, c.t[1] * f, c.t[2] * f};
  for (int row = 0; row < 3; ++row) {  // R^T, -R^T t
    for (int col = 0; col < 3; ++col)
      out[row][col] = static_cast<float>(r[col][row]);
    out[row][3] = static_cast<float>(-(r[0][row] * t[0] + r[1][row] * t[1] + r[2][row] * t[2]));
  }
}

void draw_warp(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target, const webgpu::Viewport& present,
               double f, bool timed) {
  Uniform u{};
  fill_transform(s_shown.c, f, u.r);
  const auto& p = s_shown.proj;
  const auto& v = s_shown.viewport;
  const float lowest = 1.f - v.zfar, highest = 1.f - v.znear;
  const float z[4] = {p[4], p[5], lowest, std::max(highest - lowest, 1e-6f)};
  const float proj[4] = {p[0], p[1], p[2], p[3]};
  const float vp[4] = {v.left, v.top, v.width, v.height};
  const float pr[4] = {present.left, present.top, present.width, present.height};
  std::memcpy(u.proj, proj, sizeof proj);
  std::memcpy(u.z, z, sizeof z);
  std::memcpy(u.vp, vp, sizeof vp);
  std::memcpy(u.rect, s_shown.rect, sizeof u.rect);
  std::memcpy(u.present, pr, sizeof pr);
  g_queue.WriteBuffer(s_uniform, 0, &u, sizeof u);
  if (!s_bind || s_bindColor.Get() != s_shown.color.Get() || s_bindDepth.Get() != s_depthView.Get()) {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = s_uniform, .size = sizeof(Uniform)},
        wgpu::BindGroupEntry{.binding = 1, .sampler = s_sampler},
        wgpu::BindGroupEntry{.binding = 2, .textureView = s_shown.color},
        wgpu::BindGroupEntry{.binding = 3, .textureView = s_depthView},
    };
    const wgpu::BindGroupDescriptor desc{
        .layout = s_pipeline.GetBindGroupLayout(0),
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    s_bind = g_device.CreateBindGroup(&desc);
    s_bindColor = s_shown.color;
    s_bindDepth = s_depthView;
  }
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
  }};
  const wgpu::PassTimestampWrites writes{.querySet = s_queries, .beginningOfPassWriteIndex = 0, .endOfPassWriteIndex = 1};
  const wgpu::RenderPassDescriptor passDesc{
      .label = "Timewarp pass",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = timed && s_queries ? &writes : nullptr,
  };
  const auto pass = encoder.BeginRenderPass(&passDesc);
  pass.SetPipeline(s_pipeline);
  pass.SetBindGroup(0, s_bind);
  pass.SetViewport(present.left, present.top, present.width, present.height, 0.f, 1.f);
  pass.Draw(3);
  pass.End();
}

// Render worker. Gives way to a true frame: nothing is drawn once that frame has been handed over.
void warp_job(int64_t due) {
  if (!s_shown.valid || !s_pipeline || s_trueComing.load(std::memory_order_acquire) || !webgpu::g_surface) {
    s_warpsDropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const int64_t started = ns_now();
  if (started - due > 2'500'000) {  // too late for its moment: the next one is nearly due
    s_warpsDropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  wgpu::Texture surfaceTexture;
  {
    window::SurfaceLock lock;
    if (!window::is_presentable())
      return;
    wgpu::SurfaceTexture st;
    webgpu::g_surface.GetCurrentTexture(&st);
    if (st.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
      s_warpsDropped.fetch_add(1, std::memory_order_relaxed);
      return;  // the true frame's present deals with the surface
    }
    surfaceTexture = std::move(st.texture);
  }
  const auto& config = webgpu::g_graphicsConfig.surfaceConfiguration;
  const auto present = webgpu::calculate_present_viewport(config.width, config.height, s_shown.colorSize.width,
                                                          s_shown.colorSize.height);
  const double f = std::clamp(static_cast<double>(started - s_trueNs.load(std::memory_order_relaxed)) / 1e6 / TrueFrameMs,
                              0.0, 1.25);
  static uint64_t count = 0;
  const bool timed = s_queries && ++count % 120 == 0 && s_queryState.load(std::memory_order_acquire) == 0;
  const auto encoder = g_device.CreateCommandEncoder();
  draw_warp(encoder, surfaceTexture.CreateView(), present, f, timed);
  if (timed) {
    encoder.ResolveQuerySet(s_queries, 0, 4, s_queryResolve, 0);
    encoder.CopyBufferToBuffer(s_queryResolve, 0, s_queryRead, 0, 32);
  }
  const auto buffer = encoder.Finish();
  g_queue.Submit(1, &buffer);
  if (count % 120 == 0) {
    const int64_t submitted = ns_now();
    std::atomic<int64_t> finished{0};
    g_queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                [&finished](wgpu::QueueWorkDoneStatus, wgpu::StringView) { finished.store(ns_now()); });
    for (int i = 0; i < 2000 && finished.load() == 0; ++i)
      webgpu::g_instance.ProcessEvents();
    if (finished.load() != 0)
      s_doneUs = static_cast<double>(finished.load() - submitted) / 1e3;
  }
  if (timed) {
    s_queryState.store(1, std::memory_order_release);
    s_queryRead.MapAsync(wgpu::MapMode::Read, 0, 32, wgpu::CallbackMode::AllowSpontaneous,
                         [](wgpu::MapAsyncStatus status, wgpu::StringView) {
                           s_queryState.store(status == wgpu::MapAsyncStatus::Success ? 2 : 0,
                                              std::memory_order_release);
                         });
  }
  {
    window::SurfaceLock lock;
    if (window::is_presentable())
      (void)webgpu::g_surface.Present();
  }
  s_warps.fetch_add(1, std::memory_order_relaxed);
  s_warpCpuUs += 0.05 * (static_cast<double>(ns_now() - started) / 1e3 - s_warpCpuUs);
}

// Test pictures: one warp of the frame at a fraction, into a file of raw pixels (w, h as 32-bit words, then 4-byte
// pixels in the surface's format), through the same shader. Fraction 0 is the true frame through the warp.
void shoot(const char* name, double f) {
  const auto& config = webgpu::g_graphicsConfig.surfaceConfiguration;
  const uint32_t w = config.width, h = config.height;
  const wgpu::TextureDescriptor desc{
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .size = {w, h, 1},
      .format = s_targetFormat,
  };
  const auto target = g_device.CreateTexture(&desc);
  const uint32_t rowBytes = (w * 4u + 255u) & ~255u;
  const wgpu::BufferDescriptor bufDesc{
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = static_cast<uint64_t>(rowBytes) * h,
  };
  const auto buffer = g_device.CreateBuffer(&bufDesc);
  const auto present = webgpu::calculate_present_viewport(w, h, s_shown.colorSize.width, s_shown.colorSize.height);
  const auto encoder = g_device.CreateCommandEncoder();
  draw_warp(encoder, target.CreateView(), present, f, false);
  const wgpu::TexelCopyTextureInfo src{.texture = target};
  const wgpu::TexelCopyBufferInfo dst{.layout = {.bytesPerRow = rowBytes, .rowsPerImage = h}, .buffer = buffer};
  const wgpu::Extent3D size{w, h, 1};
  encoder.CopyTextureToBuffer(&src, &dst, &size);
  const auto commands = encoder.Finish();
  g_queue.Submit(1, &commands);
  std::atomic<int> done{0};
  buffer.MapAsync(wgpu::MapMode::Read, 0, bufDesc.size, wgpu::CallbackMode::AllowSpontaneous,
                  [&done](wgpu::MapAsyncStatus status, wgpu::StringView) {
                    done.store(status == wgpu::MapAsyncStatus::Success ? 1 : 2);
                  });
  for (int i = 0; i < 4000 && done.load() == 0; ++i) {
    webgpu::g_instance.ProcessEvents();
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  if (done.load() == 1) {
    const std::string path = s_shotPrefix + name + ".raw";
    if (FILE* out = std::fopen(path.c_str(), "wb")) {
      std::fwrite(&w, 4, 1, out);
      std::fwrite(&h, 4, 1, out);
      const auto* mapped = static_cast<const u8*>(buffer.GetConstMappedRange(0, bufDesc.size));
      for (uint32_t y = 0; y < h; ++y)
        std::fwrite(mapped + static_cast<size_t>(y) * rowBytes, 1, w * 4u, out);
      std::fclose(out);
    }
    buffer.Unmap();
  }
  std::fprintf(stderr, "[timewarp] shot %s, fraction %.2f: %s\n", name, f, done.load() == 1 ? "written" : "failed");
}

void shoot_depth() {
  const uint32_t w = s_depthSize.width, h = s_depthSize.height;
  const uint32_t rowBytes = (w * 4u + 255u) & ~255u;
  const wgpu::BufferDescriptor bufDesc{
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = static_cast<uint64_t>(rowBytes) * h,
  };
  const auto buffer = g_device.CreateBuffer(&bufDesc);
  const auto encoder = g_device.CreateCommandEncoder();
  const wgpu::TexelCopyTextureInfo src{.texture = s_depth};
  const wgpu::TexelCopyBufferInfo dst{.layout = {.bytesPerRow = rowBytes, .rowsPerImage = h}, .buffer = buffer};
  const wgpu::Extent3D size{w, h, 1};
  encoder.CopyTextureToBuffer(&src, &dst, &size);
  const auto commands = encoder.Finish();
  g_queue.Submit(1, &commands);
  std::atomic<int> done{0};
  buffer.MapAsync(wgpu::MapMode::Read, 0, bufDesc.size, wgpu::CallbackMode::AllowSpontaneous,
                  [&done](wgpu::MapAsyncStatus status, wgpu::StringView) {
                    done.store(status == wgpu::MapAsyncStatus::Success ? 1 : 2);
                  });
  for (int i = 0; i < 4000 && done.load() == 0; ++i) {
    webgpu::g_instance.ProcessEvents();
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  if (done.load() == 1) {
    const std::string path = s_shotPrefix + "depth.raw";
    if (FILE* out = std::fopen(path.c_str(), "wb")) {
      std::fwrite(&w, 4, 1, out);
      std::fwrite(&h, 4, 1, out);
      const auto* mapped = static_cast<const u8*>(buffer.GetConstMappedRange(0, bufDesc.size));
      for (uint32_t y = 0; y < h; ++y)
        std::fwrite(mapped + static_cast<size_t>(y) * rowBytes, 1, w * 4u, out);
      std::fclose(out);
    }
    buffer.Unmap();
  }
  const auto& c = s_shown.c;
  const auto& p = s_shown.proj;
  const auto& v = s_shown.viewport;
  std::fprintf(stderr,
               "[timewarp] shot state: proj %g %g %g %g %g %g; viewport %g %g %g %g z %g %g; rect %g %g %g %g; depth %ux%u; "
               "C rows %.5f %.5f %.5f %.3f / %.5f %.5f %.5f %.3f / %.5f %.5f %.5f %.3f\n",
               p[0], p[1], p[2], p[3], p[4], p[5], v.left, v.top, v.width, v.height, v.znear, v.zfar, s_shown.rect[0],
               s_shown.rect[1], s_shown.rect[2], s_shown.rect[3], w, h, c.r[0][0], c.r[0][1], c.r[0][2], c.t[0], c.r[1][0],
               c.r[1][1], c.r[1][2], c.t[1], c.r[2][0], c.r[2][1], c.r[2][2], c.t[2]);
  std::fflush(stderr);
}

void pacer_main() {
#if defined(_WIN32)
  timeBeginPeriod(1);
  SetThreadDescription(GetCurrentThread(), L"Aurora timewarp");
#endif
  const double period = TrueFrameMs / (s_warpsAFrame + 1);
  while (s_running.load(std::memory_order_acquire)) {
    const int64_t base = s_trueNs.load(std::memory_order_acquire);
    const int index = s_nextWarp.load(std::memory_order_acquire);
    if (base == 0 || index > s_warpsAFrame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    const int64_t due = base + static_cast<int64_t>(index * period * 1e6);
    const int64_t left = due - ns_now();
    if (left > 1'500'000) {
      std::this_thread::sleep_for(std::chrono::microseconds(left / 1000 - 1200));
      continue;  // (a new true frame may have moved the moment)
    }
    if (left > 0) {
      std::this_thread::yield();
      continue;
    }
    int expected = index;
    if (!s_nextWarp.compare_exchange_strong(expected, index + 1))
      continue;
    if (s_trueNs.load(std::memory_order_acquire) != base)
      continue;
    render_worker::enqueue_work([due] { warp_job(due); });
  }
#if defined(_WIN32)
  timeEndPeriod(1);
#endif
}

double median_of(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// Render worker, after the true frame's present.
void true_frame_shown(Solved solved) {
  const int64_t now = ns_now();
  make_resources();
  s_shown.colorTexture = s_coming.colorTexture;
  s_shown.color = s_coming.color;
  s_shown.colorSize = s_coming.colorSize;
  std::memcpy(s_shown.rect, s_coming.rect, sizeof s_shown.rect);
  s_shown.c = solved.c;
  s_shown.proj = solved.proj;
  s_shown.viewport = solved.viewport;
  s_shown.valid = solved.valid && s_shown.color && s_depthView;
  ++s_trueFrames;
  s_validFrames += s_shown.valid ? 1 : 0;
  s_lastSolved = solved;
  if (s_lastTrueNs != 0)
    s_trueGaps.push_back(static_cast<double>(now - s_lastTrueNs) / 1e6);
  s_lastTrueNs = now;
  s_trueNs.store(now, std::memory_order_release);
  s_nextWarp.store(1, std::memory_order_release);
  s_trueComing.store(false, std::memory_order_release);

  if (s_queryState.load(std::memory_order_acquire) == 2) {
    const auto* t = static_cast<const uint64_t*>(s_queryRead.GetConstMappedRange(0, 32));
    if (t[1] > t[0])
      s_warpGpuUs = static_cast<double>(t[1] - t[0]) / 1e3;
    if (t[3] > t[2])
      s_depthGpuUs = static_cast<double>(t[3] - t[2]) / 1e3;
    s_queryRead.Unmap();
    s_queryState.store(0, std::memory_order_release);
  }
  if (!s_shotPrefix.empty() && s_shown.valid) {
    if (s_trueFrames == s_shotAt) {
      shoot_depth();
      {
        // every projection and depth range the frame's main view used, with how many models
        for (const auto& t : s_shotTable)
          std::fprintf(stderr, "[timewarp] table: %d models, proj %g %g %g %g %g %g, z %g %g\n", t.count, t.p[0], t.p[1],
                       t.p[2], t.p[3], t.p[4], t.p[5], t.zn, t.zf);
      }
      shoot("true", 0.0);
      shoot("warp-025", 0.25);
      shoot("warp-050", 0.5);
      shoot("warp-075", 0.75);
      shoot("warp-100", 1.0);
    } else if (s_trueFrames == s_shotAt + 1) {
      shoot("next-true", 0.0);
    }
  }
  if (s_trueFrames - s_lastLogFrame >= 600) {
    const auto warps = s_warps.load(), dropped = s_warpsDropped.load();
    const double seconds = static_cast<double>(now - s_logStartNs) / 1e9;
    std::fprintf(stderr,
                 "[timewarp] true frame %llu: %.1f pictures/s (%.1f true, %.1f warped); warps %llu, gave way %llu; "
                 "frames with a camera %llu of %llu; true gap median %.2f ms; warp submit to done %.0f us, warp GPU %.0f us, depth copy GPU %.0f us, worker %.0f us; "
                 "last camera: %d of %d agree, %.0f%%, turn %.4f, move %.2f\n",
                 static_cast<unsigned long long>(s_trueFrames),
                 (static_cast<double>(s_trueFrames - s_lastLogFrame) + static_cast<double>(warps - s_warpsAtLog)) / seconds,
                 static_cast<double>(s_trueFrames - s_lastLogFrame) / seconds,
                 static_cast<double>(warps - s_warpsAtLog) / seconds, static_cast<unsigned long long>(warps - s_warpsAtLog),
                 static_cast<unsigned long long>(dropped - s_droppedAtLog),
                 static_cast<unsigned long long>(s_validFrames), static_cast<unsigned long long>(s_trueFrames - s_lastLogFrame),
                 median_of(s_trueGaps), s_doneUs, s_warpGpuUs, s_depthGpuUs, s_warpCpuUs, solved.agree, solved.candidates, solved.share * 100.0,
                 turn_of(solved.c), std::sqrt(solved.c.t[0] * solved.c.t[0] + solved.c.t[1] * solved.c.t[1] + solved.c.t[2] * solved.c.t[2]));
    std::fflush(stderr);
    s_lastLogFrame = s_trueFrames;
    s_warpsAtLog = warps;
    s_droppedAtLog = dropped;
    s_logStartNs = now;
    s_validFrames = 0;
    s_trueGaps.clear();
  }
}
} // namespace

void start() {
  const char* v = std::getenv("AURORA_TIMEWARP");
  if (v == nullptr || v[0] != '1')
    return;
  const char* hz = std::getenv("AURORA_TIMEWARP_HZ");
  const double rate = hz != nullptr ? std::atof(hz) : 240.0;
  s_warpsAFrame = std::clamp(static_cast<int>(std::lround(rate / 59.94)) - 1, 1, 7);
  if (const char* shot = std::getenv("AURORA_TIMEWARP_SHOT"))
    s_shotPrefix = shot;
  if (const char* at = std::getenv("AURORA_TIMEWARP_SHOT_AT"))
    s_shotAt = std::strtoull(at, nullptr, 10);
  s_now.reserve(4096);
  s_before.reserve(4096);
  s_logStartNs = ns_now();
  g_on = true;
  s_running.store(true, std::memory_order_release);
  s_pacer = std::thread(pacer_main);
  std::fprintf(stderr, "[timewarp] on: %d warps a true frame\n", s_warpsAFrame);
}

void stop() {
  if (!g_on)
    return;
  s_running.store(false, std::memory_order_release);
  if (s_pacer.joinable())
    s_pacer.join();
  render_worker::synchronize();
  g_on = false;
  // (GPU objects go before Aurora's device does)
  render_worker::enqueue_work([] {
    s_shown = {};
    s_coming = {};
    s_bind = {};
    s_bindColor = {};
    s_bindDepth = {};
    s_depth = {};
    s_depthView = {};
    s_pipeline = {};
    s_uniform = {};
    s_sampler = {};
    s_queries = {};
    s_queryResolve = {};
    s_queryRead = {};
  });
  render_worker::synchronize();
}

void note_draw(uint8_t cmd, uint16_t vtxCount, const uint8_t* vertices, size_t available) noexcept {
  const auto& state = gx::g_gxState;
  if (state.projType != GX_PERSPECTIVE || vtxCount < 3 || s_now.size() >= 8192)
    return;
  const auto& pos = state.pnMtx[std::min<u32>(state.currentPnMtx, gx::MaxPnMtx - 1)].pos;
  std::array<float, 12> mv;
  std::memcpy(mv.data(), &pos, sizeof mv);
  if (mv == s_lastMv)
    return;
  const auto& vp = state.renderViewport;
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  // only the main view: the shadows, faces and emblems are drawn in small viewports
  if (vp.width < 0.8f * static_cast<float>(targetWidth) || vp.height < 0.5f * static_cast<float>(targetHeight))
    return;
  s_lastMv = mv;
  // the model: its vertex data (for a model, indices), its arrays
  u64 key = 0xcbf29ce484222325ull;
  const auto mix = [&key](u64 value) { key = (key ^ value) * 0x100000001b3ull; };
  mix(cmd);
  mix(vtxCount);
  mix(reinterpret_cast<uintptr_t>(state.arrays[GX_VA_POS].data));
  const size_t bytes = std::min<size_t>(available, 32);
  for (size_t i = 0; i < bytes; ++i)
    mix(vertices[i]);
  const auto& proj = state.proj;
  s_now.push_back({
      .key = key,
      .mv = mv,
      .weight = static_cast<float>(vtxCount),
      .proj = {proj.m0[0], proj.m0[2], proj.m1[1], proj.m1[2], proj.m2[2], proj.m2[3]},
      .viewport = vp,
  });
}

void capture(const wgpu::CommandEncoder& encoder, const wgpu::Texture& colorTexture, const wgpu::TextureView& colorView,
             wgpu::Extent3D colorSize, const wgpu::TextureView& depthView, uint32_t msaaSamples, wgpu::Extent3D fbSize,
             int32_t rectX, int32_t rectY, int32_t rectWidth, int32_t rectHeight) {
  if (!s_depth || s_depthSize.width != fbSize.width || s_depthSize.height != fbSize.height) {
    const wgpu::TextureDescriptor desc{
        .label = "Timewarp depth",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .size = {fbSize.width, fbSize.height, 1},
        .format = wgpu::TextureFormat::R32Float,
    };
    s_depth = g_device.CreateTexture(&desc);
    s_depthView = s_depth.CreateView();
    s_depthSize = fbSize;
    s_shown.valid = false;
  }
  if (s_queries)
    encoder.WriteTimestamp(s_queries, 2);
  tex_copy_conv::snapshot_depth(encoder, depthView, msaaSamples, s_depthView);
  if (s_queries)
    encoder.WriteTimestamp(s_queries, 3);
  s_coming.colorTexture = colorTexture;
  s_coming.color = colorView;
  s_coming.colorSize = colorSize;
  s_coming.rect[0] = static_cast<float>(rectX);
  s_coming.rect[1] = static_cast<float>(rectY);
  s_coming.rect[2] = static_cast<float>(rectWidth);
  s_coming.rect[3] = static_cast<float>(rectHeight);
}

void true_frame_coming() noexcept {}

void true_frame_done() {
  if (!g_on)
    return;
  // the frame has drained and its end is with the worker: warps not yet begun give way
  s_trueComing.store(true, std::memory_order_release);
  // the frame's samples are this thread's now
  Solved solved = solve_camera();
  if (!s_shotPrefix.empty()) {
    std::vector<TableRow> table;
    for (const auto& one : s_now) {
      auto it = std::find_if(table.begin(), table.end(), [&](const TableRow& t) {
        return t.p == one.proj && t.zn == one.viewport.znear && t.zf == one.viewport.zfar;
      });
      if (it == table.end())
        table.push_back({one.proj, one.viewport.znear, one.viewport.zfar, 1});
      else
        ++it->count;
    }
    render_worker::enqueue_work([table = std::move(table)]() mutable { s_shotTable = std::move(table); });
  }
  s_before.swap(s_now);
  s_now.clear();
  s_lastMv = {};
  render_worker::enqueue_work([solved] { true_frame_shown(solved); });
}
} // namespace aurora::gfx::timewarp
