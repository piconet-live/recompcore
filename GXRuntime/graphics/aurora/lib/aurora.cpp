#include "gfx/stagelog.hpp"
#include <aurora/aurora.h>
#include <aurora/time.hpp>

#ifdef AURORA_ENABLE_GX
#include "gfx/efb_readback.hpp"
#include "gfx/resources.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/render_worker.hpp"
#include "gx/command_processor.hpp"
#include "gx/rt_capture.hpp"
#include "gx/redraw.hpp"
#include "gx/fifo.hpp"
#include "gx/gx.hpp"
#include "gx/texture.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/hdr.hpp"
#include "webgpu/gpu_prof.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "gamepad.hpp"
#include "internal.hpp"
#include "thread.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

#include <atomic>
#include <chrono>

// Frame timing for offline tools (dolgx_replay --timing): the steady-clock time the render worker submitted the
// last frame, and the time the GPU reported that submission done (queue OnSubmittedWorkDone).
static std::atomic<uint64_t> g_timingSubmitNs{0};
static std::atomic<uint64_t> g_timingGpuDoneNs{0};
static uint64_t timing_now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
extern "C" uint64_t aurora_timing_last_submit_ns() { return g_timingSubmitNs.load(std::memory_order_acquire); }
extern "C" uint64_t aurora_timing_last_gpu_done_ns() { return g_timingGpuDoneNs.load(std::memory_order_acquire); }
// a true frame's (not a picture between's) GPU done after its submit, smoothed (AURORA_PRESENT_OVERLAP times pictures from it)
static std::atomic<uint64_t> g_timingTrueLagNs{0};
extern "C" uint64_t aurora_timing_true_gpu_lag_ns() { return g_timingTrueLagNs.load(std::memory_order_acquire); }

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
constexpr Module Log{"aurora"};

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  AURORA_ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  AURORA_ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  AURORA_ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  AURORA_ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  AURORA_ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();
  thread::set_current({
      .name = "Main thread",
      .affinity = thread::Affinity::SharedCache,
  });

#ifdef AURORA_ENABLE_GX
  gfx::initialize();
  gx::fifo::init();
  gx::rt::start();
  gx::redraw::start();
  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gx::fifo::shutdown();
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  gamepad::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    if (!g_config.embedded) {
      gamepad::initialize();
    }
  }
#ifdef AURORA_ENABLE_GX
  gx::update();
#endif
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
  gx::fifo::begin_frame();
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  { gfx::stagelog::Scope t{"drain"}; gx::fifo::drain(); }
  if (gfx::g_afterDrain != nullptr)
    gfx::g_afterDrain();
  gx::fifo::end_frame();
  gx::texture::end_frame();
  { gfx::stagelog::Scope t{"finish"}; gfx::finish(); }
  auto imguiDrawData = imgui::freeze();
  webgpu::PresentHook::Draw hookDraw;
  if (webgpu::present_hook().capture != nullptr) {
    hookDraw = webgpu::present_hook().capture();
  }

  const auto& presentSource = webgpu::present_source();
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                           webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                           presentSource.size.width, presentSource.size.height);

  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlBindGroup = std::move(rmlFrame.bindGroup);
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  gfx::end_frame([rmlBindGroup = std::move(rmlBindGroup), rmlOverlay, viewportMain = viewport,
                  imguiDrawData = std::move(imguiDrawData), hookDraw = std::move(hookDraw)](
                     wgpu::CommandEncoder& encoder, std::vector<gfx::AfterSubmitCallback> afterSubmitCallbacks) {
    wgpu::Texture currentTexture;
    wgpu::TextureView currentView;
    auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
    {
      window::SurfaceLock surfaceLock;
      if (window::is_presentable() && g_surface) {
        ZoneScopedN("Acquire texture");
        wgpu::SurfaceTexture surfaceTexture;
        gfx::stagelog::Scope acquireTime{"acquire"};
        g_surface.GetCurrentTexture(&surfaceTexture);
        surfaceStatus = surfaceTexture.status;
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
          currentTexture = std::move(surfaceTexture.texture);
          currentView = currentTexture.CreateView();
        }
      }
    }

    const bool canPresent = currentTexture && currentView;
    if (canPresent) {
      // This frame's present source is known only here, on the render worker after its passes were encoded (the
      // display copy sets it); the main thread's viewport came from a frame before (or the EFB's 640x480 at the
      // first). The letterbox follows the copy's own size.
      const auto& frameSource = webgpu::present_source();
      const auto viewport = rmlBindGroup ? viewportMain
                                         : webgpu::calculate_present_viewport(
                                               webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                               webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                               frameSource.size.width, frameSource.size.height);
      wgpu::BindGroup presentBindGroup;
      if (rmlBindGroup && !rmlOverlay) {
        presentBindGroup = rmlBindGroup;
      } else {
        const auto& resampledSource = webgpu::resample_present_source(encoder, viewport);
        presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
      }
      // A filter of the whole picture: the present and everything over it into its view, then its last pass
      wgpu::TextureView filterView;
      if (webgpu::present_hook().finalTarget != nullptr) {
        filterView = webgpu::present_hook().finalTarget(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                        webgpu::g_graphicsConfig.surfaceConfiguration.height);
      }
      // AURORA_LOOK_HDR=1: everything to here as for an 8-bit surface, in float; the last step to the surface in scRGB
      const wgpu::TextureView outputView =
          webgpu::hdr::on() ? webgpu::hdr::target(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                  webgpu::g_graphicsConfig.surfaceConfiguration.height)
                            : currentView;
      const wgpu::TextureView& drawView = filterView ? filterView : outputView;
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = drawView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        set_present_viewport(pass, viewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                             webgpu::g_graphicsConfig.surfaceConfiguration.height);

        pass.Draw(3);
        if (rmlBindGroup && rmlOverlay) {
          pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
          pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
          pass.Draw(3);
        }
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = drawView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass, imguiDrawData);
        if (hookDraw) {
          hookDraw(pass, viewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                   webgpu::g_graphicsConfig.surfaceConfiguration.height);
        }
        pass.End();
      }
      if (filterView) {
        webgpu::present_hook().finalPass(encoder, filterView, outputView);
      }
      if (webgpu::hdr::on()) {
        webgpu::hdr::final_pass(encoder, currentView);
      }
    } else {
      Log.info("Skipping present; window not presentable");
    }
    webgpu::gpu_prof::frame_end(encoder);
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    wgpu::CommandBuffer buffer;
    {
      gfx::stagelog::Scope finishTime{"ef_finish"};
      buffer = encoder.Finish(&cmdBufDescriptor);
    }
    {
      ZoneScopedN("Queue Submit");
      if (gfx::g_beforeFrameSubmit != nullptr)
        gfx::g_beforeFrameSubmit();
      gfx::stagelog::Scope submitTime{"submit"};
      g_queue.Submit(1, &buffer);
      if (gfx::g_afterFrameSubmit != nullptr)
        gfx::g_afterFrameSubmit();
      const uint64_t submitNs = timing_now_ns();
      g_timingSubmitNs.store(submitNs, std::memory_order_release);
      if (gfx::g_overlap && !gfx::g_drawingBetween) {
        g_queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                    [submitNs](wgpu::QueueWorkDoneStatus, wgpu::StringView) {
                                      const uint64_t done = timing_now_ns();
                                      g_timingGpuDoneNs.store(done, std::memory_order_release);
                                      const uint64_t lag = done > submitNs ? done - submitNs : 0, was = g_timingTrueLagNs.load();
                                      g_timingTrueLagNs.store(was == 0 ? lag : was + (static_cast<int64_t>(lag) - static_cast<int64_t>(was)) / 8,
                                                              std::memory_order_release);
                                      if (gfx::stagelog::on())
                                        gfx::stagelog::note("true_gpu_lag", std::chrono::steady_clock::now(), lag / 1000);
                                    });
      } else {
        g_queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                    [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {
                                      g_timingGpuDoneNs.store(timing_now_ns(), std::memory_order_release);
                                    });
      }
    }
    {
      gfx::stagelog::Scope readbackTime{"ef_readback"};
      webgpu::gpu_prof::after_submit();
      gfx::efb_readback::after_submit();
    }
    if (canPresent && g_surface) {
      ZoneScopedN("Present");
      wgpu::ConvertibleStatus status = wgpu::Status::Error;
      if (gfx::g_beforePresent != nullptr)
        gfx::g_beforePresent();
      {
        window::SurfaceLock surfaceLock;
        if (window::is_presentable()) {
          gfx::stagelog::Scope presentTime{"present"};
          status = g_surface.Present();
        }
      }
      if (status) {
        gfx::after_present();
      } else {
        Log.warn("Surface present failed");
        webgpu::release_surface();
      }
    } else if (g_surface) {
      switch (surfaceStatus) {
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        Log.warn("Surface texture acquisition timed out");
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
        Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
        window::push_custom_event(window::CustomEvent::RefreshSurface);
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceStatus));
        webgpu::release_surface();
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Error:
        Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceStatus));
        g_surface = {};
        break;
      default:
        if (!window::is_presentable()) {
          webgpu::release_surface();
        } else {
          Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
        }
        break;
      }
    }
    for (auto& callback : afterSubmitCallbacks) {
      if (callback) {
        callback();
      }
    }

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    const auto& stats = gfx::detail::resources().stats;
    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(stats.lastIndexSize));
    TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(stats.lastStorageSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(stats.lastTextureUploadSize));
  });

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
SDL_Window* aurora_get_window() { return aurora::window::get_sdl_window(); }
AuroraWindowSize aurora_get_window_size() {
  return aurora::window::get_sdl_window() != nullptr ? aurora::window::get_window_size() : AuroraWindowSize{};
}
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }
