add_library(aurora_gx STATIC
        lib/gfx/clear.cpp
        lib/gfx/encoding.cpp
        lib/gfx/frame.cpp
        lib/gfx/efb_readback.cpp
        lib/gfx/timewarp.cpp
        lib/gfx/pipeline_cache.cpp
        lib/gfx/readback/copy_pack.cpp
        lib/gfx/readback/efb_gather.cpp
        lib/gfx/readback/gpu.cpp
        lib/gfx/readback/readback.cpp
        lib/gfx/recording.cpp
        lib/gfx/render_worker.cpp
        lib/gfx/resource_cache.cpp
        lib/gfx/dds_io.cpp
        lib/gfx/tex_copy_conv.cpp
        lib/gfx/tex_palette_conv.cpp
        lib/gfx/texture.cpp
        lib/gfx/texture_format.cpp
        lib/gfx/texture_convert.cpp
        lib/gfx/texture_replacement.cpp
        lib/gx/attr_fmt.cpp
        lib/gx/command_processor.cpp
        lib/gx/rt_capture.cpp
        lib/gx/rt_scene.cpp
        lib/gx/regs.cpp
        lib/gx/recomp.cpp
        lib/gx/dl.cpp
        lib/gx/fifo.cpp
        lib/gx/gx.cpp
        lib/gx/texture.cpp
        lib/gx/pipeline.cpp
        lib/gx/readback.cpp
        lib/gx/shader.cpp
        lib/gx/shader_info.cpp
        lib/dolphin/gx/GXBump.cpp
        lib/dolphin/gx/GXCull.cpp
        lib/dolphin/gx/GXCpu2Efb.cpp
        lib/dolphin/gx/GXDispList.cpp
        lib/dolphin/gx/GXDraw.cpp
        lib/dolphin/gx/GXExtra.cpp
        lib/dolphin/gx/GXFifo.cpp
        lib/dolphin/gx/GXFrameBuffer.cpp
        lib/dolphin/gx/GXGeometry.cpp
        lib/dolphin/gx/GXGet.cpp
        lib/dolphin/gx/GXLighting.cpp
        lib/dolphin/gx/GXManage.cpp
        lib/dolphin/gx/GXPerf.cpp
        lib/dolphin/gx/GXPixel.cpp
        lib/dolphin/gx/GXTev.cpp
        lib/dolphin/gx/GXTexture.cpp
        lib/dolphin/gx/GXTransform.cpp
        lib/dolphin/gx/GXVert.cpp
        lib/dolphin/gx/GXAurora.cpp
        lib/gfx/png_io.cpp
        lib/gfx/png_io.hpp
)
add_library(aurora::gx ALIAS aurora_gx)
set_target_properties(aurora_gx PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_gx PUBLIC aurora::core dawn::webgpu_dawn xxHash::xxhash)
if (WIN32 AND DAWN_ENABLE_D3D12)
    # Ray tracing's live pass: the shader, as bytes for the DXC of Dawn's own package
    target_sources(aurora_gx PRIVATE lib/gx/rt_live.cpp)
    target_compile_definitions(aurora_gx PRIVATE AURORA_RT_LIVE=1)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/lib/gx/rt.hlsl" _rt_hlsl HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _rt_hlsl "${_rt_hlsl}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rt_gen/rt_shader.inc" "${_rt_hlsl}")
    target_include_directories(aurora_gx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/rt_gen")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/gx/rt.hlsl")
endif ()
target_link_libraries(aurora_gx PRIVATE absl::btree absl::flat_hash_map sqlite3 Tracy::TracyClient PNG::PNG)
target_compile_definitions(aurora_gx PRIVATE WEBGPU_DAWN)

if (AURORA_ENABLE_RMLUI)
    target_sources(aurora_gx PRIVATE
        lib/rmlui/pipeline.cpp
        lib/rmlui/pipeline.hpp
    )
endif ()
