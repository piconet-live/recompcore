# Aurora patches on top of upstream 5179721

One commit each on the recompcore branch `gxr/upstream`, subject `Aurora patch NN: ...`. Export the series with

    git format-patch --relative=GXRuntime/graphics/aurora <base commit "Aurora: upstream 5179721 as the base">..

It applies with `git am` to a clean encounter/aurora checkout at `5179721` and reproduces this directory (checked
2026-10-30, all 16 files: 15 patches and the fixup of 15).

Kinds: **upstream** = fit to send to encounter/aurora as is; **recomp** = the static-recomp seam (Aurora's decomp
ports do not need it); **local** = tools and diagnostics for GXRuntime's harness.

| # | Patch | Kind | From (gxr/stage2) |
|---|---|---|---|
| 01 | Guest-address seam for retail GX streams (`lib/gx/recomp.*`, CALL_DL, CP array bases, BP copy/TLUT/texture registers, indexed-array spans) | recomp | bootstrap 0001-0004, 4172723df3 (index scan) |
| 02 | Triangle lists draw only whole triangles | upstream | fe5d3048ce |
| 03 | GX_CULL_ALL draws nothing | upstream | bootstrap 0007 |
| 04 | Min-filter decoding and signed LOD bias | upstream backport (26371ce, after the pin; drops at the next re-pin) | 5368099b56 |
| 05 | Scissor box offset and viewport origin as the hardware (encoder and decoder) | upstream, to discuss (changes Aurora's register encoding) | 6b711915f2 |
| 06 | Retail texture binds kept within a texture generation (`AURORA_RAW_TEXCACHE`) | recomp | 4172723df3 |
| 07 | The display (XFB) copy, presented (`AURORA_XFB`, `AURORA_XFB_FILTER`) | upstream | 5368099b56 |
| 08 | Hardware sampling defaults: the game's anisotropy, no arbitrary-mip detection (`AURORA_ANISO_GX`, `AURORA_ARB_MIPS`) | upstream as a config option | 5368099b56 |
| 09 | Present-source readback, its buffer released outside the map callback (`aurora_*_framebuffer_readback`) | local; the release fix is the upstreamable part | bootstrap 0006, b1a74e7a6d |
| 10 | Pixel centre at 7/12 (`AURORA_PIXEL_CENTER`) | upstream | d0ee8a2b6e |
| 11 | Emboss offsets along the NBT vectors as Dolphin | upstream | e7c9db4372 |
| 12 | Upstream readback: staging buffers never freed inside their map callback | upstream (found by reading) | new |
| 13 | Hooks for offline tools: frame timing, `AURORA_HIDDEN_WINDOW`, BUILD_TESTING restored | local | 6b711915f2, c3ddf9a7e2 |
| 14 | No out-of-range read with zero TEV stages | upstream | bootstrap 0006 |
| 15 | Draw diagnostics (`AURORA_DRAW_LOG`, `_TCG`, `AURORA_IND_LOG`, `AURORA_SKIP_TEX`) + fixup | local | 4172723df3, e6a85563d4 |

## Not carried, and why

- **gxcore** (`lib/gfx/gxcore_draw.*`, the GXCore shader type, its pipeline-cache rows): dropped by the owner's
  decision to keep Aurora's own GX path.
- **The fork's retail content cache** (`gx.cpp` `s_rawTextureCache`): upstream's content cache (XXH3-128 of texels
  and palette) does the same; only the per-generation bind memo (06) is kept.
- **Bootstrap 0005, XF channel-control attenuation bits**: upstream fixed it (regs.cpp `xf_chan_ctrl`).
- **`AURORA_SYNC_PIPELINES`** (0006): upstream's `AuroraSetDrawsBlocking` replaces it.
- **Larger vertex/index buffers**: they were for gxcore's triangle expansion.
- **Indexed XF bounds skip** (0004): upstream asserts on the same bounds; retail streams stay inside them.
- **Depth-peek default, depth-copy formats**: upstream rewrote peeks as `aurora::gfx::readback` and has the depth
  copy formats.
- **Copy log, texture-cache statistics, device-feature log, forced device synchronization** (diagnostics): upstream
  has texture statistics; the device lock proved unrelated to the stage-4 crash.
