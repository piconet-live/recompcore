# Aurora — upstream pin and patch list

This directory is upstream Aurora at a pinned commit, plus a short list of named patches on top. Each patch is
one commit on the recompcore branch that touches this directory, so the list can be exported with
`git format-patch --relative=GXRuntime/graphics/aurora <pin-commit>..` and applied to an Aurora checkout.

- upstream: `https://github.com/encounter/aurora` (MIT; `LICENSE` kept in this directory)
- pin: `5179721f` (`Add AuroraSetDrawsBlocking to control async pipelines`, 2026-10-12), the newest upstream commit
  at least two weeks old on 2026-10-30
- replaces: the owned hard fork of `0549581` (2026-06-22) with its seven folded bootstrap patches and the
  GXRuntime stage 2-5 work (branch `gxr/stage2`)
- decision (owner, 2026-10-30): keep Aurora's own GX path (`lib/gx`); gxcore is dropped; rebase once, at the
  start of runner integration

## Rules

- Change upstream code only through a named patch (one commit, subject `Aurora patch NN: ...`).
- Mark each patch as **upstream** (fit to send to encounter/aurora as is), **recomp** (the static-recomp seam
  Aurora's decomp ports do not need) or **local** (diagnostics and tools for our harness).
- Every switch a patch adds defaults to the hardware/Dolphin behaviour.

## Patches

The list, with each patch's kind and origin, is in `PATCHES.md`.
