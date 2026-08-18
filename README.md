# ComfyUI LTX 2.3 RunPod Template (CUDA 13.0)

This repository contains all the configuration files and scripts needed to build, publish, and deploy a custom **RunPod Community Template** for running ComfyUI with MiniMax H3 and LTX 2.3 workflows.

## Features

- **CUDA 13.0 Multi-Stage Build**: Utilizes `runpod/comfyui:cuda13.0` with discarded `sagebuilder` stage to keep runtime image slim.
- **ComfyUI Pinned to `v0.33.2`**: The base image bakes ComfyUI `0.30.0`, which is below the `0.33.0` floor the latest workflows need, so the build checks out `v0.33.2` explicitly. See below.
- **SageAttention 2 Multi-Arch Compilation**: Compiled from git source targeting `TORCH_CUDA_ARCH_LIST="8.6;8.9"` for both RTX 3090 (`sm_86`) and RTX 4090 (`sm_89`). **The `;` is mandatory — see below.**
- **Spectrum Sampler Acceleration**: Pinned `ComfyUI-Spectrum-MiniMax-H3` (`b5fd9db33267623eb3469ee7d6d4ddf397240025`).
- **Stability Flags**: Pre-configured `--disable-dynamic-vram`, `--disable-async-offload`, `--disable-smart-memory`, and `--reserve-vram 6`.
- **Driver Gate**: Automatically checks for NVIDIA host driver `>= 580` required for CUDA 13 images.

---

## Technical Notes

### The arch-list separator is `;`, never a space (root cause of the build failure)

SageAttention's `setup.py` parses the arch list with exactly this expression:

```python
for item in arch_list_env.replace(",", ";").split(";"):
```

It splits on `;` and `,` **only** — never on whitespace. So `"8.6 8.9"` is not two
capabilities, it is the single token `"8.6 8.9"`. That token matches
`capability.startswith("8.6")`, which sets `HAS_SM86 = True` and leaves
`HAS_SM89 = False`. The build then:

- emits only `-gencode arch=compute_86,code=sm_86`,
- skips the `_qattn_sm89` extension entirely,
- compiles `_qattn_sm80` and `_fused` with sm_86 SASS only.

The wheel builds and installs cleanly, so the failure is silent until an RTX 4090
hits `AssertionError: SM89 kernel is not available`, or any card outside sm_86
hits `no kernel image is available for execution on the device`.

| `TORCH_CUDA_ARCH_LIST` | parsed as | extensions built | RTX 4090 |
| --- | --- | --- | --- |
| `8.6 8.9` | `['8.6 8.9']` | `_qattn_sm80`, `_fused` | broken |
| `8.6;8.9` | `['8.6', '8.9']` | `_qattn_sm80`, `_qattn_sm89`, `_fused` | works |

A previous version of the Dockerfile actively converted `;` to spaces
(`tr ';,' '  '`) on the mistaken premise that `setup.py` wanted a space-delimited
list. It also validated the result with
`torch.utils.cpp_extension._get_cuda_arch_flags()`, which **does** split on
whitespace — so torch cheerfully reported `compute_86` + `compute_89` for a list
that SageAttention itself had already collapsed. That probe has been removed and
replaced with a preflight that runs SageAttention's own parser.

### `SM86_ENABLED` does not exist

`sageattention/core.py` defines only three capability flags:

```python
SM80_ENABLED    SM89_ENABLED    SM90_ENABLED
```

There is no `SM86_ENABLED`, so `getattr(core, "SM86_ENABLED", None)` is
permanently `None` and any gate asserting on it can never pass, regardless of how
the wheel was compiled. The RTX 3090 (`sm_86`) is served by the **sm80**
extension — `setup.py` builds `_qattn_sm80` when any of
`HAS_SM80 / HAS_SM86 / HAS_SM89 / HAS_SM90 / ...` is true.

Likewise, `hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda")` is useless as a
fallback detector: that Python function always exists and only asserts
`SM89_ENABLED` when it is actually called.

Both gates now call `verify-sage.py`, which checks the compiled `.so` files on
disk, the real `core` flags, and (on a pod) runs live forward passes.

### Runtime kernel dispatch

`sageattn()` selects by compute capability, not by what was compiled:

| GPU | capability | kernel chosen by `sageattn()` | needs |
| --- | --- | --- | --- |
| RTX 3090 | `sm_86` | `sageattn_qk_int8_pv_fp16_triton` | **Triton** |
| RTX 4090 | `sm_89` | `sageattn_qk_int8_pv_fp8_cuda` | `_qattn_sm89` |

Note the 3090 default goes through **Triton**, not the CUDA extension. To force
the compiled CUDA kernel on a 3090, set `Patch Sage Attention KJ` to
`sageattn_qk_int8_pv_fp16_cuda`. `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` are pointed at `/workspace/.cache/` so JIT compilation
is paid once per volume rather than once per pod start.

### SageAttention SM89 Error Explanation (Patch 7 Correction)

`AssertionError: SM89 kernel is not available` means the installed SageAttention
wheel was compiled without an `sm_89` cubin. It is a build-time defect and is
independent of which GPU is present. Rebuilding with
`TORCH_CUDA_ARCH_LIST="8.6;8.9"` and verifying `core.SM89_ENABLED` is `True`
fixes this. Separately, the RTX 3090 is `sm_86` and has no FP8 tensor cores, so
on a 3090 set `PatchSageAttentionKJ` to `auto` or
`sageattn_qk_int8_pv_fp16_cuda`. A wheel compiled for both architectures is
correct for both cards; the kernel choice is made at runtime.

### The base image ships ComfyUI 0.30.0, below the 0.31.0 floor

`runpod/comfyui:cuda13.0` (digest `sha256:976ebfd8…`) bakes ComfyUI **0.30.0**.
`pin-comfyui.sh` defaults to a *verify only* mode that never mutates the checkout
and simply asserts `baked >= COMFYUI_MIN_VERSION`. Against this base that check
can never pass:

```
version file: 0.30.0
required min: 0.31.0
explicit pin: <none - verify only>
RESULT: FAIL - baked ComfyUI 0.30.0 is below the required 0.31.0.
```

That is the gate working correctly, not a gate bug. Because the base image is
pinned by digest it will never drift upward on its own, so the fix is to pin the
ComfyUI ref explicitly:

```dockerfile
ARG COMFYUI_REF=v0.33.2
```

`v0.33.2` is the newest tag on origin. When moving to a newer ComfyUI,
bump `COMFYUI_REF` — do **not** lower `COMFYUI_MIN_VERSION`, which exists to stop
the MiniMax H3 core nodes silently loading against a too-old ComfyUI.

Two hardening changes went in alongside it:

- The explicit-pin path now re-checks the floor **after** checkout. Previously
  `COMFYUI_REF=v0.29.0` would have sailed straight through the script.
- `git fetch --depth 1` is now used only when the checkout is already shallow.
  The baked tree is a full clone, and shallowing it degrades `git describe` and
  the build manifest generated later in the Dockerfile.

After checkout, `requirements.txt` is reinstalled through `filter-req.py`, which
strips `torch` / `numpy` / `transformers` pins and any `--extra-index-url` so the
CUDA 13 runtime cannot be replaced out from under the SageAttention wheel. The
resolved `comfyui-frontend-package` version is printed to the build log — a
ComfyUI source bump left with a stale frontend package is the usual cause of a
blank or half-broken web UI.

### Line endings

This repo is authored on Windows. `.gitattributes` forces `eol=lf` on all shipped
scripts. A `.sh` file committed with CRLF gets a shebang of `#!/bin/bash\r`, and
the kernel then looks for an interpreter literally named `/bin/bash\r`, failing
at runtime with a confusing `no such file or directory`.

### Spectrum Step Budget Operational Note (Patch 8)

At 8 total steps (e.g., in a 20-step run with `warmup_steps = 5` and
`tail_actual_steps = 1`), Spectrum silently floors `tail_actual_steps` at 3 with a
deterministic sampler:
`warmup (5) + tail (3) = 8 of 8 steps executed normally (0 steps forecast)`.
Spectrum is skipping no steps while paying history-buffer overhead. To gain
speedup:

1. Raise step count to 16–20 so there is a middle section to forecast, OR
2. Lower `warmup_steps` to 1–2 inside the 8-step budget.

*Note:* Spectrum is incompatible with `EasyCache` / `LazyCache` on the same model
branch. Do not enable both simultaneously.

---

## Build

```bash
docker buildx build \
  --build-arg TORCH_CUDA_ARCH_LIST='8.6;8.9' \
  --build-arg MAX_JOBS=2 \
  --build-arg EXT_PARALLEL=2 \
  --build-arg NVCC_THREADS=2 \
  --build-arg SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 \
  --build-arg COMFYUI_MIN_VERSION=0.33.0 \
  --build-arg COMFYUI_REF=v0.33.2 \
  -t youruser/sam-ltx23-comfyui:cuda13-v2 .
```

Quote the arch list in your shell — an unquoted `8.6;8.9` is parsed as a command
separator and silently truncates to `8.6`.

Measured on a GitHub-hosted `ubuntu-latest` runner at `MAX_JOBS=2`:

| build | wheel size | `_qattn_sm89` | compile |
| --- | --- | --- | --- |
| broken (`8.6 8.9`) | 8,334,376 B | absent | 206 s |
| fixed (`8.6;8.9`) | 21,590,575 B | 36 MB, `sm_86` + `sm_89` | 361 s |

The GitHub Actions workflow uses a registry build cache keyed on `SAGE_REF` and
the arch list, so that cost is paid once.

On the first run — and after **any** failed run — buildx logs:

```
ERROR: failed to configure registry cache importer:
       .../sam-ltx23-comfyui:buildcache: not found
```

That is expected and non-fatal. buildx only exports the cache tag when a build
succeeds, so it cannot exist yet. It stops appearing after the first green run.

## Verify on the pod

```bash
python3 /usr/local/bin/verify-sage.py
```
