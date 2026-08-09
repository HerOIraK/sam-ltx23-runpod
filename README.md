# ComfyUI LTX 2.3 RunPod Template (CUDA 13.0)

This repository contains all the configuration files and scripts needed to build, publish, and deploy a custom **RunPod Community Template** for running ComfyUI with MiniMax H3 and LTX 2.3 workflows.

## Features

- **CUDA 13.0 Multi-Stage Build**: Utilizes `runpod/comfyui:cuda13.0` with discarded `sagebuilder` stage to keep runtime image slim.
- **SageAttention 2 Multi-Arch Compilation**: Compiled from git source targeting `TORCH_CUDA_ARCH_LIST="8.6 8.9"` for both RTX 3090 (`sm_86`) and RTX 4090 (`sm_89`).
- **Spectrum Sampler Acceleration**: Pinned `ComfyUI-Spectrum-MiniMax-H3` (`b5fd9db33267623eb3469ee7d6d4ddf397240025`).
- **Stability Flags**: Pre-configured `--disable-dynamic-vram`, `--disable-async-offload`, `--disable-smart-memory`, and `--reserve-vram 6`.
- **Driver Gate**: Automatically checks for NVIDIA host driver `>= 580` required for CUDA 13 images.

---

## Technical Notes

### SageAttention SM89 Error Explanation (Patch 7 Correction)
`AssertionError: SM89 kernel is not available` means the installed SageAttention wheel was compiled without an `sm_89` cubin. It is a build-time defect and is independent of which GPU is present. Rebuilding with `TORCH_CUDA_ARCH_LIST="8.6 8.9"` and verifying `core.SM89_ENABLED` is `True` fixes this. Separately, the RTX 3090 is `sm_86` and has no FP8 tensor cores, so on a 3090 set `PatchSageAttentionKJ` to `auto` or `sageattn_qk_int8_pv_fp16_cuda`. A wheel compiled for both architectures is correct for both cards; the kernel choice is made at runtime.

### Spectrum Step Budget Operational Note (Patch 8)
At 8 total steps (e.g., in a 20-step run with `warmup_steps = 5` and `tail_actual_steps = 1`), Spectrum silently floors `tail_actual_steps` at 3 with a deterministic sampler:
`warmup (5) + tail (3) = 8 of 8 steps executed normally (0 steps forecast)`.
Spectrum is skipping no steps while paying history-buffer overhead. To gain speedup:
1. Raise step count to 16–20 so there is a middle section to forecast, OR
2. Lower `warmup_steps` to 1–2 inside the 8-step budget.

*Note:* Spectrum is incompatible with `EasyCache` / `LazyCache` on the same model branch. Do not enable both simultaneously.
