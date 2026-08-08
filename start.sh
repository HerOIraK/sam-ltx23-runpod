#!/usr/bin/env bash
set -uo pipefail

COMFYUI_DIR="/opt/ComfyUI"
VOLUME_DIR="/workspace"

echo "=============================================================="
echo " MiniMax H3 + LTX-2.3 ComfyUI  |  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================================="

# 1. Warm up NVIDIA CUDA Driver & Boot Environment Diagnostics
nvidia-smi >/dev/null 2>&1 || true

python3 - <<'PYCHK'
import sys
print("Python          :", sys.version.split()[0])
try:
    import torch
    print("PyTorch         :", torch.__version__, "| CUDA", torch.version.cuda)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("GPU Device      : %s  sm_%d%d  %.1f GiB" % (name, cap[0], cap[1], total))
        SM = cap[0] * 10 + cap[1]
        if SM >= 89:
            print("FP8 Tensor Cores: SUPPORTED - sageattn_qk_int8_pv_fp8_cuda++ will run natively")
        else:
            print("FP8 Tensor Cores: NOT PRESENT ON SM<89 (RTX 3090 / Ampere)")
            print("                  Note: On RTX 3090, set 'Patch Sage Attention KJ' -> 'auto'")
    else:
        print("GPU Device      : NONE VISIBLE")
        SM = 0
except Exception as e:
    print("PyTorch         : FAILED ->", e)

try:
    import sageattention as s
    ks = sorted(k for k in dir(s) if k.startswith("sageattn"))
    print("SageAttention   :", len(ks), "kernels ->", ", ".join(ks) or "NONE")
    if "sageattn_qk_int8_pv_fp8_cuda" not in ks and "sageattn" not in ks:
        print("  !! SageAttention kernels missing")
    import torch
    if torch.cuda.is_available():
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        o = s.sageattn(q, q.clone(), q.clone(), tensor_layout="HND")
        torch.cuda.synchronize()
        print("SageAttention   : Forward pass OK, output shape=", tuple(o.shape))
except Exception as e:
    print("SageAttention   : Diagnostic note ->", e)
PYCHK

echo "--------------------------------------------------------------"

# 2. Connect Persistent Volume Storage (/workspace)
if [[ -d "$VOLUME_DIR" ]]; then
    mkdir -p \
        "$VOLUME_DIR/models" \
        "$VOLUME_DIR/input" \
        "$VOLUME_DIR/output" \
        "$VOLUME_DIR/user/default/workflows" \
        "$VOLUME_DIR/user/__manager" \
        "$VOLUME_DIR/downloads"

    rm -rf "$COMFYUI_DIR/models"; ln -s "$VOLUME_DIR/models" "$COMFYUI_DIR/models"
    rm -rf "$COMFYUI_DIR/input";  ln -s "$VOLUME_DIR/input"  "$COMFYUI_DIR/input"
    rm -rf "$COMFYUI_DIR/output"; ln -s "$VOLUME_DIR/output" "$COMFYUI_DIR/output"

    cp -n "$COMFYUI_DIR/user/default/workflows/"*.json "$VOLUME_DIR/user/default/workflows/" 2>/dev/null || true
    cp -n "$COMFYUI_DIR/user/default/comfy.settings.json" "$VOLUME_DIR/user/default/comfy.settings.json" 2>/dev/null || true
    cp -n "$COMFYUI_DIR/user/__manager/config.ini" "$VOLUME_DIR/user/__manager/config.ini" 2>/dev/null || true

    rm -rf "$COMFYUI_DIR/user"
    ln -s "$VOLUME_DIR/user" "$COMFYUI_DIR/user"
fi

# 3. Model Downloads (Disabled by default)
if [ "${AUTO_DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_LTX_MODELS:-false}" = "true" ]; then
    echo "[models] AUTO_DOWNLOAD_LTX_MODELS=true -> Running LTX model downloader..."
    /download-ltx-models.sh || echo "WARN: LTX downloader reported errors; continuing"
elif [ "${AUTO_DOWNLOAD_SCAIL2_MODELS:-false}" = "true" ]; then
    echo "[models] AUTO_DOWNLOAD_SCAIL2_MODELS=true -> Running SCAIL2 model downloader..."
    /download-scail2-models.sh || echo "WARN: SCAIL2 downloader reported errors; continuing"
else
    echo "[models] Auto model downloads are DISABLED (default). No models downloaded on boot."
fi

# 4. Start VS Code code-server on port 8000 in background
echo "Starting code-server on port 8000..."
mkdir -p /workspace/code-server
rm -f /workspace/root-fs || true
ln -sf / /workspace/root-fs || true
nohup code-server --bind-addr 0.0.0.0:8000 --auth none --user-data-dir /workspace/code-server /workspace >/workspace/code-server.log 2>&1 &

# 5. Build ComfyUI launch command with stability flags for MiniMax H3 / int8 convrot
ARGS=(main.py --listen 0.0.0.0 --port 8188 --enable-cors-header)

[ "${ENABLE_MANAGER:-true}"        = "true" ] && ARGS+=(--enable-manager)
[ "${DISABLE_DYNAMIC_VRAM:-true}"  = "true" ] && ARGS+=(--disable-dynamic-vram)
[ "${DISABLE_ASYNC_OFFLOAD:-true}" = "true" ] && ARGS+=(--disable-async-offload)
[ "${DISABLE_PINNED_MEMORY:-false}" = "true" ] && ARGS+=(--disable-pinned-memory)
[ -n "${RESERVE_VRAM:-}" ] && ARGS+=(--reserve-vram "${RESERVE_VRAM}")

if [ -n "${COMFY_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${COMFY_EXTRA_ARGS})
fi

echo "--------------------------------------------------------------"
echo "Launching ComfyUI: python3 ${ARGS[*]}"
echo "--------------------------------------------------------------"
cd "$COMFYUI_DIR"
exec python3 "${ARGS[@]}"
