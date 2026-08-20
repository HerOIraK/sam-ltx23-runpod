#!/usr/bin/env bash
set -uo pipefail

COMFYUI_DIR="/opt/ComfyUI"
VOLUME_DIR="/workspace"

echo "=============================================================="
echo " MiniMax H3 + LTX-2.3 ComfyUI  |  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================================="

# 1. Driver Warmup & CUDA 13 Driver Gate (Patch 3)
nvidia-smi >/dev/null 2>&1 || true

DRV_FULL="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
DRV_MAJOR="${DRV_FULL%%.*}"
if [ -n "${DRV_MAJOR}" ] && [ "${DRV_MAJOR}" -lt 580 ] 2>/dev/null; then
    echo "============================================================="
    echo " FATAL: NVIDIA driver ${DRV_FULL} detected."
    echo " This image is CUDA 13 and requires driver >= 580."
    echo ""
    echo " Fix: terminate this pod and redeploy. In the RunPod console,"
    echo " open 'Additional Filters' -> 'CUDA Version' and select 13.0"
    echo " before choosing a GPU."
    echo "============================================================="
    exit 1
fi
echo "driver ${DRV_FULL} OK for CUDA 13"

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true

# Display build manifest on boot (Patch 4b)
if [ -f /opt/build-manifest.txt ]; then
    echo "------------------ Build Manifest ------------------"
    cat /opt/build-manifest.txt
    echo "----------------------------------------------------"
fi

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
            print("FP8 Tensor Cores: not present on sm_%d%d" % cap)
            print("                  Set 'Patch Sage Attention KJ' -> sageattn_qk_int8_pv_fp16_cuda")
    else:
        print("GPU Device      : NONE VISIBLE")
        SM = 0
except Exception as e:
    print("PyTorch         : FAILED ->", e)

PYCHK

# SageAttention verification -- the exact script the build gate uses.
if [[ -x /usr/local/bin/verify-sage.py ]]; then
    python3 /usr/local/bin/verify-sage.py \
        || echo "WARNING: SageAttention verification failed -- continuing boot so you can debug on the pod"
fi

echo "--------------------------------------------------------------"

# 2. Connect Persistent Volume Storage (/workspace)
if [[ -d "$VOLUME_DIR" ]]; then
    mkdir -p \
        "$VOLUME_DIR/models" \
        "$VOLUME_DIR/models/clip_projections" \
        "$VOLUME_DIR/input" \
        "$VOLUME_DIR/output" \
        "$VOLUME_DIR/user/default/workflows" \
        "$VOLUME_DIR/user/__manager" \
        "$VOLUME_DIR/downloads" \
        "$VOLUME_DIR/.cache/triton" \
        "$VOLUME_DIR/.cache/inductor"

    rm -rf "$COMFYUI_DIR/models"; ln -s "$VOLUME_DIR/models" "$COMFYUI_DIR/models"
    rm -rf "$COMFYUI_DIR/input";  ln -s "$VOLUME_DIR/input"  "$COMFYUI_DIR/input"
    rm -rf "$COMFYUI_DIR/output"; ln -s "$VOLUME_DIR/output" "$COMFYUI_DIR/output"

    cp -n "$COMFYUI_DIR/user/default/workflows/"*.json "$VOLUME_DIR/user/default/workflows/" 2>/dev/null || true
    cp -n "$COMFYUI_DIR/user/default/comfy.settings.json" "$VOLUME_DIR/user/default/comfy.settings.json" 2>/dev/null || true
    cp -n "$COMFYUI_DIR/user/__manager/config.ini" "$VOLUME_DIR/user/__manager/config.ini" 2>/dev/null || true

    rm -rf "$COMFYUI_DIR/user"
    ln -s "$VOLUME_DIR/user" "$COMFYUI_DIR/user"
fi

# 3. Optional Model Fetch (Patch 6)
if [ "${DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_LTX_MODELS:-false}" = "true" ]; then
    echo "[models] Fetching required models into $VOLUME_DIR/models..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    M="$VOLUME_DIR/models"
    mkdir -p "$M/diffusion_models" "$M/text_encoders" "$M/vae" "$M/loras" "$M/clip_projections"

    fetch() {
        if [ -s "$1" ]; then echo "present: $(basename "$1")"; return 0; fi
        echo "downloading: $(basename "$1")"
        aria2c -x 8 -s 8 -c --dir "$(dirname "$1")" --out "$(basename "$1")" \
            ${HF_TOKEN:+--header "Authorization: Bearer ${HF_TOKEN}"} "$2" \
            || echo "WARN: failed $(basename "$1")"
    }

    HF="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
    fetch "$M/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" "$HF/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    fetch "$M/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"        "$HF/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    fetch "$M/vae/minimax_h3_video_vae_fp16.safetensors"                          "$HF/vae/minimax_h3_video_vae_fp16.safetensors"
    fetch "$M/vae/minimax_h3_audio_vae_fp32.safetensors"                          "$HF/vae/minimax_h3_audio_vae_fp32.safetensors"

    df -h "$VOLUME_DIR" | tail -n1
else
    echo "[models] DOWNLOAD_MODELS=false (default). No models fetched automatically on boot."
fi

# 4. VS Code code-server startup (Port 8000 - Enabled by default)
if [ "${ENABLE_CODE_SERVER:-true}" = "true" ]; then
    mkdir -p /workspace/code-server
    if [ -n "${CODE_SERVER_PASSWORD:-}" ]; then
        echo "Starting code-server on port 8000 (password authentication enabled)..."
        PASSWORD="${CODE_SERVER_PASSWORD}" nohup code-server \
            --bind-addr 0.0.0.0:8000 \
            --auth password \
            --disable-telemetry \
            --user-data-dir /workspace/code-server \
            /workspace \
            >/workspace/code-server.log 2>&1 &
    else
        echo "Starting code-server on port 8000 (no password required)..."
        nohup code-server \
            --bind-addr 0.0.0.0:8000 \
            --auth none \
            --disable-telemetry \
            --user-data-dir /workspace/code-server \
            /workspace \
            >/workspace/code-server.log 2>&1 &
    fi
fi

# 5. Build ComfyUI launch command (Opt-in memory management flags)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(main.py --listen 0.0.0.0 --port 8188 --enable-cors-header)

[ "${ENABLE_MANAGER:-true}" = "true" ] && ARGS+=(--enable-manager)
[ "${DISABLE_DYNAMIC_VRAM:-0}" = "1" ] && ARGS+=(--disable-dynamic-vram)
[ "${DISABLE_ASYNC_OFFLOAD:-0}" = "1" ] && ARGS+=(--disable-async-offload)
[ "${DISABLE_SMART_MEMORY:-0}" = "1" ] && ARGS+=(--disable-smart-memory)
[ "${DISABLE_PINNED_MEMORY:-0}" = "1" ] && ARGS+=(--disable-pinned-memory)

ARGS+=(--reserve-vram "${RESERVE_VRAM:-0.5}")

if [ -n "${COMFY_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${COMFY_EXTRA_ARGS})
fi

# Boot diagnostics - echo resolved ComfyUI version and expanded ARGS array
COMFY_VERSION="$(python3 -c "import pathlib, re; p=pathlib.Path('/opt/ComfyUI/comfyui_version.py'); print(re.search(r'__version__\s*=\s*[\"']([^\"']+)[\"']', p.read_text()).group(1) if p.exists() else 'unknown')" 2>/dev/null || echo "unknown")"

echo "--------------------------------------------------------------"
echo "Resolved ComfyUI Version: ${COMFY_VERSION}"
echo "Launching ComfyUI: python3 ${ARGS[*]}"
echo "--------------------------------------------------------------"
cd "$COMFYUI_DIR"
exec python3 "${ARGS[@]}"
