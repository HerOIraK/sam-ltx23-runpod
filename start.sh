#!/usr/bin/env bash
set -uo pipefail

COMFYUI_DIR="/opt/ComfyUI"
VOLUME_DIR="/workspace"

echo "============================================================"
echo " ComfyUI / MiniMax H3 + LTX-2.3  |  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# 1. Report GPU Environment & Tensor Core Capabilities
python3 - <<'PY'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA version    : {torch.version.cuda}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU device      : {name}")
    print(f"Capability      : sm_{cc[0]}{cc[1]}")
    print(f"VRAM            : {vram:.1f} GiB")
    cc_num = cc[0] * 10 + cc[1]
    if cc_num >= 89:
        print("FP8 Tensor Cores: SUPPORTED - Native FP8 CUDA kernels will execute")
    else:
        print("FP8 Tensor Cores: NOT PRESENT ON SM<89 (RTX 3090 / Ampere)")
        print("                  Note: On RTX 3090, set 'Patch Sage Attention KJ' -> 'auto'")
        print("                  or use MiniMaxH3MemoryEfficientSageAttentionPatch")
else:
    print("GPU device      : NONE VISIBLE")
PY

# 2. Verify SageAttention & Run GPU Forward Pass
python3 - <<'PY'
import sys
try:
    import sageattention as s
except Exception as e:
    print(f"SageAttention   : IMPORT FAILED - {e}")
    sys.exit(0)

ks = sorted(n for n in dir(s) if n.startswith("sageattn"))
print(f"SageAttention   : {len(ks)} kernels -> {', '.join(ks)}")

try:
    import torch
    if torch.cuda.is_available():
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        o = s.sageattn(q, q.clone(), q.clone(), tensor_layout="HND")
        torch.cuda.synchronize()
        print(f"SageAttention   : GPU forward pass OK, out={tuple(o.shape)} {o.dtype}")
except Exception as e:
    print(f"SageAttention   : Forward pass error - {type(e).__name__}: {e}")
PY

echo "------------------------------------------------------------"

# 3. Connect Persistent Volume Storage (/workspace)
mkdir -p \
    "$VOLUME_DIR/models" \
    "$VOLUME_DIR/input" \
    "$VOLUME_DIR/output" \
    "$VOLUME_DIR/user" \
    "$VOLUME_DIR/downloads"

rm -rf "$COMFYUI_DIR/models";  ln -s "$VOLUME_DIR/models" "$COMFYUI_DIR/models"
rm -rf "$COMFYUI_DIR/input";   ln -s "$VOLUME_DIR/input"  "$COMFYUI_DIR/input"
rm -rf "$COMFYUI_DIR/output";  ln -s "$VOLUME_DIR/output" "$COMFYUI_DIR/output"

mkdir -p "$VOLUME_DIR/user/default/workflows" "$VOLUME_DIR/user/__manager"
cp -n "$COMFYUI_DIR/user/default/workflows/"*.json "$VOLUME_DIR/user/default/workflows/" 2>/dev/null || true
cp -n "$COMFYUI_DIR/user/default/comfy.settings.json" "$VOLUME_DIR/user/default/comfy.settings.json" 2>/dev/null || true
cp -n "$COMFYUI_DIR/user/__manager/config.ini" "$VOLUME_DIR/user/__manager/config.ini" 2>/dev/null || true

rm -rf "$COMFYUI_DIR/user"
ln -s "$VOLUME_DIR/user" "$COMFYUI_DIR/user"

# 4. Model Downloads (Disabled by default)
if [ "${AUTO_DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_LTX_MODELS:-false}" = "true" ]; then
    echo "[models] AUTO_DOWNLOAD_LTX_MODELS=true -> Running LTX model downloader..."
    /download-ltx-models.sh
elif [ "${AUTO_DOWNLOAD_SCAIL2_MODELS:-false}" = "true" ]; then
    echo "[models] AUTO_DOWNLOAD_SCAIL2_MODELS=true -> Running SCAIL2 model downloader..."
    /download-scail2-models.sh
else
    echo "[models] Auto model downloads are DISABLED (default). No models downloaded on boot."
fi

# 5. Start VS Code code-server on port 8000 in background
echo "Starting code-server on port 8000..."
mkdir -p /workspace/code-server
rm -f /workspace/root-fs || true
ln -sf / /workspace/root-fs || true
nohup code-server --bind-addr 0.0.0.0:8000 --auth none --user-data-dir /workspace/code-server /workspace >/workspace/code-server.log 2>&1 &

# 6. Launch ComfyUI on port 8188
cd "$COMFYUI_DIR"
echo "[comfy] Starting ComfyUI server on 0.0.0.0:8188..."
exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
