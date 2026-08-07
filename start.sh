#!/bin/bash
set -e

COMFYUI_DIR="/opt/ComfyUI"
VOLUME_DIR="/workspace"

echo "Starting Sam LTX 2.3 ComfyUI template"

# Persistent directories
mkdir -p \
    "$VOLUME_DIR/models" \
    "$VOLUME_DIR/input" \
    "$VOLUME_DIR/output" \
    "$VOLUME_DIR/user" \
    "$VOLUME_DIR/downloads"

# Connect persistent storage to the ComfyUI installation
rm -rf "$COMFYUI_DIR/models"
ln -s "$VOLUME_DIR/models" "$COMFYUI_DIR/models"

rm -rf "$COMFYUI_DIR/input"
ln -s "$VOLUME_DIR/input" "$COMFYUI_DIR/input"

rm -rf "$COMFYUI_DIR/output"
ln -s "$VOLUME_DIR/output" "$COMFYUI_DIR/output"

# Copy included workflows and default settings to persistent user storage
mkdir -p "$VOLUME_DIR/user/default/workflows" "$VOLUME_DIR/user/__manager"

cp -n \
    "$COMFYUI_DIR/user/default/workflows/"*.json \
    "$VOLUME_DIR/user/default/workflows/" 2>/dev/null || true

cp -n "$COMFYUI_DIR/user/default/comfy.settings.json" "$VOLUME_DIR/user/default/comfy.settings.json" 2>/dev/null || true
cp -n "$COMFYUI_DIR/user/__manager/config.ini" "$VOLUME_DIR/user/__manager/config.ini" 2>/dev/null || true

rm -rf "$COMFYUI_DIR/user"
ln -s "$VOLUME_DIR/user" "$COMFYUI_DIR/user"

# Download models only when requested
if [ "${AUTO_DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_LTX_MODELS:-false}" = "true" ]; then
    /download-ltx-models.sh
fi

if [ "${AUTO_DOWNLOAD_SCAIL2_MODELS:-false}" = "true" ]; then
    /download-scail2-models.sh
fi

# Start VS Code code-server on port 8000 in the background
echo "Starting code-server on port 8000..."
mkdir -p /workspace/code-server
rm -f /workspace/root-fs || true
ln -sf / /workspace/root-fs || true
nohup code-server --bind-addr 0.0.0.0:8000 --auth none --user-data-dir /workspace/code-server /workspace &


# Ensure SageAttention and GPU capability routing for RTX 3090 / 4090
set +e
echo "Detecting GPU capability..."
GPU_CC=$(python3 -c "import torch; print('%d%d' % torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else print('none')" 2>/dev/null)
echo "   Compute capability: ${GPU_CC}"

# GPUs below SM 8.9 (e.g. RTX 3090 = SM 8.6) do not have FP8 tensor cores.
# Auto-rewrite any hardcoded FP8 kernel selections in saved workflows to 'auto'.
if [ "$GPU_CC" != "89" ] && [ "$GPU_CC" != "90" ] && [ "$GPU_CC" != "120" ]; then
    echo "   SM ${GPU_CC}: No FP8 hardware tensor cores -> Normalizing PatchSageAttentionKJ workflow settings to 'auto'..."
    WFDIR="$VOLUME_DIR/user/default/workflows" python3 - <<'PY'
import json, os, pathlib
BAD = {"sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++",
       "sageattn_qk_int8_pv_fp8_cuda_sm90", "sageattn_3_blackwell"}
def scrub(n):
    c = 0
    if isinstance(n, dict):
        for k, v in n.items():
            if isinstance(v, str) and v in BAD: n[k] = "auto"; c += 1
            else: c += scrub(v)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            if isinstance(v, str) and v in BAD: n[i] = "auto"; c += 1
            else: c += scrub(v)
    return c
d = pathlib.Path(os.environ["WFDIR"])
if d.is_dir():
    for f in d.rglob("*.json"):
        try: data = json.loads(f.read_text(encoding="utf-8"))
        except Exception: continue
        if scrub(data):
            f.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"      Scrubbed FP8 kernel in {f.name} -> auto")
PY
fi

echo "Verifying SageAttention installation..."
if ! python3 -c "from sageattention import sageattn" 2>/dev/null; then
    echo "   Installing SageAttention (Triton, SM80/SM86 safe)..."
    pip install --no-cache-dir sageattention==1.0.6 || pip install --no-cache-dir sageattention || true
fi

python3 - <<'PY' || true
try:
    import sageattention
    ks = sorted(n for n in dir(sageattention) if n.startswith("sageattn"))
    print("   SageAttention kernels available: " + (", ".join(ks) if ks else "none"))
except Exception as e:
    print(f"   SageAttention unavailable ({e}) - ComfyUI will use PyTorch attention")
PY
set -e

cd "$COMFYUI_DIR"

exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
