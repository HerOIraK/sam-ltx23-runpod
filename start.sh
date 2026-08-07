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


# Ensure SageAttention CUDA kernels are available for RTX 3090 / 4090
echo "📦 Verifying SageAttention CUDA kernel installation..."
python3 -c "from sageattention import sageattn_qk_int8_pv_fp8_cuda; print('SageAttention CUDA kernels loaded successfully!')" 2>/dev/null || {
    echo "⚡ Setting up CUDA build environment for SageAttention..."
    if [ ! -f /usr/local/cuda/include/cusparse.h ] && [ ! -f /usr/include/cusparse.h ]; then
        echo "Installing libcusparse-dev headers..."
        apt-get update && apt-get install -y libcusparse-dev libcublas-dev libcusolver-dev || true
    fi
    if [ ! -f /usr/local/cuda/include/cusparse.h ]; then
        CUSPARSE_HEADER=$(find /usr/include /usr/lib -name cusparse.h 2>/dev/null | head -n 1)
        if [ -n "$CUSPARSE_HEADER" ]; then
            mkdir -p /usr/local/cuda/include
            ln -sf "$CUSPARSE_HEADER" /usr/local/cuda/include/cusparse.h
        fi
    fi
    export CUDA_HOME="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CPATH="/usr/local/cuda/include:/usr/include:/usr/include/x86_64-linux-gnu:${CPATH}"
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:/usr/include:/usr/include/x86_64-linux-gnu:${CPLUS_INCLUDE_PATH}"

    echo "⚡ Compiling official SageAttention from source..."
    if ! pip install --no-cache-dir --no-build-isolation git+https://github.com/thu-ml/SageAttention.git; then
        echo "⚠️ Compilation failed; installing PyPI package with symbol compatibility bridge..."
        pip install --no-cache-dir sageattention || true
    fi
    python3 -c "import sageattention, site, os; p = os.path.join(site.getsitepackages()[0], 'sageattention', '__init__.py'); open(p, 'a').write('\n\nif not hasattr(sageattention, \"sageattn_qk_int8_pv_fp8_cuda\"): sageattn_qk_int8_pv_fp8_cuda = getattr(sageattention, \"sageattn\", None)\n')" 2>/dev/null || true
}

cd "$COMFYUI_DIR"

exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
