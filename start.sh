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


# Build & Compile SageAttention CUDA kernels for RTX 3090 / 4090 on container startup
echo "📦 Checking/Building SageAttention CUDA extension for RTX 3090 / 4090..."
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"
export MAX_JOBS=2
export CPATH="/usr/local/cuda/include:${CPATH}"
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="/usr/local/cuda/include:${C_INCLUDE_PATH}"
export CXX_APPEND_FLAGS="-I/usr/local/cuda/include"
export NVCC_APPEND_FLAGS="-I/usr/local/cuda/include"

python3 -c "import sageattention._fused; print('SageAttention CUDA extension loaded successfully!')" 2>/dev/null || {
    echo "⚡ Compiling SageAttention CUDA extension..."
    pip uninstall -y sageattention || true
    rm -rf /tmp/SageAttention
    if git clone --depth 1 https://github.com/woct0rdho/SageAttention.git /tmp/SageAttention; then
        (cd /tmp/SageAttention && pip install --no-cache-dir --no-build-isolation .) || echo "⚠️ SageAttention compilation failed; falling back to native attention"
        rm -rf /tmp/SageAttention
    fi
}

cd "$COMFYUI_DIR"

exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
