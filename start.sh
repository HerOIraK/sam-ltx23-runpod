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


# Build & Compile SageAttention CUDA kernels for RTX 4090 on container startup
echo "📦 Checking/Building SageAttention CUDA extension for RTX 4090..."
python3 -c "import sageattention; print('SageAttention module loaded successfully!')" 2>/dev/null || {
    echo "⚡ Compiling SageAttention CUDA extension..."
    (cd /tmp && git clone --depth 1 https://github.com/thu-ml/SageAttention.git && cd SageAttention && TORCH_CUDA_ARCH_LIST="8.9" python3 setup.py install && rm -rf /tmp/SageAttention) || true
}

cd "$COMFYUI_DIR"

exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
