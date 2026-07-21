#!/bin/bash
set -e

COMFYUI_DIR="/workspace/ComfyUI"
BAKED_COMFYUI="/opt/ComfyUI"
VOLUME_DIR="/workspace"

echo "Starting Sam LTX 2.3 ComfyUI template"

# 1. Ensure /workspace/ComfyUI exists and is fully populated
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "Initializing ComfyUI installation at $COMFYUI_DIR..."
    cp -a "$BAKED_COMFYUI" "$COMFYUI_DIR"
else
    echo "Updating custom nodes in $COMFYUI_DIR..."
    # Sync pre-baked custom nodes from image into /workspace/ComfyUI/custom_nodes
    cp -rn "$BAKED_COMFYUI/custom_nodes/"* "$COMFYUI_DIR/custom_nodes/" 2>/dev/null || true
fi

# 2. Ensure model directories exist under /workspace/ComfyUI
mkdir -p \
    "$COMFYUI_DIR/models" \
    "$COMFYUI_DIR/input" \
    "$COMFYUI_DIR/output" \
    "$COMFYUI_DIR/user/default/workflows" \
    "$COMFYUI_DIR/user/__manager" \
    "$VOLUME_DIR/downloads"

# Symlink top-level /workspace/models, input, output, user to /workspace/ComfyUI for convenience
ln -sfn "$COMFYUI_DIR/models" "$VOLUME_DIR/models"
ln -sfn "$COMFYUI_DIR/input" "$VOLUME_DIR/input"
ln -sfn "$COMFYUI_DIR/output" "$VOLUME_DIR/output"
ln -sfn "$COMFYUI_DIR/user" "$VOLUME_DIR/user"

# Copy default workflows and settings if not present
cp -n "$BAKED_COMFYUI/user/default/workflows/"*.json "$COMFYUI_DIR/user/default/workflows/" 2>/dev/null || true
cp -n "$BAKED_COMFYUI/user/default/comfy.settings.json" "$COMFYUI_DIR/user/default/comfy.settings.json" 2>/dev/null || true
cp -n "$BAKED_COMFYUI/user/__manager/config.ini" "$COMFYUI_DIR/user/__manager/config.ini" 2>/dev/null || true

# 3. Auto-download models if requested
if [ "${AUTO_DOWNLOAD_MODELS:-false}" = "true" ] || [ "${AUTO_DOWNLOAD_LTX_MODELS:-false}" = "true" ]; then
    /download-ltx-models.sh
fi

if [ "${AUTO_DOWNLOAD_SCAIL2_MODELS:-false}" = "true" ]; then
    /download-scail2-models.sh
fi

# 4. Start VS Code code-server on port 8000 in background
echo "Starting code-server on port 8000..."
mkdir -p /workspace/code-server
rm -f /workspace/root-fs || true
ln -sf / /workspace/root-fs || true
nohup code-server --bind-addr 0.0.0.0:8000 --auth none --user-data-dir /workspace/code-server /workspace &

# 5. Launch ComfyUI directly from /workspace/ComfyUI
cd "$COMFYUI_DIR"

exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --enable-manager
