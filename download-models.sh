#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Model directory paths
MODEL_DIR="/workspace/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
VAE_DIR="${MODEL_DIR}/vae"
UPSCALE_DIR="${MODEL_DIR}/latent_upscale_models"
PREVIEW_DIR="${MODEL_DIR}/vae_approx"
LORA_DIR="${MODEL_DIR}/loras"

echo "Creating MiniMax H3 / SEEDHUNTER model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$VAE_DIR" "$UPSCALE_DIR" "$PREVIEW_DIR" "$LORA_DIR"

HF_TOKEN="${HF_TOKEN:-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}"
export HF_TOKEN

fetch() {
    local dest_file="$1"
    local url="$2"
    local dir
    local fname

    dir="$(dirname "$dest_file")"
    fname="$(basename "$dest_file")"
    mkdir -p "$dir"

    if [ -s "$dest_file" ]; then
        echo "[EXISTS] ${fname} is already present, skipping."
        return 0
    fi

    echo "[DOWNLOADING] ${fname} -> ${dir}..."
    aria2c \
        --console-log-level=warn \
        --summary-interval=5 \
        -x 16 -s 16 -k 1M -c \
        --dir "$dir" \
        --out "$fname" \
        ${HF_TOKEN:+--header "Authorization: Bearer ${HF_TOKEN}"} \
        "$url" \
        || echo "WARN: failed downloading ${fname}"
}

echo "=== STARTING MINIMAX H3 & SEEDHUNTER MODEL DOWNLOADS ==="

# 1. Base Diffusion Model (~14 GB)
fetch "$DIFFUSION_DIR/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"

# 2. Text Encoder (~18 GB)
fetch "$TEXT_ENC_DIR/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

# 3. Video VAE (~1.5 GB)
fetch "$VAE_DIR/minimax_h3_video_vae_fp16.safetensors" \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors"

# 4. Audio VAE (~600 MB)
fetch "$VAE_DIR/minimax_h3_audio_vae_fp32.safetensors" \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"

# 5. 3D Latent Upscaler (~1.2 GB)
fetch "$UPSCALE_DIR/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_bf16.safetensors"

# 6. Preview Approx VAE (~2 MB)
fetch "$PREVIEW_DIR/taeh3.safetensors" \
      "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors"

# 7. Turbo 8-Step LoRA (~500 MB)
fetch "$LORA_DIR/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
      "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

echo "=== ALL MINIMAX H3 & SEEDHUNTER DOWNLOADS COMPLETED ==="
