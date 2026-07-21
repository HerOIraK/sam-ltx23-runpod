#!/bin/bash
set -e

# Model directory paths under /workspace/ComfyUI/models
MODEL_DIR="/workspace/ComfyUI/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
VAE_DIR="${MODEL_DIR}/vae"
UPSCALE_DIR="${MODEL_DIR}/latent_upscale_models"
LORA_DIR="${MODEL_DIR}/loras/LTX2"

echo "Creating LTX 2.3 model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$VAE_DIR" "$UPSCALE_DIR" "$LORA_DIR"

HF_TOKEN="${HF_TOKEN:-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}"

download_file() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    if [ -f "${dest_dir}/${filename}" ] && [ ! -f "${dest_dir}/${filename}.aria2" ]; then
        echo "[EXISTS] ${filename} is already present in ${dest_dir}, skipping download."
    else
        echo "[DOWNLOADING] ${filename} -> ${dest_dir}..."
        aria2c \
            -x 16 -s 16 -k 1M \
            --continue=true \
            --auto-file-renaming=false \
            --console-log-level=notice \
            --summary-interval=5 \
            ${HF_TOKEN:+--header="Authorization: Bearer ${HF_TOKEN}"} \
            -d "$dest_dir" \
            -o "$filename" \
            "$url"
    fi
}

echo "=== STARTING LTX 2.3 MODEL DOWNLOADS WITH LIVE STATUS ==="

# 1. Base Video Model (~23 GB)
echo "[1/8] Base Video Model (ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors)"
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" \
    "$DIFFUSION_DIR" \
    "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"

# 2. Text Encoders & Projections (~12 GB)
echo "[2/8] Gemma 3 Text Encoder (gemma_3_12B_it_fp8_e4m3fn.safetensors)"
download_file \
    "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors" \
    "$TEXT_ENC_DIR" \
    "gemma_3_12B_it_fp8_e4m3fn.safetensors"

echo "[3/8] Text Projection (ltx-2.3_text_projection_bf16.safetensors)"
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$TEXT_ENC_DIR" \
    "ltx-2.3_text_projection_bf16.safetensors"

# 3. VAE Models (~500 MB)
echo "[4/8] Video VAE (LTX23_video_vae_bf16.safetensors)"
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
    "$VAE_DIR" \
    "LTX23_video_vae_bf16.safetensors"

echo "[5/8] Audio VAE (LTX23_audio_vae_bf16.safetensors)"
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
    "$VAE_DIR" \
    "LTX23_audio_vae_bf16.safetensors"

echo "[6/8] Preview VAE (taeltx2_3.safetensors)"
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" \
    "$VAE_DIR" \
    "taeltx2_3.safetensors"

# 4. Latent Upscale Model (~1.2 GB)
echo "[7/8] Video Spatial Upscaler (ltx-2.3-spatial-upscaler-x2-1.1.safetensors)"
download_file \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$UPSCALE_DIR" \
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# 5. IC-LoRA Union Control (~620 MB)
echo "[8/8] Motion Control LoRA (ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)"
download_file \
    "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    "$LORA_DIR" \
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

echo "=== ALL LTX 2.3 MODEL DOWNLOADS COMPLETED ==="
