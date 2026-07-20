#!/bin/bash
set -e

# Model directory paths
MODEL_DIR="/workspace/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
VAE_DIR="${MODEL_DIR}/vae"
UPSCALE_DIR="${MODEL_DIR}/latent_upscale_models"
LORA_DIR="${MODEL_DIR}/loras/LTX2"

echo "Creating LTX 2.3 model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$VAE_DIR" "$UPSCALE_DIR" "$LORA_DIR"

download_file() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    if [ -f "${dest_dir}/${filename}" ] && [ ! -f "${dest_dir}/${filename}.aria2" ]; then
        echo "[EXISTS] ${filename} already present in ${dest_dir}, skipping."
    else
        echo "[DOWNLOADING] ${filename} to ${dest_dir}..."
        aria2c -x 16 -s 16 -k 1M --console-log-level=warn --summary-interval=10 -d "$dest_dir" -o "$filename" "$url"
    fi
}

echo "=== STARTING LTX 2.3 MODEL DOWNLOADS ==="

# 1. Diffusion Model
download_file "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" "$DIFFUSION_DIR" "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"

# 2. Text Encoders & Projections
download_file "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors" "$TEXT_ENC_DIR" "gemma_3_12B_it_fp8_e4m3fn.safetensors"
download_file "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" "$TEXT_ENC_DIR" "ltx-2.3_text_projection_bf16.safetensors"

# 3. VAE Models
download_file "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" "$VAE_DIR" "taeltx2_3.safetensors"
download_file "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" "$VAE_DIR" "LTX23_audio_vae_bf16.safetensors"
download_file "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" "$VAE_DIR" "LTX23_video_vae_bf16.safetensors"

# 4. Latent Upscale Model
download_file "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$UPSCALE_DIR" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# 5. IC-LoRA Union Control
download_file "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" "$LORA_DIR" "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

echo "=== ALL LTX 2.3 MODEL DOWNLOADS COMPLETED ==="
