#!/bin/bash
set -e

MODELS="/workspace/models"

mkdir -p \
    "$MODELS/diffusion_models" \
    "$MODELS/text_encoders" \
    "$MODELS/vae" \
    "$MODELS/latent_upscale_models" \
    "$MODELS/loras/LTX2"

download_file() {
    URL="$1"
    OUTPUT="$2"

    if [ -f "$OUTPUT" ] && [ ! -f "${OUTPUT}.aria2" ]; then
        echo "Already downloaded: $OUTPUT"
        return
    fi

    echo "Downloading: $OUTPUT"

    ARIA_ARGS=(
        --continue=true
        --max-connection-per-server=16
        --split=16
        --min-split-size=1M
        --auto-file-renaming=false
    )

    if [ -n "${HF_TOKEN:-}" ]; then
        ARIA_ARGS+=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi

    aria2c \
        "${ARIA_ARGS[@]}" \
        --dir="$(dirname "$OUTPUT")" \
        --out="$(basename "$OUTPUT")" \
        "$URL"
}

echo "Downloading LTX 2.3 model package..."

# 1. Diffusion Models
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" \
    "$MODELS/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"

# 2. Text Encoders & Projections
download_file \
    "https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors" \
    "$MODELS/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"

download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$MODELS/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

# 3. VAE Models
download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" \
    "$MODELS/vae/taeltx2_3.safetensors"

download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
    "$MODELS/vae/LTX23_audio_vae_bf16.safetensors"

download_file \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
    "$MODELS/vae/LTX23_video_vae_bf16.safetensors"

# 4. Latent Upscale Models
download_file \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$MODELS/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# 5. IC-LoRA Union Control Lora
download_file \
    "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    "$MODELS/loras/LTX2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

echo "Model provisioning completed."
