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

# Enable Hugging Face Ultra-Fast Rust Parallel Transfer Engine
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-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}"

echo "=== STARTING LTX 2.3 MODEL DOWNLOADS (MAX UNCAPPED SPEED) ==="

# 1. Base Video Model (~23 GB)
echo "[1/8] Downloading Base Video Model..."
hf download Kijai/LTX2.3_comfy diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors --local-dir "$DIFFUSION_DIR" --local-dir-use-symlinks False

# 2. Text Encoders & Projections (~12 GB)
echo "[2/8] Downloading Gemma 3 Text Encoder..."
hf download GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn gemma_3_12B_it_fp8_e4m3fn.safetensors --local-dir "$TEXT_ENC_DIR" --local-dir-use-symlinks False

echo "[3/8] Downloading Text Projection..."
hf download Kijai/LTX2.3_comfy text_encoders/ltx-2.3_text_projection_bf16.safetensors --local-dir "$TEXT_ENC_DIR" --local-dir-use-symlinks False

# 3. VAE Models (~500 MB)
echo "[4/8] Downloading Video VAE..."
hf download Kijai/LTX2.3_comfy vae/LTX23_video_vae_bf16.safetensors --local-dir "$VAE_DIR" --local-dir-use-symlinks False

echo "[5/8] Downloading Audio VAE..."
hf download Kijai/LTX2.3_comfy vae/LTX23_audio_vae_bf16.safetensors --local-dir "$VAE_DIR" --local-dir-use-symlinks False

echo "[6/8] Downloading Preview VAE..."
hf download Kijai/LTX2.3_comfy vae/taeltx2_3.safetensors --local-dir "$VAE_DIR" --local-dir-use-symlinks False

# 4. Latent Upscale Model (~1.2 GB)
echo "[7/8] Downloading Video Spatial Upscaler..."
hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$UPSCALE_DIR" --local-dir-use-symlinks False

# 5. IC-LoRA Union Control (~620 MB)
echo "[8/8] Downloading Motion Control LoRA..."
hf download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors --local-dir "$LORA_DIR" --local-dir-use-symlinks False

echo "=== ALL LTX 2.3 MODEL DOWNLOADS COMPLETED ==="
