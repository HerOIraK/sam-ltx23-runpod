#!/bin/bash
set -e

# Model directory paths
MODEL_DIR="/workspace/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
CLIP_VISION_DIR="${MODEL_DIR}/clip_vision"
VAE_DIR="${MODEL_DIR}/vae"
SAM_DIR="${MODEL_DIR}/checkpoints"
LORA_DIR="${MODEL_DIR}/loras"
RIFE_DIR="${MODEL_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"

echo "Creating SCAIL-2 model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$CLIP_VISION_DIR" "$VAE_DIR" "$SAM_DIR" "$LORA_DIR" "$RIFE_DIR"

# Enable Hugging Face High-Performance Engine
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN="${HF_TOKEN:-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}"

echo "=== STARTING SCAIL-2 MODEL DOWNLOADS (MAX UNCAPPED SPEED) ==="

# 1. SCAIL-2 Diffusion Model
echo "[1/8] Downloading SCAIL-2 Diffusion Model..."
hf download Comfy-Org/SCAIL-2 split_files/diffusion_models/wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors --local-dir "$DIFFUSION_DIR"

# 2. UMT5 XXL Text Encoder
echo "[2/8] Downloading UMT5 XXL Text Encoder..."
hf download Comfy-Org/SCAIL-2 split_files/text_encoders/umt5_xxl_fp8_e4m3fn.safetensors --local-dir "$TEXT_ENC_DIR"

# 3. CLIP Vision Model
echo "[3/8] Downloading CLIP Vision Model..."
hf download Comfy-Org/SCAIL-2 split_files/clip_vision/clip_vision_h.safetensors --local-dir "$CLIP_VISION_DIR"

# 4. Wan2.1 VAE
echo "[4/8] Downloading Wan2.1 VAE..."
hf download Comfy-Org/SCAIL-2 split_files/vae/Wan2_1_VAE_bf16.safetensors --local-dir "$VAE_DIR"

# 5. SAM 3.1 Checkpoint
echo "[5/8] Downloading SAM 3.1 Checkpoint..."
hf download Comfy-Org/sam3.1 sam3.1_multiplex_fp16.safetensors --local-dir "$SAM_DIR"

# 6. LightX2V LoRA
echo "[6/8] Downloading LightX2V LoRA..."
hf download Kijai/WanVideo_comfy LightX2v/lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors --local-dir "$LORA_DIR"

# 7. PusaV1 LoRA
echo "[7/8] Downloading PusaV1 LoRA..."
hf download Kijai/WanVideo_comfy Wan21_PusaV1_LoRA_rank512_bf16.safetensors --local-dir "$LORA_DIR"

# 8. RIFE 4.9 Model
echo "[8/8] Downloading RIFE 4.9 Model..."
hf download VMTamashii/rife49 rife49.pth --local-dir "$RIFE_DIR"

echo "=== ALL SCAIL-2 MODEL DOWNLOADS COMPLETED ==="
