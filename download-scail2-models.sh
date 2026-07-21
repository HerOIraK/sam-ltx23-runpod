#!/bin/bash
set -e

# Model directory paths under /workspace/ComfyUI/models
MODEL_DIR="/workspace/ComfyUI/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
CLIP_VISION_DIR="${MODEL_DIR}/clip_vision"
VAE_DIR="${MODEL_DIR}/vae"
SAM_DIR="${MODEL_DIR}/checkpoints"
LORA_DIR="${MODEL_DIR}/loras"
RIFE_DIR="/workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"

echo "Creating SCAIL-2 model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$CLIP_VISION_DIR" "$VAE_DIR" "$SAM_DIR" "$LORA_DIR" "$RIFE_DIR"

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

echo "=== STARTING SCAIL-2 MODEL DOWNLOADS WITH LIVE STATUS ==="

# 1. SCAIL-2 Diffusion Model
echo "[1/8] SCAIL-2 Diffusion Model (wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors)"
download_file \
    "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/split_files/diffusion_models/wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors" \
    "$DIFFUSION_DIR" \
    "wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors"

# 2. UMT5 XXL Text Encoder
echo "[2/8] UMT5 XXL Text Encoder (umt5_xxl_fp8_e4m3fn.safetensors)"
download_file \
    "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn.safetensors" \
    "$TEXT_ENC_DIR" \
    "umt5_xxl_fp8_e4m3fn.safetensors"

# 3. CLIP Vision Model
echo "[3/8] CLIP Vision Model (clip_vision_h.safetensors)"
download_file \
    "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "$CLIP_VISION_DIR" \
    "clip_vision_h.safetensors"

# 4. Wan2.1 VAE
echo "[4/8] Wan2.1 VAE (Wan2_1_VAE_bf16.safetensors)"
download_file \
    "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/split_files/vae/Wan2_1_VAE_bf16.safetensors" \
    "$VAE_DIR" \
    "Wan2_1_VAE_bf16.safetensors"

# 5. SAM 3.1 Checkpoint
echo "[5/8] SAM 3.1 Checkpoint (sam3.1_multiplex_fp16.safetensors)"
download_file \
    "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/sam3.1_multiplex_fp16.safetensors" \
    "$SAM_DIR" \
    "sam3.1_multiplex_fp16.safetensors"

# 6. LightX2V LoRA
echo "[6/8] LightX2V LoRA (lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors)"
download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LightX2v/lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors" \
    "$LORA_DIR" \
    "lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors"

# 7. PusaV1 LoRA
echo "[7/8] PusaV1 LoRA (Wan21_PusaV1_LoRA_rank512_bf16.safetensors)"
download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_PusaV1_LoRA_rank512_bf16.safetensors" \
    "$LORA_DIR" \
    "Wan21_PusaV1_LoRA_rank512_bf16.safetensors"

# 8. RIFE 4.9 Model
echo "[8/8] RIFE 4.9 Model (rife49.pth)"
download_file \
    "https://huggingface.co/VMTamashii/rife49/resolve/main/rife49.pth" \
    "$RIFE_DIR" \
    "rife49.pth"

echo "=== ALL SCAIL-2 MODEL DOWNLOADS COMPLETED ==="
