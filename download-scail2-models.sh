#!/bin/bash
set -e

# Model directory paths
MODEL_DIR="/workspace/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
CLIP_VIS_DIR="${MODEL_DIR}/clip_vision"
VAE_DIR="${MODEL_DIR}/vae"
CKPT_DIR="${MODEL_DIR}/checkpoints"
LORA_DIR="${MODEL_DIR}/loras"
LIGHTX2V_LORA_DIR="${MODEL_DIR}/loras/Lightx2v"
RIFE_DIR="${MODEL_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"

echo "Creating model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$CLIP_VIS_DIR" "$VAE_DIR" "$CKPT_DIR" "$LORA_DIR" "$LIGHTX2V_LORA_DIR" "$RIFE_DIR"

download_file() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    if [ -f "${dest_dir}/${filename}" ]; then
        echo "[EXISTS] ${filename} already present in ${dest_dir}, skipping."
    else
        echo "[DOWNLOADING] ${filename} to ${dest_dir}..."
        aria2c -x 16 -s 16 -k 1M --console-log-level=warn --summary-interval=10 -d "$dest_dir" -o "$filename" "$url"
    fi
}

echo "=== STARTING SCAIL-2 MODEL DOWNLOADS ==="

# 1. SCAIL-2 Diffusion Model
download_file "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors" "$DIFFUSION_DIR" "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"

# 2. Text Encoder (UMT5 XXL)
download_file "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$TEXT_ENC_DIR" "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# 3. CLIP Vision
download_file "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" "$CLIP_VIS_DIR" "clip_vision_h.safetensors"

# 4. VAE
download_file "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "$VAE_DIR" "Wan2_1_VAE_bf16.safetensors"

# 5. SAM 3.1 Multiplex Checkpoint
download_file "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors" "$CKPT_DIR" "sam3.1_multiplex_fp16.safetensors"

# 6. LightX2V Distill LoRA
download_file "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors" "$LIGHTX2V_LORA_DIR" "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

# 7. PusaV1 LoRA
download_file "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors" "$LORA_DIR" "Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors"

# 8. RIFE 49 Model
download_file "https://huggingface.co/VMTamashii/rife49/resolve/main/rife49.pth" "$RIFE_DIR" "rife49.pth"

echo "=== ALL SCAIL-2 MODEL DOWNLOADS COMPLETED ==="
