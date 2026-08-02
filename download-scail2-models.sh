#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1

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

HF_TOKEN="${HF_TOKEN:-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}"
export HF_TOKEN

download_file() {
    local repo_id="$1"
    local repo_filename="$2"
    local dest_dir="$3"
    local target_filename="$4"

    if [ -f "${dest_dir}/${target_filename}" ]; then
        echo "[EXISTS] ${target_filename} is already present in ${dest_dir}, skipping download."
    else
        echo "[DOWNLOADING HF] ${repo_id}/${repo_filename} -> ${dest_dir}/${target_filename}..."
        python3 -c "
import os, sys
os.environ['PYTHONUNBUFFERED'] = '1'
from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
filename = sys.argv[2]
local_dir = sys.argv[3]
token = os.getenv('HF_TOKEN')

print(f'--> Fetching {filename} from {repo_id}...', flush=True)
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    token=token
)
print(f'--> Completed {filename}.', flush=True)
" "$repo_id" "$repo_filename" "$dest_dir"

        if [ -f "${dest_dir}/${repo_filename}" ] && [ "${repo_filename}" != "${target_filename}" ]; then
            mv "${dest_dir}/${repo_filename}" "${dest_dir}/${target_filename}"
            local top_subfolder
            top_subfolder=$(echo "$repo_filename" | cut -d'/' -f1)
            if [ -d "${dest_dir}/${top_subfolder}" ] && [ "$top_subfolder" != "$repo_filename" ]; then
                rm -rf "${dest_dir}/${top_subfolder}" 2>/dev/null || true
            fi
        fi
    fi
}

echo "=== STARTING SCAIL-2 MODEL DOWNLOADS WITH LIVE PROGRESS BAR ==="

# 1. SCAIL-2 Diffusion Model
echo "[1/8] SCAIL-2 Diffusion Model (wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors)"
download_file \
    "Comfy-Org/SCAIL-2" \
    "split_files/diffusion_models/wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors" \
    "$DIFFUSION_DIR" \
    "wan2.1_14B_SCAIL_2_fp8_e4m3fn.safetensors"

# 2. UMT5 XXL Text Encoder
echo "[2/8] UMT5 XXL Text Encoder (umt5_xxl_fp8_e4m3fn.safetensors)"
download_file \
    "Comfy-Org/SCAIL-2" \
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn.safetensors" \
    "$TEXT_ENC_DIR" \
    "umt5_xxl_fp8_e4m3fn.safetensors"

# 3. CLIP Vision Model
echo "[3/8] CLIP Vision Model (clip_vision_h.safetensors)"
download_file \
    "Comfy-Org/SCAIL-2" \
    "split_files/clip_vision/clip_vision_h.safetensors" \
    "$CLIP_VISION_DIR" \
    "clip_vision_h.safetensors"

# 4. Wan2.1 VAE
echo "[4/8] Wan2.1 VAE (Wan2_1_VAE_bf16.safetensors)"
download_file \
    "Comfy-Org/SCAIL-2" \
    "split_files/vae/Wan2_1_VAE_bf16.safetensors" \
    "$VAE_DIR" \
    "Wan2_1_VAE_bf16.safetensors"

# 5. SAM 3.1 Checkpoint
echo "[5/8] SAM 3.1 Checkpoint (sam3.1_multiplex_fp16.safetensors)"
download_file \
    "Comfy-Org/sam3.1" \
    "sam3.1_multiplex_fp16.safetensors" \
    "$SAM_DIR" \
    "sam3.1_multiplex_fp16.safetensors"

# 6. LightX2V LoRA
echo "[6/8] LightX2V LoRA (lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors)"
download_file \
    "Kijai/WanVideo_comfy" \
    "LightX2v/lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors" \
    "$LORA_DIR" \
    "lightx2v_I2V_14B_480p_cfg_step_dpo_lora_rank64_bf16.safetensors"

# 7. PusaV1 LoRA
echo "[7/8] PusaV1 LoRA (Wan21_PusaV1_LoRA_rank512_bf16.safetensors)"
download_file \
    "Kijai/WanVideo_comfy" \
    "Wan21_PusaV1_LoRA_rank512_bf16.safetensors" \
    "$LORA_DIR" \
    "Wan21_PusaV1_LoRA_rank512_bf16.safetensors"

# 8. RIFE 4.9 Model
echo "[8/8] RIFE 4.9 Model (rife49.pth)"
download_file \
    "VMTamashii/rife49" \
    "rife49.pth" \
    "$RIFE_DIR" \
    "rife49.pth"

echo "=== ALL SCAIL-2 MODEL DOWNLOADS COMPLETED ==="
