#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1

# Model directory paths
MODEL_DIR="/workspace/models"
DIFFUSION_DIR="${MODEL_DIR}/diffusion_models"
TEXT_ENC_DIR="${MODEL_DIR}/text_encoders"
VAE_DIR="${MODEL_DIR}/vae"
UPSCALE_DIR="${MODEL_DIR}/latent_upscale_models"
LORA_DIR="${MODEL_DIR}/loras/LTX2"

echo "Creating LTX 2.3 model target directories..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENC_DIR" "$VAE_DIR" "$UPSCALE_DIR" "$LORA_DIR"

HF_TOKEN="${HF_TOKEN:-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}"
export HF_TOKEN
CIVITAI_API_KEY="${CIVITAI_API_KEY:-42dcfd655ef61d81453c77102f3b64ac}"

download_hf() {
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

download_civitai() {
    local version_id="$1"
    local dest_dir="$2"
    local target_filename="$3"

    if [ -f "${dest_dir}/${target_filename}" ]; then
        echo "[EXISTS] ${target_filename} is already present in ${dest_dir}, skipping download."
    else
        echo "[DOWNLOADING CIVITAI] Version ${version_id} -> ${dest_dir}/${target_filename}..."
        aria2c \
            --console-log-level=warn \
            --summary-interval=5 \
            -x 16 -s 16 -k 1M \
            --header="Authorization: Bearer ${CIVITAI_API_KEY}" \
            -d "${dest_dir}" \
            -o "${target_filename}" \
            "https://civitai.com/api/download/models/${version_id}"
    fi
}

echo "=== STARTING LTX 2.3 BASE & CIVITAI MODEL DOWNLOADS ==="

# 1. Base Video Model (~23 GB)
echo "[1/14] Base Video Model (ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors)"
download_hf \
    "Kijai/LTX2.3_comfy" \
    "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" \
    "$DIFFUSION_DIR" \
    "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"

# 2. Text Encoders & Projections (~12.5 GB)
echo "[2/14] Gemma 3 Text Encoder (gemma_3_12B_it_fp8_e4m3fn.safetensors)"
download_hf \
    "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn" \
    "gemma_3_12B_it_fp8_e4m3fn.safetensors" \
    "$TEXT_ENC_DIR" \
    "gemma_3_12B_it_fp8_e4m3fn.safetensors"

echo "[3/14] Text Projection (ltx-2.3_text_projection_bf16.safetensors)"
download_hf \
    "Kijai/LTX2.3_comfy" \
    "text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$TEXT_ENC_DIR" \
    "ltx-2.3_text_projection_bf16.safetensors"

# 3. VAE Models (~1 GB)
echo "[4/14] Video VAE (LTX23_video_vae_bf16.safetensors)"
download_hf \
    "Kijai/LTX2.3_comfy" \
    "vae/LTX23_video_vae_bf16.safetensors" \
    "$VAE_DIR" \
    "LTX23_video_vae_bf16.safetensors"

echo "[5/14] Audio VAE (LTX23_audio_vae_bf16.safetensors)"
download_hf \
    "Kijai/LTX2.3_comfy" \
    "vae/LTX23_audio_vae_bf16.safetensors" \
    "$VAE_DIR" \
    "LTX23_audio_vae_bf16.safetensors"

echo "[6/14] Preview VAE (taeltx2_3.safetensors)"
download_hf \
    "Kijai/LTX2.3_comfy" \
    "vae/taeltx2_3.safetensors" \
    "$VAE_DIR" \
    "taeltx2_3.safetensors"

# 4. Latent Upscale Model (~1.2 GB)
echo "[7/14] Video Spatial Upscaler (ltx-2.3-spatial-upscaler-x2-1.1.safetensors)"
download_hf \
    "Lightricks/LTX-2.3" \
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$UPSCALE_DIR" \
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# 5. IC-LoRA Union Control (~620 MB)
echo "[8/14] Motion Control LoRA (ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)"
download_hf \
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control" \
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    "$LORA_DIR" \
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

# 6. Civitai Checkpoints (~56.7 GB)
echo "[9/14] Civitai Checkpoint 1 (ltx2310eros_v14.safetensors)"
download_civitai "3109610" "$DIFFUSION_DIR" "ltx2310eros_v14.safetensors"

echo "[10/14] Civitai Checkpoint 2 (DasiwaLTX23_dragonleapV4.safetensors)"
download_civitai "3092188" "$DIFFUSION_DIR" "DasiwaLTX23_dragonleapV4.safetensors"

# 7. Civitai LoRAs (~6.2 GB)
echo "[11/14] Civitai LoRA 1 (DR34ML4Y_LT3X_V3.safetensors)"
download_civitai "3082662" "$LORA_DIR" "DR34ML4Y_LT3X_V3.safetensors"

echo "[12/14] Civitai LoRA 2 (LTX2.3_reasoning_Sulphur-2_I2V_V4.safetensors)"
download_civitai "3025398" "$LORA_DIR" "LTX2.3_reasoning_Sulphur-2_I2V_V4.safetensors"

echo "[13/14] Civitai LoRA 3 (Sulphur_LTX23_better_NSFW_motion.safetensors)"
download_civitai "2986751" "$LORA_DIR" "Sulphur_LTX23_better_NSFW_motion.safetensors"

echo "[14/14] Civitai LoRAs 4, 5, 6 (Enhancers & Physics)"
download_civitai "2849716" "$LORA_DIR" "LTX2.3_Crisp_Enhance.safetensors"
download_civitai "2996907" "$LORA_DIR" "DaSiWa_LTX23_NSFW_Bodyphysics_Fluid_Motion_Enhancer_v01.safetensors"
download_civitai "2952846" "$LORA_DIR" "LTX2.3_Physics_V2_000002000.safetensors"

echo "=== ALL LTX 2.3 BASE & CIVITAI DOWNLOADS COMPLETED ==="
