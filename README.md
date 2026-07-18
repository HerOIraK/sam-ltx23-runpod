# ComfyUI LTX 2.3 RunPod Template (CUDA 13.0)

This repository contains all the configuration files and scripts needed to build, publish, and deploy a custom **RunPod Community Template** for running ComfyUI with LTX 2.3 workflows, including full automation for custom node packages and model downloads.

## Repository Structure

```
sam-ltx23-runpod/
├── Dockerfile
├── start.sh
├── download-models.sh
├── README.md
├── .dockerignore
└── workflows/
    ├── ltxDirector2SEED_v10.json
    └── director_00103-audio_Reddit.json
```

## Features

- **CUDA 13.0 Base**: Utilizes the optimized `runpod/comfyui:cuda13.0` foundation.
- **Pre-installed Custom Nodes**: Automatically bakes 14 required custom node suites (such as `WhatDreamsCost-ComfyUI`, `KJNodes`, `rgthree`, `Comfyroll`, `TinyTerraNodes`, `ArtVenture`, and `CRT-Nodes`) into the Docker image, eliminating initial start delays.
- **Model Provisioning Automation**: Runs a high-speed `download-models.sh` script using `aria2c` to download large models directly to persistent volume storage on demand.
- **Persistent Symlinking**: Prevents ComfyUI files from being overwritten or lost when RunPod mounts a persistent network volume at `/workspace`.
- **Preloaded Workflows**: Ships with two optimized LTX 2.3 workflows in the user's workspace.

---

## 1. How to Build & Push the Docker Image
You can build the Docker image in two ways: **Cloud Build (Recommended)** or **Local Build**.

### Option A: Cloud Build (No downloads on your PC)
This method uses **GitHub Actions** to build the image entirely in the cloud, requiring zero local downloads or disk space.

1. **Create a GitHub Repository**: Create a new repository on GitHub (e.g., `sam-ltx23-runpod`).
2. **Add Secrets**: In your GitHub repository, go to **Settings → Secrets and variables → Actions** and add two Secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username (`mkv420`).
   - `DOCKERHUB_TOKEN`: A Personal Access Token from Docker Hub settings (not your password).
3. **Push to GitHub**: Push these files to your repository. The workflow in `.github/workflows/docker-image.yml` will automatically trigger, build the container, and push it directly to your Docker Hub repository.

### Option B: Local Build (Requires Docker Desktop)
From the root directory of this repository:

1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t mkv420/sam-ltx23-comfyui:1.0.0 .
   ```
   *(Note: The build process clones all custom node repositories and installs Python packages. This may take several minutes.)*

3. **Push the Image**:
   ```bash
   docker push mkv420/sam-ltx23-comfyui:1.0.0
   ```

---

## 2. Setting Up the RunPod Template

Go to the **RunPod Console → Templates → New Template** and configure the fields as follows:

| Field | Configuration Value |
| :--- | :--- |
| **Template Name** | `Sam LTX 2.3 ComfyUI (CUDA 13.0)` |
| **Template Type** | `Pods` |
| **Compute Type** | `NVIDIA GPU` |
| **Container Image** | `mkv420/sam-ltx23-comfyui:1.0.0` |
| **Container Disk** | `30 GB` |
| **Volume Disk** | `100 - 150 GB` |
| **Volume Mount Path** | `/workspace` |
| **Expose HTTP Port** | `8188` |
| **Start Command** | *Leave Empty* (allows container's own `CMD ["/start.sh"]` to run) |

### Environment Variables

Under the **Environment Variables** section of the template config, add:

1. **Enable Model Auto-Download**:
   - **Key**: `AUTO_DOWNLOAD_MODELS`
   - **Value**: `true`
2. **Hugging Face Authentication Token (Gated Access)**:
   - **Key**: `HF_TOKEN`
   - **Value**: `your_huggingface_read_token` *(Needed to download gated weights like Google Gemma 3 and Lightricks LTX 2.3 upscalers).*

---

## 3. Included Workflows

Workflows are copied automatically to `/workspace/user/default/workflows/` on launch:

1. **`ltxDirector2SEED_v10.json`**: An all-in-one timeline editing workflow utilizing the LTX Director node, supporting frame scheduling, camera prompts, and prompt relaying.
2. **`director_00103-audio_Reddit.json`**: An advanced multi-stage timeline workflow featuring ControlNet, custom audio reference inputs, lipsync mapping, and RTX Video Super Resolution upscaling.

---

## 4. Installed Custom Nodes

The image includes the following node packages cloned directly into `/opt/ComfyUI/custom_nodes`:
- `ComfyUI-LTXVideo` (LTX video core nodes)
- `WhatDreamsCost-ComfyUI` (LTX Director core)
- `ComfyUI-KJNodes` (Loaders, VAE selectors, etc.)
- `ComfyUI-VideoHelperSuite` (Video load/save pipelines)
- `rgthree-comfy` (Bypasser switch clusters)
- `ComfyUI-Impact-Pack` (General flow pipelines)
- `ComfyUI-Easy-Use` (UI nodes & pipe relays)
- `ComfyUI-mxToolkit` (mxSliders and UI utilities)
- `ComfyUI_tinyterraNodes` (TinyTerra nodes)
- `ComfyUI_Comfyroll_CustomNodes` (Comfyroll Studio nodes)
- `Nvidia_RTX_Nodes_ComfyUI` (NVIDIA RTX Video Super Resolution)
- `comfyui-art-venture` (Art Venture preprocess pipelines)
- `CRT-Nodes` (Graph utils, WAN & LTX samplers)
- `ComfyUI-Manager` (Node update, verification, and diagnostics management)
