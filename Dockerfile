ARG BASE=runpod/comfyui:cuda13.0

# ===========================================================================
# STAGE 1 - Builder Stage: Compile SageAttention wheel and discard toolchain
# ===========================================================================
FROM ${BASE} AS sagebuilder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ARG CUDA_PKG=13-0
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ca-certificates build-essential ninja-build \
        cuda-nvcc-${CUDA_PKG} \
        cuda-cudart-dev-${CUDA_PKG} \
        cuda-profiler-api-${CUDA_PKG} \
        libcusparse-dev-${CUDA_PKG} \
        libcublas-dev-${CUDA_PKG} \
        libcusolver-dev-${CUDA_PKG} \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/cuda/include
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include

RUN set -eux; \
    nvcc --version; \
    HDR="$(find /usr /usr/local -name cusparse.h -print -quit)"; \
    test -n "$HDR"; \
    echo "cusparse.h -> $HDR"

ARG TORCH_CUDA_ARCH_LIST="8.6 8.9"
ARG MAX_JOBS=4
ARG EXT_PARALLEL=1
ARG NVCC_THREADS=2
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=main

RUN set -eux; \
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"; \
    export MAX_JOBS="${MAX_JOBS}"; \
    export EXT_PARALLEL="${EXT_PARALLEL}"; \
    export NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}"; \
    echo "Building SageAttention wheel: arch=${TORCH_CUDA_ARCH_LIST} jobs=${MAX_JOBS}"; \
    mkdir -p /opt/wheels; \
    pip wheel --no-cache-dir --no-build-isolation --no-deps -w /opt/wheels \
        "git+${SAGE_REPO}@${SAGE_REF}"; \
    ls -la /opt/wheels; \
    test -n "$(ls /opt/wheels/sageattention-*.whl 2>/dev/null)"

# ===========================================================================
# STAGE 2 - Runtime Stage: Clean image with prebuilt SageAttention wheel
# ===========================================================================
FROM ${BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget aria2 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=sagebuilder /opt/wheels /opt/wheels
RUN pip install --no-cache-dir /opt/wheels/sageattention-*.whl

# Patch 1: Robust SageAttention non-tautological verification
RUN python3 - <<'PY'
import sys, torch, sageattention
from sageattention import core

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)

if not hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda"):
    sys.exit("FATAL: fp8 CUDA symbol missing -- pip fell back to sageattention 1.0.x")

flags = {n: getattr(core, n, None) for n in ("SM80_ENABLED", "SM86_ENABLED", "SM89_ENABLED", "SM90_ENABLED")}
print("compiled arch flags:", flags)

missing = [n for n in ("SM86_ENABLED", "SM89_ENABLED") if not flags.get(n)]
if missing:
    sys.exit(f"FATAL: SageAttention built WITHOUT {missing}. Check TORCH_CUDA_ARCH_LIST.")

print("Stage 2 SageAttention initial verification PASSED")
PY

# Copy baked ComfyUI to /opt/ComfyUI
RUN mkdir -p /opt && cp -a /opt/comfyui-baked /opt/ComfyUI

# Patch 4a: Pin ComfyUI core version
ARG COMFYUI_REF=v0.31.0
RUN git config --global --add safe.directory /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git fetch --tags --depth 1 origin "${COMFYUI_REF}" \
    && git checkout "${COMFYUI_REF}" \
    && pip install --no-cache-dir --upgrade comfyui-frontend-package comfyui-manager huggingface_hub[cli] hf_transfer

WORKDIR /opt/ComfyUI/custom_nodes

RUN rm -rf ComfyUI-LTXVideo WhatDreamsCost-ComfyUI ComfyUI-KJNodes ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-Impact-Pack ComfyUI-Manager ComfyUI-Easy-Use ComfyUI-mxToolkit ComfyUI_tinyterraNodes ComfyUI_Comfyroll_CustomNodes Nvidia_RTX_Nodes_ComfyUI comfyui-art-venture CRT-Nodes ComfyUI-DaSiWa-Nodes comfyui_controlnet_aux ComfyUI-Frame-Interpolation Civicomfy ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Lora-Manager ComfyUI_Steudio ComfyUI-Pixaroma

# Clone required custom node packs (Patch 5: Pinned Spectrum, deleted conflicting ports)
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    python3 -c "f = 'ComfyUI-LTXVideo/pyramid_blending.py'; c = open(f).read().replace('    pad,\n)', ')\nfrom torch.nn.functional import pad'); open(f, 'w').write(c)" && \
    git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone --depth 1 https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone --depth 1 https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git && \
    git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git && \
    git clone --depth 1 https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git && \
    git clone --depth 1 https://github.com/sipherxyz/comfyui-art-venture.git && \
    git clone --depth 1 https://github.com/plugcrypt/CRT-Nodes.git && \
    git clone --depth 1 https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone --depth 1 https://github.com/KBYSHanahira/Civicomfy.git && \
    git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git && \
    git -C ComfyUI-Spectrum-MiniMax-H3 checkout b5fd9db33267623eb3469ee7d6d4ddf397240025 && \
    git clone --depth 1 https://github.com/willmiao/ComfyUI-Lora-Manager.git && \
    git clone --depth 1 https://github.com/Steudio/ComfyUI_Steudio.git && \
    git clone --depth 1 https://github.com/pixaroma/ComfyUI-Pixaroma.git

# Patch 2: Filter core pinned dependencies from custom node requirements & assert torch version unchanged
RUN set -eux; \
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"; \
    echo "torch before custom-node deps: ${TORCH_BEFORE}"; \
    for dir in /opt/ComfyUI/custom_nodes/*; do \
        req="$dir/requirements.txt"; \
        [ -f "$req" ] || continue; \
        echo "--- $(basename "$dir") ---"; \
        grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio|triton|xformers|numpy|opencv-python|opencv-contrib-python|opencv-python-headless|transformers|tokenizers|accelerate)([[:space:]]*[<>=!~].*)?$' "$req" > /tmp/req.filtered || true; \
        if ! diff -q "$req" /tmp/req.filtered >/dev/null 2>&1; then \
            echo "NOTE: filtered pinned core deps out of $(basename "$dir")/requirements.txt:"; \
            diff "$req" /tmp/req.filtered || true; \
        fi; \
        pip install --no-cache-dir -r /tmp/req.filtered || echo "WARN: $(basename "$dir") deps failed (non-fatal)"; \
    done; \
    pip uninstall -y onnxruntime-gpu || true; \
    pip install --no-cache-dir onnxruntime; \
    TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"; \
    echo "torch after custom-node deps: ${TORCH_AFTER}"; \
    if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then \
        echo "FATAL: a custom node changed torch ${TORCH_BEFORE} -> ${TORCH_AFTER}"; exit 1; \
    fi; \
    python3 -c "import torch; assert torch.version.cuda and torch.version.cuda.startswith('13'), f'FATAL: torch is not a CUDA 13 build: {torch.version.cuda}'"; \
    rm -rf /root/.cache/pip

# Copy workflows & settings
RUN mkdir -p /opt/ComfyUI/user/default/workflows /opt/ComfyUI/user/__manager
COPY workflows/ /opt/ComfyUI/user/default/workflows/
COPY config/comfy.settings.json /opt/ComfyUI/user/default/comfy.settings.json
COPY config/config.ini /opt/ComfyUI/user/__manager/config.ini

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Copy entrypoint scripts
COPY start.sh /start.sh
COPY download-models.sh /download-models.sh
COPY download-ltx-models.sh /download-ltx-models.sh
COPY download-scail2-models.sh /download-scail2-models.sh
RUN chmod +x /start.sh /download-models.sh /download-ltx-models.sh /download-scail2-models.sh

# Re-run Patch 1 verification as final step
RUN python3 - <<'PY'
import sys, torch, sageattention
from sageattention import core
if not hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda"):
    sys.exit("FATAL: fp8 CUDA symbol missing in final image")
flags = {n: getattr(core, n, None) for n in ("SM80_ENABLED", "SM86_ENABLED", "SM89_ENABLED", "SM90_ENABLED")}
missing = [n for n in ("SM86_ENABLED", "SM89_ENABLED") if not flags.get(n)]
if missing:
    sys.exit(f"FATAL: Final image SageAttention missing {missing}")
print("Final SageAttention verification PASSED")
PY

# Patch 4b: Record build manifest
RUN set -eux; \
    { \
      echo "# Build manifest"; \
      echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
      echo "torch: $(python3 -c 'import torch;print(torch.__version__)')"; \
      echo "torch_cuda: $(python3 -c 'import torch;print(torch.version.cuda)')"; \
      echo "comfyui: $(git -C /opt/ComfyUI describe --tags --always)"; \
      echo "custom_nodes:"; \
      for d in /opt/ComfyUI/custom_nodes/*/; do \
        [ -d "$d/.git" ] || continue; \
        echo "  $(basename "$d"): $(git -C "$d" rev-parse HEAD)"; \
      done; \
    } > /opt/build-manifest.txt; \
    cat /opt/build-manifest.txt

WORKDIR /opt/ComfyUI
EXPOSE 8188 8000

CMD ["/start.sh"]
