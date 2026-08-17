# ===========================================================================
# Stage 1: compile SageAttention against CUDA 13 + PyTorch 2.10
# ===========================================================================
FROM runpod/pytorch:2.10.0-py3.12-cuda13.0.0-devel-ubuntu24.04 AS sage-builder

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp/sage-build

# MUST be ';' separated. Supports RTX 3090 (8.6), RTX 4090 (8.9), and RTX 5090 (12.0).
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ARG MAX_JOBS=2
ARG EXT_PARALLEL=2
ARG NVCC_THREADS=2
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

# ---------------------------------------------------------------------------
# Stage 1 arch normaliser  ->  ';' separated
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(printf '%s' "${TORCH_CUDA_ARCH_LIST}" \
            | tr -d '\042\047' \
            | tr ' ,' ';;' \
            | sed -e 's/;;*/;/g' -e 's/^;//' -e 's/;$//')"; \
    test -n "$ARCH" || { echo "FATAL: TORCH_CUDA_ARCH_LIST is empty"; exit 1; }; \
    printf '%s' "$ARCH" > /etc/sage-arch; \
    echo "normalised TORCH_CUDA_ARCH_LIST = [$ARCH]"

# Preflight with SageAttention's OWN parser.
RUN python3 - <<'PY'
import os, sys
raw = open("/etc/sage-arch").read().strip()
os.environ["TORCH_CUDA_ARCH_LIST"] = raw
try:
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    flags = _get_cuda_arch_flags()
    print("torch.utils.cpp_extension parsed arch flags:", flags)
    assert flags, "arch flags evaluated to empty list"
except Exception as e:
    sys.exit(f"FATAL: TORCH_CUDA_ARCH_LIST={raw!r} failed torch preflight: {e}")
PY

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "setuptools>=68" "wheel" "torch==2.10.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130

RUN git clone "${SAGE_REPO}" sageattention \
    && cd sageattention \
    && git checkout "${SAGE_REF}" \
    && git status

# Build wheel with arch list explicitly set
RUN cd sageattention && \
    ARCH="$(cat /etc/sage-arch)" && \
    echo "Building SageAttention wheel with TORCH_CUDA_ARCH_LIST=[$ARCH] (MAX_JOBS=${MAX_JOBS}, EXT_PARALLEL=${EXT_PARALLEL}, NVCC_THREADS=${NVCC_THREADS})" && \
    TORCH_CUDA_ARCH_LIST="$ARCH" \
    MAX_JOBS="${MAX_JOBS}" \
    EXT_PARALLEL="${EXT_PARALLEL}" \
    NVCC_THREADS="${NVCC_THREADS}" \
    python3 setup.py bdist_wheel --dist-dir /tmp/sage-dist

RUN ls -la /tmp/sage-dist

# Copy the shared verifier script into the builder and run it
COPY verify-sage.py /usr/local/bin/verify-sage.py
RUN chmod +x /usr/local/bin/verify-sage.py

# Smoke test the built wheel inside the builder itself
RUN pip install --no-cache-dir /tmp/sage-dist/*.whl
RUN python3 /usr/local/bin/verify-sage.py

# ===========================================================================
# Stage 2: Final ComfyUI Application Image
# ===========================================================================
FROM runpod/pytorch:2.10.0-py3.12-cuda13.0.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        wget \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        build-essential \
        ninja-build \
        jq \
        ca-certificates \
        rsync \
        procps \
        unzip \
        nano \
        aria2 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Fix libcuda.so stub symlink so build-time steps find it
RUN if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ] && [ ! -f /usr/local/cuda/lib64/libcuda.so ]; then \
        ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so; \
    fi

# Copy the compiled SageAttention wheel from Stage 1 and install it
COPY --from=sage-builder /tmp/sage-dist /tmp/sage-dist
COPY --from=sage-builder /etc/sage-arch /etc/sage-arch
COPY verify-sage.py /usr/local/bin/verify-sage.py
RUN chmod +x /usr/local/bin/verify-sage.py

RUN set -eux; \
    WHEEL="$(ls -1 /tmp/sage-dist/sageattention-*.whl 2>/dev/null | head -n 1)"; \
    test -n "$WHEEL" || { echo "FATAL: no wheel copied from sage-builder"; exit 1; }; \
    pip install --no-cache-dir "$WHEEL"; \
    rm -rf /tmp/sage-dist

# Verify installation immediately using the shared verifier
RUN python3 /usr/local/bin/verify-sage.py

# Install specific triton version for compatibility
RUN pip install --no-cache-dir "triton==3.6.0"

# Preserve base image's ComfyUI installation if present
RUN if [ -d /opt/ComfyUI ]; then \
        mv /opt/ComfyUI /opt/comfyui-baked; \
    else \
        mkdir -p /opt/comfyui-baked && \
        git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui-baked; \
    fi

# Copy baked ComfyUI to /opt/ComfyUI
RUN mkdir -p /opt && cp -a /opt/comfyui-baked /opt/ComfyUI

# Pin ComfyUI explicitly to latest v0.33.1
ARG COMFYUI_MIN_VERSION=0.33.0
ARG COMFYUI_REF=v0.33.1

COPY filter-req.py /usr/local/bin/filter-req.py
COPY pin-comfyui.sh /usr/local/bin/pin-comfyui.sh
RUN chmod +x /usr/local/bin/filter-req.py /usr/local/bin/pin-comfyui.sh

RUN COMFYUI_MIN_VERSION="${COMFYUI_MIN_VERSION}" \
    COMFYUI_REF="${COMFYUI_REF}" \
    /usr/local/bin/pin-comfyui.sh

# Upgrade huggingface_hub without touching pinned frontend/manager packages
RUN pip install --no-cache-dir --upgrade "huggingface_hub[cli]" hf_transfer

# Install ComfyUI-Manager dependencies directly from tree if present
RUN cd /opt/ComfyUI && [ -f manager_requirements.txt ] \
    && python3 /usr/local/bin/filter-req.py manager_requirements.txt /tmp/mgr.txt \
    && pip install --no-cache-dir -r /tmp/mgr.txt || true

WORKDIR /opt/ComfyUI/custom_nodes

# Cleanup list (added ComfyUI-SolAttn_triton and ComfyUI-INT8-Fast)
RUN rm -rf ComfyUI-LTXVideo WhatDreamsCost-ComfyUI ComfyUI-KJNodes ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-Impact-Pack ComfyUI-Manager ComfyUI-Easy-Use ComfyUI-mxToolkit ComfyUI_tinyterraNodes ComfyUI_Comfyroll_CustomNodes Nvidia_RTX_Nodes_ComfyUI comfyui-art-venture CRT-Nodes ComfyUI-DaSiWa-Nodes comfyui_controlnet_aux ComfyUI-Frame-Interpolation Civicomfy ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Lora-Manager ComfyUI_Steudio ComfyUI-Pixaroma ComfyUI-JITBlockSwap comfyui-h3-mlp-chunk ComfyUI-SolAttn_triton ComfyUI-INT8-Fast

# Copy custom node: comfyui-h3-mlp-chunk
COPY custom_nodes/comfyui-h3-mlp-chunk ./comfyui-h3-mlp-chunk

# Clone required custom node packs
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
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
    git clone --depth 1 https://github.com/pixaroma/ComfyUI-Pixaroma.git && \
    git clone https://github.com/lovemachine100/ComfyUI-JITBlockSwap.git && \
    git -C ComfyUI-JITBlockSwap checkout 3b56b2d3514d730c8bec8354d6e9a6ca35c60fdf && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-SolAttn_triton.git && \
    git clone --depth 1 https://github.com/BobJohnson24/ComfyUI-INT8-Fast.git

# Robust LTXVideo import patch
RUN python3 - <<'PY'
import pathlib, sys
p = pathlib.Path("/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py")
if not p.exists():
    sys.exit(f"FATAL: {p} not found -- did the clone succeed?")
src = p.read_text()
old = "    pad,\n)"
new = ")\nfrom torch.nn.functional import pad"
if new in src:
    print("LTXVideo import patch already applied upstream; skipping.")
elif old in src:
    p.write_text(src.replace(old, new, 1))
    print("LTXVideo import patch applied.")
else:
    sys.exit("FATAL: LTXVideo import patch pattern no longer matches upstream pyramid_blending.py")
PY

# Filter core pinned dependencies from custom node requirements using filter-req.py
RUN set -eux; \
    TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"; \
    echo "torch before custom-node deps: ${TORCH_BEFORE}"; \
    for dir in /opt/ComfyUI/custom_nodes/*; do \
        req="$dir/requirements.txt"; \
        [ -f "$req" ] || continue; \
        echo "--- $(basename "$dir") ---"; \
        python3 /usr/local/bin/filter-req.py "$req" /tmp/req.filtered; \
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

# Final gate: re-run the shared verifier
RUN python3 /usr/local/bin/verify-sage.py

# Generate pip constraints file to lock ABI-critical packages
COPY make-pip-constraints.py /usr/local/bin/make-pip-constraints.py
RUN python3 /usr/local/bin/make-pip-constraints.py /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Record build manifest using standalone build-manifest.sh script
COPY build-manifest.sh /usr/local/bin/build-manifest.sh
RUN chmod +x /usr/local/bin/build-manifest.sh \
 && /usr/local/bin/build-manifest.sh /opt/build-manifest.txt

WORKDIR /opt/ComfyUI
EXPOSE 8188 8000

CMD ["/start.sh"]
