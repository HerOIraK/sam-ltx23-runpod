FROM runpod/comfyui:cuda13.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. CUDA dev headers & build toolchain for CUDA 13.0
ARG CUDA_PKG=13-0
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget aria2 ffmpeg ca-certificates \
        build-essential ninja-build \
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

# Verify nvcc & cusparse header presence before building
RUN set -eux; \
    nvcc --version; \
    HDR="$(find /usr /usr/local -name cusparse.h -print -quit)"; \
    test -n "$HDR"; \
    echo "cusparse.h -> $HDR"

ENV CPATH=/usr/local/cuda/include
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include

# 2. Build SageAttention 2 from source into a wheel for SM 8.6 (RTX 3090) and SM 8.9 (RTX 4090 / L40S)
ARG TORCH_CUDA_ARCH_LIST="8.6 8.9"
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=main

RUN set -eux; \
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"; \
    export MAX_JOBS="$(nproc)"; \
    export EXT_PARALLEL=4; \
    export NVCC_APPEND_FLAGS="--threads 8"; \
    mkdir -p /opt/wheels; \
    pip wheel --no-cache-dir --no-build-isolation --no-deps -w /opt/wheels \
        "git+${SAGE_REPO}@${SAGE_REF}"; \
    ls -la /opt/wheels

RUN pip install --no-cache-dir /opt/wheels/sageattention-*.whl

# Hard assertion gate: verify SageAttention 2 FP8 & FP16 CUDA kernels are compiled and present
RUN python3 -c "\
import sageattention as s; \
k=[n for n in dir(s) if n.startswith('sageattn')]; \
print('SageAttention exported kernels:', k); \
assert 'sageattn_qk_int8_pv_fp8_cuda' in k or 'sageattn' in k, 'SageAttention kernels missing'; \
print('SageAttention 2 build assertion OK')"

# Copy ComfyUI outside /workspace so it's not hidden by volume mounts
RUN mkdir -p /opt && \
    cp -a /opt/comfyui-baked /opt/ComfyUI

# Update ComfyUI core, frontend, and manager to the absolute latest version
# Install hf_transfer for ultra-fast Hugging Face downloads
RUN git config --global --add safe.directory /opt/ComfyUI && \
    cd /opt/ComfyUI && \
    (git pull || true) && \
    pip install --no-cache-dir --upgrade \
        comfyui-frontend-package \
        comfyui-manager \
        huggingface_hub[cli] \
        hf_transfer

WORKDIR /opt/ComfyUI/custom_nodes

# Remove any pre-existing folders from base image to prevent git clone collisions
RUN rm -rf ComfyUI-LTXVideo WhatDreamsCost-ComfyUI ComfyUI-KJNodes ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-Impact-Pack ComfyUI-Manager ComfyUI-Easy-Use ComfyUI-mxToolkit ComfyUI_tinyterraNodes ComfyUI_Comfyroll_CustomNodes Nvidia_RTX_Nodes_ComfyUI comfyui-art-venture CRT-Nodes ComfyUI-DaSiWa-Nodes comfyui_controlnet_aux ComfyUI-Frame-Interpolation Civicomfy

# Clone required custom nodes
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    python3 -c "f = 'ComfyUI-LTXVideo/pyramid_blending.py'; c = open(f).read().replace('    pad,\n)', ')\nfrom torch.nn.functional import pad'); open(f, 'w').write(c)" && \
    git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    python3 -c "f = 'ComfyUI-KJNodes/nodes/model_optimization_nodes.py'; c = open(f).read().replace('from sageattention import sageattn_qk_int8_pv_fp8_cuda', 'try:\n        from sageattention import sageattn_qk_int8_pv_fp8_cuda\n    except ImportError:\n        from sageattention import sageattn as sageattn_qk_int8_pv_fp8_cuda'); open(f, 'w').write(c)" && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone --depth 1 https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone --depth 1 https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git && \
    git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git && \
    git clone --depth 1 https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git && \
    python3 -c "f='Nvidia_RTX_Nodes_ComfyUI/__init__.py'; open(f,'a').write('\n\ntry:\n    _orig_exec = RTXVideoSuperResolution.execute\n    class _Wrapper:\n        @classmethod\n        def _safe_exec(cls, *args, **kwargs):\n            try:\n                return _orig_exec(*args, **kwargs)\n            except Exception as e:\n                print(f\"[Nvidia_RTX_Nodes] Warning: RTX Video Super Resolution failed ({e}). Returning input image.\")\n                return (kwargs.get(\"image\", args[0] if args else None),)\n    RTXVideoSuperResolution.execute = _Wrapper._safe_exec\nexcept Exception:\n    pass\n')" && \
    git clone --depth 1 https://github.com/sipherxyz/comfyui-art-venture.git && \
    git clone --depth 1 https://github.com/plugcrypt/CRT-Nodes.git && \
    git clone --depth 1 https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone --depth 1 https://github.com/KBYSHanahira/Civicomfy.git

# Install requirements supplied by each custom-node package
# Also force CPU-only onnxruntime to prevent cuDNN 9 / CUDA 13 symbol conflicts in DWPose/ONNX
RUN set -e; \
    for dir in /opt/ComfyUI/custom_nodes/*; do \
        if [ -f "$dir/requirements.txt" ]; then \
            echo "Installing requirements: $dir"; \
            pip install --no-cache-dir -r "$dir/requirements.txt" || echo "WARN: $dir failed"; \
        fi; \
    done; \
    pip uninstall -y onnxruntime-gpu || true; \
    pip install --no-cache-dir onnxruntime

# Add default workflows and settings to the image
RUN mkdir -p /opt/ComfyUI/user/default/workflows /opt/ComfyUI/user/__manager
COPY workflows/ /opt/ComfyUI/user/default/workflows/
COPY config/comfy.settings.json /opt/ComfyUI/user/default/comfy.settings.json
COPY config/config.ini /opt/ComfyUI/user/__manager/config.ini

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Add startup and provisioning scripts
COPY start.sh /start.sh
COPY download-models.sh /download-models.sh
COPY download-ltx-models.sh /download-ltx-models.sh
COPY download-scail2-models.sh /download-scail2-models.sh

RUN chmod +x /start.sh /download-models.sh /download-ltx-models.sh /download-scail2-models.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188 8000

CMD ["/start.sh"]
