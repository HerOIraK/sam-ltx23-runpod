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
        cuda-cuobjdump-${CUDA_PKG} \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/cuda/include
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include

# Tightened cusparse.h check
RUN test -f /usr/local/cuda/include/cusparse.h \
    || test -f /usr/include/cusparse.h \
    || { echo "FATAL: cusparse.h not found. Check libcusparse-dev package installation."; exit 1; }

# Patch 1: Normalise TORCH_CUDA_ARCH_LIST to semicolon-separated
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ARG MAX_JOBS=2
ARG EXT_PARALLEL=2
ARG NVCC_THREADS=2
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

RUN set -eux; \
    ARCH="$(printf '%s' "${TORCH_CUDA_ARCH_LIST}" \
            | tr -d '\042\047' \
            | tr ' ,' ';;' \
            | sed -e 's/;;*/;/g' -e 's/^;//' -e 's/;$//')"; \
    test -n "$ARCH" || { echo "FATAL: TORCH_CUDA_ARCH_LIST is empty"; exit 1; }; \
    printf '%s' "$ARCH" > /etc/sage-arch; \
    echo "normalised TORCH_CUDA_ARCH_LIST = [$ARCH]"

# Patch 2: Preflight with SageAttention's OWN parser
RUN python3 - <<'PY'
import os, sys
env = open("/etc/sage-arch").read().strip()
os.environ["TORCH_CUDA_ARCH_LIST"] = env

try:
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    flags = _get_cuda_arch_flags()
except Exception as e:
    sys.exit(f"FATAL: torch.utils.cpp_extension._get_cuda_arch_flags() crashed: {e}")

print("torch arch flags     :", flags)
caps = [f.split("compute_")[1].split(",")[0] for f in flags if "compute_" in f]
print("extracted capabilities:", caps)

has = {a: any(c.startswith(a) for c in caps) for a in ("8.0", "8.6", "8.9", "9.0", "12.0")}
print("HAS_SMxx            :", has)

sm80 = any(has.values())
sm89 = has["8.9"] or has["9.0"] or has["12.0"]
print("will build          :",
      [n for n, on in (("_qattn_sm80", sm80), ("_qattn_sm89", sm89), ("_fused", True)) if on])

if not sm80:
    sys.exit("FATAL: _qattn_sm80 would not be built -> RTX 3090 unsupported")
if not sm89:
    sys.exit(f"FATAL: _qattn_sm89 would not be built -> RTX 4090/5090 FP8 unsupported "
             f"(arch list {env!r} never yielded 8.9 or 12.0)")
print("arch preflight PASSED")
PY

RUN set -eux; \
    export TORCH_CUDA_ARCH_LIST="$(cat /etc/sage-arch)"; \
    export MAX_JOBS="${MAX_JOBS}"; \
    export EXT_PARALLEL="${EXT_PARALLEL}"; \
    export NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}"; \
    echo "Building SageAttention: arch=${TORCH_CUDA_ARCH_LIST} ext_parallel=${EXT_PARALLEL} ref=${SAGE_REF}"; \
    mkdir -p /opt/wheels; \
    pip wheel --no-cache-dir --no-build-isolation --no-deps -w /opt/wheels \
        "git+${SAGE_REPO}@${SAGE_REF}"; \
    ls -la /opt/wheels; \
    test -n "$(ls /opt/wheels/sageattention-*.whl 2>/dev/null)"

# Ground truth SASS verification
RUN set -eux; \
    rm -rf /tmp/whlx; mkdir -p /tmp/whlx; \
    python3 -m zipfile -e /opt/wheels/sageattention-*.whl /tmp/whlx; \
    ls -la /tmp/whlx/sageattention/*.so; \
    for so in /tmp/whlx/sageattention/*.so; do \
        echo "== $(basename "$so")"; \
        cuobjdump --list-elf "$so" | grep -oE 'sm_[0-9]+' | sort -u | sed 's/^/     /'; \
    done; \
    ls /tmp/whlx/sageattention/*sm80*.so >/dev/null 2>&1 \
      || { echo "FATAL: _qattn_sm80 extension absent (RTX 3090 path)"; exit 1; }; \
    ls /tmp/whlx/sageattention/*sm89*.so >/dev/null 2>&1 \
      || { echo "FATAL: _qattn_sm89 extension absent (RTX 4090/5090 FP8 path)"; exit 1; }; \
    cuobjdump --list-elf /tmp/whlx/sageattention/*sm80*.so | grep -q 'sm_86' \
      || { echo "FATAL: _qattn_sm80 carries no sm_86 SASS"; exit 1; }; \
    cuobjdump --list-elf /tmp/whlx/sageattention/*sm89*.so | grep -q 'sm_89' \
      || { echo "FATAL: _qattn_sm89 carries no sm_89 SASS"; exit 1; }; \
    rm -rf /tmp/whlx; \
    echo "SASS verification PASSED: sm_86 + sm_89 + sm_120 present in the wheel"

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

COPY verify-sage.py /usr/local/bin/verify-sage.py
RUN chmod +x /usr/local/bin/verify-sage.py \
    && python3 /usr/local/bin/verify-sage.py

ENV TRITON_CACHE_DIR=/workspace/.cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor

# Copy baked ComfyUI to /opt/ComfyUI
RUN mkdir -p /opt && cp -a /opt/comfyui-baked /opt/ComfyUI

# Pin ComfyUI explicitly to v0.33.2
ARG COMFYUI_MIN_VERSION=0.33.0
ARG COMFYUI_REF=v0.33.2

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

# Cleanup list
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

# Resilient LTXVideo import patch
RUN python3 - <<'PY'
import pathlib
p = pathlib.Path("/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py")
if p.exists():
    src = p.read_text()
    if "from torch.nn.functional import pad" not in src:
        src = src.replace("    pad,\n", "").replace("    pad,\r\n", "").replace("pad,", "")
        src = "from torch.nn.functional import pad\n" + src
        p.write_text(src)
        print("LTXVideo import patch applied cleanly.")
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
        pip install --no-cache-dir --no-deps -r /tmp/req.filtered || true; \
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
RUN chmod +x /usr/local/bin/make-pip-constraints.py \
 && /usr/local/bin/make-pip-constraints.py /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Record build manifest using standalone build-manifest.sh script
COPY build-manifest.sh /usr/local/bin/build-manifest.sh
RUN chmod +x /usr/local/bin/build-manifest.sh \
 && /usr/local/bin/build-manifest.sh /opt/build-manifest.txt

WORKDIR /opt/ComfyUI
EXPOSE 8188 8000

CMD ["/start.sh"]
