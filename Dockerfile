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

# Tightened cusparse.h check
RUN set -eux; \
    nvcc --version; \
    echo "CUDA_HOME=${CUDA_HOME}"; \
    readlink -f "${CUDA_HOME}" || true; \
    test -f "${CUDA_HOME}/include/cusparse.h" \
      || { echo "FATAL: cusparse.h missing from ${CUDA_HOME}/include"; \
           echo "other copies found:"; find /usr -name cusparse.h 2>/dev/null; exit 1; }; \
    echo "cusparse.h present in ${CUDA_HOME}/include"

ARG TORCH_CUDA_ARCH_LIST="8.6 8.9"
ARG MAX_JOBS=4
ARG EXT_PARALLEL=1
ARG NVCC_THREADS=2
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention.git
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

# Stage 1 arch sanitiser (space-delimited for SageAttention setup.py) + _get_cuda_arch_flags probe
RUN set -eux; \
    ARCH="$(printf '%s' "${TORCH_CUDA_ARCH_LIST}" \
            | tr -d '\042\047' \
            | tr ';,' '  ' \
            | sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//')"; \
    test -n "$ARCH" || { echo "FATAL: TORCH_CUDA_ARCH_LIST is empty"; exit 1; }; \
    echo "normalised TORCH_CUDA_ARCH_LIST = [$ARCH]"; \
    export TORCH_CUDA_ARCH_LIST="$ARCH"; \
    python3 -c "import sys; from torch.utils.cpp_extension import _get_cuda_arch_flags as g; f=g(); print('nvcc arch flags:', f); sys.exit(0 if all(any(w in x for x in f) for w in ('compute_86','compute_89')) else 'FATAL: arch flags missing compute_86 and/or compute_89')"; \
    export MAX_JOBS="${MAX_JOBS}"; \
    export EXT_PARALLEL="${EXT_PARALLEL}"; \
    export NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}"; \
    echo "Building SageAttention: arch=${TORCH_CUDA_ARCH_LIST} jobs=${MAX_JOBS} ref=${SAGE_REF}"; \
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

# Patch 4a (v2): verify the baked ComfyUI meets floor via pin-comfyui.sh
ARG COMFYUI_MIN_VERSION=0.31.0
ARG COMFYUI_REF=

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

RUN rm -rf ComfyUI-LTXVideo WhatDreamsCost-ComfyUI ComfyUI-KJNodes ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-Impact-Pack ComfyUI-Manager ComfyUI-Easy-Use ComfyUI-mxToolkit ComfyUI_tinyterraNodes ComfyUI_Comfyroll_CustomNodes Nvidia_RTX_Nodes_ComfyUI comfyui-art-venture CRT-Nodes ComfyUI-DaSiWa-Nodes comfyui_controlnet_aux ComfyUI-Frame-Interpolation Civicomfy ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Lora-Manager ComfyUI_Steudio ComfyUI-Pixaroma

# Clone required custom node packs (Pip Manager only, ComfyUI-Manager removed from custom_nodes)
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
    git clone --depth 1 https://github.com/pixaroma/ComfyUI-Pixaroma.git

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

# Record build manifest with safe.directory '*'
RUN set -eux; \
    git config --global --add safe.directory '*'; \
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
