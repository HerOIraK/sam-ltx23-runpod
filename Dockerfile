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

compute_capabilities = set()
for item in env.replace(",", ";").split(";"):
    it = item.strip().lower().replace("sm_", "").replace("compute_", "").replace("a", "")
    if it:
        if len(it) == 2 and it.isdigit():
            it = f"{it[0]}.{it[1]}"
        compute_capabilities.add(it)

print("parsed capabilities  :", compute_capabilities)
HAS_SM80 = any(c.startswith("8.0") for c in compute_capabilities)
HAS_SM86 = any(c.startswith("8.6") for c in compute_capabilities)
HAS_SM89 = any(c.startswith("8.9") for c in compute_capabilities)
HAS_SM90 = any(c.startswith("9.0") for c in compute_capabilities)
HAS_SM120 = any(c.startswith("12.0") for c in compute_capabilities)

sm80 = HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM120
sm89 = HAS_SM89 or HAS_SM90 or HAS_SM120
print("will build           :",
      [n for n, on in (("_qattn_sm80", sm80), ("_qattn_sm89", sm89), ("_fused", True)) if on])

if not sm80:
    sys.exit("FATAL: _qattn_sm80 would not be built -> RTX 3090 unsupported")
if not sm89:
    sys.exit(f"FATAL: _qattn_sm89 would not be built -> RTX 4090/5090 FP8 unsupported (arch list {env!r})")
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
RUN python3 - <<'PY'
import glob, os, re, subprocess, sys, zipfile

wheel = glob.glob("/opt/wheels/sageattention-*.whl")
if not wheel:
    sys.exit("FATAL: No SageAttention wheel found in /opt/wheels")

extract_dir = "/tmp/whlx"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(wheel[0], 'r') as z:
    z.extractall(extract_dir)

sos = glob.glob(f"{extract_dir}/sageattention/*.so")
print("Compiled extensions:", [os.path.basename(s) for s in sos])

has_sm80 = any("sm80" in s for s in sos)
has_sm89 = any("sm89" in s for s in sos)

if not has_sm80:
    sys.exit("FATAL: _qattn_sm80 extension absent (RTX 3090 path)")
if not has_sm89:
    sys.exit("FATAL: _qattn_sm89 extension absent (RTX 4090/5090 FP8 path)")

for so in sos:
    out = subprocess.check_output(["cuobjdump", "--list-elf", so], text=True)
    archs = sorted(set(re.findall(r"sm_\d+", out)))
    print(f"== {os.path.basename(so)}: {archs}")
    if "sm80" in so and "sm_86" not in archs:
        sys.exit(f"FATAL: {os.path.basename(so)} carries no sm_86 SASS")
    if "sm89" in so and "sm_89" not in archs:
        sys.exit(f"FATAL: {os.path.basename(so)} carries no sm_89 SASS")

subprocess.run(["rm", "-rf", extract_dir])
print("SASS verification PASSED: sm_86 + sm_89 + sm_120 present in the wheel")
PY

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

# Clean custom_nodes directory to leave a pristine environment with ComfyUI-Manager only
RUN find /opt/ComfyUI/custom_nodes -mindepth 1 -maxdepth 1 ! -name 'ComfyUI-Manager' -exec rm -rf {} + \
    && if [ ! -d "/opt/ComfyUI/custom_nodes/ComfyUI-Manager" ]; then \
        git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git /opt/ComfyUI/custom_nodes/ComfyUI-Manager; \
    fi

# Pre-install core multimedia, vision, and helper libraries
RUN pip install --no-cache-dir \
    av \
    imageio \
    imageio-ffmpeg \
    scikit-image \
    matplotlib \
    colorama \
    librosa \
    soundfile \
    scipy \
    einops \
    rich \
    pydantic \
    onnxruntime

# Clone requested custom node packs
RUN git clone --depth 1 https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone --depth 1 https://github.com/KBYSHanahira/Civicomfy.git && \
    git clone --depth 1 https://github.com/Azornes/Comfyui-Resolution-Master.git

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
RUN chmod +x /start.sh /download-models.sh

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
