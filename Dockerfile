FROM runpod/comfyui:cuda13.0

USER root

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    aria2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy ComfyUI outside /workspace so it's not hidden by volume mounts
RUN mkdir -p /opt && \
    cp -a /opt/comfyui-baked /opt/ComfyUI

# Update ComfyUI core & frontend to the absolute latest version
RUN cd /opt/ComfyUI && \
    (git pull || true) && \
    pip install --no-cache-dir --upgrade comfyui-frontend

WORKDIR /opt/ComfyUI/custom_nodes

# Remove any pre-existing folders from base image to prevent git clone collisions
RUN rm -rf ComfyUI-LTXVideo WhatDreamsCost-ComfyUI ComfyUI-KJNodes ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-Impact-Pack ComfyUI-Manager ComfyUI-Easy-Use ComfyUI-mxToolkit ComfyUI_tinyterraNodes ComfyUI_Comfyroll_CustomNodes Nvidia_RTX_Nodes_ComfyUI comfyui-art-venture CRT-Nodes

# Clone required custom nodes
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
    git clone --depth 1 https://github.com/plugcrypt/CRT-Nodes.git


# Install requirements supplied by each custom-node package
RUN set -e; \
    for dir in /opt/ComfyUI/custom_nodes/*; do \
        if [ -f "$dir/requirements.txt" ]; then \
            echo "Installing requirements: $dir"; \
            pip install --no-cache-dir -r "$dir/requirements.txt"; \
        fi; \
    done

# Add your workflows to the image
RUN mkdir -p /opt/ComfyUI/user/default/workflows
COPY workflows/ /opt/ComfyUI/user/default/workflows/

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Add startup and provisioning scripts
COPY start.sh /start.sh
COPY download-models.sh /download-models.sh

RUN chmod +x /start.sh /download-models.sh

EXPOSE 8188 8000

CMD ["/start.sh"]
