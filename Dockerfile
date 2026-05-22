# Learn2Splat — interactive demo for a Hugging Face Space (Docker SDK, GPU).
#
# Installs the optgs package + prebuilt CUDA-extension wheels, then runs
# demo.py's viser GUI: SfM-initialize a COLMAP scene and refine the Gaussians
# with the learned optimizer live in the browser.
#
# The CUDA extensions are NOT compiled here — the HF Docker builder runs out of
# RAM doing it. They are prebuilt into wheels/ on a machine matching this image
# (Python 3.12, torch 2.7.1+cu128, glibc 2.35); see huggingface_space/DEPLOY.md.
#
# Build context = the Space repo root (optgs source + wheels/, see DEPLOY.md).
# Hardware: pick a GPU in the Space settings — A10G (24 GB) recommended; the
# GUI holds the dense and sparse checkpoints in VRAM at once.

# CUDA 12.8 + Ubuntu 22.04 — matches the wheels' build environment.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.12 (via deadsnakes) — optgs uses PEP 695 generic syntax that
# Ubuntu 22.04's stock Python 3.10 cannot parse. Also: build tools, extension
# headers (libglm-dev), and the OpenCV runtime libs (libgl1, libglib2.0-0 —
# optgs's COLMAP loader imports cv2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git build-essential ninja-build libglm-dev \
        libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as a non-root user (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch
WORKDIR /home/user/app

# All Python work happens in a venv on PATH (no system-Python writes).
RUN python3.12 -m venv /home/user/venv
ENV PATH=/home/user/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8) — pinned to setup.sh.
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Python requirements (copied first so this layer caches across code edits).
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

# HF serves the Space over HTTP/2, and the WebSocket subprotocol viser uses to
# announce its client version doesn't survive the proxy — so viser's server
# reads the client version as "unknown" and rejects the connection (the GUI
# then hangs on "connecting"). Client and server are the same viser build
# here, so treat an undeterminable client version as a match, not a reject.
RUN VISER_INFRA="$(python -c 'import viser.infra._infra as m; print(m.__file__)')" \
 && sed -i 's/client_version_str = "unknown"/client_version_str = viser.__version__/' "$VISER_INFRA" \
 && grep -q 'client_version_str = viser.__version__' "$VISER_INFRA"

# Prebuilt CUDA-extension wheels — gsplat, nerfacc, pycolmap, fused-ssim,
# simple-knn, pointops, fused_knn_attn. Built on a matching machine (see
# DEPLOY.md) so the HF builder never compiles CUDA and never OOMs.
COPY --chown=user:user wheels/ ./wheels/
RUN pip install --no-deps ./wheels/*.whl

# The optgs repo, then optgs itself (pure Python — editable install).
COPY --chown=user:user . .
RUN pip install --no-build-isolation --no-deps -e .

# viser serves the GUI here — must equal app_port in README.md.
EXPOSE 7860

# client mode: viser ships the splats to the browser's WebGL renderer, so the
# GPU is used only for optimization. viser binds 0.0.0.0 by default.
CMD ["python", "demo.py", "--with-gui", "client", "--gui-port", "7860"]
