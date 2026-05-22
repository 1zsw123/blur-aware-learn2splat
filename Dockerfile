# Learn2Splat — interactive demo for a Hugging Face Space (Docker SDK, GPU).
#
# Builds the optgs package + its CUDA extensions and runs demo.py's viser GUI:
# SfM-initialize a COLMAP scene, then refine the Gaussians with the learned
# optimizer live in the browser. Mirrors setup.sh, minus conda — the CUDA
# toolkit ships in the base image.
#
# Build context = the optgs repo root (see huggingface_space/DEPLOY.md).
# Hardware: pick a GPU in the Space settings — A10G (24 GB) recommended; the
# GUI holds the dense and sparse checkpoints in VRAM at once.

# CUDA 12.8 devel (nvcc + headers); Ubuntu 22.04 — the OS setup.sh is tested on.
# A devel base is required: gsplat / nerfacc JIT-compile CUDA on first use, so
# nvcc must also be present at runtime.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Compile the CUDA extensions for every GPU a Space may run on
    # (T4 7.5 · A100 8.0 · A10G 8.6 · L4/L40S 8.9 · H100 9.0). Trim this to
    # your chosen GPU to shorten the build.
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"

# Build tools + extension headers (libglm-dev) and the OpenCV runtime libs
# (libgl1, libglib2.0-0 — optgs's COLMAP loader imports cv2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
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
RUN python3 -m venv /home/user/venv
ENV PATH=/home/user/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8) — pinned to setup.sh.
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Python requirements (copied first so this layer caches across code edits).
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

# gsplat + nerfacc — built from git against the torch installed above.
RUN pip install --no-build-isolation \
        git+https://github.com/nerfstudio-project/nerfacc \
        git+https://github.com/nerfstudio-project/gsplat.git

# The optgs repo.
COPY --chown=user:user . .

# CUDA-extension submodules, then optgs itself. pycolmap is the pure-Python
# COLMAP reader (no C++ build); the other four compile CUDA kernels.
RUN pip install submodules/pycolmap \
 && pip install --no-build-isolation submodules/fused-ssim \
 && pip install --no-build-isolation submodules/simple-knn \
 && pip install --no-build-isolation submodules/pointops \
 && pip install --no-build-isolation submodules/fused_knn_attn \
 && pip install --no-build-isolation --no-deps -e .

# viser serves the GUI here — must equal app_port in README.md.
EXPOSE 7860

# client mode: viser ships the splats to the browser's WebGL renderer, so the
# GPU is used only for optimization. viser binds 0.0.0.0 by default.
CMD ["python", "demo.py", "--with-gui", "client", "--gui-port", "7860"]
