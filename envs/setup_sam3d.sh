#!/usr/bin/env bash
# SAM 3D Objects env (E43, docs/EXPERIMENTS.md): a parked vehicle in one real frame (image + instance mask) -> 3D
# Gaussians with its layout. The official setup is a conda env with the full research requirements; this is the lean
# subset the inference path imports: torch 2.5.1 cu121 (official), kaolin / flash-attn / xformers wheels for it,
# pytorch3d and gsplat built from the official pinned commits, MoGe (pointmaps), the package itself without deps, and
# the official hydra patch. Weights facebook/sam-3d-objects (gated, SAM License) into <src>/checkpoints/hf.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/sam3d"
SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722  # 2026-06-02 HEAD
SRC="$ROOT/.venvs/src/sam-3d-objects"
export CUDA_HOME=${CUDA_HOME:-$HOME/cuda-12.6} TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16

uv venv "$VIRTUAL_ENV" --python 3.11
uv pip install torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
# every later install keeps this torch (MoGe's own dependencies pull the newest torch otherwise)
C="$VIRTUAL_ENV/constraints.txt"
printf 'torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\nxformers==0.0.28.post3\n' > "$C"
export UV_CONSTRAINT="$C" UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 UV_INDEX_STRATEGY=unsafe-best-match
uv pip install hydra-core==1.3.2 omegaconf loguru easydict einops timm==0.9.16 roma==1.5.1 open3d==0.18.0 opencv-python==4.9.0.80 \
    scikit-image==0.23.1 trimesh scipy seaborn==0.13.2 matplotlib safetensors "huggingface-hub[cli]<1.0" spconv-cu121==2.3.8 \
    lightning==2.3.3 pymeshfix==0.17.0 igraph==0.11.8 optree==0.14.1 plotly xatlas pyvista plyfile imageio astor ninja setuptools wheel
uv pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
uv pip install git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b  # the commit SAM 3D was released with (later MoGe needs triton >= 3.2)
uv pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
if [ ! -d "$SRC" ]; then git clone https://github.com/facebookresearch/sam-3d-objects.git "$SRC"; fi
git -C "$SRC" checkout "$SAM3D_COMMIT"
uv pip install -e "$SRC" --no-deps
(cd "$SRC" && "$VIRTUAL_ENV/bin/python" patching/hydra)
if [ ! -d "$SRC/checkpoints/hf" ]; then
  "$VIRTUAL_ENV/bin/hf" download --repo-type model --local-dir "$SRC/checkpoints/hf-download" --max-workers 1 facebook/sam-3d-objects
  mv "$SRC/checkpoints/hf-download/checkpoints" "$SRC/checkpoints/hf" && rm -rf "$SRC/checkpoints/hf-download"
fi
