#!/usr/bin/env bash
# TRELLIS image-to-3D env (E43, docs/EXPERIMENTS.md): a parked vehicle's image crop -> 3D Gaussians (structured latents
# decoded with the Gaussian decoder; no mesh extraction, so no kaolin / nvdiffrast / diff-gaussian-rasterization).
# torch 2.5.0 cu124 with the matching xformers wheel (TRELLIS's setup.sh table) and spconv; TRELLIS source at a pinned
# commit, put on the path with a .pth file (it has no setup.py). Weights microsoft/TRELLIS-image-large (MIT) into the
# HF cache; the DINOv2 image encoder comes through torch.hub from GitHub on first use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/trellis"
TRELLIS_COMMIT=442aa1e1afb9014e80681d3bf604e8d728a86ee7  # 2025-11-05 HEAD
SRC="$ROOT/.venvs/src/TRELLIS"

uv venv "$VIRTUAL_ENV" --python 3.10
uv pip install torch==2.5.0 torchvision==0.20.0 xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124
uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh \
    xatlas igraph transformers plyfile spconv-cu124 huggingface_hub safetensors open3d
# kaolin: imported by TRELLIS's FlexiCubes at module load
uv pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
if [ ! -d "$SRC" ]; then git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$SRC"; fi
git -C "$SRC" checkout "$TRELLIS_COMMIT"
echo "$SRC" > "$("$VIRTUAL_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')/trellis_src.pth"
"$VIRTUAL_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('microsoft/TRELLIS-image-large'))"
