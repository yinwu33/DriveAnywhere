#!/usr/bin/env bash
# NKSR env (Phase 7 surface reconstruction). Upstream now builds from source (package/) with conda +
# CUDA 12.8; here the same source is built with uv against the system CUDA 12.1 toolkit and torch cu121
# (driver 555 supports CUDA <= 12.5). setup.py fetches Eigen 3.4 and NanoVDB itself via GitPython.
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.1 toolkit, e.g. /usr/local/cuda-12.1}"
: "${TORCH_CUDA_ARCH_LIST:?set TORCH_CUDA_ARCH_LIST, e.g. 8.6 for RTX A6000}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/nksr"
export PATH="$CUDA_HOME/bin:$PATH"
NKSR_COMMIT=e40336845e67761343a756788e5a98b827d4a143  # 2026-08-24
SRC="$ROOT/.venvs/src/NKSR"

uv venv "$VIRTUAL_ENV" --python 3.10.16
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
uv pip install "setuptools<81" wheel ninja GitPython "python-pycg>=1.0.1" pykdtree plyfile numpy scipy pytest
# nksr imports pycg.vis at module level, which requires Open3D
uv pip install open3d==0.19.0
if [ ! -d "$SRC" ]; then git clone https://github.com/nv-tlabs/NKSR.git "$SRC"; fi
git -C "$SRC" checkout "$NKSR_COMMIT"
(cd "$SRC" && uv pip install --no-build-isolation package/)
