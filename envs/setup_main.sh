#!/usr/bin/env bash
# Main 3DGS training env: Python 3.10 + torch 2.4.1 cu121 + gsplat 1.3.0 (see docs/DECISIONS.md, section D).
# uv ships no CUDA toolkit: CUDA extensions compile against $CUDA_HOME, which must be a 12.1 toolkit.
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.1 toolkit, e.g. /usr/local/cuda-12.1}"
: "${TORCH_CUDA_ARCH_LIST:?set TORCH_CUDA_ARCH_LIST, e.g. 8.6 for RTX A6000}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/main"
export PATH="$CUDA_HOME/bin:$PATH"

uv venv "$VIRTUAL_ENV" --python 3.10.16
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r "$ROOT/envs/main-requirements.txt"
# build deps for the CUDA extensions below (they build against the installed torch)
# setuptools<81: torchmetrics 0.10.3 imports pkg_resources, removed from later setuptools
uv pip install "setuptools<81" wheel ninja
uv pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0"
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
uv pip install -e "$ROOT/third_party/smplx"
