#!/usr/bin/env bash
# Phase 8 Fixer env (DECISIONS D14): NVIDIA Fixer (single-step Cosmos-Predict2 0.6B restorer) outside its NGC
# container. torch 2.6 cu124 (runs on driver 555); cosmos-predict2 and megatron-core without dependencies, as the
# official Dockerfile does (their training-only dependencies such as tensorstore do not build here); flash-attn
# as the prebuilt wheel (the Qwen text-encoder module asserts it at import, Fixer never runs it); transformer_engine
# compiled against this torch with the system CUDA 12.1 toolkit. The worker (dashrecon/gen/fixer_worker.py) needs
# the venv's nvidia/*/lib directories on LD_LIBRARY_PATH (transformer_engine loads cuDNN through ctypes);
# dashrecon/gen/fixer_client.py sets that when it starts the worker.
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.x toolkit, e.g. /usr/local/cuda-12.1}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/fixer"
export PATH="$CUDA_HOME/bin:$PATH"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r "$ROOT/envs/fixer-requirements.txt"
uv pip install --no-deps cosmos-predict2==1.0.9 megatron-core==0.10.0
# --native-tls: GitHub downloads go through the host's TLS proxy
uv pip install --native-tls "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
CUDNN_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn" NVTE_FRAMEWORK=pytorch MAX_JOBS=12 \
    uv pip install --no-build-isolation "transformer_engine[pytorch]==2.2.0"
# Fixer source (Apache-2.0) at the verified commit; weights nvidia/Fixer (NVIDIA Open Model License) into the HF cache
if [ ! -d "$ROOT/.venvs/src/Fixer" ]; then git clone https://github.com/nv-tlabs/Fixer.git "$ROOT/.venvs/src/Fixer"; fi
git -C "$ROOT/.venvs/src/Fixer" checkout b39dfcaf4eeec90dc943b057ff368c16252c6c6e
"$VIRTUAL_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/Fixer'))"
