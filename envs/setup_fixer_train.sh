#!/usr/bin/env bash
# E34 Fixer fine-tuning env (docs/PLAN_AFTER_E30.md): the Fixer env of envs/setup_fixer.sh (same pins, same transformer_engine
# build) plus the training dependencies of the official training script (Fixer README "Training": accelerate, wandb in
# offline mode, OpenAI CLIP, vision-aided-loss, clean-fid), and the official tokenizer patch (removes @torch.no_grad from
# the Cosmos tokenizer encode / decode so gradients reach the VAE LoRA). A separate venv, so .venvs/fixer, which the
# inference worker of every Fixer run uses, stays unpatched.
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.x toolkit, e.g. /usr/local/cuda-12.1}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/fixer_train"
export PATH="$CUDA_HOME/bin:$PATH"
SRC="$ROOT/.venvs/src/Fixer"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-deps -r "$ROOT/envs/fixer-requirements.txt"
uv pip install --no-deps cosmos-predict2==1.0.9 megatron-core==0.10.0
uv pip install --native-tls "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
CUDNN_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn" NVTE_FRAMEWORK=pytorch MAX_JOBS=12 \
    uv pip install --no-build-isolation "transformer_engine[pytorch]==2.2.0"
uv pip install accelerate==1.15.0 wandb==0.30.0 clean-fid==0.1.35 vision-aided-loss==0.1.0 ftfy regex
uv pip install --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
if [ ! -d "$SRC" ]; then git clone https://github.com/nv-tlabs/Fixer.git "$SRC"; fi
git -C "$SRC" checkout b39dfcaf4eeec90dc943b057ff368c16252c6c6e
patch "$VIRTUAL_ENV/lib/python3.12/site-packages/cosmos_predict2/tokenizers/tokenizer.py" "$SRC/tokenizer.patch"
