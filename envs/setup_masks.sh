#!/usr/bin/env bash
# Phase 4 mask env: Grounding DINO + SAM 2.1 + SegFormer, all through Hugging Face transformers.
# torch >= 2.6 is required: transformers 5.x refuses to torch.load .bin checkpoints (e.g. SegFormer)
# on older torch (CVE-2025-32434). cu124 wheels run on driver 555 (CUDA 12.5).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/masks"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers==5.17.0 accelerate==1.15.0 pillow numpy scipy pytest
