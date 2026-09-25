#!/usr/bin/env bash
# Phase 9 video completion env (DECISIONS D18): Wan2.1-VACE through diffusers 0.40 (WanVACEPipeline), torch 2.6
# cu124 (driver 555). Weights Wan-AI/Wan2.1-VACE-1.3B-diffusers (Apache-2.0, 19 GB with the UMT5-XXL text encoder)
# into the HF cache; run the Phase 9 scripts with HF_HUB_OFFLINE=1 afterwards.
# Video-to-video repainting (scripts/repaint_views.py) adds the Wan2.1-T2V-1.3B transformer (Apache-2.0, 5.7 GB); its
# text encoder (same UMT5-XXL, stored in fp32) and VAE (identical weights) are taken from the VACE snapshot.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/wan"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install diffusers==0.40.0 transformers==5.17.0 accelerate==1.15.0 huggingface-hub==1.33.0 ftfy==6.3.1 \
    sentencepiece==0.2.2 safetensors==0.8.0 tokenizers==0.23.2 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
    opencv-python-headless==5.0.0.93 pillow==12.3.0 numpy==2.5.2
"$VIRTUAL_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.1-VACE-1.3B-diffusers'))"
"$VIRTUAL_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', revision='0fad780a534b6463e45facd96134c9f345acfa5b', allow_patterns=['model_index.json', 'scheduler/*', 'transformer/*']))"
