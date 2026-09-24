#!/usr/bin/env bash
# Phase 8 weights into the Hugging Face cache (~9.6 GB): SDXL base 1.0 (fp16 UNet and text encoders, no VAE),
# the SDXL depth ControlNet (fp16) and the fp16-safe SDXL VAE. dashrecon/gen/ggds.py loads them by repo id;
# run the Phase 8 scripts with HF_HUB_OFFLINE=1 afterwards.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/.venvs/main/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

for repo, patterns in [
    ("stabilityai/stable-diffusion-xl-base-1.0", [
        "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json"]),
    ("diffusers/controlnet-depth-sdxl-1.0", ["config.json", "diffusion_pytorch_model.fp16.safetensors"]),
    ("madebyollin/sdxl-vae-fp16-fix", ["config.json", "diffusion_pytorch_model.safetensors"]),
]:
    path = snapshot_download(repo, allow_patterns=patterns)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    print(f"{repo} -> {path} ({size / 1e9:.2f} GB)")
PY
