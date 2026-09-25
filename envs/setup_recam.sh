#!/usr/bin/env bash
# ReCamMaster (camera-controlled re-rendering of a video; user request 2026-09-25, DECISIONS P): KwaiVGI/ReCamMaster
# @ fcf98bc (MIT code, bundles its own DiffSynth), weights Wan-AI/Wan2.1-T2V-1.3B (original format, Apache-2.0) and
# KwaiVGI/ReCamMaster-Wan2.1 step20000.ckpt (Apache-2.0), 20 GB, into the repo's models/ directory where
# inference_recammaster.py expects them. Python 3.10 so every dependency has a wheel (no Rust toolchain needed).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/.venvs/src/ReCamMaster"
export VIRTUAL_ENV="$ROOT/.venvs/recam"

[ -d "$SRC" ] || git clone https://github.com/KwaiVGI/ReCamMaster.git "$SRC"
git -C "$SRC" checkout -q fcf98bc86e876bb534518cd99e8a65b282f0f16e
uv venv "$VIRTUAL_ENV" --python 3.10
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install setuptools==80.9.0 wheel
uv pip install --no-build-isolation -e "$SRC"  # setup.py imports pkg_resources
uv pip install lightning pandas
"$VIRTUAL_ENV/bin/python" - <<EOF
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
                  local_dir="$SRC/models/Wan-AI/Wan2.1-T2V-1.3B",
                  allow_patterns=["Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth",
                                  "google/*", "config.json"])
hf_hub_download("KwaiVGI/ReCamMaster-Wan2.1", "step20000.ckpt", revision="4f3f7391743dfd25067ab27e0f1eb8928d11b91d",
                local_dir="$SRC/models/ReCamMaster/checkpoints")
EOF
