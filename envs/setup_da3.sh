#!/usr/bin/env bash
# Depth Anything 3 env (D-G1, docs/PLAN_AFTER_E30.md): any-view poses, intrinsics and depth from FRONT images only,
# compared with GLOMAP + MapAnything. torch 2.6 cu124 with the matching xformers wheel; DA3 source at a pinned commit.
# Weights depth-anything/DA3NESTED-GIANT-LARGE-1.1 (metric, CC BY-NC 4.0) into the HF cache.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/da3"
DA3_COMMIT=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4  # 2026-10-03 HEAD
SRC="$ROOT/.venvs/src/Depth-Anything-3"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
if [ ! -d "$SRC" ]; then git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$SRC"; fi
git -C "$SRC" checkout "$DA3_COMMIT"
uv pip install -e "$SRC"
uv pip install addict==2.4.0 pytest  # addict: imported by depth_anything_3 but not in its declared dependencies
"$VIRTUAL_ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('depth-anything/DA3NESTED-GIANT-LARGE-1.1'))"
