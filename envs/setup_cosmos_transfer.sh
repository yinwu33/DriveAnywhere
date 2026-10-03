#!/usr/bin/env bash
# E33 env (docs/PLAN_AFTER_E30.md): NVIDIA Cosmos-Transfer2.5-2B as a video-level restorer of rendered sweeps (vis / depth
# controls). Official uv project (`uv sync --extra=cu128`, torch 2.7 cu128 wheels; driver >= 570), synced into
# .venvs/cosmos_transfer. Single-GPU inference needs 65.4 GB (official docs), so it only fits an H100 with nothing else
# large on it. Checkpoints (NVIDIA Open Model License, gated on Hugging Face: accept the license, then `hf auth login`)
# are downloaded by the inference script on first use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=2ff49d0717af02057ae79bc75c00fbff9da1b4e7  # 2026-10-03 HEAD
SRC="$ROOT/.venvs/src/cosmos-transfer2.5"

if [ ! -d "$SRC" ]; then git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$SRC"; fi
git -C "$SRC" checkout "$COMMIT"
(cd "$SRC" && UV_PROJECT_ENVIRONMENT="$ROOT/.venvs/cosmos_transfer" uv sync --python 3.10 --extra=cu128)  # flash-attn wheels are cp310 only
