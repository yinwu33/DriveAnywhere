#!/usr/bin/env bash
# MapAnything env (Phase 3 pose + pointmap backend). Upstream recommends Python 3.12 and does not pin torch.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/mapanything"
MAPANYTHING_COMMIT=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9  # 2026-08-07, v1.1.4

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "mapanything @ git+https://github.com/facebookresearch/map-anything.git@${MAPANYTHING_COMMIT}"
uv pip install pytest  # tests/ run in this venv (numpy + PIL only)
