#!/usr/bin/env bash
# Waymo preprocessing env: the devkit pinned by docs/Waymo.md needs TF 2.11 and Python <= 3.10.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/waymo"

uv venv "$VIRTUAL_ENV" --python 3.10.16
uv pip install "waymo-open-dataset-tf-2-11-0==1.6.0" tqdm
