#!/usr/bin/env bash
# Phase 3 fix (DECISIONS D13): COLMAP 4.2 / GLOMAP self-calibration through pycolmap (CPU wheels; SIFT and
# global mapping run on the CPU, about 3-8 min per 200-frame scene on 16 cores).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export VIRTUAL_ENV="$ROOT/.venvs/sfm"

uv venv "$VIRTUAL_ENV" --python 3.12.12
uv pip install pycolmap==4.2.0 opencv-python-headless==5.0.0.93 numpy==2.5.3 pillow==12.3.0 scipy==1.18.1
