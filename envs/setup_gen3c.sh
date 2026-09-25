#!/usr/bin/env bash
# GEN3C (NVIDIA, 3D-cache-conditioned camera-controlled video generation on Cosmos-Predict1 7B; DECISIONS P) for
# side / rear views of the FRONT video. Code nv-tlabs/GEN3C @ db2ffe1 (Apache-2.0), weights nvidia/GEN3C-Cosmos-7B
# (NVIDIA Open Model License), nvidia/Cosmos-Tokenize1-CV8x8x8-720p, google-t5/t5-11b (PyTorch weights only),
# 76 GB into the repository's checkpoints/ directory. uv instead of the official conda env (python 3.10, torch 2.6
# cu124); CUDA extensions (transformer_engine 1.12) compiled against the system CUDA 12.x toolkit, as for
# envs/setup_fixer.sh. megatron-core without dependencies (its tensorstore dependency does not build here, same as
# the Fixer env). apex with its C++ / CUDA extensions (NVIDIA/apex @ e74a67b): the inference pipeline imports the
# training model module, which imports apex's amp_C at module level (CUDA version check relaxed to the major version). Guardrail and prompt-upsampler models are not
# downloaded: run with --disable_guardrail --disable_prompt_encoder.
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.x toolkit, e.g. /usr/local/cuda-12.1}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/.venvs/src/GEN3C"
export VIRTUAL_ENV="$ROOT/.venvs/gen3c"
export PATH="$CUDA_HOME/bin:$PATH"

[ -d "$SRC" ] || git clone https://github.com/nv-tlabs/GEN3C.git "$SRC"
git -C "$SRC" checkout -q db2ffe12ced12ddafcec5e0422ee46ce8520746b
uv venv "$VIRTUAL_ENV" --python 3.10
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
REQ=$(mktemp)
grep -v "^megatron-core" "$SRC/requirements.txt" > "$REQ"
uv pip install -r "$REQ"
rm "$REQ"
uv pip install --no-deps megatron-core==0.10.0
uv pip install ninja pybind11 wheel
CUDNN_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn" NVTE_FRAMEWORK=pytorch MAX_JOBS=12 \
    uv pip install --no-build-isolation "transformer_engine[pytorch]==1.12.0"
uv pip install "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
APEX="$ROOT/.venvs/src/apex"
[ -d "$APEX" ] || git clone https://github.com/NVIDIA/apex.git "$APEX"
# May 2025 (GEN3C's time): apex from August 2025 on registers torch.library ops that torch 2.6 rejects
git -C "$APEX" checkout -q e74a67bba3ee679f778670e17edc21639008ae0a
# apex refuses any CUDA version other than torch's (12.4); only 12.1 is installed here, and extensions built with 12.1
# run with torch cu124 (transformer_engine above, the Fixer env), so compare the major version only
sed -i 's/    if (bare_metal_version != torch_binary_version):/    if (bare_metal_version.major != torch_binary_version.major):/' "$APEX/setup.py"
grep -q "bare_metal_version.major != torch_binary_version.major" "$APEX/setup.py"
uv pip install setuptools==76.0.0 wheel
(cd "$APEX" && rm -rf build dist && MAX_JOBS=12 "$VIRTUAL_ENV/bin/python" setup.py bdist_wheel --cpp_ext --cuda_ext)
uv pip install "$APEX"/dist/apex-*.whl
"$VIRTUAL_ENV/bin/python" - <<EOF
from huggingface_hub import snapshot_download
ckpt = "$SRC/checkpoints"
snapshot_download("nvidia/GEN3C-Cosmos-7B", revision="9bcfdb4f3924f41376daeadf6200826c12a3bf8e",
                  local_dir=f"{ckpt}/Gen3C-Cosmos-7B", allow_patterns=["README.md", "config.json", "model.pt"])
snapshot_download("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", revision="b6af495317c76f287a4131e9299936b1533f5f9f",
                  local_dir=f"{ckpt}/Cosmos-Tokenize1-CV8x8x8-720p")
snapshot_download("google-t5/t5-11b", revision="90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
                  local_dir=f"{ckpt}/google-t5/t5-11b",
                  allow_patterns=["config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json"])
EOF
