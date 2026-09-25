#!/usr/bin/env bash
# Phase 9 (E8, DECISIONS D16-D18): progressive generative completion of one scene, one round per camera move.
#
# Round r (0-based) with move M_r:
#   1. render_views.py  (main venv)        the current model along every <frame_stride>-th FRONT frame moved by M_r,
#                                          hole masks;
#                                          the views of rounds < r are the 3D memory (--memory_dirs)
#   2. fill_views.py    (wan venv)         Wan2.1-VACE-1.3B fills the holes (<fill> vace), or
#      gen3c_fill.py    (gen3c venv)       GEN3C regenerates the view from the render as its 3D cache (<fill> gen3c:
#                                          rendered at 704 x 1280, the move ramped in over 20 views from the FRONT pose)
#   3. depth_views.py   (mapanything venv) depth of the filled frames, aligned to the rendered depth
#   4. train_fill.py    (main venv)        spawn Gaussians in the holes, distil all rounds' views so far
# The current model is <init_log_dir> for round 0 and <out_log_dir> afterwards. Views go to <out_log_dir>/views/
# r<r>_<move with "=" dropped and "," -> "_">. Rounds before <first_round> must already be complete (their views
# dirs are used as memory and training views; <out_log_dir>/checkpoint_final.pth is their result).
#
# Usage (repo root):
#   bash scripts/run_phase9.sh <scene_id> <init_log_dir> <out_log_dir> <first_round> <frame_stride> <fill> <move_0> [<move_1> ...]
#   bash scripts/run_phase9.sh val039 results/E5c/val039 results/E8/val039 0 1 vace right=1.5,yaw=15 right=-1.5,yaw=-15
# (frame_stride applies to the rounds run now; earlier rounds keep whatever they were rendered with)
set -euo pipefail
[ $# -ge 7 ] || { echo "usage: $0 <scene_id> <init_log_dir> <out_log_dir> <first_round> <frame_stride> <fill> <move_0> [<move_1> ...]" >&2; exit 2; }
scene=$1; init=$2; out=$3; first=$4; stride=$5; fill=$6; shift 6
case "$fill" in
  vace) render_opts=(); ;;
  gen3c) render_opts=(--ramp_frames 20 --render_hw 704 1280); ;;
  *) echo "fill must be vace or gen3c, got $fill" >&2; exit 2; ;;
esac
moves=("$@")
[ "$first" -lt "${#moves[@]}" ] || { echo "first_round $first >= number of moves ${#moves[@]}" >&2; exit 2; }
[ -d /usr/local/cuda-12.1 ] || { echo "needs /usr/local/cuda-12.1" >&2; exit 2; }
export CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAIN="env PATH=$PWD/.venvs/main/bin:$CUDA_HOME/bin:$PATH .venvs/main/bin/python"

views=()
for r in "${!moves[@]}"; do
  views+=("$out/views/r${r}_$(echo "${moves[$r]}" | sed 's/=//g; s/,/_/g')")
done
for ((r = 0; r < first; r++)); do
  [ -f "${views[$r]}/depth.json" ] || { echo "round $r is not complete: ${views[$r]}/depth.json missing" >&2; exit 1; }
done

for ((r = first; r < ${#moves[@]}; r++)); do
  vd=${views[$r]}
  src=$init
  [ "$r" -eq 0 ] || src=$out
  echo "[run_phase9] $scene round $r: ${moves[$r]} from $src -> $vd ($(date +%H:%M))"
  $MAIN scripts/render_views.py --log_dir "$src" --move "${moves[$r]}" --out_dir "$vd" --frame_stride "$stride" \
      "${render_opts[@]}" --memory_dirs "${views[@]:0:$r}"
  if [ "$fill" = vace ]; then
    .venvs/wan/bin/python scripts/fill_views.py --views_dir "$vd"
  else
    .venvs/gen3c/bin/python scripts/gen3c_fill.py --views_dir "$vd"
  fi
  .venvs/mapanything/bin/python scripts/depth_views.py --views_dir "$vd" --model_id facebook/map-anything
  $MAIN scripts/train_fill.py --scene_id "$scene" --init_log_dir "$src" --out_log_dir "$out" --round "$r" \
      --views_dirs "${views[@]:0:$((r + 1))}"
done
echo "[run_phase9] $scene: rounds $first..$((${#moves[@]} - 1)) done ($(date +%H:%M))"
