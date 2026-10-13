#!/usr/bin/env bash
# The E54 recipe (the simulation / demo operating point, docs/EXPERIMENTS.md E54) on any scene from its FRONT images
# only: masks of the original images, the E46 pipeline (MoGe focal prior, fixed-focal GLOMAP recalibration, masks,
# MapAnything depth, moving-object masks, fusion, NKSR mesh, planes), E54 training, sky completion (E54s) and the
# FRONT held-out evaluation. Waymo (user 10-13: several scenes, Waymo and Mapillary) or an online video fetched with
# scripts/fetch_mapillary.py; steps whose output exists are skipped (the Waymo dev scenes already have them from G1).
# The GT-based evaluations (D-Z1, side cameras) run only when $3 names a Waymo processed split with GT.
# DIST=1 (G2b): the moving-object test's frame offsets are rescaled to the video's camera step (refine_dynamic_masks
# --offset_step 0.5, the step of the Waymo dev scenes at 10 Hz): masks gsam2mvd, training E54d, sky E54ds.
#   [DIST=1] scripts/run_scene_e54.sh <scene_id> <processed_root> [gt_root]
set -euo pipefail
cd /home/tjhu78u/workspace/DriveAnywhere
export CUDA_HOME=$HOME/cuda-12.6 TORCH_CUDA_ARCH_LIST=9.0 HF_HUB_OFFLINE=1 PATH=$PWD/.venvs/main/bin:$HOME/cuda-12.6/bin:$PATH
S=$1; G=$2; GT=${3:-}
U=data/dashrecon/_undistorted/calib-glomapf; D=data/dashrecon/$S
if [ "${DIST:-0}" = 1 ]; then MVT=gsam2mvd; E=E54d; OFFSTEP="--offset_step 0.5"; else MVT=gsam2mv; E=E54; OFFSTEP=""; fi
M=$D/mask-gsam2_sky-segformer; M0=$D/mask-gsam2_sky-segformer_img-glomapf; MV=$D/mask-${MVT}_sky-segformer_img-glomapf
VRU=$D/mask-gsam2_sky-segformer_img-glomapf-vru
P=$D/pose-glomapf_depth-mapanything; F=$D/pose-glomapf_depth-mapanything__mask-${MVT}_sky-segformer_img-glomapf
# training holds one of N slots (N re-read from results/_logs/.train_slots every minute, default 1): the GPU is shared
train_slot() {
  while :; do
    for k in $(seq 1 "$(cat results/_logs/.train_slots 2>/dev/null || echo 1)"); do
      exec {fd}>results/_logs/.train_slot$k.lock
      if flock -n $fd; then echo "(training slot $k)"; "$@"; flock -u $fd; exec {fd}>&-; return 0; fi
      exec {fd}>&-
    done
    sleep 60
  done
}
MASKARGS="--detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large --box_threshold 0.25 --text_threshold 0.25 --dilate_px 5 --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_hw 1024 1536"
echo "== masks (original images) $(date +%T)"
[ -f $M/meta.json ] || .venvs/masks/bin/python scripts/run_masks.py --scene_id $S --processed_root $G --out_root data/dashrecon $MASKARGS
echo "== focal $(date +%T)"
[ -f $D/focal-moge/focal.json ] || .venvs/sam3d/bin/python scripts/run_focal_prior.py --scene_id $S --processed_root $G --out_root data/dashrecon --frames 30
echo "== calib $(date +%T)"
[ -f $D/calib-glomapf/meta.json ] || .venvs/sfm/bin/python scripts/run_calib.py --scene_id $S --processed_root $G --mask_dir $M --out_root data/dashrecon --undistorted_root $U --max_features 8192 --overlap 20 --seed 0 --focal_prior $D/focal-moge/focal.json
echo "== masks $(date +%T)"
[ -f $M0/meta.json ] || .venvs/masks/bin/python scripts/run_masks.py --scene_id $S --processed_root $U --out_root data/dashrecon $MASKARGS --image_tag img-glomapf
[ -f $VRU/meta.json ] || .venvs/masks/bin/python scripts/run_masks.py --scene_id $S --processed_root $U --out_root data/dashrecon $MASKARGS --prompts person bicycle motorcycle --image_tag img-glomapf-vru
echo "== pose / depth $(date +%T)"
# one MapAnything at a time, with a fixed dense-head minibatch: the GPU is shared with training runs
[ -f $P/meta.json ] || HF_HUB_OFFLINE=0 flock results/_logs/.mapanything.lock .venvs/mapanything/bin/python scripts/run_pose.py --scene_id $S --backend mapanything --model_id facebook/map-anything --processed_root $U --camera_dir $D/calib-glomapf --out_root data/dashrecon --scale model --max_views 300 --minibatch_size 8
echo "== moving-object masks $(date +%T)"
[ -f $MV/meta.json ] || PYTHONPATH=. .venvs/main/bin/python scripts/refine_dynamic_masks.py --scene_id $S --mask_dir $M0 --pose_dir $P --out_root data/dashrecon --test_stride 10 --conf_percentile 0 --max_depth 60 --offsets 5 10 15 20 $OFFSTEP --consistency_rel 0.05 --consistency_min 2 --static_share 0.5 --min_tested 30 --always_dir $VRU
echo "== fusion $(date +%T)"
[ -f $F/points_fused.ply ] || .venvs/main/bin/python scripts/run_fusion.py --scene_id $S --pose_dir $P --mask_dir $MV --processed_root $U --out_root data/dashrecon --test_stride 10 --conf_percentile 30 --max_depth 60 --consistency_k 4 --consistency_rel 0.05 --consistency_min 2 --voxel 0.05 --outlier_nb 20 --outlier_std 2.0 --road_cell 0.5 --road_sigma 2.0 --road_min_points 5 --road_max_dz 0.3 --normal_knn 30 --rejected_sample 200000 --seed 0
echo "== mesh $(date +%T)"
[ -f $F/mesh_nksr.ply ] || PATH=$PWD/.venvs/nksr/bin:$HOME/cuda-12.6/bin:$PATH .venvs/nksr/bin/python scripts/run_mesh.py --fusion_dir $F --detail_level 0.5 --mise_iter 1 --solver_tol 1e-4 --coverage_radius 0.15
echo "== planes $(date +%T)"
[ -f $F/planes.npz ] || PYTHONPATH=. .venvs/main/bin/python scripts/fit_planes.py --scene_id $S --fused_dir $F --pose_dir $P --processed_root $U --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --test_stride 10 --frame_stride 3 --depth_tol 0.05 --cell 0.25 --min_cell_points 3 --smooth_cells 1.0 --support_dilate 2 --inlier 0.04 --min_inliers 150 --max_tilt_deg 15 --patch_eps 0.3 --max_planes 80 --vertical_source mesh --out $F/planes.npz
echo "== train $(date +%T)"
[ -f results/$E/$S/checkpoint_final.pth ] || train_slot .venvs/main/bin/python scripts/train_gs.py --exp $E --scene_id $S --output_root results
echo "== sky $(date +%T)"
[ -f results/${E}s/$S/checkpoint_final.pth ] || PYTHONPATH=. .venvs/main/bin/python scripts/complete_sky.py --log_dir results/$E/$S --out_log_dir results/${E}s/$S --exp ${E}s
echo "== eval $(date +%T)"
export PYTHONPATH=.
[ -f results/$E/$S/front_heldout/metrics.json ] || .venvs/main/bin/python scripts/eval_front_heldout.py --log_dir results/$E/$S --example_frames 50 100 150
if [ -n "$GT" ]; then
  [ -d results/${E}s/$S/view_range ] || .venvs/main/bin/python scripts/eval_view_range.py --log_dir results/${E}s/$S --gt_root $GT --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --frame_stride 10 --example_frames 50 100 150
  [ -f results/${E}s/$S/realism_fs/metrics.json ] || .venvs/main/bin/python scripts/eval_cross_camera.py --log_dir results/${E}s/$S --gt_root $GT --frame_stride 5 --alpha 0.5 --example_frames 50 100 150 --cams 1 2 --save_renders --out_subdir realism_fs
fi
echo "== done $(date +%T)"
