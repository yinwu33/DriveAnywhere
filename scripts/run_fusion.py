"""Phase 5 entry point: fuse per-frame depth into one cleaned point cloud (AGENTS.md Phase 5).

Inputs (via dashrecon.io): the Phase 3 pose/depth directory and the Phase 4 mask directory of one scene,
plus the FRONT images for colour. Only training frames are used (DECISIONS D7): held-out frames neither
contribute points nor serve as consistency neighbours. Steps, each switchable with --skip for ablation:
    1. back-project depth; drop dynamic / sky pixels, low confidence and far depth (always on)
    2. consistency   multi-view depth consistency filter (dashrecon/fusion/consistency.py)
    3. voxel         voxel averaging (dashrecon/fusion/cleanup.py)
    4. outlier       statistical outlier removal (Open3D)
    5. road          road smoothing onto a height field (dashrecon/fusion/road.py)
    6. normals       PCA normals oriented towards the nearest camera centre (always on)
Output: <out_root>/<scene_id>/<pose_tag>__<mask_tag>/ with points_fused.ply (xyz, normals, rgb,
label 0 = other / 1 = road), frames.txt (frames used; --frame_range limits them to one traversal of a combined
multi-traversal scene, MT1), meta.json (parameters, per-step counts, road
statistics, runtime) and diagnostics/consistency_rejected.ply (a sample of rejected points).

Example (main venv):
    .venvs/main/bin/python scripts/run_fusion.py --scene_id val056 \
        --pose_dir data/dashrecon/val056/pose-mapanything_depth-mapanything \
        --mask_dir data/dashrecon/val056/mask-gsam2_sky-segformer \
        --processed_root data/waymo/processed/validation --out_root data/dashrecon --test_stride 10 \
        --conf_percentile 30 --max_depth 60 --consistency_k 4 --consistency_rel 0.05 --consistency_min 2 \
        --voxel 0.05 --outlier_nb 20 --outlier_std 2.0 --road_cell 0.5 --road_sigma 2.0 --road_min_points 5 \
        --road_max_dz 0.3 --normal_knn 30 --rejected_sample 200000 --seed 0
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.fusion.backproject import backproject  # noqa: E402
from dashrecon.fusion.cleanup import oriented_normals, statistical_outlier_keep, voxel_downsample  # noqa: E402
from dashrecon.fusion.consistency import consistency_counts  # noqa: E402
from dashrecon.fusion.road import smooth_road  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene, train_frame_mask  # noqa: E402

STEPS = ("consistency", "voxel", "outlier", "road")
LABEL_NAMES = ("other", "road")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--test_stride", type=int, required=True)
    parser.add_argument("--conf_percentile", type=float, required=True)
    parser.add_argument("--max_depth", type=float, required=True)
    parser.add_argument("--consistency_k", type=int, required=True)
    parser.add_argument("--consistency_rel", type=float, required=True)
    parser.add_argument("--consistency_min", type=int, required=True)
    parser.add_argument("--voxel", type=float, required=True)
    parser.add_argument("--outlier_nb", type=int, required=True)
    parser.add_argument("--outlier_std", type=float, required=True)
    parser.add_argument("--road_cell", type=float, required=True)
    parser.add_argument("--road_sigma", type=float, required=True)
    parser.add_argument("--road_min_points", type=int, required=True)
    parser.add_argument("--road_max_dz", type=float, required=True)
    parser.add_argument("--normal_knn", type=int, required=True)
    parser.add_argument("--rejected_sample", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip", nargs="*", choices=STEPS, default=[], help="steps to disable (ablation)")
    parser.add_argument("--frame_range", type=int, nargs=2, metavar=("START", "END"),
                        help="use only the frames in [START, END), e.g. one traversal of a combined scene (MT1)")
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.pose_dir)
    np.testing.assert_array_equal(io.read_frames(args.mask_dir), frames)
    poses_all = io.read_poses(args.pose_dir)
    k_img = io.read_intrinsics(args.pose_dir)
    pose_meta, mask_meta = io.read_meta(args.pose_dir), io.read_meta(args.mask_dir)
    grid = pose_meta["depth_grid"]
    k_all = io.depth_intrinsics(k_img, grid)
    train = train_frame_mask(frames, scene.start_timestep, args.test_stride)
    if args.frame_range is not None:
        train &= (frames >= args.frame_range[0]) & (frames < args.frame_range[1])
    tf = frames[train]
    poses, ks = poses_all[train], k_all[train]
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    print(f"[run_fusion] {args.scene_id}: {len(tf)} training frames of {len(frames)} (skip: {args.skip or 'none'})", flush=True)

    depths = np.stack([io.read_depth(args.pose_dir, int(t)) for t in tf])
    confs = np.stack([io.read_depth_conf(args.pose_dir, int(t)) for t in tf])
    conf_thr = float(np.percentile(confs[depths > 0], args.conf_percentile))

    counts = {"depth_pixels": 0, "after_masks": 0, "after_conf_range": 0}
    xyz_l, rgb_l, lab_l, rej_l = [], [], [], []
    for i, t in enumerate(tf):
        t = int(t)
        dyn = io.mask_to_depth_grid(io.read_mask(args.mask_dir, "dynamic", t), grid)
        sky = io.mask_to_depth_grid(io.read_mask(args.mask_dir, "sky", t), grid)
        road = io.mask_to_depth_grid(io.read_mask(args.mask_dir, "road", t), grid)
        has_depth = depths[i] > 0
        static = has_depth & ~dyn & ~sky
        valid = static & (confs[i] >= conf_thr) & (depths[i] < args.max_depth)
        counts["depth_pixels"] += int(has_depth.sum())
        counts["after_masks"] += int(static.sum())
        counts["after_conf_range"] += int(valid.sum())
        pts = backproject(depths[i], ks[i], poses[i], valid)
        rgb = io.image_to_depth_grid(os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg"), grid)[valid]
        lab = road[valid].astype(np.uint8)
        if "consistency" not in args.skip:
            ok = consistency_counts(depths, ks, poses, valid, i, args.consistency_k, args.consistency_rel) >= args.consistency_min
            rej_l.append(pts[~ok])
            pts, rgb, lab = pts[ok], rgb[ok], lab[ok]
        xyz_l.append(pts)
        rgb_l.append(rgb)
        lab_l.append(lab)
    xyz, rgb, lab = np.concatenate(xyz_l), np.concatenate(rgb_l), np.concatenate(lab_l)
    counts["after_consistency"] = int(len(xyz))
    print(f"[run_fusion] back-projected {counts['after_conf_range']:,} -> {len(xyz):,} after consistency ({time.time() - t0:.0f}s)", flush=True)

    if "voxel" not in args.skip:
        xyz, rgb, lab = voxel_downsample(xyz, rgb, lab, args.voxel)
    counts["after_voxel"] = int(len(xyz))
    if "outlier" not in args.skip:
        keep = statistical_outlier_keep(xyz, args.outlier_nb, args.outlier_std)
        xyz, rgb, lab = xyz[keep], rgb[keep], lab[keep]
    counts["after_outlier"] = int(len(xyz))
    road_stats = None
    if "road" not in args.skip:
        xyz, road_stats = smooth_road(xyz, lab == 1, args.road_cell, args.road_sigma, args.road_min_points, args.road_max_dz)
    normals = oriented_normals(xyz, args.normal_knn, poses[:, :3, 3])
    counts["final"] = int(len(xyz))

    out_dir = io.scene_dir(args.out_root, args.scene_id,
                           f"{os.path.basename(os.path.normpath(args.pose_dir))}__{os.path.basename(os.path.normpath(args.mask_dir))}")
    io.write_frames(out_dir, tf)
    io.write_ply(os.path.join(out_dir, "points_fused.ply"), xyz.astype(np.float32), rgb, normals.astype(np.float32), lab)
    if "consistency" not in args.skip:
        rej = np.concatenate(rej_l)
        pick = rng.choice(len(rej), size=min(args.rejected_sample, len(rej)), replace=False)
        io.write_ply(os.path.join(out_dir, "diagnostics", "consistency_rejected.ply"), rej[pick].astype(np.float32),
                     np.zeros((len(pick), 3), dtype=np.uint8))
    runtime = time.time() - t0
    io.write_meta(out_dir, {
        "scene_id": args.scene_id,
        "params": {k: v for k, v in vars(args).items() if k not in ("scene_id",)},
        "conf_threshold": conf_thr,
        "label_names": list(LABEL_NAMES),
        "num_frames_total": int(len(frames)),
        "num_frames_used": int(len(tf)),
        "counts": counts,
        "rejected_by_consistency": counts["after_conf_range"] - counts["after_consistency"],
        "road": road_stats,
        "runtime_s": runtime,
        "units": "backend units (scale strategy of the pose run)",
        "sources": {"pose": {"dir": args.pose_dir, "dashrecon_commit": pose_meta["dashrecon_commit"]},
                    "mask": {"dir": args.mask_dir, "dashrecon_commit": mask_meta["dashrecon_commit"]}},
        "uses_oracle": False,
        "uses_calibration": False,
        "dashrecon_commit": commit,
    })
    print(f"[run_fusion] {args.scene_id}: " + " -> ".join(f"{k} {v:,}" for k, v in counts.items())
          + (f"; road moved {road_stats['moved']:,} (mean |dz| {road_stats['mean_abs_dz_moved']:.3f})" if road_stats else "")
          + f"; {runtime:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
