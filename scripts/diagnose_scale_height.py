"""D-G2 (docs/PLAN_AFTER_E30.md R5): scale from the camera's height above the road. DIAGNOSTIC CODE: reads GT.

The reconstruction's scale is 0.43-0.78 of the truth (OPEN_QUESTIONS 15) and D-G1's depth models do not fix it. In a
driving scene the one length that is always there is the camera's height above the road. This measures, per scene and
without any calibration, the camera height in scene units:
    for every frame, the fused road points (points_fused.ply labels == 1, Phase 5) within --radius scene units of the
    camera centre (horizontally) are fitted by a plane (least squares, --ransac_iters RANSAC rounds with --inlier_dist),
    and the height is the signed distance of the camera centre to that plane; frames with fewer than --min_points road
    points are skipped; the scene's height is the median over frames.
Then, evaluation side, the GT height (DECISIONS A: FRONT at 2.12 m on Waymo, read from extrinsics/0.txt) gives the
scale the height would imply, h_scene / h_gt, compared with the Sim(3) scale of the GT trajectory onto the estimated one
(as scripts/diagnose_da3.py / eval_cross_camera.py): how precisely a known camera height fixes the scale. A generic
mount-height prior (--prior_height, e.g. 1.5 m for a dashcam) adds its own error on top: both are reported.

Output: --out JSON (per scene and mean) and a printed table.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/diagnose_scale_height.py --scene_ids val056 val039 val041 val087 val094 \\
        --processed_root data/waymo/processed/validation --dashrecon_root data/dashrecon --prior_height 1.5 \\
        --out data/dashrecon/diagnostics/dg2_height.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from diagnose_da3 import OPENCV2DATASET, umeyama  # noqa: E402

POSE_TAG = "pose-glomap_depth-mapanything"
FUSION_TAG = "pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap"


def fit_plane(pts: np.ndarray, rng: np.random.Generator, iters: int, inlier_dist: float) -> tuple[np.ndarray, float]:
    """RANSAC plane (unit normal n, offset d with n . x + d = 0) refitted on its inliers by least squares. All ``iters``
    candidate triplets are scored at once (a (iters, N) distance matrix): numpy's per-call sampling without replacement
    made the loop version take 0.5 s per iteration on 90k points."""
    tri = pts[rng.integers(0, len(pts), size=(iters, 3))]  # (iters, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n, p0 = n[ok] / norm[ok][:, None], tri[ok, 0]
    d = -(n * p0).sum(1)
    dist = np.abs(n @ pts.T + d[:, None])  # (iters, N)
    best = int(np.argmax((dist < inlier_dist).sum(1)))
    q = pts[dist[best] < inlier_dist]
    c = q.mean(0)
    n_fit = np.linalg.svd(q - c, full_matrices=False)[2][-1]  # full_matrices would build an (N, N) U
    n_fit = n_fit / np.linalg.norm(n_fit)
    return n_fit, float(-n_fit @ c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_ids", nargs="+", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--dashrecon_root", required=True)
    parser.add_argument("--radius", type=float, default=8.0, help="scene units around the camera, horizontally")
    parser.add_argument("--min_points", type=int, default=500)
    parser.add_argument("--max_points", type=int, default=20000, help="road points per frame after random subsampling")
    parser.add_argument("--ransac_iters", type=int, default=200)
    parser.add_argument("--inlier_dist", type=float, default=0.05)
    parser.add_argument("--prior_height", type=float, required=True, help="generic camera mount height prior, metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = git_commit()
    rng = np.random.default_rng(args.seed)
    rows = {}
    for sid in args.scene_ids:
        scene = get_scene(sid)
        gt_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}")
        pose_dir = os.path.join(args.dashrecon_root, sid, POSE_TAG)
        ply = io.read_ply(os.path.join(args.dashrecon_root, sid, FUSION_TAG, "points_fused.ply"))
        assert "labels" in ply, "points_fused.ply without road labels"
        road = ply["xyz"][ply["labels"] == 1].astype(np.float64)
        poses = io.read_poses(pose_dir)
        frames = io.read_frames(pose_dir)
        heights, tilts = [], []
        for c2w in poses:
            c = c2w[:3, 3]
            near = road[np.linalg.norm((road - c)[:, :2], axis=1) < args.radius]
            if len(near) < args.min_points:
                continue
            if len(near) > args.max_points:
                near = near[rng.choice(len(near), args.max_points, replace=False)]
            n, d = fit_plane(near, rng, args.ransac_iters, args.inlier_dist)
            if n[2] < 0:
                n, d = -n, -d
            heights.append(float(n @ c + d))
            tilts.append(float(np.degrees(np.arccos(min(1.0, n[2])))))
        assert len(heights) >= 10, f"{sid}: only {len(heights)} frames with enough road points"
        h_scene = float(np.median(heights))
        # evaluation side: GT height and the Sim(3) scale (GT metres -> scene units)
        front_to_ego = np.loadtxt(os.path.join(gt_dir, "extrinsics", "0.txt")) @ OPENCV2DATASET
        h_gt = float(front_to_ego[2, 3])  # vehicle frame z of the camera; vehicle ground z ~ -0.05 (DECISIONS A)
        gt_c = np.stack([(np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{int(t):03d}.txt")) @ front_to_ego)[:3, 3] for t in frames])
        s_gt = umeyama(gt_c, poses[:, :3, 3])[0]
        rows[sid] = {"frames_used": len(heights), "road_points": int(len(road)), "height_scene_units": h_scene,
                     "height_frame_iqr": [float(np.percentile(heights, 25)), float(np.percentile(heights, 75))],
                     "road_tilt_deg_median": float(np.median(tilts)), "gt_height_m": h_gt, "gt_to_est_scale": s_gt,
                     "height_in_metres_via_gt_scale": h_scene / s_gt,
                     "scale_from_gt_height": h_scene / h_gt, "scale_from_prior": h_scene / args.prior_height}
        r = rows[sid]
        print(f"[diagnose_scale_height] {sid}: height {h_scene:.3f} units ({r['height_in_metres_via_gt_scale']:.2f} m via GT scale, GT {h_gt:.2f} m), "
              f"road tilt {r['road_tilt_deg_median']:.1f} deg; scale GT {s_gt:.3f}, from GT height {r['scale_from_gt_height']:.3f} "
              f"(err {r['scale_from_gt_height'] / s_gt - 1:+.1%}), from prior {args.prior_height} m {r['scale_from_prior']:.3f} "
              f"(err {r['scale_from_prior'] / s_gt - 1:+.1%}); {len(heights)} frames", flush=True)
    mean = {"abs_err_scale_from_gt_height": float(np.mean([abs(r["scale_from_gt_height"] / r["gt_to_est_scale"] - 1) for r in rows.values()])),
            "abs_err_scale_from_prior": float(np.mean([abs(r["scale_from_prior"] / r["gt_to_est_scale"] - 1) for r in rows.values()])),
            "abs_err_current": float(np.mean([abs(1 / r["gt_to_est_scale"] - 1) for r in rows.values()]))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, **vars(args), "scenes": rows, "mean": mean, "dashrecon_commit": commit}, f, indent=2)
    print(f"[diagnose_scale_height] mean |scale error|: current {mean['abs_err_current']:.1%}, from GT height "
          f"{mean['abs_err_scale_from_gt_height']:.1%}, from prior {args.prior_height} m {mean['abs_err_scale_from_prior']:.1%}", flush=True)


if __name__ == "__main__":
    main()
