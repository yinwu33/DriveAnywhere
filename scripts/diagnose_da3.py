"""D-G1 (docs/PLAN_AFTER_E30.md): Depth Anything 3 against the current pose backend. DIAGNOSTIC CODE: reads GT.

For each scene, DA3 (any-view, no poses or intrinsics given) runs on the raw (distorted) FRONT images of every
--frame_stride-th frame, and the result is compared with the GT FRONT camera (Waymo calibration and ego poses,
evaluation side only) and with the current backend (GLOMAP self-calibration + MapAnything-scaled poses,
pose-glomap_depth-mapanything, DECISIONS L):
    focal: DA3's mean fx rescaled to the 1920 px image width vs the GT fx (and GLOMAP's shared f);
    scale: Sim(3) (Umeyama) scale mapping GT FRONT camera centres onto the estimated ones; 1.0 = metric. The current
        backend is 0.43-0.79 (OPEN_QUESTIONS 15);
    ATE: RMSE of the aligned camera centres in GT metres, and the trajectory length ratio est / GT (after the scale).
DA3's camera centres come from its predicted world-to-camera extrinsics; is_metric says whether the model's depth
(and so its translation) is in metres.

With --undistorted_root, DA3 also runs (any-view, no poses) on the undistorted FRONT images the current backend's
depth was computed on (calib-glomap), and its depth is compared with that depth on the same grid: r = median over
frames of the per-frame median of DA3 depth / our depth (valid pixels). If DA3's depth is metric, our scale times r
is 1: the "rescaled" scale says how well DA3 depth alone would fix the current backend's scale, keeping its poses.

Output: --out JSON (per scene and mean) and a printed table.

Example (da3 venv):
    PYTHONPATH=. .venvs/da3/bin/python scripts/diagnose_da3.py --scene_ids val056 val039 val041 val087 val094 \
        --processed_root data/waymo/processed/validation --dashrecon_root data/dashrecon \
        --model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 --frame_stride 2 --out data/dashrecon/diagnostics/dg1_da3.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402

# Waymo camera frame (x forward, y left, z up) -> OpenCV, as scripts/eval_cross_camera.py
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
POSE_TAG = "pose-glomap_depth-mapanything"


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Sim(3) (s, R, t) with dst ~ s R src + t for (N, 3) point sets."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = float(np.trace(np.diag(d) @ sign) / ((xs ** 2).sum() / len(src)))
    return s, r, md - s * r @ ms


def compare(gt_c: np.ndarray, est_c: np.ndarray) -> dict:
    """Scale of GT -> estimate, and the ATE of the estimate brought back to GT metres."""
    s, r, t = umeyama(gt_c, est_c)
    s_inv, r_inv, t_inv = umeyama(est_c, gt_c)
    aligned = (s_inv * (r_inv @ est_c.T)).T + t_inv
    gt_len = float(np.linalg.norm(np.diff(gt_c, axis=0), axis=1).sum())
    est_len = float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum())
    return {"gt_to_est_scale": s, "ate_rmse_m": float(np.sqrt(((aligned - gt_c) ** 2).sum(1).mean())),
            "path_length_ratio": est_len / gt_len}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_ids", nargs="+", required=True)
    parser.add_argument("--processed_root", required=True, help="original processed split (raw FRONT images, GT)")
    parser.add_argument("--dashrecon_root", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--frame_stride", type=int, required=True)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--undistorted_root", help="calib-glomap undistorted image root (depth-scale check)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = git_commit()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(args.model_id).to(torch.device("cuda")).eval()
    rows = {}
    for sid in args.scene_ids:
        scene = get_scene(sid)
        gt_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}")
        pose_dir = os.path.join(args.dashrecon_root, sid, POSE_TAG)
        ours_frames = io.read_frames(pose_dir)
        frames = ours_frames[::args.frame_stride]
        paths = [os.path.join(gt_dir, "images", f"{t:03d}_0.jpg") for t in frames]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        pred = model.inference(paths, process_res=args.process_res)
        runtime, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 30
        w_proc = pred.processed_images.shape[2]
        w2c = np.tile(np.eye(4), (len(frames), 1, 1))
        w2c[:, :3, :4] = pred.extrinsics[:, :3, :4]
        da3_c = np.linalg.inv(w2c)[:, :3, 3]

        k_gt = np.loadtxt(os.path.join(gt_dir, "intrinsics", "0.txt"))
        front_to_ego = np.loadtxt(os.path.join(gt_dir, "extrinsics", "0.txt")) @ OPENCV2DATASET
        gt_c = np.stack([(np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{t:03d}.txt")) @ front_to_ego)[:3, 3] for t in frames])
        ours_c = io.read_poses(pose_dir)[::args.frame_stride][:, :3, 3]
        cam = io.read_camera(os.path.join(args.dashrecon_root, sid, "calib-glomap"))
        rows[sid] = {
            "frames": len(frames), "runtime_s": runtime, "peak_gb": peak, "is_metric": int(pred.is_metric),
            "process_hw": list(pred.processed_images.shape[1:3]),
            "focal_gt": float(k_gt[0]), "focal_da3": float(pred.intrinsics[:, 0, 0].mean() * 1920 / w_proc),
            "focal_da3_std": float(pred.intrinsics[:, 0, 0].std() * 1920 / w_proc), "focal_glomap": float(cam["K"][0][0]),
            "da3": compare(gt_c, da3_c), "ours": compare(gt_c, ours_c)}
        if args.undistorted_root is not None:
            und = [os.path.join(args.undistorted_root, f"{scene.scene_idx:03d}", "images", f"{t:03d}_0.jpg") for t in frames]
            pred_u = model.inference(und, process_res=args.process_res)
            grid = io.read_meta(pose_dir)["depth_grid"]
            (h_res, w_res), (top, left), (h_d, w_d) = grid["resized_hw"], grid["crop_top_left"], grid["depth_hw"]
            ratios = []
            for j, t in enumerate(frames):
                ours_d = io.read_depth(pose_dir, int(t))
                da3_d = cv2.resize(pred_u.depth[j], (w_res, h_res), interpolation=cv2.INTER_LINEAR)[top:top + h_d, left:left + w_d]
                valid = (ours_d > 0) & (da3_d > 0)
                ratios.append(float(np.median(da3_d[valid] / ours_d[valid])))
            r_med = float(np.median(ratios))
            rows[sid]["depth_ratio_da3_over_ours"] = r_med
            rows[sid]["depth_ratio_frame_iqr"] = [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))]
            rows[sid]["ours_rescaled_scale"] = rows[sid]["ours"]["gt_to_est_scale"] * r_med
            print(f"[diagnose_da3] {sid}: DA3 / our depth {r_med:.3f}, our scale x ratio = {rows[sid]['ours_rescaled_scale']:.3f}", flush=True)
        r = rows[sid]
        print(f"[diagnose_da3] {sid}: focal GT {r['focal_gt']:.0f} DA3 {r['focal_da3']:.0f} (std {r['focal_da3_std']:.0f}) GLOMAP {r['focal_glomap']:.0f}; "
              f"scale DA3 {r['da3']['gt_to_est_scale']:.3f} ours {r['ours']['gt_to_est_scale']:.3f}; ATE DA3 {r['da3']['ate_rmse_m']:.2f} m "
              f"ours {r['ours']['ate_rmse_m']:.2f} m; {runtime:.0f} s, {peak:.1f} GB", flush=True)
    mean = {k: float(np.mean([abs(r[k] / r["focal_gt"] - 1) for r in rows.values()])) for k in ("focal_da3", "focal_glomap")}
    for b in ("da3", "ours"):
        mean[f"{b}_scale"] = float(np.mean([r[b]["gt_to_est_scale"] for r in rows.values()]))
        mean[f"{b}_ate_rmse_m"] = float(np.mean([r[b]["ate_rmse_m"] for r in rows.values()]))
    if args.undistorted_root is not None:
        mean["ours_rescaled_scale"] = float(np.mean([r["ours_rescaled_scale"] for r in rows.values()]))
        mean["ours_rescaled_abs_err"] = float(np.mean([abs(r["ours_rescaled_scale"] - 1) for r in rows.values()]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, "model_id": args.model_id, "frame_stride": args.frame_stride, "process_res": args.process_res,
                   "input": "raw (distorted) FRONT images, no poses or intrinsics given", "scenes": rows,
                   "mean": mean, "dashrecon_commit": commit}, f, indent=2)
    print(f"[diagnose_da3] mean |focal error|: DA3 {mean['focal_da3']:.3f}, GLOMAP {mean['focal_glomap']:.3f}; "
          f"mean scale DA3 {mean['da3_scale']:.3f}, ours {mean['ours_scale']:.3f}", flush=True)


if __name__ == "__main__":
    main()
