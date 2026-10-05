"""M7 (docs/PAPER_PLAN.md) for E45 (docs/EXPERIMENTS.md): floaters and road height against LiDAR. EVALUATION CODE: reads GT
LiDAR, ego poses and calibration.

LiDAR of every --lidar_stride-th frame is put into the run's world the way scripts/eval_cross_camera.py places the
side cameras: vehicle frame -> GT FRONT camera frame (GT extrinsics) -> scaled by the Sim(3) scale between the GT and
the estimated FRONT trajectories -> that frame's estimated FRONT camera (pose_dir, before CamPose refinement). Points
closer than --min_range metres to the sensor are dropped (ego car), the rest voxelised (--voxel scene units).

Per run (--runs label=log_dir), at the D-Z1 moves (--moves, every --frame_stride-th frame, cameras placed like the
LiDAR) and at the real FRONT_LEFT / FRONT_RIGHT placements:
    LiDAR depth: the aggregated points projected and z-buffered on the render grid, then a --min_filter_px minimum
        filter (thin far points leaking between near ones);
    depth calibration: the trajectory Sim(3) scale does not fix the reconstruction's own depth scale, and the mismatch
        depends on depth (val056 E41s: renders 1.2-1.35 x deeper than the LiDAR near the camera, 1.06-1.16 x far away).
        Per frame, log(render depth) = a + b log(LiDAR depth) is fitted on the frame's own FRONT view (opaque pixels,
        two passes, residuals over 0.2 dropped) and every moved view of that frame is compared with
        d_hat = exp(a) LiDAR^b, so what is measured is the extra error of moving, not the reconstruction's scale;
    floater rate: share of the pixels with LiDAR depth where the render is opaque (alpha > 0.5) and its depth is less
        than (1 - --floater_rel) x d_hat: something rendered in front of every real surface;
    depth AbsRel: |render - d_hat| / d_hat over the opaque pixels.
Road height: LiDAR ground points (Waymo ground label) give a median ground height per --cell grid cell; per run, the
opacity-weighted distribution of (Gaussian z - LiDAR ground z) for Gaussians between 0.6 units under and 0.15 units over
the ground (above that sit cars and kerbs), in metres; its p90 - p10 is the thickness of the road layer (the offset
carries the depth-scale mismatch above); with --planes also (height field - LiDAR ground) of scripts/fit_planes.py.

Output: --out json and a printed table.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/eval_floaters.py --runs E41s=results/E41s/val056 E45s=results/E45s/val056 \\
        --gt_root data/waymo/processed/validation --out results/_diagnostics/floaters/val056.json \\
        --planes data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomap/planes.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.novel import attach_view_gate, build_trainer, to_device, view_gate_path  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from eval_cross_camera import gt_cam_to_ego, umeyama_scale  # noqa: E402


def lidar_world(gt_dir, frames, est, front_to_ego, s, stride, min_range, voxel):
    """Aggregated LiDAR in the run's world (scene units) and its ground flag."""
    cam_from_veh = np.linalg.inv(front_to_ego)
    pts, gnd = [], []
    for i in range(0, len(frames), stride):
        a = np.fromfile(os.path.join(gt_dir, "lidar", f"{int(frames[i]):03d}.bin"), dtype=np.float32).reshape(-1, 14)
        o, p = a[:, 0:3].astype(np.float64), a[:, 3:6].astype(np.float64)
        keep = np.linalg.norm(p - o, axis=1) > min_range
        p = p[keep]
        pc = (np.c_[p, np.ones(len(p))] @ cam_from_veh.T)[:, :3] * s
        pw = (np.c_[pc, np.ones(len(pc))] @ est[i].T)[:, :3]
        pts.append(pw)
        gnd.append(a[keep, 10] > 0)
    pts, gnd = np.concatenate(pts), np.concatenate(gnd)
    ijk = np.floor(pts / voxel).astype(np.int64) + (1 << 20)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]  # one int64 per voxel (2D np.unique is far slower)
    _, first = np.unique(key, return_index=True)
    return pts[first], gnd[first]


def lidar_depth(pts_t, w2c, k, h, w, min_filter):
    pc = pts_t @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 0.1
    pc, z = pc[ok], z[ok]
    u = (k[0, 0] * pc[:, 0] / z + k[0, 2]).long()
    v = (k[1, 1] * pc[:, 1] / z + k[1, 2]).long()
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = torch.full((h * w,), float("inf"), device=pts_t.device)
    d.scatter_reduce_(0, (v[ok] * w + u[ok]), z[ok], reduce="amin")
    d = d.view(1, 1, h, w)
    if min_filter > 1:
        d = -F.max_pool2d(-d, min_filter, 1, min_filter // 2)
    d = d[0, 0]
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--planes", default=None)
    parser.add_argument("--moves", nargs="+", default=["right=0", "right=-1.29", "right=1.29", "yaw=-20", "yaw=20", "yaw=-45", "yaw=45", "up=0.86"])
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--lidar_stride", type=int, default=2)
    parser.add_argument("--min_range", type=float, default=3.0)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--min_filter_px", type=int, default=3)
    parser.add_argument("--floater_rel", type=float, default=0.2)
    parser.add_argument("--cell", type=float, default=0.25)
    args = parser.parse_args()
    commit = git_commit()
    dev = torch.device("cuda")
    from datasets.driving_dataset import DrivingDataset

    t0 = time.time()
    results, ground_ref = {}, None
    for spec in args.runs:
        label, log_dir = spec.split("=", 1)
        cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
        ds = DrivingDataset(data_cfg=cfg.data)
        tr = build_trainer(cfg, ds, dev)
        tr.resume_from_checkpoint(ckpt_path=os.path.join(log_dir, "checkpoint_final.pth"), load_only_model=True)
        if view_gate_path(cfg) is not None:
            attach_view_gate(tr, view_gate_path(cfg))
        tr.set_eval()
        frames = np.arange(ds.start_timestep, ds.end_timestep)
        est = []
        for i in range(len(frames)):
            _, ci = ds.full_image_set.get_image(front_image_index(ds, i), 1)
            est.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
        est = np.stack(est)
        gt_dir = os.path.join(args.gt_root, f"{int(cfg.data.scene_idx):03d}")
        front_to_ego = gt_cam_to_ego(gt_dir, 0)
        gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{int(t):03d}.txt")) @ front_to_ego for t in frames])
        s = umeyama_scale(gt_front[:, :3, 3], est[:, :3, 3])
        if ground_ref is None:
            pts, gnd = lidar_world(gt_dir, frames, est, front_to_ego, s, args.lidar_stride, args.min_range, args.voxel)
            pts_t = torch.tensor(pts, dtype=torch.float32, device=dev)
            # LiDAR ground height grid
            g = pts[gnd]
            ij = np.floor(g[:, :2] / args.cell).astype(np.int64)
            org = ij.min(0)
            ij -= org
            shape = tuple(ij.max(0) + 1)
            flat = ij[:, 0] * shape[1] + ij[:, 1]
            order = np.argsort(flat)
            uniq, start, cnt = np.unique(flat[order], return_index=True, return_counts=True)
            gz = np.full(shape[0] * shape[1], np.nan)
            gzs = g[order, 2]
            gz[uniq] = [np.median(gzs[a:a + c]) for a, c in zip(start, cnt)]
            ground_ref = (org, gz.reshape(shape), s)
            print(f"[eval_floaters] {len(pts):,} LiDAR points ({int(gnd.sum()):,} ground) in the run world, scale {s:.4f} ({time.time() - t0:.0f} s)", flush=True)
        org, gz, s_ref = ground_ref

        def ground_at(xy):
            ij = np.floor(xy / args.cell).astype(np.int64) - org
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < gz.shape[0]) & (ij[:, 1] < gz.shape[1])
            out = np.full(len(xy), np.nan)
            out[ok] = gz[ij[ok, 0], ij[ok, 1]]
            return out

        bg = tr.models["Background"]
        m = bg._means.detach().cpu().numpy()
        op = torch.sigmoid(bg._opacities.detach()[:, 0]).cpu().numpy()
        gzm = ground_at(m[:, :2])
        dz = m[:, 2] - gzm
        near = np.isfinite(dz) & (dz > -0.6) & (dz < 0.15)
        wq = lambda x, w, q: float(np.interp(q, np.cumsum(w[np.argsort(x)]) / w.sum(), np.sort(x)))  # noqa: E731
        qs = [wq(dz[near], op[near], q) / s_ref for q in (0.1, 0.5, 0.9)]
        road = {"gaussians_near_ground": int(near.sum()), "offset_m_p10_p50_p90": qs, "thickness_m_p90_p10": qs[2] - qs[0]}
        # per-frame depth calibration on the frame's own FRONT view
        calib = {}
        with torch.no_grad():
            for i in range(0, len(frames), args.frame_stride):
                ii, ci = ds.full_image_set.get_image(front_image_index(ds, i), 1)
                ii, ci = to_device(ii, dev), to_device(ci, dev)
                c2w = torch.tensor(est[i], dtype=torch.float32, device=dev)
                h, w = int(ci["height"]), int(ci["width"])
                out = render_at(tr, ii, ci, c2w, ci["intrinsics"], (h, w))
                d_l = lidar_depth(pts_t, torch.linalg.inv(c2w), ci["intrinsics"], h, w, args.min_filter_px)
                m = (d_l > 0) & (out["opacity"][..., 0] > 0.5)
                x, y = torch.log(d_l[m]), torch.log(out["depth"][..., 0][m].clamp(min=1e-3))
                keep = torch.ones_like(x, dtype=torch.bool)
                for _ in range(2):
                    A = torch.stack([torch.ones_like(x[keep]), x[keep]], 1)
                    sol = torch.linalg.lstsq(A, y[keep][:, None]).solution[:, 0]
                    keep = (y - (sol[0] + sol[1] * x)).abs() < 0.2
                calib[i] = (float(sol[0]), float(sol[1]))
        # depth / floaters at the moves
        per_move = {}
        with torch.no_grad():
            for spec_m in args.moves:
                mv = ViewMove.parse(spec_m)
                fl, n_px, absrel = 0, 0, []
                for i in range(0, len(frames), args.frame_stride):
                    ii, ci = ds.full_image_set.get_image(front_image_index(ds, i), 1)
                    ii, ci = to_device(ii, dev), to_device(ci, dev)
                    c2w = moved_c2w(torch.tensor(est[i], dtype=torch.float32, device=dev), mv)
                    h, w = int(ci["height"]), int(ci["width"])
                    out = render_at(tr, ii, ci, c2w, ci["intrinsics"], (h, w))
                    d_r = out["depth"][..., 0]
                    a_r = out["opacity"][..., 0]
                    d_l = lidar_depth(pts_t, torch.linalg.inv(c2w), ci["intrinsics"], h, w, args.min_filter_px)
                    has = d_l > 0
                    opq = has & (a_r > 0.5)
                    if opq.sum() < 100:
                        continue
                    a_c, b_c = calib[i]
                    d_hat = torch.exp(a_c + b_c * torch.log(d_l.clamp(min=1e-3)))
                    fl += int((opq & (d_r < (1 - args.floater_rel) * d_hat)).sum())
                    n_px += int(has.sum())
                    absrel.append(float(((d_r[opq] - d_hat[opq]).abs() / d_hat[opq]).mean()))
                per_move[spec_m] = {"floater_rate": fl / max(n_px, 1), "depth_absrel_calibrated": float(np.mean(absrel)) if absrel else None}
        results[label] = {"log_dir": log_dir, "road_layer": road, "moves": per_move}
        print(f"[eval_floaters] {label}: road layer offset p10/p50/p90 " + " / ".join(f"{x:+.2f}" for x in road["offset_m_p10_p50_p90"])
              + f" m (thickness {road['thickness_m_p90_p10']:.2f} m); calibration exponent median {np.median([c[1] for c in calib.values()]):.3f}; floater rate " + "  ".join(f"{k} {v['floater_rate']:.3f}" for k, v in per_move.items()), flush=True)
        del tr, ds
        torch.cuda.empty_cache()
    planes = None
    if args.planes:
        z = np.load(args.planes)
        planes = {}
        for gi in range(int(z["num_ground"])):
            hgt, sup, o = z[f"ground{gi}_height"], z[f"ground{gi}_support"], z[f"ground{gi}_origin"]
            cc = (np.argwhere(sup) + 0.5) * float(z["cell"]) + o
            diff = hgt[sup] - ground_at(cc)
            diff = diff[np.isfinite(diff)]
            planes[str(z["ground_classes"][gi])] = {"cells": int(len(diff)), "offset_m_p10_p50_p90": [float(np.percentile(diff, q)) / s_ref for q in (10, 50, 90)]}
        print(f"[eval_floaters] height fields vs LiDAR ground: {planes}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, "params": vars(args), "metre_to_units": s_ref, "results": results, "height_fields": planes,
                   "dashrecon_commit": commit}, f, indent=1)


if __name__ == "__main__":
    main()
