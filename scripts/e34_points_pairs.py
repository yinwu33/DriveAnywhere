"""E34 (docs/E34_DATA.md): "points" pairs, FreeVS-style: FRONT depth splatted into the target cameras.

For every target of scripts/e34_undistort.py (camera c, timestep t), the FRONT frames whose camera lies on the driven
path between --path_back world units behind and --path_fwd units ahead of the FRONT camera at t (content beside the
car was seen by FRONT 10-30 m earlier, so a frame window would depend on the speed), evenly subsampled to at most
--max_window_frames, are back-projected and splatted into camera c at t:
    - points: MapAnything depth of the undistorted FRONT images (scripts/run_pose.py --camera_dir, depth grid), kept
      where depth > 0, not dynamic, not sky (the warped FRONT masks), confidence >= the --conf_percentile of all
      frames and depth < --max_depth (the Phase 5 rule of scripts/run_fusion.py); colour = the undistorted FRONT image
      on the depth grid (dashrecon.io.image_to_depth_grid);
    - camera: the target's own estimated K on the pair grid (targets.json) and its joint-calibration pose in the
      reconstruction world (dashrecon.e34.geometry);
    - splat: each point covers its footprint, a depth-grid pixel's world size seen from the target camera
      (the closest odd width, at least 1 px, at most --max_splat_px per axis), nearest point wins (z-buffer, ties to the lower point
      index); empty pixels are black.
Reads no image of the target cameras. Writes <pairs_dir>/points/<t:03d>_<cam>.jpg (pair grid, JPEG 95) and
<pairs_dir>/points.json (parameters, per-image coverage, runtime).

Example (main venv):
    .venvs/main/bin/python scripts/e34_points_pairs.py --scene_id wt646 \
        --calib_dir data/e34/work/wt646/calib-glomap-cams --pose_dir data/e34/work/wt646/pose-glomap-cams_depth-mapanything \
        --mask_dir data/e34/work/wt646/mask-gsam2_sky-segformer_img-glomap-cams --front_root data/e34/front \
        --pairs_dir data/e34/pairs/wt646 --path_back 40 --path_fwd 10 --max_window_frames 40 --conf_percentile 30 \
        --max_depth 60 --max_splat_px 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.geometry import check_front_consistency, target_world_poses  # noqa: E402
from dashrecon.e34.layout import FRONT  # noqa: E402
from dashrecon.fusion.backproject import backproject  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402

JPEG_QUALITY = 95


def front_points(pose_dir: str, mask_dir: str, image_dir: str, conf_percentile: float, max_depth: float) -> dict:
    """Per FRONT frame: world points, colours and world footprint (size of a depth-grid pixel at that depth)."""
    frames = io.read_frames(pose_dir)
    np.testing.assert_array_equal(io.read_frames(mask_dir), frames)
    grid = io.read_meta(pose_dir)["depth_grid"]
    k_depth = io.depth_intrinsics(io.read_intrinsics(pose_dir), grid)
    poses = io.read_poses(pose_dir)
    depths = [io.read_depth(pose_dir, int(t)) for t in frames]
    confs = [io.read_depth_conf(pose_dir, int(t)) for t in frames]
    thr = float(np.percentile(np.concatenate([c[d > 0] for c, d in zip(confs, depths)]), conf_percentile))
    out = {}
    for i, t in enumerate(frames):
        t = int(t)
        dyn = io.mask_to_depth_grid(io.read_mask(mask_dir, "dynamic", t), grid)
        sky = io.mask_to_depth_grid(io.read_mask(mask_dir, "sky", t), grid)
        d = depths[i]
        valid = (d > 0) & ~dyn & ~sky & (confs[i] >= thr) & (d < max_depth)
        xyz = backproject(d, k_depth[i], poses[i], valid)
        rgb = io.image_to_depth_grid(os.path.join(image_dir, f"{t:03d}_{FRONT}.jpg"), grid)[valid]
        foot = d[valid] / np.sqrt(k_depth[i, 0, 0] * k_depth[i, 1, 1])
        out[t] = (xyz.astype(np.float32), rgb, foot.astype(np.float32))
    return {"frames": frames, "points": out, "conf_threshold": thr}


def path_window(centres: np.ndarray, v: int, back: float, fwd: float, max_frames: int) -> list[int]:
    """Frames whose position along the driven path (cumulative FRONT camera-centre distance) is within [-back, +fwd]
    of frame v, evenly subsampled to at most max_frames (always keeping the first and last)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))])
    sel = np.nonzero((s >= s[v] - back) & (s <= s[v] + fwd))[0]
    if len(sel) > max_frames:
        sel = sel[np.unique(np.round(np.linspace(0, len(sel) - 1, max_frames)).astype(np.int64))]
    return [int(x) for x in sel]


def splat(xyz: torch.Tensor, rgb: torch.Tensor, foot: torch.Tensor, K: np.ndarray, c2w: np.ndarray, hw: tuple,
          max_px: int) -> tuple[np.ndarray, float]:
    """Z-buffered footprint splat of world points into one camera; returns (H,W,3) uint8 and covered fraction."""
    h, w = hw
    dev = xyz.device
    w2c = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)
    pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 1e-3
    pc, z, rgb, foot = pc[front], z[front], rgb[front], foot[front]
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    u = torch.round(fx * pc[:, 0] / z + cx).long()
    v = torch.round(fy * pc[:, 1] / z + cy).long()
    # the odd splat width 2 * half + 1 closest to the footprint in pixels
    half_x = torch.clamp(torch.round((foot * fx / z - 1) / 2), 0, (max_px - 1) // 2).long()
    half_y = torch.clamp(torch.round((foot * fy / z - 1) / 2), 0, (max_px - 1) // 2).long()
    ids = torch.arange(len(z), device=dev)
    H = (max_px - 1) // 2
    pix_l, z_l, id_l = [], [], []
    for dy in range(-H, H + 1):
        for dx in range(-H, H + 1):
            sel = (half_x >= abs(dx)) & (half_y >= abs(dy))
            uu, vv = u[sel] + dx, v[sel] + dy
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            pix_l.append((vv * w + uu)[inside])
            z_l.append(z[sel][inside])
            id_l.append(ids[sel][inside])
    pix, zz, pid = torch.cat(pix_l), torch.cat(z_l), torch.cat(id_l)
    zbuf = torch.full((h * w,), float("inf"), device=dev).scatter_reduce(0, pix, zz, reduce="amin")
    win = zz <= zbuf[pix]
    owner = torch.full((h * w,), len(z), dtype=torch.long, device=dev).scatter_reduce(0, pix[win], pid[win], reduce="amin")
    covered = owner < len(z)
    img = torch.zeros((h * w, 3), dtype=torch.uint8, device=dev)
    img[covered] = rgb[owner[covered]]
    return img.reshape(h, w, 3).cpu().numpy(), float(covered.float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--mask_dir", required=True, help="warped FRONT masks (scripts/e34_undistort.py)")
    parser.add_argument("--front_root", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--path_back", type=float, required=True, help="world units")
    parser.add_argument("--path_fwd", type=float, required=True, help="world units")
    parser.add_argument("--max_window_frames", type=int, required=True)
    parser.add_argument("--conf_percentile", type=float, required=True)
    parser.add_argument("--max_depth", type=float, required=True)
    parser.add_argument("--max_splat_px", type=int, required=True, help="odd")
    args = parser.parse_args()
    assert args.max_splat_px % 2 == 1, args.max_splat_px
    commit = git_commit()
    t0 = time.time()
    scene = get_scene(args.scene_id)
    out_dir = os.path.join(args.pairs_dir, "points")
    assert not os.path.exists(out_dir), f"{out_dir} exists"
    check_front_consistency(args.calib_dir, args.pose_dir)
    calib = io.read_meta(args.calib_dir)
    a = calib["frame_offset"]
    targets = io.read_json(os.path.join(args.pairs_dir, "targets.json"))
    image_dir = os.path.join(args.front_root, f"{scene.scene_idx:03d}", "images")
    fp = front_points(args.pose_dir, args.mask_dir, image_dir, args.conf_percentile, args.max_depth)
    dev = torch.device("cuda")
    gpu = {v: tuple(torch.from_numpy(x).to(dev) for x in p) for v, p in fp["points"].items()}
    centres = io.read_poses(args.pose_dir)[:, :3, 3]
    np.testing.assert_array_equal(fp["frames"], np.arange(len(centres)))
    poses = {}
    for c_str in targets["cameras"]:
        frames, c2w = target_world_poses(args.calib_dir, args.pose_dir, int(c_str))
        poses[int(c_str)] = dict(zip((int(t) for t in frames), c2w))
    t1 = time.time()
    os.makedirs(out_dir)
    hw = tuple(targets["pair_hw"])
    rows = []
    for tg in targets["targets"]:
        cam, t = tg["cam"], tg["t"]
        v = t - a
        src = path_window(centres, v, args.path_back, args.path_fwd, args.max_window_frames)
        xyz, rgb, foot = (torch.cat([gpu[s][k] for s in src]) for k in range(3))
        K = np.asarray(targets["cameras"][str(cam)]["K_pair"])
        img, cov = splat(xyz, rgb, foot, K, poses[cam][t], hw, args.max_splat_px)
        Image.fromarray(img).save(os.path.join(out_dir, f"{t:03d}_{cam}.jpg"), quality=JPEG_QUALITY)
        rows.append({"cam": cam, "t": t, "input": f"points/{t:03d}_{cam}.jpg", "front_frames": [src[0] + a, src[-1] + a], "num_front_frames": len(src),
                     "num_points": int(len(xyz)), "coverage": cov})
    covs = {}
    for r in rows:
        covs.setdefault(str(r["cam"]), []).append(r["coverage"])
    with open(os.path.join(args.pairs_dir, "points.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, "params": {k: v for k, v in vars(args).items() if k != "scene_id"},
                   "conf_threshold": fp["conf_threshold"], "frame_offset": a,
                   "mean_coverage": {c: float(np.mean(v)) for c, v in covs.items()},
                   "runtime_s": {"load": t1 - t0, "splat": time.time() - t1}, "jpeg_quality": JPEG_QUALITY,
                   "uses_oracle": False, "oracle_note": "FRONT depth, masks and joint self-calibration only",
                   "dashrecon_commit": commit, "pairs": rows}, f, indent=2)
    print(f"[e34_points_pairs] {args.scene_id}: {len(rows)} inputs, coverage "
          + ", ".join(f"cam{c} {np.mean(v):.2f}" for c, v in sorted(covs.items()))
          + f" ({t1 - t0:.0f} s load, {time.time() - t1:.0f} s splat) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
