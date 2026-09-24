"""Qualitative Phase 3 visualisation: back-projected point cloud, BEV image and web-viewer data.

Reads only the section-5 products of one scene (via ``dashrecon.io``) plus the FRONT images for
colour. Outputs go to ``<scene_dir>/vis/`` and are not part of the data contract:
    points_vis.ply   subsampled coloured point cloud (world frame of poses_c2w.npy)
    bev.png          top-down orthographic view with the camera trajectory in red
    viewer_data.js   compact point cloud + cameras for the HTML viewer

Example:
    .venvs/mapanything/bin/python scripts/vis_pose.py --scene_id val_static_a \
        --scene_dir data/dashrecon/val_static_a/pose-mapanything_depth-mapanything \
        --processed_root data/waymo/processed/validation --frame_stride 2 --conf_percentile 30 \
        --max_depth 60 --max_points 400000 --seed 0
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene  # noqa: E402


def image_on_depth_grid(path: str, grid: dict) -> np.ndarray:
    """Resize + crop an image exactly like MapAnything's ``crop_resize_if_necessary``."""
    img = Image.open(path).convert("RGB")
    assert [img.size[1], img.size[0]] == grid["image_hw"], (img.size, grid["image_hw"])
    rh, rw = grid["resized_hw"]
    top, left = grid["crop_top_left"]
    th, tw = grid["depth_hw"]
    img = img.resize((rw, rh), resample=Image.LANCZOS)
    return np.asarray(img)[top : top + th, left : left + tw]


def backproject(depth: np.ndarray, k: np.ndarray, c2w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points of the masked pixels of a z-depth map (pixel centres at integer coordinates)."""
    v, u = np.nonzero(mask)
    z = depth[v, u]
    x = (u - k[0, 2]) / k[0, 0] * z
    y = (v - k[1, 2]) / k[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def render_bev(xyz: np.ndarray, rgb: np.ndarray, cam_xyz: np.ndarray, size: int) -> Image.Image:
    """Top-down orthographic render (x forward -> image up, y left -> image left)."""
    lo = np.minimum(xyz[:, :2].min(0), cam_xyz[:, :2].min(0)) - 2.0
    hi = np.maximum(xyz[:, :2].max(0), cam_xyz[:, :2].max(0)) + 2.0
    res = (hi - lo).max() / size
    h = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    w = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    order = np.argsort(xyz[:, 2])  # paint low points first so higher surfaces stay on top
    rows = ((hi[0] - xyz[order, 0]) / res).astype(int)
    cols = ((hi[1] - xyz[order, 1]) / res).astype(int)
    canvas[rows, cols] = rgb[order]
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    traj = [((hi[1] - p[1]) / res, (hi[0] - p[0]) / res) for p in cam_xyz]
    draw.line(traj, fill=(230, 30, 30), width=3)
    draw.ellipse([traj[0][0] - 6, traj[0][1] - 6, traj[0][0] + 6, traj[0][1] + 6], outline=(230, 30, 30), width=3)
    return img


def viewer_payload(scene_id: str, xyz: np.ndarray, rgb: np.ndarray, poses: np.ndarray, k: np.ndarray,
                   image_hw: list[int], meta: dict) -> str:
    """JS snippet registering the scene for the HTML viewer (uint16-quantised positions, base64)."""
    lo, hi = xyz.min(0), xyz.max(0)
    q = np.round((xyz - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype("<u2")
    payload = {
        "id": scene_id,
        "n": int(len(xyz)),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "pos": base64.b64encode(q.tobytes()).decode("ascii"),
        "col": base64.b64encode(rgb.astype(np.uint8).tobytes()).decode("ascii"),
        "cams": [np.round(p, 5).tolist() for p in poses],
        "fx": float(k[0, 0]), "fy": float(k[1, 1]), "w": image_hw[1], "h": image_hw[0],
        "stats": {
            "frames": meta["num_views"],
            "path_m": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()),
            "runtime_s": meta["runtime_s"],
            "peak_vram_gb": meta["peak_vram_gb"],
            "valid_depth": meta["valid_depth_fraction"],
        },
    }
    return f"window.SCENES = window.SCENES || {{}};\nwindow.SCENES[{json.dumps(scene_id)}] = {json.dumps(payload)};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--frame_stride", type=int, required=True)
    parser.add_argument("--conf_percentile", type=float, required=True, help="drop pixels below this confidence percentile")
    parser.add_argument("--max_depth", type=float, required=True)
    parser.add_argument("--max_points", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.scene_dir)
    poses = io.read_poses(args.scene_dir)
    k_img = io.read_intrinsics(args.scene_dir)
    meta = io.read_meta(args.scene_dir)
    grid = meta["depth_grid"]
    k_depth = io.depth_intrinsics(k_img, grid)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")

    sel = np.arange(0, len(frames), args.frame_stride)
    confs = np.concatenate([io.read_depth_conf(args.scene_dir, int(frames[i]))[io.read_depth(args.scene_dir, int(frames[i])) > 0] for i in sel])
    conf_thr = float(np.percentile(confs, args.conf_percentile))

    all_xyz, all_rgb = [], []
    for i in sel:
        t = int(frames[i])
        depth = io.read_depth(args.scene_dir, t)
        conf = io.read_depth_conf(args.scene_dir, t)
        mask = (depth > 0) & (depth < args.max_depth) & (conf >= conf_thr)
        rgb = image_on_depth_grid(os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg"), grid)
        all_xyz.append(backproject(depth, k_depth[i], poses[i], mask))
        all_rgb.append(rgb[mask])
    xyz = np.concatenate(all_xyz).astype(np.float32)
    rgb = np.concatenate(all_rgb).astype(np.uint8)
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(len(xyz), size=min(args.max_points, len(xyz)), replace=False)
    xyz, rgb = xyz[keep], rgb[keep]

    out_dir = os.path.join(args.scene_dir, "vis")
    io.write_ply(os.path.join(out_dir, "points_vis.ply"), xyz, rgb)
    render_bev(xyz, rgb, poses[:, :3, 3], size=1400).save(os.path.join(out_dir, "bev.png"))
    k_mean = k_img.mean(axis=0) if k_img.ndim == 3 else k_img
    with open(os.path.join(out_dir, "viewer_data.js"), "w") as f:
        f.write(viewer_payload(args.scene_id, xyz, rgb, poses[:: args.frame_stride], k_mean, grid["image_hw"], meta))
    with open(os.path.join(out_dir, "vis_params.json"), "w") as f:
        json.dump({**vars(args), "conf_threshold": conf_thr, "num_points": int(len(xyz))}, f, indent=2)
    print(f"[vis_pose] {args.scene_id}: {len(xyz)} points (conf >= {conf_thr:.3f}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
