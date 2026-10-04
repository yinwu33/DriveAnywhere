"""E39d (docs/EXPERIMENTS.md): Depth Anything 3 depth at high resolution for given poses, as a dashrecon pose directory.
Runs in .venvs/da3 (envs/setup_da3.sh).

The current depth (MapAnything, DECISIONS L) is predicted on a 518 x 336 grid; rendered as surfels its holes and
depth noise, not the point count, limit the observed surfaces (E39). DA3 (any-view, DECISIONS AB / D-G1) accepts the
cameras as input: every window of --window consecutive frames (--overlap frames shared with the next) is run with our
GLOMAP poses (--pose_dir, scene units) and the shared undistorted intrinsics (--calib_dir camera.json) at
--process_res (upper_bound_resize: the image is resized full-extent so that its longer side is process_res, then to a
multiple of 14), with DA3's scale alignment to the given extrinsics, so the depth comes out in the pose directory's
units. Per frame the depth of the window that has it farthest from a window edge is kept. Output <out_root>/<scene>/
<tag>/: frames.txt, poses_c2w.npy and intrinsics.npy copied from --pose_dir (image grid), depth/<t>.npy (float32,
DA3 grid, 0 = invalid), depth_conf/<t>.npy, meta.json with depth_grid (a full-extent resize, no crop) so
scripts/run_fusion.py reads it like any other backend. GT is not read.

Example:
    CUDA_HOME=$HOME/cuda-12.6 HF_HUB_OFFLINE=1 PYTHONPATH=. .venvs/da3/bin/python scripts/run_depth_da3.py --scene_id val056 \\
        --processed_root data/dashrecon/_undistorted/calib-glomap --calib_dir data/dashrecon/val056/calib-glomap \\
        --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything --out_root data/dashrecon/_e39d \\
        --model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 --process_res 1008 --window 24 --overlap 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402

TAG = "pose-glomap_depth-da3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--processed_root", required=True, help="undistorted image root (calib-glomap)")
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--pose_dir", required=True, help="poses (scene units) and frames to reuse")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--process_res", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--max_frames", type=int, default=0, help="pilot: only the first n frames (0 = all)")
    args = parser.parse_args()
    assert 0 < args.overlap < args.window, (args.window, args.overlap)
    commit = git_commit()
    from depth_anything_3.api import DepthAnything3

    scene = get_scene(args.scene_id)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    frames = io.read_frames(args.pose_dir)
    poses = io.read_poses(args.pose_dir)
    k_img = io.read_intrinsics(args.pose_dir)
    if k_img.ndim == 2:
        k_img = np.tile(k_img, (len(frames), 1, 1))
    cam = io.read_camera(args.calib_dir)
    assert np.allclose(cam["K"], k_img, rtol=1e-3), "pose_dir intrinsics differ from the calibration"
    if args.max_frames:
        frames, poses, k_img = frames[:args.max_frames], poses[:args.max_frames], k_img[:args.max_frames]
    n = len(frames)
    out_dir = io.scene_dir(args.out_root, args.scene_id, TAG)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth_conf"), exist_ok=True)

    model = DepthAnything3.from_pretrained(args.model_id).to(torch.device("cuda")).eval()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    step = args.window - args.overlap
    starts = list(range(0, max(n - args.window, 0) + 1, step))
    if starts[-1] + args.window < n:
        starts.append(n - args.window)
    best = np.full(n, -1)  # distance from the nearest window edge of the kept prediction
    depth_hw, scales = None, []
    w2c_all = np.linalg.inv(poses)
    for s in starts:
        idx = list(range(s, min(s + args.window, n)))
        paths = [os.path.join(img_dir, f"{int(frames[i]):03d}_0.jpg") for i in idx]
        pred = model.inference(paths, extrinsics=w2c_all[idx].astype(np.float32), intrinsics=k_img[idx].astype(np.float32),
                               process_res=args.process_res)
        scales.append(float(pred.scale_factor) if pred.scale_factor is not None else None)
        if depth_hw is None:
            depth_hw = tuple(pred.depth.shape[1:])
        assert tuple(pred.depth.shape[1:]) == depth_hw, (pred.depth.shape, depth_hw)
        for j, i in enumerate(idx):
            edge = min(j, len(idx) - 1 - j)
            if edge > best[i]:
                best[i] = edge
                d = pred.depth[j].astype(np.float32)
                d[~np.isfinite(d) | (d <= 0)] = 0.0
                io.write_depth(out_dir, int(frames[i]), d)
                io.write_depth_conf(out_dir, int(frames[i]), pred.conf[j].astype(np.float32))
        print(f"[run_depth_da3] window {s}..{idx[-1]}: is_metric {int(pred.is_metric)}, scale {scales[-1]}, "
              f"{time.time() - t0:.0f} s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    assert (best >= 0).all()
    with open(os.path.join(img_dir, f"{int(frames[0]):03d}_0.jpg"), "rb") as f:
        from PIL import Image
        w_img, h_img = Image.open(f).size
    io.write_frames(out_dir, frames)
    io.write_poses(out_dir, poses)
    io.write_intrinsics(out_dir, k_img)
    src_meta = io.read_meta(args.pose_dir)
    io.write_meta(out_dir, {
        "backend": "da3", "model_id": args.model_id, "params": vars(args), "frames": n, "windows": len(starts),
        "depth_grid": {"image_hw": [h_img, w_img], "resized_hw": list(depth_hw), "crop_top_left": [0, 0], "depth_hw": list(depth_hw)},
        "poses_source": args.pose_dir, "world_frame": src_meta["world_frame"], "scale_strategy": "aligned to the given extrinsics (DA3 align_to_input_ext_scale)",
        "window_scale_factors": scales, "runtime_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "uses_oracle": False, "uses_calibration": False, "oracle_note": "undistorted FRONT images + self-calibration + GLOMAP poses only",
        "dashrecon_commit": commit})
    print(f"[run_depth_da3] wrote {out_dir}: {n} frames at {depth_hw[1]}x{depth_hw[0]}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
