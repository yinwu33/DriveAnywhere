"""E42c (docs/EXPERIMENTS.md): a high-resolution road texture as surfels, from full-resolution FRONT pixels on the mesh.

E42 put the fused cloud's road points (518 px depth grid, 5 cm voxels = 12 cm on val056) in as surfels: the lane
markings stay straight under sideways moves, but the near road is much blurrier than the images (FRONT held-out PSNR
32.5 -> 30.7). Here every road pixel of every training frame at full image resolution is cast onto the scene mesh
(--mesh, nvdiffrast z-depth; the mesh is smooth on the road, so the texture lies on one surface) and lands in a cell of
--cell scene units on the ground plane (world z is up). Each cell keeps the observation with the smallest ground
footprint (depth / fx over the cosine between the ray and the surface normal): the closest, least grazing view, so a
cell is coloured by one sharp pixel, not an average of blurred ones. Road pixels are SegFormer road (--mask_dir road)
outside the Grounded-SAM-2 vehicle mask (--mask_dir dynamic, all vehicles: a car in front of the road would colour
the road behind it), nearer than --max_depth; normals come from the mesh depth map.

Output: --out_ply with xyz, rgb, normals and labels = 1 (road), as scripts/road_surfel_composite.py --ply reads it;
meta json next to it. GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/road_ortho_surfels.py --scene_id val056 \\
        --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything \\
        --mask_dir data/dashrecon/val056/mask-gsam2_sky-segformer_img-glomap \\
        --mesh data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap/mesh_nksr.ply \\
        --processed_root data/dashrecon/_undistorted/calib-glomap --test_stride 10 --cell 0.015 --max_depth 25 \\
        --out_ply results/E42c/val056_road/road_ortho.ply
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
from dashrecon.mesh.render import MeshDepthRenderer  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene, train_frame_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--pose_dir", "--mask_dir", "--mesh", "--processed_root", "--out_ply"):
        parser.add_argument(a, required=True)
    parser.add_argument("--test_stride", type=int, required=True)
    parser.add_argument("--cell", type=float, required=True)
    parser.add_argument("--max_depth", type=float, required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    dev = torch.device("cuda")
    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.pose_dir)
    poses = io.read_poses(args.pose_dir)
    k_img = io.read_intrinsics(args.pose_dir)
    if k_img.ndim == 2:
        k_img = np.tile(k_img, (len(frames), 1, 1))
    train = train_frame_mask(frames, scene.start_timestep, args.test_stride)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    mesh = io.read_mesh_ply(args.mesh)
    rend = MeshDepthRenderer(mesh["vertices"], mesh["faces"], 0.05, 500.0, dev)

    best_key = np.zeros(0, np.int64)
    best = {"score": np.zeros(0, np.float32), "xyz": np.zeros((0, 3), np.float32), "rgb": np.zeros((0, 3), np.uint8),
            "nrm": np.zeros((0, 3), np.float32)}
    for i in np.nonzero(train)[0]:
        t = int(frames[i])
        img = np.asarray(Image.open(os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg")).convert("RGB"))
        h, w = img.shape[:2]
        k = torch.as_tensor(k_img[i], dtype=torch.float32, device=dev)
        c2w = torch.as_tensor(poses[i], dtype=torch.float32, device=dev)
        d = rend.depth(k, c2w, h, w)
        # camera-frame points of every pixel and their normals from the depth map
        v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        p = torch.stack([(u - k[0, 2]) / k[0, 0] * d, (v - k[1, 2]) / k[1, 1] * d, d], -1)
        n = torch.zeros_like(p)
        n[1:-1, 1:-1] = torch.cross(p[1:-1, 2:] - p[1:-1, :-2], p[2:, 1:-1] - p[:-2, 1:-1], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ray = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cos = (n * ray).sum(-1).abs()
        n = torch.where((n * ray).sum(-1, keepdim=True) > 0, -n, n)  # towards the camera
        road = torch.from_numpy(io.read_mask(args.mask_dir, "road", t) & ~io.read_mask(args.mask_dir, "dynamic", t)).to(dev)
        valid = road & (d > 0) & (d < args.max_depth) & (cos > 0.05)
        valid[0], valid[-1], valid[:, 0], valid[:, -1] = False, False, False, False
        score = (d / k[0, 0] / cos.clamp(min=0.15))[valid].cpu().numpy().astype(np.float32)
        pw = (p[valid] @ c2w[:3, :3].T + c2w[:3, 3]).cpu().numpy()
        nw = (n[valid] @ c2w[:3, :3].T).cpu().numpy()
        rgb = img[valid.cpu().numpy()]
        cells = np.floor(pw[:, :2] / args.cell).astype(np.int64)
        key = (cells[:, 0] + 2**20) * 2**21 + (cells[:, 1] + 2**20)
        # merge with the running best: per cell the smallest footprint
        all_key = np.concatenate([best_key, key])
        all_score = np.concatenate([best["score"], score])
        order = np.lexsort((all_score, all_key))
        first = np.ones(len(order), bool)
        first[1:] = all_key[order][1:] != all_key[order][:-1]
        sel = order[first]
        cat = {"xyz": np.concatenate([best["xyz"], pw.astype(np.float32)]), "rgb": np.concatenate([best["rgb"], rgb]),
               "nrm": np.concatenate([best["nrm"], nw.astype(np.float32)])}
        best_key = all_key[sel]
        best = {"score": all_score[sel], **{kk: vv[sel] for kk, vv in cat.items()}}
        if i % 20 == 0:
            print(f"[road_ortho_surfels] frame {t}: {int(valid.sum()):,} road pixels, {len(best_key):,} cells ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    io.write_ply(args.out_ply, best["xyz"].astype(np.float64), best["rgb"], normals=best["nrm"].astype(np.float64),
                 labels=np.ones(len(best_key), np.uint8))
    meta = {"scene_id": args.scene_id, "params": vars(args), "cells": int(len(best_key)),
            "footprint_median": float(np.median(best["score"])), "footprint_p90": float(np.percentile(best["score"], 90)),
            "uses_oracle": False, "runtime_s": time.time() - t0, "dashrecon_commit": commit}
    with open(os.path.splitext(args.out_ply)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[road_ortho_surfels] {len(best_key):,} road cells, footprint median {meta['footprint_median']:.4f} -> {args.out_ply} "
          f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
