"""E45 (docs/EXPERIMENTS.md): planar proxies of a scene's structured surfaces, from the fused cloud and FRONT images only.

User, 10-05: the ground is often broken and walls could be planes too; side-view floaters and fragmented road
(E39 vs E35, E42) come from Gaussians that only have to look right along the forward rays. Two kinds of proxy:
    ground  per class in --ground_classes (Cityscapes road, sidewalk): a height field on a --cell grid of world xy
            (world z is up): median z of the class's points per cell (cells with >= --min_cell_points), Gaussian
            smoothing over valid cells (--smooth_cells), support = valid cells dilated by --support_dilate cells
            (heights there from the smoothed field, normalised convolution);
    vertical  points of --vertical_classes (building, wall, fence): sequential RANSAC planes whose normal is within
            --max_tilt_deg of horizontal (inlier distance --inlier, at least --min_inliers), each split into connected
            patches in plane coordinates (DBSCAN --patch_eps); a patch's support is its occupancy grid (--cell) dilated
            by --support_dilate cells. A patch is kept only when it is thin (--min_planarity: of the points within 3 x --inlier of
the plane inside its footprint, that share is within --inlier; a blob of points admits many planes) and at least
--min_extent wide and --min_height tall.
Point classes: every --frame_stride-th training frame is segmented with SegFormer-B5 Cityscapes (the Phase 4 model);
each fused point projected into a frame votes the class at its pixel when the frame's estimated depth there agrees
(relative --depth_tol); the majority over frames is the point's class.

Output: --out (npz: ground fields and vertical patches, the format dashrecon.train.planar reads), <out>_meta.json
(counts, per-class point numbers, per-plane inliers / extent), <out>_topdown.jpg. GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/fit_planes.py --scene_id val056 \\
        --fused_dir data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomap \\
        --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything --processed_root data/dashrecon/_undistorted/calib-glomap \\
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --test_stride 10 --frame_stride 3 --depth_tol 0.05 \\
        --cell 0.25 --min_cell_points 3 --smooth_cells 1.0 --support_dilate 2 --inlier 0.04 --min_inliers 400 \\
        --max_tilt_deg 10 --patch_eps 0.4 --max_planes 80 \\
        --out data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomap/planes.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene, train_frame_mask  # noqa: E402

CITYSCAPES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
              "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def point_classes(xyz, frames, poses, k_depth, grid, depth_dir, img_dir, seg_model_id, depth_tol, dev):
    """Majority SegFormer class of every point over the frames that see it at their estimated depth."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    proc = SegformerImageProcessor.from_pretrained(seg_model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(seg_model_id).to(dev).eval()
    votes = torch.zeros(len(xyz), len(CITYSCAPES), dtype=torch.int32, device=dev)
    pts = torch.tensor(xyz, dtype=torch.float32, device=dev)
    th, tw = grid["depth_hw"]
    for i, t in frames:
        img = Image.open(os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg")).convert("RGB")
        w, h = img.size
        with torch.no_grad():
            inp = proc(images=img, size={"height": 32 * round(h * 1536 / w / 32), "width": 1536}, return_tensors="pt").to(dev)
            seg = proc.post_process_semantic_segmentation(model(**inp), target_sizes=[(h, w)])[0]
        rh, rw = grid["resized_hw"]
        top, left = grid["crop_top_left"]
        lab = torch.nn.functional.interpolate(seg[None, None].float(), size=(rh, rw), mode="nearest")[0, 0].long()[top:top + th, left:left + tw]
        d = torch.tensor(io.read_depth(depth_dir, t), dtype=torch.float32, device=dev)
        w2c = torch.tensor(np.linalg.inv(poses[i]), dtype=torch.float32, device=dev)
        k = torch.tensor(k_depth[i], dtype=torch.float32, device=dev)
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = (k[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + k[0, 2]).round().long()
        v = (k[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + k[1, 2]).round().long()
        ok = (z > 0.05) & (u >= 0) & (u < tw) & (v >= 0) & (v < th)
        idx = torch.nonzero(ok)[:, 0]
        dz = d[v[idx], u[idx]]
        good = (dz > 0) & ((z[idx] - dz).abs() / dz.clamp(min=1e-3) < depth_tol)
        idx = idx[good]
        votes[idx, lab[v[idx], u[idx]]] += 1
    n_votes = votes.sum(1)
    cls = votes.argmax(1)
    cls[n_votes == 0] = -1
    return cls.cpu().numpy(), n_votes.cpu().numpy()


def height_field(p, cell, min_pts, smooth, dilate):
    """Smoothed median-height grid of points p (N, 3): origin, heights, support mask, raw valid mask."""
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    org = ij.min(0) - dilate - 2
    ij -= org
    shape = tuple(ij.max(0) + dilate + 3)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], p[order, 2]
    uniq, start, cnt = np.unique(fs, return_index=True, return_counts=True)
    med = np.array([np.median(zs[a:a + c]) for a, c in zip(start, cnt)])
    hgt = np.zeros(shape[0] * shape[1])
    val = np.zeros(shape[0] * shape[1], bool)
    keep = cnt >= min_pts
    hgt[uniq[keep]], val[uniq[keep]] = med[keep], True
    hgt, val = hgt.reshape(shape), val.reshape(shape)
    num = ndimage.gaussian_filter(hgt * val, smooth)
    den = ndimage.gaussian_filter(val.astype(float), smooth)
    support = ndimage.binary_dilation(val, np.ones((3, 3), bool), dilate)
    # heights over the dilated support: normalised convolution with a wider kernel where the narrow one has no data
    wide_num = ndimage.gaussian_filter(hgt * val, smooth + dilate)
    wide_den = ndimage.gaussian_filter(val.astype(float), smooth + dilate)
    sm = np.where(den > 1e-3, num / np.maximum(den, 1e-9), wide_num / np.maximum(wide_den, 1e-9))
    return org * cell, sm.astype(np.float32), support, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--fused_dir", "--pose_dir", "--processed_root", "--seg_model_id", "--out"):
        parser.add_argument(a, required=True)
    parser.add_argument("--test_stride", type=int, required=True)
    parser.add_argument("--frame_stride", type=int, required=True)
    parser.add_argument("--depth_tol", type=float, required=True)
    parser.add_argument("--ground_classes", nargs="+", default=["road", "sidewalk"])
    parser.add_argument("--vertical_classes", nargs="+", default=["building", "wall", "fence"])
    parser.add_argument("--cell", type=float, required=True)
    parser.add_argument("--min_cell_points", type=int, required=True)
    parser.add_argument("--smooth_cells", type=float, required=True)
    parser.add_argument("--support_dilate", type=int, required=True)
    parser.add_argument("--inlier", type=float, required=True)
    parser.add_argument("--min_inliers", type=int, required=True)
    parser.add_argument("--max_tilt_deg", type=float, required=True)
    parser.add_argument("--patch_eps", type=float, required=True)
    parser.add_argument("--max_planes", type=int, required=True)
    parser.add_argument("--vertical_source", choices=["points", "mesh"], default="mesh")
    parser.add_argument("--min_planarity", type=float, default=0.6, help="inliers / points within 3 x inlier of the plane")
    parser.add_argument("--min_extent", type=float, default=1.0, help="smallest horizontal extent of a patch (scene units)")
    parser.add_argument("--min_height", type=float, default=0.3, help="smallest vertical extent of a patch (scene units)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    scene = get_scene(args.scene_id)
    fused = io.read_ply(os.path.join(args.fused_dir, "points_fused.ply"))
    xyz = fused["xyz"].astype(np.float64)
    frames = io.read_frames(args.pose_dir)
    poses = io.read_poses(args.pose_dir)
    grid = io.read_meta(args.pose_dir)["depth_grid"]
    k_depth = io.depth_intrinsics(io.read_intrinsics(args.pose_dir), grid)
    train = np.nonzero(train_frame_mask(frames, scene.start_timestep, args.test_stride))[0][::args.frame_stride]
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    cls, n_votes = point_classes(xyz, [(int(i), int(frames[i])) for i in train], poses, k_depth, grid, args.pose_dir, img_dir,
                                 args.seg_model_id, args.depth_tol, dev)
    counts = {CITYSCAPES[c]: int((cls == c).sum()) for c in range(len(CITYSCAPES)) if (cls == c).any()}
    counts["unlabelled"] = int((cls < 0).sum())
    print(f"[fit_planes] {len(xyz):,} points classified over {len(train)} frames ({time.time() - t0:.0f} s): {counts}", flush=True)

    out = {"cell": np.float32(args.cell)}
    meta_ground = []
    for g, name in enumerate(args.ground_classes):
        p = xyz[cls == CITYSCAPES.index(name)]
        org, hgt, support, val = height_field(p, args.cell, args.min_cell_points, args.smooth_cells, args.support_dilate)
        out[f"ground{g}_origin"], out[f"ground{g}_height"], out[f"ground{g}_support"] = org.astype(np.float32), hgt, support
        resid = p[:, 2] - hgt[np.clip(np.floor(p[:, 0] / args.cell).astype(int) - int(round(org[0] / args.cell)), 0, hgt.shape[0] - 1),
                               np.clip(np.floor(p[:, 1] / args.cell).astype(int) - int(round(org[1] / args.cell)), 0, hgt.shape[1] - 1)]
        meta_ground.append({"class": name, "points": int(len(p)), "cells_with_data": int(val.sum()), "support_cells": int(support.sum()),
                            "residual_abs_median": float(np.median(np.abs(resid))), "residual_abs_p90": float(np.percentile(np.abs(resid), 90))})
        print(f"[fit_planes] ground {name}: {len(p):,} points, {int(val.sum())} cells, |residual| median {meta_ground[-1]['residual_abs_median']:.3f}", flush=True)
    out["ground_classes"] = np.array(args.ground_classes)

    # vertical planes: sequential RANSAC with a near-horizontal normal, split into connected patches. With
    # --vertical_source mesh the candidates are the NKSR mesh vertices (class of the nearest fused point, vertex normal
    # within --max_tilt_deg of horizontal): the mesh averages the monocular depth noise that makes walls ~0.5 units
    # thick in the fused cloud (val056: planarity 0.33-0.45, i.e. no thin surface at all)
    if args.vertical_source == "mesh":
        from scipy.spatial import cKDTree

        mesh = io.read_mesh_ply(os.path.join(args.fused_dir, "mesh_nksr.ply"))
        mv, mf = mesh["vertices"].astype(np.float64), mesh["faces"].astype(np.int64)
        fn = np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]])
        vn = np.zeros_like(mv)
        for j in range(3):
            np.add.at(vn, mf[:, j], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        dist, nn = cKDTree(xyz).query(mv, k=1, workers=-1)
        vcls = np.where(dist < 0.15, cls[nn], -1)
        xyz, cls = mv, vcls  # the vertical fit below runs on the mesh vertices
        vert_ok = np.abs(vn[:, 2]) < np.sin(np.radians(args.max_tilt_deg))
    else:
        vert_ok = np.ones(len(xyz), bool)
    vmask = np.isin(cls, [CITYSCAPES.index(c) for c in args.vertical_classes]) & vert_ok
    rest = np.nonzero(vmask)[0]
    print(f"[fit_planes] vertical candidates: {int(vmask.sum()):,} of {len(xyz):,} {args.vertical_source} points "
          f"({int(vert_ok.sum()):,} with a near-horizontal normal)", flush=True)
    max_nz = np.sin(np.radians(args.max_tilt_deg))
    patches = []
    for _ in range(args.max_planes * 3):
        if len(rest) < args.min_inliers or len(patches) >= args.max_planes:
            break
        p = xyz[rest]
        best, best_n = None, 0
        for _ in range(400):
            a, b = p[rng.choice(len(p), 2, replace=False)]
            hdir = (b - a)[:2]
            if np.linalg.norm(hdir) < 0.2:
                continue
            n = np.array([-hdir[1], hdir[0], 0.0]) / np.linalg.norm(hdir)  # vertical plane through a and b
            inl = np.abs((p - a) @ n) < args.inlier
            if inl.sum() > best_n:
                best, best_n = (n, -float(a @ n)), int(inl.sum())
        if best is None or best_n < args.min_inliers:
            break
        n, d = best
        inl = np.abs(p @ n + d) < args.inlier
        # least-squares refit (normal kept within max_tilt of horizontal)
        q = p[inl]
        c = q.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((q - c).T))
        n2 = evecs[:, 0]
        if abs(n2[2]) <= max_nz:
            n, d = n2 / np.linalg.norm(n2), -float(c @ (n2 / np.linalg.norm(n2)))
            inl = np.abs(p @ n + d) < args.inlier
        u_axis = np.array([-n[1], n[0], 0.0]) / max(np.linalg.norm(n[:2]), 1e-9)  # horizontal in-plane axis
        q = p[inl]
        uv = np.stack([q @ u_axis, q[:, 2]], 1)
        lab = DBSCAN(eps=args.patch_eps, min_samples=5).fit_predict(uv)
        for l in sorted(set(lab) - {-1}):
            m = lab == l
            if m.sum() < args.min_inliers:
                continue
            # a real surface is thin: of the points near the patch (within 3 x inlier of the plane, inside its uv
            # footprint), most must be within the inlier distance; a blob of points gives many planes that are not
            uv_all = np.stack([p @ u_axis, p[:, 2]], 1)
            box = (uv_all[:, 0] >= uv[m, 0].min()) & (uv_all[:, 0] <= uv[m, 0].max()) & (uv_all[:, 1] >= uv[m, 1].min()) & (uv_all[:, 1] <= uv[m, 1].max())
            near = box & (np.abs(p @ n + d) < 3 * args.inlier)
            planarity = float(m.sum() / max(near.sum(), 1))
            if os.environ.get("FIT_PLANES_DEBUG"):
                print(f"[fit_planes] candidate {int(m.sum())} pts {np.ptp(uv[m, 0]):.1f}x{np.ptp(uv[m, 1]):.1f} planarity {planarity:.2f}", flush=True)
            if planarity < args.min_planarity or np.ptp(uv[m, 0]) < args.min_extent or np.ptp(uv[m, 1]) < args.min_height:
                continue
            ij = np.floor(uv[m] / args.cell).astype(np.int64)
            org = ij.min(0) - args.support_dilate - 1
            ij -= org
            occ = np.zeros(tuple(ij.max(0) + args.support_dilate + 2), bool)
            occ[ij[:, 0], ij[:, 1]] = True
            occ = ndimage.binary_dilation(ndimage.binary_closing(occ, np.ones((3, 3), bool)), np.ones((3, 3), bool), args.support_dilate)
            cnames = [CITYSCAPES[c] for c in cls[rest[inl][m]]]
            patches.append({"normal": n.astype(np.float32), "d": np.float32(d), "u_axis": u_axis.astype(np.float32),
                            "origin_uv": (org * args.cell).astype(np.float32), "support": occ,
                            "points": int(m.sum()), "class": max(set(cnames), key=cnames.count), "planarity": planarity,
                            "extent_u": float(np.ptp(uv[m, 0])), "extent_z": float(np.ptp(uv[m, 1]))})
        rest = rest[~inl]
    for k, pa in enumerate(patches):
        out[f"plane{k}_normal"], out[f"plane{k}_d"], out[f"plane{k}_u"] = pa["normal"], pa["d"], pa["u_axis"]
        out[f"plane{k}_origin_uv"], out[f"plane{k}_support"] = pa["origin_uv"], pa["support"]
    out["num_ground"], out["num_planes"] = np.int32(len(args.ground_classes)), np.int32(len(patches))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[fit_planes] {len(patches)} vertical patches: " + ", ".join(f"{p['class']} {p['points']} pts {p['extent_u']:.1f}x{p['extent_z']:.1f} pl {p['planarity']:.2f}"
                                                          for p in patches[:12]), flush=True)

    # top-down picture: points by class, ground support, plane patches as lines
    lo, hi = np.percentile(xyz[:, :2], 1, 0) - 2, np.percentile(xyz[:, :2], 99, 0) + 2
    S = 1600 / max(hi - lo)
    W, H = int((hi[0] - lo[0]) * S) + 1, int((hi[1] - lo[1]) * S) + 1
    img = np.full((H, W, 3), 25, np.uint8)
    pal = {0: (110, 110, 120), 1: (170, 140, 120), 2: (200, 80, 80), 3: (220, 160, 60), 4: (220, 220, 80), 8: (60, 160, 60), 9: (120, 150, 80), 13: (60, 90, 220)}
    u = ((xyz[:, 0] - lo[0]) * S).astype(int)
    v = ((hi[1] - xyz[:, 1]) * S).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for c, col in pal.items():
        m = ok & (cls == c)
        img[v[m], u[m]] = col
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for pa in patches:
        ij = np.argwhere(pa["support"])
        us = (ij[:, 0] + 0.5) * args.cell + pa["origin_uv"][0]
        a = np.array([us.min(), us.max()])
        pts = [(-pa["normal"][0] * pa["d"] + pa["u_axis"][0] * s, -pa["normal"][1] * pa["d"] + pa["u_axis"][1] * s) for s in a]
        dr.line([((x - lo[0]) * S, (hi[1] - y) * S) for x, y in pts], fill=(255, 255, 255), width=3)
    im.save(os.path.splitext(args.out)[0] + "_topdown.jpg", quality=88)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump({"scene_id": args.scene_id, "params": vars(args), "point_classes": counts, "ground": meta_ground,
                   "planes": [{k: v for k, v in p.items() if k not in ("support", "normal", "u_axis", "origin_uv", "d")} |
                              {"normal": p["normal"].tolist(), "d": float(p["d"])} for p in patches],
                   "uses_oracle": False, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=1)
    print(f"[fit_planes] -> {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
