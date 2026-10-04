"""E43 step 1 (docs/EXPERIMENTS.md): parked vehicles of a scene as 3D instances with their best image crops, without GT.

Detections are the per-instance Grounded-SAM-2 masks (--inst_dir, scripts/run_instance_masks.py) of every training
frame. A detection is static when at least half of it lies in the pixels scripts/refine_dynamic_masks.py released as
static (in the vehicle mask --mask_dir, not in --refined_dir). Its pixels, eroded by --erode_px against depth bleeding
at the silhouette, are back-projected with the estimated depth (--pose_dir, depth grid) and summarised by their median
point; the medians of all static detections are clustered with DBSCAN (--eps) and a cluster with detections in at least
--min_frames frames is one vehicle (a track). Its points are the union of its detections' points, voxelised (--voxel). World z is up (val056: mean road normal (0.01, 0.02, 1.00)). Per vehicle: the ground-plane principal axis
(length direction), extents (2nd-98th percentiles; the side facing away from the camera is never seen, so width is a
lower bound), the ground height (median z of the fused road points within --ground_ring of the footprint), and up to
--crops crops: the frames whose view of it is largest without touching the image border, picked greedily so that the
viewing directions differ by at least --min_view_sep degrees. A crop is the vehicle's mask component in that frame,
cut out on a white background, square and padded (TRELLIS input).

Output --out_dir: instances.json (per vehicle: id, points, frames, centre, axis, extents, ground z, crops with frame,
pixel box and viewing direction; detections as (frame, instance id)), points_<id>.npy (voxel centres),
crops/<id>_<frame>.png (RGBA), overview.jpg.
GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/vehicle_instances.py --scene_id val056 \\
        --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything \\
        --mask_dir data/dashrecon/val056/mask-gsam2_sky-segformer_img-glomap \\
        --refined_dir data/dashrecon/val056/mask-gsam2mv_sky-segformer_img-glomap \\
        --inst_dir data/dashrecon/val056/inst-gsam2_img-glomap \\
        --fused_ply data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomap/points_fused.ply \\
        --processed_root data/dashrecon/_undistorted/calib-glomap --out_dir results/E43/val056/instances \\
        --test_stride 10 --max_depth 40 --erode_px 6 --min_px 400 --voxel 0.05 --eps 0.35 --min_frames 5 \\
        --ground_ring 0.6 --crops 4 --min_view_sep 20 --crop_size 518
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.fusion.backproject import backproject  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene, train_frame_mask  # noqa: E402


def square_crop(img: np.ndarray, mask: np.ndarray, size: int, pad: float = 0.1) -> Image.Image:
    """The masked object on white, centred in a square with ``pad`` margin, resized to ``size``; RGBA."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    rgba = np.zeros((side, side, 4), np.uint8)
    rgba[..., :3] = 255
    sy0, sx0 = cy - side // 2, cx - side // 2
    ty0, tx0 = max(0, -sy0), max(0, -sx0)
    iy0, ix0 = max(0, sy0), max(0, sx0)
    iy1, ix1 = min(img.shape[0], sy0 + side), min(img.shape[1], sx0 + side)
    m = mask[iy0:iy1, ix0:ix1]
    tile = rgba[ty0:ty0 + iy1 - iy0, tx0:tx0 + ix1 - ix0]
    tile[m, :3] = img[iy0:iy1, ix0:ix1][m]
    tile[m, 3] = 255
    return Image.fromarray(rgba, "RGBA").resize((size, size), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--pose_dir", "--mask_dir", "--refined_dir", "--inst_dir", "--fused_ply", "--processed_root", "--out_dir"):
        parser.add_argument(a, required=True)
    parser.add_argument("--test_stride", type=int, required=True)
    parser.add_argument("--max_depth", type=float, required=True)
    parser.add_argument("--erode_px", type=int, required=True)
    parser.add_argument("--voxel", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--min_px", type=int, required=True, help="smallest detection (pixels) used")
    parser.add_argument("--min_frames", type=int, required=True)
    parser.add_argument("--ground_ring", type=float, required=True)
    parser.add_argument("--crops", type=int, required=True)
    parser.add_argument("--min_view_sep", type=float, required=True)
    parser.add_argument("--crop_size", type=int, required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)
    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.pose_dir)
    grid = io.read_meta(args.pose_dir)["depth_grid"]
    poses = io.read_poses(args.pose_dir)
    ks = io.depth_intrinsics(io.read_intrinsics(args.pose_dir), grid)
    train = train_frame_mask(frames, scene.start_timestep, args.test_stride)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")

    # static detections: back-projected points and their median
    dets, st, inst_maps = [], np.ones((3, 3), bool), {}
    for i in np.nonzero(train)[0]:
        t = int(frames[i])
        inst_map = np.asarray(Image.open(os.path.join(args.inst_dir, "inst", f"{t:06d}.png")))
        static = io.read_mask(args.mask_dir, "dynamic", t) & ~io.read_mask(args.refined_dir, "dynamic", t)
        d = io.read_depth(args.pose_dir, t)
        inst_maps[t] = inst_map
        for j in np.unique(inst_map):
            m = inst_map == j
            if j == 0 or m.sum() < args.min_px or (m & static).sum() < 0.5 * m.sum():
                continue
            small = io.mask_to_depth_grid(ndimage.binary_erosion(m, st, args.erode_px), grid)
            valid = small & (d > 0) & (d < args.max_depth)
            if valid.sum() < 10:
                continue
            p = backproject(d, ks[i], poses[i], valid)
            ys, xs = np.nonzero(m)
            dets.append({"frame": t, "inst": int(j), "i": int(i), "points": p, "median": np.median(p, 0), "area": int(m.sum()),
                         "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                         "border": bool(ys.min() <= 2 or xs.min() <= 2 or ys.max() >= m.shape[0] - 3 or xs.max() >= m.shape[1] - 3)})
    print(f"[vehicle_instances] {len(dets)} static vehicle detections ({time.time() - t0:.0f} s)", flush=True)
    lab = DBSCAN(eps=args.eps, min_samples=2).fit_predict(np.stack([x["median"] for x in dets]))

    fused = io.read_ply(args.fused_ply)
    road = fused["xyz"][fused["labels"] == 1]
    instances = []
    for c in sorted(set(lab) - {-1}):
        track = sorted((dets[k] for k in np.nonzero(lab == c)[0]), key=lambda x: x["frame"])
        fr_c = sorted({x["frame"] for x in track})
        if len(fr_c) < args.min_frames:
            continue
        pts = np.concatenate([x["points"] for x in track])
        vox = np.unique(np.floor(pts / args.voxel).astype(np.int64), axis=0) * args.voxel + args.voxel / 2
        xy = vox[:, :2]
        mu = xy.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((xy - mu).T))
        ax = evecs[:, np.argmax(evals)]
        perp = np.array([-ax[1], ax[0]])
        a, b = (xy - mu) @ ax, (xy - mu) @ perp
        lo_a, hi_a, lo_b, hi_b = np.percentile(a, 2), np.percentile(a, 98), np.percentile(b, 2), np.percentile(b, 98)
        near = road[(np.abs((road[:, :2] - mu) @ ax) < (hi_a - lo_a) / 2 + args.ground_ring)
                    & (np.abs((road[:, :2] - mu) @ perp) < (hi_b - lo_b) / 2 + args.ground_ring)]
        ground = float(np.median(near[:, 2])) if len(near) >= 20 else float(np.percentile(vox[:, 2], 2))
        centre = np.array([*(mu + ax * (lo_a + hi_a) / 2 + perp * (lo_b + hi_b) / 2), ground])
        inst = {"id": len(instances), "voxels": int(len(vox)), "frames": fr_c, "centre": centre.round(4).tolist(),
                "detections": [[x["frame"], x["inst"]] for x in track],
                "axis": [float(ax[0]), float(ax[1]), 0.0], "extent": [float(hi_a - lo_a), float(hi_b - lo_b), float(np.percentile(vox[:, 2], 98) - ground)],
                "ground_z": ground, "ground_points": int(len(near))}
        np.save(os.path.join(args.out_dir, f"points_{inst['id']}.npy"), vox.astype(np.float32))

        # crops: the largest detections not touching the image border, with differing viewing directions
        cands = []
        for x in track:
            view = centre - poses[x["i"]][:3, 3]
            view[2] = 0.0
            ang = float(np.degrees(np.arctan2(view @ np.array([*perp, 0.0]), view @ np.array([*ax, 0.0]))))
            cands.append({"frame": x["frame"], "inst": x["inst"], "area": x["area"], "border": x["border"], "box": x["box"], "view_deg": ang})
        cands.sort(key=lambda x: (x["border"], -x["area"]))
        chosen = []
        for cd in cands:
            if cd["border"] and any(not x["border"] for x in cands):
                continue
            if all(abs((cd["view_deg"] - x["view_deg"] + 180) % 360 - 180) >= args.min_view_sep for x in chosen):
                chosen.append(cd)
            if len(chosen) == args.crops:
                break
        for cd in chosen:
            img = np.asarray(Image.open(os.path.join(img_dir, f"{cd['frame']:03d}_{FRONT_CAM_ID}.jpg")).convert("RGB"))
            name = f"{inst['id']}_{cd['frame']:03d}.png"
            square_crop(img, inst_maps[cd["frame"]] == cd["inst"], args.crop_size).save(os.path.join(args.out_dir, "crops", name))
            cd["crop"] = f"crops/{name}"
        inst["crops"] = chosen
        instances.append(inst)
        print(f"[vehicle_instances] vehicle {inst['id']}: {len(vox)} voxels, {len(fr_c)} frames, extent "
              f"{inst['extent'][0]:.2f} x {inst['extent'][1]:.2f} x {inst['extent'][2]:.2f}, crops {[c['frame'] for c in chosen]}", flush=True)

    # overview: the first crop of every vehicle
    tiles = [Image.open(os.path.join(args.out_dir, inst["crops"][0]["crop"])).convert("RGB").resize((160, 160)) for inst in instances if inst["crops"]]
    if tiles:
        cols = 8
        sheet = Image.new("RGB", (160 * cols, 180 * ((len(tiles) + cols - 1) // cols)), (40, 40, 40))
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % cols) * 160, (k // cols) * 180))
            ImageDraw.Draw(sheet).text(((k % cols) * 160 + 4, (k // cols) * 180 + 162), str(instances[k]["id"]), fill=(255, 255, 0))
        sheet.save(os.path.join(args.out_dir, "overview.jpg"), quality=88)
    with open(os.path.join(args.out_dir, "instances.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, "params": vars(args), "instances": instances, "uses_oracle": False,
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=1)
    print(f"[vehicle_instances] {len(instances)} vehicles -> {args.out_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
