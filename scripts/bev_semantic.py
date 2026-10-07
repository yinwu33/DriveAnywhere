"""BEV semantic map of a reconstructed scene (user 10-07: "给我看一个 BEV 的语义图"), from the fused cloud and FRONT images
only (no GT).

Every fused point gets the majority Cityscapes class of the SegFormer predictions of the training frames that see it at
its estimated depth (scripts/fit_planes.py point_classes, same model and tolerance). On a --cell grid of world xy (z is
up for this pipeline):
    rgb       mean colour of the points within --ground_band of the local ground (orthophoto of the ground);
    ground    majority class of the same near-ground points (what lies on the ground: road, sidewalk, terrain, car
              bottoms, pole feet);
    full      top-down view: the class of the cell's points with the highest priority (car > pole / sign > building /
              wall / fence > vegetation > ground classes), objects at any height, tree crowns over the road included;
    drivable  the road support of --planes (fit_planes.py height field, class road) minus cells whose near-ground class
              is another ground class (the support is dilated into the kerb), minus cells with an obstacle (car,
              pole, sign, building, wall, fence, vegetation trunks: more than --obstacle_points points between
              --obstacle_low and --obstacle_high above the road surface) and minus the footprints of --parked
              (scripts/vehicle_instances.py); its boundary typed as kerb (the cell across has observed non-road ground)
              or open (nothing observed across: field-of-view edge, occlusion, range end); the ego path on top;
    state     the planner raster: 1 drivable, 2 non-drivable (observed and not drivable, obstacles and parked vehicles
              included), 0 unknown (nothing observed). Kerb = 1 next to 2, open edge = 1 next to 0.
The panels are rotated so that the ego path runs left to right (the grids in the npz are not rotated).
Local ground: the road height field where supported, else the sidewalk field, else the 5th height percentile of the
cell's points.

Output --out: one PNG with the four layers stacked (scale bar in scene units; a legend), and <out>.npz with the class
grids. Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/bev_semantic.py --scene_id val056 \\
        --fused_dir data/dashrecon/val056/pose-glomapf_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomapf \\
        --pose_dir data/dashrecon/val056/pose-glomapf_depth-mapanything --processed_root data/dashrecon/_undistorted/calib-glomapf \\
        --parked results/E56/val056/parked/instances.json --out results/E56/val056/bev/bev_semantic.png
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene, train_frame_mask  # noqa: E402
from fit_planes import CITYSCAPES, point_classes  # noqa: E402

PALETTE = {"road": (128, 64, 128), "sidewalk": (244, 35, 232), "building": (70, 70, 70), "wall": (102, 102, 156),
           "fence": (190, 153, 153), "pole": (153, 153, 153), "traffic light": (250, 170, 30), "traffic sign": (220, 220, 0),
           "vegetation": (107, 142, 35), "terrain": (152, 251, 152), "sky": (70, 130, 180), "person": (220, 20, 60),
           "rider": (255, 0, 0), "car": (0, 0, 142), "truck": (0, 0, 70), "bus": (0, 60, 100), "train": (0, 80, 100),
           "motorcycle": (0, 0, 230), "bicycle": (119, 11, 32)}
PRIORITY = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider", "pole", "traffic sign", "traffic light",
            "building", "wall", "fence", "vegetation", "terrain", "sidewalk", "road"]
OBSTACLES = ["car", "truck", "bus", "pole", "traffic sign", "traffic light", "building", "wall", "fence", "vegetation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--fused_dir", "--pose_dir", "--processed_root", "--parked", "--out"):
        parser.add_argument(a, required=True)
    parser.add_argument("--seg_model_id", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    parser.add_argument("--test_stride", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--depth_tol", type=float, default=0.05)
    parser.add_argument("--cell", type=float, default=0.25)
    parser.add_argument("--ground_band", type=float, default=0.25)
    parser.add_argument("--obstacle_low", type=float, default=0.3)
    parser.add_argument("--obstacle_high", type=float, default=1.2)
    parser.add_argument("--obstacle_points", type=int, default=3)
    parser.add_argument("--min_island", type=int, default=50, help="smallest drivable connected region kept (cells)")
    parser.add_argument("--px", type=int, default=4, help="pixels per cell")
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scene = get_scene(args.scene_id)
    fused = io.read_ply(os.path.join(args.fused_dir, "points_fused.ply"))
    xyz = fused["xyz"].astype(np.float64)
    rgb = fused["rgb"].astype(np.float64)
    rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
    frames = io.read_frames(args.pose_dir)
    poses = io.read_poses(args.pose_dir)
    grid = io.read_meta(args.pose_dir)["depth_grid"]
    k_depth = io.depth_intrinsics(io.read_intrinsics(args.pose_dir), grid)
    train = np.nonzero(train_frame_mask(frames, scene.start_timestep, args.test_stride))[0][::args.frame_stride]
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    cls, _ = point_classes(xyz, [(int(i), int(frames[i])) for i in train], poses, k_depth, grid, args.pose_dir, img_dir,
                           args.seg_model_id, args.depth_tol, torch.device("cuda"))
    print(f"[bev_semantic] {len(xyz):,} points classified ({time.time() - t0:.0f} s)", flush=True)

    planes = np.load(os.path.join(args.fused_dir, "planes.npz"))
    names = [str(x) for x in planes["ground_classes"]]
    assert abs(float(planes["cell"]) - args.cell) < 1e-6, "use the cell of planes.npz"

    # grid over all points, aligned with the height fields (integer cell indices of world xy / cell)
    ij = np.floor(xyz[:, :2] / args.cell).astype(np.int64)
    lo = ij.min(0)
    shape = tuple(ij.max(0) - lo + 1)
    ij -= lo
    flat = ij[:, 0] * shape[1] + ij[:, 1]

    def field(name):
        g = names.index(name)
        org = np.round(planes[f"ground{g}_origin"] / args.cell).astype(np.int64) - lo
        h, s = planes[f"ground{g}_height"], planes[f"ground{g}_support"]
        H = np.full(shape, np.nan)
        S = np.zeros(shape, bool)
        a0, b0 = max(org[0], 0), max(org[1], 0)
        a1, b1 = min(org[0] + h.shape[0], shape[0]), min(org[1] + h.shape[1], shape[1])
        H[a0:a1, b0:b1] = np.where(s[a0 - org[0]:a1 - org[0], b0 - org[1]:b1 - org[1]], h[a0 - org[0]:a1 - org[0], b0 - org[1]:b1 - org[1]], np.nan)
        S[a0:a1, b0:b1] = s[a0 - org[0]:a1 - org[0], b0 - org[1]:b1 - org[1]]
        return H, S

    road_h, road_s = field("road")
    side_h, _ = field("sidewalk") if "sidewalk" in names else (np.full(shape, np.nan), None)
    p5 = np.full(shape[0] * shape[1], np.nan)
    order = np.argsort(flat, kind="stable")
    uniq, start, cnt = np.unique(flat[order], return_index=True, return_counts=True)
    zs = xyz[order, 2]
    p5[uniq] = [np.percentile(zs[a:a + c], 5) for a, c in zip(start, cnt)]
    ground = np.where(np.isfinite(road_h), road_h, np.where(np.isfinite(side_h), side_h, p5.reshape(shape)))
    dz = xyz[:, 2] - ground.reshape(-1)[flat]

    n_cls = len(CITYSCAPES)
    near = np.abs(dz) <= args.ground_band
    cnt_ground = np.zeros((shape[0] * shape[1], n_cls), np.int32)
    ok = near & (cls >= 0)
    np.add.at(cnt_ground, (flat[ok], cls[ok]), 1)
    cnt_all = np.zeros_like(cnt_ground)
    ok_all = cls >= 0
    np.add.at(cnt_all, (flat[ok_all], cls[ok_all]), 1)
    ground_cls = np.where(cnt_ground.sum(1) > 0, cnt_ground.argmax(1), -1).reshape(shape)
    full_cls = np.full(shape[0] * shape[1], -1)
    for name in reversed(PRIORITY):  # lowest priority first, overwritten by higher ones
        c = CITYSCAPES.index(name)
        need = 2 if name in ("road", "sidewalk", "terrain") else 3
        full_cls[cnt_all[:, c] >= need] = c
    full_cls = full_cls.reshape(shape)
    col_sum = np.zeros((shape[0] * shape[1], 3))
    np.add.at(col_sum, flat[near], rgb[near])
    n_near = np.bincount(flat[near], minlength=shape[0] * shape[1])
    ortho = (col_sum / np.maximum(n_near, 1)[:, None]).reshape(shape + (3,))
    ortho[n_near.reshape(shape) == 0] = 0

    # drivable: road support minus obstacles and parked footprints; boundary typing
    obst = np.zeros(shape[0] * shape[1], np.int32)
    is_obst = np.isin(cls, [CITYSCAPES.index(c) for c in OBSTACLES]) & (dz > args.obstacle_low) & (dz < args.obstacle_high)
    np.add.at(obst, flat[is_obst], 1)
    obst = obst.reshape(shape) >= args.obstacle_points
    parked = np.zeros(shape, bool)
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cx, cy = (gi + lo[0] + 0.5) * args.cell, (gj + lo[1] + 0.5) * args.cell
    vehicles = json.load(open(args.parked))["instances"]
    for v in vehicles:
        ax, c = np.array(v["axis"][:2]), np.array(v["centre"][:2])
        a = (cx - c[0]) * ax[0] + (cy - c[1]) * ax[1]
        b = -(cx - c[0]) * ax[1] + (cy - c[1]) * ax[0]
        parked |= (np.abs(a) < v["extent"][0] / 2 + 0.15) & (np.abs(b) < max(v["extent"][1], 0.8) / 2 + 0.15)
    # the road support is dilated outwards (fit_planes --support_dilate): cells whose near-ground points are mostly
    # another class (sidewalk, terrain, ...) are not road
    other_ground = (ground_cls >= 0) & (ground_cls != CITYSCAPES.index("road"))
    drivable = ndimage.binary_closing(road_s & ~other_ground, iterations=1) & road_s & ~other_ground
    drivable &= ~ndimage.binary_dilation(obst | parked, iterations=1)
    lab, n_lab = ndimage.label(drivable)  # drop small islands (speckle inside sidewalks)
    sizes = ndimage.sum(drivable, lab, np.arange(1, n_lab + 1))
    drivable &= np.isin(lab, 1 + np.nonzero(sizes >= args.min_island)[0])
    observed = (n_near.reshape(shape) > 0) | (cnt_all.sum(1).reshape(shape) > 0)
    edge = drivable & ~ndimage.binary_erosion(drivable)
    across = ndimage.binary_dilation(~drivable & observed, iterations=2) & edge
    kerb, open_edge = across, edge & ~across

    # drawing: x (first grid axis) to the right, y up
    def to_img(a):
        return np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))[::-1]

    def colour(c):
        out = np.zeros(c.shape + (3,), np.uint8)
        for k, name in enumerate(CITYSCAPES):
            out[c == k] = PALETTE[name]
        return out

    dr_img = np.zeros(shape + (3,), np.uint8)
    dr_img[observed] = (40, 40, 40)
    dr_img[road_s] = (90, 60, 90)
    dr_img[drivable] = (128, 64, 128)
    dr_img[obst & road_s] = (0, 0, 142)
    dr_img[parked] = (0, 0, 200)
    dr_img[kerb] = (255, 255, 255)
    dr_img[open_edge] = (255, 140, 0)
    state = np.zeros(shape, np.uint8)
    state[observed | obst | parked] = 2
    state[drivable] = 1
    st_img = np.zeros(shape + (3,), np.uint8)
    st_img[state == 1] = (128, 64, 128)
    st_img[state == 2] = (230, 230, 230)
    layers = [("rgb: ground orthophoto (fused points within %.2f of the ground)" % args.ground_band, (np.clip(ortho, 0, 1) * 255).astype(np.uint8)),
              ("ground: class of near-ground points (SegFormer votes)", colour(ground_cls)),
              ("full: top-down class (objects at any height)", colour(full_cls)),
              ("drivable: road minus obstacles / parked cars; white = kerb, orange = open (unobserved) edge; yellow = ego path", dr_img),
              ("state (planner raster): purple = drivable, light grey = non-drivable, black = unknown", st_img)]
    ego = poses[:, :2, 3]
    heading = np.degrees(np.arctan2(*(ego[-1] - ego[0])[::-1]))
    tiles = []
    for title, arr in layers:
        im = Image.fromarray(to_img(arr)).resize((shape[0] * args.px, shape[1] * args.px), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        if title.startswith("drivable") or title.startswith("state"):
            pts = [(float((p[0, 3] / args.cell - lo[0]) * args.px), float((shape[1] - (p[1, 3] / args.cell - lo[1])) * args.px)) for p in poses]
            dr.line(pts, fill=(255, 230, 0), width=3)
        im = im.rotate(-heading, resample=Image.NEAREST, expand=True)  # ego path left to right
        bbox = Image.fromarray((np.asarray(im).sum(-1) > 0).astype(np.uint8) * 255).getbbox()
        tiles.append((title, im, bbox))
    x0, y0 = min(b[0] for _, _, b in tiles), min(b[1] for _, _, b in tiles)
    x1, y1 = max(b[2] for _, _, b in tiles), max(b[3] for _, _, b in tiles)
    out_tiles = []
    for title, im, _ in tiles:
        im = im.crop((max(x0 - 10, 0), max(y0 - 24, 0), min(x1 + 10, im.width), min(y1 + 24, im.height)))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 900, 18], fill=(0, 0, 0))
        dr.text((6, 3), title, fill=(255, 255, 255))
        bar = int(10 / args.cell * args.px)  # 10 scene units
        dr.rectangle([10, im.height - 14, 10 + bar, im.height - 8], fill=(255, 255, 255))
        dr.text((14 + bar, im.height - 18), "10 units", fill=(255, 255, 255))
        out_tiles.append(np.asarray(im.convert("RGB")))
    tiles = out_tiles
    legend = Image.new("RGB", (tiles[0].shape[1], 40), (20, 20, 20))
    dl = ImageDraw.Draw(legend)
    x = 6
    for name in PRIORITY:
        dl.rectangle([x, 8, x + 14, 22], fill=PALETTE[name])
        dl.text((x + 18, 10), name, fill=(255, 255, 255))
        x += 18 + 7 * len(name) + 14
    gap = np.full((6, tiles[0].shape[1], 3), 20, np.uint8)
    sheet = np.concatenate(sum([[t, gap] for t in tiles], []) + [np.asarray(legend)], 0)
    Image.fromarray(sheet).save(args.out)
    stats = {"cells": int(np.prod(shape)), "road_support": int(road_s.sum()), "drivable": int(drivable.sum()),
             "removed_obstacles": int((road_s & obst).sum()), "removed_parked": int((road_s & parked).sum()),
             "edge_kerb": int(kerb.sum()), "edge_open": int(open_edge.sum()),
             "state_cells": {"unknown": int((state == 0).sum()), "drivable": int((state == 1).sum()), "non_drivable": int((state == 2).sum())},
             "classes": {CITYSCAPES[c]: int((cls == c).sum()) for c in range(n_cls) if (cls == c).any()}}
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", origin_cell=lo, cell=args.cell, ground_cls=ground_cls, full_cls=full_cls,
                        drivable=drivable, kerb=kerb, open_edge=open_edge, road_support=road_s, state=state)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"params": vars(args), "stats": stats, "uses_oracle": False, "dashrecon_commit": commit, "runtime_s": time.time() - t0}, f, indent=1)
    print(f"[bev_semantic] {stats} -> {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
