"""E17a (docs/EXPERIMENTS.md): one point cloud of the observed scene and the generated geometry that passed validation.

The observed part is the run's Phase 5 fused cloud (--real_fusion_dir/points_fused.ply: xyz, oriented normals, rgb,
labels). The generated part comes from every --views_dirs trajectory (render_views.py + gen3c_fill.py +
depth_views.py): pixels its --validation_dirs entry accepted (status 4, validate_generated_views.py) with positive
depth, every --pixel_stride-th pixel, back-projected with the view's camera; normals from the cross product of the
point map's central differences, turned towards the camera; colour from the generated frame. Generated points are
thinned to one per --voxel cell (the first in view order). scripts/run_mesh.py then turns the result into one
"imagined" NKSR mesh, whose depth replaces the per-round depth of the generated views (scripts/render_mesh_depth.py,
train_fill.py --depth_dirs), so all rounds share one geometry.

Output in --out_dir (must not exist): points_fused.ply (labels: the real cloud's codes plus "generated") and
meta.json (label_names, counts per trajectory, parameters, sources, commit; generative: true).

Example (main venv):
    .venvs/main/bin/python scripts/fuse_generated.py \
        --real_fusion_dir data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap \
        --views_dirs results/E14/val056/views/r0 results/E14/val056/views/r1 \
        --validation_dirs results/E14/val056/validation/r0 results/E14/val056/validation/r1 \
        --pixel_stride 2 --voxel 0.05 --out_dir results/E17/val056/imagined_fusion
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.gen.views import backproject  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

ACCEPTED = 4  # validate_generated_views.py status code


def view_points(views_dir: str, validation_dir: str, cam: dict, k: int, stride: int) -> tuple:
    """(xyz, normals, rgb uint8) of the accepted, positive-depth pixels of generated view k."""
    depth = np.load(os.path.join(views_dir, "filled_depth", f"{k:03d}.npy")).astype(np.float32)
    status = np.asarray(Image.open(os.path.join(validation_dir, "status", f"{k:03d}.png")))
    rgb = np.asarray(Image.open(os.path.join(views_dir, "filled", f"{k:03d}.png")).convert("RGB"))
    h, w = depth.shape
    assert status.shape == (h, w) and rgb.shape[:2] == (h, w) and list(cam["hw"]) == [h, w], (views_dir, k)
    c2w = torch.tensor(cam["c2w"], dtype=torch.float32)
    pts = backproject(torch.from_numpy(depth), torch.tensor(cam["K"], dtype=torch.float32), c2w).reshape(h, w, 3)
    dx = torch.zeros_like(pts)
    dy = torch.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1] = pts[2:] - pts[:-2]
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    n = torch.where(((c2w[:3, 3] - pts) * n).sum(-1, keepdim=True) < 0, -n, n)
    sub = np.zeros((h, w), bool)
    sub[1:-1:stride, 1:-1:stride] = True  # central differences need both neighbours
    # a depth discontinuity makes the normal meaningless: keep pixels whose four neighbours are valid too
    valid = depth > 0
    inner = np.zeros_like(valid)
    inner[1:-1, 1:-1] = valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    sel = (status == ACCEPTED) & inner & sub
    sel_t = torch.from_numpy(sel)
    return pts[sel_t].numpy(), n[sel_t].numpy(), rgb[sel]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real_fusion_dir", required=True)
    parser.add_argument("--views_dirs", nargs="+", required=True)
    parser.add_argument("--validation_dirs", nargs="+", required=True)
    parser.add_argument("--pixel_stride", type=int, required=True)
    parser.add_argument("--voxel", type=float, required=True, help="scene units")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    assert len(args.views_dirs) == len(args.validation_dirs), "one validation dir per views dir"
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before fusing experiment inputs"
    os.makedirs(args.out_dir, exist_ok=False)
    t0 = time.time()
    real = io.read_ply(os.path.join(args.real_fusion_dir, "points_fused.ply"))
    real_meta = io.read_meta(args.real_fusion_dir)
    label_names = list(real_meta["label_names"]) + ["generated"]
    gen_code = len(label_names) - 1
    xyz, nrm, rgb, per_dir = [], [], [], {}
    for vd, vald in zip(args.views_dirs, args.validation_dirs):
        report = json.load(open(os.path.join(vald, "validation.json")))
        assert os.path.realpath(report["views_dir"]) == os.path.realpath(vd), f"{vald} does not validate {vd}"
        cams = json.load(open(os.path.join(vd, "cams.json")))["cams"]
        before = sum(len(p) for p in xyz)
        for k, cam in enumerate(cams):
            p, n, c = view_points(vd, vald, cam, k, args.pixel_stride)
            xyz.append(p)
            nrm.append(n)
            rgb.append(c)
        per_dir[vd] = int(sum(len(p) for p in xyz) - before)
        print(f"[fuse_generated] {vd}: {per_dir[vd]:,} accepted points", flush=True)
    xyz, nrm, rgb = np.concatenate(xyz), np.concatenate(nrm), np.concatenate(rgb)
    _, first = np.unique(np.floor(xyz / args.voxel).astype(np.int64), axis=0, return_index=True)
    first = np.sort(first)
    xyz, nrm, rgb = xyz[first], nrm[first], rgb[first]
    out_xyz = np.concatenate([real["xyz"], xyz]).astype(np.float32)
    out_nrm = np.concatenate([real["normals"], nrm]).astype(np.float32)
    out_rgb = np.concatenate([real["rgb"], rgb]).astype(np.uint8)
    out_lab = np.concatenate([real["labels"], np.full(len(xyz), gen_code, np.uint8)]).astype(np.uint8)
    io.write_ply(os.path.join(args.out_dir, "points_fused.ply"), out_xyz, out_rgb, normals=out_nrm, labels=out_lab)
    io.write_meta(args.out_dir, {
        "kind": "observed fused cloud + validated generated geometry (E17a)", "label_names": label_names,
        "counts": {"real": int(len(real["xyz"])), "generated_before_voxel": int(sum(per_dir.values())),
                   "generated": int(len(xyz)), "generated_per_views_dir": per_dir},
        "params": vars(args), "generative": True, "uses_oracle": False,
        "sources": {"real": args.real_fusion_dir, "views_dirs": args.views_dirs, "validation_dirs": args.validation_dirs},
        "units": "pose-run units (as the real fused cloud)", "runtime_s": time.time() - t0, "dashrecon_commit": commit})
    print(f"[fuse_generated] {len(real['xyz']):,} real + {len(xyz):,} generated points -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
