"""E17a (docs/EXPERIMENTS.md): depth of one mesh at every camera of generated trajectories (main venv, nvdiffrast).

For each --views_dirs trajectory (cams.json of render_views.py) writes <out_root>/<trajectory name>/mesh_depth/<k>.npy
(float16 z-depth at the view's resolution, 0 where the mesh is absent) and <out_root>/<name>/mesh_depth.json with, per
view, the fraction of hole pixels (mask/) the mesh covers and the median |log(mesh / per-round depth)| over pixels
the round's validation accepted, i.e. how far the consolidated geometry moves the accepted per-round depth.
train_fill.py --depth_dirs <out_root>/<name> then uses this depth for the generated views of that trajectory.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH .venvs/main/bin/python scripts/render_mesh_depth.py \
        --mesh_ply results/E17/val056/imagined_fusion/mesh_nksr.ply \
        --views_dirs results/E14/val056/views/r0 --validation_dirs results/E14/val056/validation/r0 \
        --near 0.05 --far 500 --out_root results/E17/val056/mesh_depth
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
from dashrecon.mesh.render import MeshDepthRenderer  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh_ply", required=True)
    parser.add_argument("--views_dirs", nargs="+", required=True)
    parser.add_argument("--validation_dirs", nargs="+", required=True)
    parser.add_argument("--near", type=float, required=True)
    parser.add_argument("--far", type=float, required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()
    assert len(args.views_dirs) == len(args.validation_dirs), "one validation dir per views dir"
    names = [os.path.basename(os.path.normpath(d)) for d in args.views_dirs]
    assert len(set(names)) == len(names), f"trajectory names must be unique: {names}"
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before rendering experiment inputs"
    device = torch.device("cuda")
    mesh = io.read_mesh_ply(args.mesh_ply)
    renderer = MeshDepthRenderer(mesh["vertices"], mesh["faces"], args.near, args.far, device)
    for vd, vald, name in zip(args.views_dirs, args.validation_dirs, names):
        out = os.path.join(args.out_root, name)
        os.makedirs(os.path.join(out, "mesh_depth"), exist_ok=False)
        cams = json.load(open(os.path.join(vd, "cams.json")))["cams"]
        t0, rows = time.time(), []
        for k, cam in enumerate(cams):
            h, w = cam["hw"]
            d = renderer.depth(torch.tensor(cam["K"], dtype=torch.float32, device=device),
                               torch.tensor(cam["c2w"], dtype=torch.float32, device=device), h, w).cpu().numpy()
            np.save(os.path.join(out, "mesh_depth", f"{k:03d}.npy"), d.astype(np.float16))
            hole = np.asarray(Image.open(os.path.join(vd, "mask", f"{k:03d}.png"))) > 127
            status = np.asarray(Image.open(os.path.join(vald, "status", f"{k:03d}.png")))
            own = np.load(os.path.join(vd, "filled_depth", f"{k:03d}.npy")).astype(np.float32)
            acc = (status == 4) & (d > 0) & (own > 0)
            rows.append({"k": k, "hole_covered": float(((d > 0) & hole).sum() / max(1, hole.sum())),
                         "accepted_pixels": int(acc.sum()),
                         "accepted_median_abs_log_ratio": float(np.median(np.abs(np.log(d[acc] / own[acc])))) if acc.any() else None})
        covered = float(np.mean([r["hole_covered"] for r in rows]))
        moved = [r["accepted_median_abs_log_ratio"] for r in rows if r["accepted_median_abs_log_ratio"] is not None]
        with open(os.path.join(out, "mesh_depth.json"), "w") as f:
            json.dump({"views_dir": vd, "validation_dir": vald, "mesh_ply": args.mesh_ply, "params": vars(args),
                       "mean_hole_covered": covered, "median_accepted_abs_log_ratio": float(np.median(moved)),
                       "frames": rows, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
        print(f"[render_mesh_depth] {name}: mesh covers {covered:.3f} of the holes; accepted per-round depth moves by "
              f"median |log| {np.median(moved):.3f}", flush=True)


if __name__ == "__main__":
    main()
