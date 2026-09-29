"""E17c (docs/EXPERIMENTS.md): each imagined surface point is taught by one generated view only.

E14 / E17a spawned Gaussians from every generated view that saw a place, each coloured by its own imagination, and
distilled all of them: the overlapping, differently coloured Gaussians render as fine noise (E17a). Borrowing view
selection from mesh texturing, every supervised pixel of every generated view gets a score
    confidence (validation status 4 accepted: 1, 1 unknown / 2 weak: --unknown_w) x |cos(view ray, surface normal)| / depth
on the shared imagined geometry (--depth_dirs: render_mesh_depth.py of E17a; normals from the depth map). A pixel is
owned by its view unless another generated view (any trajectory, frame within --frame_window) sees the same surface
point (its depth within --depth_tol) with a higher score; ties go to the earlier trajectory / view. train_fill.py
--owner_dirs then keeps only owned pixels for spawning and supervision, so each point is shaped by its best view.

Output: <out_root>/<trajectory>/owner/<k:03d>.png (255 = owned) and <out_root>/<trajectory>/owner.json (owned fraction
of the supervised pixels per view, parameters, commit).

Example (main venv):
    .venvs/main/bin/python scripts/best_view_ownership.py --views_dirs results/E14/val056/views/r0 results/E14/val056/views/r1 \
        --validation_dirs results/E14/val056/validation/r0 results/E14/val056/validation/r1 \
        --depth_dirs results/E17/val056/consolidated/mesh_depth/r0 results/E17/val056/consolidated/mesh_depth/r1 \
        --unknown_w 0.3 --frame_window 12 --depth_tol 0.05 --out_root results/E17/val056/bc/ownership
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
from dashrecon.gen.views import backproject  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def load_view(views_dir: str, validation_dir: str, depth_dir: str, cam: dict, k: int, unknown_w: float, device) -> dict:
    """Mesh depth, supervision weight and score map of generated view k (all on ``device``)."""
    d = torch.from_numpy(np.load(os.path.join(depth_dir, "mesh_depth", f"{k:03d}.npy")).astype(np.float32)).to(device)
    status = torch.from_numpy(np.asarray(Image.open(os.path.join(validation_dir, "status", f"{k:03d}.png")))).to(device)
    weight = torch.where(status == 4, 1.0, torch.where((status == 1) | (status == 2), unknown_w, 0.0))
    kk = torch.tensor(cam["K"], dtype=torch.float32, device=device)
    c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device=device)
    h, w = d.shape
    pts = backproject(d, kk, c2w).reshape(h, w, 3)
    dx = torch.zeros_like(pts)
    dy = torch.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1] = pts[2:] - pts[:-2]
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    ray = pts - c2w[:3, 3]
    ray = ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = (n * ray).sum(-1).abs()
    valid = (d > 0) & (weight > 0)
    score = torch.where(valid, weight * cos / d.clamp(min=1e-3), torch.zeros_like(d))
    return {"depth": d, "score": score, "valid": valid, "pts": pts, "K": kk, "w2c": torch.linalg.inv(c2w), "hw": (h, w)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dirs", nargs="+", required=True)
    parser.add_argument("--validation_dirs", nargs="+", required=True)
    parser.add_argument("--depth_dirs", nargs="+", required=True)
    parser.add_argument("--unknown_w", type=float, required=True)
    parser.add_argument("--frame_window", type=int, required=True)
    parser.add_argument("--depth_tol", type=float, required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()
    assert len(args.views_dirs) == len(args.validation_dirs) == len(args.depth_dirs)
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before computing experiment inputs"
    device = torch.device("cuda")
    t0 = time.time()
    views = []  # (trajectory index, k, frame, data)
    for t, (vd, vald, dd) in enumerate(zip(args.views_dirs, args.validation_dirs, args.depth_dirs)):
        cams = json.load(open(os.path.join(vd, "cams.json")))["cams"]
        for k, cam in enumerate(cams):
            views.append((t, k, cam["frame"], load_view(vd, vald, dd, cam, k, args.unknown_w, device)))
    print(f"[best_view_ownership] {len(views)} views loaded ({time.time() - t0:.0f} s)", flush=True)
    names = [os.path.basename(os.path.normpath(d)) for d in args.views_dirs]
    stats = {n: [] for n in names}
    for n in names:
        os.makedirs(os.path.join(args.out_root, n, "owner"), exist_ok=False)
    for i, (t, k, frame, v) in enumerate(views):
        owned = v["valid"].clone()
        x = v["pts"][owned]
        s_own = v["score"][owned]
        beaten = torch.zeros(len(x), dtype=torch.bool, device=device)
        for j, (t2, k2, frame2, u) in enumerate(views):
            if j == i or abs(frame2 - frame) > args.frame_window:
                continue
            pc = x @ u["w2c"][:3, :3].T + u["w2c"][:3, 3]
            z = pc[:, 2]
            h, w = u["hw"]
            px = u["K"][0, 0] * pc[:, 0] / z.clamp(min=1e-6) + u["K"][0, 2] - 0.5  # +0.5 pixel-centre convention
            py = u["K"][1, 1] * pc[:, 1] / z.clamp(min=1e-6) + u["K"][1, 2] - 0.5
            xi, yi = px.round().long(), py.round().long()
            inside = (z > 0) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xi, yi = xi.clamp(0, w - 1), yi.clamp(0, h - 1)
            dz = u["depth"][yi, xi]
            same = inside & (dz > 0) & ((dz - z).abs() < args.depth_tol * z)
            s_other = u["score"][yi, xi]
            earlier = (t2, k2) < (t, k)
            beaten |= same & ((s_other > s_own) | ((s_other == s_own) & earlier))
        mask = torch.zeros_like(v["valid"])
        mask[owned] = ~beaten
        Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.out_root, names[t], "owner", f"{k:03d}.png"))
        stats[names[t]].append({"k": k, "supervised": int(v["valid"].sum()), "owned": int(mask.sum())})
        if i % 50 == 0:
            print(f"[best_view_ownership] {i}/{len(views)}: {names[t]} view {k} owns {int(mask.sum())} of {int(v['valid'].sum())}", flush=True)
    for t, n in enumerate(names):
        rows = stats[n]
        frac = sum(r["owned"] for r in rows) / max(1, sum(r["supervised"] for r in rows))
        with open(os.path.join(args.out_root, n, "owner.json"), "w") as f:
            json.dump({"views_dir": args.views_dirs[t], "validation_dir": args.validation_dirs[t], "depth_dir": args.depth_dirs[t],
                       "params": vars(args), "owned_fraction_of_supervised": frac, "frames": rows,
                       "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
        print(f"[best_view_ownership] {n}: owns {frac:.3f} of its supervised pixels", flush=True)


if __name__ == "__main__":
    main()
