"""E39 (docs/EXPERIMENTS.md): a fused point cloud rendered as fixed surfels, as a drivestudio run, without any optimisation.

Every point of --ply (scripts/run_fusion.py output: xyz, rgb, normals) becomes one Gaussian with the real pixel
colour (SH degree 0 only), opacity --opacity, a disc of radius --radius_knn x (mean distance to its --knn nearest
neighbours, scipy cKDTree) in the tangent plane and --thickness x that along the normal. The run is written in the
format of a trained drivestudio run (checkpoint_final.pth, config.yaml, meta.json) by copying --template (a run of the
same scene, e.g. the sky-completed E5fs) and replacing its Background Gaussians: the Sky (completed), Affine and the
loader config stay; CamPose embeddings are zeroed (the cloud was built with the loader's own poses). The evaluation
scripts (eval_front_heldout, eval_view_range, eval_cross_camera, vis_freeview, view_gs) then run unchanged.

--sh_degree0 stores the colour as logit(rgb) with no higher SH bands (trainer.gaussian_ctrl_general_cfg.sh_degree 0 in the copied
config): 14 instead of 59 floats per point, so tens of millions of surfels fit on the GPU.

The question it answers (user, 10-04): is 3DGS needed for the observed content, or is a dense coloured point cloud
with real pixels better than the optimised Gaussians (the haze and floaters of yaw 20-45 come from the optimisation)?

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/surfel_run.py --ply data/dashrecon/_e39_dense/val056/<fusion tag>/points_fused.ply \\
        --template results/E5fs/val056 --out_log_dir results/E39/val056/dense --exp E39 --knn 4 --radius_knn 0.7
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

C0 = 0.28209479177387814  # SH band-0 constant (models/gaussians/basics.py RGB2SH)


def rotation_from_normals(n: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z; gsplat convention) whose local z axis is the normal: the third scale is the
    surfel thickness."""
    z = n / np.linalg.norm(n, axis=1, keepdims=True)
    a = np.where(np.abs(z[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    x = np.cross(a, z)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    r = np.stack([x, y, z], axis=2)  # columns = local axes
    t = np.trace(r, axis1=1, axis2=2)
    q = np.empty((len(r), 4))
    s = np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2.0
    q[:, 0] = 0.25 * s
    q[:, 1] = (r[:, 2, 1] - r[:, 1, 2]) / s
    q[:, 2] = (r[:, 0, 2] - r[:, 2, 0]) / s
    q[:, 3] = (r[:, 1, 0] - r[:, 0, 1]) / s
    bad = t + 1.0 < 1e-6  # 180-degree rotations: rebuild from the largest diagonal element
    for i in np.nonzero(bad)[0]:
        m = r[i]
        k = int(np.argmax(np.diag(m)))
        j, l = (k + 1) % 3, (k + 2) % 3
        sk = np.sqrt(max(m[k, k] - m[j, j] - m[l, l] + 1.0, 1e-12)) * 2.0
        qq = np.zeros(4)
        qq[k + 1] = 0.25 * sk
        qq[0] = (m[l, j] - m[j, l]) / sk
        qq[j + 1] = (m[j, k] + m[k, j]) / sk
        qq[l + 1] = (m[l, k] + m[k, l]) / sk
        q[i] = qq
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--template", required=True, help="run of the same scene whose Sky / Affine / config are reused")
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--radius_knn", type=float, default=0.7, help="disc radius = this x mean kNN distance")
    parser.add_argument("--thickness", type=float, default=0.1, help="normal-axis scale as a fraction of the radius")
    parser.add_argument("--opacity", type=float, default=0.99)
    parser.add_argument("--max_points", type=int, default=0, help="random subset (0 = all)")
    parser.add_argument("--sh_degree0", action="store_true", help="logit colours, no SH bands (memory for > 20M points)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_log_dir, exist_ok=False)
    commit = git_commit()
    t0 = time.time()
    ply = io.read_ply(args.ply)
    assert "normals" in ply, "surfels need normals (run_fusion.py --normal_knn)"
    xyz, rgb, nrm = ply["xyz"].astype(np.float64), ply["rgb"].astype(np.float32) / 255.0, ply["normals"].astype(np.float64)
    if args.max_points and len(xyz) > args.max_points:
        pick = np.random.default_rng(args.seed).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb, nrm = xyz[pick], rgb[pick], nrm[pick]
    n = len(xyz)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=args.knn + 1, workers=-1)
    radius = args.radius_knn * d[:, 1:].mean(1)
    radius = np.maximum(radius, 1e-4)
    print(f"[surfel_run] {n:,} points; kNN radius median {np.median(radius):.4f}, p90 {np.percentile(radius, 90):.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    state = torch.load(os.path.join(args.template, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    bg = state["models"]["Background"]
    assert bg["_features_rest"].shape[1:] == (15, 3), bg["_features_rest"].shape
    f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731
    bg["_means"] = f32(xyz)
    bg["_scales"] = f32(np.log(np.stack([radius, radius, radius * args.thickness], 1)))
    bg["_quats"] = f32(rotation_from_normals(nrm))
    if args.sh_degree0:
        bg["_features_dc"] = f32(np.log(np.clip(rgb, 1e-4, 1 - 1e-4) / (1 - np.clip(rgb, 1e-4, 1 - 1e-4))))
        bg["_features_rest"] = torch.zeros(n, 0, 3)
    else:
        bg["_features_dc"] = f32((rgb - 0.5) / C0)
        bg["_features_rest"] = torch.zeros(n, 15, 3)
    bg["_opacities"] = torch.full((n, 1), float(np.log(args.opacity / (1 - args.opacity))))
    state["models"]["CamPose"]["embeds.weight"].zero_()
    torch.save(state, os.path.join(args.out_log_dir, "checkpoint_final.pth"))
    cfg = OmegaConf.load(os.path.join(args.template, "config.yaml"))
    if args.sh_degree0:
        assert int(cfg.trainer.gaussian_ctrl_general_cfg.sh_degree) == 3, cfg.trainer.gaussian_ctrl_general_cfg.sh_degree
        cfg.trainer.gaussian_ctrl_general_cfg.sh_degree = 0
    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"exp": args.exp, "kind": "fixed surfels from a fused point cloud, no optimisation; Sky / Affine from the template",
                   "ply": os.path.abspath(args.ply), "template": os.path.abspath(args.template), "points": n, **vars(args),
                   "radius_median": float(np.median(radius)), "generative": False, "uses_oracle": False,
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[surfel_run] -> {args.out_log_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
