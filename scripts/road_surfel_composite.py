"""E42 (docs/EXPERIMENTS.md): the road of a trained run replaced by fixed surfels of the fused point cloud, no training.

User, 10-04: in E39 (the fused cloud as fixed surfels) the lane markings stay sharp from every position and angle,
while in E5fs / E35 they break into fragments as soon as the camera moves sideways. The optimised road Gaussians only
have to look right from the FRONT trajectory and sit at slightly wrong heights and thicknesses; the surfels lie on the
fused road surface (normals from the cloud) and carry the real pixel colours, so the markings are a texture on the
right plane.

Road Gaussians of --log_dir: within --max_plane_dist (scene units, in the road plane) of a road point of --ply
(labels == 1, the SegFormer road of the fusion) and between --below and --above along that point's normal (the normal
is oriented towards the cameras, i.e. up). They are removed, and every road point becomes a surfel exactly as
scripts/surfel_run.py builds them (disc radius --radius_knn x mean distance to its --knn nearest road neighbours,
thickness --thickness x radius, opacity --opacity, colour as SH band 0, higher bands zero). Everything else of the run
(other Gaussians, Sky, Affine, CamPose, its view gate restricted to the kept Gaussians) is unchanged. Gaussians added
after the surfels are never gated (they are appended after the gated ones).

With --feather > 0 the surfels fade out over the last --feather scene units before the edge of their coverage (ground
grid of twice the surfel spacing, gaps closed, distance to the nearest uncovered cell), and the run's road Gaussians are only removed
where the surfels are fully opaque, so the edge is a blend, not a seam (E42c, camera raised 2 m).

Output: --out_log_dir with checkpoint_final.pth, config.yaml (view_gate -> the restricted gate), view_gate.pt,
meta.json (counts). The evaluation scripts run on it unchanged.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/road_surfel_composite.py --log_dir results/E35/val056/model \\
        --ply data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap/points_fused.ply \\
        --out_log_dir results/E42/val056 --exp E42 --max_plane_dist 0.3 --below 0.5 --above 0.08 --knn 4 --radius_knn 0.7
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon import io  # noqa: E402
from dashrecon.gen.floaters import ViewGate  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from surfel_run import C0, rotation_from_normals  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--max_plane_dist", type=float, required=True)
    parser.add_argument("--below", type=float, required=True)
    parser.add_argument("--above", type=float, required=True)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--radius_knn", type=float, default=0.7)
    parser.add_argument("--thickness", type=float, default=0.1)
    parser.add_argument("--opacity", type=float, default=0.99)
    parser.add_argument("--feather", type=float, default=0.0, help="fade-out width at the coverage edge (scene units)")
    args = parser.parse_args()
    os.makedirs(args.out_log_dir, exist_ok=False)
    commit = git_commit()
    t0 = time.time()

    ply = io.read_ply(args.ply)
    road = ply["labels"] == 1
    xyz, rgb, nrm = ply["xyz"][road].astype(np.float64), ply["rgb"][road].astype(np.float32) / 255.0, ply["normals"][road].astype(np.float64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    tree = cKDTree(xyz)

    state = torch.load(os.path.join(args.log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    bg = state["models"]["Background"]
    means = bg["_means"].double().numpy()
    _, nn = tree.query(means, k=1, workers=-1)
    d = means - xyz[nn]
    along = (d * nrm[nn]).sum(1)
    in_plane = np.linalg.norm(d - along[:, None] * nrm[nn], axis=1)
    is_road = (in_plane < args.max_plane_dist) & (along > -args.below) & (along < args.above)
    edge_dist = np.full(len(xyz), np.inf)
    if args.feather > 0:
        from scipy.ndimage import binary_closing, distance_transform_edt

        # coverage on a grid of twice the surfel spacing, small gaps closed, then the distance to uncovered cells
        cell = 2 * float(np.median(tree.query(xyz[:20000], k=2)[0][:, 1]))
        ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
        ij -= ij.min(0) - 2
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0], ij[:, 1]] = True
        occ = binary_closing(occ, np.ones((3, 3), bool))
        edge_dist = distance_transform_edt(occ)[ij[:, 0], ij[:, 1]] * cell
        is_road &= edge_dist[nn] >= args.feather  # keep the run's own road where the surfels fade
    keep = np.nonzero(~is_road)[0]
    print(f"[road_surfel_composite] {len(means):,} Gaussians, {int(is_road.sum()):,} on the road removed; {len(xyz):,} road surfels "
          f"({time.time() - t0:.0f} s)", flush=True)

    dk, _ = tree.query(xyz, k=args.knn + 1, workers=-1)
    radius = np.maximum(args.radius_knn * dk[:, 1:].mean(1), 1e-4)
    n_s = len(xyz)
    f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731
    kt = torch.from_numpy(keep)
    rest_shape = bg["_features_rest"].shape[1:]
    surf = {"_means": f32(xyz), "_scales": f32(np.log(np.stack([radius, radius, radius * args.thickness], 1))),
            "_quats": f32(rotation_from_normals(nrm)), "_features_dc": f32((rgb - 0.5) / C0),
            "_features_rest": torch.zeros((n_s,) + tuple(rest_shape)),
            "_opacities": f32(np.log(np.clip(args.opacity * np.clip(edge_dist / max(args.feather, 1e-9), 0, 1), 1e-4, args.opacity)
                                     / (1 - np.clip(args.opacity * np.clip(edge_dist / max(args.feather, 1e-9), 0, 1), 1e-4, args.opacity)))[:, None])}
    assert set(bg.keys()) == set(surf.keys()), bg.keys()
    for k in surf:
        bg[k] = torch.cat([bg[k][kt], surf[k]], 0)
    torch.save(state, os.path.join(args.out_log_dir, "checkpoint_final.pth"))

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    gate_info = None
    if "view_gate" in cfg:
        gate = ViewGate.load(cfg.view_gate, torch.device("cpu"))
        gk = kt[kt < gate.n]  # kept indices are sorted, so the gated ones stay first
        new = ViewGate(axis=gate.axis[gk], half_angle=gate.half_angle[gk], observed=gate.observed[gk], selected=gate.selected[gk],
                       margin=gate.margin, fade=gate.fade, double_sided=gate.double_sided)
        gate_path = os.path.join(args.out_log_dir, "view_gate.pt")
        new.save(gate_path, {"source": cfg.view_gate, "restricted_to": "Gaussians kept by road_surfel_composite.py"})
        cfg.view_gate = gate_path
        gate_info = {"source": str(gate_path), "gated_before": gate.n, "gated_after": new.n}
    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"exp": args.exp, "kind": "road Gaussians replaced by fixed road surfels of the fused cloud, no training",
                   "source_run": os.path.abspath(args.log_dir), "ply": os.path.abspath(args.ply), **vars(args),
                   "gaussians_before": int(len(means)), "road_removed": int(is_road.sum()), "road_surfels": int(n_s),
                   "radius_median": float(np.median(radius)), "gate": gate_info, "generative": bool(json.load(open(os.path.join(args.log_dir, "meta.json")))["generative"]),
                   "uses_oracle": False, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[road_surfel_composite] -> {args.out_log_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
