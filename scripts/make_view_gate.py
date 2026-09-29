"""Observation-cone view gate of a trained FRONT-only run (D-A3 / E14 in docs/EXPERIMENTS.md).

For every Background Gaussian of --log_dir: its blending weight in every FRONT training view (only over pixels the
photometric loss constrained), the observation cone of those views (dashrecon.gen.floaters.observation_cone, views
with >= --cone_min_weight pixels of weight) and its distance to the nearest vertex of --mesh_ply (the run's own NKSR
mesh). Gaussians farther than --mesh_distance from the mesh are gated: in a novel view seen more than --margin
degrees outside their cone they fade out over --fade degrees (dashrecon.gen.floaters.cone_gate); unobserved ones
vanish from novel views. D-A3 chose mesh distance 0.3, margin 15, fade 15, min weight 0.5 on val056 E5c.

With --views_dirs / --validation_dirs (D-A5) the generated views a run was distilled with count as observers too, over
the hole pixels they supervised, and the gate covers every Gaussian of the run (spawned ones included): a Gaussian made
from side views is then gated where no real or generated view constrained it (E14's rear view showed spawned side
Gaussians from behind). The run may already have a gate (--replace_gate); observations are rendered without it.

Output: --out (a ViewGate, torch.save) and <--out without .pt>.json (parameters, counts, percentiles, commit).

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/make_view_gate.py --log_dir results/E5c/val056 \
        --mesh_ply data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap/mesh_nksr.ply \
        --mesh_distance 0.3 --margin 15 --fade 15 --cone_min_weight 0.5 --out results/E14/val056/view_gate.pt
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon import io  # noqa: E402
from dashrecon.gen.floaters import ViewGate, iterate_observations, observation_cone_streaming  # noqa: E402
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import DASHRECON_ROOT, assert_non_oracle  # noqa: E402
from dashrecon.train.seeds import seed_scene_optimization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--mesh_ply", required=True)
    parser.add_argument("--mesh_distance", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--fade", type=float, required=True)
    parser.add_argument("--cone_min_weight", type=float, required=True)
    parser.add_argument("--views_dirs", nargs="*", default=[], help="generated trajectories the run was distilled with")
    parser.add_argument("--validation_dirs", nargs="*", default=[], help="their validations (status maps)")
    parser.add_argument("--replace_gate", action="store_true", help="the run already has a gate; make a new one for it")
    parser.add_argument("--double_sided", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert len(args.views_dirs) == len(args.validation_dirs), "one validation dir per views dir"
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before making a view gate"
    assert args.mesh_ply.startswith(DASHRECON_ROOT), f"mesh must be a dashrecon product: {args.mesh_ply}"
    assert args.out.endswith(".pt") and not os.path.exists(args.out), args.out
    started = time.time()
    seed_scene_optimization(0)
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    assert ("view_gate" in cfg) == args.replace_gate, f"{args.log_dir}: view_gate in config must match --replace_gate"
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    assert set(trainer.gaussian_classes.keys()) == {"Background"}, trainer.gaussian_classes
    bg = trainer.models["Background"]
    observations = lambda: iterate_observations(trainer, dataset, device, list(zip(args.views_dirs, args.validation_dirs)))
    axis, half, observed, n_views = observation_cone_streaming(bg._means.detach(), observations, args.cone_min_weight)
    with torch.no_grad():
        mesh = io.read_mesh_ply(args.mesh_ply)
        dist = torch.from_numpy(cKDTree(mesh["vertices"]).query(bg._means.cpu().numpy(), k=1, workers=16)[0]).float().to(device)
        selected = dist > args.mesh_distance
    gate = ViewGate(axis=axis, half_angle=half, observed=observed, selected=selected, margin=args.margin, fade=args.fade,
                    double_sided=args.double_sided)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    q = torch.tensor([0.1, 0.5, 0.9], device=device)
    summary = {
        "log_dir": args.log_dir, "params": vars(args), "gaussians": gate.n, "observer_views": n_views,
        "observed": int(observed.sum()), "selected": int(selected.sum()),
        "selected_and_unobserved": int((selected & ~observed).sum()),
        "half_angle_deg_observed_p10_p50_p90": torch.quantile(half[observed], q).tolist(),
        "uses_oracle": False, "oracle_information": "none; FRONT images, masks, estimated cameras and mesh of the run",
        "dashrecon_commit": commit, "runtime_s": time.time() - started}
    gate.save(args.out, summary)
    with open(args.out[:-3] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[make_view_gate] {gate.n:,} Gaussians, {summary['selected']:,} gated (off mesh), "
          f"{summary['observed']:,} observed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
