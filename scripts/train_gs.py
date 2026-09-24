"""Phase 6 entry point: train one experiment (E3 / E4 / E5, AGENTS.md section 7) on one scene with drivestudio.

    E3  init: raw back-projected estimated points (Phase 5 with every cleanup step skipped)
    E4  init: Phase 5 fused cloud (points_fused.ply)
    E5  init: flat Gaussians on the NKSR mesh + mesh depth / normal regularisation (DECISIONS D9)
All: MapAnything poses / intrinsics / depth, Grounded-SAM-2 dynamic masks, SegFormer sky masks, FRONT only.

Checks GT isolation on the merged config (dashrecon.train.guard) before training, then runs
tools/train.py:main. Output: <output_root>/<exp>/<scene_id>/ (drivestudio log dir: config.yaml, checkpoints,
videos/, metrics/) plus meta.json (commit, versions, runtime, peak VRAM, oracle flags) and metrics.json.

Example (main venv, from the repo root):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/train_gs.py --exp E4 --scene_id val056 --output_root results
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402

CONFIG = "configs/dashrecon/static_bg.yaml"
POSE_TAG = "pose-mapanything_depth-mapanything"
MASK_TAG = "mask-gsam2_sky-segformer"
FUSION_TAG = f"{POSE_TAG}__{MASK_TAG}"
E3_ROOT = "data/dashrecon/_e3_nocleanup"
EXPERIMENTS = ("E3", "E4", "E5")


def experiment_opts(exp: str, scene_id: str) -> list[str]:
    """CLI overrides (OmegaConf dot-list) for one experiment and scene."""
    scene = get_scene(scene_id)
    pose_dir = io.scene_dir("data/dashrecon", scene_id, POSE_TAG)
    mask_dir = io.scene_dir("data/dashrecon", scene_id, MASK_TAG)
    fusion_dir = io.scene_dir("data/dashrecon", scene_id, FUSION_TAG)
    opts = [
        f"data.scene_idx={scene.scene_idx}",
        f"data.pixel_source.pose_dir={pose_dir}",
        f"data.pixel_source.mask_dir={mask_dir}",
    ]
    if exp == "E3":
        init = ["source=ply", f"path={io.scene_dir(E3_ROOT, scene_id, FUSION_TAG)}/points_fused.ply"]
    elif exp == "E4":
        init = ["source=ply", f"path={fusion_dir}/points_fused.ply"]
    else:
        init = ["source=mesh", f"path={fusion_dir}/mesh_nksr.ply"]
        opts += [f"trainer.losses.mesh.path={fusion_dir}/mesh_nksr.ply", "trainer.losses.mesh.depth_w=0.1",
                 "trainer.losses.mesh.normal_w=0.05", "trainer.losses.mesh.min_alpha=0.5",
                 "trainer.losses.mesh.near=0.05", "trainer.losses.mesh.far=500.0"]
    opts += [f"model.Background.init.from_dashrecon.{kv}" for kv in init]
    return opts


def collect_metrics(log_dir: str, splits: list[str]) -> dict:
    """Image metrics written by drivestudio's do_evaluation, one file per rendered split."""
    out = {}
    for split in splits:
        files = sorted(glob.glob(os.path.join(log_dir, "metrics", f"images_{split}_*.json")))
        assert len(files) == 1, f"expected one images_{split} metrics file in {log_dir}/metrics, got {files}"
        with open(files[0]) as f:
            out[split] = json.load(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, choices=EXPERIMENTS)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="extra OmegaConf overrides (e.g. trainer.optim.num_iters=2000)")
    args = parser.parse_args()
    commit = git_commit()

    opts = experiment_opts(args.exp, args.scene_id) + list(args.opts)
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG), OmegaConf.from_cli(opts))
    assert_non_oracle(cfg)

    from tools.train import main as train_main

    train_args = argparse.Namespace(
        config_file=CONFIG, output_root=args.output_root, project=args.exp, run_name=args.scene_id,
        resume_from=None, render_video_postfix=None, enable_wandb=False, entity="none",
        enable_viewer=False, viewer_port=8080, opts=opts,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_main(train_args)
    runtime = time.time() - t0
    log_dir = os.path.join(args.output_root, args.exp, args.scene_id)
    splits = [s for s, on in (("test", cfg.render.render_test), ("full", cfg.render.render_full)) if on]
    metrics = collect_metrics(log_dir, splits)
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump({"exp": args.exp, "scene_id": args.scene_id, **metrics}, f, indent=2)
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump({
            "exp": args.exp, "scene_id": args.scene_id, "opts": opts,
            "dashrecon_commit": commit, "torch": torch.__version__,
            "runtime_s": runtime, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "uses_oracle": False,
            "oracle_note": "FRONT images + dashrecon products only; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[train_gs] {args.exp} {args.scene_id}: {runtime / 60:.1f} min -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
