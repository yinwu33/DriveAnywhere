"""E34 (docs/E34_DATA.md): train the FRONT-only E5f 3DGS of one training segment, for the "gs" (Fixer cross reference)
pairs.

The experiment is scripts/train_gs.py's E5f (configs/dashrecon/static_bg.yaml, Gaussians initialised on the NKSR mesh,
mesh depth / normal regularisation, MTGS-style flatten and max-scale regularisers; FRONT images only), with the E34
directories given explicitly instead of train_gs.py's fixed data/dashrecon layout:
    --front_root   undistorted, renumbered FRONT tree of scripts/e34_undistort.py (images only)
    --pose_dir     scripts/run_pose.py --camera_dir output; --mask_dir the warped FRONT masks; --fusion_dir with
                   points_fused.ply and mesh_nksr.ply (scripts/run_fusion.py, scripts/run_mesh.py)
The GT-isolation guard runs with --front_root as the only image root and --product_root as the only product root.
The end-of-training renders (test / full / novel videos) are switched off: only the checkpoint is needed. Extra
OmegaConf overrides (e.g. trainer.optim.num_iters=15000) are appended and recorded.

Output: <output_root>/E5f/<scene_id>/ (drivestudio log dir with checkpoint_final.pth) plus meta.json.

Example (main venv, from the repo root):
    PATH=$PWD/.venvs/main/bin:$HOME/cuda-12.6/bin:$PATH CUDA_HOME=$HOME/cuda-12.6 \
        .venvs/main/bin/python scripts/e34_train_gs.py --scene_id wt646 --front_root data/e34/front \
        --pose_dir data/e34/work/wt646/pose-glomap-cams_depth-mapanything \
        --mask_dir data/e34/work/wt646/mask-gsam2_sky-segformer_img-glomap-cams \
        --fusion_dir data/e34/work/wt646/pose-glomap-cams_depth-mapanything__mask-gsam2_sky-segformer_img-glomap-cams \
        --product_root data/e34/work --output_root data/e34/work/wt646/gs
"""
import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from train_gs import CONFIG, E5F_REG  # noqa: E402

EXP = "E5f"
NO_RENDERS = ["render.render_test=False", "render.render_full=False", "render.render_novel=null"]


def e5f_opts(scene_idx: int, front_root: str, pose_dir: str, mask_dir: str, fusion_dir: str) -> list[str]:
    """scripts/train_gs.py experiment_opts("E5f", ...) with explicit directories."""
    mesh = os.path.join(fusion_dir, "mesh_nksr.ply")
    return [f"data.data_root={front_root}", f"data.scene_idx={scene_idx}", f"data.pixel_source.pose_dir={pose_dir}",
            f"data.pixel_source.mask_dir={mask_dir}",
            f"trainer.losses.mesh.path={mesh}", "trainer.losses.mesh.depth_w=0.1", "trainer.losses.mesh.normal_w=0.05",
            "trainer.losses.mesh.min_alpha=0.5", "trainer.losses.mesh.near=0.05", "trainer.losses.mesh.far=500.0",
            "model.Background.init.from_dashrecon.source=mesh", f"model.Background.init.from_dashrecon.path={mesh}",
            *E5F_REG]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--front_root", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--product_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="extra OmegaConf overrides")
    args = parser.parse_args()
    commit = git_commit()
    scene = get_scene(args.scene_id)
    absd = lambda p: os.path.abspath(p)  # noqa: E731
    front_root, product_root = absd(args.front_root), absd(args.product_root) + os.sep
    opts = e5f_opts(scene.scene_idx, front_root, absd(args.pose_dir), absd(args.mask_dir), absd(args.fusion_dir))
    opts += NO_RENDERS + list(args.opts)
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG), OmegaConf.from_cli(opts))
    assert_non_oracle(cfg, image_roots=(front_root,), product_roots=(product_root,))

    from tools.train import main as train_main

    train_args = argparse.Namespace(
        config_file=CONFIG, output_root=args.output_root, project=EXP, run_name=args.scene_id,
        resume_from=None, render_video_postfix=None, enable_wandb=False, entity="none",
        enable_viewer=False, viewer_port=8080, opts=opts,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_main(train_args)
    runtime = time.time() - t0
    log_dir = os.path.join(args.output_root, EXP, args.scene_id)
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump({
            "exp": EXP, "scene_id": args.scene_id, "purpose": "E34 cross-reference pairs", "opts": opts,
            "num_iters": int(cfg.trainer.optim.num_iters), "dashrecon_commit": commit, "torch": torch.__version__,
            "runtime_s": runtime, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "uses_oracle": False,
            "oracle_note": "FRONT images + dashrecon products only; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[e34_train_gs] {args.scene_id}: {runtime / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB "
          f"-> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
