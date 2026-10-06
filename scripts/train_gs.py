"""Phase 6 entry point: train one experiment (E3 / E4 / E5, AGENTS.md section 7) on one scene with drivestudio.

    E3  init: raw back-projected estimated points (Phase 5 with every cleanup step skipped)
    E4  init: Phase 5 fused cloud (points_fused.ply)
    E5  init: flat Gaussians on the NKSR mesh + mesh depth / normal regularisation (DECISIONS D9)
All: MapAnything poses / intrinsics / depth, Grounded-SAM-2 dynamic masks, SegFormer sky masks, FRONT only.
    E4c, E5c  as E4, E5 on the self-calibrated camera (DECISIONS D13): undistorted FRONT images, shared
              GLOMAP intrinsics, GLOMAP poses scaled to MapAnything depth, masks of the undistorted images.
    E5f  E5c + drivestudio's flatten (shortest axis L1) and max_s_square (largest axis squared) Background
         regularisers, MTGS-style, against the large off-mesh Gaussians behind the side-view junk (DECISIONS W1,
         docs/EXPERIMENTS.md E5f); weights untuned.
    MT1  E5f on a combined multi-traversal scene (scripts/combine_traversals.py; docs/EXPERIMENTS.md MT1).
    MT1A MT1 with the first traversal only (data.end_timestep), initialised from a mesh of its frames only.
    E40  E5f with the depth of Depth Anything 3 at 1008 px for the GLOMAP poses (scripts/run_depth_da3.py,
         pose-glomap_depth-da3) instead of MapAnything's 518 px: fusion, NKSR mesh, depth supervision all from it
         (docs/EXPERIMENTS.md E39 / E40: depth accuracy, not point density, limits the observed surfaces).
    E41  E5f with the dynamic mask reduced to the moving objects (scripts/refine_dynamic_masks.py: per-region multi-view
         depth consistency, people and two-wheelers always masked): parked vehicles are fused, meshed and supervised
         like the rest of the static scene instead of being excluded everywhere (docs/EXPERIMENTS.md E41).
    E45  E41 with the planar proxies of scripts/fit_planes.py (road and sidewalk height fields; vertical patches where
         the reconstruction has thin walls) as a hard constraint (dashrecon.train.planar): Background Gaussians near a
         proxy are snapped onto it every 10 steps (docs/EXPERIMENTS.md E45).
    E46  E45 on a recalibration with the focal fixed to a single-image prior (scripts/run_focal_prior.py, MoGe v1:
         calib-glomapf): GLOMAP's own focal was 14 % short on val056, which stretches the scene sideways / vertically
         (D-F2, docs/EXPERIMENTS.md E46); everything downstream (masks, depth, fusion, mesh, planes) redone on it.
    E48  E46 with free-space carving (dashrecon.train.carve): Gaussians never on the NKSR mesh surface but in front of /
         behind it in the cached training views are removed during training (docs/EXPERIMENTS.md E48).
    E49  E48, but a Gaussian on the raw estimated depth (MapAnything, no confidence filter; within 20 %) in a view also
         counts as on the surface there, so trees and the far side of the street, which the mesh misses, are kept.
    E50  E49, but the selected Gaussians are slid along the nearest training ray onto the surface instead of removed.
    E52  E50, but only free-space evidence counts (in front of the nearest surface; behind it is occlusion) and Gaussians
         on a planar proxy are left alone.
    E53  E51 + the E52 carving (estimated-depth loss on every static non-sky pixel, front-only relocation).
    E54  E53 with behind-the-surface evidence counted again (as E50), planar-proxy Gaussians still exempt: is the
         FRONT cost of E49 / E50 the road (exempt here) or the behind rule?
    E51  E46 with the estimated-depth loss on every static non-sky pixel (no confidence percentile, no range cut):
         trees and the far side of the street, which the filter removed, get a depth target (D-F3).
    MT1app MT1 with per-traversal appearance (dashrecon.train.traversal: DC colour residual per traversal, one sky per
         traversal); novel views get the first traversal's appearance.
    MT1e / MT1x / MT1m  MT1 / MT1app / MTGS-style (per-traversal higher SH bands, one sky per traversal), all three with
         MTGS's per-image exposure (dashrecon.train.exposure) instead of drivestudio's Affine.

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
# per camera pipeline: (pose tag, mask tag, image root)
PIPELINES = {
    "mapanything": ("pose-mapanything_depth-mapanything", "mask-gsam2_sky-segformer", "data/waymo/processed/validation"),
    "glomap": ("pose-glomap_depth-mapanything", "mask-gsam2_sky-segformer_img-glomap", "data/dashrecon/_undistorted/calib-glomap"),
    "glomap-mt1": ("pose-glomap-mt1_depth-mapanything", "mask-gsam2_sky-segformer_img-glomap-mt1",
                   "data/dashrecon/_undistorted/calib-glomap-mt1"),
    "glomap-da3": ("pose-glomap_depth-da3", "mask-gsam2_sky-segformer_img-glomap", "data/dashrecon/_undistorted/calib-glomap"),
    "glomap-mv": ("pose-glomap_depth-mapanything", "mask-gsam2mv_sky-segformer_img-glomap", "data/dashrecon/_undistorted/calib-glomap"),
    "glomapf-mv": ("pose-glomapf_depth-mapanything", "mask-gsam2mv_sky-segformer_img-glomapf", "data/dashrecon/_undistorted/calib-glomapf"),
}
E3_ROOT = "data/dashrecon/_e3_nocleanup"
EXPERIMENTS = {"E3": "mapanything", "E4": "mapanything", "E5": "mapanything", "E4c": "glomap", "E5c": "glomap",
               "E5f": "glomap", "MT1": "glomap-mt1", "MT1A": "glomap-mt1", "MT1app": "glomap-mt1",
               "MT1e": "glomap-mt1", "MT1x": "glomap-mt1", "MT1m": "glomap-mt1", "E40": "glomap-da3", "E41": "glomap-mv", "E45": "glomap-mv", "E46": "glomapf-mv", "E48": "glomapf-mv", "E49": "glomapf-mv", "E50": "glomapf-mv", "E51": "glomapf-mv", "E52": "glomapf-mv", "E53": "glomapf-mv", "E54": "glomapf-mv"}
MT1_EXPS = ("MT1", "MT1A", "MT1app", "MT1e", "MT1x", "MT1m")
# per-traversal appearance of the MT1 runs: Gaussian colour residuals (dashrecon.train.traversal) and one sky per traversal
TRAVERSAL_RESIDUAL = {"MT1app": ["dc"], "MT1x": ["dc"], "MT1m": ["rest"]}
# MTGS's per-image exposure (dashrecon.train.exposure) instead of drivestudio's Affine, whose zero-initialised MLP never learns
EXPOSURE = ["model.Affine.type=dashrecon.train.exposure.PerImageExposure", "model.Affine.optim.all.lr=0.001",
            "model.Affine.optim.all.lr_final=0.0001", "model.Affine.optim.all.warmup_steps=5000",
            "model.Affine.optim.all.lr_pre_warmup=0.00001", "model.Affine.optim.all.weight_decay=0.0"]
# MT1A: the first traversal of the combined scene alone, with a mesh fused from its frames only (run_fusion --frame_range)
MT1A_ROOT = "data/dashrecon/_mt1a_only"
E5F_REG = ["model.Background.reg.flatten.w=1.0", "model.Background.reg.max_s_square_reg.w=0.05"]


def experiment_opts(exp: str, scene_id: str) -> list[str]:
    """CLI overrides (OmegaConf dot-list) for one experiment and scene."""
    scene = get_scene(scene_id)
    pose_tag, mask_tag, image_root = PIPELINES[EXPERIMENTS[exp]]
    fusion_tag = f"{pose_tag}__{mask_tag}"
    pose_dir = io.scene_dir("data/dashrecon", scene_id, pose_tag)
    mask_dir = io.scene_dir("data/dashrecon", scene_id, mask_tag)
    fusion_dir = io.scene_dir(MT1A_ROOT if exp == "MT1A" else "data/dashrecon", scene_id, fusion_tag)
    opts = [
        f"data.data_root={image_root}",
        f"data.scene_idx={scene.scene_idx}",
        f"data.pixel_source.pose_dir={pose_dir}",
        f"data.pixel_source.mask_dir={mask_dir}",
    ]
    if exp == "E3":
        init = ["source=ply", f"path={io.scene_dir(E3_ROOT, scene_id, fusion_tag)}/points_fused.ply"]
    elif exp in ("E4", "E4c"):
        init = ["source=ply", f"path={fusion_dir}/points_fused.ply"]
    else:
        init = ["source=mesh", f"path={fusion_dir}/mesh_nksr.ply"]
        opts += [f"trainer.losses.mesh.path={fusion_dir}/mesh_nksr.ply", "trainer.losses.mesh.depth_w=0.1",
                 "trainer.losses.mesh.normal_w=0.05", "trainer.losses.mesh.min_alpha=0.5",
                 "trainer.losses.mesh.near=0.05", "trainer.losses.mesh.far=500.0"]
    opts += [f"model.Background.init.from_dashrecon.{kv}" for kv in init]
    if exp in ("E5f", "E40", "E41", "E45", "E46", "E48", "E49", "E50", "E51", "E52", "E53", "E54") + MT1_EXPS:
        opts += E5F_REG
    if exp in ("E48", "E49", "E50", "E52", "E53", "E54"):
        opts += ["trainer.carve.every=500", "trainer.carve.start=1500", "trainer.carve.stop=28000", "trainer.carve.margin_rel=0.05",
                 "trainer.carve.margin_abs=0.1", "trainer.carve.min_views=3", "trainer.carve.max_views=90", "trainer.carve.down=4"]
    if exp in ("E49", "E50", "E52", "E53", "E54"):
        opts += ["trainer.carve.est_rel=0.2", "data.pixel_source.raw_depth=true"]
    if exp in ("E50", "E52", "E53", "E54"):
        opts += ["trainer.carve.relocate=true"]
    if exp in ("E52", "E53"):
        opts += ["trainer.carve.front_only=true"]
    if exp in ("E52", "E53", "E54"):
        opts += ["trainer.carve.exempt_planar=true"]
    if exp in ("E51", "E53", "E54"):
        opts += ["data.pixel_source.depth_conf_percentile=0", "data.pixel_source.depth_max=1000"]
    if exp in ("E45", "E46", "E48", "E49", "E50", "E51", "E52", "E53", "E54"):
        opts += [f"trainer.planar.path={fusion_dir}/planes.npz", "trainer.planar.every=10", "trainer.planar.start=500",
                 "trainer.planar.below=0.6", "trainer.planar.above=0.05", "trainer.planar.band=0.1", "trainer.planar.thickness=0.01"]
    if exp in MT1_EXPS:
        # drivestudio's front_center_interp video would interpolate across the jump between traversals, and its rays
        # for all frames are precomputed on the GPU (16 GB for the 395 frames of mt1)
        opts += ["render.render_novel=null"]
    if exp in TRAVERSAL_RESIDUAL:
        meta = io.read_meta(pose_dir)
        starts = [meta["traversals"][m]["virtual_frames"][0] for m in meta["members"]]
        opts += ["model.Background.type=dashrecon.train.traversal.TraversalGaussians",
                 f"model.Background.num_traversals={len(starts)}", "model.Background.reg.traversal_residual.w=0.01",
                 f"model.Background.residual=[{','.join(TRAVERSAL_RESIDUAL[exp])}]",
                 "model.Background.optim.sh_dc_tr.lr=0.0025", "model.Background.optim.sh_rest_tr.lr=0.000125",
                 "model.Sky.type=dashrecon.train.traversal.TraversalEnvLight", f"model.Sky.params.num_traversals={len(starts)}",
                 f"trainer.traversal.starts=[{','.join(str(x) for x in starts)}]", "trainer.traversal.novel=0"]
    if exp in ("MT1e", "MT1x", "MT1m"):
        opts += EXPOSURE
    if exp == "MT1A":
        first = io.read_meta(pose_dir)["members"][0]
        opts += [f"data.end_timestep={io.read_meta(pose_dir)['traversals'][first]['virtual_frames'][1]}"]
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
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
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
