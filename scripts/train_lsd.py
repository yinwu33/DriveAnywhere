"""E9: LSD-3D-style appearance generation on the reconstructed geometry (user decision 2026-09-25; DECISIONS O).

Reproduces the second half of LSD-3D (Ost et al., arXiv 2508.19204) on our own scene instead of a generated
layout: Gaussians already on the NKSR mesh (the E5c run, --init_exp) are optimised with Geometry-Grounded
Distillation Sampling towards the frozen SDXL + depth ControlNet (dashrecon.gen.ggds.SDXLRefiner: DDIM inversion to
t, 5 guided DDIM steps, mesh disparity as control), trading fidelity to the FRONT frames for realism:
    - no real-frame loss unless --real_w > 0 (LSD-3D has none; its scenes are generated);
    - free viewpoints: every --pool_stride-th training frame, from its refined pose, moved by right ~ U(-right_max,
      right_max), up ~ U(0, up_max), yaw ~ U(-yaw_max, yaw_max), pitch ~ U(-pitch_max, pitch_max / 2) (keeps the
      horizon in view; dashrecon.gen.views conventions), or not moved with probability --p_orig; new moves and
      targets for the whole pool every --refresh_every steps (LSD-3D regenerates every step: ~3 s per target here);
    - noise level t ~ U(t_lo(r), --t_max) with t_lo annealed linearly from --t_lo_start to --t_lo_end ("annealed by
      dropping the lower sampling bound"); loss weight w(t) = sqrt(alpha_bar) as in E6 (the paper's omega(t) is
      not given);
    - losses on the novel view: gen_w * w(t) * (L1 + lpips_w * LPIPS-VGG) over kept pixels (mesh or opacity >
      --keep_alpha; sky and empty background keep the detached render, E6), the E5 mesh depth / normal losses, and
      tv_w * total variation of the kept render (LSD-3D's TV loss; weight not given in the paper);
    - densification back on until --densify_steps after the start (gradients of the novel views), culling as usual.
Not reproduced: the Waymo fine-tuning of SDXL (AGENTS §2, user choice (a)), 2DGS (3DGS kept), SGLD perturbations and
the generated environment map (the EnvLight sky of the run is kept, frozen). The deferred rendering of LSD-3D is
scripts/vis_freeview.py --deferred_t.

Novel views render with novel_view=True and rays recomputed for the moved camera (dashrecon.gen.views.render_at);
Affine and Sky are frozen. Opacity reset is off; every learning rate warms up over --warmup_steps (E6).

Outputs in <output_root>/E9/<scene_id>/: config.yaml, checkpoint_final.pth, metrics/, videos/, metrics.json,
meta.json (generative: true), lsd_params.json and gen/round<r>_view<k>.jpg (render | control | target) for the
first --save_views views of rounds 0, the middle one and the last one.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python scripts/train_lsd.py --scene_id val056 --output_root results
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.ggds import SDXLRefiner, disparity_image  # noqa: E402
from dashrecon.gen.novel import build_trainer, mesh_losses, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, moved_c2w, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from train_ggds import NEGATIVE, PROMPT, RENDER_KEYS, save_example  # noqa: E402
from train_gs import collect_metrics  # noqa: E402

EXP = "E9"


def draw_move(args, rng: np.random.Generator) -> ViewMove:
    if rng.uniform() < args.p_orig:
        return ViewMove()
    return ViewMove(right=float(rng.uniform(-args.right_max, args.right_max)), up=float(rng.uniform(0.0, args.up_max)),
                    yaw=float(rng.uniform(-args.yaw_max, args.yaw_max)), pitch=float(rng.uniform(-args.pitch_max, args.pitch_max / 2)))


def total_variation(rgb: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between horizontal and vertical neighbours, both inside ``keep`` [H, W, 1]."""
    dx = (rgb[:, 1:] - rgb[:, :-1]).abs() * keep[:, 1:] * keep[:, :-1]
    dy = (rgb[1:] - rgb[:-1]).abs() * keep[1:] * keep[:-1]
    return (dx.sum() + dy.sum()) / (3.0 * keep.sum()).clamp(min=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--init_exp", default="E5c")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--refresh_every", type=int, default=200)
    parser.add_argument("--pool_stride", type=int, default=4)
    parser.add_argument("--p_orig", type=float, default=0.2)
    parser.add_argument("--right_max", type=float, default=2.5)
    parser.add_argument("--up_max", type=float, default=1.5)
    parser.add_argument("--yaw_max", type=float, default=30.0)
    parser.add_argument("--pitch_max", type=float, default=10.0)
    parser.add_argument("--t_max", type=float, default=0.85)
    parser.add_argument("--t_lo_start", type=float, default=0.7)
    parser.add_argument("--t_lo_end", type=float, default=0.3)
    parser.add_argument("--real_w", type=float, default=0.0)
    parser.add_argument("--gen_w", type=float, default=1.0)
    parser.add_argument("--lpips_w", type=float, default=0.5)
    parser.add_argument("--tv_w", type=float, default=0.01)
    parser.add_argument("--keep_alpha", type=float, default=0.5)
    parser.add_argument("--densify_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--gen_hw", type=int, nargs=2, default=[832, 1248])
    parser.add_argument("--disp_pct", type=float, default=90.0)
    parser.add_argument("--save_views", type=int, default=4)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    src_dir = os.path.join(args.output_root, args.init_exp, args.scene_id)
    log_dir = os.path.join(args.output_root, EXP, args.scene_id)
    cfg = OmegaConf.load(os.path.join(src_dir, "config.yaml"))
    assert_non_oracle(cfg)
    assert "mesh" in cfg.trainer.losses, f"{src_dir} was trained without the mesh losses"
    init_ckpt = os.path.join(src_dir, "checkpoint_final.pth")
    start = int(torch.load(init_ckpt, map_location="cpu", weights_only=False)["step"])
    cfg.trainer.optim.num_iters = start + args.steps
    cfg.trainer.gaussian_ctrl_general_cfg.reset_alpha_interval = 10**9
    cfg.trainer.gaussian_ctrl_general_cfg.stop_split_at = start + args.densify_steps
    cfg.log_dir = log_dir
    for sub in ("metrics", "videos", "gen"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    params = {**vars(args), "prompt": PROMPT, "negative_prompt": NEGATIVE, "init_ckpt": init_ckpt, "start_step": start,
              "dashrecon_commit": commit}
    with open(os.path.join(log_dir, "lsd_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    from datasets.driving_dataset import DrivingDataset
    from tools.eval import do_evaluation
    import lpips

    torch.cuda.reset_peak_memory_stats()
    t_begin = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=init_ckpt, load_only_model=True)
    assert trainer.step == start, (trainer.step, start)
    trainer.initialize_optimizer()
    base_lr = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    refiner = SDXLRefiner(device, PROMPT, NEGATIVE, args.guidance_scale, args.controlnet_scale, args.num_steps, tuple(args.gen_hw))
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    frozen_params = list(trainer.models["Affine"].parameters()) + list(trainer.models["Sky"].parameters())
    frames = []
    for idx in range(0, len(dataset.train_image_set), args.pool_stride):
        ii, ci = dataset.train_image_set.get_image(idx, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        frames.append({"ii": ii, "ci": ci, "frame": int(ii["img_idx"].flatten()[0]), "c2w": refined_c2w(trainer, ii, ci),
                       "hw": (int(ci["height"]), int(ci["width"]))})
    n_rounds = -(-args.steps // args.refresh_every)
    example_rounds = sorted({0, n_rounds // 2, n_rounds - 1})
    print(f"[train_lsd] {args.scene_id}: pool {len(frames)} views, {n_rounds} rounds, steps {start + 1}..{start + args.steps}, "
          f"real_w {args.real_w}", flush=True)

    gen_s, rounds, pool, running, t_print = 0.0, [], [], {}, time.time()
    for step in range(start + 1, start + args.steps + 1):
        r = (step - start - 1) // args.refresh_every
        if (step - start - 1) % args.refresh_every == 0:
            t0 = time.time()
            t_lo = args.t_lo_start + (args.t_lo_end - args.t_lo_start) * (r / max(n_rounds - 1, 1))
            pool = []
            trainer.set_eval()
            with torch.no_grad():
                for k, fr in enumerate(frames):
                    move = draw_move(args, rng)
                    ii, ci = dict(fr["ii"]), dict(fr["ci"])
                    c2w = moved_c2w(fr["c2w"], move)
                    out = render_at(trainer, ii, ci, c2w, ci["intrinsics"], fr["hw"])
                    maps = trainer._mesh_maps(trainer._last_cam)
                    control = disparity_image(maps["mesh_depth"], maps["mesh_valid"], args.disp_pct)
                    t = float(rng.uniform(t_lo, args.t_max))
                    target = refiner.refine(out["rgb"].clamp(0, 1), control, t)
                    pool.append({"ii": ii, "ci": ci, "c2w": c2w, "hw": fr["hw"], "frame": fr["frame"], "move": vars(move), "t": t,
                                 "w": refiner.signal_weight(t), "target": target.half()})
                    if r in example_rounds and k < args.save_views:
                        save_example(os.path.join(log_dir, "gen", f"round{r}_view{k}.jpg"), out["rgb"], control, target)
            gen_s += time.time() - t0
            rounds.append({"round": r, "step": step, "t_range": [t_lo, args.t_max], "seconds": time.time() - t0,
                           "pool": [{"frame": v["frame"], "move": v["move"], "t": v["t"]} for v in pool]})
            print(f"[train_lsd] round {r} at step {step}: t in [{t_lo:.2f}, {args.t_max}], {time.time() - t0:.0f} s", flush=True)
            t_print += time.time() - t0

        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        loss_dict = {}
        if args.real_w > 0:
            ii, ci = dataset.train_image_set.next(trainer._get_downscale_factor())
            ii, ci = to_device(ii, device), to_device(ci, device)
            outputs = trainer(ii, ci)
            trainer.update_visibility_filter()
            loss_dict = {k: args.real_w * val for k, val in trainer.compute_losses(outputs=outputs, image_infos=ii, cam_infos=ci).items()}

        v = pool[int(rng.integers(len(pool)))]
        for p in frozen_params:
            p.requires_grad_(False)
        nv = render_at(trainer, dict(v["ii"]), dict(v["ci"]), v["c2w"], v["ci"]["intrinsics"], v["hw"])
        for p in frozen_params:
            p.requires_grad_(True)
        trainer.update_visibility_filter()  # postprocess_per_train_step reads the last render's radii and gradients
        rgb = nv["rgb"]
        with torch.no_grad():
            keep = (nv["mesh_valid"] | (nv["opacity"][..., 0] > args.keep_alpha)).float()[..., None]
        target = keep * v["target"].float() + (1.0 - keep) * rgb.detach()
        l1 = (rgb - target).abs().sum() / (3.0 * keep.sum()).clamp(min=1.0)
        lp = lpips_vgg(rgb.permute(2, 0, 1)[None] * 2 - 1, target.permute(2, 0, 1)[None] * 2 - 1).mean()
        loss_dict["gen_loss"] = args.gen_w * v["w"] * (l1 + args.lpips_w * lp)
        loss_dict["tv_loss"] = args.tv_w * total_variation(rgb, keep)
        for k, val in mesh_losses(nv, nv["mesh_valid"].float(), cfg.trainer.losses.mesh).items():
            loss_dict["nv_" + k] = val

        for k, val in loss_dict.items():
            if not torch.isfinite(val).all():
                raise ValueError(f"non-finite loss {k} at step {step}")
        ramp = min(1.0, (step - start) / args.warmup_steps)
        for g in trainer.optimizer.param_groups:
            sched = trainer.lr_schedulers.get(g["name"])
            g["lr"] = ramp * (sched(step) if sched is not None else base_lr[g["name"]])
        trainer.backward(loss_dict)
        trainer.postprocess_per_train_step(step=step)

        for k, val in {**{k: val.item() for k, val in loss_dict.items()}, "gen_l1": l1.item(), "gen_lpips": lp.item(),
                       "gen_keep": keep.mean().item()}.items():
            running[k] = running.get(k, 0.0) + val
        if (step - start) % args.print_every == 0:
            msg = "  ".join(f"{k} {val / args.print_every:.4f}" for k, val in running.items())
            print(f"[train_lsd] step {step}  {msg}  gaussians {trainer.get_gaussian_count()}  "
                  f"{args.print_every / (time.time() - t_print):.1f} it/s", flush=True)
            running, t_print = {}, time.time()

    trainer.save_checkpoint(log_dir=log_dir, save_only_model=True, is_final=True)
    eval_args = argparse.Namespace(enable_wandb=False, render_video_postfix=None)
    do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, args=eval_args, render_keys=RENDER_KEYS)
    runtime = time.time() - t_begin
    splits = [s for s, on in (("test", cfg.render.render_test), ("full", cfg.render.render_full)) if on]
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump({"exp": EXP, "scene_id": args.scene_id, **collect_metrics(log_dir, splits)}, f, indent=2)
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump({
            "exp": EXP, "scene_id": args.scene_id, "init": src_dir, "dashrecon_commit": commit, "torch": torch.__version__,
            "runtime_s": runtime, "generation_s": gen_s, "rounds": rounds, "example_rounds": example_rounds,
            "gaussians": trainer.get_gaussian_count(), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT images (init run only unless real_w > 0) + dashrecon products + frozen SDXL/ControlNet; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[train_lsd] {args.scene_id}: {runtime / 60:.1f} min ({gen_s / 60:.1f} min generation) -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
