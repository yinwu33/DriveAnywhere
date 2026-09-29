"""Phase 8 E7: progressive distillation of NVIDIA Fixer into a trained model (DECISIONS D14; Difix3D+ recipe).

Stage 2 continues <output_root>/<init_exp>/<scene_id> (default E5c, the self-calibrated pipeline of D13) for
--steps steps. Every step keeps all losses of the init run on one real training frame and adds, for one novel
view of the current pool,
    gen_loss                = gen_w * (L1 + lpips_w * LPIPS-VGG)(render, target) over the kept pixels
    nv_mesh_depth_loss,
    nv_mesh_normal_loss     = the E5 mesh losses (same weights) over the pixels the mesh covers.
Every --refresh_every steps the pool is rebuilt (progressive update as in Difix3D+): every --pool_stride-th
training frame (never a held-out one), moved sideways from its refined pose by |offset| ~ U(lo(r), hi(r)) scene
units to a random side and turned by yaw ~ U(-yaw_max, yaw_max) degrees, where [lo, hi] grows linearly from
[off_lo_start, off_hi_start] in the first round to [off_lo_end, off_hi_end] in the last. Each pool view is rendered
with the current model and restored by Fixer (single step at --timestep; dashrecon.gen.fixer_client), and the
restored image is its target until the next refresh. Fixer is ~0.12 s per 960x640 image, so the pool is refreshed
much more often than E6's SDXL targets (every 1000 steps).

Loss weights: LPIPS-VGG between a render and its Fixer target is ~0.15 and L1 ~0.015 (val056 smoke run), so
gen_w 0.5 and lpips_w 0.2 put gen_loss (~0.02) level with the real-frame rgb + ssim losses (~0.025) instead of
the E6 weights (1.0, 0.5), which would make it ~4x larger.

As in E6: kept pixels are those the mesh covers or where the rendered Gaussian opacity exceeds --keep_alpha
(elsewhere the target is the detached render); novel views run with novel_view=True and frozen Affine and Sky;
the opacity reset is disabled and every learning rate warms up over --warmup_steps (the checkpoint has no
optimizer state); see scripts/train_ggds.py for the reasons.

Outputs in <output_root>/E7/<scene_id>/: config.yaml, checkpoint_final.pth, metrics/, videos/, metrics.json,
meta.json (generative: true), fixer_params.json and gen/round<r>_view<k>.jpg (render | Fixer target) for the
first --save_views views of rounds 0, the middle one and the last one.

E19 (docs/EXPERIMENTS.md) options: --init_log_dir / --out_log_dir / --exp name the runs explicitly (instead of
<output_root>/<init_exp> -> <output_root>/E7); --yaw_choices turns every pool view to one of these yaws (plus the
uniform +-yaw_max jitter), e.g. the Phase 9 turning angles, with --off_* 0 for no sideways move. A run trained with a
view gate (E14 / E17) keeps rendering its novel views through it; refinement is then off (as train_fill.py
--no_refine) so the gate's Gaussians keep their indices.

Example (main venv, from the repo root):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/train_fixer.py --scene_id val056 --output_root results
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.fixer_client import FIXER_REPO_ID, FIXER_REVISION, FixerClient  # noqa: E402
from dashrecon.gen.novel import (attach_view_gate, build_trainer, mesh_losses, refined_c2w, shifted_c2w, to_device,  # noqa: E402
                                 view_gate_path)
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from train_gs import collect_metrics  # noqa: E402

EXP = "E7"
RENDER_KEYS = ["gt_rgbs", "rgbs", "Background_rgbs"]


def pool_frames(dataset, stride: int, device: torch.device) -> list:
    """Training frames of the pool with their image/camera infos (camera poses are set per round)."""
    frames = []
    for idx in range(0, len(dataset.train_image_set), stride):
        ii, ci = dataset.train_image_set.get_image(idx, 1)
        frames.append({"ii": to_device(ii, device), "ci": to_device(ci, device), "frame": int(ii["img_idx"].flatten()[0])})
    return frames


def save_example(path: str, render: np.ndarray, target: np.ndarray) -> None:
    img = (np.concatenate([np.clip(render, 0, 1), target], axis=1) * 255).round().astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.LANCZOS).save(path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--init_exp", default="E5c")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--refresh_every", type=int, default=250)
    parser.add_argument("--pool_stride", type=int, default=4)
    parser.add_argument("--off_lo_start", type=float, default=0.25)
    parser.add_argument("--off_hi_start", type=float, default=0.75)
    parser.add_argument("--off_lo_end", type=float, default=1.5)
    parser.add_argument("--off_hi_end", type=float, default=2.5)
    parser.add_argument("--yaw_max", type=float, default=3.0)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--gen_w", type=float, default=0.5)
    parser.add_argument("--lpips_w", type=float, default=0.2)
    parser.add_argument("--keep_alpha", type=float, default=0.5)
    parser.add_argument("--save_views", type=int, default=4)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_log_dir", help="start run (default <output_root>/<init_exp>/<scene_id>)")
    parser.add_argument("--out_log_dir", help="output run (default <output_root>/E7/<scene_id>)")
    parser.add_argument("--exp", default=EXP)
    parser.add_argument("--yaw_choices", type=float, nargs="*", default=[], help="pool yaws (degrees) before the jitter")
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    src_dir = os.path.join(args.output_root, args.init_exp, args.scene_id) if args.init_log_dir is None else args.init_log_dir
    log_dir = os.path.join(args.output_root, EXP, args.scene_id) if args.out_log_dir is None else args.out_log_dir
    cfg = OmegaConf.load(os.path.join(src_dir, "config.yaml"))
    assert_non_oracle(cfg)
    assert "mesh" in cfg.trainer.losses, f"{src_dir} was trained without the mesh losses"
    start = int(cfg.trainer.optim.num_iters)
    cfg.trainer.optim.num_iters = start + args.steps
    cfg.trainer.gaussian_ctrl_general_cfg.reset_alpha_interval = 10**9
    gate_path = view_gate_path(cfg)
    if gate_path is not None:  # keep the gated Gaussians' indices (train_fill.py --no_refine)
        cfg.trainer.gaussian_ctrl_general_cfg.refine_interval = 10**9
        if "ctrl" in cfg.model.Background:
            cfg.model.Background.ctrl.refine_interval = 10**9
    cfg.log_dir = log_dir
    for sub in ("metrics", "videos", "gen"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    params = {**vars(args), "generator": FIXER_REPO_ID, "generator_revision": FIXER_REVISION,
              "init_ckpt": os.path.join(src_dir, "checkpoint_final.pth"), "start_step": start, "dashrecon_commit": commit}
    with open(os.path.join(log_dir, "fixer_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    from datasets.driving_dataset import DrivingDataset
    from tools.eval import do_evaluation
    import lpips

    torch.cuda.reset_peak_memory_stats()
    t_begin = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=params["init_ckpt"], load_only_model=True)
    assert trainer.step == start, (trainer.step, start)
    if gate_path is not None:
        attach_view_gate(trainer, gate_path)
    trainer.initialize_optimizer()
    base_lr = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    fixer = FixerClient(args.timestep, os.path.join(log_dir, "fixer_io"), os.environ["CUDA_HOME"])
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    frozen_params = list(trainer.models["Affine"].parameters()) + list(trainer.models["Sky"].parameters())
    frames = pool_frames(dataset, args.pool_stride, device)
    n_rounds = -(-args.steps // args.refresh_every)
    example_rounds = sorted({0, n_rounds // 2, n_rounds - 1})
    print(f"[train_fixer] {args.scene_id}: pool {len(frames)} frames, {n_rounds} rounds, steps {start + 1}..{start + args.steps}", flush=True)

    rounds, pool, running, t_print = [], [], {}, time.time()
    for step in range(start + 1, start + args.steps + 1):
        r = (step - start - 1) // args.refresh_every
        if (step - start - 1) % args.refresh_every == 0:
            t0 = time.time()
            frac = r / max(n_rounds - 1, 1)
            lo = args.off_lo_start + (args.off_lo_end - args.off_lo_start) * frac
            hi = args.off_hi_start + (args.off_hi_end - args.off_hi_start) * frac
            pool, renders = [], []
            trainer.set_eval()
            with torch.no_grad():
                for fr in frames:
                    off = float(rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi))
                    yaw = float(rng.uniform(-args.yaw_max, args.yaw_max))
                    if args.yaw_choices:
                        yaw += float(rng.choice(args.yaw_choices))
                    ci = dict(fr["ci"])
                    ci["camera_to_world"] = shifted_c2w(refined_c2w(trainer, fr["ii"], fr["ci"]), off, yaw)
                    renders.append(trainer(fr["ii"], ci, novel_view=True)["rgb"].clamp(0, 1).float().cpu().numpy())
                    pool.append({"ii": fr["ii"], "ci": ci, "frame": fr["frame"], "offset": off, "yaw": yaw})
            renders = np.stack(renders)
            targets = fixer.refine(renders)
            for k, v in enumerate(pool):
                v["target"] = torch.from_numpy(targets[k]).to(device).half()
                if r in example_rounds and k < args.save_views:
                    save_example(os.path.join(log_dir, "gen", f"round{r}_view{k}.jpg"), renders[k], targets[k])
            rounds.append({"round": r, "step": step, "offset_range": [lo, hi], "seconds": time.time() - t0,
                           "mean_abs_change": float(np.abs(targets - renders).mean()),
                           "pool": [{"frame": v["frame"], "offset": v["offset"], "yaw": v["yaw"]} for v in pool]})
            if r % 4 == 0 or r == n_rounds - 1:
                print(f"[train_fixer] round {r} at step {step}: |offset| in [{lo:.2f}, {hi:.2f}], "
                      f"mean |target - render| {rounds[-1]['mean_abs_change']:.4f}, {time.time() - t0:.1f} s", flush=True)
            t_print += time.time() - t0

        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        ii, ci = dataset.train_image_set.next(trainer._get_downscale_factor())
        ii, ci = to_device(ii, device), to_device(ci, device)
        outputs = trainer(ii, ci)
        trainer.update_visibility_filter()
        loss_dict = trainer.compute_losses(outputs=outputs, image_infos=ii, cam_infos=ci)
        with torch.no_grad():
            metrics = trainer.compute_metrics(outputs=outputs, image_infos=ii)

        v = pool[int(rng.integers(len(pool)))]
        for p in frozen_params:
            p.requires_grad_(False)
        nv = trainer(v["ii"], v["ci"], novel_view=True)
        for p in frozen_params:
            p.requires_grad_(True)
        trainer.update_visibility_filter()  # postprocess_per_train_step reads the last render's radii and gradients
        rgb = nv["rgb"]
        with torch.no_grad():
            keep = (nv["mesh_valid"] | (nv["opacity"][..., 0] > args.keep_alpha)).float()[..., None]
        target = keep * v["target"].float() + (1.0 - keep) * rgb.detach()
        l1 = (rgb - target).abs().sum() / (3.0 * keep.sum()).clamp(min=1.0)
        lp = lpips_vgg(rgb.permute(2, 0, 1)[None] * 2 - 1, target.permute(2, 0, 1)[None] * 2 - 1).mean()
        loss_dict["gen_loss"] = args.gen_w * (l1 + args.lpips_w * lp)
        for k, val in mesh_losses(nv, nv["mesh_valid"].float(), cfg.trainer.losses.mesh).items():
            loss_dict["nv_" + k] = val

        for k, val in loss_dict.items():
            if not torch.isfinite(val).all():
                raise ValueError(f"non-finite loss {k} at step {step}")
        ramp = args.lr_scale * min(1.0, (step - start) / args.warmup_steps)
        for g in trainer.optimizer.param_groups:
            sched = trainer.lr_schedulers.get(g["name"])
            g["lr"] = ramp * (sched(step) if sched is not None else base_lr[g["name"]])
        trainer.backward(loss_dict)
        trainer.postprocess_per_train_step(step=step)

        for k, val in {**{k: val.item() for k, val in loss_dict.items()}, "psnr": metrics["psnr"].item(),
                       "gen_l1": l1.item(), "gen_lpips": lp.item(), "gen_keep": keep.mean().item()}.items():
            running[k] = running.get(k, 0.0) + val
        if (step - start) % args.print_every == 0:
            msg = "  ".join(f"{k} {val / args.print_every:.4f}" for k, val in running.items())
            print(f"[train_fixer] step {step}  {msg}  gaussians {trainer.get_gaussian_count()}  "
                  f"{args.print_every / (time.time() - t_print):.1f} it/s", flush=True)
            running, t_print = {}, time.time()

    fixer.close()
    trainer.save_checkpoint(log_dir=log_dir, save_only_model=True, is_final=True)
    eval_args = argparse.Namespace(enable_wandb=False, render_video_postfix=None)
    do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, args=eval_args, render_keys=RENDER_KEYS)
    runtime = time.time() - t_begin
    splits = [s for s, on in (("test", cfg.render.render_test), ("full", cfg.render.render_full)) if on]
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump({"exp": args.exp, "scene_id": args.scene_id, **collect_metrics(log_dir, splits)}, f, indent=2)
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump({
            "exp": args.exp, "scene_id": args.scene_id, "init": src_dir, "dashrecon_commit": commit, "torch": torch.__version__,
            "runtime_s": runtime, "generation_s": fixer.seconds, "generator": fixer.info, "rounds": rounds,
            "pool": rounds[0]["pool"], "example_rounds": example_rounds,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT images + dashrecon products + frozen NVIDIA Fixer; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[train_fixer] {args.scene_id}: {runtime / 60:.1f} min ({fixer.seconds / 60:.1f} min Fixer) -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
