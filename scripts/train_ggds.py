"""Phase 8 E6: GGDS-style generative distillation into a trained E5 model (AGENTS Phase 8, DECISIONS D10-D12).

Stage 2 continues <output_root>/<init_exp>/<scene_id> (checkpoint_final.pth) for --steps steps. Every step keeps
all E5 losses on one real training frame and adds losses on one novel view drawn from a fixed pool:
    gen_loss                = gen_w * w(t) * (L1 + lpips_w * LPIPS-VGG)(render, target) over the kept pixels
    nv_mesh_depth_loss,
    nv_mesh_normal_loss     = the E5 mesh losses (same weights) over the pixels the mesh covers.
The pool is every --pool_stride-th training frame (held-out frames are never used). Each view starts from the
frame's refined pose, moves sideways by |offset| ~ U(off_min, off_max) scene units to a random side, and turns
by yaw ~ U(-yaw_max, yaw_max) degrees. Targets are dashrecon.gen.ggds.SDXLRefiner outputs for the current render
of each view, regenerated for the whole pool every --refresh_every steps. This is an iterative dataset update;
GGDS regenerates every step, which would cost about 4 s per step here. Round r draws t ~ U(t_min, t_max(r)), with
t_max annealed linearly from t_max_start to t_max_end; w(t) = sqrt(alpha_bar) at the start timestep.

Kept pixels are those the mesh covers or where the rendered Gaussian opacity exceeds --keep_alpha; elsewhere
(sky, empty background) the target is replaced by the detached render. Without this, SDXL turns faint cloud
streaks into bold ones and the shared sky model carries them into every view: a 300-step val056 test raised
held-out LPIPS from 0.16 to 0.28 while non-sky PSNR barely moved. Novel views render with novel_view=True
(no CamPose), and the Affine and Sky parameters are frozen during the novel-view forward, so the generated
targets move only the Gaussians. The mask roughly halves the loss, so gen_w defaults to 1.0 (the unmasked test
used 0.5).

Changes to the E5 config for stage 2:
    trainer.optim.num_iters = start step + steps (the xyz learning-rate schedule stretches accordingly);
    gaussian_ctrl_general_cfg.reset_alpha_interval = 10**9, because VanillaGaussians resets opacity whenever
    step % interval == refine_interval, which would hit 30100 and 33100. Densification is already off
    (stop_split_at 15000); culling of transparent Gaussians continues every refine_interval steps.
drivestudio checkpoints hold no optimizer state, so stage 2 starts a fresh Adam, whose first steps move every
parameter by about its full learning rate; the first step would also use the unscheduled initial xyz rate.
Every learning rate (scheduled value, or the configured constant) is therefore multiplied by
lr_scale * min(1, (step - start) / warmup_steps). With the warmup and gen_w = 0, 300 val056 steps left the
held-out metrics unchanged or slightly better (PSNR 27.41 -> 27.64, LPIPS 0.163 -> 0.163).

Outputs in <output_root>/E6/<scene_id>/: config.yaml, checkpoint_final.pth, metrics/, videos/ (drivestudio
do_evaluation, as for E3-E5), metrics.json, meta.json (generative: true), ggds_params.json and
gen/round<r>_view<k>.jpg (render | control | target) for the first --save_views views of every round.

Example (main venv, from the repo root):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python scripts/train_ggds.py --scene_id val056 --output_root results
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
from dashrecon.gen.ggds import SDXLRefiner  # noqa: E402
from dashrecon.gen.novel import (build_trainer, mesh_losses, refined_c2w, render_and_refine,  # noqa: E402
                                 shifted_c2w, to_device)
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from train_gs import collect_metrics  # noqa: E402

EXP = "E6"
PROMPT = "a dashcam photo of a street, realistic, sharp, highly detailed"
NEGATIVE = "blurry, smeared, low quality, distorted, artifacts, painting, cartoon"
RENDER_KEYS = ["gt_rgbs", "rgbs", "Background_rgbs"]


def make_pool(dataset, trainer, stride: int, off_min: float, off_max: float, yaw_max: float,
              rng: np.random.Generator, device: torch.device) -> list:
    pool = []
    for idx in range(0, len(dataset.train_image_set), stride):
        ii, ci = dataset.train_image_set.get_image(idx, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        off = float(rng.choice([-1.0, 1.0]) * rng.uniform(off_min, off_max))
        yaw = float(rng.uniform(-yaw_max, yaw_max))
        ci["camera_to_world"] = shifted_c2w(refined_c2w(trainer, ii, ci), off, yaw)
        pool.append({"ii": ii, "ci": ci, "frame": int(ii["img_idx"].flatten()[0]), "offset": off, "yaw": yaw})
    return pool


def save_example(path: str, render: torch.Tensor, control: torch.Tensor, target: torch.Tensor) -> None:
    row = torch.cat([render.clamp(0, 1), control[..., None].expand(-1, -1, 3), target], dim=1)
    img = (row.cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.LANCZOS).save(path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--init_exp", default="E5")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--refresh_every", type=int, default=1000)
    parser.add_argument("--pool_stride", type=int, default=4)
    parser.add_argument("--off_min", type=float, default=0.5)
    parser.add_argument("--off_max", type=float, default=2.5)
    parser.add_argument("--yaw_max", type=float, default=3.0)
    parser.add_argument("--t_min", type=float, default=0.3)
    parser.add_argument("--t_max_start", type=float, default=0.7)
    parser.add_argument("--t_max_end", type=float, default=0.4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--gen_w", type=float, default=1.0)
    parser.add_argument("--keep_alpha", type=float, default=0.5)
    parser.add_argument("--lpips_w", type=float, default=0.5)
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
    start = int(cfg.trainer.optim.num_iters)
    cfg.trainer.optim.num_iters = start + args.steps
    cfg.trainer.gaussian_ctrl_general_cfg.reset_alpha_interval = 10**9
    cfg.log_dir = log_dir
    for sub in ("metrics", "videos", "gen"):
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    params = {**vars(args), "prompt": PROMPT, "negative_prompt": NEGATIVE, "init_ckpt": os.path.join(src_dir, "checkpoint_final.pth"),
              "start_step": start, "dashrecon_commit": commit}
    with open(os.path.join(log_dir, "ggds_params.json"), "w") as f:
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
    trainer.initialize_optimizer()
    base_lr = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    refiner = SDXLRefiner(device, PROMPT, NEGATIVE, args.guidance_scale, args.controlnet_scale, args.num_steps, tuple(args.gen_hw))
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    frozen_params = list(trainer.models["Affine"].parameters()) + list(trainer.models["Sky"].parameters())
    pool = make_pool(dataset, trainer, args.pool_stride, args.off_min, args.off_max, args.yaw_max, rng, device)
    n_rounds = -(-args.steps // args.refresh_every)
    print(f"[train_ggds] {args.scene_id}: pool {len(pool)} views, {n_rounds} rounds, steps {start + 1}..{start + args.steps}", flush=True)

    gen_s, rounds, running, t_print = 0.0, [], {}, time.time()
    for step in range(start + 1, start + args.steps + 1):
        r = (step - start - 1) // args.refresh_every
        if (step - start - 1) % args.refresh_every == 0:
            t_hi = args.t_max_start + (args.t_max_end - args.t_max_start) * (r / max(n_rounds - 1, 1))
            t0 = time.time()
            ts = []
            for k, v in enumerate(pool):
                t = float(rng.uniform(args.t_min, t_hi))
                out, control, target = render_and_refine(trainer, refiner, v["ii"], v["ci"], t, args.disp_pct, novel_view=True)
                v.update(target=target.half(), t=t, w=refiner.signal_weight(t))
                ts.append(t)
                if k < args.save_views:
                    save_example(os.path.join(log_dir, "gen", f"round{r}_view{k}.jpg"), out["rgb"], control, target)
            gen_s += time.time() - t0
            rounds.append({"round": r, "step": step, "t_max": t_hi, "t_mean": float(np.mean(ts)), "seconds": time.time() - t0})
            print(f"[train_ggds] round {r} at step {step}: t in [{args.t_min}, {t_hi:.2f}], {time.time() - t0:.0f} s", flush=True)

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
        loss_dict["gen_loss"] = args.gen_w * v["w"] * (l1 + args.lpips_w * lp)
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
            print(f"[train_ggds] step {step}  {msg}  gaussians {trainer.get_gaussian_count()}  "
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
            "runtime_s": runtime, "generation_s": gen_s, "rounds": rounds, "pool": [
                {"frame": v["frame"], "offset": v["offset"], "yaw": v["yaw"]} for v in pool],
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT images + dashrecon products + frozen SDXL/ControlNet; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[train_ggds] {args.scene_id}: {runtime / 60:.1f} min ({gen_s / 60:.1f} min generation) -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
