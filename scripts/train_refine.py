"""Continue a trained run on its real FRONT frames with other densification settings (OPEN_QUESTIONS 38).

Diagnostic for why renders have about 1/3 of the real images' high-frequency detail: if more Gaussians (densification
back on for --densify_steps with gradient threshold --densify_grad_thresh) make the held-out frames sharper, capacity
limits the detail; if not, the views disagree (pose / rolling shutter / distortion) and the model averages them.
Only the init run's own losses on real training frames are used (no generated content); opacity reset off and
learning rates warmed up over --warmup_steps as in the other continuation scripts (the checkpoint has no optimizer
state). --campose_lr_scale scales the camera-refinement learning rate (a second diagnostic: better-aligned views).

Outputs in --out_log_dir: config.yaml, checkpoint_final.pth, metrics/, videos/, metrics.json, meta.json.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/train_refine.py --scene_id val056 --init_log_dir results/E5c/val056 \
        --out_log_dir results/E5c_dens/val056 --steps 6000 --densify_steps 4000 --densify_grad_thresh 0.0002
"""
import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from train_gs import collect_metrics  # noqa: E402

RENDER_KEYS = ["gt_rgbs", "rgbs", "Background_rgbs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--init_log_dir", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--densify_steps", type=int, default=4000)
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0002)
    parser.add_argument("--campose_lr_scale", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(os.path.join(args.init_log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    init_ckpt = os.path.join(args.init_log_dir, "checkpoint_final.pth")
    start = int(torch.load(init_ckpt, map_location="cpu", weights_only=False)["step"])
    cfg.trainer.optim.num_iters = start + args.steps
    ctrl = cfg.trainer.gaussian_ctrl_general_cfg
    ctrl.reset_alpha_interval = 10**9
    ctrl.stop_split_at = start + args.densify_steps
    ctrl.densify_grad_thresh = args.densify_grad_thresh
    cfg.log_dir = args.out_log_dir
    for sub in ("metrics", "videos"):
        os.makedirs(os.path.join(args.out_log_dir, sub), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))

    from datasets.driving_dataset import DrivingDataset
    from tools.eval import do_evaluation

    torch.cuda.reset_peak_memory_stats()
    t_begin = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=init_ckpt, load_only_model=True)
    assert trainer.step == start, (trainer.step, start)
    n_before = sum(trainer.get_gaussian_count().values())
    trainer.initialize_optimizer()
    base_lr = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    campose_groups = [g["name"] for g in trainer.optimizer.param_groups if g["name"].startswith("CamPose")]
    assert campose_groups, [g["name"] for g in trainer.optimizer.param_groups]
    print(f"[train_refine] {args.scene_id}: {n_before:,} Gaussians, steps {start + 1}..{start + args.steps}, densify until "
          f"{start + args.densify_steps} at grad {args.densify_grad_thresh}, CamPose lr x{args.campose_lr_scale} ({campose_groups})", flush=True)

    running, t_print = {}, time.time()
    for step in range(start + 1, start + args.steps + 1):
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        ii, ci = dataset.train_image_set.next(trainer._get_downscale_factor())
        ii, ci = to_device(ii, device), to_device(ci, device)
        outputs = trainer(ii, ci)
        trainer.update_visibility_filter()
        loss_dict = trainer.compute_losses(outputs=outputs, image_infos=ii, cam_infos=ci)
        for k, val in loss_dict.items():
            if not torch.isfinite(val).all():
                raise ValueError(f"non-finite loss {k} at step {step}")
        ramp = min(1.0, (step - start) / args.warmup_steps)
        for g in trainer.optimizer.param_groups:
            sched = trainer.lr_schedulers.get(g["name"])
            scale = args.campose_lr_scale if g["name"] in campose_groups else 1.0
            g["lr"] = ramp * scale * (sched(step) if sched is not None else base_lr[g["name"]])
        trainer.backward(loss_dict)
        trainer.postprocess_per_train_step(step=step)
        with torch.no_grad():
            psnr = trainer.compute_metrics(outputs=outputs, image_infos=ii)["psnr"].item()
        for k, val in {**{k: val.item() for k, val in loss_dict.items()}, "psnr": psnr}.items():
            running[k] = running.get(k, 0.0) + val
        if (step - start) % args.print_every == 0:
            msg = "  ".join(f"{k} {val / args.print_every:.4f}" for k, val in running.items())
            print(f"[train_refine] step {step}  {msg}  gaussians {trainer.get_gaussian_count()}  "
                  f"{args.print_every / (time.time() - t_print):.1f} it/s", flush=True)
            running, t_print = {}, time.time()

    trainer.save_checkpoint(log_dir=args.out_log_dir, save_only_model=True, is_final=True)
    do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, args=argparse.Namespace(enable_wandb=False, render_video_postfix=None),
                  render_keys=RENDER_KEYS)
    runtime = time.time() - t_begin
    splits = [s for s, on in (("test", cfg.render.render_test), ("full", cfg.render.render_full)) if on]
    with open(os.path.join(args.out_log_dir, "metrics.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, **collect_metrics(args.out_log_dir, splits)}, f, indent=2)
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, "init": args.init_log_dir, "params": vars(args), "start_step": start,
                   "gaussians_before": n_before, "gaussians_after": int(sum(trainer.get_gaussian_count().values())),
                   "dashrecon_commit": commit, "torch": torch.__version__, "runtime_s": runtime,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": False, "uses_oracle": False,
                   "oracle_note": "FRONT images + dashrecon products; checked by dashrecon.train.guard"}, f, indent=2)
    print(f"[train_refine] {args.scene_id}: {runtime / 60:.1f} min, {n_before:,} -> {sum(trainer.get_gaussian_count().values()):,} "
          f"Gaussians -> {args.out_log_dir}", flush=True)


if __name__ == "__main__":
    main()
