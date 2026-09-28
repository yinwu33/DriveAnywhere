"""Phase 9 step 4 (DECISIONS D16-D18, experiment E8): lift filled views into the model and distil them.

Continues --init_log_dir (E5c for the first round, then E8 itself) with the filled trajectories of all rounds so far
(--views_dirs, each from render_views.py + fill_views.py + depth_views.py; keeping earlier rounds is the 3D memory).

1. Spawn: in every --spawn_stride-th frame of the newest views dir, hole pixels (every --pixel_stride-th) are
   back-projected with the aligned estimated depth and become new Gaussians: colour from the filled frame,
   isotropic scale = pixel footprint x --scale_factor, opacity --init_opacity; one per voxel of half the median
   footprint.
2. Distil for --steps steps: every step keeps all losses of the init run on one real training frame and adds one
   filled view (rays recomputed for its camera, novel_view=True, Affine and Sky frozen):
       fill_rgb_loss   = fill_w * (L1 + lpips_w * LPIPS-VGG), L1 weighted 1 on holes and --seen_w elsewhere, LPIPS
                         on the render pasted into the target outside the holes;
       fill_depth_loss = depth_w * inverse-depth L1 against the aligned depth on hole pixels.
   Opacity reset off, densification off (stage 2 is past stop_split_at), learning rates warm up over
   --warmup_steps (the checkpoint has no optimizer state), as in scripts/train_fixer.py.

E14 options (docs/EXPERIMENTS.md):
    --view_gate PATH   a dashrecon.gen.floaters.ViewGate (scripts/make_view_gate.py) applied to every novel-view render
                       (the generated views here, and everywhere the output is rendered later: it is written to the
                       output config as ``view_gate``); a run trained with one keeps it.
    --no_refine        no densification / culling / opacity reset during distillation (refine_interval 1e9 in
                       trainer.gaussian_ctrl_general_cfg and, if present, model.Background.ctrl), so the
                       Gaussians the gate covers keep their indices across rounds.
    --unknown_w W      with --validation_dirs: hole pixels without conflicting evidence that were not accepted
                       (status 1 unknown, 2 weak; validate_generated_views.py) get confidence W instead of 0;
                       conflicting pixels stay at 0.
    --round_affine     one 3x4 colour transform per views dir (a "virtual traversal" appearance, after MTGS), applied
                       to the render before it is compared with that dir's generated frames; identity at the start
                       (or the previous round's value), L2 to identity weighted --round_affine_reg. Real FRONT frames
                       keep the run's own Affine model.

Output: <out_log_dir>/ config.yaml, checkpoint_final.pth (also rounds/ckpt_r<round>.pth), metrics/ (this round's;
earlier rounds' in rounds/metrics_r<k>/), videos/,
metrics.json, meta.json (generative: true, per round: views, spawned Gaussians, runtime), fill_params_r<round>.json.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/train_fill.py --scene_id val039 --init_log_dir results/E5c/val039 \
        --out_log_dir results/E8/val039 --round 0 --views_dirs results/E8/val039/views/r0_right1.5_yaw15
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.novel import attach_view_gate, build_trainer, to_device, view_gate_path  # noqa: E402
from dashrecon.gen.views import backproject, front_image_index, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from dashrecon.train.seeds import seed_scene_optimization  # noqa: E402
from train_gs import collect_metrics  # noqa: E402

SH_C0 = 0.28209479177387814
RENDER_KEYS = ["gt_rgbs", "rgbs", "Background_rgbs"]


def load_views(views_dir: str, device: torch.device, validation_dir: str | None = None,
               view_indices: list[int] | None = None, unknown_w: float = 0.0) -> list:
    """Load generated supervision, optionally restricted to a recorded local window. With a validation dir, the
    confidence is the accepted confidence, plus unknown_w on unknown / weak hole pixels (status 1, 2)."""
    with open(os.path.join(views_dir, "cams.json")) as f:
        cams = json.load(f)["cams"]
    if validation_dir is not None:
        with open(os.path.join(validation_dir, "validation.json")) as f:
            validation = json.load(f)
        if os.path.realpath(validation["views_dir"]) != os.path.realpath(views_dir):
            raise ValueError(f"validation directory does not belong to {views_dir}")
    views = []
    indices = list(range(len(cams))) if view_indices is None else view_indices
    if not indices or len(indices) != len(set(indices)) or any(k < 0 or k >= len(cams) for k in indices):
        raise ValueError("view_indices must be distinct valid generated view indices")
    for k in indices:
        c = cams[k]
        rgb = np.asarray(Image.open(os.path.join(views_dir, "filled", f"{k:03d}.png")).convert("RGB"), dtype=np.float32) / 255.0
        hole = np.asarray(Image.open(os.path.join(views_dir, "mask", f"{k:03d}.png"))) > 127
        depth = np.load(os.path.join(views_dir, "filled_depth", f"{k:03d}.npy")).astype(np.float32)
        confidence = np.ones(hole.shape, np.float32)
        if validation_dir is not None:
            confidence = np.load(os.path.join(validation_dir, "confidence", f"{k:03d}.npy")).astype(np.float32)
            if confidence.shape != hole.shape or not np.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
                raise ValueError(f"invalid generation confidence in {views_dir}, view {k}")
            if unknown_w > 0:
                status = np.asarray(Image.open(os.path.join(validation_dir, "status", f"{k:03d}.png")))
                if status.shape != hole.shape or not np.array_equal(status > 0, hole):
                    raise ValueError(f"validation status does not match the hole mask in {views_dir}, view {k}")
                confidence = np.where((status == 1) | (status == 2), np.float32(unknown_w), confidence)
        views.append({"confidence": torch.from_numpy(confidence).to(device), "k": k, "frame": c["frame"], "dir": views_dir,
                      "c2w": torch.tensor(c["c2w"], dtype=torch.float32, device=device),
                      "K": torch.tensor(c["K"], dtype=torch.float32, device=device),
                      "rgb": torch.from_numpy(rgb).to(device).half(), "hole": torch.from_numpy(hole).to(device),
                      "depth": torch.from_numpy(depth).to(device).half()})
    return views


def spawn(views: list, spawn_stride: int, pixel_stride: int, scale_factor: float, init_opacity: float, sh_rest_shape) -> dict:
    """New Background Gaussians (state-dict tensors) for the hole pixels of every spawn_stride-th view."""
    pts, cols, foot = [], [], []
    for v in views[::spawn_stride]:
        hole = v["hole"].clone()
        sub = torch.zeros_like(hole)
        sub[::pixel_stride, ::pixel_stride] = True
        sel = (hole & sub & (v["depth"] > 0) & (v["confidence"] > 0)).reshape(-1)
        if sel.sum() == 0:
            continue
        p = backproject(v["depth"].float(), v["K"], v["c2w"])[sel]
        pts.append(p)
        cols.append(v["rgb"].float().reshape(-1, 3)[sel])
        foot.append(v["depth"].float().reshape(-1)[sel] / v["K"][0, 0] * pixel_stride)
    if not pts:
        raise ValueError("no validated positive-depth hole pixels to spawn; inspect geometry before distillation")
    pts, cols, foot = torch.cat(pts), torch.cat(cols), torch.cat(foot)
    voxel = 0.5 * float(foot.median())
    _, first = np.unique(np.floor(pts.cpu().numpy() / voxel).astype(np.int64), axis=0, return_index=True)
    first = torch.from_numpy(np.sort(first)).to(pts.device)
    pts, cols, foot = pts[first], cols[first], foot[first]
    n = len(pts)
    quats = torch.zeros(n, 4, device=pts.device)
    quats[:, 0] = 1.0
    return {"_means": pts, "_scales": torch.log(foot * scale_factor)[:, None].repeat(1, 3), "_quats": quats,
            "_features_dc": (cols - 0.5) / SH_C0, "_features_rest": torch.zeros((n,) + tuple(sh_rest_shape), device=pts.device),
            "_opacities": torch.logit(torch.full((n, 1), init_opacity, device=pts.device)), "voxel": voxel}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--init_log_dir", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True, help="experiment id written to metrics.json / meta.json (E8, E10)")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--views_dirs", nargs="+", required=True, help="all rounds so far, newest last (spawning uses the newest)")
    parser.add_argument("--validation_dirs", nargs="*", help="one validation directory per views_dir; confidence gates spawning and losses")
    parser.add_argument("--view_indices", type=int, nargs="+", help="local generated view window, applied to each views_dir; real FRONT supervision is unchanged")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--spawn_stride", type=int, default=3)
    parser.add_argument("--pixel_stride", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=0.7)
    parser.add_argument("--init_opacity", type=float, default=0.5)
    parser.add_argument("--fill_w", type=float, default=0.5)
    parser.add_argument("--lpips_w", type=float, default=0.2)
    parser.add_argument("--seen_w", type=float, default=0.1)
    parser.add_argument("--depth_w", type=float, default=0.1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view_gate", help="ViewGate for novel views when --init_log_dir has none (E14 round 0)")
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--unknown_w", type=float, default=0.0)
    parser.add_argument("--round_affine", action="store_true")
    parser.add_argument("--round_affine_reg", type=float, default=1.0)
    parser.add_argument("--round_affine_lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.unknown_w > 0 and args.validation_dirs is None:
        raise ValueError("--unknown_w needs --validation_dirs (the status maps)")
    if args.validation_dirs is not None and len(args.validation_dirs) != len(args.views_dirs):
        raise ValueError("one --validation_dirs entry per --views_dirs required")
    if args.validation_dirs is not None and args.seen_w != 0:
        raise ValueError("validated distillation requires --seen_w 0; real frames anchor observed appearance")
    commit = git_commit()
    device = torch.device("cuda")
    seed_scene_optimization(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = OmegaConf.load(os.path.join(args.init_log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    init_ckpt = os.path.join(args.init_log_dir, "checkpoint_final.pth")
    state = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    start = int(state["step"])
    cfg.trainer.optim.num_iters = start + args.steps
    cfg.trainer.gaussian_ctrl_general_cfg.reset_alpha_interval = 10**9
    # drivestudio copies trainer.gaussian_ctrl_general_cfg into model.Background.ctrl when it builds the trainer
    # (models/trainers/base.py update_gaussian_cfg), so configs saved after training carry both; the class one wins
    if args.no_refine:
        cfg.trainer.gaussian_ctrl_general_cfg.refine_interval = 10**9
        if "ctrl" in cfg.model.Background:
            cfg.model.Background.ctrl.refine_interval = 10**9
    gate_path = view_gate_path(cfg)
    if args.view_gate is not None:
        assert gate_path is None or gate_path == args.view_gate, f"{args.init_log_dir} was trained with view gate {gate_path}"
        gate_path = args.view_gate
    if gate_path is not None:
        refine = cfg.model.Background.ctrl.refine_interval if "ctrl" in cfg.model.Background else cfg.trainer.gaussian_ctrl_general_cfg.refine_interval
        assert refine > start + args.steps, "a view gate needs --no_refine (stable indices)"
        cfg.view_gate = gate_path
    cfg.log_dir = args.out_log_dir
    for sub in ("metrics", "videos", "rounds"):
        os.makedirs(os.path.join(args.out_log_dir, sub), exist_ok=True)
    if args.round > 0:  # the previous round's drivestudio metrics files, so collect_metrics finds this round's only
        old = sorted(os.listdir(os.path.join(args.out_log_dir, "metrics")))
        assert old, f"round {args.round}: no metrics of round {args.round - 1} in {args.out_log_dir}/metrics"
        keep = os.path.join(args.out_log_dir, "rounds", f"metrics_r{args.round - 1}")
        os.makedirs(keep)
        for name in old:
            shutil.move(os.path.join(args.out_log_dir, "metrics", name), os.path.join(keep, name))
    params = {**vars(args), "init_ckpt": init_ckpt, "start_step": start, "dashrecon_commit": commit}
    with open(os.path.join(args.out_log_dir, f"fill_params_r{args.round}.json"), "w") as f:
        json.dump(params, f, indent=2)

    from datasets.driving_dataset import DrivingDataset
    from tools.eval import do_evaluation
    import lpips

    torch.cuda.reset_peak_memory_stats()
    t_begin = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    validation_dirs = [None] * len(args.views_dirs) if args.validation_dirs is None else args.validation_dirs
    views = [v for d, vd in zip(args.views_dirs, validation_dirs) for v in load_views(d, device, vd, args.view_indices, args.unknown_w)]
    newest = [v for v in views if v["dir"] == args.views_dirs[-1]]
    bg = state["models"]["Background"]
    new = spawn(newest, args.spawn_stride, args.pixel_stride, args.scale_factor, args.init_opacity, bg["_features_rest"].shape[1:])
    n_before = bg["_means"].shape[0]
    for key in ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities"):
        bg[key] = torch.cat([bg[key], new[key].to(bg[key].dtype).cpu()], dim=0)
    trainer.load_state_dict(state, load_only_model=True, strict=True)
    assert trainer.step == start, (trainer.step, start)
    if gate_path is not None:
        attach_view_gate(trainer, gate_path)
    eye = torch.eye(3, 4, device=device)
    affines = None
    if args.round_affine:
        init = {}
        if args.round > 0:
            prev = torch.load(os.path.join(args.out_log_dir, f"round_affine_r{args.round - 1}.pt"), map_location=device)
            init = prev["affines"]
        affines = torch.nn.Parameter(torch.stack([init[d].to(device) if d in init else eye.clone() for d in args.views_dirs]))
        affine_opt = torch.optim.Adam([affines], lr=args.round_affine_lr)
        dir_index = {d: i for i, d in enumerate(args.views_dirs)}
    trainer.initialize_optimizer()
    base_lr = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    frozen_params = list(trainer.models["Affine"].parameters()) + list(trainer.models["Sky"].parameters())
    print(f"[train_fill] {args.scene_id} round {args.round}: {len(views)} filled views from {len(args.views_dirs)} trajectories; "
          f"spawned {len(new['_means']):,} Gaussians (voxel {new['voxel']:.4f}) on top of {n_before:,}", flush=True)

    running, t_print = {}, time.time()
    sampled_real, sampled_generated, sampled_pool_indices = [], [], []
    for step in range(start + 1, start + args.steps + 1):
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        ii, ci = dataset.train_image_set.next(trainer._get_downscale_factor())
        sampled_real.append(int(ii["img_idx"].flatten()[0]))
        ii, ci = to_device(ii, device), to_device(ci, device)
        outputs = trainer(ii, ci)
        trainer.update_visibility_filter()
        loss_dict = trainer.compute_losses(outputs=outputs, image_infos=ii, cam_infos=ci)

        pool_index = int(rng.integers(len(views)))
        sampled_pool_indices.append(pool_index)
        v = views[pool_index]
        sampled_generated.append(v["frame"])
        fi, fc = dataset.full_image_set.get_image(front_image_index(dataset, v["frame"] - dataset.start_timestep), 1)
        fi, fc = to_device(fi, device), to_device(fc, device)
        for p in frozen_params:
            p.requires_grad_(False)
        nv = render_at(trainer, fi, fc, v["c2w"], v["K"], v["rgb"].shape[:2])
        for p in frozen_params:
            p.requires_grad_(True)
        trainer.update_visibility_filter()  # postprocess_per_train_step reads the last render's radii and gradients
        rgb, target = nv["rgb"], v["rgb"].float()
        if affines is not None:
            a = affines[dir_index[v["dir"]]]
            rgb = rgb @ a[:, :3].T + a[:, 3]
            loss_dict["round_affine_reg"] = args.round_affine_reg * ((a - eye) ** 2).sum()
        hole = v["hole"].float()[..., None]
        confidence = v["confidence"][..., None]
        accepted = hole * confidence
        wmap = accepted + args.seen_w * (1.0 - hole)
        l1 = ((rgb - target).abs() * wmap).sum() / (3.0 * wmap.sum().clamp(min=1.0))
        pasted = accepted * target + (1.0 - accepted) * rgb.detach()
        lp = lpips_vgg(rgb.permute(2, 0, 1)[None] * 2 - 1, pasted.permute(2, 0, 1)[None] * 2 - 1).mean()
        loss_dict["fill_rgb_loss"] = args.fill_w * (l1 + args.lpips_w * lp)
        dmask = (v["hole"] & (v["depth"] > 0)).float() * v["confidence"]
        inv = (1.0 / nv["depth"][..., 0].clamp(min=1e-3) - 1.0 / v["depth"].float().clamp(min=1e-3)).abs()
        loss_dict["fill_depth_loss"] = args.depth_w * (inv * dmask).sum() / dmask.sum().clamp(min=1.0)

        for k, val in loss_dict.items():
            if not torch.isfinite(val).all():
                raise ValueError(f"non-finite loss {k} at step {step}")
        ramp = min(1.0, (step - start) / args.warmup_steps)
        for g in trainer.optimizer.param_groups:
            sched = trainer.lr_schedulers.get(g["name"])
            g["lr"] = ramp * (sched(step) if sched is not None else base_lr[g["name"]])
        if affines is not None:
            affine_opt.zero_grad()
            scale = trainer.grad_scaler.get_scale()
        trainer.backward(loss_dict)
        if affines is not None:
            affines.grad /= scale
            affine_opt.step()
        trainer.postprocess_per_train_step(step=step)

        for k, val in {**{k: val.item() for k, val in loss_dict.items()}, "fill_l1": l1.item(), "fill_lpips": lp.item()}.items():
            running[k] = running.get(k, 0.0) + val
        if (step - start) % args.print_every == 0:
            msg = "  ".join(f"{k} {val / args.print_every:.4f}" for k, val in running.items())
            print(f"[train_fill] step {step}  {msg}  gaussians {trainer.get_gaussian_count()}  "
                  f"{args.print_every / (time.time() - t_print):.1f} it/s", flush=True)
            running, t_print = {}, time.time()

    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))
    with open(os.path.join(args.out_log_dir, f"sampled_views_r{args.round}.json"), "w") as f:
        json.dump({"seed": args.seed, "real_full_image_indices": sampled_real,
                   "generated_frames": sampled_generated, "generated_pool_indices": sampled_pool_indices}, f, indent=2)
    trainer.save_checkpoint(log_dir=args.out_log_dir, save_only_model=True, is_final=True)
    if affines is not None:
        final = {d: affines[i].detach().cpu() for d, i in dir_index.items()}
        torch.save({"affines": final}, os.path.join(args.out_log_dir, f"round_affine_r{args.round}.pt"))
        with open(os.path.join(args.out_log_dir, f"round_affine_r{args.round}.json"), "w") as f:
            json.dump({d: a.tolist() for d, a in final.items()}, f, indent=2)
    shutil.copyfile(os.path.join(args.out_log_dir, "checkpoint_final.pth"), os.path.join(args.out_log_dir, "rounds", f"ckpt_r{args.round}.pth"))
    do_evaluation(step=step, cfg=cfg, trainer=trainer, dataset=dataset, args=argparse.Namespace(enable_wandb=False, render_video_postfix=None),
                  render_keys=RENDER_KEYS)
    runtime = time.time() - t_begin
    splits = [s for s, on in (("test", cfg.render.render_test), ("full", cfg.render.render_full)) if on]
    with open(os.path.join(args.out_log_dir, "metrics.json"), "w") as f:
        json.dump({"exp": args.exp, "scene_id": args.scene_id, "round": args.round, **collect_metrics(args.out_log_dir, splits)}, f, indent=2)
    meta_path = os.path.join(args.out_log_dir, "meta.json")
    rounds = json.load(open(meta_path))["rounds"] if args.round > 0 else []
    assert len(rounds) == args.round, f"meta.json has {len(rounds)} rounds, this is round {args.round}"
    rounds.append({"round": args.round, "init_log_dir": args.init_log_dir, "views_dirs": args.views_dirs,
                   "spawned": int(len(new["_means"])), "voxel": new["voxel"], "gaussians_before": int(n_before),
                   "gaussians_after": int(sum(trainer.get_gaussian_count().values())), "steps": [start + 1, start + args.steps],
                   "runtime_s": runtime, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "dashrecon_commit": commit})
    with open(meta_path, "w") as f:
        json.dump({"exp": args.exp, "scene_id": args.scene_id, "init": rounds[0]["init_log_dir"], "rounds": rounds,
                   "dashrecon_commit": commit, "torch": torch.__version__, "runtime_s": sum(r["runtime_s"] for r in rounds),
                   "seed": args.seed, "seeded_random_generators": ["python", "numpy", "torch", "torch_cuda"],
                   "peak_vram_gb": max(r["peak_vram_gb"] for r in rounds), "generative": True, "uses_oracle": False,
                   "validation_dirs": validation_dirs,
                   "oracle_note": "FRONT + dashrecon products and generated RGB-D; generator/depth provenance in each views_dir fill.json/depth.json; checked by dashrecon.train.guard"},
                  f, indent=2)
    print(f"[train_fill] {args.scene_id} round {args.round}: {runtime / 60:.1f} min -> {args.out_log_dir}", flush=True)


if __name__ == "__main__":
    main()
