"""E27 (docs/EXPERIMENTS.md): fit the Sky cube map to the sky of generated turning views.

The Sky model (models.modules.EnvLight) is a 6 x R x R cube map looked up by view direction, initialised to grey 0.5.
Only the directions the FRONT camera looked along were ever trained, so a turned camera sees the untouched grey
texels: the hazy grey sky of every Phase 9 side view. GEN3C painted real-looking skies in its generated frames, but
refine_holes.py hands generated sky to the Sky model and nothing trained it. Here only the cube map is optimised:
    generated  L1 between the cube map at the view direction of every generated sky pixel (views_dir/sky_gen/<k>.png,
               SegFormer sky of the generated frame) and that pixel's colour (views_dir/filled/<k>.png); directions
               from datasets.base.pixel_source.get_rays with the view's K and c2w (as dashrecon.gen.views.render_at);
    anchor     L1 between the cube map and its initial values at the directions of the real FRONT training sky pixels
               (dataset sky_masks), so the forward sky (trained through the per-image Affine) does not move;
    smooth     --tv_w x mean absolute difference between neighbouring texels (within each face).
Every other parameter of the run is copied unchanged.

Output <out_log_dir>: checkpoint_final.pth (the run's checkpoint with the new Sky), config.yaml (the run's),
fit_sky.json (parameters, sample counts, losses, commit; generative: true).

Example (main venv):
    .venvs/main/bin/python scripts/fit_sky.py --log_dir results/E23/val056/model --out_log_dir results/E27/val056/model --exp E27 \
        --views_dirs results/E20/val056/views/r0 results/E20/val056/views/r1 --steps 3000
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402


def generated_sky(views_dir: str, per_view: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(directions (N, 3), colours (N, 3)) of up to per_view random sky pixels of every generated view."""
    from datasets.base.pixel_source import get_rays

    cams = json.load(open(os.path.join(views_dir, "cams.json")))["cams"]
    dirs, cols = [], []
    for k, c in enumerate(cams):
        sky = np.asarray(Image.open(os.path.join(views_dir, "sky_gen", f"{k:03d}.png"))) > 127
        ys, xs = np.nonzero(sky)
        if len(ys) == 0:
            continue
        pick = rng.choice(len(ys), size=min(per_view, len(ys)), replace=False)
        ys, xs = ys[pick], xs[pick]
        rgb = np.asarray(Image.open(os.path.join(views_dir, "filled", f"{k:03d}.png")).convert("RGB"))[ys, xs] / 255.0
        _, viewdirs, _ = get_rays(torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32),
                                  torch.tensor(c["c2w"], dtype=torch.float32), torch.tensor(c["K"], dtype=torch.float32))
        dirs.append(viewdirs.numpy())
        cols.append(rgb)
    return np.concatenate(dirs), np.concatenate(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--views_dirs", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tv_w", type=float, default=0.1)
    parser.add_argument("--anchor_w", type=float, default=1.0)
    parser.add_argument("--per_view", type=int, default=4000, help="sky pixels sampled per generated view")
    parser.add_argument("--per_frame", type=int, default=4000, help="sky pixels sampled per real FRONT frame")
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_log_dir, exist_ok=False)
    commit = git_commit()
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    sky = trainer.models["Sky"]
    base0 = sky.base.detach().clone()

    gen_d, gen_c = [], []
    for vd in args.views_dirs:
        d, c = generated_sky(vd, args.per_view, rng)
        gen_d.append(d)
        gen_c.append(c)
    gen_d = torch.tensor(np.concatenate(gen_d), dtype=torch.float32, device=device)
    gen_c = torch.tensor(np.concatenate(gen_c), dtype=torch.float32, device=device)
    real_d = []
    for idx in range(len(dataset.train_image_set)):
        ii, _ = dataset.train_image_set.get_image(idx, 1)
        sky_px = ii["sky_masks"] > 0.5
        dirs = ii["viewdirs"][sky_px]
        if len(dirs) == 0:
            continue
        real_d.append(dirs[torch.from_numpy(rng.choice(len(dirs), size=min(args.per_frame, len(dirs)), replace=False))])
    real_d = torch.cat(real_d).float().to(device)
    with torch.no_grad():
        real_c = sky({"viewdirs": real_d}).detach()
    print(f"[fit_sky] {len(gen_d):,} generated sky pixels from {len(args.views_dirs)} trajectories, "
          f"{len(real_d):,} real FRONT sky anchors", flush=True)

    opt = torch.optim.Adam([sky.base], lr=args.lr)
    history = []
    for step in range(args.steps):
        gi = torch.randint(len(gen_d), (args.batch,), device=device)
        ri = torch.randint(len(real_d), (args.batch,), device=device)
        gen_loss = (sky({"viewdirs": gen_d[gi]}) - gen_c[gi]).abs().mean()
        anchor_loss = (sky({"viewdirs": real_d[ri]}) - real_c[ri]).abs().mean()
        b = sky.base
        tv = (b[:, 1:] - b[:, :-1]).abs().mean() + (b[:, :, 1:] - b[:, :, :-1]).abs().mean()
        loss = gen_loss + args.anchor_w * anchor_loss + args.tv_w * tv
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sky.base.clamp_(0.0, 1.0)
        if step % 500 == 0 or step == args.steps - 1:
            history.append({"step": step, "gen_l1": gen_loss.item(), "anchor_l1": anchor_loss.item(), "tv": tv.item()})
            print(f"[fit_sky] step {step}: generated L1 {gen_loss.item():.4f}  anchor L1 {anchor_loss.item():.4f}  tv {tv.item():.5f}", flush=True)

    state = torch.load(os.path.join(args.log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    assert state["models"]["Sky"]["base"].shape == sky.base.shape, state["models"]["Sky"]["base"].shape
    state["models"]["Sky"]["base"] = sky.base.detach().cpu()
    torch.save(state, os.path.join(args.out_log_dir, "checkpoint_final.pth"))
    shutil.copyfile(os.path.join(args.log_dir, "config.yaml"), os.path.join(args.out_log_dir, "config.yaml"))
    changed = (sky.base.detach() - base0).abs().max(-1).values > 0.02
    with open(os.path.join(args.out_log_dir, "fit_sky.json"), "w") as f:
        json.dump({**vars(args), "generated_samples": len(gen_d), "real_anchor_samples": len(real_d), "history": history,
                   "texels_changed_fraction": float(changed.float().mean()), "generative": True,
                   "kind": "Sky cube map fitted to GEN3C-generated sky; all Gaussians unchanged",
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"exp": args.exp, "init": args.log_dir,
                   "generative": True, "uses_oracle": False, "dashrecon_commit": commit}, f, indent=2)
    print(f"[fit_sky] {args.out_log_dir}: {float(changed.float().mean()) * 100:.1f}% of the texels changed, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
