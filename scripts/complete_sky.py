"""E35 (docs/PLAN_AFTER_E30.md R3): complete the Sky cube map from the elevation profile of the real sky.

A turned camera looks at sky directions the FRONT camera never saw. Their cube-map texels are the 0.5 grey of the
initialisation, or (E27b / E30) a fit to the sky of a few generated frames: the grey band at the top of yaw-45
views and the blue sky cut to a rectangle (results/_vis_plan_e30). Clear-sky colour depends mostly on elevation, so
every texel that no real FRONT sky pixel reads is refitted to the colour profile of the real sky:
    profile    median colour of the run's own sky model over the real FRONT training sky pixels (dataset sky_masks),
               per --bin_deg elevation bin, smoothed over --smooth_bins; directions above the highest observed bin
               take its colour, below the lowest the lowest bin's (no extrapolated gradient);
    fill       every texel takes the bilinear-weighted mean of profile(elevation) over the random directions that read
               it: two backward passes of the cube-map lookup per batch of --dirs_per_batch uniform sphere directions
               accumulate sum(w * colour) and sum(w) per texel (--batches batches). Deterministic given the seed and
               smooth, unlike an Adam fit, where each texel is hit a few times per step and every update has the same
               size (that left salt-and-pepper noise, first test). Texels a real FRONT sky pixel reads are left exactly
               as they are (the gradient trick of scripts/fit_sky.py --freeze_real), so the forward sky does not change.
Elevation = asin of the world z of the direction (the up axis, as dashrecon.gen.views). The Gaussians are copied
unchanged; azimuth (sun direction) and clouds are not modelled. Not generative: the new texels repeat the observed
sky's colour by elevation.

Output <out_log_dir>: checkpoint_final.pth (the run's with the new Sky), config.yaml (the run's), sky_complete.json
(profile, frozen and changed fractions, commit), meta.json (generative as the init run).

Example (main venv):
    .venvs/main/bin/python scripts/complete_sky.py --log_dir results/E30/val056/model --out_log_dir results/E35/val056/model --exp E35
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402


def elevation_deg(dirs: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.asin((dirs[:, 2] / dirs.norm(dim=1)).clamp(-1.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--bin_deg", type=float, default=1.0)
    parser.add_argument("--smooth_bins", type=int, default=3)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--dirs_per_batch", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_log_dir, exist_ok=False)
    commit = git_commit()
    device = torch.device("cuda")
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

    # real FRONT sky directions, the texels they read (frozen), and the run's sky colour there
    sky.base.grad = None
    real_dirs = []
    for idx in range(len(dataset.train_image_set)):
        ii, _ = dataset.train_image_set.get_image(idx, 1)
        dirs = ii["viewdirs"][ii["sky_masks"] > 0.5].float().to(device)
        if len(dirs) == 0:
            continue
        sky({"viewdirs": dirs}).sum().backward()
        real_dirs.append(dirs)
    frozen = sky.base.grad.abs().sum(-1) > 0
    sky.base.grad = None
    real_dirs = torch.cat(real_dirs)
    with torch.no_grad():
        real_col = sky({"viewdirs": real_dirs}).clamp(0.0, 1.0)
    elev = elevation_deg(real_dirs)
    lo_bin, hi_bin = int(torch.floor(elev.min() / args.bin_deg)), int(torch.floor(elev.max() / args.bin_deg))
    centres, colours = [], []
    for b in range(lo_bin, hi_bin + 1):
        sel = (elev >= b * args.bin_deg) & (elev < (b + 1) * args.bin_deg)
        if int(sel.sum()) < 100:
            continue
        centres.append((b + 0.5) * args.bin_deg)
        colours.append(real_col[sel].median(0).values.cpu().numpy())
    centres, colours = np.asarray(centres), np.stack(colours)
    k = args.smooth_bins
    smoothed = np.stack([colours[max(0, i - k):i + k + 1].mean(0) for i in range(len(colours))])
    print(f"[complete_sky] {len(real_dirs):,} real sky pixels, elevation {elev.min():.1f}..{elev.max():.1f} deg, "
          f"{len(centres)} bins; frozen {frozen.float().mean().item() * 100:.2f}% of the texels", flush=True)
    c_t = torch.tensor(centres, dtype=torch.float32, device=device)
    s_t = torch.tensor(smoothed, dtype=torch.float32, device=device)

    def profile(d: torch.Tensor) -> torch.Tensor:
        e = elevation_deg(d).clamp(float(c_t[0]), float(c_t[-1]))
        i = torch.searchsorted(c_t, e).clamp(1, len(c_t) - 1)
        w = ((e - c_t[i - 1]) / (c_t[i] - c_t[i - 1]))[:, None]
        return (1 - w) * s_t[i - 1] + w * s_t[i]

    acc = torch.zeros_like(sky.base)
    wsum = torch.zeros(sky.base.shape[:3], device=device)
    for b in range(args.batches):
        d = torch.nn.functional.normalize(torch.randn(args.dirs_per_batch, 3, device=device), dim=1)
        sky.base.grad = None
        (sky({"viewdirs": d}) * profile(d)).sum().backward()  # d/d texel of sum_d c(d) . lookup(d) = sum_d w c(d)
        acc += sky.base.grad
        sky.base.grad = None
        sky({"viewdirs": d})[..., 0].sum().backward()  # sum_d w
        wsum += sky.base.grad[..., 0]
    sky.base.grad = None
    fill = ~frozen & (wsum > 0)
    with torch.no_grad():
        sky.base[fill] = (acc[fill] / wsum[fill][:, None]).clamp(0.0, 1.0)
    unreached = float((~frozen & (wsum == 0)).float().mean())
    print(f"[complete_sky] filled {fill.float().mean().item() * 100:.2f}% of the texels; {unreached * 100:.3f}% unreached", flush=True)
    assert torch.equal(sky.base.detach()[frozen], base0[frozen]), "a frozen texel moved"

    state = torch.load(os.path.join(args.log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    assert state["models"]["Sky"]["base"].shape == sky.base.shape, state["models"]["Sky"]["base"].shape
    state["models"]["Sky"]["base"] = sky.base.detach().cpu()
    torch.save(state, os.path.join(args.out_log_dir, "checkpoint_final.pth"))
    shutil.copyfile(os.path.join(args.log_dir, "config.yaml"), os.path.join(args.out_log_dir, "config.yaml"))
    changed = (sky.base.detach() - base0).abs().max(-1).values > 0.02
    with open(os.path.join(args.out_log_dir, "sky_complete.json"), "w") as f:
        json.dump({**vars(args), "real_sky_pixels": len(real_dirs), "profile_elevation_deg": centres.tolist(),
                   "profile_rgb": smoothed.tolist(), "texels_unreached_fraction": unreached, "texels_frozen_fraction": float(frozen.float().mean()),
                   "texels_changed_fraction": float(changed.float().mean()), "generative": False,
                   "kind": "unseen Sky texels refitted to the real sky's colour by elevation; all Gaussians unchanged",
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    # the completed model carries the init run's generated content; runs from before Phase 8 (E5f) have no such key
    init_meta = json.load(open(os.path.join(args.log_dir, "meta.json")))
    init_generative = bool(init_meta["generative"]) if "generative" in init_meta else False
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"exp": args.exp, "init": args.log_dir, "generative": init_generative, "sky_completion_generative": False,
                   "uses_oracle": False, "dashrecon_commit": commit}, f, indent=2)
    print(f"[complete_sky] {args.out_log_dir}: {float(changed.float().mean()) * 100:.1f}% of the texels changed, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
