"""E42c (docs/EXPERIMENTS.md): fit only the colours (SH band 0) of the surfels a composite appended, on the real FRONT
training frames; every other parameter (all geometry, the other Gaussians, Sky, Affine, CamPose) stays fixed.

The surfels of scripts/road_surfel_composite.py carry raw pixel colours, but the run renders through its per-image
Affine colour transform, so raw colours come out shifted (E42: darker, bluer road). Fitting the band-0 colour of the
appended surfels (index >= --first, from the composite's meta.json when omitted) with the run's own photometric loss
puts them in the run's colour space without moving anything. --reg pulls each fitted colour towards its starting (raw
pixel) colour (mean squared band-0 change): surfels the training frames hardly see keep their raw colour instead of
drifting (E42c: teal patches on a cross street seen only at grazing angles). No densification or pruning: the Gaussian set and the
view gate stay valid.

Output: --out_log_dir with checkpoint_final.pth, config.yaml (view gate unchanged), meta.json.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:$PATH CUDA_HOME=$HOME/cuda-12.6 PYTHONPATH=. .venvs/main/bin/python scripts/fit_surfel_colours.py \\
        --log_dir results/E42c/val056_init --out_log_dir results/E42c/val056 --exp E42c --steps 3000 --lr 0.0025
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_log_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--first", type=int, default=None, help="first fitted Gaussian (default: the composite's kept count)")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--reg", type=float, default=0.0, help="weight of the pull towards the starting colours")
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    commit = git_commit()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    src_meta = json.load(open(os.path.join(args.log_dir, "meta.json")))
    first = args.first if args.first is not None else src_meta["gaussians_before"] - src_meta["road_removed"]
    os.makedirs(args.out_log_dir, exist_ok=False)
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    cfg.log_dir = args.out_log_dir

    from datasets.driving_dataset import DrivingDataset

    t0 = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    for m in trainer.models.values():
        for p in m.parameters():
            p.requires_grad_(False)
    dc = trainer.models["Background"]._features_dc
    n = dc.shape[0]
    assert 0 < first < n, (first, n)
    dc.requires_grad_(True)
    # the training-mode renderer keeps the screen-space gradient of the means (densification statistics): let it flow
    # to the means, which no optimizer steps
    trainer.models["Background"]._means.requires_grad_(True)
    opt = torch.optim.Adam([dc], lr=args.lr, eps=1e-15)
    dc0 = dc.detach()[first:].clone()
    print(f"[fit_surfel_colours] fitting the colours of Gaussians {first:,}..{n - 1:,} of {n:,} for {args.steps} steps", flush=True)
    trainer.set_train()
    running = {}
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        ii, ci = dataset.train_image_set.next(trainer._get_downscale_factor())
        ii, ci = to_device(ii, device), to_device(ci, device)
        outputs = trainer(ii, ci)
        losses = trainer.compute_losses(outputs=outputs, image_infos=ii, cam_infos=ci)
        loss = losses["rgb_loss"] + losses["ssim_loss"]
        if args.reg > 0:
            loss = loss + args.reg * ((dc[first:] - dc0) ** 2).mean()
        loss.backward()
        dc.grad[:first] = 0
        opt.step()
        with torch.no_grad():
            psnr = trainer.compute_metrics(outputs=outputs, image_infos=ii)["psnr"].item()
        for k, v in (("rgb", losses["rgb_loss"].item()), ("ssim", losses["ssim_loss"].item()), ("psnr", psnr)):
            running[k] = running.get(k, 0.0) + v
        if step % args.print_every == 0:
            print(f"[fit_surfel_colours] step {step}  " + "  ".join(f"{k} {v / args.print_every:.4f}" for k, v in running.items())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
            running = {}
    dc.requires_grad_(False)
    trainer.save_checkpoint(log_dir=args.out_log_dir, save_only_model=True, is_final=True)
    if "view_gate" in cfg:  # the gate travels with the run (the Gaussian set is unchanged)
        shutil.copy(cfg.view_gate, os.path.join(args.out_log_dir, "view_gate.pt"))
        cfg.view_gate = os.path.join(args.out_log_dir, "view_gate.pt")
    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({**src_meta, "exp": args.exp, "colour_fit": {"source": os.path.abspath(args.log_dir), "first": first, **vars(args)},
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[fit_surfel_colours] -> {args.out_log_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
