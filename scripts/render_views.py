"""Phase 9 step 1 (DECISIONS D16-D18): render a free-viewpoint trajectory of a trained run and its hole masks.

Every FRONT frame's camera (as trained, CamPose refined) is moved by --move (dashrecon.gen.views.ViewMove, e.g.
"right=1.5,yaw=15") and rendered: RGB with the sky model (rays recomputed for the moved camera), Gaussian opacity
and z-depth. Holes are the pixels no training view saw at comparable resolution (dashrecon.gen.views, visibility
test against every --obs_stride-th training frame), excluding sky. With --memory_dirs (the trajectories filled in
earlier rounds, scripts/fill_views.py) every --obs_stride-th view of those, rendered by the current model, also
counts as an observer: the 3D memory that keeps later rounds from generating the same region again.

Output <out_dir>/: rgb/<k:03d>.png, mask/<k:03d>.png (255 = hole), depth/<k:03d>.npy (float16 z-depth),
count/<k:03d>.npy (float16 number of training views that saw the pixel), cams.json (per view: frame, c2w,
drivestudio intrinsics; parameters, hole fractions, commit).

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/render_views.py --log_dir results/E5c/val039 --move right=1.5,yaw=15 \
        --out_dir results/E8/val039/views/r0_right1.5_yaw15
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.gen.views import ViewMove, hole_mask, memory_observers, render_moved, training_observers  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--move", required=True, help='e.g. "right=1.5,yaw=15" (keys: right, up, forward, yaw, pitch)')
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--obs_stride", type=int, default=2)
    parser.add_argument("--memory_dirs", nargs="*", default=[], help="views dirs filled in earlier rounds")
    parser.add_argument("--depth_tol", type=float, default=0.10)
    parser.add_argument("--res_ratio", type=float, default=3.0)
    parser.add_argument("--sky_alpha", type=float, default=0.5)
    parser.add_argument("--sky_elev_deg", type=float, default=5.0)
    parser.add_argument("--open_px", type=int, default=9)
    parser.add_argument("--min_area", type=int, default=3000)
    parser.add_argument("--dilate_px", type=int, default=7)
    args = parser.parse_args()
    commit = git_commit()
    move = ViewMove.parse(args.move)
    device = torch.device("cuda")

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    observers = training_observers(trainer, dataset, args.obs_stride, device)
    n_train_obs = len(observers)
    observers += memory_observers(trainer, dataset, args.memory_dirs, args.obs_stride, device)

    for sub in ("rgb", "mask", "depth", "count"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    cams, hole_frac = [], []
    with torch.no_grad():
        for k in range(len(dataset.full_image_set)):
            out, ii, ci, c2w = render_moved(trainer, dataset, k, move, device)
            h, w = int(ci["height"]), int(ci["width"])
            depth = out["depth"][..., 0]
            hole, count, _ = hole_mask(out, ii["viewdirs"], ci["intrinsics"], c2w, observers, args.depth_tol, args.res_ratio,
                                       args.sky_alpha, args.sky_elev_deg, args.open_px, args.min_area, args.dilate_px)
            rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(args.out_dir, "rgb", f"{k:03d}.png"))
            Image.fromarray((hole * 255).astype(np.uint8)).save(os.path.join(args.out_dir, "mask", f"{k:03d}.png"))
            np.save(os.path.join(args.out_dir, "depth", f"{k:03d}.npy"), depth.cpu().numpy().astype(np.float16))
            np.save(os.path.join(args.out_dir, "count", f"{k:03d}.npy"), count.astype(np.float16))
            cams.append({"frame": int(dataset.start_timestep + k), "c2w": c2w.cpu().numpy().tolist(), "K": ci["intrinsics"].cpu().numpy().tolist(),
                         "hw": [h, w]})
            hole_frac.append(float(hole.mean()))
    with open(os.path.join(args.out_dir, "cams.json"), "w") as f:
        json.dump({"log_dir": args.log_dir, "move": vars(move), "params": vars(args), "cams": cams, "hole_frac": hole_frac,
                   "observers": len(observers), "train_observers": n_train_obs, "dashrecon_commit": commit}, f)
    print(f"[render_views] {args.log_dir} {args.move}: {len(cams)} views, hole fraction mean {np.mean(hole_frac):.3f} "
          f"max {np.max(hole_frac):.3f} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
