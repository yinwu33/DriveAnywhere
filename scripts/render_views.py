"""Phase 9 step 1 (DECISIONS D16-D18): render a free-viewpoint trajectory of a trained run and its hole masks.

Every FRONT frame's camera (as trained, CamPose refined) is moved by --move (dashrecon.gen.views.ViewMove, e.g.
"right=1.5,yaw=15") and rendered: RGB with the sky model (rays recomputed for the moved camera), Gaussian opacity
and z-depth. Holes are the pixels no training view saw at comparable resolution (dashrecon.gen.views, visibility
test against every --obs_stride-th training frame), excluding sky.

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
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, backproject, clean_mask, moved_c2w, seen_count  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--move", required=True, help='e.g. "right=1.5,yaw=15" (keys: right, up, forward, yaw, pitch)')
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--obs_stride", type=int, default=2)
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
    from datasets.base.pixel_source import get_rays
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    observers = []
    with torch.no_grad():
        for j in range(0, len(dataset.train_image_set), args.obs_stride):
            ii, ci = dataset.train_image_set.get_image(j, 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            out = trainer(ii, ci)
            observers.append({"w2c": torch.linalg.inv(trainer._last_cam.camtoworlds), "K": ci["intrinsics"], "depth": out["depth"][..., 0].half()})

    for sub in ("rgb", "mask", "depth", "count"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    full = dataset.full_image_set
    cams, hole_frac = [], []
    with torch.no_grad():
        for k in range(len(full)):
            ii, ci = full.get_image(k, 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            c2w = moved_c2w(refined_c2w(trainer, ii, ci), move)
            h, w = int(ci["height"]), int(ci["width"])
            x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="xy")
            origins, viewdirs, dnorm = get_rays(x.flatten(), y.flatten(), c2w, ci["intrinsics"])
            ii["origins"], ii["viewdirs"], ii["direction_norm"] = origins.reshape(h, w, 3), viewdirs.reshape(h, w, 3), dnorm.reshape(h, w, 1)
            ci["camera_to_world"] = c2w
            out = trainer(ii, ci, novel_view=True)
            depth = out["depth"][..., 0]
            count = seen_count(backproject(depth, ci["intrinsics"], c2w), depth.reshape(-1), observers, args.depth_tol, args.res_ratio)
            count = count.reshape(h, w).cpu().numpy()
            alpha = out["opacity"][..., 0].cpu().numpy()
            elev = np.degrees(np.arcsin(np.clip(ii["viewdirs"][..., 2].cpu().numpy(), -1.0, 1.0)))
            sky = (alpha < args.sky_alpha) & (elev >= args.sky_elev_deg)
            hole = clean_mask((count < 1) & ~sky, args.open_px, args.min_area, args.dilate_px) & ~sky
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
                   "observers": len(observers), "dashrecon_commit": commit}, f)
    print(f"[render_views] {args.log_dir} {args.move}: {len(cams)} views, hole fraction mean {np.mean(hole_frac):.3f} "
          f"max {np.max(hole_frac):.3f} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
