"""Phase 9 step 1 (DECISIONS D16-D18): render a free-viewpoint trajectory of a trained run and its hole masks.

Every --frame_stride-th FRONT frame's camera (as trained, CamPose refined) is moved by --move (dashrecon.gen.views.ViewMove, e.g.
"right=1.5,yaw=15") and rendered: RGB with the sky model (rays recomputed for the moved camera), Gaussian opacity
and z-depth. Holes are the pixels no training view saw at comparable resolution (dashrecon.gen.views, visibility
test against every --obs_stride-th training frame), excluding sky. --ramp_frames n scales the move linearly from
zero at the first view to full at view n (a camera that starts at the real FRONT pose and turns / moves away, as
GEN3C needs its first frame to be a real image). --render_hw H W renders at another size: the intrinsics are scaled
to width W and the image centre-cropped to height H (GEN3C: 704 x 1280). With --memory_dirs (the trajectories filled in
earlier rounds, scripts/fill_views.py) every --obs_stride-th view of those, rendered by the current model, also
counts as an observer: the 3D memory that keeps later rounds from generating the same region again.

Output <out_dir>/ (k = view number 0.. in frame order; cams.json gives each view's frame): rgb/<k:03d>.png,
mask/<k:03d>.png (255 = hole), depth/<k:03d>.npy (float16 z-depth), count/<k:03d>.npy (float16 number of training
views that saw the pixel), cams.json (per view: frame, c2w, drivestudio intrinsics; parameters, hole fractions,
commit).

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
from dashrecon.gen.novel import refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import (ViewMove, front_image_index, hole_mask, memory_observers, moved_c2w, render_at,  # noqa: E402
                                 training_observers)
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--move", required=True, help='e.g. "right=1.5,yaw=15" (keys: right, up, forward, yaw, pitch)')
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frame_stride", type=int, default=1, help="render every n-th frame (2 halves the Wan fill time)")
    parser.add_argument("--ramp_frames", type=int, default=0, help="0 = full move from the first view")
    parser.add_argument("--render_hw", type=int, nargs=2, help="H W; default: the training size")
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
        for v, k in enumerate(range(0, dataset.num_img_timesteps, args.frame_stride)):
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, k), 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            f = min(1.0, v / args.ramp_frames) if args.ramp_frames > 0 else 1.0
            c2w = moved_c2w(refined_c2w(trainer, ii, ci), ViewMove(**{key: f * val for key, val in vars(move).items()}))
            h0, w0 = int(ci["height"]), int(ci["width"])
            h, w = (h0, w0) if args.render_hw is None else tuple(args.render_hw)
            kk = ci["intrinsics"].clone()
            kk[:2] *= w / w0
            kk[1, 2] -= (h0 * w / w0 - h) / 2
            assert h <= h0 * w / w0 + 1e-6, f"--render_hw {h} x {w} is taller than the scaled image {h0 * w / w0:.1f}"
            out = render_at(trainer, ii, ci, c2w, kk, (h, w))
            depth = out["depth"][..., 0]
            hole, count, _ = hole_mask(out, ii["viewdirs"], kk, c2w, observers, args.depth_tol, args.res_ratio,
                                       args.sky_alpha, args.sky_elev_deg, args.open_px, args.min_area, args.dilate_px)
            rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(args.out_dir, "rgb", f"{v:03d}.png"))
            Image.fromarray((hole * 255).astype(np.uint8)).save(os.path.join(args.out_dir, "mask", f"{v:03d}.png"))
            np.save(os.path.join(args.out_dir, "depth", f"{v:03d}.npy"), depth.cpu().numpy().astype(np.float16))
            np.save(os.path.join(args.out_dir, "count", f"{v:03d}.npy"), count.astype(np.float16))
            cams.append({"frame": int(dataset.start_timestep + k), "c2w": c2w.cpu().numpy().tolist(), "K": kk.cpu().numpy().tolist(), "ramp": f,
                         "hw": [h, w]})
            hole_frac.append(float(hole.mean()))
    with open(os.path.join(args.out_dir, "cams.json"), "w") as f:
        json.dump({"log_dir": args.log_dir, "move": vars(move), "params": vars(args), "cams": cams, "hole_frac": hole_frac,
                   "observers": len(observers), "train_observers": n_train_obs, "dashrecon_commit": commit}, f)
    print(f"[render_views] {args.log_dir} {args.move}: {len(cams)} views, hole fraction mean {np.mean(hole_frac):.3f} "
          f"max {np.max(hole_frac):.3f} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
