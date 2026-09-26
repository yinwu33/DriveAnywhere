"""Held-out FRONT frames of a trained run, the same way for single- and multi-camera runs (DECISIONS Q).

drivestudio's test metrics average over every camera of the run, so a 3-camera run (the oracle upper bound U3) and a
FRONT-only run are not comparable from metrics.json. This renders only the FRONT image of every test timestep
(dashrecon.gen.views.front_image_index), in eval mode through the trainer's normal forward as drivestudio's
render_images does (test frames then use the test-set behaviour of Affine), and compares with the run's own ground
truth image (its loader's undistortion):
    PSNR, SSIM (mean of the SSIM map), LPIPS (trainer.lpips, AlexNet as drivestudio), each over the whole image and
    PSNR also over non-sky pixels (the run's sky mask);
    sharpness: variance of the Laplacian of the grey render and of the ground truth, and their ratio.
Outputs <log_dir>/front_heldout/metrics.json (means and per frame) and <t:03d>.jpg (ground truth | render) for
--example_frames.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/eval_front_heldout.py --log_dir results/U3/val056 --example_frames 50 100 150
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.gen.views import front_image_index  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def sharpness(rgb: np.ndarray) -> float:
    grey = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--example_frames", type=int, nargs="*", default=[])
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    test_ts = [int(t) for t in dataset.test_timesteps]
    assert test_ts, "the run has no test timesteps"
    out_dir = os.path.join(args.log_dir, "front_heldout")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for k in test_ts:
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, k), trainer._get_downscale_factor())
            ii, ci = to_device(ii, device), to_device(ci, device)
            assert int(ci["cam_id"].flatten()[0]) == 0, ci["cam_id"]
            out = trainer(ii, ci)
            render = out["rgb"].clamp(0, 1).cpu().numpy()
            gt = ii["pixels"].cpu().numpy()
            sky = ii["sky_masks"].cpu().numpy() > 0.5
            mse = ((render - gt) ** 2).mean()
            t = dataset.start_timestep + k
            row = {"frame": t, "psnr": float(-10 * np.log10(mse)),
                   "psnr_nonsky": float(-10 * np.log10(((render - gt) ** 2)[~sky].mean())),
                   "ssim": float(structural_similarity(render, gt, data_range=1.0, channel_axis=-1)),
                   "lpips": float(trainer.lpips(out["rgb"].clamp(0, 1).permute(2, 0, 1)[None], ii["pixels"].permute(2, 0, 1)[None])),
                   "sharp_render": sharpness(render), "sharp_gt": sharpness(gt)}
            row["sharp_ratio"] = row["sharp_render"] / row["sharp_gt"]
            rows.append(row)
            if t in args.example_frames:
                Image.fromarray((np.concatenate([gt, render], 1) * 255).round().astype(np.uint8)).save(
                    os.path.join(out_dir, f"{t:03d}.jpg"), quality=90)
    keys = ("psnr", "psnr_nonsky", "ssim", "lpips", "sharp_render", "sharp_gt", "sharp_ratio")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"log_dir": args.log_dir, "frames": len(rows), "resolution": list(render.shape[:2]), "summary": summary,
                   "per_frame": rows, "dashrecon_commit": commit}, f, indent=2)
    print(f"[eval_front_heldout] {args.log_dir}: {len(rows)} FRONT test frames at {render.shape[1]}x{render.shape[0]}: "
          + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()), flush=True)


if __name__ == "__main__":
    main()
