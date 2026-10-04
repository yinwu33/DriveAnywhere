"""E41 (docs/EXPERIMENTS.md): how well scripts/refine_dynamic_masks.py separates moving from parked objects. EVALUATION
CODE: reads the GT moving-object masks of the processed Waymo scene (dynamic_masks/all: boxes of objects moving in that
frame, drivestudio preprocessing), never used in reconstruction.

Per frame (FRONT): the GT moving pixels covered by the Grounded-SAM-2 mask and by the refined (moving-only) mask, and the
pixels the refinement released as static that are GT moving. The GT masks are on the original distorted image and ours
on the undistorted one (DECISIONS D13); both are 1920 x 1280 and FRONT distortion is small, so the comparison is per
pixel after a --dilate_px dilation of our masks.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/eval_dynamic_masks.py --scene_id val041 --gt_root data/waymo/processed/validation \\
        --mask_dir data/dashrecon/val041/mask-gsam2_sky-segformer_img-glomap --refined_dir data/dashrecon/val041/mask-gsam2mv_sky-segformer_img-glomap
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--refined_dir", required=True)
    parser.add_argument("--dilate_px", type=int, default=10)
    args = parser.parse_args()
    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.refined_dir)
    gt_px = cov_src = cov_ref = released = released_moving = 0
    for t in frames:
        t = int(t)
        gt = np.asarray(Image.open(os.path.join(args.gt_root, f"{scene.scene_idx:03d}", "dynamic_masks", "all", f"{t:03d}_0.png"))) > 0
        src, ref = io.read_mask(args.mask_dir, "dynamic", t), io.read_mask(args.refined_dir, "dynamic", t)
        assert gt.shape == src.shape, (gt.shape, src.shape)
        st = np.ones((3, 3), bool)
        src_d, ref_d = ndimage.binary_dilation(src, st, args.dilate_px), ndimage.binary_dilation(ref, st, args.dilate_px)
        rel = src & ~ref
        gt_px += int(gt.sum())
        cov_src += int((gt & src_d).sum())
        cov_ref += int((gt & ref_d).sum())
        released += int(rel.sum())
        released_moving += int((rel & ndimage.binary_dilation(gt, st, args.dilate_px)).sum())
    out = {"scene_id": args.scene_id, "frames": len(frames), "gt_moving_px": gt_px,
           "gt_moving_covered_by_gsam2": cov_src / max(1, gt_px), "gt_moving_covered_by_refined": cov_ref / max(1, gt_px),
           "released_px": released, "released_on_gt_moving": released_moving / max(1, released), "reads_gt": True}
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.refined_dir, "eval_vs_gt_moving.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
