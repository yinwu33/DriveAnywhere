"""Sky masks in drivestudio's layout for a processed scene (runs in .venvs/masks; oracle upper bound, DECISIONS Q).

drivestudio's default pixel source needs <scene>/sky_masks/<t:03d>_<cam>.png (> 0 = sky) for every camera it trains
on; upstream makes them with the mmcv-based datasets/tools/extract_masks.py, which DECISIONS D6 replaced. This
writes them with the same SegFormer-B5 Cityscapes backend as Phase 4 (dashrecon.masks.segformer, sky = class 10) on
the processed (distorted) images; drivestudio undistorts images and masks together when it loads them.

Example:
    .venvs/masks/bin/python scripts/sky_masks_processed.py --scene_dir data/waymo/processed/validation/056 --cams 0 1 2 \
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_hw 1024 1536
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.masks.segformer import SegformerSkyRoadBackend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--cams", type=int, nargs="+", required=True)
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--seg_input_hw", type=int, nargs=2, required=True)
    args = parser.parse_args()
    commit = git_commit()
    backend = SegformerSkyRoadBackend(args.seg_model_id, tuple(args.seg_input_hw))
    out = os.path.join(args.scene_dir, "sky_masks")
    os.makedirs(out, exist_ok=True)
    t0, fracs = time.time(), {}
    for cam in args.cams:
        names = sorted(f for f in os.listdir(os.path.join(args.scene_dir, "images")) if f.endswith(f"_{cam}.jpg"))
        assert names, f"no images for camera {cam}"
        sky = []
        for name in names:
            masks, _ = backend.predict(Image.open(os.path.join(args.scene_dir, "images", name)).convert("RGB"))
            Image.fromarray((masks["sky"] * 255).astype(np.uint8)).save(os.path.join(out, name.replace(".jpg", ".png")))
            sky.append(float(masks["sky"].mean()))
        fracs[cam] = {"frames": len(names), "sky_fraction_mean": float(np.mean(sky))}
        print(f"[sky_masks_processed] camera {cam}: {len(names)} masks, sky {np.mean(sky):.3f}", flush=True)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({**backend.meta(), "cams": fracs, "seconds": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)


if __name__ == "__main__":
    main()
