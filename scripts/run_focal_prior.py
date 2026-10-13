"""E46 (docs/EXPERIMENTS.md): a focal-length prior of the FRONT camera from single images (MoGe v1), FRONT images only.
Runs in .venvs/sam3d (MoGe at the commit SAM 3D was released with).

Under forward motion the self-calibration's focal is weakly constrained (val056: GLOMAP 14 % short); D-G3 found the
median MoGe v1 focal over 30 frames within 2.5 % of GT on average over the five dev scenes (GLOMAP 6.4 %). --frames
original (distorted, as recorded) FRONT frames spread over the drive; the median of the focal recovered from MoGe's
normalised intrinsics is the prior. GT is not read.

Output: <out_root>/<scene_id>/focal-moge/focal.json (fx, per-frame values, model, image size).

Example (sam3d venv):
    CUDA_HOME=$HOME/cuda-12.6 .venvs/sam3d/bin/python scripts/run_focal_prior.py --scene_id val056 \\
        --processed_root data/waymo/processed/validation --out_root data/dashrecon --frames 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--model_id", default="Ruicheng/moge-vitl")
    args = parser.parse_args()
    commit = git_commit()
    from moge.model.v1 import MoGeModel

    t0 = time.time()
    dev = torch.device("cuda")
    model = MoGeModel.from_pretrained(args.model_id).to(dev).eval()
    img_dir = os.path.join(args.processed_root, f"{get_scene(args.scene_id).scene_idx:03d}", "images")
    names = sorted(n for n in os.listdir(img_dir) if n.endswith("_0.jpg"))
    pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, args.frames).round()]
    fx = []
    for n in pick:
        pil = Image.open(os.path.join(img_dir, n)).convert("RGB")
        w, h = pil.size
        x = torch.tensor(np.asarray(pil), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            fx.append(float(model.infer(x)["intrinsics"][0, 0]) * w)
    out_dir = os.path.join(args.out_root, args.scene_id, "focal-moge")
    os.makedirs(out_dir, exist_ok=True)
    rec = {"scene_id": args.scene_id, "fx": float(np.median(fx)), "iqr": [float(np.percentile(fx, 25)), float(np.percentile(fx, 75))],
           "per_frame": dict(zip(pick, fx)), "image_wh": [w, h], "model_id": args.model_id, "uses_oracle": False,
           "oracle_note": "original FRONT images only", "runtime_s": time.time() - t0, "dashrecon_commit": commit}
    with open(os.path.join(out_dir, "focal.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[run_focal_prior] {args.scene_id}: fx {rec['fx']:.1f} (IQR {rec['iqr'][0]:.0f}-{rec['iqr'][1]:.0f}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
