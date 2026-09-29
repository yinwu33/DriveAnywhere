"""E26 (docs/EXPERIMENTS.md): Fixer as a render-time enhancement (the "+" of Difix3D+) on saved renders.

Reads <src>/renders/*.png (e.g. the realism2/ side renders of eval_cross_camera.py --save_renders), runs NVIDIA Fixer
(single step at --timestep, dashrecon.gen.fixer_client) on each frame independently and writes <out>/renders/ with the
same names; <out>/refs links to <src>/refs so eval_realism.py can compare the enhanced renders. Frames are padded to a
multiple of 16 (edge replicate) for Fixer and cropped back. Per-frame enhancement is not tied to the 3D model, so
consecutive frames can disagree; this measures the ceiling of what render-time Fixer adds, not a 3D result.

Example (main venv):
    .venvs/main/bin/python scripts/fix_renders.py --src results/E23/val056/model/realism2 \
        --out results/E26/val056/realism2 --timestep 250 --batch 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.fixer_client import FIXER_REPO_ID, FIXER_REVISION, FixerClient  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--cuda_home", default="/usr/local/cuda-12.1")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "renders"), exist_ok=False)
    os.symlink(os.path.abspath(os.path.join(args.src, "refs")), os.path.join(args.out, "refs"))
    names = sorted(os.listdir(os.path.join(args.src, "renders")))
    fixer = FixerClient(args.timestep, os.path.join(args.out, "fixer_work"), args.cuda_home)
    t0 = time.time()
    for b in range(0, len(names), args.batch):
        part = names[b:b + args.batch]
        imgs = [np.asarray(Image.open(os.path.join(args.src, "renders", n)).convert("RGB")).astype(np.float32) / 255.0 for n in part]
        h, w = imgs[0].shape[:2]
        assert all(i.shape[:2] == (h, w) for i in imgs), "renders of one size per batch"
        ph, pw = -h % 16, -w % 16
        padded = np.stack([np.pad(i, ((0, ph), (0, pw), (0, 0)), mode="edge") for i in imgs])
        fixed = fixer.refine(padded)[:, :h, :w]
        for n, f in zip(part, fixed):
            Image.fromarray((np.clip(f, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(args.out, "renders", n))
    fixer.close()
    with open(os.path.join(args.out, "fix_renders.json"), "w") as f:
        json.dump({"src": args.src, "frames": len(names), "timestep": args.timestep, "generator": FIXER_REPO_ID,
                   "generator_revision": FIXER_REVISION, "generative": True, "kind": "render-time per-frame enhancement",
                   "runtime_s": time.time() - t0, "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[fix_renders] {len(names)} frames of {args.src} -> {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
