"""E43 (docs/EXPERIMENTS.md): per-instance vehicle masks of the FRONT images (Grounded-SAM-2, one SAM 2.1 mask per box).

The Phase 4 dynamic mask is the union of all boxes' masks, so parked cars in a row ahead merge into one region; the
vehicle assets need one mask per vehicle. Reads only the FRONT images. Writes <out_root>/<scene_id>/inst-gsam2_<image_tag>/:
frames.txt, inst/<t:06d>.png (uint16 instance map, 0 = none), instances.json (per frame: id, box, score, label, area),
meta.json.

Example (masks venv):
    HF_HUB_OFFLINE=1 .venvs/masks/bin/python scripts/run_instance_masks.py --scene_id val056 \\
        --processed_root data/dashrecon/_undistorted/calib-glomap --out_root data/dashrecon --image_tag img-glomap \\
        --detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large --prompts car truck bus van \\
        --box_threshold 0.3 --text_threshold 0.25 --nms_iou 0.6
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
from dashrecon import io  # noqa: E402
from dashrecon.masks.sam_text import GroundedSam2Backend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import front_image_paths, get_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--processed_root", "--out_root", "--image_tag", "--detector_id", "--sam_id"):
        parser.add_argument(a, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--box_threshold", type=float, required=True)
    parser.add_argument("--text_threshold", type=float, required=True)
    parser.add_argument("--nms_iou", type=float, required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    scene = get_scene(args.scene_id)
    out_dir = io.scene_dir(args.out_root, args.scene_id, f"inst-gsam2_{args.image_tag}")
    os.makedirs(os.path.join(out_dir, "inst"), exist_ok=True)
    backend = GroundedSam2Backend(args.detector_id, args.sam_id, tuple(args.prompts), args.box_threshold, args.text_threshold, 0)
    frames, per_frame = [], []
    for t, path in zip(*front_image_paths(os.path.join(args.processed_root, f"{scene.scene_idx:03d}"), scene.start_timestep, scene.end_timestep)):
        inst, info = backend.predict_instances(Image.open(path).convert("RGB"), args.nms_iou)
        Image.fromarray(inst).save(os.path.join(out_dir, "inst", f"{int(t):06d}.png"))
        frames.append(int(t))
        per_frame.append({"frame": int(t), "instances": info})
    io.write_frames(out_dir, np.array(frames))
    with open(os.path.join(out_dir, "instances.json"), "w") as f:
        json.dump(per_frame, f)
    io.write_meta(out_dir, {"scene_id": args.scene_id, "params": vars(args), "backend": backend.meta(), "uses_oracle": False,
                            "oracle_note": "FRONT images only", "runtime_s": time.time() - t0,
                            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "dashrecon_commit": commit})
    print(f"[run_instance_masks] {args.scene_id}: {len(frames)} frames, {np.mean([len(p['instances']) for p in per_frame]):.1f} instances/frame "
          f"-> {out_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
