"""E43 step 2 (docs/EXPERIMENTS.md): one SAM 3D Objects asset per parked vehicle (runs in .venvs/sam3d, envs/setup_sam3d.sh).

SAM 3D Objects reconstructs an object from one real image and its mask, trained for occluded, small, in-the-wild
objects (TRELLIS, trained on object renders, turned these dashcam crops into flat cards: E43 notes). Input per vehicle
of --instances (scripts/vehicle_instances.py): the full FRONT frame of its first crop (largest unoccluded detection)
and its instance mask (--inst_dir). Output in --out_dir: asset_<id>.ply (3D Gaussians in SAM 3D's object frame,
standard 3DGS PLY), asset_<id>.json (frame, seed, the predicted object-to-camera rotation / translation / scale),
which scripts/vehicle_insert.py fits to the scene. Generated content. GT is not read.

Example (sam3d venv):
    CUDA_HOME=$HOME/cuda-12.6 .venvs/sam3d/bin/python scripts/vehicle_assets_sam3d.py --scene_id val056 \\
        --instances results/E43/val056/instances/instances.json --inst_dir data/dashrecon/val056/inst-gsam2_img-glomap \\
        --processed_root data/dashrecon/_undistorted/calib-glomap --sam3d_root .venvs/src/sam-3d-objects \\
        --out_dir results/E43/val056/assets_sam3d --seed 42
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--scene_id", "--instances", "--inst_dir", "--processed_root", "--sam3d_root", "--out_dir"):
        parser.add_argument(a, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ids", type=int, nargs="*", help="only these vehicles")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.abspath(args.sam3d_root), "notebook"))
    import torch
    from inference import Inference

    meta = json.load(open(args.instances))
    scene_idx = int(args.scene_id[3:])
    t0 = time.time()
    model = Inference(os.path.join(os.path.abspath(args.sam3d_root), "checkpoints", "hf", "pipeline.yaml"), compile=False)
    print(f"[vehicle_assets_sam3d] model loaded ({time.time() - t0:.0f} s)", flush=True)
    for inst in meta["instances"]:
        if args.ids and inst["id"] not in args.ids or not inst["crops"]:
            continue
        crop = inst["crops"][0]
        img = np.asarray(Image.open(os.path.join(args.processed_root, f"{scene_idx:03d}", "images", f"{crop['frame']:03d}_0.jpg")).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(args.inst_dir, "inst", f"{crop['frame']:06d}.png"))) == crop["inst"]
        t1 = time.time()
        out = model(img, mask, seed=args.seed)
        g = out["gaussian"][0]
        g.save_ply(os.path.join(args.out_dir, f"asset_{inst['id']}.ply"))
        pose = {k: out[k].detach().cpu().numpy().tolist() for k in ("rotation", "translation", "scale") if k in out}
        with open(os.path.join(args.out_dir, f"asset_{inst['id']}.json"), "w") as f:
            json.dump({"vehicle": inst["id"], "frame": crop["frame"], "inst": crop["inst"], "seed": args.seed, "pose_l2c": pose,
                       "gaussians": int(g.get_xyz.shape[0]), "generative": True, "runtime_s": time.time() - t1}, f, indent=1)
        print(f"[vehicle_assets_sam3d] vehicle {inst['id']} (frame {crop['frame']}): {g.get_xyz.shape[0]:,} Gaussians "
              f"({time.time() - t1:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
