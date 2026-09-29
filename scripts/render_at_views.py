"""Render a trained run at every camera of a generated trajectory (D-E2 in docs/EXPERIMENTS.md).

Writes <out_dir>/rgb/<k:03d>.png (the run's novel-view render, through its view gate when its config has one) and a
link to the trajectory's cams.json, so link_generated_for_realism.py --source rgb can pair them with real images.

Example (main venv):
    .venvs/main/bin/python scripts/render_at_views.py --log_dir results/E14/val056/model \
        --views_dir results/E14/val056/views/r2 --out_dir results/_diagnostics/DE2/val056/E14_final_r2
"""
import argparse
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import attach_view_gate, build_trainer, to_device, view_gate_path  # noqa: E402
from dashrecon.gen.views import front_image_index, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_dir, "rgb"), exist_ok=False)
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    if view_gate_path(cfg) is not None:
        attach_view_gate(trainer, view_gate_path(cfg))
    trainer.set_eval()
    cams = json.load(open(os.path.join(args.views_dir, "cams.json")))["cams"]
    with torch.no_grad():
        for k, c in enumerate(cams):
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, c["frame"] - dataset.start_timestep), 1)
            out = render_at(trainer, to_device(ii, device), to_device(ci, device), torch.tensor(c["c2w"], dtype=torch.float32, device=device),
                            torch.tensor(c["K"], dtype=torch.float32, device=device), tuple(c["hw"]))
            Image.fromarray((out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(os.path.join(args.out_dir, "rgb", f"{k:03d}.png"))
    os.symlink(os.path.abspath(os.path.join(args.views_dir, "cams.json")), os.path.join(args.out_dir, "cams.json"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"log_dir": args.log_dir, "views_dir": args.views_dir, "view_gate": view_gate_path(cfg), "views": len(cams),
                   "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[render_at_views] {len(cams)} views of {args.views_dir} rendered by {args.log_dir}", flush=True)


if __name__ == "__main__":
    main()
