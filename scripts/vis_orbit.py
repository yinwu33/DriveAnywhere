"""Orbit videos of trained runs at fixed positions (free-view check, D-F1 in docs/EXPERIMENTS.md).

At each --frames entry the camera stands at that frame's trained FRONT camera (CamPose refined) and
    1. turns a full circle at the camera's own height (yaw 0 -> 360),
    2. rises by --rise scene units while pitching down to --pitch degrees,
    3. turns another full circle from up there,
so one clip shows every direction from the road and from above. Runs sit side by side with their labels; Phase 9 runs
render through their view gate, everything over the Sky model (dashrecon.gen.views.render_moved, as vis_freeview.py).

Outputs in --out_dir: <scene>_orbit_f<t:03d>.mp4, <scene>_orbit_f<t:03d>_<yaw>.jpg (stills at yaw 0 / 90 / 180 / 270 of
the first circle) and <scene>_orbit.json (runs, frames, path, commit).

Example (main venv):
    .venvs/main/bin/python scripts/vis_orbit.py --log_dirs results/E5f/val056 results/E30/val056/model \
        --labels E5f "E30 (gen)" --frames 60 104 150 --cell_width 400 --fps 15 --out_dir results/_vis_orbit
"""
import argparse
import json
import os
import sys

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.novel import attach_view_gate, build_trainer, view_gate_path  # noqa: E402
from dashrecon.gen.views import ViewMove, render_moved  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from vis_freeview import labelled, to_uint8  # noqa: E402


def orbit_path(steps: int, rise_steps: int, rise: float, pitch: float) -> list:
    """ViewMoves of the orbit: circle at height 0, rise while pitching down, circle from above."""
    path = [ViewMove(yaw=360.0 * i / steps) for i in range(steps)]
    path += [ViewMove(up=rise * i / rise_steps, pitch=pitch * i / rise_steps) for i in range(1, rise_steps + 1)]
    path += [ViewMove(up=rise, pitch=pitch, yaw=360.0 * i / steps) for i in range(1, steps + 1)]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=120, help="frames per full circle")
    parser.add_argument("--rise_steps", type=int, default=20)
    parser.add_argument("--rise", type=float, default=2.0, help="scene units")
    parser.add_argument("--pitch", type=float, default=-15.0, help="degrees (negative looks down)")
    args = parser.parse_args()
    assert len(args.labels) == len(args.log_dirs), "one label per run"
    device = torch.device("cuda")
    from datasets.driving_dataset import DrivingDataset

    cfgs = [OmegaConf.load(os.path.join(d, "config.yaml")) for d in args.log_dirs]
    for c in cfgs[1:]:
        assert OmegaConf.to_container(c.data) == OmegaConf.to_container(cfgs[0].data), "runs must share their data config"
    dataset = DrivingDataset(data_cfg=cfgs[0].data)
    trainers = []
    for d, c in zip(args.log_dirs, cfgs):
        t = build_trainer(c, dataset, device)
        t.resume_from_checkpoint(ckpt_path=os.path.join(d, "checkpoint_final.pth"), load_only_model=True)
        if view_gate_path(c) is not None:
            attach_view_gate(t, view_gate_path(c))
        t.set_eval()
        trainers.append(t)
    scene = f"val{int(cfgs[0].data.scene_idx):03d}"
    path = orbit_path(args.steps, args.rise_steps, args.rise, args.pitch)
    stills = {round(args.steps * q / 4): int(360 * q / 4) for q in range(4)}
    os.makedirs(args.out_dir, exist_ok=True)
    files = []
    for t in args.frames:
        k = t - dataset.start_timestep
        assert 0 <= k < dataset.num_img_timesteps, t
        name = f"{scene}_orbit_f{t:03d}"
        writer = imageio.get_writer(os.path.join(args.out_dir, f"{name}.mp4"), mode="I", fps=args.fps, macro_block_size=8)
        for i, move in enumerate(path):
            cells = []
            for tr, lab in zip(trainers, args.labels):
                with torch.no_grad():
                    out = render_moved(tr, dataset, k, move, device)[0]
                cells.append(labelled(to_uint8(out["rgb"]), lab, args.cell_width))
            row = np.concatenate(cells, 1)
            writer.append_data(row)
            if i in stills:
                still = f"{name}_{stills[i]:03d}.jpg"
                Image.fromarray(row).save(os.path.join(args.out_dir, still), quality=88)
                files.append(still)
        writer.close()
        files.append(f"{name}.mp4")
        print(f"[vis_orbit] {scene} frame {t}: {len(path)} views -> {name}.mp4", flush=True)
    with open(os.path.join(args.out_dir, f"{scene}_orbit.json"), "w") as f:
        json.dump({"scene_id": scene, "runs": dict(zip(args.labels, args.log_dirs)), "frames": args.frames,
                   "path": {"steps": args.steps, "rise_steps": args.rise_steps, "rise": args.rise, "pitch": args.pitch},
                   "files": files, "generative": True, "cell_width": args.cell_width, "dashrecon_commit": git_commit()},
                  f, indent=2)


if __name__ == "__main__":
    main()
