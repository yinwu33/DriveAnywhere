"""Free-viewpoint comparison of trained runs (Phase 9, DECISIONS D17): the same camera moves rendered by each run.

Every run renders every frame of the sequence from its own trained FRONT camera (CamPose refined) moved by each
--moves entry (dashrecon.gen.views.ViewMove), with rays recomputed for the sky. Runs sit side by side, labelled.

Outputs in --out_dir: <scene>_<move>.mp4 (all frames) and <scene>_<move>_<t:03d>.jpg (--still_frames), cells
--cell_width wide; <scene>_freeview.json (runs, moves, files, commit). <move> is the move with "=" dropped and "," -> "_".

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/vis_freeview.py --log_dirs results/E5c/val039 results/E8/val039 --labels E5c E8 \
        --moves right=1.5,yaw=15 up=1.5,pitch=-10 yaw=60 --still_frames 50 100 150 --cell_width 480 --fps 10 \
        --out_dir results/_vis/freeview
"""
import argparse
import json
import os
import sys

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.gen.views import ViewMove, render_moved  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def move_tag(spec: str) -> str:
    return spec.replace("=", "").replace(",", "_")


def labelled(rgb: np.ndarray, label: str, width: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 8 * len(label), 20], fill=(0, 0, 0))
    d.text((5, 4), label, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--moves", nargs="+", required=True)
    parser.add_argument("--still_frames", type=int, nargs="+", required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    assert len(args.labels) == len(args.log_dirs), "one label per run"
    commit = git_commit()
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
        t.set_eval()
        trainers.append(t)
    scene = f"val{int(cfgs[0].data.scene_idx):03d}"
    frames = list(range(dataset.start_timestep, dataset.end_timestep))
    for t in args.still_frames:
        assert t in frames, t
    os.makedirs(args.out_dir, exist_ok=True)
    files = []
    for spec in args.moves:
        move = ViewMove.parse(spec)
        video_name = f"{scene}_{move_tag(spec)}.mp4"
        writer = imageio.get_writer(os.path.join(args.out_dir, video_name), mode="I", fps=args.fps)
        for k, t in enumerate(frames):
            cells = [labelled((render_moved(tr, dataset, k, move, device)[0]["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8),
                              f"{lab}  {spec}  t={t}", args.cell_width) for tr, lab in zip(trainers, args.labels)]
            row = np.concatenate(cells, 1)
            writer.append_data(row)
            if t in args.still_frames:
                still = f"{scene}_{move_tag(spec)}_{t:03d}.jpg"
                Image.fromarray(row).save(os.path.join(args.out_dir, still), quality=88)
                files.append(still)
        writer.close()
        files.append(video_name)
        print(f"[vis_freeview] {scene} {spec}: {len(frames)} frames -> {video_name}", flush=True)
    with open(os.path.join(args.out_dir, f"{scene}_freeview.json"), "w") as f:
        json.dump({"scene_id": scene, "runs": dict(zip(args.labels, args.log_dirs)), "moves": args.moves, "files": files,
                   "cell_width": args.cell_width, "dashrecon_commit": commit}, f, indent=2)


if __name__ == "__main__":
    main()
