"""Render checkpoints at one fixed grid of E5c cameras; no GT inputs or scores."""
import argparse
import json
from pathlib import Path
import sys
import time

import imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at
from dashrecon.provenance import git_commit
from dashrecon.train.guard import assert_non_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_log_dir", type=Path, required=True)
    parser.add_argument("--log_dirs", type=Path, nargs=2, required=True, help="drive fit, sweep fit")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    commit = git_commit()
    if commit.endswith("-dirty"):
        raise ValueError("commit code before comparison rendering")
    from datasets.driving_dataset import DrivingDataset

    started = time.time()
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.init_log_dir / "config.yaml")
    assert_non_oracle(cfg)
    dataset = DrivingDataset(data_cfg=cfg.data)
    initial = build_trainer(cfg, dataset, device)
    initial.resume_from_checkpoint(ckpt_path=str(args.init_log_dir / "checkpoint_final.pth"), load_only_model=True)
    initial.set_eval()
    cameras = []
    for frame in args.frames:
        ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, frame - dataset.start_timestep), 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        base = refined_c2w(initial, ii, ci)
        h0, w0 = int(ci["height"]), int(ci["width"])
        K = ci["intrinsics"].clone()
        K[:2] *= 1280 / w0
        K[1, 2] -= (h0 * 1280 / w0 - 704) / 2
        for yaw in [0., 15., 30., 45., 60.]:
            for right in [0., .5]:
                cameras.append({"frame": frame, "yaw": yaw, "right": right,
                                "c2w": moved_c2w(base, ViewMove(yaw=yaw, right=right)).cpu().numpy(), "K": K.cpu().numpy()})
    del initial
    torch.cuda.empty_cache()
    outputs = []
    for label, log_dir in zip(["E5c", "drive_gen", "sweep_gen"], [args.init_log_dir, *args.log_dirs]):
        config = OmegaConf.load(log_dir / "config.yaml")
        assert_non_oracle(config)
        trainer = build_trainer(config, dataset, device)
        trainer.resume_from_checkpoint(ckpt_path=str(log_dir / "checkpoint_final.pth"), load_only_model=True)
        trainer.set_eval()
        directory = args.out_dir / label
        directory.mkdir()
        images = []
        with torch.no_grad():
            for k, cam in enumerate(cameras):
                ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, cam["frame"] - dataset.start_timestep), 1)
                out = render_at(trainer, to_device(ii, device), to_device(ci, device),
                                torch.tensor(cam["c2w"], device=device), torch.tensor(cam["K"], device=device), (704, 1280))
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(directory / f"{k:03d}.png")
                np.save(directory / f"{k:03d}_depth.npy", out["depth"].cpu().numpy().astype(np.float16))
                images.append(rgb)
        outputs.append(images)
        del trainer
        torch.cuda.empty_cache()
    writer = imageio.get_writer(str(args.out_dir / "comparison.mp4"), fps=5, macro_block_size=1)
    for k, cam in enumerate(cameras):
        canvas = Image.new("RGB", (1920, 384), "#151515")
        draw = ImageDraw.Draw(canvas)
        for column, (label, images) in enumerate(zip(["E5c", "drive (gen)", "sweep (gen)"], outputs)):
            canvas.paste(Image.fromarray(images[k]).resize((640, 352)), (640 * column, 32))
            draw.text((640 * column + 8, 8), f"{label} | frame {cam['frame']} yaw {cam['yaw']:g} right {cam['right']:g}", fill="white")
        canvas.save(args.out_dir / f"{k:03d}_compare.png")
        writer.append_data(np.asarray(canvas))
    writer.close()
    rows = [{**c, "c2w": c["c2w"].tolist(), "K": c["K"].tolist()} for c in cameras]
    (args.out_dir / "cameras.json").write_text(json.dumps(rows, indent=2))
    (args.out_dir / "meta.json").write_text(json.dumps({"generative": True, "uses_oracle": False,
        "kind": "fixed-camera qualitative comparison, no ground truth", "dashrecon_commit": commit,
        "runtime_s": time.time() - started}, indent=2))
    print(f"[common cameras] {len(cameras)} views -> {args.out_dir}")


if __name__ == "__main__":
    main()
