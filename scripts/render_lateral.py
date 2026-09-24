"""Render a trained Phase 6 model along the original trajectory and with lateral offsets (AGENTS Phase 6 task 5).

For every frame of the sequence the camera is shifted along its own x axis (OpenCV: right) by each offset,
in scene units, which are MapAnything's metric estimate (scale strategy "model"; see OPEN_QUESTIONS 15 for
how far that is from true metres). Rendering goes through the normal trainer path (novel_view=False), so the
pose refinement and exposure (Affine) of that frame apply, and held-out frames use the mean embedding.

Outputs in <log_dir>/renders/: lateral_<offset>.mp4 per offset and frames/<t:03d>_<offset>.jpg for the
frames listed in --still_frames. Reads the run's config.yaml (GT isolation re-checked) and checkpoint_final.pth.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/render_lateral.py --log_dir results/E4/val056 \
        --offsets 0 0.5 1 2 --still_frames 50 100 150 --fps 10
"""
import argparse
import os
import sys

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from datasets.driving_dataset import DrivingDataset  # noqa: E402
from utils.misc import import_str  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--offsets", type=float, nargs="+", required=True)
    parser.add_argument("--still_frames", type=int, nargs="+", required=True)
    parser.add_argument("--fps", type=int, required=True)
    args = parser.parse_args()
    commit = git_commit()

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert_non_oracle(cfg)
    device = torch.device("cuda")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()

    full = dataset.full_image_set
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    assert len(frames) == len(full), (len(frames), len(full))
    for t in args.still_frames:
        assert t in frames, f"still frame {t} not in {frames[0]}..{frames[-1]}"
    out_dir = os.path.join(args.log_dir, "renders")
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    with torch.no_grad():
        for off in args.offsets:
            writer = imageio.get_writer(os.path.join(out_dir, f"lateral_{off:g}.mp4"), mode="I", fps=args.fps)
            for i, t in enumerate(frames):
                image_infos, cam_infos = full.get_image(i, trainer._get_downscale_factor())
                image_infos = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in image_infos.items()}
                cam_infos = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cam_infos.items()}
                c2w = cam_infos["camera_to_world"].clone()
                c2w[:3, 3] = c2w[:3, 3] + off * c2w[:3, 0]
                cam_infos["camera_to_world"] = c2w
                rgb = trainer(image_infos, cam_infos)["rgb"].clamp(0, 1)
                img = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
                writer.append_data(img)
                if t in args.still_frames:
                    Image.fromarray(img).save(os.path.join(out_dir, "frames", f"{t:03d}_{off:g}.jpg"), quality=90)
            writer.close()
            io.write_json(os.path.join(out_dir, "render_params.json"), {**vars(args), "dashrecon_commit": commit})
            print(f"[render_lateral] {args.log_dir}: offset {off:g} -> lateral_{off:g}.mp4", flush=True)


if __name__ == "__main__":
    main()
