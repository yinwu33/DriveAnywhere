"""Phase 8 E5+pp: per-frame generative post-processing of a trained model (the control for E6; AGENTS Phase 8).

Renders <output_root>/<init_exp>/<scene_id> the way scripts/render_lateral.py does (normal trainer path, camera
shifted along its own x axis before the pose refinement), then refines every image on its own with
dashrecon.gen.ggds.SDXLRefiner at a fixed --t, conditioned on the NKSR-mesh disparity of that view. Nothing is
trained and frames are refined independently, so consecutive frames and neighbouring viewpoints are not kept
consistent with each other; E6 distils the same refiner into the 3D model instead.

Outputs in <output_root>/E5pp/<scene_id>/:
    renders/frames/<t:03d>_<offset>.jpg     --still_frames x --offsets (same names as render_lateral.py)
    renders/lateral_<offset>.mp4           every frame, for each of --video_offsets
    metrics.json                           held-out test frames at offset 0 with drivestudio's render_images
                                           (same PSNR / SSIM / LPIPS / non-sky / dynamic-region metrics as E3-E6);
                                           test split only, the full split would take ~15 min per scene
    meta.json, render_params.json

Example (main venv, from the repo root):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python scripts/postprocess_frames.py --scene_id val056 --output_root results \
        --offsets 0 0.5 1 2 --still_frames 50 100 150 --video_offsets 1
"""
import argparse
import json
import os
import sys
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.gen.ggds import SDXLRefiner, disparity_image  # noqa: E402
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402

EXP = "E5pp"
PROMPT = "a dashcam photo of a street, realistic, sharp, highly detailed"
NEGATIVE = "blurry, smeared, low quality, distorted, artifacts, painting, cartoon"
METRIC_KEYS = ("psnr", "ssim", "lpips", "occupied_psnr", "occupied_ssim", "masked_psnr", "masked_ssim")


class RefinedTrainer:
    """Trainer proxy whose forward returns the refined image as "rgb", so drivestudio's render_images computes
    the metrics of the post-processed frames with its own code."""

    def __init__(self, trainer, refiner: SDXLRefiner, t: float, disp_pct: float) -> None:
        self.trainer, self.refiner, self.t, self.disp_pct = trainer, refiner, t, disp_pct

    def __call__(self, image_infos, cam_infos, novel_view: bool = False):
        out = self.trainer(image_infos, cam_infos, novel_view)
        maps = self.trainer._mesh_maps(self.trainer._last_cam)
        control = disparity_image(maps["mesh_depth"], maps["mesh_valid"], self.disp_pct)
        out["rgb"] = self.refiner.refine(out["rgb"].clamp(0, 1), control, self.t)
        return out

    def __getattr__(self, name):
        return getattr(self.trainer, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--init_exp", default="E5")
    parser.add_argument("--offsets", type=float, nargs="+", required=True)
    parser.add_argument("--still_frames", type=int, nargs="+", required=True)
    parser.add_argument("--video_offsets", type=float, nargs="*", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--t", type=float, default=0.6)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--gen_hw", type=int, nargs=2, default=[832, 1248])
    parser.add_argument("--disp_pct", type=float, default=90.0)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")

    src_dir = os.path.join(args.output_root, args.init_exp, args.scene_id)
    out_dir = os.path.join(args.output_root, EXP, args.scene_id)
    os.makedirs(os.path.join(out_dir, "renders", "frames"), exist_ok=True)
    cfg = OmegaConf.load(os.path.join(src_dir, "config.yaml"))
    assert_non_oracle(cfg)
    assert "mesh" in cfg.trainer.losses, f"{src_dir} has no mesh for the ControlNet condition"

    from datasets.driving_dataset import DrivingDataset
    from models.video_utils import render_images

    torch.cuda.reset_peak_memory_stats()
    t_begin = time.time()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(src_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    refiner = SDXLRefiner(device, PROMPT, NEGATIVE, args.guidance_scale, args.controlnet_scale, args.num_steps, tuple(args.gen_hw))
    model = RefinedTrainer(trainer, refiner, args.t, args.disp_pct)

    with torch.no_grad():
        res = render_images(trainer=model, dataset=dataset.test_image_set, compute_metrics=True)
    metrics = {"exp": EXP, "scene_id": args.scene_id, "test": {f"image_metrics/test/{k}": res[k] for k in METRIC_KEYS}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[postprocess_frames] {args.scene_id}: test PSNR {res['psnr']:.2f} SSIM {res['ssim']:.3f} LPIPS {res['lpips']:.3f}", flush=True)
    del res

    full = dataset.full_image_set
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    assert len(frames) == len(full), (len(frames), len(full))
    for t in args.still_frames:
        assert t in frames, f"still frame {t} not in {frames[0]}..{frames[-1]}"
    with torch.no_grad():
        for off in sorted(set(args.offsets) | set(args.video_offsets)):
            video = off in args.video_offsets
            writer = imageio.get_writer(os.path.join(out_dir, "renders", f"lateral_{off:g}.mp4"), mode="I", fps=args.fps) if video else None
            for i, t in enumerate(frames):
                if not video and (t not in args.still_frames or off not in args.offsets):
                    continue
                ii, ci = full.get_image(i, trainer._get_downscale_factor())
                ii, ci = to_device(ii, device), to_device(ci, device)
                c2w = ci["camera_to_world"].clone()
                c2w[:3, 3] = c2w[:3, 3] + off * c2w[:3, 0]
                ci["camera_to_world"] = c2w
                img = (model(ii, ci)["rgb"].cpu().numpy() * 255).round().astype(np.uint8)
                if video:
                    writer.append_data(img)
                if t in args.still_frames and off in args.offsets:
                    Image.fromarray(img).save(os.path.join(out_dir, "renders", "frames", f"{t:03d}_{off:g}.jpg"), quality=90)
            if video:
                writer.close()
            print(f"[postprocess_frames] {args.scene_id}: offset {off:g} done", flush=True)
    io.write_json(os.path.join(out_dir, "renders", "render_params.json"), {**vars(args), "dashrecon_commit": commit})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "exp": EXP, "scene_id": args.scene_id, "init": src_dir, "params": vars(args), "prompt": PROMPT,
            "negative_prompt": NEGATIVE, "dashrecon_commit": commit, "torch": torch.__version__,
            "runtime_s": time.time() - t_begin, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT images + dashrecon products + frozen SDXL/ControlNet; checked by dashrecon.train.guard",
        }, f, indent=2)
    print(f"[postprocess_frames] {args.scene_id}: {(time.time() - t_begin) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
