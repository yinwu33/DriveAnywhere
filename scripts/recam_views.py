"""Side views from the FRONT video with ReCamMaster (user request 2026-09-25, DECISIONS P; runs in .venvs/recam).

ReCamMaster (KwaiVGI, open version on Wan2.1-T2V-1.3B, envs/setup_recam.sh) re-renders a source video along a target
camera trajectory. This feeds it --num_frames (81) FRONT frames of a scene from --image_dir (<t:03d>_0.jpg; the
self-calibrated undistorted images of the E5c pipeline) and renders each preset trajectory in --cam_types
(1 pan right, 2 pan left, 3 tilt up, 4 tilt down, 5 zoom in, 6 zoom out, 7 / 8 translate up / down, 9 arc left,
10 arc right; the pan presets turn by about 20 degrees over the clip). Model loading, the preset trajectories and
the input preprocessing (cover-resize and centre crop to 832 x 480) are the repository's own
(inference_recammaster.py @ fcf98bc: TextVideoCameraDataset and its __main__ steps 1-3), so the script runs with
the repository as working directory.

Output in --out_dir: source.mp4 (the cropped input), cam<type>.mp4, compare.mp4 (source | one column per
trajectory, labelled), compare_<i:03d>.jpg for --still_frames (indices into the clip), recam.json (parameters,
seconds, peak memory, commit, generative: true).

Example (recam venv, from the DriveAnywhere repo root):
    .venvs/recam/bin/python scripts/recam_views.py --image_dir data/dashrecon/_undistorted/calib-glomap/056/images \
        --start 40 --cam_types 2 1 9 10 --still_frames 0 40 80 --out_dir results/_vis_p9/recam_val056
"""
import argparse
import json
import os
import sys
import time

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dashrecon.provenance import git_commit  # noqa: E402

RECAM = os.path.join(REPO, ".venvs", "src", "ReCamMaster")
PROMPT = ("A dashcam video recorded from a car driving down a sunny suburban street, with houses, trees, sidewalks, "
          "street lights and road markings. Realistic, sharp details.")
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")  # repository default
CAM_NAMES = {1: "pan right", 2: "pan left", 3: "tilt up", 4: "tilt down", 5: "zoom in", 6: "zoom out", 7: "translate up",
             8: "translate down", 9: "arc left", 10: "arc right"}


def labelled(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(label), 18], fill=(0, 0, 0))
    d.text((5, 3), label, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cam_types", type=int, nargs="+", required=True)
    parser.add_argument("--still_frames", type=int, nargs="*", default=[])
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    assert args.num_frames == 81, "ReCamMaster's dataset class asserts 81 frames"
    commit = git_commit()
    image_dir, out_dir = os.path.abspath(args.image_dir), os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "input", "videos"), exist_ok=True)

    # the FRONT clip as the repository's dataset layout: videos/<name>.mp4 + metadata.csv
    frames = [np.asarray(Image.open(os.path.join(image_dir, f"{t:03d}_0.jpg")).convert("RGB")) for t in range(args.start, args.start + args.num_frames)]
    imageio.mimwrite(os.path.join(out_dir, "input", "videos", "front.mp4"), frames, fps=args.fps, quality=10)
    with open(os.path.join(out_dir, "input", "metadata.csv"), "w") as f:
        f.write('file_name,text\nfront.mp4,"' + PROMPT + '"\n')

    os.chdir(RECAM)  # relative model and trajectory paths of the repository
    sys.path.insert(0, RECAM)
    import torch.nn as nn
    from diffsynth import ModelManager, WanVideoReCamMasterPipeline
    from inference_recammaster import TextVideoCameraDataset

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                               "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                               "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
    pipe.dit.load_state_dict(torch.load("models/ReCamMaster/checkpoints/step20000.ckpt", map_location="cpu"), strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    outputs, runs, source = {}, [], None
    for cam in args.cam_types:
        item = TextVideoCameraDataset(os.path.join(out_dir, "input"), os.path.join(out_dir, "input", "metadata.csv"),
                                      argparse.Namespace(cam_type=cam))[0]
        if source is None:
            source = ((item["video"].permute(1, 2, 3, 0).float().numpy() + 1) * 127.5).clip(0, 255).round().astype(np.uint8)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = pipe(prompt=[item["text"]], negative_prompt=NEGATIVE, source_video=item["video"][None], target_camera=item["camera"][None],
                     cfg_scale=args.cfg_scale, num_inference_steps=args.steps, seed=args.seed, tiled=True)
        outputs[cam] = [np.asarray(f) for f in video]
        imageio.mimwrite(os.path.join(out_dir, f"cam{cam}.mp4"), outputs[cam], fps=args.fps, quality=9)
        runs.append({"cam_type": cam, "trajectory": CAM_NAMES[cam], "seconds": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3})
        print(f"[recam_views] cam_type {cam} ({CAM_NAMES[cam]}): {runs[-1]['seconds']:.0f} s, {runs[-1]['peak_vram_gb']:.1f} GB", flush=True)

    imageio.mimwrite(os.path.join(out_dir, "source.mp4"), list(source), fps=args.fps, quality=9)
    writer = imageio.get_writer(os.path.join(out_dir, "compare.mp4"), mode="I", fps=args.fps)
    for i in range(args.num_frames):
        row = np.concatenate([labelled(source[i], f"FRONT t={args.start + i}")] +
                             [labelled(outputs[c][i], f"ReCamMaster {CAM_NAMES[c]} (gen)") for c in args.cam_types], 1)
        writer.append_data(row)
        if i in args.still_frames:
            Image.fromarray(row).save(os.path.join(out_dir, f"compare_{i:03d}.jpg"), quality=90)
    writer.close()
    with open(os.path.join(out_dir, "recam.json"), "w") as f:
        json.dump({"recammaster_commit": "fcf98bc86e876bb534518cd99e8a65b282f0f16e", "checkpoint": "KwaiVGI/ReCamMaster-Wan2.1 step20000.ckpt",
                   "image_dir": args.image_dir, "frames": [args.start, args.start + args.num_frames - 1], "prompt": PROMPT,
                   "params": vars(args), "runs": runs, "generative": True, "dashrecon_commit": commit}, f, indent=2)
    print(f"[recam_views] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
