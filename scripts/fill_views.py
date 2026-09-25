"""Phase 9 step 2 (DECISIONS D16-D18): fill the holes of a rendered trajectory with Wan2.1-VACE (runs in .venvs/wan).

Reads a scripts/render_views.py directory (rgb/, mask/) and fills the masked pixels of every frame with the frozen
Wan2.1-VACE video model (masked video-to-video, API confirmed against diffusers 0.40.0 ``WanVACEPipeline``: ``video``
and ``mask`` frame lists, white mask = generate). The sequence is cut into clips of --clip_len frames (4k + 1) that
overlap by --overlap frames; from the second clip on, the overlapping frames are passed as already-filled frames with
an all-black mask, so each clip continues the previous one. Frames are resized to --width x --height for the model
(masked pixels set to grey so the unconstrained render there does not condition the model) and the generated pixels
are feathered back into the render at full resolution; unmasked pixels keep the render exactly.

Output in the same directory: filled/<k:03d>.png, filled_compare.mp4 (render | filled), fill.json (parameters, per
clip frame range and seconds, commit).

Example (wan venv):
    HF_HUB_OFFLINE=1 .venvs/wan/bin/python scripts/fill_views.py --views_dir results/E8/val039/views/r0_right1.5_yaw15
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "A realistic dashcam video of a street, photorealistic, sharp details, daylight, static scene."
NEGATIVE = ("blurry, low quality, distorted, deformed, cartoon, painting, illustration, text, watermark, "
            "overexposed, flicker, people walking, moving cars")


def clip_starts(n: int, length: int, overlap: int) -> list[int]:
    """Clip start indices covering n frames; the last clip ends at the last frame."""
    assert n >= length, (n, length)
    starts = list(range(0, n - length + 1, length - overlap))
    if starts[-1] + length < n:
        starts.append(n - length)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--clip_len", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--width", type=int, default=816)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--flow_shift", type=float, default=3.0, help="3.0 for 480p models (diffusers Wan docs)")
    parser.add_argument("--feather_px", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert (args.clip_len - 1) % 4 == 0, "Wan clips need 4k + 1 frames"
    commit = git_commit()
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    from diffusers.utils import export_to_video

    files = sorted(glob.glob(os.path.join(args.views_dir, "rgb", "*.png")))
    renders = [Image.open(f).convert("RGB") for f in files]
    masks = [Image.open(f.replace(f"{os.sep}rgb{os.sep}", f"{os.sep}mask{os.sep}")).convert("L") for f in files]
    size = (args.width, args.height)
    full_size = renders[0].size

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.to("cuda")

    filled = [None] * len(files)
    clips = []
    for c, s in enumerate(clip_starts(len(files), args.clip_len, args.overlap)):
        t0 = time.time()
        video_in, mask_in = [], []
        for k in range(s, s + args.clip_len):
            if filled[k] is not None:  # overlap with the previous clip: condition on its result
                video_in.append(filled[k].resize(size, Image.BICUBIC))
                mask_in.append(Image.new("L", size, 0))
                continue
            fr = np.asarray(renders[k].resize(size, Image.BICUBIC)).copy()
            m = np.asarray(masks[k].resize(size, Image.NEAREST)) > 127
            fr[m] = 128
            video_in.append(Image.fromarray(fr))
            mask_in.append(Image.fromarray((m * 255).astype(np.uint8)))
        gen = pipe(video=video_in, mask=mask_in, prompt=PROMPT, negative_prompt=NEGATIVE, height=args.height, width=args.width,
                   num_frames=args.clip_len, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=torch.Generator("cuda").manual_seed(args.seed + c), output_type="pil").frames[0]
        new = 0
        for i, k in enumerate(range(s, s + args.clip_len)):
            if filled[k] is not None:
                continue
            soft = masks[k].filter(ImageFilter.GaussianBlur(args.feather_px))
            filled[k] = Image.composite(gen[i].resize(full_size, Image.BICUBIC), renders[k], soft)
            new += 1
        clips.append({"clip": c, "frames": [s, s + args.clip_len - 1], "new_frames": new, "seconds": time.time() - t0})
        print(f"[fill_views] clip {c}: frames {s}..{s + args.clip_len - 1} ({new} new), {time.time() - t0:.0f} s", flush=True)

    os.makedirs(os.path.join(args.views_dir, "filled"), exist_ok=True)
    for k, img in enumerate(filled):
        img.save(os.path.join(args.views_dir, "filled", f"{k:03d}.png"))
    export_to_video([np.concatenate([np.asarray(r), np.asarray(f)], 1).astype(np.float32) / 255.0 for r, f in zip(renders, filled)],
                    os.path.join(args.views_dir, "filled_compare.mp4"), fps=10)
    with open(os.path.join(args.views_dir, "fill.json"), "w") as f:
        json.dump({"model": MODEL_ID, "prompt": PROMPT, "negative_prompt": NEGATIVE, "params": vars(args), "clips": clips,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True, "dashrecon_commit": commit}, f, indent=2)
    print(f"[fill_views] {args.views_dir}: {len(files)} frames in {len(clips)} clips, "
          f"{sum(c['seconds'] for c in clips) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
