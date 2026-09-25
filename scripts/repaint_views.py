"""Phase 9 (DECISIONS D16-D18): repaint a rendered trajectory with Wan2.1-T2V video-to-video (runs in .venvs/wan).

SDEdit on video: the frames of a render_views.py directory (--source rgb = the render, filled = after
fill_views.py) are encoded, noised to --strengths s (the first s of the --steps denoising schedule is skipped) and
denoised by the frozen Wan2.1-T2V-1.3B with the fill prompt, so low s keeps the input and only sharpens it, high s
looks more like a real photo but changes content. API confirmed against diffusers 0.40.0
``WanVideoToVideoPipeline`` (``video``, ``strength``; noise added with the scheduler's ``add_noise``). The text encoder
and VAE come from the Wan2.1-VACE-1.3B snapshot (same UMT5-XXL and identical VAE weights, envs/setup_wan.sh).
The whole frame is repainted at --width x --height and resized back to the render size.

Frames --start .. --start + --num_frames - 1 (one clip, 4k + 1 frames). Output in <views_dir>/repaint_<source>/:
s<strength>/<k:03d>.png, compare.mp4 (input | one column per strength, labelled), compare_<k:03d>.jpg for
--still_frames, repaint.json (parameters, seconds and peak memory per strength, commit, generative: true).

Example (wan venv):
    HF_HUB_OFFLINE=1 .venvs/wan/bin/python scripts/repaint_views.py --views_dir results/E8/val056/views/r0_right1.5_yaw15 \
        --source filled --start 40 --strengths 0.3 0.5 0.7 --still_frames 60 80 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

T2V_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
T2V_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
VACE_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "A realistic dashcam video of a street, photorealistic, sharp details, daylight, static scene."
NEGATIVE = ("blurry, low quality, distorted, deformed, cartoon, painting, illustration, text, watermark, "
            "overexposed, flicker, people walking, moving cars")


def labelled(img: Image.Image, label: str, width: int) -> np.ndarray:
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(label), 18], fill=(0, 0, 0))
    d.text((5, 3), label, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--source", choices=["rgb", "filled"], required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--strengths", type=float, nargs="+", required=True)
    parser.add_argument("--still_frames", type=int, nargs="*", default=[])
    parser.add_argument("--width", type=int, default=816)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--cell_width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert (args.num_frames - 1) % 4 == 0, "Wan clips need 4k + 1 frames"
    commit = git_commit()
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel, WanVideoToVideoPipeline
    from huggingface_hub import snapshot_download
    from transformers import T5TokenizerFast, UMT5EncoderModel
    import imageio

    ks = list(range(args.start, args.start + args.num_frames))
    for k in args.still_frames:
        assert k in ks, f"still frame {k} outside the clip {ks[0]}..{ks[-1]}"
    frames = [Image.open(os.path.join(args.views_dir, args.source, f"{k:03d}.png")).convert("RGB") for k in ks]
    full_size = frames[0].size
    size = (args.width, args.height)

    # the pinned T2V snapshot as a local path: diffusers queries the Hub API for a sharded model with a revision,
    # which fails with HF_HUB_OFFLINE=1
    t2v_dir = snapshot_download(T2V_ID, revision=T2V_REVISION, local_files_only=True,
                                allow_patterns=["model_index.json", "scheduler/*", "transformer/*"])  # as envs/setup_wan.sh
    pipe = WanVideoToVideoPipeline(
        tokenizer=T5TokenizerFast.from_pretrained(VACE_ID, subfolder="tokenizer"),  # class from the VACE model_index.json
        text_encoder=UMT5EncoderModel.from_pretrained(VACE_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16),
        transformer=WanTransformer3DModel.from_pretrained(os.path.join(t2v_dir, "transformer"), torch_dtype=torch.bfloat16),
        vae=AutoencoderKLWan.from_pretrained(VACE_ID, subfolder="vae", torch_dtype=torch.float32),
        scheduler=UniPCMultistepScheduler.from_pretrained(os.path.join(t2v_dir, "scheduler")),
    )
    pipe.to("cuda")

    out_dir = os.path.join(args.views_dir, f"repaint_{args.source}")
    results, runs = {}, []
    video_in = [f.resize(size, Image.BICUBIC) for f in frames]
    for s in args.strengths:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        gen = pipe(video=video_in, prompt=PROMPT, negative_prompt=NEGATIVE, height=args.height, width=args.width,
                   num_inference_steps=args.steps, guidance_scale=args.guidance, strength=s,
                   generator=torch.Generator("cuda").manual_seed(args.seed), output_type="pil").frames[0]
        assert len(gen) == len(ks), (len(gen), len(ks))
        sub = os.path.join(out_dir, f"s{s:g}")
        os.makedirs(sub, exist_ok=True)
        results[s] = [g.resize(full_size, Image.BICUBIC) for g in gen]
        for k, img in zip(ks, results[s]):
            img.save(os.path.join(sub, f"{k:03d}.png"))
        runs.append({"strength": s, "denoising_steps": min(int(args.steps * s), args.steps), "seconds": time.time() - t0,
                     "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3})
        print(f"[repaint_views] strength {s:g}: {runs[-1]['denoising_steps']} steps, {runs[-1]['seconds']:.0f} s", flush=True)

    writer = imageio.get_writer(os.path.join(out_dir, "compare.mp4"), mode="I", fps=args.fps)
    for i, k in enumerate(ks):
        row = np.concatenate([labelled(frames[i], f"{args.source}  k={k}", args.cell_width)] +
                             [labelled(results[s][i], f"repaint s={s:g}", args.cell_width) for s in args.strengths], 1)
        writer.append_data(row)
        if k in args.still_frames:
            Image.fromarray(row).save(os.path.join(out_dir, f"compare_{k:03d}.jpg"), quality=90)
    writer.close()
    with open(os.path.join(out_dir, "repaint.json"), "w") as f:
        json.dump({"model": T2V_ID, "revision": T2V_REVISION, "text_encoder_vae": VACE_ID, "prompt": PROMPT, "negative_prompt": NEGATIVE,
                   "params": vars(args), "frames": [ks[0], ks[-1]], "runs": runs, "generative": True, "dashrecon_commit": commit}, f, indent=2)
    print(f"[repaint_views] {args.views_dir} {args.source} {ks[0]}..{ks[-1]} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
