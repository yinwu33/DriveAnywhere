"""E33 (docs/PLAN_AFTER_E30.md): Cosmos-Transfer2.5 as a video-level restorer of a rendered sweep. Runs in any venv
(numpy, imageio, PIL, cv2); the generation itself is the official examples/inference.py in .venvs/cosmos_transfer
(envs/setup_cosmos_transfer.sh).

export (default): a scripts/render_views.py directory rendered at 704 x 1280 (Cosmos' "720" 16:9 bucket, so nothing is
resized) becomes the inputs of one Cosmos-Transfer2.5 sample in --out_dir:
    input.mp4         the renders (rgb/<k>.png), holes as rendered;
    depth.mp4         depth control: inverse rendered z-depth (depth/<k>.npy), 0 in the holes, scaled to 0..255 by the
                      1st / 99th percentile of the valid pixels over the whole sweep, three identical channels (the
                      format of the official depth control videos);
    control_mask.mp4  255 where a training view saw the pixel (mask/<k>.png == 0), 0 in the holes: the mask of both
                      controls, so the model keeps the observed content (vis = blur control of the render itself, depth)
                      and is free in the holes;
    spec.json         the inference file: prompt, guidance, steps, seed, vis / depth controls with their weights and the
                      mask, plus --negative_prompt or --sigma_max when given;
    export.json       parameters, frame count, depth scaling, commit.
Run Cosmos as (single GPU, 65 GB; guardrails off, they need another gated model):
    cd .venvs/src/cosmos-transfer2.5 && ../../cosmos_transfer/bin/python examples/inference.py -i <out_dir>/spec.json \\
        -o <out_dir>/cosmos --disable_guardrails
import: --cosmos_video <out_dir>/cosmos/<name>.mp4 writes the generated frames back as <views_dir>/filled/<k:03d>.png
(the output of gen3c_fill.py / fill_views.py, so depth_views.py, validate_generated_views.py and train_fill.py run
unchanged), filled_compare.mp4 (render with holes in red | generated) and fill.json (generator, spec, generative: true).
Cosmos writes the output at the input resolution (keep_input_resolution); the frame count must match the views.

Example:
    python scripts/e33_cosmos_control.py --views_dir results/E33/val056/views/r0 --out_dir results/E33/val056/cosmos/r0 \\
        --prompt "..." --fps 10 --seed 0
    python scripts/e33_cosmos_control.py import --views_dir results/E33/val056/views/r0 \\
        --cosmos_video results/E33/val056/cosmos/r0/cosmos/r0.mp4
"""
import argparse
import json
import os
import sys

import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

HW = (704, 1280)
GENERATOR = "nvidia/Cosmos-Transfer2.5-2B"


def writer(path: str, fps: int):
    return imageio.get_writer(path, mode="I", fps=fps, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)


def load_views(views_dir: str):
    with open(os.path.join(views_dir, "cams.json")) as f:
        cams = json.load(f)["cams"]
    assert all(c["hw"] == list(HW) for c in cams), "render the views with --render_hw 704 1280 (Cosmos' 720 / 16:9 bucket)"
    return cams


def export(args) -> None:
    cams = load_views(args.views_dir)
    n = len(cams)
    os.makedirs(args.out_dir, exist_ok=True)
    rgb = [np.asarray(Image.open(os.path.join(args.views_dir, "rgb", f"{k:03d}.png")).convert("RGB")) for k in range(n)]
    hole = [np.asarray(Image.open(os.path.join(args.views_dir, "mask", f"{k:03d}.png"))) > 127 for k in range(n)]
    depth = [np.load(os.path.join(args.views_dir, "depth", f"{k:03d}.npy")).astype(np.float32) for k in range(n)]
    inv = [np.where(d > 0, 1.0 / np.maximum(d, 1e-6), 0.0) for d in depth]
    valid = np.concatenate([v[(d > 0) & ~h] for v, d, h in zip(inv, depth, hole)])
    lo, hi = np.percentile(valid, 1.0), np.percentile(valid, 99.0)
    with writer(os.path.join(args.out_dir, "input.mp4"), args.fps) as w_in, \
            writer(os.path.join(args.out_dir, "depth.mp4"), args.fps) as w_depth, \
            writer(os.path.join(args.out_dir, "control_mask.mp4"), args.fps) as w_mask:
        for k in range(n):
            w_in.append_data(rgb[k])
            d = np.clip((inv[k] - lo) / (hi - lo), 0.0, 1.0)
            d[hole[k] | (depth[k] <= 0)] = 0.0
            w_depth.append_data(np.repeat((d * 255).round().astype(np.uint8)[..., None], 3, -1))
            w_mask.append_data(np.repeat(((~hole[k]) * 255).astype(np.uint8)[..., None], 3, -1))
    mask_path = os.path.abspath(os.path.join(args.out_dir, "control_mask.mp4"))
    spec = {"name": args.name, "prompt": args.prompt, "video_path": os.path.abspath(os.path.join(args.out_dir, "input.mp4")),
            "guidance": args.guidance, "num_steps": args.num_steps, "seed": args.seed,
            "vis": {"control_weight": args.vis_weight, "mask_path": mask_path},
            "depth": {"control_path": os.path.abspath(os.path.join(args.out_dir, "depth.mp4")), "control_weight": args.depth_weight,
                      "mask_path": mask_path}}
    if args.negative_prompt is not None:
        spec["negative_prompt"] = args.negative_prompt
    if args.sigma_max is not None:
        spec["sigma_max"] = str(args.sigma_max)
    with open(os.path.join(args.out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(args.out_dir, "export.json"), "w") as f:
        json.dump({**vars(args), "views": n, "hole_fraction": float(np.mean([h.mean() for h in hole])),
                   "inverse_depth_range": [float(lo), float(hi)], "generator": GENERATOR, "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[e33_cosmos_control] {n} views -> {args.out_dir} (holes {np.mean([h.mean() for h in hole]) * 100:.1f}%)", flush=True)


def import_video(args) -> None:
    cams = load_views(args.views_dir)
    n = len(cams)
    out_dir = args.views_dir if args.out_dir is None else args.out_dir
    filled_dir = os.path.join(out_dir, "filled")
    if os.path.exists(filled_dir):
        raise FileExistsError(f"refusing to overwrite existing generated frames: {filled_dir}")
    frames = imageio.mimread(args.cosmos_video, memtest=False)
    assert len(frames) == n, f"{args.cosmos_video} has {len(frames)} frames, the views {n}"
    os.makedirs(filled_dir, exist_ok=True)
    with writer(os.path.join(out_dir, "filled_compare.mp4"), args.fps) as w:
        for k in range(n):
            gen = np.asarray(frames[k])[..., :3]
            if gen.shape[:2] != HW:
                gen = np.asarray(Image.fromarray(gen).resize((HW[1], HW[0]), Image.LANCZOS))
            Image.fromarray(gen).save(os.path.join(filled_dir, f"{k:03d}.png"))
            render = np.asarray(Image.open(os.path.join(args.views_dir, "rgb", f"{k:03d}.png")).convert("RGB")).copy()
            hole = np.asarray(Image.open(os.path.join(args.views_dir, "mask", f"{k:03d}.png"))) > 127
            render[hole] = (0.6 * render[hole] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
            row = np.concatenate([render, gen], 1)
            w.append_data(row[::2, ::2])
    spec_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.cosmos_video)))
    with open(os.path.join(spec_dir, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(out_dir, "fill.json"), "w") as f:
        json.dump({"generator": GENERATOR, "cosmos_video": os.path.abspath(args.cosmos_video), "params": spec, "views": n,
                   "composited": False, "generative": True, "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[e33_cosmos_control] {n} generated frames -> {filled_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode")
    ex = sub.add_parser("export")
    ex.add_argument("--views_dir", required=True)
    ex.add_argument("--out_dir", required=True)
    ex.add_argument("--name", default="sweep")
    ex.add_argument("--prompt", required=True)
    ex.add_argument("--negative_prompt")
    ex.add_argument("--fps", type=int, default=10)
    ex.add_argument("--guidance", type=float, default=3.0)
    ex.add_argument("--num_steps", type=int, default=35)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--vis_weight", type=float, default=0.5)
    ex.add_argument("--depth_weight", type=float, default=0.5)
    ex.add_argument("--sigma_max", type=float, help="noise added to the input video (0..200); default: Cosmos' own")
    im = sub.add_parser("import")
    im.add_argument("--views_dir", required=True)
    im.add_argument("--cosmos_video", required=True)
    im.add_argument("--out_dir")
    im.add_argument("--fps", type=int, default=10)
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("export", "import"):
        argv = ["export", *argv]
    args = parser.parse_args(argv)
    if args.mode == "export":
        export(args)
    else:
        import_video(args)


if __name__ == "__main__":
    main()
