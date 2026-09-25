"""Phase 9 with GEN3C instead of Wan VACE (DECISIONS P; runs in .venvs/gen3c, envs/setup_gen3c.sh).

GEN3C (NVIDIA, Cosmos-Predict1 7B) generates a video along a camera trajectory from a conditioning frame and, for
every frame, a "3D cache" render of that camera (warped image + mask of valid pixels), and was trained to follow
the cache where it is valid and to generate the rest consistently. Our 3DGS render is such a cache: this script
feeds a scripts/render_views.py directory rendered at GEN3C's 704 x 1280 with a ramped move (--render_hw 704 1280
--ramp_frames n, so view 0 is the real FRONT pose):
    - conditioning frame of the first chunk: the real FRONT image of view 0's frame from --image_dir (the undistorted
      1920 x 1280 images the run was trained on, scaled by 2/3 and centre-cropped like the render's intrinsics);
    - cache for each view: the render where training views saw the surface (mask/ = 0), -1 (empty, as GEN3C's own
      forward warp) and mask 0 in the holes; view 0 uses the real image with a full mask;
    - chunks of 121 frames, autoregressive as gen3c_dynamic.py: each later chunk starts from the last generated
      frame (one frame overlap), the last chunk repeats the last view to fill up.
Pipeline construction and the call follow cosmos_predict1/diffusion/inference/gen3c_dynamic.py @ db2ffe1 with all
offloading, no guardrail, no prompt encoder (the repository's low-memory settings, ~43 GB peak; the prompt is then
unused) and its defaults (35 steps, guidance 1, seed 1); --fps is passed as the frame-rate conditioning.

GEN3C regenerates the whole frame; the output is written unchanged as filled/<k:03d>.png so depth_views.py and
train_fill.py run as after fill_views.py (train_fill weights the hole pixels 1 and the rest --seen_w). Also written:
filled_compare.mp4 (render with holes in red | GEN3C), fill.json (generator, parameters, seconds, commit,
generative: true).

Example (from the DriveAnywhere repo root):
    CUDA_HOME=/usr/local/cuda-12.1 .venvs/gen3c/bin/python scripts/gen3c_fill.py \
        --views_dir results/E8/val056/views/r1_yaw30 --image_dir data/dashrecon/_undistorted/calib-glomap/056/images
"""
import argparse
import json
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dashrecon.provenance import git_commit  # noqa: E402

GEN3C = os.path.join(REPO, ".venvs", "src", "GEN3C")
H, W, CHUNK = 704, 1280, 121


def real_frame(path: str) -> np.ndarray:
    """Undistorted 1920 x 1280 FRONT image -> 704 x 1280 like render_views --render_hw 704 1280 (scale 2/3 of the
    image, i.e. 4/3 of the 960 x 640 training grid, then centre crop; < 0.4 px off the exact 853.3-row scale)."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert img.shape[:2] == (1280, 1920), (path, img.shape)
    img = cv2.resize(img, (W, 853), interpolation=cv2.INTER_AREA)
    return img[75:75 + H]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    commit = git_commit()
    views_dir, image_dir = os.path.abspath(args.views_dir), os.path.abspath(args.image_dir)
    with open(os.path.join(views_dir, "cams.json")) as f:
        cams = json.load(f)["cams"]
    n = len(cams)
    assert all(c["hw"] == [H, W] for c in cams), "render the views with --render_hw 704 1280"
    assert cams[0]["ramp"] == 0.0, "view 0 must be the real FRONT pose (--ramp_frames > 0)"
    renders = np.stack([np.asarray(Image.open(os.path.join(views_dir, "rgb", f"{k:03d}.png")).convert("RGB")) for k in range(n)])
    holes = np.stack([np.asarray(Image.open(os.path.join(views_dir, "mask", f"{k:03d}.png"))) > 127 for k in range(n)])
    first = real_frame(os.path.join(image_dir, f"{cams[0]['frame']:03d}_0.jpg"))

    warp = renders.astype(np.float32) / 127.5 - 1.0
    warp[holes] = -1.0
    warp[0] = first.astype(np.float32) / 127.5 - 1.0
    valid = (~holes).astype(np.float32)
    valid[0] = 1.0

    os.chdir(GEN3C)  # relative config and checkpoint paths of the repository
    sys.path.insert(0, GEN3C)
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline

    pipeline = Gen3cPipeline(
        inference_type="video2world", checkpoint_dir="checkpoints", checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir="Pixtral-12B", enable_prompt_upsampler=False, offload_network=True, offload_tokenizer=True,
        offload_text_encoder_model=True, offload_prompt_upsampler=True, offload_guardrail_models=True,
        disable_guardrail=True, disable_prompt_encoder=True, guidance=args.guidance, num_steps=args.num_steps,
        height=H, width=W, fps=args.fps, num_video_frames=CHUNK, seed=args.seed)
    assert pipeline.model.chunk_size == CHUNK, pipeline.model.chunk_size
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    out, chunks, cond, start = [], [], torch.from_numpy(warp[0]).permute(2, 0, 1)[None, :, None].to(device), 0
    while start < n - 1 or start == 0:
        idx = [min(i, n - 1) for i in range(start, start + CHUNK)]
        t0 = time.time()
        imgs = torch.from_numpy(warp[idx]).permute(0, 3, 1, 2)[None, :, None].to(device)   # B, F, N=1, C, H, W
        masks = torch.from_numpy(valid[idx])[None, :, None, None].to(device)                 # B, F, N=1, 1, H, W
        video, _ = pipeline.generate(prompt="", image_path=cond, rendered_warp_images=imgs, rendered_warp_masks=masks)
        out.extend(list(video) if start == 0 else list(video[1:]))
        cond = torch.from_numpy(video[-1].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device)
        chunks.append({"views": [start, start + CHUNK - 1], "seconds": time.time() - t0})
        print(f"[gen3c_fill] views {start}..{min(start + CHUNK - 1, n - 1)}: {time.time() - t0:.0f} s", flush=True)
        start += CHUNK - 1
    out = out[:n]

    os.makedirs(os.path.join(views_dir, "filled"), exist_ok=True)
    writer = imageio.get_writer(os.path.join(views_dir, "filled_compare.mp4"), mode="I", fps=args.fps)
    for k in range(n):
        Image.fromarray(out[k]).save(os.path.join(views_dir, "filled", f"{k:03d}.png"))
        shown = renders[k].copy()
        shown[holes[k]] = (0.45 * shown[holes[k]] + 0.55 * np.array([230, 40, 40])).astype(np.uint8)
        row = np.concatenate([shown, out[k]], 1)
        writer.append_data(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    writer.close()
    with open(os.path.join(views_dir, "fill.json"), "w") as f:
        json.dump({"generator": "GEN3C-Cosmos-7B", "gen3c_commit": "db2ffe12ced12ddafcec5e0422ee46ce8520746b",
                   "weights_revision": "9bcfdb4f3924f41376daeadf6200826c12a3bf8e", "params": vars(args), "chunks": chunks,
                   "conditioning": "real FRONT image of view 0, then the last generated frame", "composited": False,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True, "dashrecon_commit": commit}, f, indent=2)
    print(f"[gen3c_fill] {views_dir}: {n} views in {len(chunks)} chunks, {sum(c['seconds'] for c in chunks) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
