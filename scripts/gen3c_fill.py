"""Phase 9 with GEN3C instead of Wan VACE (DECISIONS P; runs in .venvs/gen3c, envs/setup_gen3c.sh).

GEN3C (NVIDIA, Cosmos-Predict1 7B) generates a video along a camera trajectory from a conditioning frame and, for
every frame, a "3D cache" render of that camera (warped image + mask of valid pixels), and was trained to follow
the cache where it is valid and to generate the rest consistently. Our 3DGS render is such a cache: this script
feeds a scripts/render_views.py directory rendered at GEN3C's 704 x 1280 with a ramped move (--render_hw 704 1280
--ramp_frames n, so view 0 is the real FRONT pose):
    - conditioning frame of the first chunk: the real FRONT image of view 0's frame (<data.data_root>/<scene_idx>/
      images/<t:03d>_0.jpg of the rendered run's config: the undistorted 1920 x 1280 images it was trained on),
      scaled by 2/3 and centre-cropped like the render's intrinsics;
    - cache for each view: the render where training views saw the surface (mask/ = 0), -1 (empty, as GEN3C's own
      forward warp) and mask 0 in the holes; view 0 uses the real image with a full mask;
    - --buffers picks up to two cache buffers per view (GEN3C-Cosmos-7B takes frame_buffer_max = 2, newest first):
      "render" is the above; "realN" is the real FRONT image of the view's N-th warp source (render_views.py
      --src_offsets: frame t - d_N), forward-warped into the view with GEN3C's own splatting (forward_warp_utils_pytorch,
      world points from the run's rendered depth at that source camera). Real warps carry real texture, which a
      blurry 3DGS render as the only cache does not (DECISIONS P: the render-only cache gave blurry output);
    - chunks of 121 frames, autoregressive as gen3c_dynamic.py: each later chunk starts from the last generated
      frame (one frame overlap), the last chunk repeats the last view to fill up.
Pipeline construction and the call follow cosmos_predict1/diffusion/inference/gen3c_dynamic.py @ db2ffe1 with all
offloading, no guardrail, no prompt encoder (the repository's low-memory settings, ~43 GB peak; the prompt is then
unused) and its defaults (35 steps, guidance 1, seed 1); --fps is passed as the frame-rate conditioning.
--speckle drop|fill (consistent cache only) removes or fills the sparse dots a grazing-angle warp leaves (D-C2).
--prompt TEXT enables the T5-11B prompt encoder (checkpoints/google-t5/t5-11b, offloaded after encoding) and conditions
every chunk on TEXT (D-C1 in docs/EXPERIMENTS.md); without it the pipeline is exactly the prompt-free one above.

GEN3C regenerates the whole frame; the output is written unchanged as filled/<k:03d>.png so depth_views.py and
train_fill.py run as after fill_views.py (train_fill weights the hole pixels 1 and the rest --seen_w). Also written:
buffers.mp4 (render | each cache buffer, before generation; --buffers_only stops there), filled_compare.mp4 (render with holes in red | GEN3C), fill.json (generator, parameters, seconds, commit,
generative: true).

Example (from the DriveAnywhere repo root):
    CUDA_HOME=/usr/local/cuda-12.1 .venvs/gen3c/bin/python scripts/gen3c_fill.py \
        --views_dir results/E8/val056/views/r1_yaw30
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

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
    parser.add_argument("--buffers", nargs="+", default=["render"], help='up to two of "render", "real0", "real1", ... (newest first)')
    parser.add_argument("--buffers_only", action="store_true", help="write buffers.mp4 and stop before generation")
    parser.add_argument("--out_dir", help="separate output directory; inputs in views_dir stay untouched")
    parser.add_argument("--cache_policy", choices=["fixed", "consistent"], default="fixed")
    parser.add_argument("--memory_dirs", nargs="*", default=[])
    parser.add_argument("--validation_dirs", nargs="*", default=[])
    parser.add_argument("--memory_radius", type=int, default=6)
    parser.add_argument("--memory_candidate_policy", choices=["frame", "pose"], default="frame")
    parser.add_argument("--max_memory_candidates", type=int, default=3)
    parser.add_argument("--require_memory", action="store_true", help="fail if no validated memory reaches the cache")
    parser.add_argument("--speckle", choices=["keep", "drop", "fill"], default="keep",
                        help="consistent cache only: sparse warp dots (dashrecon.gen.cache.treat_speckle, D-C2)")
    parser.add_argument("--speckle_window", type=int, default=15)
    parser.add_argument("--speckle_density", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--prompt", help="scene description; enables the T5 prompt encoder (default: no prompt encoder)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    commit = git_commit()
    from omegaconf import OmegaConf

    views_dir = os.path.abspath(args.views_dir)
    with open(os.path.join(views_dir, "cams.json")) as f:
        meta = json.load(f)
    cams = meta["cams"]
    out_dir = views_dir if args.out_dir is None else os.path.abspath(args.out_dir)
    if os.path.exists(os.path.join(out_dir, "filled")):
        raise FileExistsError(f"refusing to overwrite existing generated frames: {out_dir}")
    if args.cache_policy == "fixed" and (args.memory_dirs or args.validation_dirs or args.require_memory):
        raise ValueError("memory options require --cache_policy consistent")
    if args.cache_policy == "fixed" and args.speckle != "keep":
        raise ValueError("--speckle requires --cache_policy consistent")
    memory_dirs = [Path(p).resolve() for p in args.memory_dirs]
    validation_dirs = [Path(p).resolve() for p in args.validation_dirs]
    os.makedirs(out_dir, exist_ok=True)
    if out_dir != views_dir:
        for name in ("cams.json", "rgb", "mask", "depth", "count", "src"):
            src_path = os.path.join(views_dir, name)
            if os.path.exists(src_path):
                dst_path = os.path.join(out_dir, name)
                if os.path.lexists(dst_path):
                    if os.path.realpath(dst_path) != os.path.realpath(src_path):
                        raise FileExistsError(dst_path)
                else:
                    os.symlink(src_path, dst_path)
    data_cfg = OmegaConf.load(os.path.join(meta["log_dir"], "config.yaml")).data
    image_dir = os.path.abspath(os.path.join(data_cfg.data_root, f"{int(data_cfg.scene_idx):03d}", "images"))
    if args.cache_policy == "consistent":
        data_cfg.pixel_source.mask_dir = os.path.abspath(data_cfg.pixel_source.mask_dir)
    n = len(cams)
    assert all(c["hw"] == [H, W] for c in cams), "render the views with --render_hw 704 1280"
    assert cams[0]["ramp"] == 0.0, "view 0 must be the real FRONT pose (--ramp_frames > 0)"
    renders = np.stack([np.asarray(Image.open(os.path.join(views_dir, "rgb", f"{k:03d}.png")).convert("RGB")) for k in range(n)])
    holes = np.stack([np.asarray(Image.open(os.path.join(views_dir, "mask", f"{k:03d}.png"))) > 127 for k in range(n)])
    first = real_frame(os.path.join(image_dir, f"{cams[0]['frame']:03d}_0.jpg"))

    assert 1 <= len(args.buffers) <= 2, args.buffers
    os.chdir(GEN3C)  # relative config and checkpoint paths of the repository
    sys.path.insert(0, GEN3C)
    from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import forward_warp
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
    from dashrecon.gen.views import backproject

    device = torch.device("cuda")
    sources = {}

    def source(src: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Real FRONT image of a warp source in [-1, 1] (1, 3, H, W) and its world points (1, H, W, 3)."""
        if src["frame"] not in sources:
            img = real_frame(os.path.join(image_dir, f"{src['frame']:03d}_0.jpg")).astype(np.float32) / 127.5 - 1.0
            depth = torch.from_numpy(np.load(os.path.join(views_dir, "src", f"{src['frame']:03d}.npy")).astype(np.float32)).to(device)
            pts = backproject(depth, torch.tensor(src["K"], device=device), torch.tensor(src["c2w"], device=device)).reshape(1, H, W, 3)
            sources[src["frame"]] = (torch.from_numpy(img).permute(2, 0, 1)[None].to(device), pts, (depth > 0).float()[None, None])
        return sources[src["frame"]]

    if args.cache_policy == "consistent":
        from dashrecon.gen.cache import build_consistent_cache
        warp, valid, cache_report = build_consistent_cache(
            cams, Path(views_dir), data_cfg, Path(image_dir), memory_dirs, validation_dirs,
            real_frame, args.memory_radius, .10, .12, args.memory_candidate_policy, args.max_memory_candidates,
            args.speckle, args.speckle_window, args.speckle_density)
        if args.require_memory and cache_report["memory_coverage"] <= 0:
            raise ValueError("no validated memory reached the generator; see validation results")
    else:
        warp = np.zeros((n, len(args.buffers), H, W, 3), dtype=np.float32)
        valid = np.zeros((n, len(args.buffers), H, W), dtype=np.float32)
        with torch.no_grad():
            for k, c in enumerate(cams):
                k_cv = torch.tensor(c["K"], device=device)
                k_cv[0, 2] -= 0.5  # GEN3C projects to integer pixel centres, drivestudio intrinsics use +0.5
                k_cv[1, 2] -= 0.5
                w2c = torch.linalg.inv(torch.tensor(c["c2w"], device=device))
                for b, name in enumerate(args.buffers):
                    if name == "render":
                        warp[k, b] = np.where(holes[k][..., None], -1.0, renders[k].astype(np.float32) / 127.5 - 1.0)
                        valid[k, b] = (~holes[k]).astype(np.float32)
                        continue
                    assert name.startswith("real"), name
                    img, pts, m = source(c["src"][int(name[4:])])
                    wimg, wmask, _, _ = forward_warp(img, m, None, None, w2c[None], k_cv[None], k_cv[None], world_points1=pts)
                    warp[k, b] = wimg[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                    valid[k, b] = wmask[0, 0].cpu().numpy()
        cache_report = {"policy": "fixed", "buffers": args.buffers}
    warp[0, 0] = first.astype(np.float32) / 127.5 - 1.0  # view 0 is the real FRONT pose: the real image, fully valid
    valid[0, 0] = 1.0
    # Release CUDA source tensors before loading the 7B diffusion model.
    sources.clear()
    torch.cuda.empty_cache()
    buffer_count = warp.shape[1]
    coverage = [float(valid[:, b].mean()) for b in range(buffer_count)]
    cache_report.update({"views_dir": views_dir, "params": vars(args), "coverage": coverage,
                         "dashrecon_commit": commit})
    with open(os.path.join(out_dir, "cache.json"), "w") as f:
        json.dump(cache_report, f, indent=2)
    print(f"[gen3c_fill] buffers {args.buffers}: valid fraction {[round(x, 3) for x in coverage]}", flush=True)
    writer = imageio.get_writer(os.path.join(out_dir, "buffers.mp4"), mode="I", fps=args.fps)  # render | buffers, before generation
    for k in range(n):
        row = np.concatenate([renders[k]] + [((warp[k, b] + 1) * 127.5).round().astype(np.uint8) for b in range(buffer_count)], 1)
        writer.append_data(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    writer.close()
    if args.buffers_only:
        return

    pipeline = Gen3cPipeline(
        inference_type="video2world", checkpoint_dir="checkpoints", checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir="Pixtral-12B", enable_prompt_upsampler=False, offload_network=True, offload_tokenizer=True,
        offload_text_encoder_model=True, offload_prompt_upsampler=True, offload_guardrail_models=True,
        disable_guardrail=True, disable_prompt_encoder=args.prompt is None, guidance=args.guidance, num_steps=args.num_steps,
        height=H, width=W, fps=args.fps, num_video_frames=CHUNK, seed=args.seed)
    assert pipeline.model.chunk_size == CHUNK, pipeline.model.chunk_size
    torch.cuda.reset_peak_memory_stats()

    out, chunks, cond, start = [], [], torch.from_numpy(warp[0, 0]).permute(2, 0, 1)[None, :, None].to(device), 0
    while start < n - 1 or start == 0:
        idx = [min(i, n - 1) for i in range(start, start + CHUNK)]
        t0 = time.time()
        imgs = torch.from_numpy(warp[idx]).permute(0, 1, 4, 2, 3)[None].to(device)   # B, F, N, C, H, W
        masks = torch.from_numpy(valid[idx])[None, :, :, None].to(device)            # B, F, N, 1, H, W
        video, _ = pipeline.generate(prompt="" if args.prompt is None else args.prompt, image_path=cond,
                                     rendered_warp_images=imgs, rendered_warp_masks=masks)
        out.extend(list(video) if start == 0 else list(video[1:]))
        cond = torch.from_numpy(video[-1].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device)
        chunks.append({"views": [start, start + CHUNK - 1], "seconds": time.time() - t0})
        print(f"[gen3c_fill] views {start}..{min(start + CHUNK - 1, n - 1)}: {time.time() - t0:.0f} s", flush=True)
        start += CHUNK - 1
    out = out[:n]

    os.makedirs(os.path.join(out_dir, "filled"), exist_ok=True)
    writer = imageio.get_writer(os.path.join(out_dir, "filled_compare.mp4"), mode="I", fps=args.fps)
    for k in range(n):
        Image.fromarray(out[k]).save(os.path.join(out_dir, "filled", f"{k:03d}.png"))
        shown = renders[k].copy()
        shown[holes[k]] = (0.45 * shown[holes[k]] + 0.55 * np.array([230, 40, 40])).astype(np.uint8)
        row = np.concatenate([shown, out[k]], 1)
        writer.append_data(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    writer.close()
    with open(os.path.join(out_dir, "fill.json"), "w") as f:
        json.dump({"generator": "GEN3C-Cosmos-7B", "gen3c_commit": "db2ffe12ced12ddafcec5e0422ee46ce8520746b",
                   "weights_revision": "9bcfdb4f3924f41376daeadf6200826c12a3bf8e", "params": vars(args), "chunks": chunks,
                   "conditioning": "real FRONT image of view 0, then the last generated frame", "image_dir": image_dir,
                   "buffers": args.buffers, "cache_policy": args.cache_policy, "memory_dirs": [str(p) for p in memory_dirs], "buffer_valid_fraction": coverage, "composited": False,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True, "dashrecon_commit": commit}, f, indent=2)
    print(f"[gen3c_fill] {views_dir}: {n} views in {len(chunks)} chunks, {sum(c['seconds'] for c in chunks) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
