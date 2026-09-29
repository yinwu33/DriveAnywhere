"""GEN3C caches with real-first coverage selection and validated generated memory.

Uses the installed GEN3C forward_warp implementation, including its depth-aware
bilinear splatting. All generated RGB-D stays explicitly labelled as synthetic.
"""
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from dashrecon.gen.views import backproject
from dashrecon.gen.trajectory import memory_candidate_indices
from dashrecon.scenes import train_frame_mask


def treat_speckle(rgb: np.ndarray, valid: np.ndarray, depth: np.ndarray, mode: str, window: int,
                  density: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Handle the sparse dots a forward warp leaves where it stretches the source (grazing angles, D-C2).

``density`` of a pixel = fraction of valid pixels in the ``window`` x ``window`` box around it.
    keep: unchanged;
    drop: valid pixels with density < ``density`` become empty (the generator imagines them freely);
    fill: the cache becomes exactly the pixels with density >= ``density``: empty ones among them get the box
          average of the valid colours and depths around them (normalised convolution; foreground and background
          dots are averaged, no depth test), valid ones below it are dropped as in drop.
"""
    if mode == "keep":
        return rgb, valid, depth
    v = valid.astype(np.float32)
    local = cv2.blur(v, (window, window), borderType=cv2.BORDER_CONSTANT)
    if mode == "drop":
        kept = valid & (local >= density)
        return np.where(kept[..., None], rgb, -1.), kept, np.where(kept, depth, 0.)
    if mode != "fill":
        raise ValueError(f"unknown speckle mode {mode}")
    kept = local >= density
    grow = kept & ~valid
    den = np.maximum(local, 1e-6)
    rgb_avg = np.stack([cv2.blur(rgb[..., c] * v, (window, window), borderType=cv2.BORDER_CONSTANT) for c in range(3)], -1) / den[..., None]
    depth_avg = cv2.blur(depth * v, (window, window), borderType=cv2.BORDER_CONSTANT) / den
    rgb = np.where(grow[..., None], rgb_avg, rgb)
    depth = np.where(grow, depth_avg, depth)
    return np.where(kept[..., None], rgb, -1.), kept, np.where(kept, depth, 0.)


def build_consistent_cache(cams: list, views_dir: Path, data_cfg, image_dir: Path,
                           memory_dirs: list[Path], validation_dirs: list[Path],
                           real_frame, memory_radius: int, depth_tol: float,
                           rgb_tol: float, memory_candidate_policy: str = "frame",
                           max_memory_candidates: int = 3, speckle: str = "keep", speckle_window: int = 15,
                           speckle_density: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two buffers: best real source; complementary real source plus memory holes.

Only train FRONT pixels outside estimated dynamic/sky masks enter real caches.
Memory is accepted only on validated original holes and never overwrites real
coverage. The best projected coverage, not a fixed source offset, picks sources.
Every warp (real and memory) first goes through treat_speckle (``speckle`` keep: the E14 cache).
"""
    from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import forward_warp

    if len(memory_dirs) != len(validation_dirs) or memory_radius < 0:
        raise ValueError("one validation directory per memory, and nonnegative radius required")
    memory = []
    positive_indices = []
    for directory, validation in zip(memory_dirs, validation_dirs):
        meta = json.loads((validation / "validation.json").read_text())
        if Path(meta["views_dir"]).resolve() != directory.resolve():
            raise ValueError(f"validation does not belong to {directory}")
        mcams = json.loads((directory / "cams.json").read_text())["cams"]
        if len(meta["frames"]) != len(mcams) or [r["k"] for r in meta["frames"]] != list(range(len(mcams))):
            raise ValueError("validation rows must correspond to memory cameras")
        positive_indices.append([r["k"] for r in meta["frames"] if r["accepted_pixels"] > 0])
        memory.append((directory, validation, mcams))
    h, w = cams[0]["hw"]
    device = torch.device("cuda")
    mask_dir = Path(data_cfg.pixel_source.mask_dir).resolve()
    start = int(data_cfg.start_timestep)
    test_stride = int(data_cfg.pixel_source.test_image_stride)

    @lru_cache(maxsize=12)
    def real_source(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = real_frame(str(image_dir / f"{frame:03d}_0.jpg"))
        depth = np.load(views_dir / "src" / f"{frame:03d}.npy").astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        for kind in ("dynamic", "sky"):
            mask = np.asarray(Image.open(mask_dir / f"mask_{kind}" / f"{frame:06d}.png"))
            resized = cv2.resize(mask, (w, 853), interpolation=cv2.INTER_NEAREST)[75:75 + h]
            valid &= resized < 127
        return rgb, depth, valid

    @lru_cache(maxsize=12)
    def memory_source(d: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        directory, validation, _ = memory[d]
        rgb = np.asarray(Image.open(directory / "filled" / f"{j:03d}.png").convert("RGB"))
        depth = np.load(directory / "filled_depth" / f"{j:03d}.npy").astype(np.float32)
        conf = np.load(validation / "confidence" / f"{j:03d}.npy")
        hole = np.asarray(Image.open(directory / "mask" / f"{j:03d}.png")) > 127
        if depth.shape != (h, w) or rgb.shape != (h, w, 3) or conf.shape != (h, w):
            raise ValueError("memory must use the same render resolution")
        return rgb, depth, hole & (conf > 0) & np.isfinite(depth) & (depth > 0)

    @torch.no_grad()
    def warp(source, source_cam: dict, target_cam: dict):
        rgb, depth, valid = source
        points = backproject(torch.tensor(depth, device=device), torch.tensor(source_cam["K"], device=device),
                             torch.tensor(source_cam["c2w"], device=device)).reshape(1, h, w, 3)
        k = torch.tensor(target_cam["K"], device=device)
        k[0, 2] -= .5
        k[1, 2] -= .5
        rgb_t = torch.tensor(rgb.astype(np.float32) / 127.5 - 1, device=device).permute(2, 0, 1)[None]
        rgb_w, valid_w, depth_w, _ = forward_warp(
            rgb_t, torch.tensor(valid, device=device).float()[None, None], None, None,
            torch.linalg.inv(torch.tensor(target_cam["c2w"], device=device))[None], k[None], k[None],
            world_points1=points, render_depth=True)
        return treat_speckle(rgb_w[0].permute(1, 2, 0).float().cpu().numpy(), valid_w[0, 0].cpu().numpy() > .5,
                             depth_w[0].float().cpu().numpy(), speckle, speckle_window, speckle_density)

    out = np.full((len(cams), 2, h, w, 3), -1., np.float32)
    valid = np.zeros((len(cams), 2, h, w), np.float32)
    rows = []
    skipped_test = set()
    for k, c in enumerate(cams):
        real = []
        frames = set()
        for sc in c["src"]:
            frame = sc["frame"]
            if frame in frames:
                continue
            frames.add(frame)
            if not train_frame_mask(np.array([frame]), start, test_stride)[0]:
                skipped_test.add(frame)
                continue
            real.append((frame, warp(real_source(frame), sc, c)))
        if not real:
            raise ValueError(f"no train FRONT cache candidate for view {k}; render more --src_offsets")
        order = sorted(range(len(real)), key=lambda j: -int(real[j][1][1].sum()))
        first = order[0]
        rgb0, valid0, depth0 = real[first][1]
        out[k, 0], valid[k, 0] = rgb0, valid0
        remainder = [j for j in order if j != first]
        secondary = None
        if remainder:
            secondary = max(remainder, key=lambda j: int((real[j][1][1] & ~valid0).sum()))
            rgb1, valid1, depth1 = real[secondary][1]
            out[k, 1], valid[k, 1] = rgb1, valid1
        else:
            valid1, depth1 = np.zeros((h, w), bool), np.zeros((h, w), np.float32)
        real_union = valid0 | valid1
        best = None
        candidate_rows = []
        for d, (directory, _, mcams) in enumerate(memory):
            pool = positive_indices[d] if memory_candidate_policy == "pose" else None
            nearby = memory_candidate_indices(mcams, c, memory_radius, max_memory_candidates, memory_candidate_policy, pool)
            for j in nearby:
                rgb_m, valid_m, depth_m = warp(memory_source(d, j), mcams[j], c)
                overlap = valid_m & real_union
                ref_depth = np.where(valid0, depth0, depth1)
                ref_rgb = np.where(valid0[..., None], out[k, 0], out[k, 1])
                depth_bad = np.abs(depth_m - ref_depth) > depth_tol * np.maximum(ref_depth, 1e-4)
                rgb_bad = np.abs(rgb_m - ref_rgb).mean(-1) * .5 > rgb_tol
                conflict = overlap & (depth_bad | rgb_bad)
                usable = valid_m & ~real_union
                score = int(usable.sum())
                candidate_rows.append({"dir": str(directory), "k": j, "frame": mcams[j]["frame"],
                                       "new_pixels": score, "real_overlap_pixels": int(overlap.sum()),
                                       "real_conflict_pixels": int(conflict.sum())})
                # Reject an entire memory proposal when its observable overlap contradicts reality.
                if overlap.sum() >= 100 and conflict.sum() / overlap.sum() > .25:
                    continue
                if best is None or score > best[0]:
                    best = (score, rgb_m, usable, candidate_rows[-1])
        selected_memory = None
        memory_pixels = 0
        if best is not None and best[0] > 0:
            memory_pixels, rgb_m, usable, selected_memory = best
            out[k, 1][usable] = rgb_m[usable]
            valid[k, 1][usable] = 1.
        out[k] = np.where(valid[k, ..., None] > .5, out[k], -1.)
        rows.append({"k": k, "real_primary": real[first][0],
                     "real_secondary": None if secondary is None else real[secondary][0],
                     "real_coverage": float(real_union.mean()), "memory_pixels": memory_pixels,
                     "memory_coverage": memory_pixels / (h * w), "selected_memory": selected_memory,
                     "memory_candidates": candidate_rows})
        if k % 10 == 0:
            print(f"[cache] {k}/{len(cams)} real {real_union.mean():.3f} memory {memory_pixels/(h*w):.3f}", flush=True)
    real_source.cache_clear()
    memory_source.cache_clear()
    return out, valid, {"policy": "real_first_coverage_validated_memory", "frames": rows,
                        "memory_candidate_policy": memory_candidate_policy, "max_memory_candidates": max_memory_candidates,
                        "speckle": {"mode": speckle, "window": speckle_window, "density": speckle_density},
                        "positive_memory_frame_indices": positive_indices,
                        "candidate_filter": "positive validation before pose ranking; legacy frame ranking unchanged",
                        "skipped_heldout_source_frames": sorted(skipped_test),
                        "memory_coverage": float(np.mean([r["memory_coverage"] for r in rows])),
                        "validation_dirs": [str(p) for p in validation_dirs]}
