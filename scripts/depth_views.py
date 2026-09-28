"""Phase 9 step 3 (DECISIONS D16-D18): depth for the filled frames of a trajectory (runs in .venvs/mapanything).

MapAnything gets the filled frames (scripts/fill_views.py) with the trajectory's known intrinsics and camera poses
(cams.json, non-metric flag), so its depth is consistent across the frames of the trajectory. It is then scaled to
the scene with the rendered depth where that is reliable (pixels >= --min_count training views saw, render_views.py):
    - each hole (8-connected component of mask/) by the median rendered / MapAnything depth over its ring, the reliable
      pixels within --ring_px outside it, so the filled surface meets the observed one at the hole border (a single
      scale per frame left 13-30 % steps there, val039 round 0); a hole whose ring has fewer than --min_ring_px
      reliable pixels takes the frame scale, and how many did is recorded;
    - everything else by the frame scale, the median over all reliable pixels of the frame;
    - a frame with fewer than --min_ref_px reliable pixels (a view nothing observed, e.g. val056 E10 yaw 90) gets no
      depth (all zero: train_fill.py spawns nothing from it and applies no depth loss there; its filled image still
      supervises colour); such frames are counted and printed.

Output in the views directory: filled_depth/<k:03d>.npy (float16 z-depth on the full render grid, 0 = invalid; nearest
upsampling of the MapAnything grid), depth.json (per frame: frame scale and spread, holes with a ring scale / with the
frame scale, ring error after alignment, valid fraction; parameters; commit).

Example (mapanything venv):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 .venvs/mapanything/bin/python \
        scripts/depth_views.py --views_dir results/E8/val039/views/r0_right1.5_yaw15 --model_id facebook/map-anything
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.gen.depth_alignment import apply_scene_scale, estimate_scene_scale  # noqa: E402

MOGE_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"  # Ruicheng/moge-2-vitl-normal (MIT)


def mapanything_depth(args, paths: list, k_cv: np.ndarray, poses: np.ndarray, h: int, w: int) -> tuple[list, dict]:
    """MapAnything with the trajectory's intrinsics and poses; depth nearest-upsampled to the full grid."""
    from dashrecon.pose.mapanything import MapAnythingBackend

    backend = MapAnythingBackend(args.model_id, args.max_views, 518, args.max_intrinsics_dev)
    res = backend.estimate(paths, intrinsics=k_cv, poses_c2w=poses)
    grid = res.depth_grid
    assert grid["image_hw"] == [h, w], (grid["image_hw"], h, w)
    # full-grid pixel -> depth-grid pixel (nearest), as io.image_to_depth_grid_coords
    ys, xs = np.mgrid[0:h, 0:w]
    ud, vd = io.image_to_depth_grid_coords(xs.astype(np.float64), ys.astype(np.float64), grid)
    ui, vi = np.rint(ud).astype(np.int64), np.rint(vd).astype(np.int64)
    th, tw = grid["depth_hw"]
    inside = (ui >= 0) & (ui < tw) & (vi >= 0) & (vi < th)
    depths = []
    for k in range(len(paths)):
        d = np.zeros((h, w), dtype=np.float64)
        d[inside] = res.depth[k][vi[inside], ui[inside]]
        depths.append(d)
    # the given poses only condition MapAnything (dashrecon/pose/mapanything.py); its depth is consistent with the
    # poses it returns, in its own metric units, so record how far those are from the given ones (same world frame)
    np.save(os.path.join(args.views_dir, "mapanything_poses_c2w.npy"), res.poses_c2w)
    tg = poses[:, :3, 3] - poses[:, :3, 3].mean(0)
    tp = res.poses_c2w[:, :3, 3] - res.poses_c2w[:, :3, 3].mean(0)
    s = float((tp * tg).sum() / (tg * tg).sum())
    rel = np.einsum("nji,njk->nik", poses[:, :3, :3], res.poses_c2w[:, :3, :3])
    rot_deg = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)))
    agreement = {"centre_scale_pred_over_given": s,
                 "centre_rms_residual_given_units": float(np.sqrt(((tp / s - tg) ** 2).sum(1).mean())),
                 "given_centre_rms_spread": float(np.sqrt((tg ** 2).sum(1).mean())),
                 "rotation_error_deg_median": float(np.median(rot_deg)), "rotation_error_deg_max": float(rot_deg.max())}
    return depths, {**res.meta, "given_pose_agreement": agreement}


def moge_depth(args, paths: list, k_cv: np.ndarray, h: int, w: int) -> tuple[list, dict]:
    """MoGe-2 per frame with the known horizontal field of view (runs in .venvs/gen3c, which has MoGe)."""
    import torch
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(args.model_id, revision=MOGE_REVISION).cuda().eval()
    fov_x = float(np.degrees(2 * np.arctan(w / (2 * k_cv[0, 0]))))
    depths = []
    with torch.no_grad():
        for p in paths:
            img = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).cuda()
            d = model.infer(img, fov_x=fov_x)["depth"].float().cpu().numpy().astype(np.float64)
            assert d.shape == (h, w), (d.shape, h, w)
            depths.append(np.where(np.isfinite(d), d, 0.0))
    return depths, {"backend": "moge-2", "model_id": args.model_id, "revision": MOGE_REVISION, "fov_x_deg": fov_x,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--backend", choices=["mapanything", "moge"], default="mapanything")
    parser.add_argument("--model_id", required=True, help="facebook/map-anything or Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--alignment_policy", choices=["boundary", "scene_global"], default="boundary")
    parser.add_argument("--scene_scale", type=float, help="explicit scale reused from an earlier scene_global round")
    parser.add_argument("--max_views", type=int, default=300)
    parser.add_argument("--max_intrinsics_dev", type=float, default=0.5,
                        help="bound on MapAnything's predicted-vs-given intrinsics (val056 yaw 60: focal 27 %% off on the "
                             "mostly generated, weakly structured frames; the depth is re-scaled per hole)")
    parser.add_argument("--ring_px", type=int, default=20)
    parser.add_argument("--min_ring_px", type=int, default=200)
    parser.add_argument("--min_ref_px", type=int, default=1000)
    args = parser.parse_args()
    if args.scene_scale is not None and args.alignment_policy != "scene_global":
        raise ValueError("--scene_scale requires --alignment_policy scene_global")
    commit = git_commit()
    with open(os.path.join(args.views_dir, "cams.json")) as f:
        cams = json.load(f)["cams"]
    paths = [os.path.join(args.views_dir, "filled", f"{k:03d}.png") for k in range(len(cams))]
    h, w = cams[0]["hw"]
    # drivestudio intrinsics use +0.5 pixel centres; dashrecon / MapAnything use integer centres
    k_ds = np.array(cams[0]["K"], dtype=np.float64)
    assert all(np.allclose(np.array(c["K"]), k_ds) for c in cams), "expected one shared intrinsics"
    k_cv = k_ds.copy()
    k_cv[0, 2] -= 0.5
    k_cv[1, 2] -= 0.5
    poses = np.array([c["c2w"] for c in cams], dtype=np.float64)
    if args.backend == "mapanything":
        depths, backend_meta = mapanything_depth(args, paths, k_cv, poses, h, w)
    else:
        depths, backend_meta = moge_depth(args, paths, k_cv, h, w)
    os.makedirs(os.path.join(args.views_dir, "filled_depth"), exist_ok=True)
    if args.alignment_policy == "scene_global":
        references = [np.load(os.path.join(args.views_dir, "depth", f"{k:03d}.npy")).astype(np.float32) for k in range(len(cams))]
        counts = [np.load(os.path.join(args.views_dir, "count", f"{k:03d}.npy")) for k in range(len(cams))]
        estimated, alignment = estimate_scene_scale(depths, references, counts, args.min_count, args.min_ref_px)
        scale = estimated if args.scene_scale is None else args.scene_scale
        alignment.update(estimated_this_round=estimated, scene_scale=scale,
                         scale_reused=args.scene_scale is not None, accepted_geometry=False)
        for k, depth in enumerate(depths):
            aligned = apply_scene_scale(depth, scale)
            np.save(os.path.join(args.views_dir, "filled_depth", f"{k:03d}.npy"), aligned.astype(np.float16))
            alignment["frames"][k].update(valid_fraction=float((aligned > 0).mean()),
                                          requires_geometric_validation=True)
        with open(os.path.join(args.views_dir, "depth.json"), "w") as f:
            json.dump({"model_id": args.model_id, "params": vars(args), "backend_meta": backend_meta,
                       "alignment": alignment, "frames": alignment["frames"], "dashrecon_commit": commit}, f, indent=2)
        print(f"[depth_views] explicit scene-global scale {scale:.5f}; {len(cams)} monocular candidate depths; "
              f"{alignment['reference_pixels']} supporting scale pixels; all novel geometry still unverified", flush=True)
        return
    stats = []
    for k in range(len(cams)):
        d = depths[k]
        rendered = np.load(os.path.join(args.views_dir, "depth", f"{k:03d}.npy")).astype(np.float64)
        count = np.load(os.path.join(args.views_dir, "count", f"{k:03d}.npy"))
        ref = (count >= args.min_count) & (d > 0) & (rendered > 0)
        if ref.sum() < args.min_ref_px:
            np.save(os.path.join(args.views_dir, "filled_depth", f"{k:03d}.npy"), np.zeros_like(d, dtype=np.float16))
            stats.append({"k": k, "no_reference": True, "reference_pixels": int(ref.sum())})
            continue
        ratio = rendered[ref] / d[ref]
        s = float(np.median(ratio))
        out = d * s
        hole = np.asarray(Image.open(os.path.join(args.views_dir, "mask", f"{k:03d}.png"))) > 127
        n_comp, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.ring_px + 1, 2 * args.ring_px + 1))
        ring_scaled, frame_scaled, ring_err = 0, 0, []
        for c in range(1, n_comp):
            comp = lab == c
            ring = (cv2.dilate(comp.astype(np.uint8), ker) > 0) & ~hole & ref
            if ring.sum() < args.min_ring_px:
                frame_scaled += 1
                continue
            sc = float(np.median(rendered[ring] / d[ring]))
            out[comp] = d[comp] * sc
            ring_err.append(float(np.median(np.abs(np.log(rendered[ring] / (d[ring] * sc))))))
            ring_scaled += 1
        np.save(os.path.join(args.views_dir, "filled_depth", f"{k:03d}.npy"), out.astype(np.float16))
        stats.append({"k": k, "scale": s, "median_abs_log_ratio": float(np.median(np.abs(np.log(ratio / s)))),
                      "holes_ring_scaled": ring_scaled, "holes_frame_scaled": frame_scaled,
                      "ring_median_abs_log_ratio": ring_err, "reference_fraction": float(ref.mean()),
                      "valid_fraction": float((d > 0).mean())})
    with open(os.path.join(args.views_dir, "depth.json"), "w") as f:
        json.dump({"model_id": args.model_id, "params": vars(args), "backend_meta": backend_meta, "frames": stats,
                   "dashrecon_commit": commit}, f, indent=2)
    aligned = [s for s in stats if "scale" in s]
    sc = np.array([s["scale"] for s in aligned])
    print(f"[depth_views] {args.views_dir}: {len(stats) - len(aligned)} of {len(stats)} frames without reference pixels (no depth); "
          f"scale {np.median(sc):.3f} (p5 {np.percentile(sc, 5):.3f}, p95 {np.percentile(sc, 95):.3f}), "
          f"median |log ratio| {np.median([s['median_abs_log_ratio'] for s in aligned]):.3f}; holes with a ring scale "
          f"{sum(s['holes_ring_scaled'] for s in aligned)}, with the frame scale {sum(s['holes_frame_scaled'] for s in aligned)}, "
          f"ring |log ratio| after alignment {np.median([e for s in aligned for e in s['ring_median_abs_log_ratio']]):.3f}", flush=True)

if __name__ == "__main__":
    main()
