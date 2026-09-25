"""Phase 9 step 3 (DECISIONS D16-D18): depth for the filled frames of a trajectory (runs in .venvs/mapanything).

MapAnything gets the filled frames (scripts/fill_views.py) with the trajectory's known intrinsics and camera poses
(cams.json, non-metric flag), so its depth is consistent across the frames of the trajectory. Each frame's depth is
then scaled to the scene by the median of rendered depth / MapAnything depth over the pixels that training views saw
(count >= --min_count, render_views.py), where the rendered depth is reliable; that scale and its spread are recorded.

Output in the views directory: filled_depth/<k:03d>.npy (float16 z-depth on the full render grid, 0 = invalid; nearest
upsampling of the MapAnything grid), depth.json (per-frame scale, spread, valid fraction; parameters; commit).

Example (mapanything venv):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 .venvs/mapanything/bin/python \
        scripts/depth_views.py --views_dir results/E8/val039/views/r0_right1.5_yaw15 --model_id facebook/map-anything
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.pose.mapanything import MapAnythingBackend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=300)
    args = parser.parse_args()
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
    backend = MapAnythingBackend(args.model_id, args.max_views, 518)
    res = backend.estimate(paths, intrinsics=k_cv, poses_c2w=poses)
    grid = res.depth_grid
    assert grid["image_hw"] == [h, w], (grid["image_hw"], h, w)

    # full-grid pixel -> depth-grid pixel (nearest), as io.image_to_depth_grid_coords
    ys, xs = np.mgrid[0:h, 0:w]
    ud, vd = io.image_to_depth_grid_coords(xs.astype(np.float64), ys.astype(np.float64), grid)
    ui, vi = np.rint(ud).astype(np.int64), np.rint(vd).astype(np.int64)
    th, tw = grid["depth_hw"]
    inside = (ui >= 0) & (ui < tw) & (vi >= 0) & (vi < th)
    os.makedirs(os.path.join(args.views_dir, "filled_depth"), exist_ok=True)
    stats = []
    for k in range(len(cams)):
        d = np.zeros((h, w), dtype=np.float64)
        d[inside] = res.depth[k][vi[inside], ui[inside]]
        rendered = np.load(os.path.join(args.views_dir, "depth", f"{k:03d}.npy")).astype(np.float64)
        count = np.load(os.path.join(args.views_dir, "count", f"{k:03d}.npy"))
        ref = (count >= args.min_count) & (d > 0) & (rendered > 0)
        assert ref.sum() > 1000, f"frame {k}: only {ref.sum()} reference pixels"
        ratio = rendered[ref] / d[ref]
        s = float(np.median(ratio))
        np.save(os.path.join(args.views_dir, "filled_depth", f"{k:03d}.npy"), (d * s).astype(np.float16))
        stats.append({"k": k, "scale": s, "median_abs_log_ratio": float(np.median(np.abs(np.log(ratio / s)))),
                      "reference_fraction": float(ref.mean()), "valid_fraction": float((d > 0).mean())})
    with open(os.path.join(args.views_dir, "depth.json"), "w") as f:
        json.dump({"model_id": args.model_id, "params": vars(args), "backend_meta": res.meta, "frames": stats,
                   "dashrecon_commit": commit}, f, indent=2)
    sc = np.array([s["scale"] for s in stats])
    print(f"[depth_views] {args.views_dir}: scale {np.median(sc):.3f} (p5 {np.percentile(sc, 5):.3f}, p95 {np.percentile(sc, 95):.3f}), "
          f"median |log ratio| {np.median([s['median_abs_log_ratio'] for s in stats]):.3f}", flush=True)


if __name__ == "__main__":
    main()
