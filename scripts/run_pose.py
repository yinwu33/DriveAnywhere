"""Phase 3 entry point: estimate intrinsics, poses and depth for one scene from FRONT images only.

Reads only ``<processed_root>/<scene_idx:03d>/images/<t:03d>_0.jpg`` (no GT pose, LiDAR, boxes or
calibration: AGENTS.md section 10, DECISIONS D3) and writes the section-5 products to
``<out_root>/<scene_id>/pose-<backend>_depth-<backend>/``.

Example (mapanything venv):
    .venvs/mapanything/bin/python scripts/run_pose.py --scene_id val056 --backend mapanything \
        --model_id facebook/map-anything --processed_root data/waymo/processed/validation \
        --out_root data/dashrecon --scale model --max_views 300
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.pose.world import SCALE_STRATEGIES, apply_scale_strategy, gravity_aligned_transform  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import front_image_paths, get_scene  # noqa: E402

BACKENDS = ("mapanything",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--processed_root", required=True, help="drivestudio processed split dir")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--scale", required=True, choices=SCALE_STRATEGIES)
    parser.add_argument("--max_views", type=int, required=True)
    parser.add_argument("--resolution_set", type=int, default=518)
    args = parser.parse_args()

    commit = git_commit()  # at start: later edits must not change what this run records
    scene = get_scene(args.scene_id)
    frames, paths = front_image_paths(
        os.path.join(args.processed_root, f"{scene.scene_idx:03d}"), scene.start_timestep, scene.end_timestep
    )
    print(f"[run_pose] {args.scene_id}: {len(paths)} FRONT frames {frames[0]}..{frames[-1]}", flush=True)

    if args.backend == "mapanything":
        from dashrecon.pose.mapanything import MapAnythingBackend

        backend = MapAnythingBackend(args.model_id, args.max_views, args.resolution_set)
    result = backend.estimate(paths, intrinsics=None)

    world_t = gravity_aligned_transform(result.poses_c2w)
    poses = world_t[None] @ result.poses_c2w
    poses, depth, scale = apply_scale_strategy(args.scale, poses, result.depth)

    out_dir = io.scene_dir(args.out_root, args.scene_id, f"pose-{args.backend}_depth-{args.backend}")
    io.write_frames(out_dir, frames)
    io.write_intrinsics(out_dir, result.intrinsics)
    io.write_poses(out_dir, poses)
    for i, t in enumerate(frames):
        io.write_depth(out_dir, int(t), depth[i])
        io.write_depth_conf(out_dir, int(t), result.depth_conf[i])
    meta = dict(result.meta)
    meta.update(
        {
            "scene_id": args.scene_id,
            "segment": scene.segment,
            "image_source": [os.path.relpath(p) for p in (paths[0], paths[-1])],
            "depth_grid": result.depth_grid,
            "world_frame": "x forward, y left, z up; origin = first camera; up = mean camera -y (no road refinement yet)",
            "world_from_backend": world_t.tolist(),
            "scale_strategy": args.scale,
            "scale_factor": scale,
            "uses_oracle": False,
            "uses_calibration": False,
            "oracle_note": "FRONT images only; no GT pose / LiDAR / boxes / calibration read",
            "dashrecon_commit": commit,
        }
    )
    io.write_meta(out_dir, meta)
    print(
        f"[run_pose] wrote {out_dir}: runtime {meta['runtime_s']:.1f}s, peak VRAM {meta['peak_vram_gb']:.1f} GB, "
        f"valid depth {meta['valid_depth_fraction']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
