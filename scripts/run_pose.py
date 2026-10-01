"""Phase 3 entry point: estimate intrinsics, poses and depth for one scene from FRONT images only.

Two modes (no calibration, GT pose, LiDAR or boxes in either: AGENTS.md section 10, DECISIONS D3):
    - images only: ``<processed_root>/<scene_idx:03d>/images/<t:03d>_0.jpg`` (original images) ->
      ``<out_root>/<scene_id>/pose-<backend>_depth-<backend>/``;
    - with ``--camera_dir`` (a self-calibration from scripts/run_calib.py, DECISIONS D13): ``--processed_root`` is
      the undistorted image tree of that calibration; the backend gets the shared intrinsics and the SfM poses
      as inputs and supplies dense depth. The SfM poses are kept and scaled so that the sparse SfM points agree
      with the backend depth (median of depth / SfM z over all observations) ->
      ``<out_root>/<scene_id>/pose-glomap_depth-<backend>/`` (``calib-glomap-<group>`` from run_calib_multi.py ->
      ``pose-glomap-<group>_depth-<backend>/``: still the scene's own world, see scripts/combine_traversals.py).
Both then re-express the poses in the gravity-aligned world of dashrecon.pose.world.

Example (mapanything venv):
    .venvs/mapanything/bin/python scripts/run_pose.py --scene_id val056 --backend mapanything \
        --model_id facebook/map-anything --processed_root data/dashrecon/_undistorted/calib-glomap \
        --camera_dir data/dashrecon/val056/calib-glomap --out_root data/dashrecon --scale model --max_views 300
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


def umeyama_scale(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """Sim(3) fit of (N,3) points src onto dst: (scale, RMS residual in dst units)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = float(np.trace(np.diag(d) @ sign) / ((xs ** 2).sum() / len(src)))
    res = xd - s * xs @ r.T
    return s, float(np.sqrt((res ** 2).sum(1).mean()))


def depth_scale(obs: dict, frames: np.ndarray, depth: np.ndarray, grid: dict) -> tuple[float, dict]:
    """Scale taking SfM units to the backend depth: median over observations of depth(u, v) / z_sfm."""
    u, v = io.image_to_depth_grid_coords(obs["u"], obs["v"], grid)
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    th, tw = grid["depth_hw"]
    row = np.searchsorted(frames, obs["frame"])
    assert np.array_equal(frames[row], obs["frame"]), "observation frames not in the sequence"
    inside = (ui >= 0) & (ui < tw) & (vi >= 0) & (vi < th)
    d = np.zeros_like(obs["z"])
    d[inside] = depth[row[inside], vi[inside], ui[inside]]
    ok = inside & (d > 0)
    assert ok.sum() > 1000, f"only {ok.sum()} SfM observations fall on valid depth"
    ratio = d[ok] / obs["z"][ok]
    s = float(np.median(ratio))
    per_frame = np.array([np.median(ratio[row[ok] == i]) for i in np.unique(row[ok])])
    log_dev = np.abs(np.log(ratio / s))
    return s, {
        "observations_used": int(ok.sum()), "observations_total": int(len(ok)),
        "median_abs_log_ratio": float(np.median(log_dev)),
        "per_frame_scale_p5_p95": [float(np.percentile(per_frame, 5) / s), float(np.percentile(per_frame, 95) / s)],
        "frames_with_observations": int(len(per_frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--processed_root", required=True, help="drivestudio processed split, or the undistorted tree of --camera_dir")
    parser.add_argument("--camera_dir", default=None, help="self-calibration dir (scripts/run_calib.py); omit for images only")
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
    extra = {}
    if args.camera_dir is None:
        result = backend.estimate(paths, intrinsics=None, poses_c2w=None)
        poses_backend = result.poses_c2w
        tag = f"pose-{args.backend}_depth-{args.backend}"
    else:
        calib_meta = io.read_meta(args.camera_dir)
        assert calib_meta["backend"] == "glomap", calib_meta["backend"]
        assert os.path.relpath(os.path.dirname(paths[0])) == calib_meta["undistorted_images"], (
            f"--processed_root must be the undistorted tree of {args.camera_dir}: {calib_meta['undistorted_images']}")
        np.testing.assert_array_equal(io.read_frames(args.camera_dir), frames)
        camera = io.read_camera(args.camera_dir)
        sfm_poses = io.read_poses(args.camera_dir)
        result = backend.estimate(paths, intrinsics=camera["K"], poses_c2w=sfm_poses)
        s, stats = depth_scale(io.read_sparse_obs(args.camera_dir), frames, result.depth, result.depth_grid)
        poses_backend = sfm_poses.copy()
        poses_backend[:, :3, 3] *= s
        s_pred, res_pred = umeyama_scale(poses_backend[:, :3, 3], result.poses_c2w[:, :3, 3])
        extra = {
            "camera_dir": os.path.relpath(args.camera_dir),
            "calib_commit": calib_meta["dashrecon_commit"],
            "poses_source": "SfM poses of camera_dir, scaled by depth_scale; the backend's own poses are not used",
            "depth_scale": s, "depth_scale_stats": stats,
            "backend_vs_sfm_poses": {"scale": s_pred, "rms_residual": res_pred},
            "distortion_k1_k2": camera["dist"].tolist(),
        }
        # calib-glomap -> pose-glomap_depth-<backend>; calib-glomap-mt1 (joint, run_calib_multi.py) -> pose-glomap-mt1_...
        tag = f"pose-{os.path.basename(os.path.normpath(args.camera_dir)).removeprefix('calib-')}_depth-{args.backend}"
        print(f"[run_pose] {args.scene_id}: depth scale {s:.4f} (median |log ratio| {stats['median_abs_log_ratio']:.3f}, "
              f"{stats['observations_used']} obs); backend poses vs scaled SfM: scale {s_pred:.3f}, rms {res_pred:.3f}", flush=True)

    world_t = gravity_aligned_transform(poses_backend)
    poses = world_t[None] @ poses_backend
    poses, depth, scale = apply_scale_strategy(args.scale, poses, result.depth)

    out_dir = io.scene_dir(args.out_root, args.scene_id, tag)
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
            **extra,
            "uses_oracle": False,
            "uses_calibration": False,
            "oracle_note": "FRONT images (and a self-calibration from them) only; no GT pose / LiDAR / boxes / calibration read",
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
