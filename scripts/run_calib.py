"""Phase 3 fix (DECISIONS D13): self-calibrate the FRONT camera from the video itself and undistort the images.

Reads the original FRONT images ``<processed_root>/<scene_idx:03d>/images/<t:03d>_0.jpg`` and the Phase 4 masks
of the same images (no features on dynamic objects or sky); no calibration, GT pose or LiDAR. Writes
    <out_root>/<scene_id>/calib-glomap/     camera.json (shared K, radial k1 k2), frames.txt, poses_c2w.npy
                                             (SfM frame and units), sparse_obs.npz, meta.json, colmap/sparse/
    <undistorted_root>/<scene_idx:03d>/images/<t:03d>_0.jpg
                                             undistorted FRONT images: same size, pinhole K from camera.json
The undistorted tree has the layout of the drivestudio processed split and holds nothing but FRONT images, so
downstream scripts take it as ``--processed_root`` and cannot reach any other file of the original split.
dashrecon.pose.calib explains the COLMAP setup.

Example (sfm venv):
    .venvs/sfm/bin/python scripts/run_calib.py --scene_id val056 --processed_root data/waymo/processed/validation \
        --mask_dir data/dashrecon/val056/mask-gsam2_sky-segformer --out_root data/dashrecon \
        --undistorted_root data/dashrecon/_undistorted/calib-glomap --max_features 8192 --overlap 20 --seed 0
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.pose.calib import run_glomap  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import front_image_paths, get_scene  # noqa: E402

TAG = "calib-glomap"
EDGE_TOLERANCE_PX = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--processed_root", required=True, help="original drivestudio processed split")
    parser.add_argument("--mask_dir", required=True, help="Phase 4 masks of the original images")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--undistorted_root", required=True)
    parser.add_argument("--max_features", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--focal_prior", default=None, help="focal.json of scripts/run_focal_prior.py: fix the focal to it "
                        "(output calib-glomapf/ instead of calib-glomap/; E46)")
    parser.add_argument("--view_graph_calibration", action="store_true",
                        help="estimate focal lengths from the view graph before global mapping (dashrecon.pose.calib)")
    args = parser.parse_args()
    commit = git_commit()
    import pycolmap

    scene = get_scene(args.scene_id)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    frames, paths = front_image_paths(os.path.dirname(img_dir), scene.start_timestep, scene.end_timestep)
    np.testing.assert_array_equal(io.read_frames(args.mask_dir), frames)
    names = [os.path.basename(p) for p in paths]
    focal = None if args.focal_prior is None else float(json.load(open(args.focal_prior))["fx"])
    calib_dir = io.scene_dir(args.out_root, args.scene_id, TAG if focal is None else TAG + "f")
    out_img_dir = os.path.join(args.undistorted_root, f"{scene.scene_idx:03d}", "images")
    assert not os.path.exists(calib_dir), f"{calib_dir} exists; delete it to recalibrate"
    assert not os.path.exists(out_img_dir), f"{out_img_dir} exists; delete it to recalibrate"
    work_dir = os.path.join(calib_dir, "colmap")
    os.makedirs(work_dir)
    print(f"[run_calib] {args.scene_id}: {len(frames)} FRONT frames {frames[0]}..{frames[-1]}", flush=True)

    res = run_glomap(img_dir, names, frames, args.mask_dir, work_dir, args.max_features, args.overlap, args.seed,
                     args.view_graph_calibration, focal=focal)
    # the COLMAP database and feature masks are large and fully determined by the inputs above
    os.remove(os.path.join(work_dir, "database.db"))
    shutil.rmtree(os.path.join(work_dir, "feature_masks"))

    h, w = cv2.imread(paths[0]).shape[:2]
    dist5 = np.array([res.dist[0], res.dist[1], 0.0, 0.0, 0.0])

    def undistort_maps(zoom: float) -> tuple:
        k_out = res.K.copy()
        k_out[0, 0] *= zoom
        k_out[1, 1] *= zoom
        mx, my = cv2.initUndistortRectifyMap(res.K, dist5, None, k_out, (w, h), cv2.CV_32FC1)
        return k_out, mx, my, max(-0.5 - mx.min(), mx.max() - (w - 0.5), -0.5 - my.min(), my.max() - (h - 0.5), 0.0)
    # barrel distortion keeps the undistorted image (same K) inside the original one up to a thin border: allow
    # EDGE_TOLERANCE_PX outside the image (filled by replicating the edge pixel). A lens whose corners would need more
    # (mly03, Mapillary) gets a cropped K instead: the focal of the undistorted images zoomed by the smallest factor that
    # keeps them inside (bisection); the sparse observations, pinhole projections with K, move with it.
    zoom = 1.0
    k_und, map_x, map_y, outside = undistort_maps(zoom)
    if outside > EDGE_TOLERANCE_PX:
        lo, hi = 1.0, 1.5
        assert undistort_maps(hi)[3] <= EDGE_TOLERANCE_PX, f"undistortion {outside:.1f} px outside even at zoom {hi}"
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if undistort_maps(mid)[3] <= EDGE_TOLERANCE_PX else (mid, hi)
        zoom = hi
        print(f"[run_calib] undistortion {outside:.1f} px outside the image at the estimated K: focal zoomed by {zoom:.4f}", flush=True)
        k_und, map_x, map_y, outside = undistort_maps(zoom)
        for key, c in (("u", 0), ("v", 1)):
            res.obs[key] = res.K[c, 2] + zoom * (res.obs[key] - res.K[c, 2])
    os.makedirs(out_img_dir)
    for name, path in zip(names, paths):
        img = cv2.imread(path)
        assert img.shape[:2] == (h, w), (path, img.shape)
        cv2.imwrite(os.path.join(out_img_dir, name),
                    cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    io.write_camera(calib_dir, {"model": "COLMAP RADIAL, principal point fixed", "image_hw": [h, w],
                                "K": k_und.tolist(), "dist": res.dist.tolist(),
                                **({} if zoom == 1.0 else {"K_distorted": res.K.tolist(), "undistort_zoom": zoom})})
    io.write_frames(calib_dir, frames)
    io.write_poses(calib_dir, res.poses_c2w)
    io.write_sparse_obs(calib_dir, res.obs["frame"], res.obs["u"], res.obs["v"], res.obs["z"])
    io.write_meta(calib_dir, {
        "focal_prior": None if focal is None else {"fx": focal, "source": os.path.relpath(args.focal_prior), "fixed": True},
        "backend": "glomap", "pycolmap": pycolmap.__version__, "scene_id": args.scene_id, "segment": scene.segment,
        "params": {k: getattr(args, k) for k in ("max_features", "overlap", "seed", "view_graph_calibration")},
        "camera_model": "RADIAL (f, cx, cy, k1, k2), shared by all frames, principal point fixed at the image centre",
        "feature_mask": "dynamic | sky from " + os.path.relpath(args.mask_dir),
        "poses_frame": "SfM world frame and units (not metric, not gravity-aligned)",
        "undistorted_images": os.path.relpath(out_img_dir),
        "undistort_map_range": {"x": [float(map_x.min()), float(map_x.max())], "y": [float(map_y.min()), float(map_y.max())]},
        "undistort_outside_px": float(outside),
        "undistort_zoom": zoom,
        "num_observations": int(len(res.obs["z"])),
        **res.stats,
        "uses_oracle": False, "uses_calibration": False,
        "oracle_note": "original FRONT images and Phase 4 masks only; no calibration, GT pose, LiDAR or boxes read",
        "dashrecon_commit": commit,
    })
    print(f"[run_calib] {args.scene_id}: f {res.K[0, 0]:.1f} k1 {res.dist[0]:.4f} k2 {res.dist[1]:.4f}, "
          f"{res.stats['points3D']} points, reprojection {res.stats['mean_reprojection_error_px']:.2f} px, "
          f"{sum(res.stats['runtime_s'].values()):.0f} s -> {calib_dir}", flush=True)


if __name__ == "__main__":
    main()
