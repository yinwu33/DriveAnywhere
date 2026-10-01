"""MT1 (docs/EXPERIMENTS.md): self-calibrate several traversals of the same road jointly in one SfM frame.

Like run_calib.py (DECISIONS D13), but the original FRONT images of every traversal in --scene_ids go into one GLOMAP
reconstruction (dashrecon.pose.calib.run_glomap_multi): one RADIAL camera per traversal, sequential pairs within each
traversal, exhaustive pairs between the keyframes (every --cross_stride-th frame) of different traversals. Nothing
reads calibration, GT pose or LiDAR; which frames see the same place is left to feature matching. Writes, for every
traversal <sid>,
    <out_root>/<sid>/calib-glomap-<group>/   camera.json, frames.txt, poses_c2w.npy (the shared SfM frame and
                                             units), sparse_obs.npz, meta.json
    <undistorted_root>/<scene_idx:03d>/images/<t:03d>_0.jpg   undistorted with that traversal's camera
and the shared model in <out_root>/<group>/calib-glomap-<group>/colmap/sparse/ with meta.json (joint statistics).

Example (sfm venv):
    .venvs/sfm/bin/python scripts/run_calib_multi.py --group mt1 --scene_ids mt1a mt1b \
        --processed_root data/waymo/processed/training --mask_tag mask-gsam2_sky-segformer --out_root data/dashrecon \
        --undistorted_root data/dashrecon/_undistorted/calib-glomap-mt1 --max_features 8192 --overlap 20 \
        --cross_stride 4 --seed 0
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.pose.calib import run_glomap_multi  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import front_image_paths, get_scene  # noqa: E402
from run_calib import EDGE_TOLERANCE_PX  # noqa: E402


def undistort_tree(paths: list[str], names: list[str], K: np.ndarray, dist: np.ndarray, out_img_dir: str) -> dict:
    """Undistort the original images with one traversal's camera (same K, as run_calib.py); returns the map range."""
    h, w = cv2.imread(paths[0]).shape[:2]
    dist5 = np.array([dist[0], dist[1], 0.0, 0.0, 0.0])
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist5, None, K, (w, h), cv2.CV_32FC1)
    outside = max(-0.5 - map_x.min(), map_x.max() - (w - 0.5), -0.5 - map_y.min(), map_y.max() - (h - 0.5), 0.0)
    assert outside <= EDGE_TOLERANCE_PX, f"undistortion samples {outside:.1f} px outside the image: a cropped K would be needed"
    os.makedirs(out_img_dir)
    for name, path in zip(names, paths):
        img = cv2.imread(path)
        assert img.shape[:2] == (h, w), (path, img.shape)
        cv2.imwrite(os.path.join(out_img_dir, name),
                    cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"image_hw": [h, w], "x": [float(map_x.min()), float(map_x.max())],
            "y": [float(map_y.min()), float(map_y.max())], "outside_px": float(outside)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True)
    parser.add_argument("--scene_ids", nargs="+", required=True)
    parser.add_argument("--processed_root", required=True, help="original drivestudio processed split")
    parser.add_argument("--mask_tag", required=True, help="Phase 4 mask tag of the original images")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--undistorted_root", required=True)
    parser.add_argument("--max_features", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--cross_stride", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--view_graph_calibration", action="store_true",
                        help="estimate focal lengths from the view graph before global mapping (dashrecon.pose.calib)")
    args = parser.parse_args()
    assert len(args.scene_ids) >= 2, "a joint calibration needs at least two traversals"
    commit = git_commit()
    import pycolmap

    tag = f"calib-glomap-{args.group}"
    group_dir = io.scene_dir(args.out_root, args.group, tag)
    assert not os.path.exists(group_dir), f"{group_dir} exists; delete it to recalibrate"
    groups, info = [], {}
    for sid in args.scene_ids:
        scene = get_scene(sid)
        img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
        frames, paths = front_image_paths(os.path.dirname(img_dir), scene.start_timestep, scene.end_timestep)
        mask_dir = io.scene_dir(args.out_root, sid, args.mask_tag)
        np.testing.assert_array_equal(io.read_frames(mask_dir), frames)
        calib_dir = io.scene_dir(args.out_root, sid, tag)
        out_img_dir = os.path.join(args.undistorted_root, f"{scene.scene_idx:03d}", "images")
        assert not os.path.exists(calib_dir), f"{calib_dir} exists; delete it to recalibrate"
        assert not os.path.exists(out_img_dir), f"{out_img_dir} exists; delete it to recalibrate"
        names = [os.path.basename(p) for p in paths]
        groups.append({"label": sid, "image_dir": img_dir, "names": names, "frames": frames, "mask_root": mask_dir})
        info[sid] = {"scene": scene, "paths": paths, "names": names, "frames": frames, "mask_dir": mask_dir,
                     "calib_dir": calib_dir, "out_img_dir": out_img_dir}
        print(f"[run_calib_multi] {sid}: {len(frames)} FRONT frames {frames[0]}..{frames[-1]}", flush=True)
    work_dir = os.path.join(group_dir, "colmap")
    os.makedirs(work_dir)

    results, stats = run_glomap_multi(groups, work_dir, args.max_features, args.overlap, args.cross_stride, args.seed,
                                      args.view_graph_calibration)
    # the database, feature masks and image links are large or fully determined by the inputs above
    os.remove(os.path.join(work_dir, "database.db"))
    shutil.rmtree(os.path.join(work_dir, "feature_masks"))
    shutil.rmtree(os.path.join(work_dir, "images"))

    params = {k: getattr(args, k) for k in ("max_features", "overlap", "cross_stride", "seed", "view_graph_calibration")}
    common = {"backend": "glomap", "joint": "one GLOMAP model of all traversals, one camera per traversal",
              "pycolmap": pycolmap.__version__,
              "group": args.group, "traversals": args.scene_ids, "params": params,
              "poses_frame": "SfM world frame and units shared by all traversals of the group (not metric, not gravity-aligned)",
              "uses_oracle": False, "uses_calibration": False,
              "oracle_note": "original FRONT images and Phase 4 masks only; no calibration, GT pose, LiDAR or boxes read",
              "dashrecon_commit": commit}
    for sid in args.scene_ids:
        r, d = results[sid], info[sid]
        und = undistort_tree(d["paths"], d["names"], r.K, r.dist, d["out_img_dir"])
        io.write_camera(d["calib_dir"], {"model": "COLMAP RADIAL, principal point fixed, one camera per traversal",
                                         "image_hw": und["image_hw"], "K": r.K.tolist(), "dist": r.dist.tolist()})
        io.write_frames(d["calib_dir"], d["frames"])
        io.write_poses(d["calib_dir"], r.poses_c2w)
        io.write_sparse_obs(d["calib_dir"], r.obs["frame"], r.obs["u"], r.obs["v"], r.obs["z"])
        io.write_meta(d["calib_dir"], {**common, "scene_id": sid, "segment": d["scene"].segment,
                                       "feature_mask": "dynamic | sky from " + os.path.relpath(d["mask_dir"]),
                                       "undistorted_images": os.path.relpath(d["out_img_dir"]),
                                       "undistort_map_range": und, "num_observations": int(len(r.obs["z"])),
                                       "joint_model": os.path.relpath(group_dir), **r.stats})
        print(f"[run_calib_multi] {sid}: f {r.K[0, 0]:.1f} k1 {r.dist[0]:.4f} k2 {r.dist[1]:.4f} -> {d['calib_dir']}", flush=True)
    io.write_meta(group_dir, {**common, **stats,
                              "members": {sid: os.path.relpath(info[sid]["calib_dir"]) for sid in args.scene_ids}})
    print(f"[run_calib_multi] {args.group}: {stats['points3D']} points, {stats['points3D_seen_by_several_traversals']} seen "
          f"by several traversals, reprojection {stats['mean_reprojection_error_px']:.2f} px, "
          f"{sum(stats['runtime_s'].values()):.0f} s", flush=True)


if __name__ == "__main__":
    main()
