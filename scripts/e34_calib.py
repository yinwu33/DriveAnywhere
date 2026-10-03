"""E34 (docs/E34_DATA.md): joint self-calibration of all cameras of one training segment, and its acceptance check.

dashrecon.e34.calib.run_glomap_cameras on the original images: FRONT every frame, the target cameras at the frames of
scripts/e34_masks.py, Phase 4 masks as feature masks. No Waymo calibration, ego pose or LiDAR is read (the raw tree
holds images only, scripts/e34_fetch.py). The side cameras' intrinsics, distortion and per-image poses come out in the
frame of the FRONT trajectory; the later FRONT-only reconstruction never sees their images. The principal points of
--refine_pp_cams (SIDE_LEFT / SIDE_RIGHT, whose 1920x886 images are sensor crops) are refined in a final bundle
adjustment, the others stay at the image centre (dashrecon.e34.calib).

Acceptance (both must hold, else the segment is skipped with the reason in meta.json; nothing falls back to GT):
    FRONT    the longest run of consecutive registered FRONT frames covers >= --min_front_frac of the frames, and
             undistorting FRONT with the shared K stays within EDGE_TOLERANCE_PX of the image (as scripts/run_calib.py);
    targets  every target camera registers >= --min_target_frac of its images.
The FRONT run [a, b) becomes the scene: its frames are renumbered from 0 (virtual frame v = t - a) because the FRONT
tree of scripts/e34_undistort.py holds exactly these frames (dashrecon.scenes.training_scene spans all its images).

Writes <out_dir> (e.g. <work>/<scene_id>/calib-glomap-cams):
    meta.json                     accepted / reason, frame range, per-camera registration and reprojection error,
                                  joint statistics, parameters; ``undistorted_images`` names the FRONT tree for
                                  scripts/run_pose.py --camera_dir
    camera.json, frames.txt (virtual), poses_c2w.npy, sparse_obs.npz     FRONT, the scripts/run_calib.py layout
    cam<c>/camera.json, frames.txt (original timesteps), poses_c2w.npy    each target camera (shared SfM frame)
    colmap/sparse/                the GLOMAP model (database, feature masks and image links are deleted)

Example (sfm venv):
    .venvs/sfm/bin/python scripts/e34_calib.py --scene_id wt646 --raw_root data/e34/raw \
        --mask_dir data/e34/work/wt646/masks_orig --out_dir data/e34/work/wt646/calib-glomap-cams \
        --front_root data/e34/front --target_cams 1 2 3 4 --overlap_front 20 --overlap_target 10 \
        --cross 0-1 0-2 1-3 2-4 --cross_window 10 --cross_key_stride 2 --max_features 8192 --seed 0 \
        --num_threads 16 --refine_pp_cams 3 4 --min_front_frac 0.9 --min_target_frac 0.8
"""
import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.calib import run_glomap_cameras  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES, FRONT, cam_dir  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from run_calib import EDGE_TOLERANCE_PX  # noqa: E402


def longest_run(frames: np.ndarray) -> tuple[int, int]:
    """[a, b) of the longest run of consecutive integers in ascending ``frames`` (first one on ties)."""
    assert len(frames) > 0 and np.all(np.diff(frames) > 0), frames
    breaks = np.nonzero(np.diff(frames) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(frames)]])
    i = int(np.argmax(ends - starts))
    return int(frames[starts[i]]), int(frames[ends[i] - 1]) + 1


def undistort_outside_px(K: np.ndarray, dist: np.ndarray, hw: list) -> float:
    """How far (px) undistorting with the same K samples outside the original image (scripts/run_calib.py)."""
    h, w = hw
    map_x, map_y = cv2.initUndistortRectifyMap(K, np.array([dist[0], dist[1], 0.0, 0.0, 0.0]), None, K, (w, h),
                                               cv2.CV_32FC1)
    return float(max(-0.5 - map_x.min(), map_x.max() - (w - 0.5), -0.5 - map_y.min(), map_y.max() - (h - 0.5), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--mask_dir", required=True, help="scripts/e34_masks.py output (cam<c>/ per camera)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--front_root", required=True, help="undistorted FRONT tree written later by e34_undistort.py")
    parser.add_argument("--target_cams", type=int, nargs="+", required=True)
    parser.add_argument("--overlap_front", type=int, required=True)
    parser.add_argument("--overlap_target", type=int, required=True)
    parser.add_argument("--cross", nargs="+", required=True, help="camera pairs a-b matched across cameras")
    parser.add_argument("--cross_window", type=int, required=True, help="frames")
    parser.add_argument("--cross_key_stride", type=int, required=True)
    parser.add_argument("--max_features", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_threads", type=int, required=True)
    parser.add_argument("--refine_pp_cams", type=int, nargs="*", required=True,
                        help="cameras whose principal point the final bundle adjustment refines (dashrecon.e34.calib)")
    parser.add_argument("--min_front_frac", type=float, required=True)
    parser.add_argument("--min_target_frac", type=float, required=True)
    args = parser.parse_args()
    commit = git_commit()
    import pycolmap

    scene = get_scene(args.scene_id)
    assert not os.path.exists(args.out_dir), f"{args.out_dir} exists; delete it to recalibrate"
    img_dir = os.path.join(args.raw_root, f"{scene.scene_idx:03d}", "images")
    groups = []
    for cam in [FRONT] + args.target_cams:
        mroot = cam_dir(args.mask_dir, cam)
        frames = io.read_frames(mroot)
        groups.append({"cam": cam, "label": f"cam{cam}", "image_dir": img_dir, "frames": frames,
                       "names": [f"{t:03d}_{cam}.jpg" for t in frames], "mask_root": mroot,
                       "overlap": args.overlap_front if cam == FRONT else args.overlap_target})
    cross = []
    for spec in args.cross:
        a, b = (int(x) for x in spec.split("-"))
        cross.append((a, b, args.cross_window, args.cross_key_stride))
    work_dir = os.path.join(args.out_dir, "colmap")
    os.makedirs(work_dir)
    print(f"[e34_calib] {args.scene_id}: " + ", ".join(f"{CAM_NAMES[g['cam']]} {len(g['frames'])}" for g in groups), flush=True)
    t0 = time.time()
    res, stats = run_glomap_cameras(groups, cross, work_dir, args.max_features, args.seed, args.num_threads,
                                     args.refine_pp_cams)
    os.remove(os.path.join(work_dir, "database.db"))
    shutil.rmtree(os.path.join(work_dir, "feature_masks"))
    shutil.rmtree(os.path.join(work_dir, "images"))

    n_front = len(groups[0]["frames"])
    reasons = []
    if FRONT in res:
        a, b = longest_run(res[FRONT].frames)
        front_frac = (b - a) / n_front
        outside = undistort_outside_px(res[FRONT].K, res[FRONT].dist, res[FRONT].image_hw)
        if outside > EDGE_TOLERANCE_PX:
            reasons.append(f"FRONT undistortion samples {outside:.1f} px outside the image")
    else:
        a, b, front_frac, outside = 0, 0, 0.0, None
    if front_frac < args.min_front_frac:
        reasons.append(f"FRONT: longest registered run {b - a} of {n_front} frames ({front_frac:.2f} < {args.min_front_frac})")
    target_frac = {}
    for g in groups[1:]:
        reg = len(res[g["cam"]].frames) if g["cam"] in res else 0
        target_frac[str(g["cam"])] = reg / len(g["frames"])
        if target_frac[str(g["cam"])] < args.min_target_frac:
            reasons.append(f"{CAM_NAMES[g['cam']]}: {reg} of {len(g['frames'])} images registered "
                           f"({target_frac[str(g['cam'])]:.2f} < {args.min_target_frac})")
    accepted = not reasons
    cameras = {str(c): {"name": CAM_NAMES[c], "K": r.K.tolist(), "dist": r.dist.tolist(), "image_hw": r.image_hw}
               for c, r in res.items()}
    meta = {
        "backend": "glomap", "joint": "one GLOMAP model of all cameras of the segment, one RADIAL camera per camera, no rig",
        "pycolmap": pycolmap.__version__, "scene_id": args.scene_id, "segment": scene.segment,
        "params": {k: v for k, v in vars(args).items() if k != "scene_id"},
        "accepted": accepted, "reasons": reasons,
        "front_frames_total": n_front, "front_range": [a, b], "frame_offset": a, "front_frac": front_frac,
        "front_undistort_outside_px": outside, "target_frac": target_frac, "cameras": cameras,
        **stats, "runtime_total_s": time.time() - t0,
        "poses_frame": "SfM world frame and units shared by all cameras (not metric, not gravity-aligned)",
        "frames_note": "frames.txt of FRONT holds virtual frames v = t - frame_offset; cam<c>/frames.txt original timesteps",
        "uses_oracle": False, "uses_calibration": False,
        "oracle_note": "original camera images and Phase 4 masks only; no calibration, GT pose, LiDAR or boxes read",
        "dashrecon_commit": commit,
    }
    if accepted:
        f = res[FRONT]
        keep = (f.frames >= a) & (f.frames < b)
        assert keep.sum() == b - a
        meta["undistorted_images"] = os.path.relpath(os.path.join(args.front_root, f"{scene.scene_idx:03d}", "images"))
        io.write_camera(args.out_dir, {"model": "COLMAP RADIAL, principal point fixed, joint with the other cameras",
                                       "image_hw": f.image_hw, "K": f.K.tolist(), "dist": f.dist.tolist()})
        io.write_frames(args.out_dir, f.frames[keep] - a)
        io.write_poses(args.out_dir, f.poses_c2w[keep])
        ok = (f.obs["frame"] >= a) & (f.obs["frame"] < b)
        io.write_sparse_obs(args.out_dir, f.obs["frame"][ok] - a, f.obs["u"][ok], f.obs["v"][ok], f.obs["z"][ok])
        for c in args.target_cams:
            r, d = res[c], cam_dir(args.out_dir, c)
            io.write_camera(d, {"model": "COLMAP RADIAL, principal point fixed, joint with FRONT", "image_hw": r.image_hw,
                                "K": r.K.tolist(), "dist": r.dist.tolist()})
            io.write_frames(d, r.frames)
            io.write_poses(d, r.poses_c2w)
    io.write_meta(args.out_dir, meta)
    print(f"[e34_calib] {args.scene_id}: {'ACCEPTED' if accepted else 'REJECTED ' + '; '.join(reasons)}; FRONT run "
          f"[{a}, {b}) of {n_front}; " + ", ".join(f"{CAM_NAMES[int(c)]} {v['registered']}/{v['images']}"
                                                 + (f" {v['mean_reprojection_error_px']:.2f}px" if v['registered'] else "")
                                                 for c, v in stats["per_camera"].items())
          + f"; {stats['models']} model(s), {sum(stats['runtime_s'].values()):.0f} s", flush=True)


if __name__ == "__main__":
    main()
