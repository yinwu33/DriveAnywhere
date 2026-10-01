"""MT1 (docs/EXPERIMENTS.md): merge the traversals of a group into one virtual scene with one metric, gravity-aligned world.

Inputs for every member <sid> of --group (dashrecon.scenes.MULTI_TRAVERSALS), all from FRONT images only:
    <out_root>/<sid>/<pose_tag>/    run_pose.py --camera_dir <sid>/calib-glomap-<group>: intrinsics, poses_c2w,
                                    depth/, depth_conf/, meta.json (depth_scale s_t, world_from_backend W_t, depth_grid)
    <out_root>/<sid>/<mask_tag>/    run_masks.py on the undistorted images: mask_dynamic / mask_sky / mask_road
    <undistorted_root>/<idx>/images/<t:03d>_0.jpg
run_pose.py handled each member on its own: it scaled the member's part of the joint SfM frame to that member's
MapAnything depth (s_t) and gravity-aligned it (W_t). Here the members go back to the joint SfM frame
(P_sfm = W_t^-1 P, translation / s_t), get one common scale s_c (the observation-weighted geometric mean of the s_t;
each member's depth is multiplied by s_c / s_t to stay consistent with its poses) and one gravity alignment over all
cameras (dashrecon.pose.world.gravity_aligned_transform; origin = first camera of the first member). Frames are
renumbered member after member: virtual frame = member offset + (t - member's first frame).

Outputs (the layout of a single scene, so dashrecon.train.pixel_source, run_fusion.py and train_gs.py read them as is):
    <out_root>/<group>/<pose_tag>/   frames.txt, intrinsics.npy (N,3,3), poses_c2w.npy, depth/, depth_conf/, meta.json
                                     (members, frame ranges, s_t, s_c, the world transform)
    <out_root>/<group>/<mask_tag>/   mask_dynamic / mask_sky / mask_road, frames.txt, meta.json
    <undistorted_root>/<group idx>/images/<t:03d>_0.jpg   links to the members' undistorted images

Example (any venv with numpy / PIL):
    .venvs/main/bin/python scripts/combine_traversals.py --group mt1 --pose_tag pose-glomap-mt1_depth-mapanything \
        --mask_tag mask-gsam2_sky-segformer_img-glomap-mt1 --out_root data/dashrecon \
        --undistorted_root data/dashrecon/_undistorted/calib-glomap-mt1
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.pose.world import gravity_aligned_transform  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, MULTI_TRAVERSALS, get_scene  # noqa: E402

MASK_KINDS = ("dynamic", "sky", "road")


def common_scale(scales: np.ndarray, weights: np.ndarray) -> float:
    """Weighted geometric mean of the members' depth scales."""
    assert scales.shape == weights.shape and (scales > 0).all() and (weights > 0).all(), (scales, weights)
    return float(np.exp((weights * np.log(scales)).sum() / weights.sum()))


def to_joint_sfm(poses: np.ndarray, world_from_backend: np.ndarray, s: float) -> np.ndarray:
    """Undo run_pose.py's per-member gravity alignment and depth scale: poses in the joint SfM frame and units."""
    out = np.linalg.inv(world_from_backend)[None] @ poses
    out[:, :3, 3] /= s
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, choices=list(MULTI_TRAVERSALS))
    parser.add_argument("--pose_tag", required=True)
    parser.add_argument("--mask_tag", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--undistorted_root", required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    members = MULTI_TRAVERSALS[args.group]
    group = get_scene(args.group)
    pose_out = io.scene_dir(args.out_root, args.group, args.pose_tag)
    mask_out = io.scene_dir(args.out_root, args.group, args.mask_tag)
    img_out = os.path.join(args.undistorted_root, f"{group.scene_idx:03d}", "images")
    for d in (pose_out, mask_out, img_out):
        assert not os.path.exists(d), f"{d} exists; delete it to recombine"

    src = []
    for sid in members:
        pose_dir = io.scene_dir(args.out_root, sid, args.pose_tag)
        mask_dir = io.scene_dir(args.out_root, sid, args.mask_tag)
        meta = io.read_meta(pose_dir)
        frames = io.read_frames(pose_dir)
        np.testing.assert_array_equal(io.read_frames(mask_dir), frames)
        assert meta["camera_dir"] == os.path.relpath(io.scene_dir(args.out_root, sid, f"calib-glomap-{args.group}")), (
            f"{pose_dir} was not made from the joint calibration of {args.group}: {meta['camera_dir']}")
        assert meta["scale_strategy"] == "model" and meta["scale_factor"] == 1.0, meta["scale_strategy"]
        k = io.read_intrinsics(pose_dir)
        k = np.broadcast_to(k, (len(frames), 3, 3)) if k.ndim == 2 else k
        assert k.shape == (len(frames), 3, 3), k.shape
        src.append({"sid": sid, "pose_dir": pose_dir, "mask_dir": mask_dir, "meta": meta, "frames": frames, "K": k,
                    "sfm": to_joint_sfm(io.read_poses(pose_dir), np.array(meta["world_from_backend"]), meta["depth_scale"]),
                    "s": float(meta["depth_scale"]), "obs": int(meta["depth_scale_stats"]["observations_used"]),
                    "img_dir": os.path.join(args.undistorted_root, f"{get_scene(sid).scene_idx:03d}", "images")})
    grid = src[0]["meta"]["depth_grid"]
    for m in src[1:]:
        assert m["meta"]["depth_grid"] == grid, (m["sid"], m["meta"]["depth_grid"], grid)

    s_c = common_scale(np.array([m["s"] for m in src]), np.array([m["obs"] for m in src], dtype=np.float64))
    poses = np.concatenate([m["sfm"] for m in src])
    poses[:, :3, 3] *= s_c
    world = gravity_aligned_transform(poses)
    poses = world[None] @ poses
    n = sum(len(m["frames"]) for m in src)
    scales = ", ".join(f"{m['sid']} {m['s']:.4f}" for m in src)
    print(f"[combine] {args.group}: {n} frames from {members}; depth scales {scales} -> common {s_c:.4f}", flush=True)

    ranges, offset = {}, 0
    os.makedirs(img_out)
    for m in src:
        ratio = s_c / m["s"]
        for i, t in enumerate(m["frames"]):
            v = offset + i
            io.write_depth(pose_out, v, io.read_depth(m["pose_dir"], int(t)) * np.float32(ratio))
            io.write_depth_conf(pose_out, v, io.read_depth_conf(m["pose_dir"], int(t)))
            for kind in MASK_KINDS:
                io.write_mask(mask_out, kind, v, io.read_mask(m["mask_dir"], kind, int(t)))
            os.symlink(os.path.abspath(os.path.join(m["img_dir"], f"{int(t):03d}_{FRONT_CAM_ID}.jpg")),
                       os.path.join(img_out, f"{v:03d}_{FRONT_CAM_ID}.jpg"))
        ranges[m["sid"]] = {"virtual_frames": [offset, offset + len(m["frames"])],
                            "member_frames": [int(m["frames"][0]), int(m["frames"][-1]) + 1],
                            "depth_scale": m["s"], "depth_multiplier": ratio, "pose_dir": os.path.relpath(m["pose_dir"]),
                            "mask_dir": os.path.relpath(m["mask_dir"]), "pose_commit": m["meta"]["dashrecon_commit"]}
        offset += len(m["frames"])
    frames = np.arange(n)
    io.write_frames(pose_out, frames)
    io.write_intrinsics(pose_out, np.concatenate([m["K"] for m in src]))
    io.write_poses(pose_out, poses)
    io.write_frames(mask_out, frames)
    common = {"group": args.group, "members": list(members), "traversals": ranges, "uses_oracle": False,
              "uses_calibration": False,
              "oracle_note": "members' FRONT images and products only; frames are matched by SfM, no GT pose read",
              "dashrecon_commit": commit}
    io.write_meta(pose_out, {
        **common, "backend": "combined", "depth_grid": grid, "scale_strategy": "model", "scale_factor": 1.0,
        "depth_scale": s_c, "depth_scale_rule": "observation-weighted geometric mean of the members' depth scales",
        "world_frame": "x forward, y left, z up; origin = first camera of the first member; up = mean camera -y of all members",
        "world_from_joint_sfm_scaled": world.tolist(), "runtime_s": time.time() - t0})
    io.write_meta(mask_out, {**common, "backend": "combined", "kinds": list(MASK_KINDS)})
    centres = poses[:, :3, 3]
    lengths = ", ".join(f"{sid} {np.linalg.norm(np.diff(centres[r['virtual_frames'][0]:r['virtual_frames'][1]], axis=0), axis=1).sum():.1f}"
                        for sid, r in ranges.items())
    print(f"[combine] path length per member (metres): {lengths} -> {pose_out}, {mask_out}, {img_out} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
