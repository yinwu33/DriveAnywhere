"""E34 (docs/E34_DATA.md): close one segment: per-segment meta.json, keep the calibration, delete the intermediates.

Writes <pairs_dir>/meta.json from the stage outputs (each stage records its own runtime):
    fetch (raw fetch.json), masks (masks meta.json), calib (acceptance, registration, reprojection error, GLOMAP
    statistics), undistort (targets.json), pose (MapAnything run_pose meta.json: depth scale, runtime), points
    (points.json), gs (gs.json + the 3DGS meta.json, when present), and the pair counts per type and camera.
For an accepted segment the joint calibration (camera.json, frames.txt, poses_c2w.npy of FRONT and of every target
camera, meta.json; not the COLMAP model) is copied to <pairs_dir>/calib/, and the target cameras' poses in the
reconstruction world to <pairs_dir>/calib/cam<c>/poses_world.npy.
A rejected segment gets a meta.json with the reasons and no pairs.

Then deletes, with --cleanup: the raw images (<raw_root>/<idx:03d>), the original-image masks and the GLOMAP model; and
unless --keep_reconstruction (segments still waiting for "gs" pairs): the undistorted FRONT tree, the MapAnything depth,
the warped FRONT masks, the fusion / mesh directory and the 3DGS run. With --keep_reconstruction only the raw images,
original masks and GLOMAP model go.

Example (any venv):
    python scripts/e34_finalize.py --scene_id wt646 --raw_root data/e34/raw --work_dir data/e34/work/wt646 \
        --front_root data/e34/front --pairs_dir data/e34/pairs/wt646 --cleanup
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.geometry import target_world_poses  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES, FRONT, cam_dir  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402

CALIB = "calib-glomap-cams"
POSE = "pose-glomap-cams_depth-mapanything"
FRONT_MASKS = "mask-gsam2_sky-segformer_img-glomap-cams"
FUSION = f"{POSE}__{FRONT_MASKS}"
GS = "gs"


def copy_calib(calib_dir: str, pose_dir: str, out_dir: str) -> None:
    """Joint calibration without the COLMAP model, plus the target poses in the reconstruction world."""
    os.makedirs(out_dir)
    for f in ("camera.json", "frames.txt", "poses_c2w.npy", "meta.json"):
        shutil.copy(os.path.join(calib_dir, f), out_dir)
    for c_str in io.read_meta(calib_dir)["cameras"]:
        c = int(c_str)
        if c == FRONT:
            continue
        d = cam_dir(out_dir, c)
        shutil.copytree(cam_dir(calib_dir, c), d)
        _, world = target_world_poses(calib_dir, pose_dir, c)
        np.save(os.path.join(d, "poses_world.npy"), world)
    shutil.copy(os.path.join(pose_dir, "meta.json"), os.path.join(out_dir, "pose_meta.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--front_root", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--keep_reconstruction", action="store_true")
    args = parser.parse_args()
    commit = git_commit()
    scene = get_scene(args.scene_id)
    raw_dir = os.path.join(args.raw_root, f"{scene.scene_idx:03d}")
    calib_dir = os.path.join(args.work_dir, CALIB)
    pose_dir = os.path.join(args.work_dir, POSE)
    calib = io.read_meta(calib_dir)
    masks = io.read_meta(os.path.join(args.work_dir, "masks_orig"))
    fetch = io.read_json(os.path.join(raw_dir, "fetch.json"))
    meta = {"scene_id": args.scene_id, "idx": scene.scene_idx, "segment": scene.segment, "accepted": calib["accepted"],
            "reasons": calib["reasons"],
            "registration": {"front_range": calib["front_range"], "front_frac": calib["front_frac"],
                             "target_frac": calib["target_frac"], "per_camera": calib["per_camera"],
                             "models": calib["models"], "points3D": calib["points3D"],
                             "points3D_seen_by_several_cameras": calib["points3D_seen_by_several_cameras"],
                             "mean_reprojection_error_px": calib["mean_reprojection_error_px"],
                             "cameras": calib["cameras"]},
            "timing_s": {"fetch": fetch["seconds"], "masks": masks["runtime_s"], "calib": calib["runtime_total_s"]},
            "masks": {"seconds_per_image": masks["seconds_per_image"], "peak_vram_gb": masks["peak_vram_gb"],
                      "dynamic_mean": {c: v["dynamic_mean"] for c, v in masks["per_camera"].items()}},
            "stage_commits": {"masks": masks["dashrecon_commit"], "calib": calib["dashrecon_commit"]},
            "dashrecon_commit": commit}
    if calib["accepted"]:
        targets = io.read_json(os.path.join(args.pairs_dir, "targets.json"))
        pose = io.read_meta(pose_dir)
        points = io.read_json(os.path.join(args.pairs_dir, "points.json"))
        meta["timing_s"].update({"undistort": targets["runtime_s"], "mapanything": pose["runtime_s"],
                                 "points": sum(points["runtime_s"].values())})
        meta["stage_commits"].update({"undistort": targets["dashrecon_commit"], "pose": pose["dashrecon_commit"],
                                      "points": points["dashrecon_commit"]})
        meta["pose"] = {"depth_scale": pose["depth_scale"], "depth_scale_stats": pose["depth_scale_stats"],
                        "peak_vram_gb": pose["peak_vram_gb"], "valid_depth_fraction": pose["valid_depth_fraction"]}
        meta["pair_hw"] = targets["pair_hw"]
        counts = {"targets": {}, "points": {}}
        for r in targets["targets"]:
            counts["targets"][CAM_NAMES[r["cam"]]] = counts["targets"].get(CAM_NAMES[r["cam"]], 0) + 1
        for r in points["pairs"]:
            counts["points"][CAM_NAMES[r["cam"]]] = counts["points"].get(CAM_NAMES[r["cam"]], 0) + 1
        meta["points"] = {"params": points["params"], "mean_coverage": points["mean_coverage"]}
        gs_json = os.path.join(args.pairs_dir, "gs.json")
        if os.path.exists(gs_json):
            gs = io.read_json(gs_json)
            counts["gs"] = {}
            for r in gs["pairs"]:
                counts["gs"][CAM_NAMES[r["cam"]]] = counts["gs"].get(CAM_NAMES[r["cam"]], 0) + 1
            meta["timing_s"].update({"gs_train": gs["train_meta"]["runtime_s"], "gs_render": sum(gs["runtime_s"].values())})
            meta["stage_commits"]["gs"] = gs["dashrecon_commit"]
            meta["gs"] = {"num_iters": gs["train_meta"]["num_iters"], "opts": gs["train_meta"]["opts"],
                          "peak_vram_gb": gs["train_meta"]["peak_vram_gb"], "mean_coverage": gs["mean_coverage"]}
        meta["pair_counts"] = counts
        if not os.path.exists(os.path.join(args.pairs_dir, "calib")):
            copy_calib(calib_dir, pose_dir, os.path.join(args.pairs_dir, "calib"))
    else:
        os.makedirs(args.pairs_dir, exist_ok=True)
    io.write_meta(args.pairs_dir, meta)
    removed = []
    if args.cleanup:
        gone = [raw_dir, os.path.join(args.work_dir, "masks_orig"), os.path.join(calib_dir, "colmap")]
        if not args.keep_reconstruction:
            gone += [os.path.join(args.front_root, f"{scene.scene_idx:03d}"), pose_dir,
                     os.path.join(args.work_dir, FRONT_MASKS), os.path.join(args.work_dir, FUSION),
                     os.path.join(args.work_dir, GS)]
        for p in gone:
            if os.path.exists(p):
                shutil.rmtree(p)
                removed.append(p)
    print(f"[e34_finalize] {args.scene_id}: {'accepted' if calib['accepted'] else 'rejected'}"
          + (f", pairs {meta['pair_counts']}" if calib["accepted"] else f": {calib['reasons']}")
          + f"; removed {len(removed)} directories", flush=True)


if __name__ == "__main__":
    main()
