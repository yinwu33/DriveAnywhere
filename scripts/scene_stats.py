"""Per-segment statistics used to pick the development scenes (AGENTS.md Phase 0, task 2).

Reads GT labels and ego poses. That is allowed here: this is scene *selection*, not
reconstruction, and nothing written by this script is read by the reconstruction pipeline.

Run in the waymo venv (TF 2.11 devkit):
    CUDA_VISIBLE_DEVICES="" .venvs/waymo/bin/python scripts/scene_stats.py \
        --raw_dir data/data/validation --file_list data/waymo_val_list.txt \
        --out data/dashrecon/scene_selection/validation_stats.json --workers 12
"""
import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, label_pb2

MOVING_SPEED = 1.0  # m/s, same threshold as the upstream coarse dynamic masks
FRONT_NAME = "FRONT"


def segment_stats(task: tuple[int, str]) -> dict:
    """Compute selection statistics for one tfrecord segment.

    Args:
        task: (scene index in the file list, tfrecord path).

    Returns:
        Dict of per-segment statistics.
    """
    scene_idx, path = task
    positions, yaws = [], []
    moving_front, static_front = [], []
    ctx_stats = None
    for record in tf.data.TFRecordDataset(path, compression_type=""):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())
        if ctx_stats is None:
            ctx_stats = frame.context.stats
        pose = np.array(frame.pose.transform).reshape(4, 4)
        positions.append(pose[:3, 3])
        yaws.append(math.atan2(pose[1, 0], pose[0, 0]))
        n_mov, n_sta = 0, 0
        for label in frame.laser_labels:
            if label.type != label_pb2.Label.TYPE_VEHICLE:
                continue
            if label.most_visible_camera_name != FRONT_NAME:
                continue
            speed = math.hypot(label.metadata.speed_x, label.metadata.speed_y)
            if speed > MOVING_SPEED:
                n_mov += 1
            else:
                n_sta += 1
        moving_front.append(n_mov)
        static_front.append(n_sta)

    positions = np.stack(positions)
    step = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    yaw = np.unwrap(np.array(yaws))
    return {
        "scene_idx": scene_idx,
        "segment": os.path.basename(path)[: -len(".tfrecord")],
        "num_frames": int(len(positions)),
        "time_of_day": ctx_stats.time_of_day,
        "weather": ctx_stats.weather,
        "location": ctx_stats.location,
        "ego_distance_m": float(step.sum()),
        "ego_max_speed_mps": float(step.max() * 10.0),
        "heading_change_deg": float(np.degrees(np.abs(yaw[-1] - yaw[0]))),
        "elevation_change_m": float(positions[:, 2].max() - positions[:, 2].min()),
        "mean_moving_vehicles_front": float(np.mean(moving_front)),
        "mean_static_vehicles_front": float(np.mean(static_front)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True, help="directory with the .tfrecord files")
    parser.add_argument("--file_list", required=True, help="segment names; line i = scene id i")
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    names = open(args.file_list).read().splitlines()
    tasks = [(i, os.path.join(args.raw_dir, f"{n}.tfrecord")) for i, n in enumerate(names)]
    for _, path in tasks:
        assert os.path.exists(path), f"missing segment {path}"
    with Pool(args.workers) as pool:
        stats = []
        for s in pool.imap_unordered(segment_stats, tasks):
            stats.append(s)
            print(f"[{len(stats)}/{len(tasks)}] {s['scene_idx']:03d} {s['time_of_day']:>5} "
                  f"dist={s['ego_distance_m']:.0f}m mov={s['mean_moving_vehicles_front']:.1f} "
                  f"sta={s['mean_static_vehicles_front']:.1f} dyaw={s['heading_change_deg']:.0f} "
                  f"dz={s['elevation_change_m']:.1f}", flush=True)
    stats.sort(key=lambda s: s["scene_idx"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
