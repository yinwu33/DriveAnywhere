"""LiDAR of a processed Waymo scene in drivestudio's format, from tfrecords without scene flow (runs in .venvs/waymo).

For the oracle upper bound (DECISIONS Q). drivestudio's `--process_keys lidar` (datasets/waymo/waymo_preprocess.py
save_lidar) needs the scene-flow release: it reads range_image_flow_compressed, which the local perception v1.4.3
tfrecords do not have (DECISIONS C5). This does the same conversion without flow, using the upstream helpers
(parse_range_image_and_camera_projection, datasets/waymo/waymo_utils.extract_point_cloud_from_range_image and
get_ground_np): first return of all five lasers, points in the vehicle frame, TOP laser with per-pixel poses.
Rows are written in drivestudio's 14-column layout (datasets/waymo/waymo_sourceloader.py load_lidar):
    origin (3), point (3), flow (3) = 0, flow class = -1 ("no-flow-label", the dataset's own value for points
    without flow information), ground label, intensity, elongation, laser id.
drivestudio uses flow and flow class for evaluation only (waymo_sourceloader.py: "flow/ground info are used for
evaluation only"), so training is unaffected.

Output: <target_dir>/<scene_idx:03d>/lidar/<frame:03d>.bin (float32), one per frame of the tfrecord; the frame
count is checked against the processed images.

Example:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venvs/waymo/bin/python scripts/extract_lidar.py --raw_dir data/data/validation \
        --file_list data/waymo_val_list.txt --scene_idx 56 --target_dir data/waymo/processed/validation
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waymo_open_dataset import dataset_pb2  # noqa: E402
from waymo_open_dataset.utils import range_image_utils, transform_utils  # noqa: E402
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection  # noqa: E402

from datasets.waymo.waymo_utils import extract_point_cloud_from_range_image, get_ground_np  # noqa: E402


def frame_points(frame) -> np.ndarray:
    """(N, 14) drivestudio LiDAR rows of one frame (first return), without flow."""
    range_images, _, _, range_image_top_pose = parse_range_image_and_camera_projection(frame)
    assert range_image_top_pose is not None, "frame has no LiDAR (camera-only split?)"
    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    top_pose = tf.reshape(tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims)
    top_rot = transform_utils.get_rotation_matrix(top_pose[..., 0], top_pose[..., 1], top_pose[..., 2])
    top_pose = transform_utils.get_transform(top_rot, top_pose[..., 3:])
    rows = []
    for c in sorted(frame.context.laser_calibrations, key=lambda c: c.name):
        ri = range_images[c.name][0]
        if len(c.beam_inclinations) == 0:
            incl = range_image_utils.compute_inclination(tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                                                         height=ri.shape.dims[0])
        else:
            incl = tf.constant(c.beam_inclinations)
        incl = tf.reverse(incl, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
        ri_tensor = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
        pixel_pose, frame_pose_local = None, None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose, frame_pose_local = tf.expand_dims(top_pose, axis=0), tf.expand_dims(frame_pose, axis=0)
        mask_index = tf.where(ri_tensor[..., 0] > 0)
        origins, points = extract_point_cloud_from_range_image(
            tf.expand_dims(ri_tensor[..., 0], axis=0), tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(incl), axis=0), pixel_pose=pixel_pose, frame_pose=frame_pose_local)
        origins = tf.gather_nd(tf.squeeze(origins, axis=0), mask_index).numpy()
        points = tf.gather_nd(tf.squeeze(points, axis=0), mask_index).numpy()
        intensity = tf.gather_nd(ri_tensor[..., 1], mask_index).numpy()
        elongation = tf.gather_nd(ri_tensor[..., 2], mask_index).numpy()
        n = len(points)
        rows.append((origins, points, intensity, elongation, np.full(n, c.name - 1, dtype=np.float32)))
    origins, points, intensity, elongation, laser_ids = (np.concatenate([r[i] for r in rows], axis=0) for i in range(5))
    n = len(points)
    return np.column_stack((origins, points, np.zeros((n, 3)), np.full(n, -1.0), get_ground_np(points), intensity,
                            elongation, laser_ids)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--scene_idx", type=int, required=True)
    parser.add_argument("--target_dir", required=True)
    args = parser.parse_args()
    with open(args.file_list) as f:
        names = [line.strip() for line in f if line.strip()]
    path = os.path.join(args.raw_dir, names[args.scene_idx] + ".tfrecord")
    scene_dir = os.path.join(args.target_dir, f"{args.scene_idx:03d}")
    n_images = len([x for x in os.listdir(os.path.join(scene_dir, "images")) if x.endswith("_0.jpg")])
    os.makedirs(os.path.join(scene_dir, "lidar"), exist_ok=True)
    count = 0
    for frame_idx, data in enumerate(tf.data.TFRecordDataset(path, compression_type="")):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        pts = frame_points(frame)
        pts.tofile(os.path.join(scene_dir, "lidar", f"{frame_idx:03d}.bin"))
        count += 1
        if frame_idx % 50 == 0:
            print(f"[extract_lidar] {args.scene_idx:03d} frame {frame_idx}: {len(pts):,} points", flush=True)
    assert count == n_images, f"{count} LiDAR frames vs {n_images} FRONT images"
    print(f"[extract_lidar] {args.scene_idx:03d}: {count} frames -> {scene_dir}/lidar", flush=True)


if __name__ == "__main__":
    main()
