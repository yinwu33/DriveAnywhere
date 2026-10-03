"""Target-camera placement for the E34 pairs (docs/E34_DATA.md). Only numpy and dashrecon.io (any venv).

scripts/run_pose.py --camera_dir turns the joint calibration's FRONT SfM poses into the reconstruction world: the
translations are scaled by ``depth_scale`` (SfM units -> MapAnything depth units) and the result is re-expressed with
``world_from_backend`` (gravity aligned). The target cameras' SfM poses live in the same SfM frame, so the same two
steps put them into the reconstruction world, where the FRONT-only reconstruction is rendered or splatted.
"""
import os

import numpy as np

from dashrecon import io
from dashrecon.e34.layout import cam_dir


def to_world(poses_sfm: np.ndarray, pose_meta: dict) -> np.ndarray:
    """SfM camera-to-world poses -> reconstruction world, exactly as scripts/run_pose.py does for FRONT."""
    out = poses_sfm.copy()
    out[:, :3, 3] *= pose_meta["depth_scale"]
    return np.asarray(pose_meta["world_from_backend"])[None] @ out


def target_world_poses(calib_dir: str, pose_dir: str, cam: int) -> tuple[np.ndarray, np.ndarray]:
    """(timesteps, (M,4,4) world camera-to-world poses) of one target camera of a joint calibration."""
    pose_meta = io.read_meta(pose_dir)
    assert pose_meta["camera_dir"] == os.path.relpath(calib_dir), (pose_meta["camera_dir"], calib_dir)
    d = cam_dir(calib_dir, cam)
    return io.read_frames(d), to_world(io.read_poses(d), pose_meta)


FRONT_TOL = 1e-6


def check_front_consistency(calib_dir: str, pose_dir: str) -> float:
    """Max deviation between the pose run's FRONT poses and :func:`to_world` applied to the calibration's FRONT poses
    (they must agree: same transform for every camera)."""
    pose_meta = io.read_meta(pose_dir)
    dev = float(np.abs(to_world(io.read_poses(calib_dir), pose_meta) - io.read_poses(pose_dir)).max())
    assert dev < FRONT_TOL, f"FRONT poses of {pose_dir} are not the transformed calibration poses (max dev {dev})"
    return dev
