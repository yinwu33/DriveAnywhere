"""Synthetic unit tests for dashrecon.io and dashrecon.pose.world (no Waymo data needed).

Run: .venvs/mapanything/bin/python -m pytest tests -q
"""
import numpy as np

from dashrecon import io
from dashrecon.pose.world import gravity_aligned_transform

GRID = {"image_hw": [1280, 1920], "resized_hw": [345, 518], "crop_top_left": [4, 0], "depth_hw": [336, 518]}


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_depth_intrinsics_roundtrip() -> None:
    k = np.array([[2070.0, 0, 955.0], [0, 2071.0, 640.0], [0, 0, 1]])
    back = io.image_intrinsics_from_depth(io.depth_intrinsics(k, GRID), GRID)
    np.testing.assert_allclose(back, k, atol=1e-9)


def test_depth_intrinsics_maps_pixels_consistently() -> None:
    """A 3D point must project to corresponding pixels on both grids."""
    k = np.array([[2070.0, 0, 955.0], [0, 2071.0, 640.0], [0, 0, 1]])
    kd = io.depth_intrinsics(k, GRID)
    p = np.array([1.3, -0.4, 10.0])
    u_img, v_img = (k @ p)[:2] / p[2]
    u_d, v_d = (kd @ p)[:2] / p[2]
    sx, sy = 518 / 1920, 345 / 1280
    np.testing.assert_allclose(u_d, (u_img + 0.5) * sx - 0.5 - 0, atol=1e-9)
    np.testing.assert_allclose(v_d, (v_img + 0.5) * sy - 0.5 - 4, atol=1e-9)


def test_io_roundtrip(tmp_path) -> None:
    d = str(tmp_path / "scene" / "tag")
    frames = np.arange(3, 8)
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.arange(5)
    io.write_frames(d, frames)
    io.write_intrinsics(d, np.eye(3))
    io.write_poses(d, poses)
    depth = np.random.default_rng(0).random((4, 6)).astype(np.float32)
    io.write_depth(d, 5, depth)
    mask = depth > 0.5
    io.write_mask(d, "sky", 5, mask)
    io.write_meta(d, {"backend": "test"})
    np.testing.assert_array_equal(io.read_frames(d), frames)
    np.testing.assert_array_equal(io.read_poses(d), poses)
    np.testing.assert_array_equal(io.read_depth(d, 5), depth)
    np.testing.assert_array_equal(io.read_mask(d, "sky", 5), mask)
    assert io.read_meta(d) == {"backend": "test"}


def test_write_ply_header_and_size(tmp_path) -> None:
    path = str(tmp_path / "p.ply")
    xyz = np.zeros((10, 3), dtype=np.float32)
    rgb = np.full((10, 3), 7, dtype=np.uint8)
    io.write_ply(path, xyz, rgb)
    raw = open(path, "rb").read()
    header, body = raw.split(b"end_header\n")
    assert b"element vertex 10" in header
    assert len(body) == 10 * (3 * 4 + 3)


def test_gravity_alignment_recovers_z_up() -> None:
    """Cameras driving along a tilted straight line, level (no roll/pitch relative to that frame)."""
    # OpenCV camera looking along +x of a z-up world: columns = camera x, y, z axes in world
    r_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    tilt = _rot(np.array([0.3, 1.0, 0.2]), 0.4)  # arbitrary backend world rotation
    poses = []
    for i in range(20):
        p = np.eye(4)
        p[:3, :3] = tilt @ r_cam
        p[:3, 3] = tilt @ np.array([2.0 * i, 0.0, 1.5]) + np.array([5.0, -3.0, 2.0])
        poses.append(p)
    poses = np.stack(poses)
    aligned = gravity_aligned_transform(poses)[None] @ poses
    np.testing.assert_allclose(aligned[0, :3, 3], 0, atol=1e-9)
    np.testing.assert_allclose(aligned[:, :3, :3], np.broadcast_to(r_cam, (20, 3, 3)), atol=1e-9)
    np.testing.assert_allclose(aligned[-1, :3, 3], [38.0, 0, 0], atol=1e-9)
