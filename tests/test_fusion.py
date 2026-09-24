"""Synthetic tests for Phase 5 fusion (no models, no Waymo data).

Run: PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q
"""
import numpy as np

from dashrecon.fusion.backproject import backproject, project
from dashrecon.fusion.cleanup import voxel_downsample
from dashrecon.fusion.consistency import consistency_counts
from dashrecon.fusion.road import smooth_road
from dashrecon.scenes import train_frame_mask

H, W = 48, 64
K = np.array([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]])


def _wall_scene(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cameras stepping forward 0.5 along +z towards a wall at z = 20 (OpenCV frames, identity rotation)."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = 0.5 * np.arange(n)
    depths = np.stack([np.full((H, W), 20.0 - p[2, 3]) for p in poses])
    return depths, np.tile(K, (n, 1, 1)), poses


def test_backproject_project_roundtrip() -> None:
    depths, ks, poses = _wall_scene(1)
    valid = np.zeros((H, W), dtype=bool)
    valid[10, 20] = valid[30, 5] = True
    pts = backproject(depths[0], ks[0], poses[0], valid)
    u, v, z = project(pts, ks[0], poses[0])
    np.testing.assert_allclose(np.stack([v, u], 1), [[10, 20], [30, 5]], atol=1e-9)
    np.testing.assert_allclose(z, 20.0)


def test_consistency_counts_neighbours_and_rejects_outlier() -> None:
    depths, ks, poses = _wall_scene(9)
    depths[4, 20, 30] = 12.0  # a floater in the middle frame: 40% off the wall
    valid = np.zeros((H, W), dtype=bool)
    valid[20, 30] = True  # the floater
    valid[24, 32] = True  # a wall pixel near the image centre, visible in every neighbour
    counts = consistency_counts(depths, ks, poses, valid, i=4, k_neighbors=4, rel_thresh=0.05)
    # row-major order: (20, 30) first, then (24, 32)
    assert counts[0] == 0
    assert counts[1] == 8


def test_voxel_downsample_majority_label() -> None:
    xyz = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.01, 0.02], [1.0, 1.0, 1.0]])
    rgb = np.array([[0, 0, 0], [30, 30, 30], [60, 60, 60], [9, 9, 9]], dtype=np.uint8)
    lab = np.array([1, 1, 0, 0], dtype=np.uint8)
    v_xyz, v_rgb, v_lab = voxel_downsample(xyz, rgb, lab, 0.05)
    order = np.argsort(v_xyz[:, 0])
    np.testing.assert_allclose(v_xyz[order[0]], [0.02, 0.013333333, 0.016666667], atol=1e-6)
    np.testing.assert_array_equal(v_rgb[order[0]], [30, 30, 30])
    np.testing.assert_array_equal(v_lab[order], [1, 0])


def test_smooth_road_flattens_noise_and_keeps_far_points() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 20, size=(20000, 2))
    z = 0.02 * xy[:, 0] + rng.normal(0, 0.03, size=len(xy))  # gentle slope + noise
    xyz = np.column_stack([xy, z])
    xyz[0, 2] += 1.0  # a mislabelled point 1 m above the road
    out, stats = smooth_road(xyz, np.ones(len(xyz), dtype=bool), cell=0.5, sigma_cells=2.0, min_points=5, max_dz=0.3)
    resid_before = np.std(xyz[1:, 2] - 0.02 * xyz[1:, 0])
    resid_after = np.std(out[1:, 2] - 0.02 * out[1:, 0])
    assert resid_after < 0.3 * resid_before
    assert out[0, 2] == xyz[0, 2]
    assert stats["left_unchanged_far"] >= 1


def test_train_frame_mask_matches_drivestudio() -> None:
    frames = np.arange(0, 199)
    train = train_frame_mask(frames, 0, 10)
    np.testing.assert_array_equal(frames[~train], np.arange(10, 199, 10))
    assert train_frame_mask(frames, 0, 0).all()
