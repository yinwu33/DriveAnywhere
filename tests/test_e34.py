"""E34 pair-pipeline helpers (docs/E34_DATA.md) on synthetic data: frame runs, pair lists, segment selection, pair-grid
intrinsics, world placement of the target cameras, footprint splatting and the GPU-style dilation (run on the CPU)."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e34_calib import longest_run  # noqa: E402
from e34_points_pairs import splat  # noqa: E402
from e34_select import proportional_counts  # noqa: E402
from e34_undistort import pair_intrinsics  # noqa: E402

from dashrecon.e34.calib import cross_pairs, radial_project  # noqa: E402
from dashrecon.e34.geometry import to_world  # noqa: E402
from dashrecon.e34.masks import dilate_gpu  # noqa: E402
from dashrecon.masks.sam_text import disk  # noqa: E402
from dashrecon.pose.world import gravity_aligned_transform  # noqa: E402


def test_longest_run() -> None:
    assert longest_run(np.array([0, 1, 2, 5, 6, 7, 8, 10])) == (5, 9)
    assert longest_run(np.arange(3, 50)) == (3, 50)
    assert longest_run(np.array([0, 1, 3, 4])) == (0, 2)  # first on ties


def test_cross_pairs_window_and_stride() -> None:
    fa, fb = np.arange(0, 20), np.arange(0, 20, 2)
    na, nb = [f"a{t}" for t in fa], [f"b{t}" for t in fb]
    pairs = cross_pairs(na, fa, nb, fb, window=3, key_stride=4)
    keys = sorted({int(x[1:]) for x, _ in pairs})
    assert keys == [0, 4, 8, 12, 16]
    for x, y in pairs:
        assert abs(int(x[1:]) - int(y[1:])) <= 3
    assert ("a8", "b6") in pairs and ("a8", "b10") in pairs and ("a8", "b12") not in pairs


def test_radial_project_matches_pinhole_without_distortion() -> None:
    pts = np.array([[0.3, -0.2, 2.0], [0.0, 0.0, 5.0]])
    uv = radial_project(pts, (1000.0, 960.0, 640.0, 0.0, 0.0))
    np.testing.assert_allclose(uv, [[1110.0, 540.0], [960.0, 640.0]])


def test_proportional_counts() -> None:
    c = proportional_counts({"phx": 166, "sf": 175, "other": 58}, 100)
    assert sum(c.values()) == 100 and c == {"phx": 42, "sf": 44, "other": 14}


def test_pair_intrinsics_match_resized_projection() -> None:
    K = np.array([[2000.0, 0, 959.5], [0, 2000.0, 639.5], [0, 0, 1]])
    kp = pair_intrinsics(K, [1280, 1920], [576, 1024])
    p = np.array([1.0, -0.5, 7.0])
    u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
    # integer-centre pixels: u_small = (u + 0.5) * 1024 / 1920 - 0.5
    up, vp = kp[0, 0] * p[0] / p[2] + kp[0, 2], kp[1, 1] * p[1] / p[2] + kp[1, 2]
    np.testing.assert_allclose([up, vp], [(u + 0.5) * 1024 / 1920 - 0.5, (v + 0.5) * 576 / 1280 - 0.5])


def test_to_world_matches_run_pose_for_front() -> None:
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (6, 1, 1))
    for i in range(6):
        poses[i, :3, 3] = [0.1 * i, 0.01 * i, 1.0 * i]
    scaled = poses.copy()
    scaled[:, :3, 3] *= 3.0
    t = gravity_aligned_transform(scaled)
    meta = {"depth_scale": 3.0, "world_from_backend": t.tolist()}
    np.testing.assert_allclose(to_world(poses, meta), t[None] @ scaled)
    # a rigidly attached second camera keeps its relative pose (rotation) and scaled offset
    rel = np.eye(4)
    rel[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rel[:3, 3] = rng.normal(size=3)
    side = poses @ rel
    w_front, w_side = to_world(poses, meta), to_world(side, meta)
    rel_w = np.linalg.inv(w_front) @ w_side
    np.testing.assert_allclose(rel_w[:, :3, :3], np.broadcast_to(rel[:3, :3], (6, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(rel_w[:, :3, 3], np.broadcast_to(3.0 * rel[:3, 3], (6, 3)), atol=1e-12)


def test_splat_nearest_wins_and_footprint() -> None:
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]])
    c2w = np.eye(4)
    xyz = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0], [0.2, 0.0, 10.0]])
    rgb = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=torch.uint8)
    foot = torch.tensor([0.01, 0.01, 0.31])  # 1 px, 1 px, ~3 px at z = 10
    img, cov = splat(xyz, rgb, foot, K, c2w, (80, 100), max_px=5)
    assert tuple(img[40, 50]) == (0, 255, 0)  # the nearer point hides the farther one
    assert tuple(img[40, 52]) == (0, 0, 255) and tuple(img[41, 53]) == (0, 0, 255) and tuple(img[40, 51]) == (0, 0, 255)
    assert tuple(img[40, 54]) == (0, 0, 0)
    assert abs(cov - 10 / 8000) < 1e-9


def test_dilate_gpu_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    m = rng.random((60, 90)) < 0.01
    m[0, :5] = True
    for r in (0, 1, 5):
        ref = m if r == 0 else binary_dilation(m, structure=disk(r))
        np.testing.assert_array_equal(dilate_gpu(torch.from_numpy(m), r).numpy(), ref)
