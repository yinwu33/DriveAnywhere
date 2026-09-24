"""Synthetic tests for Phase 6 Gaussian initialisation helpers (no drivestudio, no data).

Run: PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q
"""
import numpy as np

from dashrecon.train.init import frame_from_normals, rotmat_to_quat_wxyz, sample_mesh


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Same formula as gsplat.cuda_legacy._torch_impl.normalized_quat_to_rotmat (wxyz)."""
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2),
    ], axis=-1).reshape(-1, 3, 3)


def test_quaternion_roundtrip_matches_gsplat_convention() -> None:
    rng = np.random.default_rng(0)
    n = rng.normal(size=(500, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    r = frame_from_normals(n)
    np.testing.assert_allclose(r[:, :, 2], n, atol=1e-12)
    np.testing.assert_allclose(np.einsum("nij,nik->njk", r, r), np.broadcast_to(np.eye(3), (500, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
    np.testing.assert_allclose(quat_wxyz_to_rotmat(rotmat_to_quat_wxyz(r)), r, atol=1e-9)


def test_sample_mesh_area_uniform_and_flat() -> None:
    # two triangles in the z = 0 plane: a big one (area 2) and a small one (area 0.5)
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    rgb = np.full((6, 3), 128, dtype=np.uint8)
    pts, col, quats, log_scales = sample_mesh(v, f, rgb, 20000, 0.2, np.random.default_rng(0))
    frac_big = (pts[:, 0] < 5).mean()
    assert abs(frac_big - 0.8) < 0.02
    np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(col, 128 / 255, atol=1e-12)
    normal_axis = quat_wxyz_to_rotmat(quats)[:, :, 2]
    np.testing.assert_allclose(np.abs(normal_axis[:, 2]), 1.0, atol=1e-9)
    s = np.exp(log_scales)
    np.testing.assert_allclose(s[:, 2] / s[:, 0], 0.2)
    np.testing.assert_allclose(np.pi * s[0, 0] ** 2 * 20000, 2.5, rtol=1e-9)
