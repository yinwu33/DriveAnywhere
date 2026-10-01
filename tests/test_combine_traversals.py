"""Joint-frame recovery and common scale of scripts/combine_traversals.py (synthetic poses)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from combine_traversals import common_scale, to_joint_sfm  # noqa: E402

from dashrecon.pose.world import gravity_aligned_transform  # noqa: E402


def random_poses(rng: np.random.Generator, n: int) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        poses[i, :3, :3] = q * np.sign(np.linalg.det(q))
        poses[i, :3, 3] = rng.normal(size=3) * 10
    return poses


def test_to_joint_sfm_undoes_run_pose() -> None:
    rng = np.random.default_rng(0)
    sfm = random_poses(rng, 12)
    s = 3.7
    scaled = sfm.copy()
    scaled[:, :3, 3] *= s
    world = gravity_aligned_transform(scaled)  # as run_pose.py
    np.testing.assert_allclose(to_joint_sfm(world[None] @ scaled, world, s), sfm, atol=1e-9)


def test_common_scale() -> None:
    assert abs(common_scale(np.array([2.0, 2.0]), np.array([1.0, 5.0])) - 2.0) < 1e-12
    assert abs(common_scale(np.array([1.0, 4.0]), np.array([1.0, 1.0])) - 2.0) < 1e-12
