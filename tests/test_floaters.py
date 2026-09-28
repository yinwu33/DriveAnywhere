"""Synthetic checks of dashrecon.gen.floaters (no data, CPU)."""
import math

import torch

from dashrecon.gen.floaters import needle_ratio, quat_to_rotmat


def test_quat_to_rotmat_matches_axis_angle():
    a = math.radians(30)
    q = torch.tensor([[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]])  # 30 degrees about z
    r = quat_to_rotmat(q)[0]
    expected = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(r, expected, atol=1e-6)


def test_needle_along_and_across_the_ray():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    scales = torch.tensor([[5.0, 1.0, 0.5], [5.0, 1.0, 0.5]])  # long along x
    means = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    centres = torch.zeros(1, 3)
    ratio = needle_ratio(means, scales, identity, centres)
    assert torch.allclose(ratio[0], torch.tensor(5.0))  # ray along x: long axis points at the camera
    assert torch.allclose(ratio[1], torch.tensor(1.0 / 5.0))  # ray along y: long axis across the ray


def test_needle_uses_nearest_centre():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = torch.tensor([[4.0, 1.0, 1.0]])
    means = torch.tensor([[0.0, 0.0, 0.0]])
    centres = torch.tensor([[0.0, 20.0, 0.0], [-3.0, 0.0, 0.0]])  # nearest looks along x
    assert torch.allclose(needle_ratio(means, scales, q, centres), torch.tensor([4.0]))
