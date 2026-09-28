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


def test_observation_cone_and_gate():
    from dashrecon.gen.floaters import cone_gate, observation_cone

    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    # two cameras 10 units behind the first Gaussian, 0 and ~11.3 degrees off its axis
    centres = torch.tensor([[-10.0, 0.0, 0.0], [-10.0, -2.0, 0.0]])
    weights = torch.tensor([[5.0, 0.0], [5.0, 0.1]])  # the second Gaussian is never seen with >= 1 px
    axis, half, observed = observation_cone(means, centres, weights, min_weight=1.0)
    assert observed.tolist() == [True, False]
    assert 5.0 < float(half[0]) < 6.0  # axis halfway between the two directions (~5.7 deg from each)
    # looking along the axis: kept; from the side (90 deg): gone; unobserved Gaussian: gone everywhere
    g_front = cone_gate(means, torch.tensor([-20.0, 0.0, 0.0]), axis, half, observed, margin=15.0, fade=15.0)
    g_side = cone_gate(means, torch.tensor([0.0, -10.0, 0.0]), axis, half, observed, margin=15.0, fade=15.0)
    assert float(g_front[0]) == 1.0 and float(g_side[0]) == 0.0
    assert float(g_front[1]) == 0.0 and float(g_side[1]) == 0.0
    # halfway through the fade band: a camera whose direction to the Gaussian is the axis turned about z by
    # half_angle + margin + fade / 2
    a = torch.deg2rad(torch.tensor(float(half[0]) + 15.0 + 7.5))
    rot = torch.tensor([[torch.cos(a), -torch.sin(a), 0.0], [torch.sin(a), torch.cos(a), 0.0], [0.0, 0.0, 1.0]])
    centre = means[0] - 10.0 * (rot @ axis[0])
    g_mid = cone_gate(means[:1], centre, axis[:1], half[:1], observed[:1], margin=15.0, fade=15.0)
    assert abs(float(g_mid[0]) - 0.5) < 1e-3, float(g_mid[0])


def test_view_gate_apply_and_roundtrip(tmp_path):
    from dashrecon.gen.floaters import ViewGate
    from models.gaussians.basics import dataclass_gs

    # two gated Gaussians seen from -x, one of them not selected; a third Gaussian added later is never gated
    gate = ViewGate(axis=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), half_angle=torch.tensor([5.0, 5.0]),
                    observed=torch.tensor([True, True]), selected=torch.tensor([True, False]), margin=15.0, fade=15.0)
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    gs = dataclass_gs(_opacities=torch.full((3, 1), 0.8), _means=means, _rgbs=torch.zeros(3, 3),
                      _scales=torch.ones(3, 3), _quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3), detach_keys=[])
    side = gate.apply(gs, torch.tensor([0.0, -10.0, 0.0]))  # 90 degrees off the axis for the first Gaussian
    assert side._opacities[:, 0].tolist() == [0.0, 0.800000011920929, 0.800000011920929]
    front = gate.apply(gs, torch.tensor([-10.0, 0.0, 0.0]))
    assert torch.allclose(front._opacities, gs._opacities)
    gate.save(str(tmp_path / "g.pt"), {"note": "test"})
    back = ViewGate.load(str(tmp_path / "g.pt"), torch.device("cpu"))
    assert back.n == 2 and back.margin == 15.0 and torch.equal(back.selected, gate.selected)
