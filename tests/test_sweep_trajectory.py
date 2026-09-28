"""Sampling and evidence must not turn a rotation-only sweep into depth votes."""
import numpy as np
import pytest

from dashrecon.gen.consistency import RGBDView, reprojection_evidence
from dashrecon.gen.trajectory import interpolate_camera, sampling_plan, translated_references


def test_sweeps_cover_three_centres_with_continuous_transfers() -> None:
    sweep = sampling_plan("sweep", [96, 104, 112], 60., [36, 36, 37], 6, 20)
    drive = sampling_plan("drive", [96, 104, 112], 60., [36, 36, 37], 6, 20)
    assert len(sweep) == len(drive) == 121
    for anchor in [96, 104, 112]:
        part = [v for v in sweep if v["segment"] == f"anchor_{anchor}"]
        assert all(v["pose_frame"] == anchor for v in part)
        assert min(v["yaw"] for v in part) == 0 and max(v["yaw"] for v in part) == 60
    assert max(abs(b["yaw"] - a["yaw"]) for a, b in zip(sweep, sweep[1:])) < 2
    assert max(b["pose_frame"] - a["pose_frame"] for a, b in zip(sweep, sweep[1:])) < 1.2
    assert drive[20]["yaw"] == 60 and drive[-1]["pose_frame"] == 112


def test_invalid_anchor_order_is_explicit() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sampling_plan("sweep", [104, 96, 112], 60., [36, 36, 37], 6, 20)


def test_camera_interpolation_preserves_rigid_rotation() -> None:
    first, second = np.eye(4), np.eye(4)
    second[:3, 3] = [4, 2, 0]
    second[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    result = interpolate_camera(first, second, .5)
    np.testing.assert_allclose(result[:3, 3], [2, 1, 0])
    np.testing.assert_allclose(result[:3, 2], [2**-.5, 0, 2**-.5], atol=1e-7)
    np.testing.assert_allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-7)


def test_repeated_centre_cannot_supply_multiple_reference_votes() -> None:
    cams = []
    for x in [0., 0., 2., 2., 4., 4.]:
        c = np.eye(4)
        c[0, 3] = x
        cams.append({"c2w": c.tolist()})
    references = translated_references(cams, 0, 1., 4)
    assert len(references) == 2
    assert {cams[j]["c2w"][0][3] for j in references} == {2., 4.}


def test_pure_rotation_has_no_depth_evidence() -> None:
    h, w = 12, 16
    K = np.array([[10., 0, 8], [0, 10., 6], [0, 0, 1.]])
    pose = np.eye(4)
    theta = np.radians(10.)
    rotated = np.eye(4)
    rotated[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    x = (np.arange(w) + .5 - K[0, 2]) / K[0, 0]
    depth = np.broadcast_to(5 / (np.cos(theta) - np.sin(theta) * x), (h, w)).copy()
    rgb, valid = np.full((h, w, 3), .5, np.float32), np.ones((h, w), bool)
    query = RGBDView(rgb, np.full((h, w), 5.), K, pose, valid)
    reference = RGBDView(rgb, depth, K, rotated, valid)
    assert reprojection_evidence(query, reference, .1, .12)[0].any()
    support, conflict = reprojection_evidence(query, reference, .1, .12, .5)
    assert not support.any() and not conflict.any()


def test_translated_plane_retains_parallax_support() -> None:
    h, w = 12, 16
    K = np.array([[10., 0, 8], [0, 10., 6], [0, 0, 1.]])
    pose, translated = np.eye(4), np.eye(4)
    translated[0, 3] = 1.
    rgb, valid = np.full((h, w, 3), .5, np.float32), np.ones((h, w), bool)
    depth = np.full((h, w), 5.)
    query, reference = [RGBDView(rgb, depth, K, p, valid) for p in [pose, translated]]
    support, conflict = reprojection_evidence(query, reference, .1, .12, .5)
    assert support[:, 2:].all() and not support[:, :2].any() and not conflict.any()
