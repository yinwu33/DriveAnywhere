"""Geometry checks must distinguish occlusion, contradiction and absent evidence."""
import numpy as np

from dashrecon.gen.consistency import RGBDView, accepted_confidence, reprojection_evidence


def plane(z: float = 5.0, x: float = 0.0) -> RGBDView:
    c2w = np.eye(4)
    c2w[0, 3] = x
    return RGBDView(np.full((12, 16, 3), 0.5, np.float32), np.full((12, 16), z, np.float32),
                    np.array([[10., 0, 8], [0, 10, 6], [0, 0, 1]]), c2w, np.ones((12, 16), bool))


def test_shifted_plane_support_and_outside_unknown() -> None:
    support, conflict = reprojection_evidence(plane(), plane(x=1.), 0.1, 0.12)
    assert support[:, 2:].all() and not support[:, :2].any()
    assert not conflict.any()


def test_occluded_points_are_unknown_but_free_space_is_conflict() -> None:
    support, conflict = reprojection_evidence(plane(), plane(z=2.), 0.1, 0.12)
    assert not support.any() and not conflict.any()
    support, conflict = reprojection_evidence(plane(), plane(z=8.), 0.1, 0.12)
    assert not support.any() and conflict.all()


def test_colour_conflict_and_invalid_depth_are_not_support() -> None:
    ref = plane()
    ref.rgb[:6] = 0.9
    ref.depth[6:] = 0
    support, conflict = reprojection_evidence(plane(), ref, 0.1, 0.12)
    assert not support.any() and conflict[:6].all() and not conflict[6:].any()


def test_confidence_requires_independent_support_and_limits_conflicts() -> None:
    conf = accepted_confidence(np.array([0, 1, 2, 2, 3]), np.array([0, 0, 0, 2, 1]), 2, .25)
    np.testing.assert_allclose(conf, [0, 0, 1, 0, .75])
