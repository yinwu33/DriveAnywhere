"""Synthetic tests for the Phase 9 camera moves and visibility test (dashrecon.gen.views; no drivestudio, no data).

Run: PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q
"""
import math

import torch

from dashrecon.gen.views import ViewMove, backproject, moved_c2w, seen_count


def intrinsics(f: float, h: int, w: int) -> torch.Tensor:
    """drivestudio convention: principal point at the image centre in +0.5 pixel-centre coordinates."""
    return torch.tensor([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def test_parse_rejects_unknown_keys() -> None:
    assert ViewMove.parse("right=1.5,yaw=15") == ViewMove(right=1.5, yaw=15.0)
    try:
        ViewMove.parse("roll=3")
    except AssertionError:
        return
    raise AssertionError("roll should be rejected")


def test_moves_in_the_camera_frame() -> None:
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)  # forward = world x, up = world z
    c2w[:3, 3] = torch.tensor([10.0, 5.0, 2.0], dtype=torch.float64)
    right, down, fwd = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
    m = moved_c2w(c2w, ViewMove(right=1.0, up=2.0, forward=3.0))
    torch.testing.assert_close(m[:3, 3], c2w[:3, 3] + right - 2.0 * down + 3.0 * fwd)
    torch.testing.assert_close(m[:3, :3], c2w[:3, :3])
    # yaw +90: looks where the camera's right axis was; pitch +90: looks up
    torch.testing.assert_close(moved_c2w(c2w, ViewMove(yaw=90.0))[:3, 2], right, atol=1e-12, rtol=0)
    torch.testing.assert_close(moved_c2w(c2w, ViewMove(pitch=90.0))[:3, 2], -down, atol=1e-12, rtol=0)
    r = moved_c2w(c2w, ViewMove(yaw=37.0, pitch=-12.0))[:3, :3]
    torch.testing.assert_close(r.T @ r, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)


def test_backproject_hits_the_depth_plane() -> None:
    h, w, f = 6, 8, 5.0
    k = intrinsics(f, h, w)
    depth = torch.full((h, w), 4.0)
    c2w = moved_c2w(torch.eye(4), ViewMove(right=1.0, yaw=20.0))
    pts = backproject(depth, k, c2w)
    pc = (pts - c2w[:3, 3]) @ c2w[:3, :3]
    torch.testing.assert_close(pc[:, 2], torch.full((h * w,), 4.0))
    # pixel (row 0, col 0) centre is at (0.5, 0.5): x = (0.5 - w / 2) / f * z
    torch.testing.assert_close(pc[0, :2], torch.tensor([(0.5 - w / 2) / f * 4.0, (0.5 - h / 2) / f * 4.0]))


def test_seen_count_occlusion_and_resolution() -> None:
    h, w, f = 12, 16, 10.0
    k = intrinsics(f, h, w)
    depth = torch.full((h, w), 5.0)
    c2w = torch.eye(4)
    pts = backproject(depth, k, c2w)
    ones = torch.ones(h, w, dtype=torch.bool)
    img = torch.full((h, w, 3), 0.5).half()
    rgb = torch.full((h * w, 3), 0.5)
    same = {"w2c": torch.linalg.inv(c2w), "K": k, "depth": depth.half(), "valid": ones, "rgb": img}
    occluded = {"w2c": torch.linalg.inv(c2w), "K": k, "depth": torch.full((h, w), 2.0).half(), "valid": ones, "rgb": img}
    far_c2w = moved_c2w(c2w, ViewMove(forward=-20.0))  # 25 away: more than 3x the novel view's 5
    far = {"w2c": torch.linalg.inv(far_c2w), "K": k, "depth": torch.full((h, w), 25.0).half(), "valid": ones, "rgb": img}
    count = seen_count(pts, depth.reshape(-1), rgb, [same, occluded, far], depth_tol=0.1, res_ratio=3.0, rgb_tol=0.1)
    torch.testing.assert_close(count, torch.ones(h * w))
    count = seen_count(pts, depth.reshape(-1), rgb, [far], depth_tol=0.1, res_ratio=6.0, rgb_tol=0.1)
    assert count.sum() > 0 and bool((count <= 1).all())
    # a view turned away does not see the points
    back = {"w2c": torch.linalg.inv(moved_c2w(c2w, ViewMove(yaw=180.0))), "K": k, "depth": depth.half(), "valid": ones, "rgb": img}
    assert seen_count(pts, depth.reshape(-1), rgb, [back], 0.1, 3.0, 0.1).sum() == 0
    assert math.isclose(float(seen_count(pts, depth.reshape(-1), rgb, [same, same], 0.1, 3.0, 0.1).mean()), 2.0)


def test_seen_count_ignores_masked_observer_pixels() -> None:
    h, w, f = 12, 16, 10.0
    k = intrinsics(f, h, w)
    depth = torch.full((h, w), 5.0)
    pts = backproject(depth, k, torch.eye(4))
    valid = torch.ones(h, w, dtype=torch.bool)
    valid[:, : w // 2] = False  # left half masked as dynamic in the observer
    obs = {"w2c": torch.eye(4), "K": k, "depth": depth.half(), "valid": valid, "rgb": torch.full((h, w, 3), 0.5).half()}
    count = seen_count(pts, depth.reshape(-1), torch.full((h * w, 3), 0.5), [obs], 0.1, 3.0, 0.1).reshape(h, w)
    assert count[:, : w // 2].sum() == 0 and bool((count[:, w // 2:] == 1).all())


def test_seen_count_needs_matching_colour() -> None:
    """A surface at the right depth whose rendered colour differs from the observer's real image (a floater) is unseen."""
    h, w, f = 12, 16, 10.0
    k = intrinsics(f, h, w)
    depth = torch.full((h, w), 5.0)
    pts = backproject(depth, k, torch.eye(4))
    img = torch.full((h, w, 3), 0.5)
    img[: h // 2] = torch.tensor([0.9, 0.2, 0.8])  # top half of the real image is another colour
    obs = {"w2c": torch.eye(4), "K": k, "depth": depth.half(), "valid": torch.ones(h, w, dtype=torch.bool), "rgb": img.half()}
    count = seen_count(pts, depth.reshape(-1), torch.full((h * w, 3), 0.5), [obs], 0.1, 3.0, 0.1).reshape(h, w)
    assert count[: h // 2].sum() == 0 and bool((count[h // 2:] == 1).all())
