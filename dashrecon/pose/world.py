"""World-frame conventions for estimated trajectories (AGENTS.md Phase 3, tasks 4 and 6).

drivestudio assumes a metric world with x forward, y left, z up (camera AABB clamping in
``datasets/base/pixel_source.py:754-786`` and the EnvLight sky in ``models/modules.py:186``).
Backends such as MapAnything return the first camera's OpenCV frame (y down), so estimated poses
are re-expressed in a gravity-aligned frame. No calibration is used (DECISIONS D3): "up" is the mean
of the cameras' -y axes, to be refined later with a road-plane fit once road masks exist (Phase 4/5).
"""
import numpy as np

SCALE_STRATEGIES = ("model", "camera_height", "oracle")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    assert n > 1e-8, "degenerate vector"
    return v / n


def gravity_aligned_transform(poses_c2w: np.ndarray) -> np.ndarray:
    """World transform T such that ``T @ poses_c2w`` is x-forward, y-left, z-up.

    Origin = first camera centre; z = mean camera up (-y) direction; x = first camera's optical
    axis projected onto the horizontal plane.

    Args:
        poses_c2w: (N,4,4) OpenCV camera-to-world poses.

    Returns:
        (4,4) rigid transform (float64).
    """
    rots = poses_c2w[:, :3, :3]
    up = _normalize(-rots[:, :, 1].mean(axis=0))
    fwd = rots[0][:, 2]
    x = _normalize(fwd - fwd.dot(up) * up)
    y = np.cross(up, x)
    r_world = np.stack([x, y, up])  # rows: new axes in old coordinates
    t = np.eye(4)
    t[:3, :3] = r_world
    t[:3, 3] = -r_world @ poses_c2w[0, :3, 3]
    return t


def apply_scale_strategy(strategy: str, poses_c2w: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a scale strategy to poses and depth.

    Args:
        strategy: one of ``SCALE_STRATEGIES``.
        poses_c2w: (N,4,4) poses.
        depth: (N,h,w) depth maps.

    Returns:
        (poses, depth, scale factor applied).
    """
    assert strategy in SCALE_STRATEGIES, strategy
    if strategy == "model":
        return poses_c2w, depth, 1.0
    if strategy == "camera_height":
        raise NotImplementedError("camera_height has no height source without calibration (DECISIONS D3, OPEN_QUESTIONS 11)")
    raise NotImplementedError("oracle scale needs the evaluation tools, deferred with Phase 1 (DECISIONS D1)")
