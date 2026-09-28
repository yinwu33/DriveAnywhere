"""Camera sampling plans for a bounded driving-versus-sweep diagnostic.

Frame coordinates may be fractional: the renderer interpolates estimated FRONT
poses. They are positions along a static scene, not a claim about video time.
"""
import numpy as np


def surrounding_plan(anchors: list[int], pan_views: int, transfer_views: int,
                     side: int) -> list[dict]:
    """Three translated half-circle sweeps; +/- branches cover a virtual rig."""
    if len(anchors) != 3 or anchors != sorted(set(anchors)):
        raise ValueError("three strictly increasing anchors required")
    if side not in [-1, 1] or pan_views < 5 or (pan_views - 1) % 4 or transfer_views < 1:
        raise ValueError("side +/-1, a pan grid including 45-degree cameras, and transfers required")
    coordinates, angles, segments = [], [], []
    for group, anchor in enumerate(anchors):
        lo, hi = (0., side * 180.) if group % 2 == 0 else (side * 180., 0.)
        coordinates.extend([float(anchor)] * pan_views)
        angles.extend(np.linspace(lo, hi, pan_views).tolist())
        segments.extend([f"anchor_{anchor}"] * pan_views)
        if group < 2:
            coordinates.extend(np.linspace(anchor, anchors[group + 1], transfer_views + 2)[1:-1].tolist())
            angles.extend([hi] * transfer_views)
            segments.extend([f"transfer_{group}"] * transfer_views)
    return [{"frame": int(round(t)), "pose_frame": float(t), "yaw": float(a),
             "ramp": float(abs(a) / 180), "segment": s}
            for t, a, s in zip(coordinates, angles, segments)]


def memory_candidate_indices(cams: list[dict], query: dict, radius: int,
                             count: int, policy: str,
                             eligible_indices: list[int] | None = None) -> list[int]:
    """Rank memory for a repeated position by direction as well as distance."""
    if radius < 0 or count < 1:
        raise ValueError("nonnegative frame radius and positive candidate count required")
    pool = list(range(len(cams))) if eligible_indices is None else eligible_indices
    if len(set(pool)) != len(pool) or any(j < 0 or j >= len(cams) for j in pool):
        raise ValueError("eligible memory indices must be distinct and in range")
    if policy == "frame":
        nearest = sorted(pool, key=lambda j: abs(cams[j]["frame"] - query["frame"]))[:count]
        return [j for j in nearest if abs(cams[j]["frame"] - query["frame"]) <= radius]
    if policy != "pose":
        raise ValueError(f"unknown memory candidate policy: {policy}")
    target = np.asarray(query["c2w"])
    eligible = [j for j in pool if abs(cams[j]["frame"] - query["frame"]) <= radius]
    poses = np.asarray([cams[j]["c2w"] for j in eligible])
    if not eligible:
        return []  # explicitly absent memory, reported by the cache builder
    distance = np.linalg.norm(poses[:, :3, 3] - target[:3, 3], axis=1)
    angle = np.degrees(np.arccos(np.clip(poses[:, :3, 2] @ target[:3, 2], -1, 1)))
    order = np.argsort(angle + 2 * distance, kind="stable")[:count]
    return [eligible[j] for j in order]


def sampling_plan(mode: str, anchors: list[int], yaw: float,
                  sweep_views: list[int], transfer_views: int,
                  ramp_views: int) -> list[dict]:
    """Use the same window for driving and three sweeps linked by translation."""
    if len(anchors) != 3 or anchors != sorted(set(anchors)):
        raise ValueError("three strictly increasing anchors required")
    if len(sweep_views) != 3 or min(sweep_views) < 2 or transfer_views < 1:
        raise ValueError("three sweep lengths >= 2 and positive transfer length required")
    count = sum(sweep_views) + 2 * transfer_views
    if not 0 < yaw <= 90 or not 0 < ramp_views < count:
        raise ValueError("yaw in (0, 90] and ramp within the window required")
    coordinates, angles, segments = [], [], []
    if mode == "drive":
        coordinates = np.linspace(anchors[0], anchors[-1], count).tolist()
        angles = (yaw * np.minimum(np.arange(count) / ramp_views, 1)).tolist()
        segments = ["drive"] * count
    elif mode == "sweep":
        for group, (anchor, length) in enumerate(zip(anchors, sweep_views)):
            lo, hi = (0., yaw) if group % 2 == 0 else (yaw, 0.)
            coordinates.extend([float(anchor)] * length)
            angles.extend(np.linspace(lo, hi, length).tolist())
            segments.extend([f"anchor_{anchor}"] * length)
            if group < 2:
                coordinates.extend(np.linspace(anchor, anchors[group + 1], transfer_views + 2)[1:-1].tolist())
                angles.extend([hi] * transfer_views)
                segments.extend([f"transfer_{group}"] * transfer_views)
    else:
        raise ValueError(f"unknown sampling mode: {mode}")
    return [{"frame": int(round(t)), "pose_frame": float(t), "yaw": float(a),
             "ramp": float(a / yaw), "segment": s}
            for t, a, s in zip(coordinates, angles, segments)]


def interpolate_camera(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Linear translation and rotation SLERP, matching utils/camera.py's API."""
    from scipy.spatial.transform import Rotation, Slerp

    if first.shape != (4, 4) or second.shape != (4, 4) or not 0 <= fraction <= 1:
        raise ValueError("two 4x4 poses and a fraction in [0, 1] required")
    out = np.eye(4)
    out[:3, 3] = first[:3, 3] * (1 - fraction) + second[:3, 3] * fraction
    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]])))([fraction]).as_matrix()[0]
    return out


def translated_references(cams: list[dict], k: int, min_baseline: float,
                          max_references: int) -> list[int]:
    """Choose direction-matched references from sufficiently distinct centres.

    Repeated yaw samples at one centre count at most once. All selected centres
    must also be separated from each other, not just from the query.
    """
    if min_baseline <= 0 or max_references < 2:
        raise ValueError("positive baseline and at least two reference slots required")
    poses = np.asarray([c["c2w"] for c in cams])
    centres, directions = poses[:, :3, 3], poses[:, :3, 2]
    distance = np.linalg.norm(centres - centres[k], axis=1)
    angle = np.degrees(np.arccos(np.clip(directions @ directions[k], -1, 1)))
    order = np.argsort(angle + 2 * distance, kind="stable")
    selected = []
    for j in order:
        if distance[j] < min_baseline:
            continue
        if selected and np.any(np.linalg.norm(centres[selected] - centres[j], axis=1) < min_baseline):
            continue
        selected.append(int(j))
        if len(selected) == max_references:
            break
    return selected
