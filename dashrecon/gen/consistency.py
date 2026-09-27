"""Cross-view checks on estimated RGB-D, without GT or a learned confidence model.

Agreement is evidence of self-consistency, never evidence that generated content
is the true hidden scene. Occluded/out-of-frame pixels are unknown, not passes.
Intrinsics use drivestudio's +0.5 pixel centres.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RGBDView:
    """One estimated view on a common image grid, RGB in [0, 1]."""

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    valid: np.ndarray


def reprojection_evidence(query: RGBDView, reference: RGBDView, depth_tol: float,
                          rgb_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Return agreeing and conflicting pixels; all other query pixels are unknown.

Reference depths closer than query points indicate occlusion, hence no evidence.
Query points in reference free space or with inconsistent colours are conflicts.
"""
    if not 0 < depth_tol < 1 or not 0 < rgb_tol <= 1:
        raise ValueError("depth_tol and rgb_tol must be in (0, 1), (0, 1]")
    h, w = query.depth.shape
    rh, rw = reference.depth.shape
    y, x = np.mgrid[:h, :w]
    z = query.depth.astype(np.float32)
    points = np.stack(((x + 0.5 - query.K[0, 2]) / query.K[0, 0] * z,
                       (y + 0.5 - query.K[1, 2]) / query.K[1, 1] * z, z), -1)
    world = points @ query.c2w[:3, :3].T + query.c2w[:3, 3]
    local = (world - reference.c2w[:3, 3]) @ reference.c2w[:3, :3]
    rz = local[..., 2]
    valid = query.valid & np.isfinite(z) & (z > 0) & np.isfinite(local).all(-1) & (rz > 1e-4)
    denom = np.where(valid, rz, 1.0)
    u = reference.K[0, 0] * local[..., 0] / denom + reference.K[0, 2] - 0.5
    v = reference.K[1, 1] * local[..., 1] / denom + reference.K[1, 2] - 0.5
    inside = valid & (u >= 0) & (u <= rw - 1) & (v >= 0) & (v <= rh - 1)
    ui = np.rint(np.where(inside, u, 0)).astype(np.int64)
    vi = np.rint(np.where(inside, v, 0)).astype(np.int64)
    ref_depth = reference.depth[vi, ui]
    usable = inside & reference.valid[vi, ui] & np.isfinite(ref_depth) & (ref_depth > 0)
    delta = (rz - ref_depth) / np.maximum(ref_depth, 1e-4)
    tested = usable & (delta <= depth_tol)  # further points are occluded
    error = np.abs(query.rgb - reference.rgb[vi, ui]).mean(-1)
    agrees = tested & (np.abs(delta) <= depth_tol) & (error <= rgb_tol)
    return agrees, tested & ~agrees


def accepted_confidence(support: np.ndarray, conflict: np.ndarray,
                        min_support: int, max_conflict_fraction: float) -> np.ndarray:
    """Reject unsupported/conflicting pixels; accepted confidence is vote agreement."""
    if min_support < 1 or not 0 <= max_conflict_fraction < 1:
        raise ValueError("min_support >= 1 and max_conflict_fraction in [0, 1) required")
    votes = support.astype(np.float32) + conflict
    ratio = np.divide(support, votes, out=np.zeros_like(votes), where=votes > 0)
    accepted = (support >= min_support) & ((1.0 - ratio) <= max_conflict_fraction)
    return np.where(accepted, ratio, 0).astype(np.float32)
