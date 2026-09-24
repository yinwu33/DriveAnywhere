"""Multi-view depth consistency filter (AGENTS.md Phase 5, task 2).

Each valid pixel of frame i is back-projected and projected into the K previous and K next frames of
the same (training-only) sequence. It is consistent with frame j when it lands inside j, in front of
the camera, on a pixel with valid depth d_j, and |z_j - d_j| / d_j < rel_thresh. Pixels consistent
with at least ``min_consistent`` neighbours are kept.
"""
import numpy as np

from dashrecon.fusion.backproject import backproject, project


def consistency_counts(depths: np.ndarray, ks: np.ndarray, poses: np.ndarray, valid: np.ndarray,
                       i: int, k_neighbors: int, rel_thresh: float) -> np.ndarray:
    """Number of neighbouring frames each valid pixel of frame ``i`` is consistent with.

    Args:
        depths: (N,h,w) z-depth maps; 0 = invalid (neighbour depth used for checking).
        ks: (N,3,3) intrinsics on the depth grid.
        poses: (N,4,4) camera-to-world poses.
        valid: (h,w) bool, the pixels of frame i to test.
        i: index of the reference frame in the arrays.
        k_neighbors: frames on each side to check.
        rel_thresh: relative depth tolerance.

    Returns:
        (M,) int counts for the valid pixels of frame i, in row-major order of ``valid``.
    """
    n, h, w = depths.shape
    pts = backproject(depths[i], ks[i], poses[i], valid)
    counts = np.zeros(len(pts), dtype=np.int32)
    for j in range(max(0, i - k_neighbors), min(n, i + k_neighbors + 1)):
        if j == i:
            continue
        u, v, z = project(pts, ks[j], poses[j])
        ui, vi = np.rint(u), np.rint(v)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dj = np.zeros(len(pts))
        dj[inside] = depths[j][vi[inside].astype(np.int64), ui[inside].astype(np.int64)]
        ok = inside & (dj > 0)
        ok[ok] = np.abs(z[ok] - dj[ok]) / dj[ok] < rel_thresh
        counts += ok
    return counts
