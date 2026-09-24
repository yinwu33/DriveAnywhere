"""Road smoothing (AGENTS.md Phase 5, task 4).

Road points are snapped onto a smooth height field z = f(x, y) in the gravity-aligned world frame
(x forward, y left, z up; dashrecon/pose/world.py). The field is the per-cell median height on a
regular grid, smoothed with a Gaussian by normalised convolution (empty cells do not pull the surface
down), and evaluated bilinearly. This is a smoothed piecewise-constant field rather than the literal
piecewise-plane or B-spline examples in AGENTS.md (recorded in DECISIONS H). Points further than
``max_dz`` from the field are treated as mislabelled (curbs, car undersides) and left unchanged.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def cell_medians(xy: np.ndarray, z: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median z per grid cell.

    Returns:
        (origin (2,), median grid (H,W) with NaN for empty cells, count grid (H,W)).
    """
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 1
    key = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.lexsort((z, key))
    key_s, z_s = key[order], z[order]
    starts = np.flatnonzero(np.r_[True, np.diff(key_s) != 0])
    counts = np.diff(np.r_[starts, len(key_s)])
    medians = z_s[starts + (counts - 1) // 2]
    grid = np.full(shape[0] * shape[1], np.nan)
    count_grid = np.zeros(shape[0] * shape[1], dtype=np.int64)
    grid[key_s[starts]] = medians
    count_grid[key_s[starts]] = counts
    return origin, grid.reshape(shape), count_grid.reshape(shape)


def smooth_road(xyz: np.ndarray, is_road: np.ndarray, cell: float, sigma_cells: float, min_points: int,
                max_dz: float) -> tuple[np.ndarray, dict]:
    """Snap road points onto a smoothed height field.

    Args:
        xyz: (N,3) points (world frame, z up).
        is_road: (N,) bool.
        cell: grid cell size (scene units).
        sigma_cells: Gaussian smoothing sigma in cells.
        min_points: cells with fewer road points do not contribute to the field.
        max_dz: only points within this vertical distance of the field are moved.

    Returns:
        (new xyz, stats dict).
    """
    road = xyz[is_road]
    origin, med, cnt = cell_medians(road[:, :2], road[:, 2], cell)
    w = (cnt >= min_points).astype(np.float64)
    num = gaussian_filter(np.where(w > 0, med, 0.0), sigma_cells, mode="constant")
    den = gaussian_filter(w, sigma_cells, mode="constant")
    field = np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)
    # grid index coordinates of cell centres: (p - origin) / cell - 0.5
    coords = ((road[:, :2] - origin) / cell - 0.5).T
    z_fit = map_coordinates(field, coords, order=1, mode="nearest")
    dz = z_fit - road[:, 2]
    move = np.isfinite(dz) & (np.abs(dz) <= max_dz)
    assert move.any(), "road smoothing moved no points: check the road masks and max_dz"
    out = xyz.copy()
    road_idx = np.flatnonzero(is_road)
    out[road_idx[move], 2] = z_fit[move]
    stats = {
        "road_points": int(len(road)),
        "moved": int(move.sum()),
        "left_unchanged_far": int((np.isfinite(dz) & ~move).sum()),
        "left_unchanged_no_field": int((~np.isfinite(dz)).sum()),
        "mean_abs_dz_moved": float(np.abs(dz[move]).mean()),
        "p95_abs_dz_moved": float(np.percentile(np.abs(dz[move]), 95)),
        "grid_cells_used": int((cnt >= min_points).sum()),
    }
    return out, stats
