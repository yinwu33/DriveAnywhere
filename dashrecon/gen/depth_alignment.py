"""A single estimated scene scale for monocular completion depths, never GT."""
import numpy as np


def estimate_scene_scale(depths: list[np.ndarray], rendered: list[np.ndarray],
                         counts: list[np.ndarray], min_count: int,
                         min_reference_pixels: int) -> tuple[float, dict]:
    """Anchor a metric-depth prior to supported pixels, including unseen frames.

    The returned scale does not certify any unobserved depth. Those pixels still
    need independent reprojection evidence before becoming scene supervision.
    """
    if not depths or len(depths) != len(rendered) or len(depths) != len(counts):
        raise ValueError("equal nonempty depth/render/count sequences required")
    if min_count < 1 or min_reference_pixels < 1:
        raise ValueError("positive support thresholds required")
    log_ratios, frames = [], []
    for k, (depth, reference, count) in enumerate(zip(depths, rendered, counts)):
        if depth.shape != reference.shape or depth.shape != count.shape:
            raise ValueError(f"depth/reference/support shapes differ in frame {k}")
        valid = ((count >= min_count) & np.isfinite(depth) & (depth > 0)
                 & np.isfinite(reference) & (reference > 0))
        n = int(valid.sum())
        frames.append({"k": k, "reference_pixels": n, "used_for_scale": n >= min_reference_pixels})
        if n >= min_reference_pixels:
            log_ratios.append(np.log(reference[valid] / depth[valid]))
    if not log_ratios:
        raise ValueError("no supported pixels to anchor the scene scale; no assumed scale is allowed")
    ratios = np.concatenate(log_ratios)
    centre = float(np.median(ratios))
    scale = float(np.exp(centre))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid estimated scene scale")
    return scale, {"kind": "monocular_prior_in_estimated_scene_units_not_verified_geometry",
                   "scene_scale": scale, "reference_pixels": int(len(ratios)), "frames": frames,
                   "median_abs_log_ratio": float(np.median(np.abs(ratios - centre))),
                   "ratio_p5_p95": np.exp(np.percentile(ratios, [5, 95])).tolist()}


def apply_scene_scale(depth: np.ndarray, scale: float) -> np.ndarray:
    """Apply the explicit scale; invalid monocular depths remain invalid (zero)."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scene scale required")
    valid = np.isfinite(depth) & (depth > 0)
    aligned = np.zeros(depth.shape, np.float32)
    aligned[valid] = depth[valid] * scale
    if not np.isfinite(aligned).all() or np.any(aligned > np.finfo(np.float16).max):
        raise ValueError("aligned depths cannot be stored as finite float16 values")
    return aligned
