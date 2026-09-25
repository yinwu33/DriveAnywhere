"""Free-viewpoint cameras and hole masks for Phase 9 (DECISIONS D16-D18).

A target view is a FRONT camera (the pose the run was trained with) moved in its own frame and turned:
``right`` / ``up`` / ``forward`` in scene units along the camera's x / -y / z axes (OpenCV), then ``yaw`` about its
y axis (positive = to the right) and ``pitch`` about its x axis (positive = up), in degrees.

Hole mask of a rendered view: a pixel is "seen" when its rendered surface point (back-projected with the rendered
z-depth) projects into a training view, agrees with that view's rendered depth within ``depth_tol`` (relative, so
it was not occluded there) and that view was at most ``res_ratio`` times farther away (comparable resolution).
Unseen pixels that are not sky are holes; this covers both empty space and the unconstrained Gaussians a 3DGS model
leaves in unobserved regions (those render opaque, so an opacity test misses them: val039 test, 0.2 % vs 13 % of
the view). Sky = rendered opacity below ``sky_alpha`` on a ray pointing above ``sky_elev_deg``. The raw mask is
cleaned (opening, small components dropped, dilation) because foliage depth is noisy.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ViewMove:
    right: float = 0.0
    up: float = 0.0
    forward: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    @staticmethod
    def parse(spec: str) -> "ViewMove":
        """``"right=1.5,yaw=15"`` -> ViewMove; unknown keys raise."""
        kw = {}
        for item in spec.split(","):
            k, v = item.split("=")
            assert k in ViewMove.__dataclass_fields__, f"unknown view move key {k}"
            kw[k] = float(v)
        return ViewMove(**kw)


def moved_c2w(c2w: torch.Tensor, m: ViewMove) -> torch.Tensor:
    """OpenCV camera-to-world moved and turned in its own frame (see module docstring)."""
    a, b = math.radians(m.yaw), math.radians(m.pitch)
    ry = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=c2w.dtype, device=c2w.device)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]], dtype=c2w.dtype, device=c2w.device)
    out = c2w.clone()
    out[:3, 3] = c2w[:3, 3] + m.right * c2w[:3, 0] - m.up * c2w[:3, 1] + m.forward * c2w[:3, 2]
    out[:3, :3] = c2w[:3, :3] @ ry @ rx
    return out


def backproject(depth: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """(H, W) z-depth, drivestudio intrinsics (+0.5 pixel-centre convention) -> (H*W, 3) world points."""
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
    xs = (x.float() + 0.5 - k[0, 2]) / k[0, 0] * depth
    ys = (y.float() + 0.5 - k[1, 2]) / k[1, 1] * depth
    return torch.stack([xs, ys, depth], -1).reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]


def seen_count(points: torch.Tensor, depth_novel: torch.Tensor, observers: list, depth_tol: float, res_ratio: float) -> torch.Tensor:
    """Number of observer views (dicts with w2c, K, depth (h, w)) that saw each point at comparable resolution."""
    count = torch.zeros(len(points), device=points.device)
    for o in observers:
        h, w = o["depth"].shape
        pc = points @ o["w2c"][:3, :3].T + o["w2c"][:3, 3]
        z = pc[:, 2]
        u = (o["K"][0, 0] * pc[:, 0] / z.clamp(min=1e-6) + o["K"][0, 2] - 0.5).round().long()
        v = (o["K"][1, 1] * pc[:, 1] / z.clamp(min=1e-6) + o["K"][1, 2] - 0.5).round().long()
        inside = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dz = torch.zeros_like(z)
        dz[inside] = o["depth"][v[inside], u[inside]].float()
        count += (inside & ((dz - z).abs() < depth_tol * z) & (z < res_ratio * depth_novel)).float()
    return count


def clean_mask(raw: np.ndarray, open_px: int, min_area: int, dilate_px: int) -> np.ndarray:
    """Opening, drop components below ``min_area`` pixels, dilation. (H, W) bool -> bool."""
    import cv2

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    opened = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    big = np.isin(lab, [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= min_area])
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(big.astype(np.uint8), dil) > 0
