"""Free-viewpoint cameras and hole masks for Phase 9 (DECISIONS D16-D18).

A target view is a FRONT camera (the pose the run was trained with) moved in its own frame and turned:
``right`` / ``up`` / ``forward`` in scene units along the camera's x / -y / z axes (OpenCV), then ``yaw`` about its
y axis (positive = to the right) and ``pitch`` about its x axis (positive = up), in degrees.

Hole mask of a rendered view: a pixel is "seen" when its rendered surface point (back-projected with the rendered
z-depth) projects into a training view, lands on a pixel that view did not mask as dynamic or sky (masked pixels
give the Gaussians there no photometric signal: the ghosts of masked parked cars, the grey blocks floating in the
sky of val056), agrees with that view's rendered depth
within ``depth_tol`` (relative, so it was not occluded there) and that view was at most ``res_ratio`` times farther
away (comparable resolution).
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
    rx = torch.tensor([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]], dtype=c2w.dtype, device=c2w.device)
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
    """Number of observer views (dicts with w2c, K, depth (h, w), valid (h, w) bool = pixel constrained by that
    view) that saw each point at comparable resolution."""
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
        ok = torch.zeros_like(inside)
        ok[inside] = o["valid"][v[inside], u[inside]]
        count += (ok & ((dz - z).abs() < depth_tol * z) & (z < res_ratio * depth_novel)).float()
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


@torch.no_grad()
def training_observers(trainer, dataset, stride: int, device: torch.device) -> list:
    """Every stride-th training view of a run, rendered as trained (CamPose refined): w2c, K, z-depth and the pixels
    outside its dynamic and sky masks (the only ones where the photometric loss constrained a surface)."""
    from dashrecon.gen.novel import to_device

    trainer.set_eval()
    observers = []
    for j in range(0, len(dataset.train_image_set), stride):
        ii, ci = dataset.train_image_set.get_image(j, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        out = trainer(ii, ci)
        observers.append({"w2c": torch.linalg.inv(trainer._last_cam.camtoworlds), "K": ci["intrinsics"], "depth": out["depth"][..., 0].half(),
                          "valid": (ii["dynamic_masks"] < 0.5) & (ii["sky_masks"] < 0.5)})
    return observers


def render_at(trainer, image_infos: dict, cam_infos: dict, c2w: torch.Tensor, k: torch.Tensor, hw) -> dict:
    """Render a frame's image/camera infos from camera c2w (OpenCV, no CamPose) with drivestudio intrinsics k on an
    (h, w) grid, which may differ from the training size; rays (and so the sky model), the image-size fields and the
    per-pixel image-index map of the Affine model (models/modules.py AffineTransform) are set for it. The infos dicts
    are updated in place. Gradients flow when enabled (scripts/train_fill.py)."""
    from datasets.base.pixel_source import get_rays

    h, w = hw
    device = c2w.device
    image_infos["img_idx"] = torch.full((h, w), int(image_infos["img_idx"].flatten()[0]), dtype=image_infos["img_idx"].dtype, device=device)
    cam_infos["height"] = torch.tensor(h, dtype=torch.long, device=device)
    cam_infos["width"] = torch.tensor(w, dtype=torch.long, device=device)
    x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="xy")
    origins, viewdirs, dnorm = get_rays(x.flatten(), y.flatten(), c2w, k)
    image_infos["origins"], image_infos["viewdirs"] = origins.reshape(h, w, 3), viewdirs.reshape(h, w, 3)
    image_infos["direction_norm"] = dnorm.reshape(h, w, 1)
    cam_infos["camera_to_world"], cam_infos["intrinsics"] = c2w, k
    return trainer(image_infos, cam_infos, novel_view=True)


@torch.no_grad()
def render_moved(trainer, dataset, k: int, move: ViewMove, device: torch.device):
    """Frame k of the full set seen from its trained camera (CamPose refined) moved by ``move``.
    Returns (outputs, image_infos, cam_infos, c2w)."""
    from dashrecon.gen.novel import refined_c2w, to_device

    ii, ci = dataset.full_image_set.get_image(k, 1)
    ii, ci = to_device(ii, device), to_device(ci, device)
    c2w = moved_c2w(refined_c2w(trainer, ii, ci), move)
    out = render_at(trainer, ii, ci, c2w, ci["intrinsics"], (int(ci["height"]), int(ci["width"])))
    return out, ii, ci, c2w


@torch.no_grad()
def memory_observers(trainer, dataset, views_dirs: list, stride: int, device: torch.device) -> list:
    """Every stride-th view of earlier filled trajectories (render_views.py cams.json), rendered by the current model:
    the 3D memory of Phase 9, so regions filled in an earlier round count as seen and are not generated again."""
    import json
    import os

    from dashrecon.gen.novel import to_device

    trainer.set_eval()
    observers = []
    for d in views_dirs:
        with open(os.path.join(d, "cams.json")) as f:
            cams = json.load(f)["cams"]
        for c in cams[::stride]:
            ii, ci = dataset.full_image_set.get_image(c["frame"] - dataset.start_timestep, 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            c2w = torch.tensor(c["c2w"], dtype=torch.float32, device=device)
            k = torch.tensor(c["K"], dtype=torch.float32, device=device)
            out = render_at(trainer, ii, ci, c2w, k, c["hw"])
            observers.append({"w2c": torch.linalg.inv(c2w), "K": k, "depth": out["depth"][..., 0].half(),
                              "valid": torch.ones(out["depth"].shape[:2], dtype=torch.bool, device=device)})
    return observers


def hole_mask(outputs: dict, viewdirs: torch.Tensor, k: torch.Tensor, c2w: torch.Tensor, observers: list, depth_tol: float,
              res_ratio: float, sky_alpha: float, sky_elev_deg: float, open_px: int, min_area: int, dilate_px: int):
    """Hole mask of a rendered view (module docstring). Returns (hole (H, W) bool, count (H, W) float32, sky (H, W) bool)."""
    depth = outputs["depth"][..., 0]
    h, w = depth.shape
    count = seen_count(backproject(depth, k, c2w), depth.reshape(-1), observers, depth_tol, res_ratio).reshape(h, w).cpu().numpy()
    alpha = outputs["opacity"][..., 0].cpu().numpy()
    elev = np.degrees(np.arcsin(np.clip(viewdirs[..., 2].cpu().numpy(), -1.0, 1.0)))
    sky = (alpha < sky_alpha) & (elev >= sky_elev_deg)
    hole = clean_mask((count < 1) & ~sky, open_px, min_area, dilate_px) & ~sky
    return hole, count, sky
