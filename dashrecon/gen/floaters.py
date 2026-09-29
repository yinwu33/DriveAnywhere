"""Per-Gaussian quantities for finding the side-view junk of a FRONT-only run (floater diagnostic A, DECISIONS W).

support: the blending weight (alpha * transmittance) a Gaussian gets in the training views, summed over the pixels
    the photometric loss constrained. A Gaussian with near-zero support was never shaped by any training image, so
    whatever it shows from an unseen direction (its initial random colour, say) is unconstrained.
needle ratio: a Gaussian's extent along the ray from the nearest training camera over its largest extent across
    that ray. A large ratio looks small from FRONT but long from the side.
observation cone / ViewGate (D-A3, E14 in docs/EXPERIMENTS.md): the directions the training views saw a Gaussian
    from; rendering a novel view, Gaussians seen from far outside their cone fade out, so the FRONT-only "screen"
    of large off-surface Gaussians (D-A2) becomes a hole there instead of junk.
"""
from dataclasses import dataclass

import torch


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """(N,4) unit quaternions (w, x, y, z; the gsplat convention) -> (N,3,3) rotation matrices."""
    w, x, y, z = quats.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def nearest_centre(means: torch.Tensor, centres: torch.Tensor, chunk: int = 200000) -> torch.Tensor:
    """(N,3) points, (M,3) camera centres -> (N,) index of the nearest centre."""
    return torch.cat([torch.cdist(means[i:i + chunk], centres).argmin(dim=1) for i in range(0, len(means), chunk)])


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def observation_cone(means: torch.Tensor, centres: torch.Tensor, weights: torch.Tensor, min_weight: float) -> tuple:
    """Directions the training views saw each Gaussian from.

    means (N,3); centres (V,3) training camera centres; weights (V,N) blending weight of each Gaussian in each view.
    Returns (axis (N,3): weight-averaged unit direction from camera to Gaussian, half_angle (N,) degrees: largest
    angle between the axis and the direction from any view with weight >= min_weight, observed (N,) bool: some view
    has weight >= min_weight). Unobserved Gaussians get half_angle 0.
    """
    acc = torch.zeros_like(means)
    for c, w in zip(centres, weights):
        acc += w.float()[:, None] * unit(means - c)
    observed = (weights >= min_weight).any(dim=0)
    axis = torch.where(observed[:, None], unit(acc + 1e-12), torch.zeros_like(acc))
    half = torch.zeros(len(means), device=means.device)
    for c, w in zip(centres, weights):
        ang = torch.rad2deg(torch.arccos((unit(means - c) * axis).sum(-1).clamp(-1.0, 1.0)))
        half = torch.where(w >= min_weight, torch.maximum(half, ang), half)
    return axis, half, observed


def cone_gate(means: torch.Tensor, centre: torch.Tensor, axis: torch.Tensor, half_angle: torch.Tensor,
              observed: torch.Tensor, margin: float, fade: float, double_sided: bool = False) -> torch.Tensor:
    """Opacity factor (N,) for a camera at ``centre``: 1 while the viewing direction is within half_angle + margin
    degrees of the observation axis, falling linearly to 0 over the next ``fade`` degrees; 0 if never observed.

    double_sided measures the angle to the axis *line* (min(a, 180 - a)): a view along the observed rays but facing the
    other way (looking back) hides depth errors along those rays just as the observing view did, so only views across
    the rays (the sides) are gated (D-A4 in docs/EXPERIMENTS.md; E14's one-sided gate emptied the rear view)."""
    ang = torch.rad2deg(torch.arccos((unit(means - centre) * axis).sum(-1).clamp(-1.0, 1.0)))
    if double_sided:
        ang = torch.minimum(ang, 180.0 - ang)
    excess = (ang - half_angle).clamp(min=0.0)
    g = (1.0 - (excess - margin) / fade).clamp(0.0, 1.0)
    return torch.where(observed, g, torch.zeros_like(g))


def needle_ratio(means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Extent along the viewing ray from the nearest centre over the largest extent across it, per Gaussian.

    Extents are standard deviations of the Gaussian: sqrt(d^T S d) along the unit ray d, and the square root of the
    largest eigenvalue of P S P with P = I - d d^T across it (S the covariance R diag(scales^2) R^T).
    """
    rot = quat_to_rotmat(quats)
    cov = rot @ torch.diag_embed(scales ** 2) @ rot.transpose(1, 2)
    d = means - centres[nearest_centre(means, centres)]
    d = d / d.norm(dim=-1, keepdim=True)
    along = torch.einsum("ni,nij,nj->n", d, cov, d).sqrt()
    proj = torch.eye(3, device=means.device, dtype=means.dtype) - d[:, :, None] * d[:, None, :]
    across = torch.linalg.eigvalsh(proj @ cov @ proj)[:, -1].sqrt()
    return along / across


def rasterize(trainer, gs, cam, colours: torch.Tensor) -> tuple:
    """gsplat rasterization of arbitrary per-Gaussian colours with the trainer's render settings (models/trainers/
    base.py render_gaussians); returns (image [H,W,C], alpha [H,W])."""
    from gsplat.rendering import rasterization

    rc = trainer.render_cfg
    assert "radius_clip" not in rc, "radius_clip set: pass it as the trainer does"
    img, alpha, _ = rasterization(
        means=gs.means, quats=gs.quats, scales=gs.scales, opacities=gs.opacities.squeeze(), colors=colours,
        viewmats=torch.linalg.inv(cam.camtoworlds)[None], Ks=cam.Ks[None], width=cam.W, height=cam.H,
        packed=rc.packed, near_plane=rc.near_plane, far_plane=rc.far_plane, render_mode="RGB",
        rasterize_mode="antialiased" if rc.antialiased else "classic")
    return img[0], alpha[0, ..., 0]


def blend_weight(trainer, gs, cam, mask: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian blending weight summed over the pixels of ``mask`` [H,W]: rasterize a zero one-channel colour and
    backpropagate the masked image sum (d sum / d c_i = sum of the Gaussian's alpha * transmittance)."""
    colour = torch.zeros(gs.means.shape[0], 1, device=mask.device, requires_grad=True)
    img, _ = rasterize(trainer, gs, cam, colour)
    (img[..., 0] * mask).sum().backward()
    return colour.grad[:, 0]


def training_observations(trainer, dataset, device) -> tuple:
    """(weights (V,N) float16: blending weight of every Background Gaussian in every FRONT training view over the
    pixels the photometric loss constrained (outside the dynamic and sky masks), training camera centres (V,3))."""
    from dashrecon.gen.novel import to_device

    n = trainer.models["Background"]._means.shape[0]
    weights, centres = [], []
    for j in range(len(dataset.train_image_set)):
        ii, ci = dataset.train_image_set.get_image(j, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        img_id = ii["img_idx"].flatten()[0]
        with torch.no_grad():
            cam = trainer.process_camera(camera_infos=ci, image_ids=img_id, novel_view=False)
            gs = trainer.collect_gaussians(cam=cam, image_ids=img_id)
        assert gs.means.shape[0] == n
        w = blend_weight(trainer, gs, cam, ((ii["dynamic_masks"] < 0.5) & (ii["sky_masks"] < 0.5)).float())
        assert float(w.max()) < 65504, "weight overflows float16"
        weights.append(w.half())
        centres.append(cam.camtoworlds[:3, 3].detach())
    return torch.stack(weights), torch.stack(centres)


@dataclass
class ViewGate:
    """Observation-cone gate of the first ``n`` Background Gaussians of a model (those present when it was made;
    Gaussians added later are never gated). Only ``selected`` Gaussians are gated."""

    axis: torch.Tensor
    half_angle: torch.Tensor
    observed: torch.Tensor
    selected: torch.Tensor
    margin: float
    fade: float
    double_sided: bool = False

    @property
    def n(self) -> int:
        return int(self.axis.shape[0])

    def factors(self, means: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
        """(n,) opacity factors of the first n Gaussians for a camera at ``centre``."""
        g = cone_gate(means[:self.n], centre, self.axis, self.half_angle, self.observed, self.margin, self.fade,
                      self.double_sided)
        return torch.where(self.selected, g, torch.ones_like(g))

    def apply(self, gs, centre: torch.Tensor):
        """The dataclass_gs ``gs`` (Background only) with the gated opacities; gradients flow through."""
        from models.gaussians.basics import dataclass_gs

        assert gs._means.shape[0] >= self.n, (gs._means.shape[0], self.n)
        f = self.factors(gs._means.detach(), centre.detach())
        opac = torch.cat([gs._opacities[:self.n] * f[:, None], gs._opacities[self.n:]], dim=0)
        return dataclass_gs(_means=gs._means, _scales=gs._scales, _quats=gs._quats, _rgbs=gs._rgbs, _opacities=opac,
                            detach_keys=gs.detach_keys, extras=gs.extras)

    def save(self, path: str, meta: dict) -> None:
        torch.save({"axis": self.axis.cpu(), "half_angle": self.half_angle.cpu(), "observed": self.observed.cpu(),
                    "selected": self.selected.cpu(), "margin": self.margin, "fade": self.fade,
                    "double_sided": self.double_sided, "meta": meta}, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "ViewGate":
        d = torch.load(path, map_location=device)
        # gates saved before D-A4 have no double_sided entry; they were one-sided
        double_sided = bool(d["double_sided"]) if "double_sided" in d else False
        return ViewGate(axis=d["axis"], half_angle=d["half_angle"], observed=d["observed"], selected=d["selected"],
                        margin=float(d["margin"]), fade=float(d["fade"]), double_sided=double_sided)


def iterate_observations(trainer, dataset, device, generated: list):
    """Yield (camera centre (3,), blending weight (N,)) for every view that supervised the run's Background Gaussians:
    each FRONT training view over the pixels the photometric loss constrained (outside the dynamic and sky masks), and
    each view of the ``generated`` (views_dir, validation_dir) trajectories over the hole pixels its distillation
    supervised (validate_generated_views.py status 1 unknown, 2 weak, 4 accepted; 3 conflict got no weight in
    train_fill.py --unknown_w). Generated views are rendered without any view gate (the trainer must have none)."""
    import json
    import os

    import numpy as np
    from PIL import Image

    from dashrecon.gen.novel import to_device
    from dashrecon.gen.views import front_image_index, render_at

    assert trainer.view_gate is None, "observations must see every Gaussian"
    for j in range(len(dataset.train_image_set)):
        ii, ci = dataset.train_image_set.get_image(j, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        img_id = ii["img_idx"].flatten()[0]
        with torch.no_grad():
            cam = trainer.process_camera(camera_infos=ci, image_ids=img_id, novel_view=False)
            gs = trainer.collect_gaussians(cam=cam, image_ids=img_id)
        mask = ((ii["dynamic_masks"] < 0.5) & (ii["sky_masks"] < 0.5)).float()
        yield cam.camtoworlds[:3, 3].detach(), blend_weight(trainer, gs, cam, mask)
    for views_dir, validation_dir in generated:
        cams = json.load(open(os.path.join(views_dir, "cams.json")))["cams"]
        for k, c in enumerate(cams):
            status = np.asarray(Image.open(os.path.join(validation_dir, "status", f"{k:03d}.png")))
            mask = torch.from_numpy(np.isin(status, (1, 2, 4))).float().to(device)
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, c["frame"] - dataset.start_timestep), 1)
            c2w = torch.tensor(c["c2w"], dtype=torch.float32, device=device)
            with torch.no_grad():
                render_at(trainer, to_device(ii, device), to_device(ci, device), c2w,
                          torch.tensor(c["K"], dtype=torch.float32, device=device), tuple(c["hw"]))
            yield c2w[:3, 3], blend_weight(trainer, trainer._last_gs, trainer._last_cam, mask)


def observation_cone_streaming(means: torch.Tensor, observations, min_weight: float) -> tuple:
    """observation_cone without holding all weights: ``observations()`` returns a fresh iterator of (centre, weight)
    and is called twice (axis, then half-angle)."""
    acc = torch.zeros_like(means)
    observed = torch.zeros(len(means), dtype=torch.bool, device=means.device)
    views = 0
    for c, w in observations():
        acc += w.float()[:, None] * unit(means - c)
        observed |= w >= min_weight
        views += 1
    axis = torch.where(observed[:, None], unit(acc + 1e-12), torch.zeros_like(acc))
    half = torch.zeros(len(means), device=means.device)
    for c, w in observations():
        ang = torch.rad2deg(torch.arccos((unit(means - c) * axis).sum(-1).clamp(-1.0, 1.0)))
        half = torch.where(w >= min_weight, torch.maximum(half, ang), half)
    return axis, half, observed, views
