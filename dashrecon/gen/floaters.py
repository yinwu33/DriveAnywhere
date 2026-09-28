"""Per-Gaussian quantities for finding the side-view junk of a FRONT-only run (floater diagnostic A, DECISIONS W).

support: the blending weight (alpha * transmittance) a Gaussian gets in the training views, summed over the pixels
    the photometric loss constrained. A Gaussian with near-zero support was never shaped by any training image, so
    whatever it shows from an unseen direction (its initial random colour, say) is unconstrained.
needle ratio: a Gaussian's extent along the ray from the nearest training camera over its largest extent across
    that ray. A large ratio looks small from FRONT but long from the side.
"""
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
              observed: torch.Tensor, margin: float, fade: float) -> torch.Tensor:
    """Opacity factor (N,) for a camera at ``centre``: 1 while the viewing direction is within half_angle + margin
    degrees of the observation axis, falling linearly to 0 over the next ``fade`` degrees; 0 if never observed."""
    ang = torch.rad2deg(torch.arccos((unit(means - centre) * axis).sum(-1).clamp(-1.0, 1.0)))
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
