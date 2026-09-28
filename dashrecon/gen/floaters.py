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
