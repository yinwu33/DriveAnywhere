"""Planar proxies as a hard training constraint for Background Gaussians (E45, docs/EXPERIMENTS.md).

Proxies come from scripts/fit_planes.py: ground height fields per class (world z up) and vertical plane patches.
Every ``every`` steps from ``start`` on, each Background Gaussian close to a proxy (inside its support; for a height
field between ``below`` under and ``above`` over the surface, for a vertical patch within ``band`` of the plane) is
snapped onto it: its centre is projected onto the surface, it is rotated so that its shortest axis lies along the
surface normal (the smallest rotation that does so), and that axis is capped at ``thickness``. Everything below a
ground surface within ``below`` is pulled up onto it: nothing real lies under the road. Gaussians farther away are left
alone (cars, kerbs higher than ``above``, vegetation). The optimizer's moment estimates are not touched; the snapping is
repeated, so drift between snaps stays small.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from pytorch3d.transforms import matrix_to_quaternion


@dataclass
class _Field:
    origin: torch.Tensor  # (2,) world xy of cell (0, 0)
    height: torch.Tensor  # (H, W)
    support: torch.Tensor  # (H, W) bool
    grad: torch.Tensor  # (H, W, 2) dh/dx, dh/dy


@dataclass
class _Patch:
    normal: torch.Tensor  # (3,)
    d: float
    u: torch.Tensor  # (3,) horizontal in-plane axis
    origin_uv: torch.Tensor  # (2,)
    support: torch.Tensor  # (Hu, Hz) bool


class PlanarConstraint:
    def __init__(self, path: str, every: int, start: int, below: float, above: float, band: float, thickness: float,
                 device: torch.device) -> None:
        z = np.load(path)
        self.cell = float(z["cell"])
        self.every, self.start, self.below, self.above, self.band, self.thickness = every, start, below, above, band, thickness
        self.fields = []
        for g in range(int(z["num_ground"])):
            h = torch.tensor(z[f"ground{g}_height"], dtype=torch.float32, device=device)
            gx, gy = torch.gradient(h, spacing=self.cell)
            self.fields.append(_Field(torch.tensor(z[f"ground{g}_origin"], dtype=torch.float32, device=device), h,
                                      torch.tensor(z[f"ground{g}_support"], dtype=torch.bool, device=device), torch.stack([gx, gy], -1)))
        self.patches = [_Patch(torch.tensor(z[f"plane{k}_normal"], dtype=torch.float32, device=device), float(z[f"plane{k}_d"]),
                               torch.tensor(z[f"plane{k}_u"], dtype=torch.float32, device=device),
                               torch.tensor(z[f"plane{k}_origin_uv"], dtype=torch.float32, device=device),
                               torch.tensor(z[f"plane{k}_support"], dtype=torch.bool, device=device))
                        for k in range(int(z["num_planes"]))]
        self.last_counts = {}

    def _field_lookup(self, f: _Field, xy: torch.Tensor):
        """Bilinear height and gradient at xy, and whether the cell is supported."""
        g = (xy - f.origin) / self.cell - 0.5
        i0, j0 = g[:, 0].floor().long(), g[:, 1].floor().long()
        H, W = f.height.shape
        inb = (i0 >= 0) & (j0 >= 0) & (i0 < H - 1) & (j0 < W - 1)
        i0c, j0c = i0.clamp(0, H - 2), j0.clamp(0, W - 2)
        a, b = (g[:, 0] - i0).clamp(0, 1), (g[:, 1] - j0).clamp(0, 1)
        def bil(t):
            return ((1 - a) * (1 - b))[..., None] * t[i0c, j0c].reshape(len(xy), -1) + (a * (1 - b))[..., None] * t[i0c + 1, j0c].reshape(len(xy), -1) \
                + ((1 - a) * b)[..., None] * t[i0c, j0c + 1].reshape(len(xy), -1) + (a * b)[..., None] * t[i0c + 1, j0c + 1].reshape(len(xy), -1)
        h = bil(f.height)[:, 0]
        grad = bil(f.grad)
        sup = inb & f.support[i0c, j0c] & f.support[i0c + 1, j0c + 1]
        return h, grad, sup

    @torch.no_grad()
    def snap(self, model) -> dict:
        means, quats, scales = model._means, model._quats, model._scales
        n = means.shape[0]
        best = torch.full((n,), float("inf"), device=means.device)
        target = means.data.clone()
        normal = torch.zeros_like(target)
        for f in self.fields:
            h, grad, sup = self._field_lookup(f, means[:, :2])
            dz = means[:, 2] - h
            cand = sup & (dz > -self.below) & (dz < self.above) & (dz.abs() < best)
            best = torch.where(cand, dz.abs(), best)
            target[cand, 2] = h[cand]
            nz = F.normalize(torch.cat([-grad, torch.ones_like(grad[:, :1])], -1), dim=-1)
            normal[cand] = nz[cand]
        for p in self.patches:
            dist = means @ p.normal + p.d
            uv = torch.stack([means @ p.u, means[:, 2]], -1)
            ij = ((uv - p.origin_uv) / self.cell).floor().long()
            Hu, Hz = p.support.shape
            inb = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < Hu) & (ij[:, 1] < Hz)
            sup = inb & p.support[ij[:, 0].clamp(0, Hu - 1), ij[:, 1].clamp(0, Hz - 1)]
            cand = sup & (dist.abs() < self.band) & (dist.abs() < best)
            best = torch.where(cand, dist.abs(), best)
            target[cand] = (means - dist[:, None] * p.normal)[cand]
            normal[cand] = p.normal
        sel = torch.isfinite(best)
        if not sel.any():
            return {"snapped": 0}
        idx = torch.nonzero(sel)[:, 0]
        means.data[idx] = target[idx]
        # rotate the shortest axis onto the normal
        q = F.normalize(quats.data[idx], dim=-1)
        R = quat_to_rotmat(q)
        s = scales.data[idx]
        a = s.argmin(dim=-1)
        v = R[torch.arange(len(idx), device=R.device), :, a]
        nrm = normal[idx]
        nrm = torch.where(((v * nrm).sum(-1, keepdim=True) < 0), -nrm, nrm)
        axis = torch.cross(v, nrm, dim=-1)
        sin = axis.norm(dim=-1).clamp(max=1.0)
        cos = (v * nrm).sum(-1).clamp(-1.0, 1.0)
        k = axis / sin.clamp(min=1e-8)[:, None]
        K = torch.zeros(len(idx), 3, 3, device=R.device)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
        eye = torch.eye(3, device=R.device).expand(len(idx), 3, 3)
        Ralign = eye + sin[:, None, None] * K + (1 - cos)[:, None, None] * (K @ K)
        Ralign = torch.where((sin < 1e-6)[:, None, None], eye, Ralign)
        quats.data[idx] = matrix_to_quaternion(Ralign @ R)
        cap = float(np.log(self.thickness))
        s[torch.arange(len(idx), device=s.device), a] = torch.clamp(s[torch.arange(len(idx), device=s.device), a], max=cap)
        scales.data[idx] = s
        self.last_counts = {"snapped": int(len(idx)), "of": int(n)}
        return self.last_counts
