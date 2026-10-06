"""Free-space carving of Background Gaussians against the NKSR mesh during training (E48, docs/EXPERIMENTS.md).

The side-view floaters of a FRONT-only run (D-F2, E45g) are Gaussians that only have to look right along the forward
rays: they sit in front of, or behind, the surface the training views see. The mesh loss already rasterises the mesh in
every training view; its depth is cached here at 1/``down`` resolution for up to ``max_views`` training images. Every
``every`` steps between ``start`` and ``stop`` (right after a densification step, when drivestudio's per-Gaussian
statistics are reset), each Background Gaussian is projected into the cached views: where the mesh has depth d at its
pixel, it is on the surface if |z - d| <= max(margin_abs, margin_rel x d), off it otherwise. A Gaussian that is never on
the surface but off it in at least ``min_views`` views is removed (with its optimizer state). Gaussians that never land
on mesh depth (far background beyond the mesh, sky directions) are left alone, so this only carves where the mesh says
what is there.
"""
import torch
from torch.nn import Parameter

from models.gaussians.basics import remove_from_optim

PARAMS = ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities")


class FreeSpaceCarver:
    def __init__(self, every: int, start: int, stop: int, margin_rel: float, margin_abs: float, min_views: int,
                 max_views: int, down: int) -> None:
        self.every, self.start, self.stop = every, start, stop
        self.margin_rel, self.margin_abs, self.min_views = margin_rel, margin_abs, min_views
        self.max_views, self.down = max_views, down
        self.views = {}  # image index -> (w2c, K at 1/down, mesh depth at 1/down)
        self.history = []

    @torch.no_grad()
    def add(self, img_idx: int, cam, mesh_depth: torch.Tensor) -> None:
        if img_idx in self.views or len(self.views) >= self.max_views:
            return
        d = mesh_depth[None, None]
        # the nearest surface in each low-resolution cell (min over the cell, invalid = 0 treated as far)
        far = torch.where(d > 0, d, torch.full_like(d, float("inf")))
        small = -torch.nn.functional.max_pool2d(-far, self.down)[0, 0]
        small = torch.where(torch.isfinite(small), small, torch.zeros_like(small))
        k = cam.Ks.clone()
        k[:2] /= self.down
        self.views[img_idx] = (torch.linalg.inv(cam.camtoworlds).detach(), k, small)

    def due(self, step: int) -> bool:
        return self.start <= step <= self.stop and step % self.every == 0 and len(self.views) >= self.min_views

    @torch.no_grad()
    def carve(self, model, optimizer, step: int) -> int:
        means = model._means.data
        n = means.shape[0]
        on = torch.zeros(n, dtype=torch.bool, device=means.device)
        off = torch.zeros(n, dtype=torch.int32, device=means.device)
        for w2c, k, d in self.views.values():
            h, w = d.shape
            pc = means @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            u = (k[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + k[0, 2]).floor().long()
            v = (k[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + k[1, 2]).floor().long()
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx = torch.nonzero(ok)[:, 0]
            dm = d[v[idx], u[idx]]
            has = dm > 0
            idx, dm = idx[has], dm[has]
            tol = torch.clamp(self.margin_rel * dm, min=self.margin_abs)
            hit = (z[idx] - dm).abs() <= tol
            on[idx[hit]] = True
            off[idx[~hit]] += 1
        prune = ~on & (off >= self.min_views)
        n_prune = int(prune.sum())
        if n_prune:
            keep = ~prune
            for name in PARAMS:
                setattr(model, name, Parameter(getattr(model, name)[keep].detach()))
            for acc in ("xys_grad_norm", "vis_counts", "max_2Dsize"):
                if getattr(model, acc, None) is not None:
                    setattr(model, acc, getattr(model, acc)[keep])
            remove_from_optim(optimizer, prune, model.get_gaussian_param_groups())
        self.history.append({"step": step, "gaussians": n, "carved": n_prune, "views": len(self.views)})
        return n_prune
