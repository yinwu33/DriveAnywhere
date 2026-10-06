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

With ``est_rel`` (E49) a Gaussian is also on the surface in a view if it lies within max(margin_abs, est_rel x e) of the
estimated depth e there (the raw per-frame depth of the pose backend: no confidence or range filter, static non-sky
pixels; image_infos ``raw_depth_map``), and views where only the estimated depth exists count too. The NKSR mesh misses
most trees and the far side of the street (low depth confidence there); without this, real Gaussians there are never
on the mesh, only in front of or behind it, and are carved (E48).

With ``relocate`` (E50) the selected Gaussians are moved instead of removed: removing them leaves holes at +-45 deg
(E48, E45g), but what a training view saw through a floater is surface colour at the wrong depth. Each is slid along the
ray from the nearest cached camera that has depth at its pixel (mesh or estimated, the nearer of the two) onto that
depth, and its scales are multiplied by the depth ratio (clamped to [1/3, 3]) so its footprint in that view stays about
the same; the photometric loss then refits it in place.
"""
import torch
from torch.nn import Parameter

from models.gaussians.basics import remove_from_optim

PARAMS = ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities")


class FreeSpaceCarver:
    def __init__(self, every: int, start: int, stop: int, margin_rel: float, margin_abs: float, min_views: int,
                 max_views: int, down: int, est_rel: float = None, relocate: bool = False) -> None:
        self.every, self.start, self.stop = every, start, stop
        self.est_rel, self.relocate = est_rel, relocate
        self.margin_rel, self.margin_abs, self.min_views = margin_rel, margin_abs, min_views
        self.max_views, self.down = max_views, down
        self.views = {}  # image index -> (w2c, K at 1/down, mesh depth at 1/down, estimated depth at 1/down or None)
        self.history = []

    def _nearest(self, depth: torch.Tensor) -> torch.Tensor:
        """The nearest surface in each low-resolution cell (min over the cell, invalid = 0 treated as far)."""
        d = depth.float()[None, None]
        far = torch.where(d > 0, d, torch.full_like(d, float("inf")))
        small = -torch.nn.functional.max_pool2d(-far, self.down)[0, 0]
        return torch.where(torch.isfinite(small), small, torch.zeros_like(small))

    @torch.no_grad()
    def add(self, img_idx: int, cam, mesh_depth: torch.Tensor, est_depth: torch.Tensor = None) -> None:
        if img_idx in self.views or len(self.views) >= self.max_views:
            return
        if self.est_rel is not None:
            assert est_depth is not None, "est_rel needs the raw estimated depth (data.pixel_source.raw_depth)"
            assert est_depth.shape == mesh_depth.shape, (est_depth.shape, mesh_depth.shape)
        k = cam.Ks.clone()
        k[:2] /= self.down
        est = self._nearest(est_depth) if self.est_rel is not None else None
        self.views[img_idx] = (torch.linalg.inv(cam.camtoworlds).detach(), k, self._nearest(mesh_depth), est)

    def due(self, step: int) -> bool:
        return self.start <= step <= self.stop and step % self.every == 0 and len(self.views) >= self.min_views

    @torch.no_grad()
    def carve(self, model, optimizer, step: int) -> int:
        means = model._means.data
        n = means.shape[0]
        on = torch.zeros(n, dtype=torch.bool, device=means.device)
        off = torch.zeros(n, dtype=torch.int32, device=means.device)
        # relocation target: nearest camera (smallest z) with depth at the Gaussian's pixel, and that depth
        best_z = torch.full((n,), float("inf"), device=means.device)
        best_t = torch.zeros(n, device=means.device)
        best_c = torch.zeros(n, 3, device=means.device)
        for w2c, k, d, e in self.views.values():
            h, w = d.shape
            pc = means @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            u = (k[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + k[0, 2]).floor().long()
            v = (k[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + k[1, 2]).floor().long()
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx = torch.nonzero(ok)[:, 0]
            zz = z[idx]
            dm = d[v[idx], u[idx]]
            has = dm > 0
            hit = has & ((zz - dm).abs() <= torch.clamp(self.margin_rel * dm, min=self.margin_abs))
            if e is not None:
                em = e[v[idx], u[idx]]
                has_e = em > 0
                hit = hit | (has_e & ((zz - em).abs() <= torch.clamp(self.est_rel * em, min=self.margin_abs)))
                has = has | has_e
            on[idx[hit]] = True
            off[idx[has & ~hit]] += 1
            if self.relocate:
                t = dm
                if e is not None:
                    t = torch.where((em > 0) & ((dm <= 0) | (em < dm)), em, dm)
                better = (t > 0) & (zz < best_z[idx])
                j = idx[better]
                best_z[j], best_t[j] = zz[better], t[better]
                best_c[j] = -w2c[:3, :3].T @ w2c[:3, 3]
        prune = ~on & (off >= self.min_views)
        n_prune = int(prune.sum())
        if n_prune and self.relocate:
            j = torch.nonzero(prune & torch.isfinite(best_z))[:, 0]
            ratio = best_t[j] / best_z[j]
            model._means.data[j] = best_c[j] + (means[j] - best_c[j]) * ratio[:, None]
            model._scales.data[j] += torch.log(ratio.clamp(1 / 3, 3))[:, None]
            self.history.append({"step": step, "gaussians": n, "relocated": int(len(j)), "views": len(self.views),
                                 "median_ratio": float(ratio.median()) if len(j) else None})
            return int(len(j))
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
