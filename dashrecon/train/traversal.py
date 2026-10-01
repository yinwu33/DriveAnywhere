"""Per-traversal appearance for combined multi-traversal scenes (MT1app in docs/EXPERIMENTS.md; after MTGS).

Two passes of the same road on different days share the geometry but not the light: sun and hard shadows against an
overcast day, a road resealed in between, a blue against a grey sky. drivestudio has one colour per Gaussian, one sky
and a per-image affine colour transform, which can brighten an image but cannot move a shadow. Here

    TraversalGaussians  VanillaGaussians + per-traversal residuals (``residual``): "dc" on the SH DC colour, (N, T, 3);
                        "rest" on the higher SH bands, (N, T, 15, 3). They enter with their mean over the traversals
                        removed, so the shared colour is the average appearance and the residuals carry only what
                        differs between the days; ``reg.traversal_residual.w`` x mean |centred residual| keeps them to
                        what the shared colour cannot explain. MTGS (mtgs/scene_model/gaussian_model/
                        multi_color_gaussian_splatting.py, config MTGS.py) shares the DC and gives each traversal its
                        own higher bands (its DC adapters exist but have learning rate 0): residual ["rest"]. Shadows
                        and a resealed road are view-independent, which is why MT1app tried ["dc"] (the default).
    TraversalEnvLight   one EnvLight cube map per traversal.

Both have ``current_traversal`` (-1 = the shared average: the mean residual is zero, the mean sky), which
dashrecon.train.trainer.DashreconTrainer sets before every render from the image's frame index and the virtual frame
ranges of the combined scene (``traversal_starts``, scripts/combine_traversals.py); novel views use
``trainer.traversal.novel``.

Config: model.Background.type = dashrecon.train.traversal.TraversalGaussians with ``num_traversals``, ``residual``,
``reg.traversal_residual.w`` and ``optim.sh_dc_tr`` / ``optim.sh_rest_tr``; model.Sky.type = dashrecon.train.traversal.TraversalEnvLight with
``params.num_traversals``; trainer.traversal.starts / novel.
"""
from typing import Dict

import nvdiffrast.torch as dr
import torch
from torch.nn import Parameter

from models.gaussians.basics import spherical_harmonics
from models.gaussians.vanilla import VanillaGaussians
from models.modules import EnvLight


RESIDUALS = {"dc": "_features_dc_tr", "rest": "_features_rest_tr"}


class TraversalGaussians(VanillaGaussians):
    """VanillaGaussians with per-traversal residuals on the DC colour and / or the higher SH bands."""

    def __init__(self, num_traversals: int, residual=("dc",), **kwargs) -> None:
        super().__init__(**kwargs)
        assert num_traversals >= 2, num_traversals
        assert len(residual) > 0 and set(residual) <= set(RESIDUALS), residual
        self.num_traversals = num_traversals
        self.residual = tuple(residual)
        self.current_traversal = -1
        for kind in self.residual:
            setattr(self, RESIDUALS[kind], torch.zeros((1, num_traversals) + self._band_shape(kind), device=self.device))

    def _band_shape(self, kind: str) -> tuple:
        return (3,) if kind == "dc" else tuple(self._features_rest.shape[1:])

    def centred(self, kind: str) -> torch.Tensor:
        """(N, T, ...) residuals of one kind with their mean over the traversals removed."""
        x = getattr(self, RESIDUALS[kind])
        return x - x.mean(dim=1, keepdim=True)

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)
        for kind in self.residual:
            setattr(self, RESIDUALS[kind], Parameter(torch.zeros((self.num_points, self.num_traversals) + self._band_shape(kind),
                                                                 device=self.device)))

    def get_gaussian_param_groups(self) -> Dict:
        names = {"dc": "sh_dc_tr", "rest": "sh_rest_tr"}
        return {**super().get_gaussian_param_groups(),
                **{self.class_prefix + names[k]: [getattr(self, RESIDUALS[k])] for k in self.residual}}

    def split_gaussians(self, split_mask: torch.Tensor, samps: int):
        # refinement_after splits, then duplicates, then concatenates [old, split, dup] for every parameter before
        # updating the optimizer: the residuals are concatenated in the same order in dup_gaussians
        self._split_tr = {}
        for kind in self.residual:
            x = getattr(self, RESIDUALS[kind])[split_mask]
            self._split_tr[kind] = x.repeat(samps, *([1] * (x.dim() - 1)))
        return super().split_gaussians(split_mask, samps)

    def dup_gaussians(self, dup_mask: torch.Tensor):
        out = super().dup_gaussians(dup_mask)
        for kind in self.residual:
            x = getattr(self, RESIDUALS[kind])
            setattr(self, RESIDUALS[kind], Parameter(torch.cat([x.detach(), self._split_tr[kind], x[dup_mask].detach()], dim=0)))
        del self._split_tr
        return out

    def cull_gaussians(self):
        culls = super().cull_gaussians()
        for kind in self.residual:
            setattr(self, RESIDUALS[kind], Parameter(getattr(self, RESIDUALS[kind])[~culls].detach()))
        return culls

    def get_gaussians(self, cam) -> Dict:
        """VanillaGaussians.get_gaussians with the current traversal's colour (no filtering, as there)."""
        assert -1 <= self.current_traversal < self.num_traversals, self.current_traversal
        dc, rest = self._features_dc, self._features_rest
        if self.current_traversal >= 0:
            if "dc" in self.residual:
                dc = dc + self.centred("dc")[:, self.current_traversal]
            if "rest" in self.residual:
                rest = rest + self.centred("rest")[:, self.current_traversal]
        colors = torch.cat((dc[:, None, :], rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = torch.clamp(spherical_harmonics(n, viewdirs, colors) + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        self.filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        gs_dict = dict(_means=self._means, _opacities=self.get_opacity, _rgbs=rgbs, _scales=self.get_scaling,
                       _quats=self.get_quats)
        for k, v in gs_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"NaN or Inf in gaussian {k} at step {self.step}")
        return gs_dict

    def compute_reg_loss(self) -> Dict:
        loss_dict = super().compute_reg_loss()
        loss_dict["traversal_residual"] = self.reg_cfg.traversal_residual.w * sum(
            self.centred(kind).abs().mean() for kind in self.residual)
        return loss_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        n = state_dict["_means"].shape[0]
        for kind in self.residual:
            setattr(self, RESIDUALS[kind], Parameter(torch.zeros((n, self.num_traversals) + self._band_shape(kind), device=self.device)))
        return super().load_state_dict(state_dict, **kwargs)


class TraversalEnvLight(EnvLight):
    """One EnvLight cube map per traversal; -1 renders their mean."""

    def __init__(self, num_traversals: int, resolution: int = 1024, **kwargs) -> None:
        super().__init__(resolution=resolution, **kwargs)
        assert num_traversals >= 2, num_traversals
        self.num_traversals = num_traversals
        self.current_traversal = -1
        self.base = Parameter(0.5 * torch.ones(num_traversals, 6, resolution, resolution, 3))

    def forward(self, image_infos):
        base = self.base.mean(dim=0) if self.current_traversal < 0 else self.base[self.current_traversal]
        l = image_infos["viewdirs"]
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape).contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])
        light = dr.texture(base[None, ...].contiguous(), l, filter_mode="linear", boundary_mode="cube")
        return light.view(*prefix, -1)
