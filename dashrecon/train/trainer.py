"""drivestudio trainer with NKSR-mesh regularisation for E5 (DECISIONS D9; selected with trainer.type).

With ``trainer.losses.mesh`` present, every training step additionally
    - rasterises the NKSR mesh with nvdiffrast into the current (pose-refined) camera: z-depth and
      face normals flipped towards the camera, both without gradient;
    - renders Gaussian normals: per Gaussian the rotation axis of its smallest scale, flipped towards the
      camera centre, alpha-blended with gsplat;
    - adds ``mesh_depth_loss`` = w_depth * mean |1/d_gs - 1/d_mesh| and
      ``mesh_normal_loss`` = w_normal * mean (1 - <n_gs, n_mesh>) over pixels covered by the mesh, not
      dynamic, not sky, and (for normals) with Gaussian alpha > ``min_alpha``.
Without the key the class behaves exactly like MultiTrainer (E3/E4).

Config (``trainer.losses.mesh``): path, depth_w, normal_w, min_alpha, near, far.

``trainer.planar`` (E45, dashrecon.train.planar): planar proxies of scripts/fit_planes.py as a hard constraint; after
every densification step whose number is a multiple of ``every`` (from ``start`` on), Background Gaussians near a proxy
are snapped onto it. Keys: path, every, start, below, above, band, thickness.

``trainer.carve`` (E48, dashrecon.train.carve; needs ``trainer.losses.mesh``): the mesh depth of up to ``max_views``
training images is cached, and every ``every`` steps from ``start`` to ``stop`` Background Gaussians that are never on
the mesh surface but off it in at least ``min_views`` of them are removed. Keys: every, start, stop, margin_rel,
margin_abs, min_views, max_views, down.

``trainer.losses.colour_anchor`` (E55): ``colour_anchor_loss`` = w * mean |c_i - I(p_i)| over the Background Gaussians
whose centre projects into the current training image (outside the dynamic and sky masks) within ``depth_tol`` (relative)
of the rendered depth there, i.e. the front layer; c_i is the view-independent (DC) colour, I(p_i) the image at the
projected centre. Without it, overlapping Gaussians can take garish colours that only add up to the right colour along
the training rays and come apart as coloured blobs from the side (D-F3, DECISIONS W). Keys: w, depth_tol.

``trainer.traversal`` (combined multi-traversal scenes with per-traversal appearance, dashrecon.train.traversal):
``starts`` (first virtual frame of every traversal) and ``novel`` (the traversal whose appearance novel views get, -1 =
the shared average). Before every render the Background and Sky models get ``current_traversal`` from the image's frame
index (dataset frames counted from 0: data.start_timestep must be 0), or ``novel`` for a novel view.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.rendering import rasterization

from dashrecon import io
from dashrecon.scenes import traversal_of
from models.trainers.scene_graph import MultiTrainer


class DashreconTrainer(MultiTrainer):
    """MultiTrainer + optional mesh depth/normal regularisation."""

    def __init__(self, traversal=None, planar=None, carve=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.carver = None
        if carve is not None:
            from dashrecon.train.carve import FreeSpaceCarver

            self.carver = FreeSpaceCarver(int(carve.every), int(carve.start), int(carve.stop), float(carve.margin_rel),
                                          float(carve.margin_abs), int(carve.min_views), int(carve.max_views), int(carve.down),
                                          float(carve.est_rel) if carve.get("est_rel") is not None else None,
                                          bool(carve.get("relocate", False)), bool(carve.get("front_only", False)))
            self.carve_exempt_planar = bool(carve.get("exempt_planar", False))
        self.planar = None
        if planar is not None:
            from dashrecon.train.planar import PlanarConstraint

            self.planar = PlanarConstraint(planar.path, int(planar.every), int(planar.start), float(planar.below), float(planar.above),
                                           float(planar.band), float(planar.thickness), self.device)
        self.traversal_starts = None if traversal is None else list(traversal.starts)
        self.traversal_novel = None if traversal is None else int(traversal.novel)
        self.mesh_cfg = self.losses_dict.mesh if "mesh" in self.losses_dict else None
        self._last_cam = None
        self._last_gs = None
        self._last_novel = False
        # dashrecon.gen.floaters.ViewGate applied to novel views only (E14); set by dashrecon.gen.novel.attach_view_gate
        self.view_gate = None
        # E17b: novel-view renders see the first protect_first_n Background Gaussians detached, so generated views only
        # shape the Gaussians spawned from them while the original ones learn from the real frames alone
        self.protect_first_n = 0
        if self.mesh_cfg is not None:
            import nvdiffrast.torch as dr

            mesh = io.read_mesh_ply(self.mesh_cfg.path)
            v = torch.from_numpy(mesh["vertices"]).float().to(self.device)
            f = torch.from_numpy(mesh["faces"]).int().to(self.device)
            tri = v[f.long()]
            n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
            self.mesh_v, self.mesh_f = v, f.contiguous()
            self.mesh_face_n = F.normalize(n, dim=-1)
            self.mesh_face_c = tri.mean(dim=1)
            self.glctx = dr.RasterizeCudaContext()
            self._dr = dr

    def process_camera(self, camera_infos, image_ids, novel_view: bool = False):
        cam = super().process_camera(camera_infos=camera_infos, image_ids=image_ids, novel_view=novel_view)
        self._last_cam = cam
        self._last_novel = novel_view
        return cam

    def collect_gaussians(self, cam, image_ids):
        gs = super().collect_gaussians(cam=cam, image_ids=image_ids)
        if self.protect_first_n > 0 and self._last_novel:
            from models.gaussians.basics import dataclass_gs

            n = self.protect_first_n
            assert gs._means.shape[0] >= n, (gs._means.shape[0], n)
            part = lambda x: torch.cat([x[:n].detach(), x[n:]], dim=0)
            gs = dataclass_gs(_means=part(gs._means), _scales=part(gs._scales), _quats=part(gs._quats), _rgbs=part(gs._rgbs),
                              _opacities=part(gs._opacities), detach_keys=gs.detach_keys, extras=gs.extras)
        if self.view_gate is not None and self._last_novel:
            gs = self.view_gate.apply(gs, cam.camtoworlds[:3, 3])
        self._last_gs = gs
        return gs

    @torch.no_grad()
    def _mesh_maps(self, cam) -> Dict[str, torch.Tensor]:
        c2w, k = cam.camtoworlds, cam.Ks
        h, w = int(cam.H), int(cam.W)
        w2c = torch.linalg.inv(c2w)
        pc = self.mesh_v @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        # pixel coordinates in the +0.5 convention, then NDC with row 0 at the top of the image
        u = k[0, 0] * pc[:, 0] / z + k[0, 2]
        v = k[1, 1] * pc[:, 1] / z + k[1, 2]
        n_, f_ = self.mesh_cfg.near, self.mesh_cfg.far
        clip = torch.stack([(2 * u / w - 1) * z, (2 * v / h - 1) * z,
                            (z * (f_ + n_) - 2 * f_ * n_) / (f_ - n_), z], dim=-1)
        rast, _ = self._dr.rasterize(self.glctx, clip[None].contiguous(), self.mesh_f, resolution=[h, w])
        tri_id = rast[0, ..., 3].long()
        valid = tri_id > 0
        depth, _ = self._dr.interpolate(z[None, :, None].contiguous(), rast, self.mesh_f)
        face = (tri_id - 1).clamp(min=0)
        n = self.mesh_face_n[face]
        to_cam = c2w[:3, 3] - self.mesh_face_c[face]
        n = torch.where(((n * to_cam).sum(-1, keepdim=True) < 0), -n, n)
        return {"mesh_valid": valid, "mesh_depth": depth[0, ..., 0] * valid, "mesh_normal": n * valid[..., None]}

    def _gaussian_normals(self, gs, cam) -> Dict[str, torch.Tensor]:
        rot = quat_to_rotmat(gs.quats)
        axis = gs.scales.argmin(dim=-1)
        n = rot[torch.arange(len(axis), device=axis.device), :, axis]
        to_cam = cam.camtoworlds[:3, 3] - gs.means
        n = torch.where(((n * to_cam).sum(-1, keepdim=True) < 0), -n, n)
        renders, alphas, _ = rasterization(
            means=gs.means, quats=gs.quats, scales=gs.scales, opacities=gs.opacities.squeeze(), colors=n,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None], Ks=cam.Ks[None], width=int(cam.W), height=int(cam.H),
            packed=self.render_cfg.packed, render_mode="RGB",
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            near_plane=self.render_cfg.near_plane, far_plane=self.render_cfg.far_plane,
        )
        return {"gs_normal": F.normalize(renders[0], dim=-1), "gs_normal_alpha": alphas[0, ..., 0]}

    def postprocess_per_train_step(self, step: int) -> None:
        super().postprocess_per_train_step(step)
        if self.planar is not None and step >= self.planar.start and step % self.planar.every == 0:
            counts = self.planar.snap(self.models["Background"])
            if step % 1000 == 0:
                print(f"[planar] step {step}: snapped {counts.get('snapped', 0):,} of {counts.get('of', 0):,} Background Gaussians", flush=True)
        if self.carver is not None and self.carver.due(step):
            exempt = None
            if self.carve_exempt_planar:
                assert self.planar is not None, "carve.exempt_planar needs trainer.planar"
                exempt = self.planar.near(self.models["Background"]._means.data)
            n = self.carver.carve(self.models["Background"], self.optimizer, step, exempt)
            verb = "relocated" if self.carver.relocate else "removed"
            print(f"[carve] step {step}: {verb} {n:,} Background Gaussians off the mesh surface ({len(self.carver.views)} cached views)", flush=True)

    def set_traversal(self, k: int) -> None:
        """Appearance of traversal k (-1 = shared average) for the Background and Sky models."""
        self.models["Background"].current_traversal = k
        self.models["Sky"].current_traversal = k

    def forward(self, image_infos, camera_infos, novel_view: bool = False):
        if self.traversal_starts is not None:
            self.set_traversal(self.traversal_novel if novel_view else
                               traversal_of(int(image_infos["frame_idx"].flatten()[0]), self.traversal_starts))
        outputs = super().forward(image_infos, camera_infos, novel_view)
        if self.mesh_cfg is not None and self.training:
            outputs.update(self._mesh_maps(self._last_cam))
            outputs.update(self._gaussian_normals(self._last_gs, self._last_cam))
            if self.carver is not None and not novel_view:
                self.carver.add(int(image_infos["img_idx"].flatten()[0]), self._last_cam, outputs["mesh_depth"],
                                image_infos.get("raw_depth_map"))
        return outputs

    def _colour_anchor_loss(self, outputs, image_infos) -> torch.Tensor:
        cfg = self.losses_dict.colour_anchor
        bg, cam = self.models["Background"], self._last_cam
        with torch.no_grad():
            w2c = torch.linalg.inv(cam.camtoworlds)
            pc = bg._means @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            u = (cam.Ks[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + cam.Ks[0, 2]).floor().long()
            v = (cam.Ks[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + cam.Ks[1, 2]).floor().long()
            h, w = image_infos["pixels"].shape[:2]
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx = torch.nonzero(ok)[:, 0]
            uu, vv = u[idx], v[idx]
            d = outputs["depth"][..., 0].detach()[vv, uu]
            static = (image_infos["dynamic_masks"][vv, uu] < 0.5) & (image_infos["sky_masks"][vv, uu] < 0.5)
            front = static & (d > 0) & ((z[idx] - d).abs() <= float(cfg.depth_tol) * d)
            idx, target = idx[front], image_infos["pixels"][vv[front], uu[front]]
        if len(idx) == 0:
            return bg._features_dc.sum() * 0.0
        return float(cfg.w) * (bg.colors[idx] - target).abs().mean()

    def compute_losses(self, outputs, image_infos, cam_infos):
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        if "colour_anchor" in self.losses_dict:
            loss_dict["colour_anchor_loss"] = self._colour_anchor_loss(outputs, image_infos)
        if self.mesh_cfg is None:
            return loss_dict
        keep = outputs["mesh_valid"].float() * (1.0 - image_infos["dynamic_masks"]) * (1.0 - image_infos["sky_masks"])
        d_gs = outputs["depth"][..., 0]
        m_depth = keep * (d_gs > 1e-4).float()
        inv_err = (1.0 / d_gs.clamp(min=1e-4) - 1.0 / outputs["mesh_depth"].clamp(min=1e-4)).abs()
        loss_dict["mesh_depth_loss"] = self.mesh_cfg.depth_w * (inv_err * m_depth).sum() / m_depth.sum().clamp(min=1.0)
        m_n = keep * (outputs["gs_normal_alpha"] > self.mesh_cfg.min_alpha).float()
        cos = (outputs["gs_normal"] * outputs["mesh_normal"]).sum(-1)
        loss_dict["mesh_normal_loss"] = self.mesh_cfg.normal_w * ((1.0 - cos) * m_n).sum() / m_n.sum().clamp(min=1.0)
        return loss_dict
