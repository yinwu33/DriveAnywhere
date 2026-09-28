"""z-depth of a triangle mesh seen from a camera, with nvdiffrast (runs in the main venv).

Same projection as dashrecon.train.trainer.DashreconTrainer._mesh_maps (drivestudio intrinsics, +0.5 pixel-centre
convention, OpenCV camera-to-world), for meshes other than the run's own (E17a's imagined mesh).
"""
import torch


class MeshDepthRenderer:
    """Holds one mesh on the GPU and renders its z-depth for arbitrary cameras."""

    def __init__(self, vertices, faces, near: float, far: float, device: torch.device) -> None:
        import nvdiffrast.torch as dr

        self.dr = dr
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.f = torch.as_tensor(faces, dtype=torch.int32, device=device).contiguous()
        self.near, self.far = near, far

    @torch.no_grad()
    def depth(self, k: torch.Tensor, c2w: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(h, w) z-depth of the nearest surface; 0 where the mesh is absent."""
        w2c = torch.linalg.inv(c2w)
        pc = self.v @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = k[0, 0] * pc[:, 0] / z + k[0, 2]
        v = k[1, 1] * pc[:, 1] / z + k[1, 2]
        n_, f_ = self.near, self.far
        clip = torch.stack([(2 * u / w - 1) * z, (2 * v / h - 1) * z, (z * (f_ + n_) - 2 * f_ * n_) / (f_ - n_), z], dim=-1)
        rast, _ = self.dr.rasterize(self.glctx, clip[None].contiguous(), self.f, resolution=[h, w])
        valid = rast[0, ..., 3] > 0
        d, _ = self.dr.interpolate(z[None, :, None].contiguous(), rast, self.f)
        return d[0, ..., 0] * valid
