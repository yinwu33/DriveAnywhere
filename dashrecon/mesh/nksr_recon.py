"""NKSR surface reconstruction from the fused point cloud (AGENTS.md Phase 7, tasks 1-3; runs in .venvs/nksr).

API confirmed in nv-tlabs/NKSR @ e403368 (package/nksr):
    - ``Reconstructor(device).reconstruct(xyz, normal=..., detail_level=..., ...)`` returns a field.
      The pretrained model has a fixed voxel size in the units it was trained on (metres);
      ``detail_level`` instead rescales the input so its point density falls in the model's trained
      density range, which is what we need because our clouds are in pose-run units, not metres.
      ``detail_level=None`` would assume the trained (metric) scale.
    - ``field.set_texture_field(fields.PCNNField(xyz, rgb))`` attaches colours;
      ``field.extract_dual_mesh(mise_iter=..., trim=True)`` returns ``MeshingResult(v, f, c)`` trimmed by
      the built-in mask field (level set 2 x voxel size) so the surface stays near the input points.
"""
import time

import numpy as np
import torch


def reconstruct_mesh(xyz: np.ndarray, normals: np.ndarray, rgb: np.ndarray, detail_level: float,
                     mise_iter: int, solver_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run NKSR on an oriented, coloured point cloud.

    Args:
        xyz: (N,3) points. normals: (N,3) oriented unit normals. rgb: (N,3) uint8 colours.
        detail_level: NKSR detail level in [0, 1] (density-adaptive input scaling).
        mise_iter: Multi-IsoSurface Extraction iterations.
        solver_tol: PCG solver tolerance.

    Returns:
        (vertices (V,3) float32, faces (F,3) int32, vertex colours (V,3) uint8, info dict).
    """
    import nksr
    from nksr import fields

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    recon = nksr.Reconstructor(device)
    pts = torch.from_numpy(xyz.astype(np.float32)).to(device)
    nrm = torch.from_numpy(normals.astype(np.float32)).to(device)
    col = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device)
    field = recon.reconstruct(pts, normal=nrm, detail_level=detail_level, approx_kernel_grad=True,
                              solver_tol=solver_tol, fused_mode=True)
    field.set_texture_field(fields.PCNNField(pts, col))
    mesh = field.extract_dual_mesh(mise_iter=mise_iter, trim=True)
    torch.cuda.synchronize(device)
    v = mesh.v.float().cpu().numpy()
    f = mesh.f.int().cpu().numpy()
    c = np.clip(np.rint(mesh.c.float().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    info = {
        "model_voxel_size": float(recon.hparams.voxel_size),
        "runtime_s": time.time() - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "nksr_version": nksr.__version__,
        "torch": torch.__version__,
    }
    return v, f, c, info


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """(E,2) edges used by exactly one triangle (mesh borders and hole rims)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]
