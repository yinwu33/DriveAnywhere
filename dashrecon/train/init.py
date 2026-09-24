"""Background Gaussian initialisation from dashrecon products (upstream hook P3, DECISIONS D9).

``from_dashrecon`` config (under ``model.Background.init``):
    source: "ply"   sample ``num_samples`` points of a point cloud written by dashrecon.io.write_ply
            "mesh"  sample ``num_samples`` points uniformly by area on a mesh (dashrecon.io.write_mesh_ply);
                    each sample becomes a flat Gaussian lying in its triangle's plane (LSD-3D-style
                    geometry grounding): tangential scale = radius of a disc of area A_total / num_samples,
                    normal scale = ``normal_scale_ratio`` x tangential (kept above 1/10 so the upstream
                    sharp_shape_reg, max ratio 10, does not fight the initialisation)
    path, num_samples, seed (and normal_scale_ratio for "mesh")
Quaternions are wxyz, the gsplat convention (gsplat.cuda_legacy._torch_impl.normalized_quat_to_rotmat).
"""
import numpy as np
import torch

from dashrecon import io


def rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) unit quaternions (w, x, y, z), w >= 0."""
    m = r
    t = np.stack([
        1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
        1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
        1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
        1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
    ], axis=1)
    k = t.argmax(axis=1)
    q = np.empty((len(m), 4))
    s = np.sqrt(np.maximum(t[np.arange(len(m)), k], 1e-12)) * 2  # s = 4 * |component k|
    for case in range(4):
        sel = k == case
        mm, ss = m[sel], s[sel]
        if case == 0:
            q[sel] = np.stack([0.25 * ss, (mm[:, 2, 1] - mm[:, 1, 2]) / ss, (mm[:, 0, 2] - mm[:, 2, 0]) / ss,
                               (mm[:, 1, 0] - mm[:, 0, 1]) / ss], axis=1)
        elif case == 1:
            q[sel] = np.stack([(mm[:, 2, 1] - mm[:, 1, 2]) / ss, 0.25 * ss, (mm[:, 0, 1] + mm[:, 1, 0]) / ss,
                               (mm[:, 0, 2] + mm[:, 2, 0]) / ss], axis=1)
        elif case == 2:
            q[sel] = np.stack([(mm[:, 0, 2] - mm[:, 2, 0]) / ss, (mm[:, 0, 1] + mm[:, 1, 0]) / ss, 0.25 * ss,
                               (mm[:, 1, 2] + mm[:, 2, 1]) / ss], axis=1)
        else:
            q[sel] = np.stack([(mm[:, 1, 0] - mm[:, 0, 1]) / ss, (mm[:, 0, 2] + mm[:, 2, 0]) / ss,
                               (mm[:, 1, 2] + mm[:, 2, 1]) / ss, 0.25 * ss], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


def frame_from_normals(n: np.ndarray) -> np.ndarray:
    """(N,3) unit normals -> (N,3,3) rotations whose third column is the normal."""
    helper = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(helper, n)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=2)


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray, num_samples: int, normal_scale_ratio: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-uniform surface samples with colours and flat-Gaussian orientation/scale.

    Returns:
        points (M,3), colours (M,3) in [0,1], quaternions (M,4) wxyz, log-scales (M,3).
    """
    tri = vertices[faces].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 0
    tri, cross, area2, fidx = tri[keep], cross[keep], area2[keep], faces[keep]
    face = rng.choice(len(tri), size=num_samples, p=area2 / area2.sum())
    r1, r2 = np.sqrt(rng.random(num_samples)), rng.random(num_samples)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    pts = np.einsum("mk,mkd->md", w, tri[face])
    col = np.einsum("mk,mkd->md", w, rgb[fidx[face]].astype(np.float64)) / 255.0
    normals = cross[face] / area2[face, None]
    quats = rotmat_to_quat_wxyz(frame_from_normals(normals))
    s_t = np.sqrt(0.5 * area2.sum() / (np.pi * num_samples))
    log_scales = np.log(np.tile([s_t, s_t, normal_scale_ratio * s_t], (num_samples, 1)))
    return pts, col, quats, log_scales


def sample_init_points(cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """Initial Background points (and, for meshes, Gaussian geometry) from a dashrecon product.

    Args:
        cfg: the ``from_dashrecon`` init config (see module docstring).
        device: torch device.

    Returns:
        (points (M,3), colours (M,3) in [0,1], geometry dict with ``quats``/``log_scales`` or None).
    """
    rng = np.random.default_rng(cfg.seed)
    if cfg.source == "ply":
        data = io.read_ply(cfg.path)
        pick = rng.choice(len(data["xyz"]), size=min(cfg.num_samples, len(data["xyz"])), replace=False)
        pts, col, geometry = data["xyz"][pick], data["rgb"][pick] / 255.0, None
    elif cfg.source == "mesh":
        mesh = io.read_mesh_ply(cfg.path)
        pts, col, quats, log_scales = sample_mesh(mesh["vertices"], mesh["faces"], mesh["rgb"], cfg.num_samples,
                                                  cfg.normal_scale_ratio, rng)
        geometry = {"quats": torch.from_numpy(quats).float().to(device),
                    "log_scales": torch.from_numpy(log_scales).float().to(device)}
    else:
        raise ValueError(f"unknown from_dashrecon.source {cfg.source!r}; expected 'ply' or 'mesh'")
    return (torch.from_numpy(np.asarray(pts)).float().to(device),
            torch.from_numpy(np.asarray(col)).float().to(device), geometry)


def apply_init_geometry(model, geometry: dict) -> None:
    """Overwrite quaternions and log-scales of the first M Gaussians (the mesh samples) after create_from_pcd."""
    m = len(geometry["quats"])
    assert m <= model._quats.shape[0], (m, model._quats.shape)
    assert model._scales.shape[1] == 3, "mesh initialisation needs 3D (non-ball, non-2D) Gaussians"
    model._quats.data[:m] = geometry["quats"]
    model._scales.data[:m] = geometry["log_scales"]
