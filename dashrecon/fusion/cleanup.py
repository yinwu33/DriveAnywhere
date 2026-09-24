"""Voxel downsampling, statistical outlier removal and normal estimation (AGENTS.md Phase 5, tasks 3 and 5).

Voxel averaging is done in numpy so per-point labels survive (Open3D's voxel_down_sample drops them);
outlier removal and normal estimation use Open3D 0.16 (``remove_statistical_outlier``,
``estimate_normals`` with ``KDTreeSearchParamKNN``).
"""
import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average points per voxel.

    Args:
        xyz: (N,3) points. rgb: (N,3) uint8. labels: (N,) uint8 in {0, 1}.
        voxel: voxel edge length (scene units).

    Returns:
        Per-voxel mean xyz (float64), mean rgb (uint8) and majority label (1 if more than half are 1).
    """
    assert set(np.unique(labels)) <= {0, 1}, "voxel majority vote expects binary labels"
    key3 = np.floor(xyz / voxel).astype(np.int64)
    key3 -= key3.min(axis=0)
    dims = key3.max(axis=0) + 1
    assert float(dims[0]) * dims[1] * dims[2] < 2**62, f"voxel grid too large: {dims}"
    key = (key3[:, 0] * dims[1] + key3[:, 1]) * dims[2] + key3[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out_xyz = np.stack([np.bincount(inv, weights=xyz[:, a]) for a in range(3)], axis=1) / cnt[:, None]
    out_rgb = np.stack([np.bincount(inv, weights=rgb[:, a].astype(np.float64)) for a in range(3)], axis=1) / cnt[:, None]
    out_lab = (np.bincount(inv, weights=labels.astype(np.float64)) / cnt > 0.5).astype(np.uint8)
    return out_xyz, np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8), out_lab


def statistical_outlier_keep(xyz: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Boolean keep-mask from Open3D statistical outlier removal."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, keep_idx = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    keep = np.zeros(len(xyz), dtype=bool)
    keep[np.asarray(keep_idx)] = True
    return keep


def oriented_normals(xyz: np.ndarray, knn: int, cam_centers: np.ndarray) -> np.ndarray:
    """PCA normals (Open3D, k nearest neighbours) flipped to face the nearest camera centre.

    Args:
        xyz: (N,3) points.
        knn: neighbours for the local plane fit.
        cam_centers: (M,3) camera centres of the frames that produced the points.

    Returns:
        (N,3) unit normals with n . (c_nearest - p) >= 0.
    """
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(pc.normals).copy()
    _, nearest = cKDTree(cam_centers).query(xyz)
    flip = np.einsum("ij,ij->i", normals, cam_centers[nearest] - xyz) < 0
    normals[flip] *= -1
    return normals
