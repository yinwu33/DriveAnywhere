"""Back-projection of z-depth maps (pixel centres at integer coordinates, OpenCV camera frame)."""
import numpy as np


def backproject(depth: np.ndarray, k: np.ndarray, c2w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """World points of the valid pixels of a z-depth map.

    Args:
        depth: (h,w) z-depth.
        k: (3,3) intrinsics on the same pixel grid.
        c2w: (4,4) camera-to-world pose.
        valid: (h,w) bool mask of pixels to back-project.

    Returns:
        (N,3) float64 world points in row-major order of ``valid``.
    """
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    pts = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], axis=1)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def project(pts_world: np.ndarray, k: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a camera.

    Returns:
        (u, v, z): pixel coordinates (float) and z-depth in that camera.
    """
    pc = (pts_world - c2w[:3, 3]) @ c2w[:3, :3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * pc[:, 0] / z + k[0, 2]
        v = k[1, 1] * pc[:, 1] / z + k[1, 2]
    return u, v, z
