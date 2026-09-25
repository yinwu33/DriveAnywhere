"""Abstract interface of pose + pointmap backends (AGENTS.md Phase 3, task 1)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseResult:
    """Output of a pose backend, in the backend's own world frame.

    Attributes:
        intrinsics: (N,3,3) intrinsics on the original image grid (pixel centres at integers).
        poses_c2w: (N,4,4) OpenCV camera-to-world poses.
        depth: (N,h,w) float32 z-depth in metres on the depth grid; 0 = invalid.
        depth_conf: (N,h,w) float32 confidence on the depth grid.
        depth_grid: mapping from the original image grid to the depth grid, see ``dashrecon.io``.
        meta: backend-specific information (model id, version, parameters, runtime, peak VRAM).
    """

    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    depth: np.ndarray
    depth_conf: np.ndarray
    depth_grid: dict
    meta: dict


class PoseBackend(ABC):
    """A model that maps an image sequence to intrinsics, poses and per-frame depth."""

    name: str

    @abstractmethod
    def estimate(self, image_paths: list[str], intrinsics: np.ndarray | None, poses_c2w: np.ndarray | None) -> PoseResult:
        """Estimate cameras and depth for an ordered image sequence.

        Args:
            image_paths: ordered image files (one per frame).
            intrinsics: known (3,3) or (N,3,3) intrinsics on the image grid, e.g. from a self-calibration
                (DECISIONS D13), or ``None`` for the backend to estimate them. Never Waymo calibration (D3).
            poses_c2w: known (N,4,4) OpenCV camera-to-world poses up to scale (needs ``intrinsics``), or ``None``.

        Returns:
            A :class:`PoseResult` with one entry per image.
        """
