"""File layout of the E34 repair-model training data (docs/E34_DATA.md). Only numpy and the standard library, so every
venv can import it.

Raw segments (scripts/e34_fetch.py): <raw_root>/<idx:03d>/images/<t:03d>_<cam>.jpg, drivestudio camera ids
(0 FRONT, 1 FRONT_LEFT, 2 FRONT_RIGHT, 3 SIDE_LEFT, 4 SIDE_RIGHT; datasets/waymo/waymo_preprocess.py).
Per-camera products of one segment live in <dir>/cam<c>/ with the dashrecon.io conventions (frames.txt, mask_*/).
"""
import os

import numpy as np

CAM_NAMES = {0: "FRONT", 1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT"}
FRONT = 0


def raw_image(raw_root: str, idx: int, t: int, cam: int) -> str:
    """Original image of camera ``cam`` at timestep ``t`` of training segment ``idx``."""
    return os.path.join(raw_root, f"{idx:03d}", "images", f"{t:03d}_{cam}.jpg")


def num_frames(raw_root: str, idx: int) -> int:
    """Number of FRONT images of a raw segment (scripts/e34_fetch.py checked every camera has frames 0..N-1)."""
    img_dir = os.path.join(raw_root, f"{idx:03d}", "images")
    n = len([f for f in os.listdir(img_dir) if f.endswith(f"_{FRONT}.jpg")])
    assert n > 0, f"no FRONT images in {img_dir}"
    return n


def camera_frames(raw_root: str, idx: int, target_cams: list[int], target_stride: int) -> dict[int, np.ndarray]:
    """Frames used per camera: FRONT every frame, each target camera every ``target_stride``-th frame from 0."""
    assert FRONT not in target_cams and set(target_cams) <= set(CAM_NAMES), target_cams
    assert target_stride >= 1, target_stride
    n = num_frames(raw_root, idx)
    out = {FRONT: np.arange(n)}
    for c in target_cams:
        out[c] = np.arange(0, n, target_stride)
    return out


def cam_dir(root: str, cam: int) -> str:
    """Directory of one camera's products below a per-segment directory."""
    assert cam in CAM_NAMES, cam
    return os.path.join(root, f"cam{cam}")
