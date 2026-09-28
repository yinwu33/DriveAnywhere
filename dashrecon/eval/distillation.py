"""Masked image diagnostics for synthetic targets and same-camera 3D renders.

Used by scripts/diagnose_distillation.py. These are fitting measurements, not
ground-truth image quality scores. Empty masks and constant images remain unknown.
"""
import cv2
import numpy as np


def fit_metrics(target: np.ndarray, render: np.ndarray, mask: np.ndarray) -> dict:
    """Measure matched RGB and high-frequency detail inside an eroded region.

Erosion excludes artificial mask boundaries. Laplacian correlation distinguishes
matching detail from unrelated sharp noise; variance alone cannot do that.
"""
    if target.dtype != np.uint8 or render.dtype != np.uint8 or mask.dtype != bool:
        raise ValueError("expected uint8 RGB images and a boolean mask")
    if target.shape != render.shape or target.shape != (*mask.shape, 3):
        raise ValueError("target/render/mask must use identical image coordinates")
    inside = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count = int(inside.sum())
    result = {"pixels": count, "mse": None, "mae": None, "psnr": None,
              "target_laplacian_variance": None, "render_laplacian_variance": None,
              "detail_variance_ratio": None, "laplacian_correlation": None}
    if count == 0:
        return result
    difference = (target.astype(np.float32) - render.astype(np.float32))[inside] / 255
    mse = float(np.mean(difference ** 2))
    result.update(mse=mse, mae=float(np.mean(np.abs(difference))),
                  psnr=float(-10 * np.log10(max(mse, 1e-12))))
    a = cv2.Laplacian(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)[inside]
    b = cv2.Laplacian(cv2.cvtColor(render, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)[inside]
    va, vb = float(a.var()), float(b.var())
    result.update(target_laplacian_variance=va, render_laplacian_variance=vb)
    if va > 0:
        result["detail_variance_ratio"] = vb / va
    if va > 0 and vb > 0:
        result["laplacian_correlation"] = float(np.mean((a - a.mean()) * (b - b.mean())) / np.sqrt(va * vb))
    return result
