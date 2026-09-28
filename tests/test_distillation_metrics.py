"""Diagnostic metrics must not reward unrelated noise or unsupported regions."""
import numpy as np
import pytest

from dashrecon.eval.distillation import fit_metrics


def test_identical_texture_and_unrelated_noise():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noise = rng.integers(0, 256, target.shape, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    same = fit_metrics(target, target, mask)
    other = fit_metrics(target, noise, mask)
    assert same["mse"] == 0
    assert same["laplacian_correlation"] == pytest.approx(1)
    assert 0.7 < other["detail_variance_ratio"] < 1.3
    assert abs(other["laplacian_correlation"]) < 0.1


def test_mask_excludes_boundary_and_unknown_is_not_success():
    target = np.zeros((20, 20, 3), dtype=np.uint8)
    render = np.full_like(target, 255)
    mask = np.zeros((20, 20), dtype=bool)
    empty = fit_metrics(target, render, mask)
    assert empty["pixels"] == 0 and empty["psnr"] is None
    mask[5:15, 5:15] = True
    render[6:14, 6:14] = 0
    matched = fit_metrics(target, render, mask)
    assert matched["pixels"] == 64 and matched["mse"] == 0
    assert matched["laplacian_correlation"] is None


def test_mismatched_coordinates_fail():
    with pytest.raises(ValueError, match="identical image coordinates"):
        fit_metrics(np.zeros((20, 20, 3), np.uint8), np.zeros((19, 20, 3), np.uint8), np.ones((20, 20), bool))
