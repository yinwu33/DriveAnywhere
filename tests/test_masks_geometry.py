"""Synthetic tests for mask helpers (no models, no Waymo data).

Run: PYTHONPATH=. .venvs/masks/bin/python -m pytest tests -q
"""
import numpy as np

from dashrecon import io

GRID = {"image_hw": [1280, 1920], "resized_hw": [345, 518], "crop_top_left": [4, 0], "depth_hw": [336, 518]}


def test_mask_to_depth_grid_shape_and_location() -> None:
    mask = np.zeros((1280, 1920), dtype=bool)
    mask[600:700, 900:1000] = True  # a 100x100 block around image row 650, col 950
    small = io.mask_to_depth_grid(mask, GRID)
    assert small.shape == (336, 518)
    rows, cols = np.nonzero(small)
    # block centre maps to ((650 + 0.5) * 345/1280 - 0.5 - 4, (950 + 0.5) * 518/1920 - 0.5)
    assert abs(rows.mean() - ((650 + 0.5) * 345 / 1280 - 0.5 - 4)) < 1.0
    assert abs(cols.mean() - ((950 + 0.5) * 518 / 1920 - 0.5)) < 1.0


def test_dilate_grows_by_radius() -> None:
    from dashrecon.masks.sam_text import dilate

    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True
    grown = dilate(mask, 5)
    assert grown[20, 25] and grown[15, 20] and not grown[20, 26] and not grown[24, 24]
    np.testing.assert_array_equal(dilate(mask, 0), mask)
