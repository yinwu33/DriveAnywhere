"""treat_speckle (dashrecon/gen/cache.py, D-C2): sparse warp dots are dropped or filled, dense regions are kept."""
import numpy as np

from dashrecon.gen.cache import treat_speckle


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """64 x 64: a dense block on the left, a lattice of single dots (1 in 16 pixels) on the right."""
    valid = np.zeros((64, 64), bool)
    valid[:, :24] = True
    valid[::4, 40::4] = True
    rgb = np.repeat(np.where(valid, np.float32(0.5), np.float32(-1.))[..., None], 3, -1)
    depth = np.where(valid, np.float32(10.), np.float32(0.))
    return rgb.astype(np.float32), valid, depth.astype(np.float32)


def test_keep_is_identity() -> None:
    rgb, valid, depth = scene()
    r, v, d = treat_speckle(rgb, valid, depth, "keep", 9, 0.5)
    assert (v == valid).all() and (r == rgb).all() and (d == depth).all()


def test_drop_removes_dots_keeps_block() -> None:
    rgb, valid, depth = scene()
    r, v, d = treat_speckle(rgb, valid, depth, "drop", 9, 0.5)
    assert v[:, 4:20].all()            # block interior stays
    assert not v[:, 40:].any()         # every lattice dot goes
    assert (r[~v] == -1).all() and (d[~v] == 0).all()


def test_fill_closes_lattice_with_average() -> None:
    rgb, valid, depth = scene()
    r, v, d = treat_speckle(rgb, valid, depth, "fill", 9, 0.04)
    assert v[8:56, 44:60].all()        # lattice interior is filled
    assert np.allclose(r[8:56, 44:60], 0.5) and np.allclose(d[8:56, 44:60], 10.)
    assert (r[valid & v] == rgb[valid & v]).all()  # original pixels that stay are untouched
    assert not v[:, 30:34].any()       # the gap between block and lattice stays empty


def test_fill_drops_isolated_dot() -> None:
    rgb, valid, depth = scene()
    valid[60, 30] = True
    r, v, d = treat_speckle(rgb, valid, depth, "fill", 9, 0.04)
    assert not v[60, 30] and r[60, 30, 0] == -1 and d[60, 30] == 0
