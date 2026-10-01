"""Pair lists of the joint multi-traversal calibration (dashrecon.pose.calib.sequence_pairs)."""
from dashrecon.pose.calib import sequence_pairs


def test_sequence_pairs_offsets() -> None:
    names = [f"{i:03d}" for i in range(40)]
    pairs = sequence_pairs(names, 3)
    offsets = {int(b) - int(a) for a, b in pairs}
    assert offsets == {1, 2, 3, 4}  # 1..3 plus 2^k < 40 for k < 3
    assert all(int(b) < 40 for _, b in pairs)
    assert len(pairs) == len(set(pairs))


def test_sequence_pairs_quadratic_reach() -> None:
    names = [str(i) for i in range(200)]
    offsets = {int(b) - int(a) for a, b in sequence_pairs(names, 20)}
    assert offsets == set(range(1, 21)) | {32, 64, 128}
