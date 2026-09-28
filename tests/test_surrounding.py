"""Surround cameras need translated support, calibrated scale and oriented memory."""
import numpy as np
import pytest
import json
from pathlib import Path
import sys
from PIL import Image

from dashrecon.gen.depth_alignment import apply_scene_scale, estimate_scene_scale
from dashrecon.gen.trajectory import memory_candidate_indices, surrounding_plan
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_generated_views import validate


def test_half_rigs_include_side_and_back_at_three_translated_centres() -> None:
    for side in [-1, 1]:
        views = surrounding_plan([96, 104, 112], 37, 5, side)
        assert len(views) == 121 and views[0]["yaw"] == views[0]["ramp"] == 0
        supervised = views[::3]
        for frame in [96, 104, 112]:
            angles = {v["yaw"] for v in supervised if v["segment"] == f"anchor_{frame}"}
            assert {side * a for a in [0, 45, 90, 135, 180]} <= angles
        assert max(abs(b["yaw"] - a["yaw"]) for a, b in zip(views, views[1:])) == 5
        assert max(abs(b["pose_frame"] - a["pose_frame"]) for a, b in zip(views, views[1:])) < 1.34


def test_unobserved_depth_uses_evidenced_scene_scale_and_remains_unverified() -> None:
    depth = [np.full((4, 4), 2.), np.full((4, 4), 9.)]
    rendered = [np.full((4, 4), 6.), np.full((4, 4), 100.)]
    count = [np.full((4, 4), 2.), np.zeros((4, 4))]
    scale, meta = estimate_scene_scale(depth, rendered, count, 2, 4)
    assert scale == pytest.approx(3.)
    assert meta["reference_pixels"] == 16 and not meta["frames"][1]["used_for_scale"]
    np.testing.assert_allclose(apply_scene_scale(depth[1], scale), 27.)


def test_no_scale_anchor_is_an_error_and_invalid_prior_depth_stays_invalid() -> None:
    d = np.ones((4, 4))
    with pytest.raises(ValueError, match="no supported pixels"):
        estimate_scene_scale([d], [d], [np.zeros_like(d)], 2, 4)
    np.testing.assert_array_equal(apply_scene_scale(np.array([0., -1., np.nan, np.inf, 2.]), 3.), [0., 0., 0., 0., 6.])


def test_back_memory_is_selected_over_front_at_the_same_frame() -> None:
    front, back = np.eye(4), np.eye(4)
    back[:3, :3] = np.diag([-1., 1., -1.])
    cams = [{"frame": 104, "c2w": front.tolist()} for _ in range(4)]
    cams.append({"frame": 104, "c2w": back.tolist()})
    query = {"frame": 104, "c2w": back.tolist()}
    assert memory_candidate_indices(cams, query, 16, 1, "pose") == [4]
    assert memory_candidate_indices(cams, query, 16, 3, "frame") == [0, 1, 2]
    assert memory_candidate_indices(cams, {**query, "frame": 150}, 16, 3, "pose") == []


def test_self_consistent_new_surface_cannot_replace_conflicting_validated_memory(tmp_path: Path) -> None:
    """Two translated new references alone pass; contradicted old memory rejects."""
    cams = []
    for frame, x in [(96, 0.), (104, 1.), (112, 2.)]:
        pose = np.eye(4)
        pose[0, 3] = x
        cams.append({"frame": frame, "c2w": pose.tolist(), "K": [[10.,0,8],[0,10.,6],[0,0,1]], "hw": [12,16]})
    directories = [tmp_path / "new", tmp_path / "memory"]
    for directory, intensity in zip(directories, [128, 230]):
        for sub in ["filled", "filled_depth", "mask"]:
            (directory / sub).mkdir(parents=True)
        (directory / "cams.json").write_text(json.dumps({"cams": cams}))
        for k in range(3):
            Image.fromarray(np.full((12,16,3), intensity, np.uint8)).save(directory / "filled" / f"{k:03d}.png")
            Image.fromarray(np.full((12,16), 255, np.uint8)).save(directory / "mask" / f"{k:03d}.png")
            np.save(directory / "filled_depth" / f"{k:03d}.npy", np.full((12,16), 5., np.float32))
    old_validation = tmp_path / "old_validation"
    (old_validation / "confidence").mkdir(parents=True)
    (old_validation / "validation.json").write_text(json.dumps({"views_dir": str(directories[1])}))
    for k in range(3):
        np.save(old_validation / "confidence" / f"{k:03d}.npy", np.ones((12,16), np.float16))
    common = dict(views_dir=directories[0], offsets=[-1,1], depth_tol=.1, rgb_tol=.12, min_support=2,
                  max_conflict_fraction=.25, reference_policy="translated", min_baseline=.5, max_references=4,
                  min_parallax_degrees=.5)
    before = validate(out_dir=tmp_path / "before", **common)
    after = validate(out_dir=tmp_path / "after", memory_dirs=[directories[1]],
                     memory_validation_dirs=[old_validation], **common)
    assert before["accepted_fraction"] > .4
    assert after["accepted_fraction"] == 0
    assert sum(r["memory_conflict_pixels"] for r in after["frames"]) > 0
