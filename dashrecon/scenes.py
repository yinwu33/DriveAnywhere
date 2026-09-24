"""Development scenes (AGENTS.md Phase 0, task 2; DECISIONS D2).

Picked from the 100 local Waymo v1.4.3 *validation* segments (they contain LiDAR for the deferred
evaluation) using ``scripts/scene_stats.py`` -> ``data/dashrecon/scene_selection/validation_stats.json``.
All are daytime, sunny, and the ego drives more than 80 m. ``scene_idx`` is the line index in
``data/waymo_val_list.txt`` and the drivestudio processed-scene index under
``data/waymo/processed/validation/``. GT labels were used only for this selection.
"""
import os
from dataclasses import dataclass

import numpy as np

FRONT_CAM_ID = 0  # drivestudio camera index of Waymo FRONT (datasets/waymo/waymo_preprocess.py:100)


@dataclass(frozen=True)
class Scene:
    """One development scene."""

    scene_id: str
    split: str
    scene_idx: int
    segment: str
    category: str
    note: str
    start_timestep: int
    end_timestep: int  # -1 = last frame


DEV_SCENES = (
    Scene("val056", "validation", 56, "segment-2367305900055174138_1881_827_1901_827_with_camera_labels",
          "static", "PHX, 219 m nearly straight, 0 moving / 2.8 parked vehicles per frame", 0, -1),
    Scene("val039", "validation", 39, "segment-15948509588157321530_7187_290_7207_290_with_camera_labels",
          "static", "162 m straight, 0.4 moving / 10.2 parked vehicles per frame", 0, -1),
    Scene("val041", "validation", 41, "segment-16979882728032305374_2719_000_2739_000_with_camera_labels",
          "dynamic", "PHX, 199 m straight, 12.9 moving vehicles per frame", 0, -1),
    Scene("val087", "validation", 87, "segment-7799643635310185714_680_000_700_000_with_camera_labels",
          "dynamic", "SF, 197 m at up to 23 m/s, 9.2 moving / 6.9 parked vehicles per frame", 0, -1),
    Scene("val094", "validation", 94, "segment-902001779062034993_2880_000_2900_000_with_camera_labels",
          "slope_curve", "SF, 182 m, 20.6 m elevation change and 68 deg heading change", 0, -1),
)


def get_scene(scene_id: str) -> Scene:
    """Look up a development scene by id."""
    matches = [s for s in DEV_SCENES if s.scene_id == scene_id]
    assert len(matches) == 1, f"unknown scene_id {scene_id}; known: {[s.scene_id for s in DEV_SCENES]}"
    return matches[0]


def front_image_paths(scene_dir: str, start: int, end: int) -> tuple[np.ndarray, list[str]]:
    """FRONT image paths for timesteps [start, end); end = -1 means up to the last frame."""
    img_dir = os.path.join(scene_dir, "images")
    front = sorted(f for f in os.listdir(img_dir) if f.endswith(f"_{FRONT_CAM_ID}.jpg"))
    num = len(front)
    assert num > 0, f"no FRONT images in {img_dir}"
    stop = num if end == -1 else end
    assert 0 <= start < stop <= num, (start, stop, num)
    frames = np.arange(start, stop)
    paths = [os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg") for t in frames]
    for p in paths:
        assert os.path.exists(p), p
    return frames, paths
