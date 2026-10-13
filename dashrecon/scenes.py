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


# Multi-traversal sites (docs/EXPERIMENTS.md D-M1 / MT1): several local Waymo segments that drive the same road at
# different times, found with scripts/find_revisits.py (GT poses used only for that selection). Each traversal is a
# Scene of its own (split "training": these segments have no local LiDAR); MULTI_TRAVERSALS maps a combined scene id
# to its traversals, in the frame order of the combined sequence. The combined scene is virtual: its scene_idx names
# the directory of the combined undistorted image tree (scripts/combine_traversals.py), not a Waymo segment.
MT_SCENES = (
    Scene("mt1a", "training", 368, "segment-17850487901509155700_9065_000_9085_000_with_camera_labels",
          "multi_traversal", "PHX arterial, MT1 traversal A, 2018-04-02, bright sun; 325 m", 0, -1),
    Scene("mt1b", "training", 662, "segment-6742105013468660925_3645_000_3665_000_with_camera_labels",
          "multi_traversal", "PHX arterial, MT1 traversal B, 2018-03-24, overcast look, road before crack sealing; 341 m", 0, -1),
    Scene("mt1", "multi", 901, "mt1a+mt1b", "multi_traversal",
          "MT1 combined: mt1a frames then mt1b frames, one world frame from a joint self-calibration", 0, -1),
)
MULTI_TRAVERSALS = {"mt1": ("mt1a", "mt1b")}

# Online dashcam videos (user 10-09 / 10-13: "a street dashcam video from the web through the E54s method"): Mapillary
# image sequences (CC BY-SA 4.0) fetched with scripts/fetch_mapillary.py into data/mapillary/processed/<scene_idx>/;
# scene_idx 950+ cannot collide with a Waymo processed index. segment = the Mapillary sequence id; the frames are the
# fetched slice (start / count in <scene_idx>/mapillary.json), renumbered from 0. No GT of any kind.
MAPILLARY_ROOT = "data/mapillary/processed"
MAPILLARY_SCENES = (
    Scene("mly01", "mapillary", 950, "lognym6ypc762dziaz1xr0", "online_dashcam",
          "Los Angeles residential arterial, iPhone 5C behind the windscreen, overcast, 2018-02-14; frames 200-359, "
          "~400 m straight, parked cars both sides", 0, -1),
    Scene("mly02", "mapillary", 951, "PmlcgfWQ5isOv0p2DtqSHj", "online_dashcam",
          "Warsaw city centre (Emilii Plater area), VANTRUE S1 Pro dashcam, wide-angle lens with strong barrel distortion, "
          "overcast, 2025-10-16; frames 200-359, ~500 m straight at ~16 m/s, high-rises, moving traffic", 0, -1),
    Scene("mly03", "mapillary", 952, "-LKoKeuP_b-lJTq6YEw1_A", "online_dashcam",
          "Paris 16e boulevard, car-mounted camera, winter overcast, 2019-01-04; frames 200-359, ~310 m straight, "
          "parked cars both sides, bare trees", 0, -1),
)

# E34 (docs/E34_DATA.md): Waymo training segments that become repair-model training pairs, addressed as
# "wt<idx:03d>", idx = line index in data/waymo_train_list.txt (the drivestudio processed-scene index). Their FRONT
# image tree is the undistorted, renumbered one of scripts/e34_calib.py (frames counted from 0 there), so the scene
# always spans all of its images.
TRAINING_LIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "waymo_train_list.txt")
TRAINING_PREFIX = "wt"


def training_scene_id(idx: int) -> str:
    """Scene id of the Waymo training segment on line ``idx`` of data/waymo_train_list.txt (E34)."""
    assert 0 <= idx < 1000, idx
    return f"{TRAINING_PREFIX}{idx:03d}"


def training_scene(scene_id: str) -> Scene:
    """Scene of an E34 training segment id ``wt<idx:03d>``."""
    assert len(scene_id) == len(TRAINING_PREFIX) + 3 and scene_id[len(TRAINING_PREFIX):].isdigit(), scene_id
    idx = int(scene_id[len(TRAINING_PREFIX):])
    names = [line.strip() for line in open(TRAINING_LIST) if line.strip()]
    return Scene(scene_id, "training", idx, names[idx], "e34_training", "E34 repair-model training segment", 0, -1)


def get_scene(scene_id: str) -> Scene:
    """Look up a development, multi-traversal, Mapillary or E34 training scene by id."""
    if scene_id.startswith(TRAINING_PREFIX):
        return training_scene(scene_id)
    known = DEV_SCENES + MT_SCENES + MAPILLARY_SCENES
    matches = [s for s in known if s.scene_id == scene_id]
    assert len(matches) == 1, f"unknown scene_id {scene_id}; known: {[s.scene_id for s in known]}"
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


def train_frame_mask(frames: np.ndarray, start_timestep: int, test_stride: int) -> np.ndarray:
    """True for training frames, False for held-out test frames (DECISIONS D7).

    Mirrors drivestudio ``DrivingDataset.split_train_test`` (datasets/driving_dataset.py:584-596): with
    stride s > 0 the test timesteps are s, 2s, ... counted from ``start_timestep``; s = 0 means no test frames.
    """
    assert test_stride >= 0, test_stride
    if test_stride == 0:
        return np.ones(len(frames), dtype=bool)
    rel = frames - start_timestep
    return ~((rel >= test_stride) & (rel % test_stride == 0))


def traversal_of(frame: int, starts: list) -> int:
    """Index of the traversal of a combined multi-traversal scene whose virtual frame range holds ``frame``
    (``starts``: first virtual frame of every traversal, ascending from 0; scripts/combine_traversals.py)."""
    assert starts[0] == 0 and all(a < b for a, b in zip(starts, starts[1:])), starts
    assert frame >= 0, frame
    return sum(1 for s in starts if s <= frame) - 1
