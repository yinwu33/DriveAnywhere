"""Self-calibration of one dashcam from its own video (DECISIONS D13; runs in the ``.venvs/sfm`` venv).

COLMAP 4.2 through pycolmap, API confirmed against pycolmap 4.2.0:
    - ``extract_features`` (SIFT, CPU) on the listed original FRONT images, one shared camera
      (``CameraMode.SINGLE``); pixels where the mask PNG ``<mask_path>/<image name>.png`` is 0 get no
      features (dynamic objects and sky, from the Phase 4 masks of the same images);
    - ``match_sequential`` (neighbouring frames, quadratic overlap, no loop detection);
    - ``global_mapping`` (GLOMAP). Incremental mapping failed on forward driving (2 of 199 frames registered
      on val039, DECISIONS K), the global mapper registered every frame of the 5 development scenes.
The camera model is COLMAP ``RADIAL`` (f, cx, cy, k1, k2) with the principal point kept at the image centre;
its distortion is the OpenCV radial model with p1 = p2 = k3 = 0. COLMAP puts pixel centres at +0.5,
dashrecon at integer coordinates: ``cx_dashrecon = cx_colmap - 0.5``.

Nothing here reads calibration, poses or LiDAR (AGENTS.md section 10).
"""
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from dashrecon import io


@dataclass
class CalibResult:
    """Shared camera and SfM trajectory of one sequence.

    Attributes:
        frames: (N,) frame indices, all registered.
        K: (3,3) pinhole intrinsics, pixel centres at integer coordinates (the undistorted images use it too).
        dist: (2,) OpenCV radial coefficients k1, k2 of the original images.
        poses_c2w: (N,4,4) OpenCV camera-to-world poses in the SfM frame and SfM units.
        obs: sparse observations for scale alignment, see :func:`dashrecon.io.write_sparse_obs`.
        stats: registration, point, reprojection-error and timing statistics.
    """

    frames: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    poses_c2w: np.ndarray
    obs: dict
    stats: dict


def write_feature_masks(mask_root: str, frames: np.ndarray, names: list[str], out_dir: str) -> None:
    """COLMAP feature masks: 255 where features may be extracted, 0 on dynamic objects and sky."""
    os.makedirs(out_dir, exist_ok=True)
    for t, name in zip(frames, names):
        blocked = io.read_mask(mask_root, "dynamic", int(t)) | io.read_mask(mask_root, "sky", int(t))
        Image.fromarray(np.where(blocked, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f"{name}.png"))


def run_glomap(image_dir: str, names: list[str], frames: np.ndarray, mask_root: str, work_dir: str,
               max_features: int, overlap: int, seed: int) -> CalibResult:
    """Self-calibrate one shared RADIAL camera and the trajectory of an ordered image sequence.

    Args:
        image_dir: directory holding the images listed in ``names`` (original, distorted FRONT images).
        names: image file names in frame order.
        frames: frame index of each name.
        mask_root: Phase 4 mask directory of the same images (``mask_dynamic``, ``mask_sky``).
        work_dir: scratch directory for the COLMAP database, feature masks and the sparse model.
        max_features: SIFT features per image.
        overlap: sequential matching overlap (quadratic overlap on top).
        seed: random seed of the global mapper.
    """
    import pycolmap

    mask_dir = os.path.join(work_dir, "feature_masks")
    write_feature_masks(mask_root, frames, names, mask_dir)
    db = os.path.join(work_dir, "database.db")
    assert not os.path.exists(db), f"{db} exists: start from an empty work dir"
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "RADIAL"
    reader.mask_path = mask_dir
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = max_features
    pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    t1 = time.time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pairing)
    t2 = time.time()
    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir)
    options = pycolmap.GlobalPipelineOptions()
    options.random_seed = seed
    recs = pycolmap.global_mapping(db, image_dir, sparse_dir, options)
    t3 = time.time()
    assert len(recs) == 1, f"global mapping produced {len(recs)} models"
    rec = next(iter(recs.values()))
    assert len(rec.cameras) == 1, f"expected one shared camera, got {len(rec.cameras)}"
    cam = next(iter(rec.cameras.values()))
    assert str(cam.model).endswith("RADIAL") and not str(cam.model).endswith("SIMPLE_RADIAL"), cam.model
    f, cx, cy, k1, k2 = (float(x) for x in cam.params)
    K = np.array([[f, 0.0, cx - 0.5], [0.0, f, cy - 0.5], [0.0, 0.0, 1.0]])

    by_name = {img.name: img for img in rec.images.values()}
    missing = [n for n in names if n not in by_name or not by_name[n].has_pose]
    assert not missing, f"{len(missing)} frames not registered: {missing[:10]}"
    poses, obs_f, obs_u, obs_v, obs_z = [], [], [], [], []
    for t, name in zip(frames, names):
        img = by_name[name]
        w2c = np.eye(4)
        w2c[:3, :4] = img.cam_from_world().matrix()
        poses.append(np.linalg.inv(w2c))
        xyz = np.array([rec.points3D[p.point3D_id].xyz for p in img.points2D if p.has_point3D()])
        pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        pc = pc[pc[:, 2] > 0]
        obs_f.append(np.full(len(pc), t))
        obs_u.append(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2])
        obs_v.append(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2])
        obs_z.append(pc[:, 2])
    stats = {
        "registered": len(names), "points3D": int(rec.num_points3D()),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "colmap_params": [f, cx, cy, k1, k2],
        "runtime_s": {"extract": t1 - t0, "match": t2 - t1, "map": t3 - t2},
    }
    return CalibResult(frames=frames, K=K, dist=np.array([k1, k2]), poses_c2w=np.stack(poses),
                       obs={"frame": np.concatenate(obs_f), "u": np.concatenate(obs_u), "v": np.concatenate(obs_v),
                            "z": np.concatenate(obs_z)}, stats=stats)
