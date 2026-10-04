"""Joint self-calibration of all cameras of one Waymo segment (E34, docs/E34_DATA.md; runs in the ``.venvs/sfm`` venv).

Like the multi-traversal calibration ``dashrecon.pose.calib.run_glomap_multi`` (MT1), with the cameras of one car in
place of the traversals: every camera's original images go into one GLOMAP reconstruction, one COLMAP RADIAL camera per
camera (``CameraMode.PER_FOLDER``, principal point at the image centre), so the target cameras' intrinsics,
distortion and per-image poses come out in the frame of the FRONT trajectory. No rig is imposed: Waymo's cameras are
not triggered together, and per-image poses put every image where it was taken. Nothing reads the Waymo calibration,
GT poses or LiDAR (AGENTS.md section 10).

Pairs (all imported, ``match_image_pairs``):
    - within a camera: COLMAP-style sequential pairs (``dashrecon.pose.calib.sequence_pairs``, own overlap per camera);
    - across cameras (``cross``: (cam a, cam b, window, key_stride)): every key_stride-th image of a with every image
      of b taken within ``window`` frames of it. The camera pairs are the neighbours by name (FRONT-FRONT_LEFT, ...);
      which images see the same place is left to feature matching and geometric verification.
Unlike MT1 not every image has to register: the reconstruction with the most FRONT images is kept and the caller
decides whether the registered fractions suffice.

Principal points: GLOMAP keeps every principal point at the image centre (as D13 does for FRONT). The SIDE cameras'
1920x886 images are crops of a taller sensor, so their principal point is far from the centre; with it fixed, the
reconstruction absorbs the offset as a ~5 degree pitch plus an off-centre distortion (wt646, found with the
evaluation-side GT diagnostic scripts/e34_diag_calib.py: SIDE -> FRONT rotation 5.3-5.4 deg off). A final bundle
adjustment therefore refines focal length, distortion and principal point of the ``refine_pp`` cameras (the cameras
whose images are crops), with the intrinsics of the other cameras held constant and every pose and point free
(Huber loss, as GLOMAP's own bundle adjustment). On wt646 it moved SIDE cy from 442 to ~270 and the SIDE -> FRONT
rotation error to 0.6-0.7 deg, at the same mean reprojection error.

API confirmed against pycolmap 4.2.0: ``extract_features``, ``match_image_pairs(database, matching_options,
pairing_options)``, ``global_mapping`` -> {id: Reconstruction}, ``FeatureExtractionOptions.num_threads``,
``FeatureMatchingOptions.num_threads``, ``GlobalPipelineOptions.num_threads / random_seed``.
"""
import os
import time
from dataclasses import dataclass

import numpy as np

from dashrecon.pose.calib import sequence_pairs, write_feature_masks


@dataclass
class CameraCalib:
    """One camera of a joint calibration.

    Attributes:
        frames: (M,) timesteps of the registered images, ascending.
        K: (3,3) pinhole intrinsics (pixel centres at integer coordinates), also those of the undistorted images.
        dist: (2,) OpenCV radial k1, k2 of the original images.
        image_hw: [h, w] of the original images.
        poses_c2w: (M,4,4) OpenCV camera-to-world poses in the shared SfM frame and units.
        obs: sparse observations of the registered images (dashrecon.io.write_sparse_obs layout).
        stats: images used / registered, reprojection error of this camera's observations, COLMAP parameters.
    """

    frames: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    image_hw: list
    poses_c2w: np.ndarray
    obs: dict
    stats: dict


def cross_pairs(names_a: list[str], frames_a: np.ndarray, names_b: list[str], frames_b: np.ndarray, window: int,
                key_stride: int) -> list[tuple[str, str]]:
    """Every key_stride-th image of camera a with every image of camera b within ``window`` frames of it."""
    out = []
    for i in range(0, len(names_a), key_stride):
        near = np.nonzero(np.abs(frames_b - frames_a[i]) <= window)[0]
        out += [(names_a[i], names_b[j]) for j in near]
    return out


def radial_project(xyz_cam: np.ndarray, params: tuple) -> np.ndarray:
    """COLMAP RADIAL projection (f, cx, cy, k1, k2; pixel centres at +0.5) of camera-frame points, (N,2)."""
    f, cx, cy, k1, k2 = params
    x, y = xyz_cam[:, 0] / xyz_cam[:, 2], xyz_cam[:, 1] / xyz_cam[:, 2]
    r2 = x * x + y * y
    d = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * x * d + cx, f * y * d + cy], axis=1)


def refine_principal_points(rec, refine_labels: list[str], num_threads: int) -> dict:
    """Bundle adjustment of a reconstruction with focal length, distortion and principal point free for the cameras of
    the images in folders ``refine_labels`` and all other intrinsics constant; poses and points free. In place."""
    import pycolmap

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        config.add_image(image_id)
    refine = {im.camera_id for im in rec.images.values() if im.has_pose and im.name.split("/")[0] in refine_labels}
    for cam_id in rec.cameras:
        if cam_id not in refine:
            config.set_constant_cam_intrinsics(cam_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_principal_point = True
    options.print_summary = False
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.solver_options.num_threads = num_threads
    t0 = time.time()
    summary = pycolmap.create_default_bundle_adjuster(options, config, rec).solve()
    return {"cameras_refined": len(refine), "seconds": time.time() - t0, "termination": str(summary.termination_type)}


def run_glomap_cameras(groups: list[dict], cross: list[tuple[int, int, int, int]], work_dir: str, max_features: int,
                       seed: int, num_threads: int, refine_pp: list[int]) -> tuple[dict, dict]:
    """Joint GLOMAP self-calibration of the cameras of one segment.

    Args:
        groups: one dict per camera: ``cam`` (id), ``label`` (folder name), ``image_dir`` (original images),
            ``names`` (file names in frame order), ``frames`` (timesteps), ``mask_root`` (Phase 4 masks of the same
            images: no features on dynamic objects or sky), ``overlap`` (sequential overlap within the camera).
        cross: (cam a, cam b, window, key_stride) for the pairs across cameras, see :func:`cross_pairs`.
        work_dir: scratch directory (images linked into ``images/<label>/``, feature masks, database, sparse model).
        max_features: SIFT features per image.
        seed: random seed of the global mapper.
        num_threads: CPU threads of extraction, matching, mapping and the final bundle adjustment.
        refine_pp: cameras whose principal point (with focal length and distortion) the final bundle adjustment
            refines, see :func:`refine_principal_points`; empty = no final bundle adjustment.

    Returns:
        ({cam: CameraCalib of its registered images} for the cameras with at least one registered image,
         statistics: per-camera registration, pairs, models, points, reprojection error, runtime).
    """
    import pycolmap

    image_root, mask_dir = os.path.join(work_dir, "images"), os.path.join(work_dir, "feature_masks")
    by_cam = {g["cam"]: g for g in groups}
    all_names = []
    for g in groups:
        os.makedirs(os.path.join(image_root, g["label"]))
        write_feature_masks(g["mask_root"], g["frames"], g["names"], os.path.join(mask_dir, g["label"]))
        for n in g["names"]:
            os.symlink(os.path.abspath(os.path.join(g["image_dir"], n)), os.path.join(image_root, g["label"], n))
            all_names.append(f"{g['label']}/{n}")
    db = os.path.join(work_dir, "database.db")
    assert not os.path.exists(db), f"{db} exists: start from an empty work dir"
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "RADIAL"
    reader.mask_path = mask_dir
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = max_features
    extraction.num_threads = num_threads
    pycolmap.extract_features(db, image_root, image_names=all_names, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=extraction)
    t1 = time.time()
    full = {g["cam"]: [f"{g['label']}/{n}" for n in g["names"]] for g in groups}
    within = []
    for g in groups:
        within += sequence_pairs(full[g["cam"]], g["overlap"])
    across = []
    for a, b, window, key_stride in cross:
        across += cross_pairs(full[a], by_cam[a]["frames"], full[b], by_cam[b]["frames"], window, key_stride)
    pair_file = os.path.join(work_dir, "pairs.txt")
    with open(pair_file, "w") as f:
        f.writelines(f"{x} {y}\n" for x, y in within + across)
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = pair_file
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    pycolmap.match_image_pairs(db, matching_options=matching, pairing_options=pairing)
    t2 = time.time()
    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir)
    options = pycolmap.GlobalPipelineOptions()
    options.random_seed = seed
    options.num_threads = num_threads
    recs = pycolmap.global_mapping(db, image_root, sparse_dir, options)
    t3 = time.time()
    assert len(recs) >= 1, "global mapping produced no model"
    front_label = by_cam[groups[0]["cam"]]["label"]
    n_front = {k: sum(1 for im in r.images.values() if im.has_pose and im.name.startswith(front_label + "/"))
               for k, r in recs.items()}
    best = max(n_front, key=lambda k: (n_front[k], -k))
    rec = recs[best]
    refine_stats = None
    if refine_pp:
        refine_stats = refine_principal_points(rec, [by_cam[c]["label"] for c in refine_pp], num_threads)
        refined_dir = os.path.join(work_dir, "sparse", "refined")
        os.makedirs(refined_dir)
        rec.write(refined_dir)
    t4 = time.time()
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}

    results, label_of, per_camera = {}, {}, {}
    for g in groups:
        reg = [(t, n) for t, n in zip(g["frames"], g["names"]) if f"{g['label']}/{n}" in by_name]
        cam_stats = {"images": len(g["names"]), "registered": len(reg)}
        per_camera[str(g["cam"])] = cam_stats
        if not reg:
            continue
        cam_ids = {by_name[f"{g['label']}/{n}"].camera_id for _, n in reg}
        assert len(cam_ids) == 1, f"{g['label']}: registered images on {len(cam_ids)} cameras"
        cam = rec.cameras[cam_ids.pop()]
        assert str(cam.model).endswith("RADIAL") and not str(cam.model).endswith("SIMPLE_RADIAL"), cam.model
        params = tuple(float(x) for x in cam.params)
        f_, cx, cy, k1, k2 = params
        K = np.array([[f_, 0.0, cx - 0.5], [0.0, f_, cy - 0.5], [0.0, 0.0, 1.0]])
        poses, obs_f, obs_u, obs_v, obs_z, err = [], [], [], [], [], []
        for t, n in reg:
            img = by_name[f"{g['label']}/{n}"]
            label_of[img.image_id] = g["cam"]
            w2c = np.eye(4)
            w2c[:3, :4] = img.cam_from_world().matrix()
            poses.append(np.linalg.inv(w2c))
            p2d = [p for p in img.points2D if p.has_point3D()]
            xyz = np.array([rec.points3D[p.point3D_id].xyz for p in p2d]).reshape(-1, 3)
            xy = np.array([p.xy for p in p2d]).reshape(-1, 2)
            pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
            front = pc[:, 2] > 0
            err.append(np.linalg.norm(radial_project(pc[front], params) - xy[front], axis=1))
            pc = pc[front]
            obs_f.append(np.full(len(pc), t))
            obs_u.append(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2])
            obs_v.append(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2])
            obs_z.append(pc[:, 2])
        err = np.concatenate(err)
        cam_stats.update({"observations": int(len(err)), "mean_reprojection_error_px": float(err.mean()),
                          "median_reprojection_error_px": float(np.median(err)), "colmap_params": list(params)})
        results[g["cam"]] = CameraCalib(
            frames=np.array([t for t, _ in reg], dtype=np.int64), K=K, dist=np.array([k1, k2]),
            image_hw=[int(cam.height), int(cam.width)], poses_c2w=np.stack(poses),
            obs={"frame": np.concatenate(obs_f), "u": np.concatenate(obs_u), "v": np.concatenate(obs_v),
                 "z": np.concatenate(obs_z)}, stats=cam_stats)
    shared = {}
    for p in rec.points3D.values():
        cams = {label_of[el.image_id] for el in p.track.elements if el.image_id in label_of}
        if len(cams) > 1:
            key = "+".join(str(c) for c in sorted(cams))
            shared[key] = shared.get(key, 0) + 1
    stats = {"images": len(all_names), "per_camera": per_camera, "models": len(recs),
             "front_registered_per_model": [n_front[k] for k in sorted(recs)],
             "kept_model_images": int(rec.num_reg_images()), "points3D": int(rec.num_points3D()),
             "points3D_seen_by_several_cameras": shared,
             "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
             "pairs_within": len(within), "pairs_across": len(across), "principal_point_refinement": refine_stats,
             "runtime_s": {"extract": t1 - t0, "match": t2 - t1, "map": t3 - t2, "refine": t4 - t3}}
    return results, stats
