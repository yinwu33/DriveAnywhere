"""E34 (docs/E34_DATA.md): undistort an accepted segment with its joint self-calibration (scripts/e34_calib.py).

FRONT (the only input of the reconstruction):
    <front_root>/<idx:03d>/images/<v:03d>_0.jpg   frames [a, b) of the calibration, renumbered v = t - a, undistorted
                                                    with the same K (as scripts/run_calib.py: <= 2 px outside, edge
                                                    pixels replicated); nothing else is in this tree
    <front_mask_dir>/                               the Phase 4 masks of the original FRONT images warped through the
                                                    same undistortion map (nearest), virtual frames: the
                                                    mask-..._img-<tag> directory of scripts/run_fusion.py and training
Warping the masks replaces re-segmenting the undistorted images (DECISIONS L did that); the undistortion is a pure
resampling, so the warped mask is the mask of the undistorted image up to nearest-neighbour rounding.

Targets (every registered target-camera image inside [a, b); never read by the reconstruction):
    <pairs_dir>/target/<t:03d>_<cam>.jpg   real image undistorted with that camera's own estimated K and distortion,
                                           then resized to --pair_hw (INTER_AREA; Fixer's resize_576x1024 grid, the
                                           aspect ratio is not kept), JPEG quality 95
    <pairs_dir>/mask/<t:03d>_<cam>.png     255 = leave out of the training loss: the camera's Phase 4 dynamic mask
                                           (objects the FRONT-only input cannot contain) or pixels whose undistortion
                                           samples outside the original image; a pair pixel is masked if any source
                                           pixel under it is
    <pairs_dir>/targets.json               per target: camera, timestep, files, masked fraction; per camera: K on the
                                           pair grid (integer pixel centres), undistortion range

Example (sfm venv):
    .venvs/sfm/bin/python scripts/e34_undistort.py --scene_id wt646 --raw_root data/e34/raw \
        --mask_dir data/e34/work/wt646/masks_orig --calib_dir data/e34/work/wt646/calib-glomap-cams \
        --front_root data/e34/front --front_mask_dir data/e34/work/wt646/mask-gsam2_sky-segformer_img-glomap-cams \
        --pairs_dir data/e34/pairs/wt646 --pair_hw 576 1024
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES, FRONT, cam_dir, raw_image  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from run_calib import EDGE_TOLERANCE_PX  # noqa: E402

JPEG_QUALITY = 95


def undistort_maps(camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undistortion maps (same K) of a camera.json and the (H,W) bool map of pixels that sample outside the image."""
    h, w = camera["image_hw"]
    dist5 = np.array([camera["dist"][0], camera["dist"][1], 0.0, 0.0, 0.0])
    map_x, map_y = cv2.initUndistortRectifyMap(camera["K"], dist5, None, camera["K"], (w, h), cv2.CV_32FC1)
    outside = (map_x < -0.5) | (map_x > w - 0.5) | (map_y < -0.5) | (map_y > h - 0.5)
    return map_x, map_y, outside


def pair_intrinsics(K: np.ndarray, image_hw: list, pair_hw: list) -> np.ndarray:
    """Intrinsics (integer pixel centres) of an image resized from image_hw to pair_hw."""
    sy, sx = pair_hw[0] / image_hw[0], pair_hw[1] / image_hw[1]
    out = np.array(K, dtype=np.float64)
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (out[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (out[1, 2] + 0.5) * sy - 0.5
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--mask_dir", required=True, help="scripts/e34_masks.py output (original images)")
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--front_root", required=True)
    parser.add_argument("--front_mask_dir", required=True)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--pair_hw", type=int, nargs=2, required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    scene = get_scene(args.scene_id)
    meta = io.read_meta(args.calib_dir)
    assert meta["accepted"], f"{args.calib_dir}: segment was not accepted: {meta['reasons']}"
    a, b = meta["front_range"]
    out_img_dir = os.path.join(args.front_root, f"{scene.scene_idx:03d}", "images")
    assert os.path.relpath(out_img_dir) == meta["undistorted_images"], (out_img_dir, meta["undistorted_images"])
    for d in (out_img_dir, args.front_mask_dir, args.pairs_dir):
        assert not os.path.exists(d), f"{d} exists"

    # FRONT tree and its masks
    camera = io.read_camera(args.calib_dir)
    map_x, map_y, outside = undistort_maps(camera)
    out_px = float(max(-0.5 - map_x.min(), map_x.max() - (camera["image_hw"][1] - 0.5), -0.5 - map_y.min(),
                       map_y.max() - (camera["image_hw"][0] - 0.5), 0.0))
    assert out_px <= EDGE_TOLERANCE_PX, f"FRONT undistortion samples {out_px:.1f} px outside the image"
    virtual = io.read_frames(args.calib_dir)
    np.testing.assert_array_equal(virtual, np.arange(b - a))
    front_masks = cam_dir(args.mask_dir, FRONT)
    os.makedirs(out_img_dir)
    ratios = []
    for v in virtual:
        t = int(v) + a
        img = cv2.imread(raw_image(args.raw_root, scene.scene_idx, t, FRONT))
        assert list(img.shape[:2]) == camera["image_hw"], (img.shape, camera["image_hw"])
        cv2.imwrite(os.path.join(out_img_dir, f"{int(v):03d}_{FRONT}.jpg"),
                    cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        row = {"frame": int(v)}
        for kind in io.MASK_KINDS:
            m = io.read_mask(front_masks, kind, t).astype(np.uint8)
            warped = cv2.remap(m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
            io.write_mask(args.front_mask_dir, kind, int(v), warped)
            row[kind] = float(warped.mean())
        ratios.append(row)
    io.write_frames(args.front_mask_dir, virtual)
    io.write_meta(args.front_mask_dir, {
        "scene_id": args.scene_id, "segment": scene.segment, "image_hw": camera["image_hw"],
        "source": "Phase 4 masks of the original FRONT images (scripts/e34_masks.py) warped (nearest) through the "
                  "undistortion map of " + os.path.relpath(args.calib_dir),
        "source_meta": io.read_meta(args.mask_dir)["dynamic"], "frame_offset": a,
        "summary": {k: float(np.mean([r[k] for r in ratios])) for k in io.MASK_KINDS},
        "uses_oracle": False, "uses_calibration": False, "dashrecon_commit": commit, "per_frame": ratios,
    })

    # targets
    os.makedirs(os.path.join(args.pairs_dir, "target"))
    os.makedirs(os.path.join(args.pairs_dir, "mask"))
    ph, pw = args.pair_hw
    rows, cams = [], {}
    for c_str in sorted(meta["cameras"]):
        cam = int(c_str)
        if cam == FRONT:
            continue
        cdir = cam_dir(args.calib_dir, cam)
        camera = io.read_camera(cdir)
        map_x, map_y, outside = undistort_maps(camera)
        frames = io.read_frames(cdir)
        frames = frames[(frames >= a) & (frames < b)]
        cams[c_str] = {"name": CAM_NAMES[cam], "K_pair": pair_intrinsics(camera["K"], camera["image_hw"], args.pair_hw).tolist(),
                       "image_hw": camera["image_hw"], "outside_frac": float(outside.mean()), "num_targets": int(len(frames))}
        for t in frames:
            t = int(t)
            img = cv2.imread(raw_image(args.raw_root, scene.scene_idx, t, cam))
            assert list(img.shape[:2]) == camera["image_hw"], (img.shape, camera["image_hw"])
            und = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dyn = io.read_mask(cam_dir(args.mask_dir, cam), "dynamic", t).astype(np.uint8)
            excl = (cv2.remap(dyn, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0) | outside
            small = cv2.resize(und, (pw, ph), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(excl.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) > 0
            name = f"{t:03d}_{cam}"
            cv2.imwrite(os.path.join(args.pairs_dir, "target", f"{name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(os.path.join(args.pairs_dir, "mask", f"{name}.png"), small_mask.astype(np.uint8) * 255)
            rows.append({"cam": cam, "camera": CAM_NAMES[cam], "t": t, "target": f"target/{name}.jpg",
                         "mask": f"mask/{name}.png", "masked_frac": float(small_mask.mean())})
    with open(os.path.join(args.pairs_dir, "targets.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, "pair_hw": args.pair_hw, "front_range": [a, b], "cameras": cams,
                   "jpeg_quality": JPEG_QUALITY, "calib_dir": os.path.relpath(args.calib_dir),
                   "mask_meaning": "255 = excluded from the loss (dynamic object or outside the original image)",
                   "dashrecon_commit": commit, "runtime_s": time.time() - t0, "targets": rows}, f, indent=2)
    print(f"[e34_undistort] {args.scene_id}: FRONT {b - a} frames -> {out_img_dir}; "
          + ", ".join(f"{v['name']} {v['num_targets']}" for v in cams.values())
          + f" targets -> {args.pairs_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
