"""E34 (docs/E34_DATA.md): how close the joint self-calibration is to Waymo's. EVALUATION CODE: reads GT.

Compares a scripts/e34_calib.py result with the GT of the segment (<gt_root>/<idx:03d>/{intrinsics,extrinsics,ego_pose},
put aside by scripts/e34_fetch.py). Nothing here feeds back into the pairs or the acceptance decision; the numbers
only describe the data quality.
    intrinsics   per camera: estimated focal / GT focal (mean of fx, fy), k1, k2 next to GT k1, k2;
    trajectory   Sim(3) (Umeyama) of the estimated FRONT camera centres onto the GT ones (GT camera pose =
                 ego_pose[t] @ extrinsics[cam] @ OPENCV2DATASET, as scripts/eval_cross_camera.py): scale (GT m per SfM
                 unit), RMS position error in m;
    targets      every registered target image mapped by the same Sim(3): position error (m) and rotation error (deg)
                 against its GT pose. The GT pose uses the ego pose of the frame, not of the camera's own exposure (the
                 cameras are not triggered together), so part of the error is GT's;
    rig          per frame, estimated FRONT -> target relative rotation against the fixed GT one (deg).
Writes --out (JSON) and prints a summary line.

Example (sfm venv):
    .venvs/sfm/bin/python scripts/e34_diag_calib.py --calib_dir data/e34/work/wt646/calib-glomap-cams \
        --gt_root data/e34/gt --scene_idx 646 --out data/e34/diag/wt646_calib_gt.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES, FRONT, cam_dir  # noqa: E402

OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = float(np.trace(np.diag(d) @ sign) / ((xs ** 2).sum() / len(src)))
    return s, r, mu_d - s * r @ mu_s


def rot_deg(r: np.ndarray) -> np.ndarray:
    """Rotation angles (deg) of (...,3,3) rotation matrices."""
    c = (np.trace(r, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--scene_idx", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = io.read_meta(args.calib_dir)
    assert meta["accepted"], meta["reasons"]
    gt = os.path.join(args.gt_root, f"{args.scene_idx:03d}")
    a = meta["frame_offset"]
    cam_to_ego = {c: np.loadtxt(os.path.join(gt, "extrinsics", f"{c}.txt")) @ OPENCV2DATASET for c in CAM_NAMES}
    ego = lambda t: np.loadtxt(os.path.join(gt, "ego_pose", f"{t:03d}.txt"))  # noqa: E731
    gt_pose = lambda c, t: ego(t) @ cam_to_ego[c]  # noqa: E731

    est = {FRONT: (io.read_frames(args.calib_dir) + a, io.read_poses(args.calib_dir), io.read_camera(args.calib_dir))}
    for c_str in meta["cameras"]:
        if int(c_str) != FRONT:
            d = cam_dir(args.calib_dir, int(c_str))
            est[int(c_str)] = (io.read_frames(d), io.read_poses(d), io.read_camera(d))
    f_frames, f_poses, _ = est[FRONT]
    g_front = np.stack([gt_pose(FRONT, int(t)) for t in f_frames])
    s, r, t_ = umeyama(f_poses[:, :3, 3], g_front[:, :3, 3])
    front_err = np.linalg.norm((s * f_poses[:, :3, 3] @ r.T + t_) - g_front[:, :3, 3], axis=1)
    out = {"reads_gt": True, "calib_dir": os.path.relpath(args.calib_dir), "scene_idx": args.scene_idx,
           "sim3_scale_gt_m_per_unit": s, "front_rms_m": float(np.sqrt((front_err ** 2).mean())),
           "front_path_m": float(np.linalg.norm(np.diff(g_front[:, :3, 3], axis=0), axis=1).sum()), "cameras": {}}
    front_by_t = dict(zip((int(t) for t in f_frames), f_poses))
    for c, (frames, poses, camera) in est.items():
        k_gt = np.loadtxt(os.path.join(gt, "intrinsics", f"{c}.txt"))
        row = {"name": CAM_NAMES[c], "focal_ratio": float(camera["K"][0, 0] / ((k_gt[0] + k_gt[1]) / 2)),
               "k1_k2": camera["dist"].tolist(), "gt_k1_k2": [float(k_gt[4]), float(k_gt[5])],
               "registered": int(len(frames))}
        if c != FRONT:
            g = np.stack([gt_pose(c, int(t)) for t in frames])
            pos = s * poses[:, :3, 3] @ r.T + t_
            rot = r @ poses[:, :3, :3]
            pos_err = np.linalg.norm(pos - g[:, :3, 3], axis=1)
            rot_err = rot_deg(np.transpose(g[:, :3, :3], (0, 2, 1)) @ rot)
            rel_gt = np.linalg.inv(cam_to_ego[FRONT]) @ cam_to_ego[c]
            both = [i for i, t in enumerate(frames) if int(t) in front_by_t]
            rel_est = np.stack([np.linalg.inv(front_by_t[int(frames[i])])[:3, :3] @ poses[i, :3, :3] for i in both])
            rig_err = rot_deg(rel_gt[:3, :3].T[None] @ rel_est)
            row.update({"pos_err_m_median": float(np.median(pos_err)), "pos_err_m_p90": float(np.percentile(pos_err, 90)),
                        "rot_err_deg_median": float(np.median(rot_err)), "rot_err_deg_p90": float(np.percentile(rot_err, 90)),
                        "rig_rot_err_deg_median": float(np.median(rig_err)),
                        "rig_rot_err_deg_p90": float(np.percentile(rig_err, 90))})
        out["cameras"][str(c)] = row
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[e34_diag_calib] {args.scene_idx:03d}: scale {s:.2f} m/unit, FRONT rms {out['front_rms_m']:.2f} m over "
          f"{out['front_path_m']:.0f} m; " + "; ".join(
              f"{v['name']} f {v['focal_ratio']:.3f}" + (f" pos {v['pos_err_m_median']:.2f} m rot {v['rot_err_deg_median']:.2f} deg "
                                                          f"rig {v['rig_rot_err_deg_median']:.2f} deg" if "pos_err_m_median" in v else "")
              for v in out["cameras"].values()), flush=True)


if __name__ == "__main__":
    main()
