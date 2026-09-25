"""Cross-camera check of a trained run (AGENTS Phase 1 task 2, simplified; DECISIONS D15). EVALUATION CODE: reads GT.

The run was trained on FRONT only. This renders FRONT_LEFT (camera 1) and FRONT_RIGHT (camera 2) at every
--frame_stride-th frame and compares them with the real images of those cameras.

Placement (per frame, relative to the run's own camera): the side camera is attached to the estimated FRONT pose
the model was trained with at that frame (before CamPose refinement),
    c2w_side = c2w_front_est @ [R_rel | s * t_rel],
where R_rel, t_rel is the GT FRONT -> side transform from the Waymo extrinsics and s the Sim(3) scale that maps
the GT FRONT camera centres onto the estimated ones (Umeyama over all frames). This measures how the scene looks
from where the side camera really was relative to the car, independent of global trajectory drift.
Intrinsics are the GT side intrinsics; the GT side images are undistorted with the GT distortion, so render and
reference are both pinhole. Rays (and so the sky model) are recomputed for the side camera.

Metrics on the overlap = rendered Gaussian opacity > --alpha and not a GT dynamic object (GT box masks of the
processed split): a per-channel affine colour fit of the render to the reference on the overlap (the run's
exposure model only knows FRONT), then PSNR and SSIM (mean of the SSIM map) over the overlap, and LPIPS (alex,
as drivestudio) of the render pasted over the reference outside the overlap; raw (no colour fit) values too.
Outputs in <log_dir>/cross_camera/: metrics.json (means per camera and overall, per-frame values), and
<t:03d>_<cam>.jpg (reference | colour-fitted render | overlap) for --example_frames.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/eval_cross_camera.py --log_dir results/E5/val056 \
        --gt_root data/waymo/processed/validation --frame_stride 5 --alpha 0.5 --example_frames 50 100 150
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from datasets.base.pixel_source import get_rays  # noqa: E402

SIDE_CAMS = {1: "FRONT_LEFT", 2: "FRONT_RIGHT"}
# Waymo camera frame (x forward, y left, z up) -> OpenCV, as datasets/waymo/waymo_sourceloader.py
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def umeyama_scale(src: np.ndarray, dst: np.ndarray) -> float:
    """Scale of the Sim(3) that maps (N,3) src onto dst."""
    xs, xd = src - src.mean(0), dst - dst.mean(0)
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    return float(np.trace(np.diag(d) @ sign) / ((xs ** 2).sum() / len(src)))


def gt_cam_to_ego(gt_dir: str, cam: int) -> np.ndarray:
    return np.loadtxt(os.path.join(gt_dir, "extrinsics", f"{cam}.txt")) @ OPENCV2DATASET


def colour_fit(render: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel least-squares a * render + b fitted to ref on mask."""
    out = np.empty_like(render)
    for c in range(3):
        a, b = np.linalg.lstsq(np.stack([render[mask, c], np.ones(mask.sum())], 1), ref[mask, c], rcond=None)[0]
        out[..., c] = a * render[..., c] + b
    return np.clip(out, 0.0, 1.0)


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(-10.0 * np.log10(((a[mask] - b[mask]) ** 2).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--gt_root", required=True, help="original processed split (GT images, calibration, poses)")
    parser.add_argument("--frame_stride", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--example_frames", type=int, nargs="*", required=True)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    scene_idx = int(cfg.data.scene_idx)
    gt_dir = os.path.join(args.gt_root, f"{scene_idx:03d}")
    full = dataset.full_image_set
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    assert len(frames) == len(full), (len(frames), len(full))

    # estimated FRONT poses as trained (pose_dir), GT FRONT camera centres, and the scale between them
    est = []
    for i in range(len(frames)):
        _, ci = full.get_image(i, 1)
        est.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
    est = np.stack(est)
    front_to_ego = gt_cam_to_ego(gt_dir, 0)
    gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{t:03d}.txt")) @ front_to_ego for t in frames])
    s = umeyama_scale(gt_front[:, :3, 3], est[:, :3, 3])

    out_dir = os.path.join(args.log_dir, "cross_camera")
    os.makedirs(out_dir, exist_ok=True)
    per_frame = []
    sel = list(range(0, len(frames), args.frame_stride))
    for t in args.example_frames:
        assert t in frames, t
    with torch.no_grad():
        for cam, cam_name in SIDE_CAMS.items():
            rel = np.linalg.inv(front_to_ego) @ gt_cam_to_ego(gt_dir, cam)
            rel[:3, 3] *= s
            k_gt = np.loadtxt(os.path.join(gt_dir, "intrinsics", f"{cam}.txt"))
            K = np.array([[k_gt[0], 0, k_gt[2]], [0, k_gt[1], k_gt[3]], [0, 0, 1.0]])
            dist = k_gt[4:9]
            for i in sel:
                t = int(frames[i])
                ii, ci = full.get_image(i, 1)
                ii, ci = to_device(ii, device), to_device(ci, device)
                h, w = int(ci["height"]), int(ci["width"])
                ref_full = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, "images", f"{t:03d}_{cam}.jpg")), cv2.COLOR_BGR2RGB)
                H, W = ref_full.shape[:2]
                ref = cv2.resize(cv2.undistort(ref_full, K, dist), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                dyn = np.asarray(Image.open(os.path.join(gt_dir, "dynamic_masks", "all", f"{t:03d}_{cam}.png")).resize((w, h), Image.NEAREST)) > 0
                # intrinsics on the render grid, +0.5 pixel-centre convention of drivestudio / gsplat
                k_r = torch.tensor([[K[0, 0] * w / W, 0, (K[0, 2] + 0.5) * w / W], [0, K[1, 1] * h / H, (K[1, 2] + 0.5) * h / H],
                                    [0, 0, 1.0]], dtype=torch.float32, device=device)
                c2w = torch.from_numpy(est[i] @ rel).float().to(device)
                x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="xy")
                origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, k_r)
                ii["origins"], ii["viewdirs"] = origins.reshape(h, w, 3), viewdirs.reshape(h, w, 3)
                ii["direction_norm"] = direction_norm.reshape(h, w, 1)
                ci["camera_to_world"], ci["intrinsics"] = c2w, k_r
                out = trainer(ii, ci, novel_view=True)
                render = out["rgb"].clamp(0, 1).cpu().numpy()
                overlap = (out["opacity"][..., 0].cpu().numpy() > args.alpha) & ~dyn
                row = {"frame": t, "cam": cam_name, "coverage": float(overlap.mean())}
                if overlap.sum() > 1000:
                    fitted = colour_fit(render, ref, overlap)
                    ssim_map = structural_similarity(fitted, ref, data_range=1.0, channel_axis=-1, full=True)[1].mean(-1)
                    paste = lambda img: torch.from_numpy(np.where(overlap[..., None], img, ref)).permute(2, 0, 1)[None].to(device)
                    ref_t = torch.from_numpy(ref).permute(2, 0, 1)[None].to(device)
                    row.update({
                        "psnr": masked_psnr(fitted, ref, overlap), "psnr_raw": masked_psnr(render, ref, overlap),
                        "ssim": float(ssim_map[overlap].mean()),
                        "lpips": float(trainer.lpips(paste(fitted), ref_t)), "lpips_raw": float(trainer.lpips(paste(render), ref_t)),
                    })
                    if t in args.example_frames:
                        vis = np.concatenate([ref, fitted, np.repeat(overlap[..., None], 3, -1).astype(np.float32)], 1)
                        Image.fromarray((vis * 255).round().astype(np.uint8)).save(os.path.join(out_dir, f"{t:03d}_{cam}.jpg"), quality=88)
                per_frame.append(row)
    keys = ("psnr", "psnr_raw", "ssim", "lpips", "lpips_raw")
    summary = {}
    for cam_name in list(SIDE_CAMS.values()) + ["all"]:
        rows = [r for r in per_frame if (cam_name == "all" or r["cam"] == cam_name)]
        scored = [r for r in rows if "psnr" in r]
        summary[cam_name] = {"frames": len(rows), "scored": len(scored), "coverage": float(np.mean([r["coverage"] for r in rows])),
                             **{k: float(np.mean([r[k] for r in scored])) for k in keys}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"reads_gt": True, "log_dir": args.log_dir, "scene_idx": scene_idx, "frame_stride": args.frame_stride,
                   "alpha": args.alpha, "gt_to_est_scale": s,
                   "placement": "per-frame: estimated FRONT pose x GT FRONT->side extrinsics (translation scaled by the Sim3 scale)",
                   "summary": summary, "per_frame": per_frame, "dashrecon_commit": commit}, f, indent=2)
    a = summary["all"]
    print(f"[eval_cross_camera] {args.log_dir}: coverage {a['coverage']:.2f}, PSNR {a['psnr']:.2f} (raw {a['psnr_raw']:.2f}), "
          f"SSIM {a['ssim']:.3f}, LPIPS {a['lpips']:.3f} over {a['scored']} images", flush=True)


if __name__ == "__main__":
    main()
