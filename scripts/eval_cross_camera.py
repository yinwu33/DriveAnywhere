"""Cross-camera check of a trained run (AGENTS Phase 1 task 2, simplified; DECISIONS D15). EVALUATION CODE: reads GT.

The run was trained on FRONT only. This renders the --cams cameras (1 FRONT_LEFT, 2 FRONT_RIGHT, 3 SIDE_LEFT,
4 SIDE_RIGHT) at every --frame_stride-th frame and compares them with the real images of those cameras. Renders
have the FRONT render width and the camera's own aspect ratio (SIDE: 1920 x 886 -> 960 x 443).

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
Unseen region (--region_ref, Phase 9 / DECISIONS D19): the pixels of each side view that no training view of the
reference run saw (the hole mask of dashrecon.gen.views with the render_views.py defaults, computed from the
reference run's rendered depth, not the evaluated one's, so every run is scored on the same pixels), minus GT
dynamic objects. PSNR / SSIM there use the overlap colour fit; LPIPS pastes the render over the reference on the
unseen pixels only. With the reference run itself these are the numbers generative completion has to improve.

Outputs in <log_dir>/<out_subdir>/: metrics.json (means per camera and overall, per-frame values), and
<t:03d>_<cam>.jpg (reference | colour-fitted render | overlap in grey, unseen in red) for --example_frames.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/eval_cross_camera.py --log_dir results/E5/val056 \
        --gt_root data/waymo/processed/validation --frame_stride 5 --alpha 0.5 --example_frames 50 100 150
    # Phase 9: all four side cameras, unseen region of the E5c run
    ... --log_dir results/E8/val039 --cams 1 2 3 4 --region_ref results/E5c/val039 --out_subdir cross_camera_p9
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
from dashrecon.gen.views import front_image_index, hole_mask, training_observers  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402
from datasets.base.pixel_source import get_rays  # noqa: E402

CAM_NAMES = {1: "FRONT_LEFT", 2: "FRONT_RIGHT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT"}
# hole mask parameters: scripts/render_views.py defaults
HOLE = {"depth_tol": 0.10, "res_ratio": 3.0, "sky_alpha": 0.5, "sky_elev_deg": 5.0, "open_px": 9, "min_area": 3000, "dilate_px": 7}
OBS_STRIDE = 2
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
    parser.add_argument("--cams", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--region_ref", help="log_dir of the run whose unseen region is scored (e.g. results/E5c/<scene>)")
    parser.add_argument("--out_subdir", default="cross_camera")
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    if args.region_ref is not None:
        ref_cfg = OmegaConf.load(os.path.join(args.region_ref, "config.yaml"))
        assert OmegaConf.to_container(ref_cfg.data) == OmegaConf.to_container(cfg.data), f"{args.region_ref} has other data than {args.log_dir}"
        ref_trainer = build_trainer(ref_cfg, dataset, device)
        ref_trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.region_ref, "checkpoint_final.pth"), load_only_model=True)
        observers = training_observers(ref_trainer, dataset, OBS_STRIDE, device)
    scene_idx = int(cfg.data.scene_idx)
    gt_dir = os.path.join(args.gt_root, f"{scene_idx:03d}")
    full = dataset.full_image_set
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    assert len(frames) == dataset.num_img_timesteps, (len(frames), dataset.num_img_timesteps)

    # estimated FRONT poses as trained (pose_dir), GT FRONT camera centres, and the scale between them
    est = []
    for i in range(len(frames)):
        _, ci = full.get_image(front_image_index(dataset, i), 1)
        est.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
    est = np.stack(est)
    front_to_ego = gt_cam_to_ego(gt_dir, 0)
    gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{t:03d}.txt")) @ front_to_ego for t in frames])
    s = umeyama_scale(gt_front[:, :3, 3], est[:, :3, 3])

    out_dir = os.path.join(args.log_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    per_frame = []
    sel = list(range(0, len(frames), args.frame_stride))
    for t in args.example_frames:
        assert t in frames, t
    with torch.no_grad():
        for cam in args.cams:
            cam_name = CAM_NAMES[cam]
            rel = np.linalg.inv(front_to_ego) @ gt_cam_to_ego(gt_dir, cam)
            rel[:3, 3] *= s
            k_gt = np.loadtxt(os.path.join(gt_dir, "intrinsics", f"{cam}.txt"))
            K = np.array([[k_gt[0], 0, k_gt[2]], [0, k_gt[1], k_gt[3]], [0, 0, 1.0]])
            dist = k_gt[4:9]
            for i in sel:
                t = int(frames[i])
                ii, ci = full.get_image(front_image_index(dataset, i), 1)
                ii, ci = to_device(ii, device), to_device(ci, device)
                ref_full = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, "images", f"{t:03d}_{cam}.jpg")), cv2.COLOR_BGR2RGB)
                H, W = ref_full.shape[:2]
                w = int(ci["width"])
                h = int(round(H * w / W))
                # the Affine model is per pixel (models/modules.py AffineTransform), so its index map takes the render shape
                ii["img_idx"] = torch.full((h, w), int(ii["img_idx"].flatten()[0]), dtype=ii["img_idx"].dtype, device=device)
                ci["height"], ci["width"] = torch.tensor(h, dtype=torch.long, device=device), torch.tensor(w, dtype=torch.long, device=device)
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
                if args.region_ref is not None:
                    ref_out = ref_trainer(ii, ci, novel_view=True)
                    unseen = hole_mask(ref_out, ii["viewdirs"], k_r, c2w, observers, **HOLE)[0] & ~dyn
                render = out["rgb"].clamp(0, 1).cpu().numpy()
                overlap = (out["opacity"][..., 0].cpu().numpy() > args.alpha) & ~dyn
                row = {"frame": t, "cam": cam_name, "coverage": float(overlap.mean())}
                if args.region_ref is not None:
                    row["unseen_frac"] = float(unseen.mean())
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
                    if args.region_ref is not None and unseen.sum() > 1000:
                        row.update({
                            "unseen_psnr": masked_psnr(fitted, ref, unseen), "unseen_ssim": float(ssim_map[unseen].mean()),
                            "unseen_lpips": float(trainer.lpips(torch.from_numpy(np.where(unseen[..., None], fitted, ref)).permute(2, 0, 1)[None].to(device), ref_t)),
                        })
                    if t in args.example_frames:
                        panel = np.repeat(overlap[..., None], 3, -1).astype(np.float32) * 0.6
                        if args.region_ref is not None:
                            panel[unseen] = (0.9, 0.2, 0.2)
                        vis = np.concatenate([ref, fitted, panel], 1)
                        Image.fromarray((vis * 255).round().astype(np.uint8)).save(os.path.join(out_dir, f"{t:03d}_{cam}.jpg"), quality=88)
                per_frame.append(row)
    keys = ("psnr", "psnr_raw", "ssim", "lpips", "lpips_raw")
    unseen_keys = ("unseen_psnr", "unseen_ssim", "unseen_lpips")
    summary = {}
    for cam_name in [CAM_NAMES[c] for c in args.cams] + ["all"]:
        rows = [r for r in per_frame if (cam_name == "all" or r["cam"] == cam_name)]
        scored = [r for r in rows if "psnr" in r]
        summary[cam_name] = {"frames": len(rows), "scored": len(scored), "coverage": float(np.mean([r["coverage"] for r in rows])),
                             **{k: float(np.mean([r[k] for r in scored])) for k in keys}}
        if args.region_ref is not None:
            unseen_scored = [r for r in rows if "unseen_psnr" in r]
            summary[cam_name].update({"unseen_frac": float(np.mean([r["unseen_frac"] for r in rows])), "unseen_scored": len(unseen_scored),
                                      **{k: float(np.mean([r[k] for r in unseen_scored])) for k in unseen_keys}})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"reads_gt": True, "log_dir": args.log_dir, "scene_idx": scene_idx, "frame_stride": args.frame_stride,
                   "alpha": args.alpha, "gt_to_est_scale": s, "cams": [CAM_NAMES[c] for c in args.cams],
                   "region_ref": args.region_ref, "hole_params": HOLE, "obs_stride": OBS_STRIDE,
                   "placement": "per-frame: estimated FRONT pose x GT FRONT->side extrinsics (translation scaled by the Sim3 scale)",
                   "summary": summary, "per_frame": per_frame, "dashrecon_commit": commit}, f, indent=2)
    a = summary["all"]
    print(f"[eval_cross_camera] {args.log_dir}: coverage {a['coverage']:.2f}, PSNR {a['psnr']:.2f} (raw {a['psnr_raw']:.2f}), "
          f"SSIM {a['ssim']:.3f}, LPIPS {a['lpips']:.3f} over {a['scored']} images", flush=True)
    if args.region_ref is not None:
        for name, v in summary.items():
            print(f"[eval_cross_camera]   {name}: unseen {v['unseen_frac']:.3f} of the view, PSNR {v['unseen_psnr']:.2f}, "
                  f"SSIM {v['unseen_ssim']:.3f}, LPIPS {v['unseen_lpips']:.3f} over {v['unseen_scored']} images", flush=True)


if __name__ == "__main__":
    main()
