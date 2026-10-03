"""D-Z1 (docs/PLAN_AFTER_E30.md, DECISIONS Z): the fixed evaluation view range. EVALUATION CODE: reads GT.

Every --frame_stride-th frame's trained FRONT camera (CamPose refined) is moved over the grid
    lateral {-3, -1.5, 0, 1.5, 3} m  x  yaw {-45, -20, 0, 20, 45} deg  x  up {0, 2} m
(dashrecon.gen.views.ViewMove: right / up along the camera's own axes, then yaw; no pitch). Metres become scene
units with the Sim(3) scale that maps the GT FRONT camera centres onto the run's estimated ones (as
scripts/eval_cross_camera.py), so every run is evaluated over the same physical range whatever its own scale.

Hidden content may be imagined, so most views have no ground truth. Per group of views the metrics are:
    KID (primary) and FID (reported, unreliable at this size) against a reference pool of this scene's real images:
        FRONT, FRONT_LEFT and FRONT_RIGHT of every 2nd frame, undistorted with the GT calibration and resized to the
        render width (torchmetrics, Inception-v3 at 299 x 299, as scripts/eval_realism.py);
    sharpness ratio: mean variance of the Laplacian of the renders over that of the reference pool;
    flat fraction: renders whose grey std at 240 px width is below 12 (scripts/eval_realism.py);
    translucency: mean fraction of pixels whose rendered opacity is in (0.2, 0.8) (the haze in turned views:
        on val056 E30 at yaw -45 it is 9 % of the pixels, at yaw 0 or 3 m lateral below 1 %);
    semantic JS divergence of the SegFormer-B5 class histogram from the reference pool's.
Opaque but averaged texture (the smeared road and ghost cars of turned views) is not translucent; KID and the
sharpness ratio carry it. The paired FRONT_LEFT / FRONT_RIGHT comparison at the real side cameras stays in
scripts/eval_cross_camera.py.

Groups: center (no move), lateral (yaw 0, up 0), up (up 2, yaw 0), yaw20, yaw45 (any lateral / up), all.

Outputs in <log_dir>/<out_subdir>/: metrics.json (per group, per view cell, scale, commit) and, for
--example_frames, <t:03d>_up<u>.jpg mosaics (rows yaw -45..45, columns lateral -3..3).

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/eval_view_range.py --log_dir results/E30/val056/model \
        --gt_root data/waymo/processed/validation --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
        --frame_stride 10 --example_frames 50 100 150
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.novel import attach_view_gate, build_trainer, view_gate_path  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, render_moved  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from eval_cross_camera import gt_cam_to_ego, umeyama_scale  # noqa: E402
from eval_realism import Segmenter, grey_std, js_divergence, sharpness  # noqa: E402

LATERAL_M = (-3.0, -1.5, 0.0, 1.5, 3.0)
YAW_DEG = (-45.0, -20.0, 0.0, 20.0, 45.0)
UP_M = (0.0, 2.0)
REF_CAMS = (0, 1, 2)
REF_STRIDE = 2
FLAT_STD = 12.0
ALPHA_LO, ALPHA_HI = 0.2, 0.8
MOSAIC_CELL_W = 256


def group_of(lat: float, yaw: float, up: float) -> str:
    if abs(yaw) == 45.0:
        return "yaw45"
    if abs(yaw) == 20.0:
        return "yaw20"
    if up > 0:
        return "up"
    if lat != 0:
        return "lateral"
    return "center"


def inception_input(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((299, 299), Image.BICUBIC))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--gt_root", required=True, help="original processed split (GT images, calibration, poses)")
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--frame_stride", type=int, required=True)
    parser.add_argument("--example_frames", type=int, nargs="*", required=True)
    parser.add_argument("--out_subdir", default="view_range")
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    t0 = time.time()
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from datasets.driving_dataset import DrivingDataset

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    if view_gate_path(cfg) is not None:
        attach_view_gate(trainer, view_gate_path(cfg))
    trainer.set_eval()
    full = dataset.full_image_set
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    assert len(frames) == dataset.num_img_timesteps, (len(frames), dataset.num_img_timesteps)
    for t in args.example_frames:
        assert t in frames[::args.frame_stride], f"example frame {t} is not on the --frame_stride grid"
    gt_dir = os.path.join(args.gt_root, f"{int(cfg.data.scene_idx):03d}")

    # metres -> scene units: Sim(3) scale of GT FRONT centres onto the estimated ones (as eval_cross_camera.py)
    est = np.stack([full.get_image(front_image_index(dataset, i), 1)[1]["camera_to_world"].cpu().numpy().astype(np.float64)
                    for i in range(len(frames))])
    front_to_ego = gt_cam_to_ego(gt_dir, 0)
    gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{int(t):03d}.txt")) @ front_to_ego for t in frames])
    s = umeyama_scale(gt_front[:, :3, 3], est[:, :3, 3])

    # reference pool: real FRONT / FRONT_LEFT / FRONT_RIGHT, undistorted (GT calibration), at the render width
    w_render = int(full.get_image(front_image_index(dataset, 0), 1)[1]["width"])
    refs = []
    for cam in REF_CAMS:
        k_gt = np.loadtxt(os.path.join(gt_dir, "intrinsics", f"{cam}.txt"))
        K = np.array([[k_gt[0], 0, k_gt[2]], [0, k_gt[1], k_gt[3]], [0, 0, 1.0]])
        for t in frames[::REF_STRIDE]:
            img = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, "images", f"{int(t):03d}_{cam}.jpg")), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            refs.append(cv2.resize(cv2.undistort(img, K, k_gt[4:9]), (w_render, round(H * w_render / W)), interpolation=cv2.INTER_AREA))
    seg = Segmenter(args.seg_model_id, device)
    ref_hist = sum(seg.histogram(r) for r in refs)
    ref_sharp = float(np.mean([sharpness(r) for r in refs]))
    ref_inception = torch.from_numpy(np.stack([inception_input(r) for r in refs])).permute(0, 3, 1, 2)

    out_dir = os.path.join(args.log_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    cells = [(lat, yaw, up) for up in UP_M for yaw in YAW_DEG for lat in LATERAL_M]
    per_view = []
    inception, hists = {}, {}
    mosaics = {(t, up): np.zeros((len(YAW_DEG), len(LATERAL_M)), dtype=object) for t in args.example_frames for up in UP_M}
    with torch.no_grad():
        for i in range(0, len(frames), args.frame_stride):
            t = int(frames[i])
            for lat, yaw, up in cells:
                move = ViewMove(right=lat * s, up=up * s, yaw=yaw)
                out = render_moved(trainer, dataset, i, move, device)[0]
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                alpha = out["opacity"][..., 0].cpu().numpy()
                g = group_of(lat, yaw, up)
                key = f"lat{lat:+.1f}_yaw{yaw:+.0f}_up{up:.0f}"
                per_view.append({"frame": t, "cell": key, "group": g, "sharpness": sharpness(rgb), "grey_std": grey_std(rgb),
                                 "translucent": float(((alpha > ALPHA_LO) & (alpha < ALPHA_HI)).mean())})
                inception.setdefault(g, []).append(inception_input(rgb))
                hists.setdefault(g, []).append(seg.histogram(rgb))
                if t in args.example_frames:
                    mosaics[(t, up)][YAW_DEG.index(yaw), LATERAL_M.index(lat)] = rgb
    groups = {}
    for g in ["center", "lateral", "up", "yaw20", "yaw45", "all"]:
        sel = [r for r in per_view if g == "all" or r["group"] == g]
        fake = np.concatenate([np.stack(v) for k, v in inception.items() if g == "all" or k == g])
        kid = KernelInceptionDistance(subset_size=min(100, len(fake), len(refs))).to(device)
        fid = FrechetInceptionDistance().to(device)
        for b in range(0, len(refs), 64):
            kid.update(ref_inception[b:b + 64].to(device), real=True)
            fid.update(ref_inception[b:b + 64].to(device), real=True)
        for b in range(0, len(fake), 64):
            part = torch.from_numpy(fake[b:b + 64]).permute(0, 3, 1, 2).to(device)
            kid.update(part, real=False)
            fid.update(part, real=False)
        kid_mean, kid_std = kid.compute()
        hist = sum(h for k, v in hists.items() if g == "all" or k == g for h in v)
        groups[g] = {"views": len(sel), "kid": float(kid_mean), "kid_std": float(kid_std), "fid": float(fid.compute()),
                     "sharpness_ratio": float(np.mean([r["sharpness"] for r in sel]) / ref_sharp),
                     "flat_fraction": float(np.mean([r["grey_std"] < FLAT_STD for r in sel])),
                     "translucent": float(np.mean([r["translucent"] for r in sel])),
                     "semantic_js": js_divergence(hist, ref_hist)}
    for (t, up), grid in mosaics.items():
        rows = []
        for r in range(len(YAW_DEG)):
            row = [np.asarray(Image.fromarray(grid[r, c]).resize((MOSAIC_CELL_W, round(grid[r, c].shape[0] * MOSAIC_CELL_W / grid[r, c].shape[1])),
                                                              Image.BILINEAR)) for c in range(len(LATERAL_M))]
            rows.append(np.concatenate(row, 1))
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out_dir, f"{t:03d}_up{up:.0f}.jpg"), quality=88)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"reads_gt": True, "log_dir": args.log_dir, "gt_to_est_scale": s, "frame_stride": args.frame_stride,
                   "grid": {"lateral_m": LATERAL_M, "yaw_deg": YAW_DEG, "up_m": UP_M},
                   "reference_pool": {"cams": REF_CAMS, "frame_stride": REF_STRIDE, "images": len(refs)},
                   "translucent_alpha": [ALPHA_LO, ALPHA_HI], "flat_std": FLAT_STD, "seg_model_id": args.seg_model_id,
                   "groups": groups, "per_view": per_view, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[eval_view_range] {args.log_dir} (scale {s:.3f}, {len(per_view)} views, {time.time() - t0:.0f} s)", flush=True)
    for g, v in groups.items():
        print(f"[eval_view_range]   {g:8s} n={v['views']:4d} KID {v['kid'] * 1000:6.1f}e-3 FID {v['fid']:6.1f} sharp {v['sharpness_ratio']:.2f} "
              f"flat {v['flat_fraction']:.2f} transl {v['translucent']:.3f} semJS {v['semantic_js']:.3f}", flush=True)


if __name__ == "__main__":
    main()
