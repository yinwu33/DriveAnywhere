"""MT1 (docs/EXPERIMENTS.md): render a run at the FRONT cameras of one traversal and compare with its real images.

In a combined multi-traversal scene (scripts/combine_traversals.py) every traversal is real ground truth for views
the other traversals never had: another lane, a few metres off, on another day. This renders the run at the held-out
FRONT frames of --member (virtual frames in the member's range that dashrecon.scenes.train_frame_mask holds out),
whether or not the run trained on that member, so a run trained on one traversal and a run trained on all of them
are scored on the same frames.

    camera   the frame's estimated pose and intrinsics in the combined pose dir (no CamPose refinement: drivestudio's
             test frames are rendered the same way, DECISIONS D4); render size = the run's training size;
    template image / camera infos of the run's own training frame whose camera centre is nearest (render_at
             replaces rays, size and intrinsics; the Affine index of that frame stays);
    image    the member's undistorted FRONT image (the combined scene's image link), resized to the render size;
    pixels   static = not the frame's dynamic mask (Grounded-SAM-2, as in training); sky excluded for the *_nosky values.
Metrics: a per-channel affine colour fit of the render to the image over the static pixels (another day, other light),
then PSNR and SSIM over the static pixels, PSNR over static non-sky pixels, LPIPS (trainer.lpips) of the fitted render
pasted over the image on the dynamic pixels; raw PSNR without the fit; and the distance from the frame's camera to the
nearest training camera of the run (scene units), the size of the viewpoint change.
No GT pose, calibration or LiDAR: everything comes from the FRONT-only products of the combined scene.

Outputs <log_dir>/<out_subdir>/metrics.json and <t:03d>.jpg (image | colour-fitted render | raw render) for
--example_frames.

Example (main venv):
    .venvs/main/bin/python scripts/eval_cross_traversal.py --log_dir results/MT1-A/mt1 --member mt1b \
        --example_frames 250 300 350 --out_subdir cross_traversal_mt1b
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon import io  # noqa: E402
from dashrecon.gen.novel import attach_view_gate, build_trainer, to_device, view_gate_path  # noqa: E402
from dashrecon.gen.views import front_image_index, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, train_frame_mask  # noqa: E402
from eval_cross_camera import colour_fit, masked_psnr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--example_frames", type=int, nargs="*", required=True)
    parser.add_argument("--out_subdir", required=True)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    if view_gate_path(cfg) is not None:
        attach_view_gate(trainer, view_gate_path(cfg))
    trainer.set_eval()

    pose_dir, mask_dir = cfg.data.pixel_source.pose_dir, cfg.data.pixel_source.mask_dir
    meta = io.read_meta(pose_dir)
    v0, v1 = meta["traversals"][args.member]["virtual_frames"]
    all_frames = io.read_frames(pose_dir)
    test_stride = int(cfg.data.pixel_source.test_image_stride)
    # held out within the combined scene's own split (start 0), whatever range the run trained on
    held = all_frames[~train_frame_mask(all_frames, 0, test_stride)]
    frames = held[(held >= v0) & (held < v1)]
    for t in args.example_frames:
        assert t in frames, f"{t} is not a held-out frame of {args.member}: {frames.tolist()}"
    poses, ks = io.read_poses(pose_dir), io.read_intrinsics(pose_dir)
    img_dir = os.path.join(cfg.data.data_root, f"{int(cfg.data.scene_idx):03d}", "images")

    assert dataset.start_timestep == 0, "the run's split must count from the combined scene's frame 0"
    run_frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    train_k = np.nonzero(train_frame_mask(run_frames, 0, test_stride))[0].tolist()
    train_centres = poses[run_frames[train_k], :3, 3]
    out_dir = os.path.join(args.log_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=False)
    rows = []
    with torch.no_grad():
        for t in frames:
            t = int(t)
            dist = np.linalg.norm(train_centres - poses[t, :3, 3], axis=1)
            k_near = train_k[int(dist.argmin())]
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, k_near), 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            h, w = int(ci["height"]), int(ci["width"])
            img_full = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"{t:03d}_{FRONT_CAM_ID}.jpg")), cv2.COLOR_BGR2RGB)
            H, W = img_full.shape[:2]
            k = ks[t]
            k_r = torch.tensor([[k[0, 0] * w / W, 0, (k[0, 2] + 0.5) * w / W], [0, k[1, 1] * h / H, (k[1, 2] + 0.5) * h / H],
                                [0, 0, 1.0]], dtype=torch.float32, device=device)
            out = render_at(trainer, ii, ci, torch.from_numpy(poses[t]).float().to(device), k_r, (h, w))
            render = out["rgb"].clamp(0, 1).cpu().numpy()
            ref = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            dyn = np.asarray(Image.fromarray(io.read_mask(mask_dir, "dynamic", t)).resize((w, h), Image.NEAREST))
            sky = np.asarray(Image.fromarray(io.read_mask(mask_dir, "sky", t)).resize((w, h), Image.NEAREST))
            static = ~dyn
            fitted = colour_fit(render, ref, static)
            ssim_map = structural_similarity(fitted, ref, data_range=1.0, channel_axis=-1, full=True)[1].mean(-1)
            ref_t = torch.from_numpy(ref).permute(2, 0, 1)[None].to(device)
            pasted = torch.from_numpy(np.where(static[..., None], fitted, ref)).permute(2, 0, 1)[None].to(device)
            rows.append({"frame": t, "nearest_train_frame": int(run_frames[k_near]), "nearest_train_dist": float(dist.min()),
                         "psnr": masked_psnr(fitted, ref, static), "psnr_raw": masked_psnr(render, ref, static),
                         "psnr_nosky": masked_psnr(fitted, ref, static & ~sky), "ssim": float(ssim_map[static].mean()),
                         "lpips": float(trainer.lpips(pasted, ref_t))})
            if t in args.example_frames:
                vis = np.concatenate([ref, fitted, render], 1)
                Image.fromarray((vis * 255).round().astype(np.uint8)).save(os.path.join(out_dir, f"{t:03d}.jpg"), quality=88)
    keys = ("psnr", "psnr_raw", "psnr_nosky", "ssim", "lpips", "nearest_train_dist")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"log_dir": args.log_dir, "member": args.member, "frames": len(rows), "test_stride": test_stride,
                   "run_frames": [int(run_frames[0]), int(run_frames[-1]) + 1], "resolution": [h, w],
                   "reads_gt": False, "summary": summary, "per_frame": rows, "dashrecon_commit": commit}, f, indent=2)
    print(f"[eval_cross_traversal] {args.log_dir} at {args.member}'s {len(rows)} held-out frames: PSNR {summary['psnr']:.2f} "
          f"(raw {summary['psnr_raw']:.2f}, non-sky {summary['psnr_nosky']:.2f}), SSIM {summary['ssim']:.3f}, "
          f"LPIPS {summary['lpips']:.3f}; nearest training camera {summary['nearest_train_dist']:.2f} away", flush=True)


if __name__ == "__main__":
    main()
