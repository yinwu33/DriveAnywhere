"""E34 (docs/E34_DATA.md): "gs" pairs, Fixer's cross reference: the FRONT-only E5f 3DGS rendered at the target cameras.

Loads <log_dir> (scripts/e34_train_gs.py: checkpoint_final.pth, config.yaml) as scripts/eval_cross_camera.py does and
renders every target of scripts/e34_undistort.py (camera c, timestep t) on the pair grid:
    - camera: c's own estimated K on the pair grid (targets.json; +0.5 for drivestudio / gsplat) and its
      joint-calibration pose in the reconstruction world (dashrecon.e34.geometry), no CamPose refinement;
    - frame: the image / camera infos of FRONT frame t (virtual t - frame_offset) for the per-image models (Affine)
      and time; rays, and so the sky model, are recomputed for c (dashrecon.gen.views.render_at).
Reads no image of the target cameras. Writes <pairs_dir>/gs/<t:03d>_<cam>.jpg (JPEG 95) and <pairs_dir>/gs.json
(per-image coverage = rendered Gaussian opacity > 0.5, runtime).

Example (main venv):
    PATH=$PWD/.venvs/main/bin:$HOME/cuda-12.6/bin:$PATH CUDA_HOME=$HOME/cuda-12.6 \
        .venvs/main/bin/python scripts/e34_gs_pairs.py --scene_id wt646 --log_dir data/e34/work/wt646/gs/E5f/wt646 \
        --calib_dir data/e34/work/wt646/calib-glomap-cams --pairs_dir data/e34/pairs/wt646
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.geometry import check_front_consistency, target_world_poses  # noqa: E402
from dashrecon.gen.novel import build_trainer, to_device  # noqa: E402
from dashrecon.gen.views import front_image_index, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

JPEG_QUALITY = 95
ALPHA = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--pairs_dir", required=True)
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    out_dir = os.path.join(args.pairs_dir, "gs")
    assert not os.path.exists(out_dir), f"{out_dir} exists"
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    pose_dir = cfg.data.pixel_source.pose_dir
    check_front_consistency(args.calib_dir, pose_dir)
    a = io.read_meta(args.calib_dir)["frame_offset"]
    targets = io.read_json(os.path.join(args.pairs_dir, "targets.json"))
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    assert dataset.start_timestep == 0, dataset.start_timestep
    poses = {}
    for c_str in targets["cameras"]:
        frames, c2w = target_world_poses(args.calib_dir, os.path.relpath(pose_dir), int(c_str))
        poses[int(c_str)] = dict(zip((int(t) for t in frames), c2w))
    t1 = time.time()
    os.makedirs(out_dir)
    h, w = targets["pair_hw"]
    rows = []
    with torch.no_grad():
        for tg in targets["targets"]:
            cam, t = tg["cam"], tg["t"]
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, t - a), 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            K = np.asarray(targets["cameras"][str(cam)]["K_pair"])
            k = torch.tensor([[K[0, 0], 0, K[0, 2] + 0.5], [0, K[1, 1], K[1, 2] + 0.5], [0, 0, 1.0]],
                             dtype=torch.float32, device=device)
            c2w = torch.from_numpy(poses[cam][t]).float().to(device)
            out = render_at(trainer, ii, ci, c2w, k, (h, w))
            rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"{t:03d}_{cam}.jpg"), quality=JPEG_QUALITY)
            rows.append({"cam": cam, "t": t, "input": f"gs/{t:03d}_{cam}.jpg",
                         "coverage": float((out["opacity"][..., 0] > ALPHA).float().mean())})
    covs = {}
    for r in rows:
        covs.setdefault(str(r["cam"]), []).append(r["coverage"])
    with open(os.path.join(args.pairs_dir, "gs.json"), "w") as f:
        json.dump({"scene_id": args.scene_id, "log_dir": os.path.relpath(args.log_dir),
                   "train_meta": io.read_json(os.path.join(args.log_dir, "meta.json")), "coverage_alpha": ALPHA,
                   "mean_coverage": {c: float(np.mean(v)) for c, v in covs.items()},
                   "runtime_s": {"load": t1 - t0, "render": time.time() - t1}, "jpeg_quality": JPEG_QUALITY,
                   "uses_oracle": False, "oracle_note": "FRONT-only 3DGS and joint self-calibration only",
                   "dashrecon_commit": commit, "pairs": rows}, f, indent=2)
    print(f"[e34_gs_pairs] {args.scene_id}: {len(rows)} renders, coverage "
          + ", ".join(f"cam{c} {np.mean(v):.2f}" for c, v in sorted(covs.items()))
          + f" ({time.time() - t1:.0f} s) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
