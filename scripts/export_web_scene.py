"""Export a trained run for the 3D review page (docs/EXPERIMENTS.md, 10-04 review artifact): Gaussians, sky and cameras.

Gaussians: the Background model's Gaussians with the colour of SH band 0 (view-dependent bands dropped), opacity,
scale and rotation, plus the run's view gate (dashrecon.gen.floaters.ViewGate, E14 / E20) so the page can apply the
same observation-cone opacity factor the evaluation renderer applies to every novel view. At most --max_splats are
kept, ranked by opacity x sqrt(largest x middle scale) (the visible footprint); --check_frames renders the run with and
without the dropped Gaussians at a few FRONT poses (straight and turned) and reports the PSNR between the two, so the
cut is chosen knowingly. Binary layout, little endian, 32 bytes per splat, written in chunks of --chunk splats:
    float32 x, y, z | float16 sx, sy, sz (linear scales) | uint8 r, g, b, a | uint8 qw, qx, qy, qz (q / |q| * 128 + 128)
    | int8 ax, ay, az (gate axis * 127) | uint8 half angle (degrees, rounded) | uint8 flags (1 gated, 2 observed) | pad 1
Sky: the run's Sky model queried on an equirectangular grid (--sky_hw), azimuth atan2(y, x) of the world direction
from -180 to 180 degrees left to right, elevation from +90 at the top to -90 at the bottom; JPEG.
Cameras: every frame's FRONT camera as rendered for novel views (CamPose refined, OpenCV c2w, world x forward, y left,
z up), its pinhole intrinsics and image size, and the evaluation-side metre scale (GT FRONT trajectory Sim(3), as
scripts/eval_view_range.py; reads GT, for labelling the page's metre sliders only).

Output <out_dir>/<label>/: splats_<k>.bin, sky.jpg, scene.json (counts, chunk files, gate parameters, cameras, scale).

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/export_web_scene.py --log_dir results/E35/val056/model --label E35 \\
        --gt_root data/waymo/processed/validation --max_splats 900000 --out_dir results/_web3d/val056
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.gen.floaters import ViewGate  # noqa: E402
from dashrecon.gen.novel import attach_view_gate, build_trainer, refined_c2w, to_device, view_gate_path  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, render_moved  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from eval_cross_camera import gt_cam_to_ego, umeyama_scale  # noqa: E402

C0 = 0.28209479177387814


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--max_splats", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=400000)
    parser.add_argument("--sky_hw", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--check_frames", type=int, nargs="*", default=[50, 100, 150])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    commit = git_commit()
    device = torch.device("cuda")
    out = os.path.join(args.out_dir, args.label)
    os.makedirs(out, exist_ok=True)
    from datasets.driving_dataset import DrivingDataset

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    gate_file = view_gate_path(cfg)
    if gate_file is not None:
        attach_view_gate(trainer, gate_file)
    trainer.set_eval()
    bg = trainer.models["Background"]
    with torch.no_grad():
        means = bg._means.detach()
        scales = torch.exp(bg._scales.detach())
        if scales.shape[1] == 2:  # 2D Gaussians store two scales
            scales = torch.cat([scales, torch.full_like(scales[:, :1], 1e-6)], 1)
        quats = torch.nn.functional.normalize(bg._quats.detach(), dim=-1)
        opac = torch.sigmoid(bg._opacities.detach())[:, 0]
        sh0 = bg._features_dc.detach()
        rgb = (sh0 * C0 + 0.5).clamp(0, 1) if bg.sh_degree > 0 else torch.sigmoid(sh0)
    n_all = means.shape[0]
    gate = ViewGate.load(gate_file, device) if gate_file is not None else None
    srt = scales.sort(dim=1, descending=True).values
    importance = opac * torch.sqrt(srt[:, 0] * srt[:, 1])
    keep = torch.argsort(importance, descending=True)[:args.max_splats]
    keep = keep.sort().values  # keep the original order (the gate covers the first n)
    print(f"[export_web_scene] {args.label}: {n_all:,} Gaussians, keeping {len(keep):,}", flush=True)

    # quality check of the cut: render with the dropped Gaussians made transparent
    full_ops = bg._opacities.data.clone()
    checks = []
    with torch.no_grad():
        for t in args.check_frames:
            for spec in ("yaw=0", "yaw=-45"):
                a = render_moved(trainer, dataset, t, ViewMove.parse(spec), device)[0]["rgb"].clamp(0, 1)
                drop = torch.ones(n_all, dtype=torch.bool, device=device)
                drop[keep] = False
                bg._opacities.data[drop] = -30.0
                b = render_moved(trainer, dataset, t, ViewMove.parse(spec), device)[0]["rgb"].clamp(0, 1)
                bg._opacities.data.copy_(full_ops)
                checks.append({"frame": t, "move": spec, "psnr_cut_vs_full": float(-10 * torch.log10(((a - b) ** 2).mean()))})
    print("[export_web_scene] cut vs full PSNR: " + "  ".join(f"t{c['frame']} {c['move']} {c['psnr_cut_vs_full']:.1f}" for c in checks), flush=True)

    k = keep.cpu().numpy()
    rec = np.zeros(len(k), dtype=[("pos", "<f4", 3), ("scale", "<f2", 3), ("rgba", "u1", 4), ("quat", "u1", 4),
                                  ("axis", "i1", 3), ("half", "u1"), ("flags", "u1"), ("pad", "u1")])
    assert rec.dtype.itemsize == 32, rec.dtype.itemsize
    rec["pos"] = means[keep].cpu().numpy()
    rec["scale"] = np.clip(scales[keep].cpu().numpy(), 1e-6, 6.0e4).astype(np.float16)
    rgba = np.concatenate([rgb[keep].cpu().numpy(), opac[keep].cpu().numpy()[:, None]], 1)
    rec["rgba"] = np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8)
    rec["quat"] = np.clip(np.round(quats[keep].cpu().numpy() * 128 + 128), 0, 255).astype(np.uint8)
    if gate is not None:
        in_gate = k < gate.n
        gk = k[in_gate]
        axis = gate.axis[gk].cpu().numpy()
        rec["axis"][in_gate] = np.clip(np.round(axis * 127), -127, 127).astype(np.int8)
        rec["half"][in_gate] = np.clip(np.round(gate.half_angle[gk].cpu().numpy()), 0, 180).astype(np.uint8)
        sel = gate.selected[gk].cpu().numpy()
        obs = gate.observed[gk].cpu().numpy()
        rec["flags"][in_gate] = (sel.astype(np.uint8) * 1) | (obs.astype(np.uint8) * 2)
    files = []
    for c, s in enumerate(range(0, len(rec), args.chunk)):
        name = f"splats_{c}.bin"
        rec[s:s + args.chunk].tofile(os.path.join(out, name))
        files.append(name)

    # sky on an equirectangular grid
    h, w = args.sky_hw
    az = np.deg2rad(np.linspace(-180, 180, w, endpoint=False) + 180.0 / w)
    el = np.deg2rad(np.linspace(90, -90, h, endpoint=False) - 90.0 / h)
    azg, elg = np.meshgrid(az, el)
    dirs = np.stack([np.cos(elg) * np.cos(azg), np.cos(elg) * np.sin(azg), np.sin(elg)], -1).astype(np.float32)
    with torch.no_grad():
        sky = trainer.models["Sky"]({"viewdirs": torch.from_numpy(dirs).to(device)}).clamp(0, 1).cpu().numpy()
    Image.fromarray((sky * 255).round().astype(np.uint8)).save(os.path.join(out, "sky.jpg"), quality=88)

    # cameras (as rendered for novel views) and the evaluation-side metre scale
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    cams, est = [], []
    with torch.no_grad():
        for i in range(len(frames)):
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, i), 1)
            est.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
            ii, ci = to_device(ii, device), to_device(ci, device)
            c2w = refined_c2w(trainer, ii, ci).cpu().numpy()
            kk = ci["intrinsics"].cpu().numpy()
            cams.append({"frame": int(frames[i]), "c2w": c2w.round(6).tolist(), "fx": float(kk[0, 0]), "fy": float(kk[1, 1]),
                         "cx": float(kk[0, 2]), "cy": float(kk[1, 2]), "w": int(ci["width"]), "h": int(ci["height"])})
    gt_dir = os.path.join(args.gt_root, f"{int(cfg.data.scene_idx):03d}")
    front_to_ego = gt_cam_to_ego(gt_dir, 0)
    gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{int(t):03d}.txt")) @ front_to_ego for t in frames])
    s = umeyama_scale(gt_front[:, :3, 3], np.stack(est)[:, :3, 3])
    lo, hi = means[keep].quantile(0.01, dim=0).cpu().numpy(), means[keep].quantile(0.99, dim=0).cpu().numpy()
    meta = {"label": args.label, "log_dir": args.log_dir, "splats": int(len(rec)), "gaussians_total": int(n_all), "files": files,
            "chunk": args.chunk, "record_bytes": 32, "gate": None if gate is None else {"margin": gate.margin, "fade": gate.fade,
            "double_sided": gate.double_sided, "gated_in_export": int(((rec["flags"] & 1) > 0).sum())},
            "sh_degree_exported": 0, "cut_check": checks, "bbox_p1_p99": [lo.round(3).tolist(), hi.round(3).tolist()],
            "metre_to_units": s, "metre_scale_source": "evaluation side: Sim(3) of the GT FRONT trajectory (reads GT)",
            "cameras": cams, "dashrecon_commit": commit}
    with open(os.path.join(out, "scene.json"), "w") as f:
        json.dump(meta, f)
    print(f"[export_web_scene] {args.label}: {len(rec):,} splats in {len(files)} file(s), {sum(os.path.getsize(os.path.join(out, x)) for x in files) / 1e6:.1f} MB "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
