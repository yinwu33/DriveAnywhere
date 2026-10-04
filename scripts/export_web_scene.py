"""Export a trained run for the 3D review page (docs/EXPERIMENTS.md, 10-04 review artifact): Gaussians, sky and cameras.

Gaussians: the Background model's Gaussians with one colour each, opacity, scale and rotation. --colour nearest_view
(default) evaluates the run's spherical harmonics for the direction from the nearest FRONT camera that has the Gaussian
in its image (the nearest camera at all if none has): the colour training supervised, where the SH band 0 alone (--colour
dc) can be far off (green road patches on val056 E35, corrected by the higher bands). SH degree 0 runs store one colour. Plus the run's view gate (dashrecon.gen.floaters.ViewGate, E14 / E20) so the page can apply the
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

The page is dashrecon/viewer/review3d.html (a WebGL2 splat renderer with the same gate); copy it to <out_dir>/index.html.
Output <out_dir>/<label>/: splats_<k>.png, sky.jpg, scene.json (counts, chunk files, gate parameters, cameras, scale).
Each chunk's bytes are stored losslessly as an RGB PNG of width PNG_WIDTH (3 bytes per pixel, zero padded; artifact
hosting serves images but not raw binary); scene.json has every file's byte count and byte sum mod 2^32 to check the
decode.

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
from gsplat.cuda._wrapper import spherical_harmonics  # noqa: E402

C0 = 0.28209479177387814
PNG_WIDTH = 4096


def write_png_bytes(data: bytes, path: str) -> dict:
    """``data`` as an RGB PNG of width PNG_WIDTH (row-major bytes, zero padded); returns its byte count and sum."""
    rows = -(-len(data) // (3 * PNG_WIDTH))
    buf = np.zeros(rows * PNG_WIDTH * 3, np.uint8)
    raw = np.frombuffer(data, np.uint8)
    buf[:len(raw)] = raw
    Image.fromarray(buf.reshape(rows, PNG_WIDTH, 3), "RGB").save(path, optimize=False, compress_level=6)
    return {"bytes": len(raw), "sum32": int(raw.astype(np.uint64).sum() % 2**32)}


def nearest_view_dirs(means: torch.Tensor, c2ws: list, ks: list, hw: tuple) -> tuple:
    """Unit direction to every point from the nearest camera that has it in its image (from the nearest camera at all
    if none has), and whether one has."""
    h, w = hw
    inf = torch.full((len(means),), float("inf"), device=means.device)
    best, best_any = inf.clone(), inf.clone()
    dirs, dirs_any = torch.zeros_like(means), torch.zeros_like(means)
    for c2w, k in zip(c2ws, ks):
        d = means - c2w[:3, 3]
        dist = d.norm(dim=-1)
        cam = d @ c2w[:3, :3]  # camera coordinates (OpenCV)
        z = cam[:, 2].clamp(min=1e-6)
        u, v = k[0, 0] * cam[:, 0] / z + k[0, 2], k[1, 1] * cam[:, 1] / z + k[1, 2]
        vis = (cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        better = vis & (dist < best)
        best = torch.where(better, dist, best)
        dirs[better] = d[better]
        nearer = dist < best_any
        best_any = torch.where(nearer, dist, best_any)
        dirs_any[nearer] = d[nearer]
    seen = torch.isfinite(best)
    dirs = torch.where(seen[:, None], dirs, dirs_any)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-9), seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--max_splats", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=400000)
    parser.add_argument("--sky_hw", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--check_frames", type=int, nargs="*", default=[50, 100, 150])
    parser.add_argument("--colour", choices=["nearest_view", "dc"], default="nearest_view")
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

    # cameras (as rendered for novel views)
    frames = np.arange(dataset.start_timestep, dataset.end_timestep)
    cams, est, c2ws, ks = [], [], [], []
    with torch.no_grad():
        for i in range(len(frames)):
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, i), 1)
            est.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
            ii, ci = to_device(ii, device), to_device(ci, device)
            c2w = refined_c2w(trainer, ii, ci)
            c2ws.append(c2w.float())
            ks.append(ci["intrinsics"].float())
            kk = ci["intrinsics"].cpu().numpy()
            cams.append({"frame": int(frames[i]), "c2w": c2w.cpu().numpy().round(6).tolist(), "fx": float(kk[0, 0]), "fy": float(kk[1, 1]),
                         "cx": float(kk[0, 2]), "cy": float(kk[1, 2]), "w": int(ci["width"]), "h": int(ci["height"])})

    with torch.no_grad():
        if bg.sh_degree == 0:
            rgb = torch.sigmoid(bg._features_dc.detach()[keep])
            colour = "sigmoid of the stored colour (SH degree 0)"
        elif args.colour == "dc":
            rgb = (bg._features_dc.detach()[keep] * C0 + 0.5).clamp(0, 1)
            colour = "SH band 0"
        else:
            dirs, seen = nearest_view_dirs(means[keep], c2ws, ks, (cams[0]["h"], cams[0]["w"]))
            coeffs = torch.cat([bg._features_dc.detach()[keep][:, None, :], bg._features_rest.detach()[keep]], 1)
            rgb = (spherical_harmonics(bg.sh_degree, dirs, coeffs) + 0.5).clamp(0, 1)
            colour = f"SH evaluated from the nearest FRONT camera that sees the Gaussian ({float(seen.float().mean()):.1%} of them; the rest from the nearest camera)"
    print(f"[export_web_scene] colour: {colour}", flush=True)

    k = keep.cpu().numpy()
    rec = np.zeros(len(k), dtype=[("pos", "<f4", 3), ("scale", "<f2", 3), ("rgba", "u1", 4), ("quat", "u1", 4),
                                  ("axis", "i1", 3), ("half", "u1"), ("flags", "u1"), ("pad", "u1")])
    assert rec.dtype.itemsize == 32, rec.dtype.itemsize
    rec["pos"] = means[keep].cpu().numpy()
    rec["scale"] = np.clip(scales[keep].cpu().numpy(), 1e-6, 6.0e4).astype(np.float16)
    rgba = np.concatenate([rgb.cpu().numpy(), opac[keep].cpu().numpy()[:, None]], 1)
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
    files, file_info = [], []
    for c, s in enumerate(range(0, len(rec), args.chunk)):
        name = f"splats_{c}.png"
        file_info.append(write_png_bytes(rec[s:s + args.chunk].tobytes(), os.path.join(out, name)))
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

    # the evaluation-side metre scale
    gt_dir = os.path.join(args.gt_root, f"{int(cfg.data.scene_idx):03d}")
    front_to_ego = gt_cam_to_ego(gt_dir, 0)
    gt_front = np.stack([np.loadtxt(os.path.join(gt_dir, "ego_pose", f"{int(t):03d}.txt")) @ front_to_ego for t in frames])
    s = umeyama_scale(gt_front[:, :3, 3], np.stack(est)[:, :3, 3])
    lo, hi = means[keep].quantile(0.01, dim=0).cpu().numpy(), means[keep].quantile(0.99, dim=0).cpu().numpy()
    meta = {"label": args.label, "log_dir": args.log_dir, "splats": int(len(rec)), "gaussians_total": int(n_all), "files": files,
            "file_info": file_info, "png_width": PNG_WIDTH, "chunk": args.chunk, "record_bytes": 32, "gate": None if gate is None else {"margin": gate.margin, "fade": gate.fade,
            "double_sided": gate.double_sided, "gated_in_export": int(((rec["flags"] & 1) > 0).sum())},
            "sh_degree_exported": 0, "colour": colour, "cut_check": checks, "bbox_p1_p99": [lo.round(3).tolist(), hi.round(3).tolist()],
            "metre_to_units": s, "metre_scale_source": "evaluation side: Sim(3) of the GT FRONT trajectory (reads GT)",
            "cameras": cams, "dashrecon_commit": commit}
    with open(os.path.join(out, "scene.json"), "w") as f:
        json.dump(meta, f)
    print(f"[export_web_scene] {args.label}: {len(rec):,} splats in {len(files)} file(s), {sum(os.path.getsize(os.path.join(out, x)) for x in files) / 1e6:.1f} MB "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
