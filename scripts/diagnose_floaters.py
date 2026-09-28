"""Floater diagnostic A (DECISIONS W): which Gaussians of a trained FRONT-only run make its side-view junk.

Per Background Gaussian (dashrecon.gen.floaters):
    support: blending weight summed over all FRONT training views, only over pixels the photometric loss constrained
        (outside the dynamic and sky masks), in pixel units. Gradient trick: rasterize a zero one-channel colour per
        Gaussian with the trainer's settings and backpropagate the masked image sum; d sum / d c_i = sum of w_i.
    needle ratio: extent along the ray from the nearest training camera over the largest extent across it.
The E5c config initialises 100k "near randoms" with random colours (models/trainers/scene_graph.py,
init.near_randoms); Gaussians that no training view shaped keep such colours, so the DC-colour saturation of
pruned and kept Gaussians is reported too.

Renders fixed cameras (each --frames FRONT camera as trained, CamPose refined, turned by --yaws; 704 x 1280 as
scripts/render_sweep_comparison.py) for these variants of the same checkpoint (parameters changed in memory only):
    base: as trained;
    dc: view-independent colour (SH rest coefficients zero): no colour extrapolated to unseen directions;
    support<t: Gaussians below support t made transparent (one variant per --support_thresholds);
    needle>r: Gaussians above needle ratio --needle_ratio made transparent;
    clean: dc + support < --clean_support + needle > --needle_ratio;
and two maps of the base model: log10 support and needle ratio of each Gaussian blended as colour (turbo; black =
nothing rendered). The FRONT held-out frames (the run's test frames, as scripts/eval_front_heldout.py) are rendered
for every variant to see what each costs on observed views.

Outputs in --out_dir (must not exist): <v>/<k:03d>.png per variant, maps/, <k:03d>_compare.png, comparison.mp4,
cameras.json, metrics.json (counts, percentiles, saturation, FRONT held-out PSNR / LPIPS and side-view mean opacity
per variant), meta.json. No GT is read.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/diagnose_floaters.py --log_dir results/E5c/val056 --frames 96 104 112 \
        --yaws -90 -60 0 60 90 --support_thresholds 1 10 --needle_ratio 4 --clean_support 10 \
        --out_dir results/_diagnostics/floaters/val056/E5c
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import cv2
import imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.floaters import needle_ratio  # noqa: E402
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import assert_non_oracle  # noqa: E402
from dashrecon.train.seeds import seed_scene_optimization  # noqa: E402

HW = (704, 1280)
SH_C0 = 0.28209479177387814
TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(np.float32) / 255


def rasterize(trainer, gs, cam, colours: torch.Tensor) -> tuple:
    """gsplat rasterization of arbitrary per-Gaussian colours with the trainer's render settings (base.py
    render_gaussians); returns (image [H,W,C], alpha [H,W])."""
    from gsplat.rendering import rasterization

    rc = trainer.render_cfg
    assert "radius_clip" not in rc, "radius_clip set: pass it as the trainer does"
    img, alpha, _ = rasterization(
        means=gs.means, quats=gs.quats, scales=gs.scales, opacities=gs.opacities.squeeze(), colors=colours,
        viewmats=torch.linalg.inv(cam.camtoworlds)[None], Ks=cam.Ks[None], width=cam.W, height=cam.H,
        packed=rc.packed, near_plane=rc.near_plane, far_plane=rc.far_plane, render_mode="RGB",
        rasterize_mode="antialiased" if rc.antialiased else "classic")
    return img[0], alpha[0, ..., 0]


def training_support(trainer, dataset, device) -> tuple:
    """(support (N,), views with >= 1 pixel of weight (N,), training camera centres (M,3))."""
    n = trainer.models["Background"]._means.shape[0]
    support = torch.zeros(n, device=device)
    views = torch.zeros(n, device=device)
    centres = []
    for j in range(len(dataset.train_image_set)):
        ii, ci = dataset.train_image_set.get_image(j, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        img_id = ii["img_idx"].flatten()[0]
        with torch.no_grad():
            cam = trainer.process_camera(camera_infos=ci, image_ids=img_id, novel_view=False)
            gs = trainer.collect_gaussians(cam=cam, image_ids=img_id)
        assert gs.means.shape[0] == n
        colour = torch.zeros(n, 1, device=device, requires_grad=True)
        img, _ = rasterize(trainer, gs, cam, colour)
        valid = ((ii["dynamic_masks"] < 0.5) & (ii["sky_masks"] < 0.5)).float()
        (img[..., 0] * valid).sum().backward()
        w = colour.grad[:, 0]
        support += w
        views += (w >= 1.0).float()
        centres.append(cam.camtoworlds[:3, 3].detach())
    return support, views, torch.stack(centres)


def to_turbo(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    idx = ((values.clamp(lo, hi) - lo) / (hi - lo) * 255).round().long()
    return torch.from_numpy(TURBO).to(values.device)[idx]


def front_heldout(trainer, dataset, device) -> dict:
    """Mean PSNR / LPIPS over the run's FRONT test frames at training resolution (as eval_front_heldout.py)."""
    psnr, lp = [], []
    with torch.no_grad():
        for k in dataset.test_timesteps:
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, int(k)), trainer._get_downscale_factor())
            ii, ci = to_device(ii, device), to_device(ci, device)
            out = trainer(ii, ci)
            render = out["rgb"].clamp(0, 1)
            psnr.append(float(-10 * torch.log10(((render - ii["pixels"]) ** 2).mean())))
            lp.append(float(trainer.lpips(render.permute(2, 0, 1)[None], ii["pixels"].permute(2, 0, 1)[None])))
    return {"frames": len(psnr), "psnr": float(np.mean(psnr)), "lpips": float(np.mean(lp))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--yaws", type=float, nargs="+", required=True)
    parser.add_argument("--support_thresholds", type=float, nargs="+", required=True, help="pixel units")
    parser.add_argument("--needle_ratio", type=float, required=True)
    parser.add_argument("--clean_support", type=float, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before a diagnostic run"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed_scene_optimization(0)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    cfg = OmegaConf.load(args.log_dir / "config.yaml")
    assert_non_oracle(cfg)
    from datasets.driving_dataset import DrivingDataset

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=str(args.log_dir / "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    assert set(trainer.gaussian_classes.keys()) == {"Background"}, trainer.gaussian_classes
    bg = trainer.models["Background"]
    assert bg.sh_degree > 0, "the DC colour formula below assumes spherical harmonics"

    support, views, centres = training_support(trainer, dataset, device)
    with torch.no_grad():
        needle = needle_ratio(bg._means, bg.get_scaling, bg.get_quats, centres)
        dc = (SH_C0 * bg._features_dc + 0.5).clamp(0, 1)
        saturation = dc.max(dim=1).values - dc.min(dim=1).values
    opacity0, rest0 = bg._opacities.data.clone(), bg._features_rest.data.clone()

    variants = {"base": (None, False), "dc": (None, True)}
    for t in args.support_thresholds:
        variants[f"support<{t:g}"] = (support < t, False)
    variants[f"needle>{args.needle_ratio:g}"] = (needle > args.needle_ratio, False)
    variants["clean"] = ((support < args.clean_support) | (needle > args.needle_ratio), True)

    cameras = []
    for frame in args.frames:
        ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, frame - dataset.start_timestep), 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        base = refined_c2w(trainer, ii, ci)
        h0, w0 = int(ci["height"]), int(ci["width"])
        k = ci["intrinsics"].clone()
        k[:2] *= HW[1] / w0
        k[1, 2] -= (h0 * HW[1] / w0 - HW[0]) / 2
        for yaw in args.yaws:
            cameras.append({"frame": frame, "yaw": yaw, "c2w": moved_c2w(base, ViewMove(yaw=yaw)), "K": k})

    def render_camera(cam: dict) -> dict:
        ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, cam["frame"] - dataset.start_timestep), 1)
        return render_at(trainer, to_device(ii, device), to_device(ci, device), cam["c2w"], cam["K"], HW)

    images, rows = {}, {}
    for name, (prune, dc_only) in variants.items():
        bg._opacities.data.copy_(opacity0)
        bg._features_rest.data.copy_(rest0)
        if prune is not None:
            bg._opacities.data[prune] = -1e4
        if dc_only:
            bg._features_rest.data.zero_()
        (args.out_dir / name).mkdir()
        images[name], opac = [], []
        with torch.no_grad():
            for k, cam in enumerate(cameras):
                out = render_camera(cam)
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(args.out_dir / name / f"{k:03d}.png")
                images[name].append(rgb)
                opac.append(float(out["opacity"].mean()))
        rows[name] = {"pruned": 0 if prune is None else int(prune.sum()), "dc_only": dc_only,
                      "side_mean_opacity": opac, "front_heldout": front_heldout(trainer, dataset, device)}
        print(f"[floaters] {name}: pruned {rows[name]['pruned']:,}, FRONT held-out {rows[name]['front_heldout']}", flush=True)
    bg._opacities.data.copy_(opacity0)
    bg._features_rest.data.copy_(rest0)

    (args.out_dir / "maps").mkdir()
    maps = {"support": to_turbo(torch.log10(support + 1e-3), -2.0, 4.0), "needle": to_turbo(needle, 1.0, 10.0)}
    for name in maps:
        images[f"{name} map"] = []
    with torch.no_grad():
        for k, cam in enumerate(cameras):
            render_camera(cam)
            for name, colours in maps.items():
                img, _ = rasterize(trainer, trainer._last_gs, trainer._last_cam, colours)
                rgb = (img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(args.out_dir / "maps" / f"{k:03d}_{name}.png")
                images[f"{name} map"].append(rgb)

    cw, ch = 480, 264
    writer = imageio.get_writer(str(args.out_dir / "comparison.mp4"), fps=2, macro_block_size=1)
    for k, cam in enumerate(cameras):
        canvas = Image.new("RGB", (cw * len(images), ch + 24), "#151515")
        draw = ImageDraw.Draw(canvas)
        for col, (name, ims) in enumerate(images.items()):
            canvas.paste(Image.fromarray(ims[k]).resize((cw, ch)), (cw * col, 24))
            draw.text((cw * col + 6, 6), f"{name} | frame {cam['frame']} yaw {cam['yaw']:g}", fill="white")
        canvas.save(args.out_dir / f"{k:03d}_compare.png")
        writer.append_data(np.asarray(canvas))
    writer.close()

    def pct(x: torch.Tensor) -> dict:
        q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=x.device)
        return {f"p{int(p * 100)}": float(v) for p, v in zip(q, torch.quantile(x.float()[torch.randperm(len(x), device=x.device)[:1000000]], q))}

    clean_mask = variants["clean"][0]
    metrics = {
        "gaussians": int(len(support)),
        "support_pixel_units": pct(support), "views_with_weight_ge_1px": pct(views), "needle_ratio": pct(needle),
        "support_below": {f"{t:g}": int((support < t).sum()) for t in [0.01, 0.1, 1, 10, 100]},
        "dc_saturation": {"pruned_by_clean": pct(saturation[clean_mask]), "kept_by_clean": pct(saturation[~clean_mask])},
        "variants": rows,
        "cameras": [{"frame": c["frame"], "yaw": c["yaw"]} for c in cameras],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "cameras.json").write_text(json.dumps(
        [{"frame": c["frame"], "yaw": c["yaw"], "c2w": c["c2w"].cpu().tolist(), "K": c["K"].cpu().tolist()} for c in cameras], indent=2))
    (args.out_dir / "meta.json").write_text(json.dumps({
        "kind": "floater diagnostic A: in-memory pruning / colour variants of one checkpoint, no training",
        "log_dir": str(args.log_dir), "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "uses_oracle": False, "oracle_information": "none; FRONT images, masks and estimated cameras of the run only",
        "generates_content": False, "model_provenance": "see the log_dir's own meta.json",
        "dashrecon_commit": commit, "runtime_s": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "versions": {name: version(name) for name in ["torch", "gsplat", "numpy"]}}, indent=2))
    print(f"[floaters] {len(cameras)} cameras x {len(variants)} variants -> {args.out_dir} "
          f"({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
