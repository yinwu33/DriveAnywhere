"""Floater diagnostic A (DECISIONS W): which Gaussians of a trained FRONT-only run make its side-view junk.

Per Background Gaussian:
    support: blending weight summed over all FRONT training views, only over pixels the photometric loss constrained
        (outside the dynamic and sky masks), in pixel units. Gradient trick: rasterize a zero one-channel colour per
        Gaussian with the trainer's settings and backpropagate the masked image sum; d sum / d c_i = sum of w_i.
    side: the same blending weight summed over the --frames x --yaws cameras (all pixels), i.e. what each Gaussian
        contributes to the views being diagnosed.
    scale: largest axis (standard deviation, scene units).
    mesh: distance from its centre to the nearest vertex of --mesh_ply (the run's own NKSR mesh, a dashrecon product).
    needle: extent along the ray from the nearest training camera over the largest extent across it
        (dashrecon.gen.floaters).
    saturation: of its view-independent (DC) colour; the E5c config initialises 100k "near randoms" with random colours
        (models/trainers/scene_graph.py init.near_randoms).
    observation cone (dashrecon.gen.floaters.observation_cone): the weight-averaged direction the training views saw
        it from and the half-angle covering every view with weight >= --cone_min_weight pixels.

Each --variants entry renders the same checkpoint with parameters changed in memory only: "base" (as trained), or
'&'-joined terms where "dc" zeroes the SH rest coefficients (no colour extrapolated to unseen directions) and
<quantity><|><value> terms (quantity support | scale | mesh | needle) select Gaussians that are made transparent when
all terms hold, e.g. "scale>1", "support<1", "scale>0.3&mesh>1", "dc&support<1". A "gate<margin>" term (degrees)
instead fades the selected Gaussians (all Gaussians when there is no other term) per camera by
dashrecon.gen.floaters.cone_gate: kept while the camera sees them within their observation cone + margin, gone
--gate_fade degrees later, and gone everywhere if no training view saw them (D-A3 in docs/EXPERIMENTS.md), e.g.
"gate30", "gate15&mesh>0.3". The first variant must be "base". For every variant: the fixed cameras
(each --frames FRONT camera as trained, CamPose refined, turned by --yaws; 704 x 1280 as
scripts/render_sweep_comparison.py), the FRONT held-out frames (the run's test frames, PSNR / LPIPS as
scripts/eval_front_heldout.py) to see what the change costs on observed views, and the share of the base side weight
it removes (static variants), and the fraction of side pixels it exposes (opacity > 0.5 in base, < 0.5 in the
variant). Maps of the base model: log10 support, mesh distance and log10 scale of each Gaussian blended as colour
(turbo; black = nothing rendered).

Outputs in --out_dir (must not exist): <variant>/<k:03d>.png, maps/, <k:03d>_compare.png, comparison.mp4, cameras.json,
metrics.json (percentiles overall and over the Gaussians making 80 % of the side weight, side-weight shares by
scale / mesh distance, per-variant counts, FRONT held-out and side weight removed), meta.json. No GT is read.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/diagnose_floaters.py --log_dir results/E5c/val056 \
        --mesh_ply data/dashrecon/val056/pose-glomap_depth-mapanything__mask-gsam2_sky-segformer_img-glomap/mesh_nksr.ply \
        --frames 96 104 112 --yaws -90 -60 0 60 90 180 --variants base "support<1" "scale>1" "mesh>3" "scale>0.3&mesh>1" \
        --out_dir results/_diagnostics/floaters/val056/E5c_b
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

import cv2
import imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon import io  # noqa: E402
from dashrecon.gen.floaters import cone_gate, needle_ratio, observation_cone  # noqa: E402
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.train.guard import DASHRECON_ROOT, assert_non_oracle  # noqa: E402
from dashrecon.train.seeds import seed_scene_optimization  # noqa: E402

HW = (704, 1280)
SH_C0 = 0.28209479177387814
TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(np.float32) / 255
TERM = re.compile(r"^(support|scale|mesh|needle)([<>])([0-9.]+)$")
GATE = re.compile(r"^gate([0-9.]+)$")


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


def blend_weight(trainer, gs, cam, mask: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian blending weight summed over the pixels of ``mask`` [H,W] (gradient of the masked image sum)."""
    colour = torch.zeros(gs.means.shape[0], 1, device=mask.device, requires_grad=True)
    img, _ = rasterize(trainer, gs, cam, colour)
    (img[..., 0] * mask).sum().backward()
    return colour.grad[:, 0]


def training_observations(trainer, dataset, device) -> tuple:
    """(weights (V,N) float16: blending weight of every Gaussian in every FRONT training view over the pixels the
    photometric loss constrained, training camera centres (V,3))."""
    n = trainer.models["Background"]._means.shape[0]
    weights, centres = [], []
    for j in range(len(dataset.train_image_set)):
        ii, ci = dataset.train_image_set.get_image(j, 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        img_id = ii["img_idx"].flatten()[0]
        with torch.no_grad():
            cam = trainer.process_camera(camera_infos=ci, image_ids=img_id, novel_view=False)
            gs = trainer.collect_gaussians(cam=cam, image_ids=img_id)
        assert gs.means.shape[0] == n
        w = blend_weight(trainer, gs, cam, ((ii["dynamic_masks"] < 0.5) & (ii["sky_masks"] < 0.5)).float())
        assert float(w.max()) < 65504, "weight overflows float16"
        weights.append(w.half())
        centres.append(cam.camtoworlds[:3, 3].detach())
    return torch.stack(weights), torch.stack(centres)


def to_turbo(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    idx = ((values.clamp(lo, hi) - lo) / (hi - lo) * 255).round().long()
    return torch.from_numpy(TURBO).to(values.device)[idx]


def front_heldout(trainer, dataset, device, prepare) -> dict:
    """Mean PSNR / LPIPS over the run's FRONT test frames at training resolution (as eval_front_heldout.py);
    ``prepare(centre)`` sets the variant's parameters for a camera at ``centre`` before each render."""
    psnr, lp = [], []
    with torch.no_grad():
        for k in dataset.test_timesteps:
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, int(k)), trainer._get_downscale_factor())
            ii, ci = to_device(ii, device), to_device(ci, device)
            prepare(ci["camera_to_world"][:3, 3])
            out = trainer(ii, ci)
            render = out["rgb"].clamp(0, 1)
            psnr.append(float(-10 * torch.log10(((render - ii["pixels"]) ** 2).mean())))
            lp.append(float(trainer.lpips(render.permute(2, 0, 1)[None], ii["pixels"].permute(2, 0, 1)[None])))
    return {"frames": len(psnr), "psnr": float(np.mean(psnr)), "lpips": float(np.mean(lp))}


def parse_variant(spec: str, quantities: dict) -> tuple:
    """"base" -> (None, False, None); "dc&scale>1" -> (selection, dc_only, None); "gate30&mesh>0.3" -> (selection,
    False, 30.0). The selection is None when no quantity term is given. Unknown terms raise."""
    if spec == "base":
        return None, False, None
    dc_only, select, margin = False, None, None
    for term in spec.split("&"):
        if term == "dc":
            dc_only = True
            continue
        g = GATE.match(term)
        if g is not None:
            margin = float(g.group(1))
            continue
        m = TERM.match(term)
        assert m is not None, f"bad variant term {term!r} in {spec!r}"
        x, v = quantities[m.group(1)], float(m.group(3))
        sel = x < v if m.group(2) == "<" else x > v
        select = sel if select is None else select & sel
    return select, dc_only, margin


def pct(x: torch.Tensor) -> dict:
    q = torch.tensor([0.1, 0.5, 0.9], device=x.device)
    sample = x.float()[torch.randperm(len(x), device=x.device)[:1000000]]
    return {f"p{int(p * 100)}": float(v) for p, v in zip(q, torch.quantile(sample, q))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--mesh_ply", required=True, help="the run's NKSR mesh (a dashrecon product)")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--yaws", type=float, nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--cone_min_weight", type=float, required=True, help="pixels of weight for a view to count")
    parser.add_argument("--gate_fade", type=float, required=True, help="degrees over which a gated Gaussian fades out")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before a diagnostic run"
    assert args.mesh_ply.startswith(DASHRECON_ROOT), f"mesh must be a dashrecon product: {args.mesh_ply}"
    assert len(set(args.variants)) == len(args.variants) and "/" not in "".join(args.variants)
    assert args.variants[0] == "base", "the first variant must be base (exposed pixels are measured against it)"
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

    weights, centres = training_observations(trainer, dataset, device)
    support = weights.float().sum(dim=0)
    views = (weights >= 1.0).float().sum(dim=0)
    side = torch.zeros_like(support)
    for cam in cameras:
        with torch.no_grad():
            render_camera(cam)
        side += blend_weight(trainer, trainer._last_gs, trainer._last_cam, torch.ones(HW, device=device))
    with torch.no_grad():
        mesh = io.read_mesh_ply(args.mesh_ply)
        mesh_dist = torch.from_numpy(cKDTree(mesh["vertices"]).query(bg._means.cpu().numpy(), k=1, workers=16)[0]).float().to(device)
        scale = bg.get_scaling.max(dim=1).values
        needle = needle_ratio(bg._means, bg.get_scaling, bg.get_quats, centres)
        axis, half_angle, observed = observation_cone(bg._means, centres, weights, args.cone_min_weight)
        dc = (SH_C0 * bg._features_dc + 0.5).clamp(0, 1)
        saturation = dc.max(dim=1).values - dc.min(dim=1).values
    del weights
    quantities = {"support": support, "scale": scale, "mesh": mesh_dist, "needle": needle}
    order = torch.argsort(side, descending=True)
    top = order[: int((torch.cumsum(side[order], 0) < 0.8 * side.sum()).sum()) + 1]
    opacity0, rest0 = bg._opacities.data.clone(), bg._features_rest.data.clone()

    images, rows, base_opacity = {}, {}, []
    for spec in args.variants:
        select, dc_only, margin = parse_variant(spec, quantities)
        bg._features_rest.data.copy_(rest0)
        if dc_only:
            bg._features_rest.data.zero_()
        static = opacity0.clone()
        if select is not None and margin is None:
            static[select] = -1e4

        def prepare(centre: torch.Tensor) -> None:
            if margin is None:
                bg._opacities.data.copy_(static)
                return
            g = cone_gate(bg._means, centre, axis, half_angle, observed, margin, args.gate_fade)
            if select is not None:
                g = torch.where(select, g, torch.ones_like(g))
            p = torch.sigmoid(opacity0[:, 0]) * g
            bg._opacities.data[:, 0] = torch.logit(p.clamp(1e-12, 1 - 1e-7))

        (args.out_dir / spec).mkdir()
        images[spec], opac, exposed = [], [], []
        with torch.no_grad():
            for k, cam in enumerate(cameras):
                prepare(cam["c2w"][:3, 3])
                out = render_camera(cam)
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(args.out_dir / spec / f"{k:03d}.png")
                images[spec].append(rgb)
                o = out["opacity"][..., 0]
                opac.append(float(o.mean()))
                if spec == "base":
                    base_opacity.append(o > 0.5)
                exposed.append(float((base_opacity[k] & (o < 0.5)).float().mean()))
        prune = select if margin is None else None
        rows[spec] = {"selected": 0 if select is None else int(select.sum()), "dc_only": dc_only, "gate_margin_deg": margin,
                      "base_side_weight_removed": None if prune is None else float(side[prune].sum() / side.sum()),
                      "selected_saturation": None if select is None else pct(saturation[select]),
                      "side_mean_opacity": opac, "side_exposed_fraction": exposed,
                      "front_heldout": front_heldout(trainer, dataset, device, prepare)}
        print(f"[floaters] {spec}: selected {rows[spec]['selected']:,}, exposed {np.mean(exposed):.3f}, "
              f"FRONT held-out {rows[spec]['front_heldout']}", flush=True)
    bg._opacities.data.copy_(opacity0)
    bg._features_rest.data.copy_(rest0)

    (args.out_dir / "maps").mkdir()
    maps = {"support": to_turbo(torch.log10(support + 1e-3), -2.0, 4.0), "mesh": to_turbo(mesh_dist, 0.0, 5.0),
            "scale": to_turbo(torch.log10(scale), -2.0, 1.0)}
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

    cw, ch, ncol = 480, 264, 4
    nrow = -(-len(images) // ncol)
    writer = imageio.get_writer(str(args.out_dir / "comparison.mp4"), fps=2, macro_block_size=1)
    for k, cam in enumerate(cameras):
        canvas = Image.new("RGB", (cw * ncol, (ch + 24) * nrow), "#151515")
        draw = ImageDraw.Draw(canvas)
        for i, (name, ims) in enumerate(images.items()):
            x, y = cw * (i % ncol), (ch + 24) * (i // ncol)
            canvas.paste(Image.fromarray(ims[k]).resize((cw, ch)), (x, y + 24))
            draw.text((x + 6, y + 6), f"{name} | frame {cam['frame']} yaw {cam['yaw']:g}", fill="white")
        canvas.save(args.out_dir / f"{k:03d}_compare.png")
        writer.append_data(np.asarray(canvas))
    writer.close()

    shares = {}
    for name, x, bins in [("scale", scale, [0, 0.05, 0.2, 0.5, 1, float("inf")]), ("mesh", mesh_dist, [0, 0.1, 0.3, 1, 3, float("inf")])]:
        shares[name] = [{"range": [a, b], "gaussians": int(((x >= a) & (x < b)).sum()),
                         "side_weight_share": float(side[(x >= a) & (x < b)].sum() / side.sum())} for a, b in zip(bins[:-1], bins[1:])]
    metrics = {
        "gaussians": int(len(support)), "gaussians_making_80pct_side_weight": int(len(top)),
        "percentiles_all": {name: pct(x) for name, x in [*quantities.items(), ("views_ge_1px", views), ("saturation", saturation)]},
        "observation_cone": {"observed_fraction": float(observed.float().mean()), "half_angle_deg_observed": pct(half_angle[observed]),
                             "min_weight_px": args.cone_min_weight, "gate_fade_deg": args.gate_fade},
        "percentiles_side_top": {name: pct(x[top]) for name, x in [*quantities.items(), ("saturation", saturation)]},
        "side_weight_share": shares, "variants": rows,
        "cameras": [{"frame": c["frame"], "yaw": c["yaw"]} for c in cameras],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "cameras.json").write_text(json.dumps(
        [{"frame": c["frame"], "yaw": c["yaw"], "c2w": c["c2w"].cpu().tolist(), "K": c["K"].cpu().tolist()} for c in cameras], indent=2))
    (args.out_dir / "meta.json").write_text(json.dumps({
        "kind": "floater diagnostic A: in-memory pruning / colour variants of one checkpoint, no training",
        "log_dir": str(args.log_dir), "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "uses_oracle": False, "oracle_information": "none; FRONT images, masks, estimated cameras and mesh of the run only",
        "generates_content": False, "model_provenance": "see the log_dir's own meta.json",
        "dashrecon_commit": commit, "runtime_s": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "versions": {name: version(name) for name in ["torch", "gsplat", "numpy", "scipy"]}}, indent=2))
    print(f"[floaters] {len(cameras)} cameras x {len(args.variants)} variants -> {args.out_dir} "
          f"({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
