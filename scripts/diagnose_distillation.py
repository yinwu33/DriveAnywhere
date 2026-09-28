"""Separate direct generation from 3D fit at identical cameras; also render
unsupervised yaw and translation. This is not ground-truth NVS scoring.
Compare common confidence regions; full-size PNG metrics precede video encoding.
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
from dashrecon.gen.novel import build_trainer, to_device
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at
from dashrecon.provenance import git_commit
from dashrecon.eval.distillation import fit_metrics
from dashrecon.train.guard import assert_non_oracle
from dashrecon.train.seeds import seed_scene_optimization


def sharpness(rgb: np.ndarray, mask: np.ndarray) -> float | None:
    """Laplacian variance inside the eroded mask; no mask-edge discontinuities."""
    inside = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not inside.any():
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F)[inside].var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--views_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_indices", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--novel_yaw", type=float, default=5.)
    parser.add_argument("--novel_right", type=float, default=.5, help="estimated scene units, not calibrated metres")
    parser.add_argument("--validation_dirs", type=Path, nargs="*", default=[], help="own validation first, other arms after it for common regions")
    parser.add_argument("--label", default="checkpoint")
    args = parser.parse_args()
    if args.novel_yaw == 0 or args.novel_right == 0:
        raise ValueError("novel yaw and translation must differ from the supervised camera")
    commit = git_commit()
    if commit.endswith("-dirty"):
        raise ValueError("commit code changes before starting a diagnostic")
    cams = json.loads((args.views_dir / "cams.json").read_text())["cams"]
    if not args.view_indices or len(set(args.view_indices)) != len(args.view_indices) or any(k < 0 or k >= len(cams) for k in args.view_indices):
        raise ValueError("view_indices must be nonempty, distinct and in range")
    for j, directory in enumerate(args.validation_dirs):
        validation = json.loads((directory / "validation.json").read_text())
        source = Path(validation["views_dir"])
        if j == 0 and source.resolve() != args.views_dir.resolve():
            raise ValueError("first validation must belong to views_dir")
        if json.loads((source / "cams.json").read_text())["cams"] != cams:
            raise ValueError("common regions require identical cameras")
        for k in args.view_indices:
            own = np.asarray(Image.open(args.views_dir / "mask" / f"{k:03d}.png"))
            other = np.asarray(Image.open(source / "mask" / f"{k:03d}.png"))
            if not np.array_equal(own, other):
                raise ValueError("common regions require identical original hole masks")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.log_dir / "config.yaml")
    assert_non_oracle(cfg)
    started = time.time()
    seed_scene_optimization(0)
    torch.cuda.reset_peak_memory_stats()
    from datasets.driving_dataset import DrivingDataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    device = torch.device("cuda")
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=str(args.log_dir / "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    rows = []
    writer = imageio.get_writer(str(args.out_dir / "comparison.mp4"), mode="I", fps=5, macro_block_size=1)
    with torch.no_grad():
        for k in args.view_indices:
            c = cams[k]
            target = np.asarray(Image.open(args.views_dir / "filled" / f"{k:03d}.png").convert("RGB"))
            hole = np.asarray(Image.open(args.views_dir / "mask" / f"{k:03d}.png")) > 127
            ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, c["frame"] - dataset.start_timestep), 1)
            ii, ci = to_device(ii, device), to_device(ci, device)
            camera = torch.tensor(c["c2w"], device=device)
            intrinsic = torch.tensor(c["K"], device=device)
            fit = render_at(trainer, ii, ci, camera, intrinsic, c["hw"])["rgb"].clamp(0, 1)
            fit = (fit.cpu().numpy() * 255).round().astype(np.uint8)
            novel = render_at(trainer, ii, ci, moved_c2w(camera, ViewMove(yaw=args.novel_yaw)), intrinsic, c["hw"])["rgb"].clamp(0, 1)
            novel = (novel.cpu().numpy() * 255).round().astype(np.uint8)
            translated_camera = moved_c2w(camera, ViewMove(right=args.novel_right))
            yaw_camera = moved_c2w(camera, ViewMove(yaw=args.novel_yaw))
            for pose in (translated_camera, yaw_camera):
                if any(np.allclose(pose.cpu().numpy(), x["c2w"], atol=1e-6) for x in cams):
                    raise ValueError("novel camera duplicates a generated camera")
            translated = render_at(trainer, ii, ci, translated_camera, intrinsic, c["hw"])["rgb"].clamp(0, 1)
            translated = (translated.cpu().numpy() * 255).round().astype(np.uint8)
            regions = {"hole": hole}
            if args.validation_dirs:
                regions["accepted"] = hole & (np.load(args.validation_dirs[0] / "confidence" / f"{k:03d}.npy") > 0)
                common = regions["accepted"].copy()
                for directory in args.validation_dirs[1:]:
                    common &= np.load(directory / "confidence" / f"{k:03d}.npy") > 0
                regions["common_accepted"] = common
            row = {"view": k, "frame": c["frame"], "hole_fraction": float(hole.mean()),
                   "generation_hole_sharpness": sharpness(target, hole), "fit_hole_sharpness": sharpness(fit, hole),
                   "regions": {name: fit_metrics(target, fit, mask) for name, mask in regions.items()},
                   "novel_yaw_c2w": yaw_camera.cpu().tolist(), "novel_right_c2w": translated_camera.cpu().tolist()}
            if hole.any():
                error = ((target.astype(np.float32) - fit) / 255) ** 2
                row["fit_to_generation_hole_psnr"] = float(-10 * np.log10(max(float(error[hole].mean()), 1e-12)))
            rows.append(row)
            panels = []
            for label, img in (("Direct generation (synthetic)", target), (f"3D {args.label}, same camera", fit),
                               (f"Unsupervised yaw +{args.novel_yaw:g}", novel),
                               (f"Unsupervised right +{args.novel_right:g} units", translated)):
                image = Image.fromarray(img)
                height = round(image.height * 384 / image.width)
                height += height % 2  # H.264 yuv420p requires an even frame height.
                image = image.resize((384, height), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 22), fill="black")
                draw.text((5, 5), label, fill="white")
                panels.append(np.asarray(image))
            Image.fromarray(np.concatenate(panels, axis=1)).save(args.out_dir / f"{k:03d}.png")
            writer.append_data(np.concatenate(panels, axis=1))
            Image.fromarray(fit).save(args.out_dir / f"{k:03d}_fit.png")
            Image.fromarray(novel).save(args.out_dir / f"{k:03d}_novel.png")
            Image.fromarray(translated).save(args.out_dir / f"{k:03d}_translated.png")
            print(f"[diagnose] {args.label} view {k}: {json.dumps(row['regions'])}", flush=True)
        writer.close()
        # Fixed-world roundtrip is only a reproducibility/inspection check.
        # It is never counted as independent geometric evidence.
        anchor = args.view_indices[len(args.view_indices) // 2]
        c = cams[anchor]
        ii, ci = dataset.full_image_set.get_image(front_image_index(dataset, c["frame"] - dataset.start_timestep), 1)
        ii, ci = to_device(ii, device), to_device(ci, device)
        camera, intrinsic = torch.tensor(c["c2w"], device=device), torch.tensor(c["K"], device=device)
        writer = imageio.get_writer(str(args.out_dir / "local_roundtrip.mp4"), mode="I", fps=10, macro_block_size=1)
        for index, amount in enumerate(np.concatenate([np.linspace(0, 1, 21), np.linspace(1, 0, 21)[1:]])):
            pose = moved_c2w(camera, ViewMove(right=float(amount * args.novel_right), yaw=float(amount * args.novel_yaw)))
            rgb = render_at(trainer, ii, ci, pose, intrinsic, c["hw"])["rgb"].clamp(0, 1)
            rgb = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
            if index == 0:
                first = rgb
            writer.append_data(np.asarray(Image.fromarray(rgb).resize((768, 422), Image.Resampling.LANCZOS)))
        writer.close()
    result = {"log_dir": str(args.log_dir), "views_dir": str(args.views_dir), "rows": rows,
              "novel_yaw": args.novel_yaw, "is_ground_truth_evaluation": False,
              "novel_right_scene_units": args.novel_right,
              "validation_dirs": [str(p) for p in args.validation_dirs],
              "roundtrip_max_rgb_difference": int(np.abs(first.astype(np.int16) - rgb.astype(np.int16)).max()),
              "note": "Sharpness alone is not realism; novel views are qualitative; fixed-world roundtrip only tests reproducibility. Historical E10/E11 input may contain heldout FRONT influence.",
              "dashrecon_commit": commit}
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    params = dict(vars(args))
    for name in ("log_dir", "views_dir", "out_dir"):
        params[name] = str(params[name])
    params["validation_dirs"] = [str(p) for p in args.validation_dirs]
    OmegaConf.save(OmegaConf.create({"diagnostic": params, "source_config": OmegaConf.to_container(cfg)}), args.out_dir / "config.yaml")
    meta = {"dashrecon_commit": commit, "runtime_s": time.time() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3, "generative": True,
            "uses_oracle": False, "oracle_note": "FRONT and generated RGB-D only; historical heldout-frame influence, implementation diagnostic only",
            "seed": 0, "versions": {name: version(name) for name in ("torch", "numpy", "Pillow", "gsplat")}}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[diagnose] {meta['runtime_s']:.1f}s, {meta['peak_vram_gb']:.2f} GB -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
