"""Separate direct generation from 3D fit at identical cameras; also render an
unsupervised +5 degree yaw. This is a diagnostic, not ground-truth NVS scoring.
All stills use the same resolution and hole masks, without per-image colour fitting.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.novel import build_trainer, to_device
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at
from dashrecon.provenance import git_commit


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
    args = parser.parse_args()
    if args.novel_yaw == 0:
        raise ValueError("novel_yaw must differ from the supervised camera")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.log_dir / "config.yaml")
    from datasets.driving_dataset import DrivingDataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    device = torch.device("cuda")
    trainer = build_trainer(cfg, dataset, device)
    trainer.resume_from_checkpoint(ckpt_path=str(args.log_dir / "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    cams = json.loads((args.views_dir / "cams.json").read_text())["cams"]
    rows = []
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
            row = {"view": k, "frame": c["frame"], "hole_fraction": float(hole.mean()),
                   "generation_hole_sharpness": sharpness(target, hole), "fit_hole_sharpness": sharpness(fit, hole)}
            if hole.any():
                error = ((target.astype(np.float32) - fit) / 255) ** 2
                row["fit_to_generation_hole_psnr"] = float(-10 * np.log10(max(float(error[hole].mean()), 1e-12)))
            rows.append(row)
            panels = []
            for label, img in (("Direct generation (synthetic)", target), ("3D fit, identical camera", fit),
                               (f"3D at untrained yaw +{args.novel_yaw:g}", novel)):
                image = Image.fromarray(img)
                image = image.resize((640, round(image.height * 640 / image.width)), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 22), fill="black")
                draw.text((5, 5), label, fill="white")
                panels.append(np.asarray(image))
            Image.fromarray(np.concatenate(panels, axis=1)).save(args.out_dir / f"{k:03d}.png")
            Image.fromarray(fit).save(args.out_dir / f"{k:03d}_fit.png")
            Image.fromarray(novel).save(args.out_dir / f"{k:03d}_novel.png")
    result = {"log_dir": str(args.log_dir), "views_dir": str(args.views_dir), "rows": rows,
              "novel_yaw": args.novel_yaw, "is_ground_truth_evaluation": False,
              "note": "Sharpness alone cannot distinguish texture from artifacts; inspect novel views too.",
              "dashrecon_commit": git_commit()}
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
