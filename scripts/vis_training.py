"""Phase 6 web-viewer data: per-scene comparison sheets and a metrics summary.

For each scene and each still frame, a JPEG sheet with rows = lateral offsets (first row = offset 0) and
columns = [ground-truth FRONT image] + experiments; the ground truth exists only in the offset-0 row.
Still frames should be held-out test frames (every 10th, DECISIONS D7) so that row 0 is an interpolation
check. Inputs: results/<exp>/<scene>/{metrics.json, meta.json, renders/frames/<t>_<offset>.jpg}
(scripts/train_gs.py, scripts/render_lateral.py). Output: <out_dir>/<scene>_<t>.jpg and <out_dir>/summary.json.

Example:
    .venvs/main/bin/python scripts/vis_training.py --results_root results --exps E3 E4 E5 \
        --frames 50 100 150 --offsets 0 0.5 1 2 --cell_width 480 \
        --processed_root data/waymo/processed/validation --out_dir results/_vis
"""
import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import DEV_SCENES, FRONT_CAM_ID  # noqa: E402

BLANK = (228, 233, 236)
METRIC_KEYS = ("psnr", "ssim", "lpips", "occupied_psnr", "masked_psnr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--exps", nargs="+", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--offsets", type=float, nargs="+", required=True)
    parser.add_argument("--cell_width", type=int, required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"exps": args.exps, "frames": args.frames, "offsets": args.offsets, "scenes": {},
               "dashrecon_commit": git_commit()}
    for sc in DEV_SCENES:
        runs = {}
        for exp in args.exps:
            run_dir = os.path.join(args.results_root, exp, sc.scene_id)
            with open(os.path.join(run_dir, "metrics.json")) as f:
                m = json.load(f)
            with open(os.path.join(run_dir, "meta.json")) as f:
                meta = json.load(f)
            runs[exp] = {
                "test": {k: m["test"][f"image_metrics/test/{k}"] for k in METRIC_KEYS},
                "full_psnr": m["full"]["image_metrics/full/psnr"],
                "runtime_min": meta["runtime_s"] / 60.0,
                "peak_vram_gb": meta["peak_vram_gb"],
                "dashrecon_commit": meta["dashrecon_commit"],
            }
        sheets = []
        for t in args.frames:
            gt = Image.open(os.path.join(args.processed_root, f"{sc.scene_idx:03d}", "images", f"{t:03d}_{FRONT_CAM_ID}.jpg"))
            cw = args.cell_width
            ch = round(gt.height * cw / gt.width)
            sheet = Image.new("RGB", (cw * (1 + len(args.exps)), ch * len(args.offsets)), BLANK)
            sheet.paste(gt.convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 0))
            for r, off in enumerate(args.offsets):
                for c, exp in enumerate(args.exps):
                    img = Image.open(os.path.join(args.results_root, exp, sc.scene_id, "renders", "frames", f"{t:03d}_{off:g}.jpg"))
                    sheet.paste(img.convert("RGB").resize((cw, ch), Image.LANCZOS), ((c + 1) * cw, r * ch))
            name = f"{sc.scene_id}_{t:03d}.jpg"
            sheet.save(os.path.join(args.out_dir, name), quality=82)
            sheets.append(name)
        summary["scenes"][sc.scene_id] = {"runs": runs, "sheets": sheets}
        print(f"[vis_training] {sc.scene_id}: " + " | ".join(
            f"{e} test PSNR {r['test']['psnr']:.2f}" for e, r in runs.items()), flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
