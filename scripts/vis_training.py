"""Phase 6 / Phase 8 web-viewer data: per-scene comparison sheets and a metrics summary.

For each scene and each still frame, a JPEG sheet with rows = lateral offsets (first row = offset 0) and
columns = [ground-truth FRONT image] + experiments; the ground truth exists only in the offset-0 row.
Still frames should be held-out test frames (every 10th, DECISIONS D7) so that row 0 is an interpolation
check. Inputs: results/<exp>/<scene>/{metrics.json, meta.json, renders/frames/<t>_<offset>.jpg}
(scripts/train_gs.py, scripts/train_ggds.py + scripts/render_lateral.py, scripts/postprocess_frames.py).
Output: <out_dir>/<scene>_<t>.jpg and <out_dir>/summary.json. Runs without a full-split evaluation (E5pp) get
full_psnr null; runs whose meta.json says generative are flagged so the viewer can mark them.
With --gen_exp, the first --gen_views target examples of the first and last round of that run
(<run>/gen/round<r>_view<k>.jpg from scripts/train_ggds.py (E6: render | disparity | SDXL target) or
scripts/train_fixer.py (E7: render | Fixer target)) are copied as <scene>_gen_r<r>_v<k>.jpg.
Runs with a cross-camera check (scripts/eval_cross_camera.py) carry its summary as "xcam" (null otherwise).
With --allow_missing, runs that are not finished and rendered yet are listed as missing (blank sheet cells) so
the viewer can show partial results while the batch is still running; without it every run must exist.

Example:
    .venvs/main/bin/python scripts/vis_training.py --results_root results --exps E3 E4 E5 E5pp E6 \
        --frames 50 100 150 --offsets 0 0.5 1 2 --cell_width 400 --gen_exp E6 --gen_views 2 \
        --processed_root data/waymo/processed/validation --out_dir results/_vis
"""
import argparse
import json
import os
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import DEV_SCENES, FRONT_CAM_ID  # noqa: E402

BLANK = (228, 233, 236)
POSTPROCESS_EXPS = ("E5pp", "E5cfx")  # scripts/postprocess_frames.py outputs
GEN_PANELS = {"sdxl": ["render", "mesh disparity (ControlNet input)", "SDXL target"], "fixer": ["render", "Fixer target"]}
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
    parser.add_argument("--allow_missing", action="store_true", help="partial results while training is running")
    parser.add_argument("--gen_exp", default=None, help="generative-distillation run whose targets are shown, e.g. E6 or E7")
    parser.add_argument("--gen_views", type=int, default=2, help="pool views per round to show for --gen_exp")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"exps": args.exps, "gen_exp": args.gen_exp, "frames": args.frames, "offsets": args.offsets, "scenes": {},
               "partial": args.allow_missing, "generated_at": time.strftime("%Y-%m-%d %H:%M"),
               "dashrecon_commit": git_commit()}
    for sc in DEV_SCENES:
        runs, missing = {}, []
        for exp in args.exps:
            run_dir = os.path.join(args.results_root, exp, sc.scene_id)
            done = os.path.exists(os.path.join(run_dir, "meta.json")) and os.path.exists(os.path.join(run_dir, "renders", "render_params.json"))
            if not done:
                assert args.allow_missing, f"{exp} {sc.scene_id} is not finished and rendered (use --allow_missing for partial output)"
                missing.append(exp)
                continue
            with open(os.path.join(run_dir, "metrics.json")) as f:
                m = json.load(f)
            with open(os.path.join(run_dir, "meta.json")) as f:
                meta = json.load(f)
            assert "generative" not in meta or meta["generative"] is True, run_dir
            xcam = None  # per-frame post-processing runs have no 3D model to place side cameras in
            if exp not in POSTPROCESS_EXPS:
                with open(os.path.join(run_dir, "cross_camera", "metrics.json")) as f:
                    summ = json.load(f)["summary"]["all"]
                xcam = {k: summ[k] for k in ("psnr", "ssim", "lpips", "coverage")}
            runs[exp] = {
                "xcam": xcam,
                "test": {k: m["test"][f"image_metrics/test/{k}"] for k in METRIC_KEYS},
                "full_psnr": m["full"]["image_metrics/full/psnr"] if "full" in m else None,
                "generative": "generative" in meta,  # only the Phase 8 scripts write the flag, and always as true
                "runtime_min": meta["runtime_s"] / 60.0,
                "peak_vram_gb": meta["peak_vram_gb"],
                "dashrecon_commit": meta["dashrecon_commit"],
            }
        sheets = []
        for t in args.frames if runs else []:
            gt = Image.open(os.path.join(args.processed_root, f"{sc.scene_idx:03d}", "images", f"{t:03d}_{FRONT_CAM_ID}.jpg"))
            cw = args.cell_width
            ch = round(gt.height * cw / gt.width)
            sheet = Image.new("RGB", (cw * (1 + len(args.exps)), ch * len(args.offsets)), BLANK)
            sheet.paste(gt.convert("RGB").resize((cw, ch), Image.LANCZOS), (0, 0))
            for r, off in enumerate(args.offsets):
                for c, exp in enumerate(args.exps):
                    if exp in missing:
                        continue
                    img = Image.open(os.path.join(args.results_root, exp, sc.scene_id, "renders", "frames", f"{t:03d}_{off:g}.jpg"))
                    sheet.paste(img.convert("RGB").resize((cw, ch), Image.LANCZOS), ((c + 1) * cw, r * ch))
            name = f"{sc.scene_id}_{t:03d}.jpg"
            sheet.save(os.path.join(args.out_dir, name), quality=82)
            sheets.append(name)
        gen_examples = []
        if args.gen_exp is not None and args.gen_exp in runs:
            run_dir = os.path.join(args.results_root, args.gen_exp, sc.scene_id)
            with open(os.path.join(run_dir, "meta.json")) as f:
                meta = json.load(f)
            for rd in (meta["rounds"][0], meta["rounds"][-1]):
                if "t_max" in rd:  # E6: SDXL targets from a fixed pool, noise level annealed per round
                    panels, caption, pool = GEN_PANELS["sdxl"], f"t ≤ {rd['t_max']:.2f}", meta["pool"]
                else:  # E7: Fixer targets from a pool rebuilt every round with a growing lateral range
                    lo, hi = rd["offset_range"]
                    panels, caption, pool = GEN_PANELS["fixer"], f"|offset| {lo:.2f}–{hi:.2f}", rd["pool"]
                summary["gen_panels"] = panels
                for k in range(args.gen_views):
                    name = f"{sc.scene_id}_gen_r{rd['round']}_v{k}.jpg"
                    shutil.copyfile(os.path.join(run_dir, "gen", f"round{rd['round']}_view{k}.jpg"), os.path.join(args.out_dir, name))
                    view = pool[k]
                    gen_examples.append({"file": name, "round": rd["round"], "step": rd["step"], "caption": caption,
                                         "frame": view["frame"], "offset": view["offset"], "yaw": view["yaw"]})
        summary["scenes"][sc.scene_id] = {"runs": runs, "sheets": sheets, "missing": missing, "gen_examples": gen_examples}
        print(f"[vis_training] {sc.scene_id}: " + " | ".join(
            f"{e} test PSNR {r['test']['psnr']:.2f}" for e, r in runs.items()) + (f" | missing {missing}" if missing else ""), flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
