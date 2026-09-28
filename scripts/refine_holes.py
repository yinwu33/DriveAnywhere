"""E14: the holes of a generated trajectory, with the sky decided by the generated frames (runs in .venvs/masks).

render_views.py counts every low-opacity pixel above sky_elev_deg of elevation as sky, never as a hole
(dashrecon.gen.views.hole_mask). With a view gate (E14) the FRONT-only junk beside the road disappears and leaves
low opacity there, so the houses and trees that belong in those pixels would be treated as sky and never distilled
(val056 E5c yaw 90 smoke test, docs/EXPERIMENTS.md E14). Here the generator decides: SegFormer-B5 Cityscapes
(the Phase 4 backend, sky = class 10) on each filled frame, and
    hole = (render hole | rendered opacity < --sky_alpha) & ~generated sky.
Pixels the generator painted as sky stay with the Sky model; everything else it painted where the render was empty or
unobserved becomes a hole to distil.

In --views_dir: the render's mask/ moves to mask_render/ (it must not exist yet), the new holes go to mask/, the
generated sky to sky_gen/, and holes.json records per-frame fractions, parameters and the commit.

Example (masks venv):
    .venvs/masks/bin/python scripts/refine_holes.py --views_dir results/E14/val056/views/r0 \
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_hw 768 1344 --sky_alpha 0.5
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.masks.segformer import SegformerSkyRoadBackend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--seg_input_hw", type=int, nargs=2, required=True)
    parser.add_argument("--sky_alpha", type=float, required=True)
    args = parser.parse_args()
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before refining experiment inputs"
    vd = args.views_dir
    with open(os.path.join(vd, "cams.json")) as f:
        n = len(json.load(f)["cams"])
    assert not os.path.exists(os.path.join(vd, "mask_render")), f"{vd} was refined already"
    shutil.move(os.path.join(vd, "mask"), os.path.join(vd, "mask_render"))
    os.makedirs(os.path.join(vd, "mask"))
    os.makedirs(os.path.join(vd, "sky_gen"))
    backend = SegformerSkyRoadBackend(args.seg_model_id, tuple(args.seg_input_hw))
    t0, rows = time.time(), []
    for k in range(n):
        filled = Image.open(os.path.join(vd, "filled", f"{k:03d}.png")).convert("RGB")
        sky = backend.predict(filled)[0]["sky"]
        opacity = np.load(os.path.join(vd, "opacity", f"{k:03d}.npy"))[..., 0]
        old = np.asarray(Image.open(os.path.join(vd, "mask_render", f"{k:03d}.png"))) > 127
        assert sky.shape == opacity.shape == old.shape, (sky.shape, opacity.shape, old.shape)
        hole = (old | (opacity < args.sky_alpha)) & ~sky
        Image.fromarray((hole * 255).astype(np.uint8)).save(os.path.join(vd, "mask", f"{k:03d}.png"))
        Image.fromarray((sky * 255).astype(np.uint8)).save(os.path.join(vd, "sky_gen", f"{k:03d}.png"))
        rows.append({"k": k, "render_hole": float(old.mean()), "hole": float(hole.mean()), "generated_sky": float(sky.mean()),
                     "render_hole_now_sky": float((old & sky).mean()), "low_opacity_added": float((hole & ~old).mean())})
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "k"}
    with open(os.path.join(vd, "holes.json"), "w") as f:
        json.dump({"params": vars(args), "segmentation": backend.meta(), "mean": summary, "frames": rows,
                   "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[refine_holes] {vd}: hole {summary['render_hole']:.3f} -> {summary['hole']:.3f} "
          f"(+{summary['low_opacity_added']:.3f} low opacity, -{summary['render_hole_now_sky']:.3f} generated sky)", flush=True)


if __name__ == "__main__":
    main()
