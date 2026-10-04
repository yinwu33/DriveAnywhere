"""D-R1 (docs/EXPERIMENTS.md): where the error of a run's FRONT_LEFT / FRONT_RIGHT renders comes from. EVALUATION CODE:
reads GT images.

Inputs: --runs label=dir, each dir with renders/<t>_<cam>.png and refs/<t>_<cam>.png on the same grid
(scripts/eval_cross_camera.py --save_renders), and --unseen_dir (the E5c unseen masks of eval_cross_camera --save_unseen,
<t>_<cam>.png on the GT grid). For every image the real reference is segmented with SegFormer-B5 Cityscapes (the Phase 4
model) and every pixel gets a class and a region (seen / unseen). Per run, per region and per class: the pixel share
and the mean absolute colour error (after the per-channel affine colour fit on the seen pixels, as eval_cross_camera),
and each cell's share of the total error. The real image is the only reference, so this measures fidelity, not realism:
it says which content (near road, parked cars, buildings, vegetation, sky) dominates the remaining error.

Output: --out JSON and a printed table (cells with >= 2 % of the error).

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/diagnose_error_classes.py --runs E35=results/E35/val056/model/realism_fs \\
        --unseen_dir results/E5c/val056/cross_camera_p9/unseen --cams 1 2 \\
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --out results/_diagnostics/error_classes/val056.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashrecon.provenance import git_commit  # noqa: E402
from eval_cross_camera import colour_fit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="label=dir")
    parser.add_argument("--unseen_dir", required=True)
    parser.add_argument("--cams", type=int, nargs="+", required=True)
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = git_commit()
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = torch.device("cuda")
    proc = SegformerImageProcessor.from_pretrained(args.seg_model_id)
    seg = SegformerForSemanticSegmentation.from_pretrained(args.seg_model_id).to(device).eval()
    labels = [seg.config.id2label[i] for i in range(seg.config.num_labels)]
    runs = dict(r.split("=", 1) for r in args.runs)
    first = next(iter(runs.values()))
    names = sorted(n for n in os.listdir(os.path.join(first, "renders")) if int(n.split("_")[1].split(".")[0]) in args.cams
                   and os.path.exists(os.path.join(args.unseen_dir, n)))  # the unseen masks exist every 5th frame only
    assert names, "no renders of these cameras with an unseen mask"
    load = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0  # noqa: E731
    classes, unseen = {}, {}
    with torch.no_grad():
        for n in names:
            ref = load(os.path.join(first, "refs", n))
            h, w = ref.shape[:2]
            inputs = proc(images=Image.fromarray((ref * 255).astype(np.uint8)), size={"height": 32 * round(h * 1536 / w / 32), "width": 1536},
                          return_tensors="pt").to(device)
            classes[n] = proc.post_process_semantic_segmentation(seg(**inputs), target_sizes=[(h, w)])[0].cpu().numpy()
            unseen[n] = np.asarray(Image.open(os.path.join(args.unseen_dir, n)).resize((w, h), Image.NEAREST)) > 127
    out = {}
    for label, d in runs.items():
        err = np.zeros((2, len(labels)))
        cnt = np.zeros((2, len(labels)))
        for n in names:
            ref, ren = load(os.path.join(d, "refs", n)), load(os.path.join(d, "renders", n))
            assert np.array_equal(ref, load(os.path.join(first, "refs", n))), f"{label}: other reference for {n}"
            fitted = colour_fit(ren, ref, ~unseen[n])
            e = np.abs(fitted - ref).mean(-1)
            for region, m in ((0, ~unseen[n]), (1, unseen[n])):
                np.add.at(err[region], classes[n][m], e[m])
                np.add.at(cnt[region], classes[n][m], 1)
        total = err.sum()
        cells = []
        for region, rname in ((0, "seen"), (1, "unseen")):
            for c, cname in enumerate(labels):
                if cnt[region, c] == 0:
                    continue
                cells.append({"region": rname, "class": cname, "pixel_share": float(cnt[region, c] / cnt.sum()),
                              "mean_abs_err": float(err[region, c] / cnt[region, c]), "error_share": float(err[region, c] / total)})
        cells.sort(key=lambda x: -x["error_share"])
        out[label] = {"mean_abs_err": float(total / cnt.sum()), "unseen_pixel_share": float(cnt[1].sum() / cnt.sum()),
                      "unseen_error_share": float(err[1].sum() / total), "cells": cells}
        print(f"[diagnose_error_classes] {label}: mean |err| {out[label]['mean_abs_err']:.4f}; unseen {out[label]['unseen_pixel_share']:.0%} of the "
              f"pixels, {out[label]['unseen_error_share']:.0%} of the error", flush=True)
        for c in cells:
            if c["error_share"] >= 0.02:
                print(f"    {c['region']:6s} {c['class']:14s} pixels {c['pixel_share']:6.1%}  |err| {c['mean_abs_err']:.3f}  error {c['error_share']:6.1%}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, "runs": runs, "cams": args.cams, "images": len(names), "unseen_dir": args.unseen_dir,
                   "seg_model_id": args.seg_model_id, "results": out, "dashrecon_commit": commit}, f, indent=2)


if __name__ == "__main__":
    main()
