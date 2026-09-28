"""D-E1 (docs/EXPERIMENTS.md): distribution-level realism of side-camera renders. EVALUATION CODE: uses GT images.

Hidden content may be imagined (AGENTS §1, D16), so matching the real side cameras pixel by pixel is not the goal;
these metrics ask whether the renders look like real street images of this kind and contain plausible things:
    KID (primary; unbiased at a few hundred images) and FID (reported, unreliable at this sample size) between the
        renders and the real undistorted images of the same cameras and frames: torchmetrics with the torch-fidelity
        Inception-v3 features;
    semantic plausibility: SegFormer-B5 Cityscapes (the Phase 4 model) class histogram over all pixels of the renders
        vs of the references, Jensen-Shannon divergence (base 2), and the fractions of the main static classes;
    sharpness: mean variance of the Laplacian of the grey renders over that of the references.
Multi-view consistency metrics (MEt3R, COLMAP-based) are for generated frames; a render of one 3DGS is consistent by
construction, so they are not used here (docs/PLAN_AFTER_E14.md D-E1, corrected).

Inputs: --runs label=dir ..., each dir holding renders/<t:03d>_<cam>.png and refs/<t:03d>_<cam>.png written by
scripts/eval_cross_camera.py --save_renders; all runs must cover the same files, with identical references.
Groups: front_side (cams 1, 2: FRONT_LEFT / FRONT_RIGHT), side (3, 4: SIDE_LEFT / SIDE_RIGHT), all.
Output: --out JSON (per run and group) and a printed table.

Example (main venv):
    .venvs/main/bin/python scripts/eval_realism.py --runs E5c=results/E5c/val056/realism E14=results/E14/val056/model/realism \
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --out results/_diagnostics/realism/val056/summary.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

GROUPS = {"front_side": (1, 2), "side": (3, 4), "all": (1, 2, 3, 4)}
MAIN_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "car"]


class Segmenter:
    """Full Cityscapes class map (19 classes) with the Phase 4 SegFormer-B5, at the image's own aspect ratio."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.proc = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def histogram(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = 32 * round(h * 1536 / w / 32), 1536  # long side 1536 (Phase 4 input width), multiples of 32
        inputs = self.proc(images=Image.fromarray(img), size={"height": th, "width": tw}, return_tensors="pt").to(self.device)
        seg = self.proc.post_process_semantic_segmentation(self.model(**inputs), target_sizes=[(h, w)])[0]
        return np.bincount(seg.cpu().numpy().ravel(), minlength=len(self.labels)).astype(np.float64)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))  # b > 0 wherever a > 0 (b is the mixture)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="label=dir")
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = git_commit()
    assert not os.path.exists(args.out), args.out
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    device = torch.device("cuda")
    runs = dict(r.split("=", 1) for r in args.runs)
    names = sorted(os.listdir(os.path.join(next(iter(runs.values())), "renders")))
    assert names, "no renders"
    ref_hash = None
    for label, d in runs.items():
        assert sorted(os.listdir(os.path.join(d, "renders"))) == names, f"{label} covers other frames"
        h = hashlib.sha256(b"".join(open(os.path.join(d, "refs", n), "rb").read() for n in names[:: max(1, len(names) // 20)])).hexdigest()
        assert ref_hash in (None, h), f"{label} has other references"
        ref_hash = h
    cam_of = {n: int(n.split("_")[1].split(".")[0]) for n in names}
    ref_dir = os.path.join(next(iter(runs.values())), "refs")
    load = lambda p: np.asarray(Image.open(p).convert("RGB"))
    seg = Segmenter(args.seg_model_id, device)
    t0 = time.time()
    ref_imgs = {n: load(os.path.join(ref_dir, n)) for n in names}
    ref_hist = {n: seg.histogram(ref_imgs[n]) for n in names}
    ref_sharp = {n: sharpness(ref_imgs[n]) for n in names}
    results = {}
    for label, d in runs.items():
        imgs = {n: load(os.path.join(d, "renders", n)) for n in names}
        hist = {n: seg.histogram(imgs[n]) for n in names}
        results[label] = {}
        for group, cams in GROUPS.items():
            sel = [n for n in names if cam_of[n] in cams]
            if not sel:
                continue  # e.g. generated frames compared with one camera only (D-C1)
            kid = KernelInceptionDistance(subset_size=min(100, len(sel))).to(device)
            fid = FrechetInceptionDistance().to(device)
            for batch in range(0, len(sel), 32):
                part = sel[batch:batch + 32]
                # Inception-v3 takes 299 x 299 (torch-fidelity resizes to it anyway); both sets get the same resize
                real = torch.from_numpy(np.stack([np.asarray(Image.fromarray(ref_imgs[n]).resize((299, 299), Image.BICUBIC))
                                                  for n in part])).permute(0, 3, 1, 2).to(device)
                fake = torch.from_numpy(np.stack([np.asarray(Image.fromarray(imgs[n]).resize((299, 299), Image.BICUBIC))
                                                  for n in part])).permute(0, 3, 1, 2).to(device)
                kid.update(real, real=True)
                kid.update(fake, real=False)
                fid.update(real, real=True)
                fid.update(fake, real=False)
            kid_mean, kid_std = kid.compute()
            hr, hf = sum(ref_hist[n] for n in sel), sum(hist[n] for n in sel)
            results[label][group] = {
                "images": len(sel), "kid": float(kid_mean), "kid_std": float(kid_std), "fid": float(fid.compute()),
                "semantic_js": js_divergence(hf, hr),
                "class_fraction_render": {c: float(hf[seg.labels.index(c)] / hf.sum()) for c in MAIN_CLASSES},
                "class_fraction_real": {c: float(hr[seg.labels.index(c)] / hr.sum()) for c in MAIN_CLASSES},
                "sharpness_ratio": float(np.mean([sharpness(imgs[n]) for n in sel]) / np.mean([ref_sharp[n] for n in sel]))}
        print(f"[eval_realism] {label}: " + "  ".join(
            f"{g} KID {v['kid']*1000:.1f}e-3 FID {v['fid']:.1f} semJS {v['semantic_js']:.3f} sharp {v['sharpness_ratio']:.2f}"
            for g, v in results[label].items()), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, "runs": runs, "files": len(names), "groups": {g: list(c) for g, c in GROUPS.items()},
                   "seg_model_id": args.seg_model_id, "kid_subset_size": "min(100, images)", "inception_input": "299 x 299 bicubic",
                   "results": results, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)


if __name__ == "__main__":
    main()
