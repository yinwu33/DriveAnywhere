"""D-G3 (docs/EXPERIMENTS.md): single-image focal-length priors against the self-calibration, to choose the focal of E46.
DIAGNOSTIC: reads GT intrinsics (evaluation side) only to score the estimators; runs in .venvs/sam3d (MoGe v1, GeoCalib).

GLOMAP's self-calibrated focal is 14 % short on val056 (1787 vs 2078 px; D-G1: 6 % on average). Under forward motion
focal is weakly observable, and a short focal leaves depth unchanged but stretches the scene sideways and vertically by
f_true / f (D-F2: renders 1.2-1.35 x deeper than trajectory-scaled LiDAR, the fitted ground 0.67 m low). Per scene,
--frames original FRONT frames (distorted, as recorded) spread over the drive go through
    GeoCalib (cvg/GeoCalib; camera_model pinhole and simple_radial): focal (and k1);
    MoGe v1 (Ruicheng/moge-vitl): focal from the predicted normalised intrinsics;
the median per estimator is compared with GT and with GLOMAP (calib-glomap/camera.json).

Output: --out json (per scene and estimator: median, IQR, relative error) and a printed table.

Example (sam3d venv):
    CUDA_HOME=$HOME/cuda-12.6 .venvs/sam3d/bin/python scripts/diagnose_focal_priors.py --scenes val039 val041 val056 val087 val094 \\
        --processed_root data/waymo/processed/validation --dashrecon_root data/dashrecon --frames 30 \\
        --out data/dashrecon/diagnostics/dg3_focal_priors.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

SCENE_IDX = {"val039": 39, "val041": 41, "val056": 56, "val087": 87, "val094": 94}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--dashrecon_root", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = git_commit()
    dev = torch.device("cuda")
    from geocalib import GeoCalib
    from moge.model.v1 import MoGeModel

    geo = GeoCalib().to(dev)
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(dev).eval()
    results = {}
    t0 = time.time()
    for sc in args.scenes:
        idx = SCENE_IDX[sc]
        img_dir = os.path.join(args.processed_root, f"{idx:03d}", "images")
        names = sorted(n for n in os.listdir(img_dir) if n.endswith("_0.jpg"))
        pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, args.frames).round()]
        est = {"geocalib_pinhole": [], "geocalib_radial": [], "geocalib_radial_k1": [], "moge_v1": []}
        for n in pick:
            path = os.path.join(img_dir, n)
            with torch.no_grad():
                img = geo.load_image(path).to(dev)
                for model_name in ("pinhole", "simple_radial"):
                    res = geo.calibrate(img, camera_model=model_name)
                    cam = res["camera"]
                    key = "geocalib_pinhole" if model_name == "pinhole" else "geocalib_radial"
                    est[key].append(float(cam.f[0, 0]))
                    if model_name == "simple_radial":
                        est["geocalib_radial_k1"].append(float(cam.dist[0, 0]))
                pil = Image.open(path).convert("RGB")
                w, h = pil.size
                x = torch.tensor(np.asarray(pil), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
                out = moge.infer(x)
                est["moge_v1"].append(float(out["intrinsics"][0, 0]) * w)
        gt = float(np.loadtxt(os.path.join(args.processed_root, f"{idx:03d}", "intrinsics", "0.txt"))[0])
        cam_json = json.load(open(os.path.join(args.dashrecon_root, sc, "calib-glomap", "camera.json")))
        glomap = float(cam_json["K"][0][0])
        row = {"gt_fx": gt, "glomap": {"fx": glomap, "rel_err": glomap / gt - 1}}
        for k, v in est.items():
            v = np.array(v)
            if k.endswith("_k1"):
                row[k] = {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
            else:
                row[k] = {"fx": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                          "rel_err": float(np.median(v)) / gt - 1}
        results[sc] = row
        print(f"[diagnose_focal_priors] {sc}: GT {gt:.0f}  GLOMAP {glomap:.0f} ({row['glomap']['rel_err']:+.1%})  "
              + "  ".join(f"{k} {row[k]['fx']:.0f} ({row[k]['rel_err']:+.1%})" for k in ("geocalib_pinhole", "geocalib_radial", "moge_v1"))
              + f"  k1 {row['geocalib_radial_k1']['median']:+.3f} ({time.time() - t0:.0f} s)", flush=True)
    summary = {k: float(np.mean([abs(r[k]["rel_err"]) for r in results.values()])) for k in ("glomap", "geocalib_pinhole", "geocalib_radial", "moge_v1")}
    print("[diagnose_focal_priors] mean |rel err|: " + "  ".join(f"{k} {v:.1%}" for k, v in summary.items()), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reads_gt": True, "params": vars(args), "results": results, "mean_abs_rel_err": summary, "dashrecon_commit": commit}, f, indent=1)


if __name__ == "__main__":
    main()
