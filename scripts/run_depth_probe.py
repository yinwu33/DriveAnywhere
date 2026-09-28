"""Depth diagnostic B (DECISIONS W): does a joint multi-view depth let more generated pixels pass validation?

Same generated frames, same validation, only the depth of the filled frames changes. For each --variants entry
<backend>:<alignment> (backend moge | mapanything, alignment scene_global | boundary; scripts/depth_views.py) this
    1. makes <out_dir>/<variant>/views with links to the source trajectory's cams.json, rgb, depth, count, mask and
       filled (nothing in the source is written);
    2. runs scripts/depth_views.py there (MoGe-2 in .venvs/gen3c, MapAnything in .venvs/mapanything, which gets the
       trajectory's intrinsics and poses and so predicts one depth for all frames jointly);
    3. validates with scripts/validate_generated_views.py using exactly the parameters of --reference_validation
       (the run's own validation of these frames), into <out_dir>/<variant>/validation;
    4. measures the aligned depth against the rendered depth where >= 2 training views saw the surface (count >= 2):
       median |log(filled / rendered)| per frame, the part of the frame the generator should have kept.
Writes <out_dir>/summary.json (per variant: accepted / unknown / conflict fraction of hole pixels overall and per
|yaw| bucket, agreement with observed depth, depth.json alignment and backend notes) and meta.json. The accepted
fraction measures self-consistency of generated RGB-D, not correctness of the hidden content.

Example (main venv):
    .venvs/main/bin/python scripts/run_depth_probe.py \
        --views_dir results/E13/val056/surround_20260928_history/views/r0 \
        --reference_validation results/E13/val056/surround_20260928_history/validation/r0 \
        --variants moge:scene_global moge:boundary mapanything:scene_global mapanything:boundary \
        --out_dir results/_diagnostics/depth_probe/val056/E13_r0
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.provenance import git_commit  # noqa: E402
from scripts.validate_generated_views import validate  # noqa: E402

LINKS = ["cams.json", "rgb", "depth", "count", "mask", "filled"]
BACKENDS = {"moge": (".venvs/gen3c/bin/python", "Ruicheng/moge-2-vitl-normal"),
            "mapanything": (".venvs/mapanything/bin/python", "facebook/map-anything")}
YAW_BUCKETS = [(0, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5), (157.5, 180.01)]


def observed_agreement(views: Path, n: int) -> list:
    """Per frame median |log(filled depth / rendered depth)| over pixels >= 2 training views saw (None if < 1000)."""
    out = []
    for k in range(n):
        d = np.load(views / "filled_depth" / f"{k:03d}.npy").astype(np.float64)
        r = np.load(views / "depth" / f"{k:03d}.npy").astype(np.float64)
        ok = (np.load(views / "count" / f"{k:03d}.npy") >= 2) & (d > 0) & (r > 0)
        out.append(float(np.median(np.abs(np.log(d[ok] / r[ok])))) if ok.sum() >= 1000 else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", type=Path, required=True)
    parser.add_argument("--reference_validation", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    commit = git_commit()
    assert not commit.endswith("-dirty"), "commit code before a diagnostic run"
    ref = json.loads((args.reference_validation / "validation.json").read_text())
    assert Path(ref["views_dir"]).resolve() == args.views_dir.resolve(), (ref["views_dir"], args.views_dir)
    p = ref["params"]
    if "memory_dirs" in p:  # absent in validations written before generation memory existed (E12)
        assert p["memory_dirs"] == [], "validation with generation memory is not reproduced here"
    params = {"offsets": p["offsets"], "depth_tol": p["depth_tol"], "rgb_tol": p["rgb_tol"],
              "min_support": p["min_support"], "max_conflict_fraction": p["max_conflict_fraction"],
              "reference_policy": p["reference_policy"], "min_baseline": p["min_baseline_scene_units"],
              "max_references": p["max_references"], "min_parallax_degrees": p["min_parallax_degrees"]}
    cams = json.loads((args.views_dir / "cams.json").read_text())["cams"]
    yaws = np.array([abs(c["yaw"]) for c in cams])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    summary = {}
    for variant in args.variants:
        backend, alignment = variant.split(":")
        python, model_id = BACKENDS[backend]
        vdir = args.out_dir / variant.replace(":", "_")
        views = vdir / "views"
        views.mkdir(parents=True)
        for name in LINKS:
            os.symlink((args.views_dir / name).resolve(), views / name)
        t0 = time.time()
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"}
        with open(vdir / "depth.log", "w") as log:
            subprocess.run([python, "-u", "scripts/depth_views.py", "--views_dir", str(views), "--backend", backend,
                            "--model_id", model_id, "--alignment_policy", alignment],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        depth_s = time.time() - t0
        result = validate(views_dir=views, out_dir=vdir / "validation", **params)
        rows = result["frames"]
        hole = np.array([r["hole_pixels"] for r in rows], dtype=np.float64)
        acc = np.array([r["accepted_pixels"] for r in rows], dtype=np.float64)
        unk = np.array([r["unknown_pixels"] for r in rows], dtype=np.float64)
        con = np.array([r["conflict_pixels"] for r in rows], dtype=np.float64)
        buckets = []
        for lo, hi in YAW_BUCKETS:
            m = (yaws >= lo) & (yaws < hi)
            if hole[m].sum() == 0:
                continue  # no view (or no hole) at these angles
            buckets.append({"abs_yaw": [lo, min(hi, 180.0)], "views": int(m.sum()),
                            "accepted": float(acc[m].sum() / hole[m].sum()),
                            "unknown": float(unk[m].sum() / hole[m].sum())})
        agree = observed_agreement(views, len(cams))
        depth_json = json.loads((views / "depth.json").read_text())
        summary[variant] = {
            "accepted": result["accepted_fraction"], "unknown": result["unknown_fraction"],
            "conflict": float(con.sum() / hole.sum()), "by_abs_yaw": buckets,
            "observed_log_error_median": float(np.median([a for a in agree if a is not None])),
            "observed_log_error_per_frame": agree,
            "backend_meta": depth_json["backend_meta"],
            "alignment": {k: v for k, v in depth_json["alignment"].items() if k != "frames"}
            if alignment == "scene_global" else "per-hole ring / frame scale (depth.json frames)",
            "depth_runtime_s": depth_s}
        print(f"[depth_probe] {variant}: accepted {result['accepted_fraction']:.4f}, unknown {result['unknown_fraction']:.4f}, "
              f"conflict {summary[variant]['conflict']:.4f}, observed |log| {summary[variant]['observed_log_error_median']:.3f}",
              flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "meta.json").write_text(json.dumps({
        "kind": "depth diagnostic B: same generated frames and validation, different depth backend / alignment",
        "views_dir": str(args.views_dir), "reference_validation": str(args.reference_validation),
        "validation_params": params, "variants": args.variants, "generative": True,
        "uses_oracle": False, "oracle_information": "none; generated frames, estimated cameras and renders only",
        "dashrecon_commit": commit, "runtime_s": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
