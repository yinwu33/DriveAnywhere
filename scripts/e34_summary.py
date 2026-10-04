"""E34 (docs/E34_DATA.md): statistics of the finished segments for the data report.

Reads data/e34/segments.json, every <pairs_root>/<sid>/meta.json and the evaluation-side GT diagnostics
<diag_dir>/<sid>_calib_gt.json (scripts/e34_diag_calib.py; GT used for reporting only), and writes --out (JSON):
    segments        selected / finished / accepted / rejected (reasons grouped), per location
    pairs           counts per type and camera, mean input coverage per camera
    timing_s        median and total per stage
    registration    median registered fraction per camera, reprojection error, GLOMAP 3D points
    calibration_vs_gt   medians over the accepted segments: focal ratio per camera, target -> FRONT rotation error,
                        target position error after the Sim(3) of the FRONT trajectory, FRONT trajectory RMS, scale
    disk_bytes      the pairs directories
Prints a short summary.

Example (any venv):
    python scripts/e34_summary.py --segments data/e34/segments.json --pairs_root data/e34/pairs --diag_dir data/e34/diag \
        --out data/e34/summary.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def median_of(rows: list, key) -> float | None:
    vals = [key(r) for r in rows]
    return float(np.median(vals)) if vals else None


def dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--diag_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    selected = io.read_json(args.segments)["selected"]
    metas, diags = {}, {}
    for s in selected:
        f = os.path.join(args.pairs_root, s["scene_id"], "meta.json")
        if os.path.exists(f):
            metas[s["scene_id"]] = io.read_json(f)
        g = os.path.join(args.diag_dir, f"{s['scene_id']}_calib_gt.json")
        if os.path.exists(g):
            diags[s["scene_id"]] = io.read_json(g)
    acc = {k: m for k, m in metas.items() if m["accepted"]}
    rej = {k: m for k, m in metas.items() if not m["accepted"]}
    reasons = {}
    for m in rej.values():
        for r in m["reasons"]:
            kind = "FRONT undistortion border" if "undistortion" in r else r.split(":")[0] + " registration"
            reasons[kind] = reasons.get(kind, 0) + 1
    loc_of = {s["scene_id"]: s["location"] for s in selected}
    per_loc = {}
    for k, m in metas.items():
        d = per_loc.setdefault(loc_of[k], {"finished": 0, "accepted": 0})
        d["finished"] += 1
        d["accepted"] += int(m["accepted"])
    pairs, coverage = {}, {}
    for m in acc.values():
        for typ, counts in m["pair_counts"].items():
            if typ == "targets":
                continue
            for cam, n in counts.items():
                pairs.setdefault(typ, {}).setdefault(cam, 0)
                pairs[typ][cam] += n
            for c, v in m[typ]["mean_coverage"].items():
                coverage.setdefault(typ, {}).setdefault(CAM_NAMES[int(c)], []).append(v)
    timing = {}
    for m in metas.values():
        for stage, sec in m["timing_s"].items():
            timing.setdefault(stage, []).append(sec)
    cams = [CAM_NAMES[c] for c in range(5)]
    reg = {"registered_frac_median": {}, "reprojection_px_median": {}}
    for c in range(5):
        rows = [m["registration"]["per_camera"][str(c)] for m in metas.values()]
        reg["registered_frac_median"][CAM_NAMES[c]] = median_of(rows, lambda r: r["registered"] / r["images"])
        reg["reprojection_px_median"][CAM_NAMES[c]] = median_of([r for r in rows if r["registered"]], lambda r: r["mean_reprojection_error_px"])
    reg["points3D_median"] = median_of(list(metas.values()), lambda m: m["registration"]["points3D"])
    reg["calib_runtime_s_median"] = median_of(list(metas.values()), lambda m: m["timing_s"]["calib"])
    dg = [diags[k] for k in acc if k in diags]
    vs_gt = {"segments": len(dg),
             "front_rms_m_median": median_of(dg, lambda d: d["front_rms_m"]),
             "front_path_m_median": median_of(dg, lambda d: d["front_path_m"]),
             "units_per_gt_m_median": median_of(dg, lambda d: 1.0 / d["sim3_scale_gt_m_per_unit"]),
             "focal_ratio_median": {CAM_NAMES[c]: median_of([d for d in dg if str(c) in d["cameras"]], lambda d: d["cameras"][str(c)]["focal_ratio"]) for c in range(5)},
             "rig_rot_err_deg_median": {CAM_NAMES[c]: median_of([d for d in dg if str(c) in d["cameras"]], lambda d: d["cameras"][str(c)]["rig_rot_err_deg_median"]) for c in range(1, 5)},
             "pos_err_m_median": {CAM_NAMES[c]: median_of([d for d in dg if str(c) in d["cameras"]], lambda d: d["cameras"][str(c)]["pos_err_m_median"]) for c in range(1, 5)}}
    out = {
        "segments": {"selected": len(selected), "finished": len(metas), "accepted": len(acc), "rejected": len(rej),
                     "rejection_reasons": reasons, "rejected_ids": {k: m["reasons"] for k, m in rej.items()},
                     "per_location": per_loc, "gs_segments": sum(1 for m in acc.values() if "gs" in m["pair_counts"]),
                     "gs_num_iters": sorted({m["gs"]["num_iters"] for m in acc.values() if "gs" in m})},
        "pairs": {"counts": pairs, "total": {typ: sum(v.values()) for typ, v in pairs.items()},
                  "mean_coverage": {typ: {cam: float(np.mean(v)) for cam, v in d.items()} for typ, d in coverage.items()}},
        "timing_s": {k: {"segments": len(v), "median": float(np.median(v)), "total": float(np.sum(v))} for k, v in timing.items()},
        "registration": reg, "cameras": cams, "calibration_vs_gt": vs_gt,
        "disk_bytes": {"pairs": dir_bytes(args.pairs_root)},
        "dashrecon_commit": git_commit(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[e34_summary] {len(metas)} finished of {len(selected)}: {len(acc)} accepted, {len(rej)} rejected {reasons}; "
          f"pairs {out['pairs']['total']}; gs segments {out['segments']['gs_segments']}; pairs dir "
          f"{out['disk_bytes']['pairs'] / 1e9:.1f} GB -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
