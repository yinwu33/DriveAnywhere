"""Evaluation-side Phase 3 diagnostics against Waymo GT (reads GT ego poses and FRONT calibration).

This is evaluation code (AGENTS.md section 2: GT may only be read by evaluation code). It is a quick
diagnostic until the Phase 1 tools exist, not the Sim(3) ATE/RPE protocol. Per scene and per run:
    fx_mean / fx_std      estimated FRONT focal length (original 1920x1280 grid)
    chord_ratio           |c_last - c_first| of the estimate divided by the same GT distance
    path_over_chord       estimated path length / its own chord (GT is ~1.00; higher = jitter)
plus runtime and peak VRAM from meta.json.

Example:
    .venvs/mapanything/bin/python scripts/diagnose_pose.py \
        --processed_root data/waymo/processed/validation --tag pose-mapanything_depth-mapanything \
        --run cc-by-nc=data/dashrecon --run apache=data/dashrecon/_checkpoint_compare/map-anything-apache \
        --primary cc-by-nc --out data/dashrecon/diagnostics/phase3_pose.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import DEV_SCENES, FRONT_CAM_ID  # noqa: E402


def trajectory_stats(centers: np.ndarray) -> tuple[float, float]:
    """(chord length, path length) of an (N,3) trajectory."""
    chord = float(np.linalg.norm(centers[-1] - centers[0]))
    path = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    return chord, path


def gt_stats(scene_dir: str, frames: np.ndarray) -> dict:
    """GT FRONT focal length and ego trajectory statistics for the given frames."""
    ego = np.stack([np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{t:03d}.txt")) for t in frames])
    chord, path = trajectory_stats(ego[:, :3, 3])
    fx = float(np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{FRONT_CAM_ID}.txt"))[0])
    return {"fx": fx, "chord_m": chord, "path_over_chord": path / chord}


def run_stats(out_dir: str, gt_chord: float) -> dict:
    """Diagnostics of one estimated run."""
    poses = io.read_poses(out_dir)
    k = io.read_intrinsics(out_dir)
    meta = io.read_meta(out_dir)
    chord, path = trajectory_stats(poses[:, :3, 3])
    fx = k[..., 0, 0]
    return {
        "model_id": meta["model_id"],
        "dashrecon_commit": meta["dashrecon_commit"],
        "fx_mean": float(fx.mean()),
        "fx_std": float(fx.std()),
        "chord_ratio": chord / gt_chord,
        "path_over_chord": path / chord,
        "runtime_s": meta["runtime_s"],
        "peak_vram_gb": meta["peak_vram_gb"],
        "valid_depth": meta["valid_depth_fraction"],
        "depth_hw": meta["depth_grid"]["depth_hw"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--tag", required=True, help="backend tag directory name")
    parser.add_argument("--run", action="append", required=True, help="LABEL=OUT_ROOT, repeatable")
    parser.add_argument("--primary", required=True, help="label of the run shown in the viewer")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = dict(r.split("=", 1) for r in args.run)
    assert len(runs) == len(args.run), "duplicate run labels"
    assert args.primary in runs, f"--primary {args.primary} not in {list(runs)}"

    scenes = []
    for sc in DEV_SCENES:
        primary_dir = io.scene_dir(runs[args.primary], sc.scene_id, args.tag)
        frames = io.read_frames(primary_dir)
        gt = gt_stats(os.path.join(args.processed_root, f"{sc.scene_idx:03d}"), frames)
        per_run = {}
        for label, root in runs.items():
            out_dir = io.scene_dir(root, sc.scene_id, args.tag)
            np.testing.assert_array_equal(io.read_frames(out_dir), frames)
            per_run[label] = run_stats(out_dir, gt["chord_m"])
        scenes.append({"scene_id": sc.scene_id, "gt": gt, "runs": per_run})
        print(f"[diagnose_pose] {sc.scene_id}: GT fx {gt['fx']:.0f} chord {gt['chord_m']:.1f} m | " + " | ".join(
            f"{label}: fx {r['fx_mean']:.0f} scale {r['chord_ratio']:.2f} jitter {r['path_over_chord']:.2f}"
            for label, r in per_run.items()), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "reads_gt": True,
            "note": "evaluation-side diagnostic; the reconstruction pipeline never reads these GT files",
            "tag": args.tag,
            "runs": list(runs),
            "primary": args.primary,
            "dashrecon_commit": git_commit(),
            "scenes": scenes,
        }, f, indent=2)
    print(f"[diagnose_pose] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
