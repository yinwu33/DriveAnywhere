"""D-B2 (docs/EXPERIMENTS.md): are E14's yaw-90 conflicts a depth problem? For each case round of E14, the generated
frames get new MapAnything depth with the given poses flagged metric (depth_views.py --poses_metric; E14 used the
non-metric flag, and at yaw 90 MapAnything shrank the camera motion to 0.16 of the given one), then the same
validation as E14 (same parameters and the same earlier rounds as memory). Nothing of E14 is written: each case works
in <output_dir>/r<k>/views with links to E14's cams.json, filled, mask, depth, count.

Example (main venv, clean committed checkout):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python -u scripts/run_db2.py --config configs/dashrecon/DB2_metric_poses.yaml
"""
import argparse
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.stage_runner import StageRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    commit = git_commit()
    if commit.endswith("-dirty") or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("start in a clean committed checkout")
    src, out = Path(cfg["source_dir"]), Path(cfg["output_dir"])
    runner = StageRunner(out, cfg, commit, args.resume, "D-B2")
    for k, memory in cfg["cases"].items():
        views = str(out / f"r{k}" / "views")
        runner.run(f"r{k}_link", "main", ["scripts/link_views.py", "--src", str(src / "views" / f"r{k}"), "--dst", views,
                   "--names", "cams.json", "filled", "mask", "depth", "count"])
        runner.run(f"r{k}_depth", "mapanything", ["scripts/depth_views.py", "--views_dir", views, "--backend", "mapanything",
                   "--model_id", cfg["depth_model_id"], "--alignment_policy", "scene_global", "--poses_metric"])
        verify = ["scripts/validate_generated_views.py", "--views_dir", views, "--out_dir", str(out / f"r{k}" / "validation"),
                  "--reference_policy", "translated", "--min_baseline", str(cfg["min_baseline_scene_units"]),
                  "--min_parallax_degrees", str(cfg["min_parallax_degrees"]), "--max_references", str(cfg["max_references"]),
                  "--depth_tol", str(cfg["depth_tolerance"]), "--rgb_tol", str(cfg["rgb_tolerance"]),
                  "--min_support", str(cfg["min_support"]), "--max_conflict_fraction", str(cfg["max_conflict_fraction"])]
        if memory:
            verify += ["--memory_dirs", *[str(src / "views" / f"r{m}") for m in memory],
                       "--memory_validation_dirs", *[str(src / "validation" / f"r{m}") for m in memory]]
        runner.run(f"r{k}_validate", "main", verify)
    runner.complete()


if __name__ == "__main__":
    main()
