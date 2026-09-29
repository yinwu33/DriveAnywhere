"""D-C2 (docs/EXPERIMENTS.md): do the sparse dots of grazing-angle cache warps make GEN3C paint foliage?

At yaw 90 about half of the valid cache pixels of E14 round 2 are isolated dots (FRONT pixels stretched across the
side view); the generated frames there are mostly vegetation-like texture (D-E1). On exactly the inputs of an E14
round (cameras, cache sources, validated memory of the earlier rounds, seed, no prompt, guidance 1) each arm changes
only the cache's speckle treatment (gen3c_fill.py --speckle, dashrecon.gen.cache.treat_speckle). Per arm:
gen3c_fill.py --out_dir (inputs linked, nothing of E14 written) -> depth_views.py (MapAnything, scene-global, poses
flagged metric as D-B2) -> validate_generated_views.py (E14's parameters and memory) -> the fully turned frames paired
with the real camera looking the same way; then eval_realism.py over the E14 round itself and every arm.
EVALUATION in the last two stages only (GT images as a distribution reference).

Example (main venv, clean committed checkout; needs D-E1's references):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venvs/main/bin/python -u scripts/run_dc2.py \
        --config configs/dashrecon/DC2_speckle.yaml
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
    out = Path(cfg["output_dir"])
    runner = StageRunner(out, cfg, commit, args.resume, "D-C2")
    base = cfg["base_label"]
    realism = {base: str(out / f"{base}_realism")}
    runner.run(f"link_{base}", "main", ["scripts/link_generated_for_realism.py", "--views_dir", cfg["base_views"],
               "--cam", str(cfg["cam"]), "--refs_dir", cfg["refs_dir"], "--out_dir", realism[base]])
    memory = ["--memory_dirs", *cfg["memory_dirs"]]
    for arm, speckle in cfg["arms"].items():
        views, validation = str(out / arm / "views"), str(out / arm / "validation")
        runner.run(f"{arm}_generate", "gen3c", ["scripts/gen3c_fill.py", "--views_dir", cfg["base_views"], "--out_dir", views,
                   "--cache_policy", "consistent", "--num_steps", str(cfg["generation_steps"]), "--seed", str(cfg["generation_seed"]),
                   "--memory_candidate_policy", "pose", "--memory_radius", str(cfg["memory_radius"]),
                   "--max_memory_candidates", str(cfg["max_memory_candidates"]), *memory,
                   "--validation_dirs", *cfg["memory_validation_dirs"], "--speckle", speckle["mode"],
                   "--speckle_window", str(speckle["window"]), "--speckle_density", str(speckle["density"])])
        runner.run(f"{arm}_depth", "mapanything", ["scripts/depth_views.py", "--views_dir", views, "--backend", "mapanything",
                   "--model_id", cfg["depth_model_id"], "--alignment_policy", "scene_global", "--poses_metric"])
        runner.run(f"{arm}_validate", "main", ["scripts/validate_generated_views.py", "--views_dir", views, "--out_dir", validation,
                   "--reference_policy", "translated", "--min_baseline", str(cfg["min_baseline_scene_units"]),
                   "--min_parallax_degrees", str(cfg["min_parallax_degrees"]), "--max_references", str(cfg["max_references"]),
                   "--depth_tol", str(cfg["depth_tolerance"]), "--rgb_tol", str(cfg["rgb_tolerance"]),
                   "--min_support", str(cfg["min_support"]), "--max_conflict_fraction", str(cfg["max_conflict_fraction"]),
                   *memory, "--memory_validation_dirs", *cfg["memory_validation_dirs"]])
        realism[arm] = str(out / f"{arm}_realism")
        runner.run(f"link_{arm}", "main", ["scripts/link_generated_for_realism.py", "--views_dir", views, "--cam", str(cfg["cam"]),
                   "--refs_dir", cfg["refs_dir"], "--out_dir", realism[arm]])
    runner.run("realism", "main", ["scripts/eval_realism.py", "--runs", *[f"{k}={v}" for k, v in realism.items()],
               "--seg_model_id", cfg["seg_model_id"], "--out", str(out / "realism.json")])
    runner.complete()


if __name__ == "__main__":
    main()
