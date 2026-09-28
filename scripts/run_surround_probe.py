"""E13: FRONT-only completion of an eight-direction virtual surrounding rig.

Two native GEN3C videos sweep right/back and left/back at three translated
centres. Only validated RGB-D becomes memory or persistent 3D supervision.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.trajectory import surrounding_plan
from dashrecon.provenance import git_commit


def main() -> None:
    """Run reproducible disk-separated stages and retain explicit failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    commit = git_commit()
    if commit.endswith("-dirty") or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("start in a clean committed checkout")
    if cfg["sides"] != [1, -1]:
        raise ValueError("right/back followed by left/back required")
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    OmegaConf.save(OmegaConf.create(cfg), out / "config.yaml")
    state = {"completed": False, "started": time.time(), "stages": [], "dashrecon_commit": commit}
    meta = {"dashrecon_commit": commit, "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT-only E5c and fresh candidates; no hidden camera, GT, E10/E11 or HUGSIM inputs",
            "generator_weights_frozen": True, "kind": "surrounding-camera completion experiment; no quality success assumed",
            "versions": {name: version(name) for name in ["torch", "numpy", "scipy", "Pillow", "gsplat", "lpips"]},
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.config, Path(cfg["init_log_dir"]) / "config.yaml", Path(cfg["init_log_dir"]) / "checkpoint_final.pth"]}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    def run(name: str, environment: str, arguments: list[str]) -> None:
        """Run one CLI stage; record its command, time and exit code before stopping."""
        command = [f".venvs/{environment}/bin/python", "-u", *arguments]
        entry = {"name": name, "command": command, "started": time.time(), "log": str(out / "logs" / f"{name}.log")}
        state["stages"].append(entry)
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        print(f"[E13] START {name} -> {entry['log']}", flush=True)
        with open(entry["log"], "w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, elapsed_s=time.time() - entry["started"])
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit {process.returncode}: {entry['log']}")
        print(f"[E13] DONE {name} {entry['elapsed_s']:.1f}s", flush=True)

    views_dirs, validations = [], []
    model = out / "model"
    initial = cfg["init_log_dir"]
    for round_index, side in enumerate(cfg["sides"]):
        plan = surrounding_plan(cfg["anchors"], cfg["pan_views"], cfg["transfer_views"], side)
        if len(plan) != 121:
            raise ValueError("one native 121-frame generation per half-circle required")
        plan_file = out / f"r{round_index}_trajectory.json"
        plan_file.write_text(json.dumps({"views": plan, "params": cfg, "round": round_index}, indent=2))
        views, validation = out / "views" / f"r{round_index}", out / "validation" / f"r{round_index}"
        run(f"r{round_index}_render", "main", ["scripts/render_views.py", "--log_dir", initial,
            "--trajectory_file", str(plan_file), "--out_dir", str(views), "--render_hw", *map(str, cfg["render_hw"]),
            "--src_offsets", *map(str, cfg["src_offsets"])])
        generation = ["scripts/gen3c_fill.py", "--views_dir", str(views), "--cache_policy", "consistent",
            "--num_steps", str(cfg["generation_steps"]), "--seed", str(cfg["generation_seed"]),
            "--memory_candidate_policy", "pose", "--memory_radius", str(cfg["memory_radius"]),
            "--max_memory_candidates", str(cfg["max_memory_candidates"])]
        if views_dirs:
            generation += ["--memory_dirs", *map(str, views_dirs), "--validation_dirs", *map(str, validations), "--require_memory"]
        run(f"r{round_index}_cache_preview", "gen3c", [*generation, "--buffers_only"])
        shutil.copyfile(views / "cache.json", out / f"r{round_index}_cache_preview.json")
        run(f"r{round_index}_generate", "gen3c", generation)
        depth = ["scripts/depth_views.py", "--views_dir", str(views), "--backend", "moge",
                 "--model_id", "Ruicheng/moge-2-vitl-normal", "--alignment_policy", "scene_global"]
        if views_dirs:
            scene_scale = json.loads((views_dirs[0] / "depth.json").read_text())["alignment"]["scene_scale"]
            depth += ["--scene_scale", str(scene_scale)]
        run(f"r{round_index}_depth", "gen3c", depth)
        verify = ["scripts/validate_generated_views.py", "--views_dir", str(views), "--out_dir", str(validation),
            "--reference_policy", "translated", "--min_baseline", str(cfg["min_baseline_scene_units"]),
            "--min_parallax_degrees", str(cfg["min_parallax_degrees"]), "--max_references", str(cfg["max_references"]),
            "--depth_tol", str(cfg["depth_tolerance"]), "--rgb_tol", str(cfg["rgb_tolerance"]),
            "--min_support", str(cfg["min_support"]), "--max_conflict_fraction", str(cfg["max_conflict_fraction"])]
        if views_dirs:
            verify += ["--memory_dirs", *map(str, views_dirs), "--memory_validation_dirs", *map(str, validations)]
        run(f"r{round_index}_validate", "main", verify)
        run(f"r{round_index}_before", "main", ["scripts/diagnose_distillation.py", "--log_dir", initial,
            "--views_dir", str(views), "--out_dir", str(out / f"r{round_index}_before"), "--label", f"r{round_index}_before",
            "--view_indices", *map(str, cfg["diagnostic_indices"]), "--novel_yaw", "7", "--validation_dirs", str(validation)])
        views_dirs.append(views)
        validations.append(validation)
        run(f"r{round_index}_train", "main", ["scripts/train_fill.py", "--scene_id", cfg["scene_id"], "--exp", "E13",
            "--init_log_dir", initial, "--out_log_dir", str(model), "--round", str(round_index),
            "--views_dirs", *map(str, views_dirs), "--validation_dirs", *map(str, validations), "--seen_w", "0",
            "--view_indices", *map(str, cfg["training_view_indices"]), "--steps", str(cfg["optimization_steps"]),
            "--seed", str(cfg["optimization_seed"]), "--spawn_stride", str(cfg["spawn_stride"]), "--pixel_stride", str(cfg["pixel_stride"])])
        run(f"r{round_index}_after", "main", ["scripts/diagnose_distillation.py", "--log_dir", str(model),
            "--views_dir", str(views), "--out_dir", str(out / f"r{round_index}_after"), "--label", f"r{round_index}_after",
            "--view_indices", *map(str, cfg["diagnostic_indices"]), "--novel_yaw", "7", "--validation_dirs", str(validation)])
        if round_index == 0:
            snapshot = out / "right_back_model"
            snapshot.mkdir()
            for name in ["checkpoint_final.pth", "config.yaml", "meta.json", "metrics.json"]:
                shutil.copyfile(model / name, snapshot / name)
        initial = str(model)
    run("common_rig_render", "main", ["scripts/render_sweep_comparison.py", "--init_log_dir", cfg["init_log_dir"],
        "--log_dirs", str(out / "right_back_model"), str(model), "--labels", "E5c", "E13_right_back_gen", "E13_surround_gen",
        "--frames", *map(str, cfg["anchors"]), "--yaws", *map(str, cfg["rig_yaws"]),
        "--rights", *map(str, cfg["rig_rights_scene_units"]), "--out_dir", str(out / "renders" / "common")])
    # Recheck the first half against its unchanged targets after the second update.
    run("right_back_retention", "main", ["scripts/diagnose_distillation.py", "--log_dir", str(model),
        "--views_dir", str(views_dirs[0]), "--out_dir", str(out / "right_back_retention"), "--label", "surround_on_first_half",
        "--view_indices", *map(str, cfg["diagnostic_indices"]), "--novel_yaw", "7", "--validation_dirs", str(validations[0])])
    state["runtime_s"] = time.time() - state["started"]
    (out / "run_status.json").write_text(json.dumps(state, indent=2))
    meta.update(runtime_s=state["runtime_s"], peak_vram_gb=max(
        json.loads((d / "fill.json").read_text())["peak_vram_gb"] for d in views_dirs))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    subprocess.run([sys.executable, "scripts/summarize_surround_probe.py", "--probe_dir", str(out)], check=True)
    state["completed"] = True
    (out / "run_status.json").write_text(json.dumps(state, indent=2))
    print(f"[E13] complete -> {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
