"""E12: equal-budget driving versus three anchored sweeps, from clean E5c inputs.

This runs one Phase 9 task-3 sampling diagnostic, without hidden cameras/GT or
generator adaptation. Each arm is one native 121-frame, 35-step generation.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.trajectory import sampling_plan
from dashrecon.provenance import git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    commit = git_commit()
    if commit.endswith("-dirty") or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("use a clean committed checkout before starting")
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    OmegaConf.save(OmegaConf.create(cfg), out / "config.yaml")
    state = {"completed": False, "stages": [], "dashrecon_commit": commit, "started": time.time()}
    meta = {"dashrecon_commit": commit, "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT-only E5c and fresh generation; no E10/E11 synthetic memory",
            "versions": {name: version(name) for name in ["torch", "numpy", "scipy", "Pillow", "gsplat", "lpips"]},
            "generator_weights_frozen": True, "kind": "sampling diagnostic; not GT image quality",
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.config, Path(cfg["init_log_dir"]) / "config.yaml", Path(cfg["init_log_dir"]) / "checkpoint_final.pth"]}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    def run(name: str, environment: str, arguments: list[str]) -> None:
        """Record exact command and fail with its retained log on a stage error."""
        command = [f".venvs/{environment}/bin/python", "-u", *arguments]
        entry = {"name": name, "command": command, "started": time.time(), "log": str(out / "logs" / f"{name}.log")}
        state["stages"].append(entry)
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        print(f"[E12] START {name} -> {entry['log']}", flush=True)
        with open(entry["log"], "w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, elapsed_s=time.time() - entry["started"])
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit {process.returncode}: {entry['log']}")
        print(f"[E12] DONE {name} {entry['elapsed_s']:.1f}s", flush=True)

    for mode in ["drive", "sweep"]:
        plan = sampling_plan(mode, cfg["anchors"], cfg["yaw_max"], cfg["sweep_views"], cfg["transfer_views"], cfg["drive_ramp_views"])
        if len(plan) != 121 or plan[0]["yaw"] != 0:
            raise ValueError("a native 121-frame window starting at a real FRONT camera required")
        plan_file = out / f"{mode}_trajectory.json"
        plan_file.write_text(json.dumps({"views": plan, "mode": mode, "params": cfg}, indent=2))
        views = out / f"{mode}_views"
        run(f"{mode}_render", "main", ["scripts/render_views.py", "--log_dir", cfg["init_log_dir"],
            "--trajectory_file", str(plan_file), "--out_dir", str(views), "--render_hw", *map(str, cfg["render_hw"]),
            "--src_offsets", *map(str, cfg["src_offsets"])])
        run(f"{mode}_generate", "gen3c", ["scripts/gen3c_fill.py", "--views_dir", str(views), "--cache_policy", "consistent",
            "--num_steps", str(cfg["generation_steps"]), "--seed", str(cfg["generation_seed"])])
        run(f"{mode}_depth", "gen3c", ["scripts/depth_views.py", "--views_dir", str(views), "--backend", "moge",
            "--model_id", "Ruicheng/moge-2-vitl-normal"])
        validation = out / f"{mode}_validation"
        run(f"{mode}_validate", "main", ["scripts/validate_generated_views.py", "--views_dir", str(views), "--out_dir", str(validation),
            "--reference_policy", "translated", "--min_baseline", str(cfg["min_baseline_scene_units"]),
            "--min_parallax_degrees", str(cfg["min_parallax_degrees"]), "--max_references", str(cfg["max_references"]),
            "--depth_tol", str(cfg["depth_tolerance"]), "--rgb_tol", str(cfg["rgb_tolerance"]),
            "--min_support", str(cfg["min_support"]), "--max_conflict_fraction", str(cfg["max_conflict_fraction"])])
        for phase, checkpoint in [("before", cfg["init_log_dir"]), ("after", str(out / f"{mode}_fit"))]:
            if phase == "after":
                run(f"{mode}_train", "main", ["scripts/train_fill.py", "--scene_id", cfg["scene_id"], "--exp", "E12",
                    "--init_log_dir", cfg["init_log_dir"], "--out_log_dir", checkpoint, "--round", "0", "--views_dirs", str(views),
                    "--validation_dirs", str(validation), "--seen_w", "0", "--view_indices", *map(str, cfg["training_view_indices"]),
                    "--steps", str(cfg["optimization_steps"]), "--seed", str(cfg["optimization_seed"])])
            run(f"{mode}_{phase}", "main", ["scripts/diagnose_distillation.py", "--log_dir", checkpoint, "--views_dir", str(views),
                "--out_dir", str(out / f"{mode}_{phase}"), "--label", f"{mode}_{phase}", "--view_indices", *map(str, cfg["diagnostic_indices"]),
                "--novel_yaw", "7", "--validation_dirs", str(validation)])
    sampling = [json.loads((out / f"{mode}_fit" / "sampled_views_r0.json").read_text()) for mode in ["drive", "sweep"]]
    state["real_sampling_identical"] = sampling[0]["real_full_image_indices"] == sampling[1]["real_full_image_indices"]
    state["pool_sampling_identical"] = sampling[0]["generated_pool_indices"] == sampling[1]["generated_pool_indices"]
    if not state["real_sampling_identical"] or not state["pool_sampling_identical"]:
        raise ValueError("paired optimization sampling differs")
    run("common_render", "main", ["scripts/render_sweep_comparison.py", "--init_log_dir", cfg["init_log_dir"],
        "--log_dirs", str(out / "drive_fit"), str(out / "sweep_fit"), "--frames", *map(str, cfg["anchors"]),
        "--out_dir", str(out / "renders" / "common")])
    state["completed"] = True
    state["runtime_s"] = time.time() - state["started"]
    (out / "run_status.json").write_text(json.dumps(state, indent=2))
    meta["runtime_s"] = state["runtime_s"]
    meta["peak_vram_gb"] = max(json.loads((out / f"{mode}_views" / "fill.json").read_text())["peak_vram_gb"] for mode in ["drive", "sweep"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    subprocess.run([sys.executable, "scripts/summarize_sweep_probe.py", "--probe_dir", str(out)], check=True)
    print(f"[E12] complete -> {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
