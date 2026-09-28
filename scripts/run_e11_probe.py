"""Run the fixed local distillation diagnostic from DECISIONS T.

Execute from a clean checkout with existing E5c/E10/E11 data and main environment.
Runs are sequential, never overwrite results, and fail explicitly on stage errors.
This reuses historical synthetic data; it is not a leakage-free quality benchmark.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.provenance import git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    commit = git_commit()
    if commit.endswith("-dirty"):
        raise ValueError("commit code before starting")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"use an isolated clean checkout: {status}")
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    local_indices = list(range(40, 65))
    diagnostic_indices = [40, 44, 48, 52, 56, 60, 64]
    config = {"scene_id": "val056", "steps": args.steps, "seed": 0,
              "local_view_indices": local_indices, "diagnostic_indices": diagnostic_indices,
              "init": "results/E5c/val056", "historical_input_warning": "E10/E11 may indirectly contain old heldout FRONT frames; diagnostic only",
              "novel_right_scene_units": .5, "novel_yaw_degrees": 5,
              "generation_weights_frozen": True, "generation_steps": {"E10": 35, "E11": 8},
              "dashrecon_commit": commit}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    state = {"completed": False, "stages": [], "dashrecon_commit": commit}

    def run(name: str, arguments: list[str]) -> None:
        """Record command and timing, fail without silently skipping a stage."""
        command = [sys.executable, "-u", *arguments]
        entry = {"name": name, "command": command, "log": str(output / "logs" / f"{name}.log"), "started": time.time()}
        state["stages"].append(entry)
        (output / "run_status.json").write_text(json.dumps(state, indent=2))
        print(f"[probe] {name} -> {entry['log']}", flush=True)
        with open(entry["log"], "w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, elapsed_s=time.time() - entry["started"])
        (output / "run_status.json").write_text(json.dumps(state, indent=2))
        if process.returncode != 0:
            raise RuntimeError(f"{name} exited {process.returncode}; inspect {entry['log']}")

    def diagnose(name: str, checkpoint: str, views: str, validations: list[str]) -> None:
        """Same-camera and novel-view report using the fixed diagnostic indices."""
        run(name, ["scripts/diagnose_distillation.py", "--log_dir", checkpoint, "--views_dir", views,
                   "--out_dir", str(output / name), "--label", name, "--view_indices", *map(str, diagnostic_indices),
                   "--validation_dirs", *validations])

    for trajectory in ("r0_yaw30", "r2_yaw60", "r4_yaw90"):
        diagnose(f"E10_{trajectory}", "results/E10/val056", f"results/E10/val056/views/{trajectory}", [])
    for arm, other in (("real_only_s8", "memory_s8"), ("memory_s8", "real_only_s8")):
        views = f"results/E11/val056/{arm}"
        validations = [f"results/E11/val056/validation/{name}" for name in (arm, other)]
        diagnose(f"{arm}_before", config["init"], views, validations)
        fitted = str(output / f"{arm}_fit")
        run(f"{arm}_train", ["scripts/train_fill.py", "--scene_id", "val056", "--exp", "E11_probe",
                             "--init_log_dir", config["init"], "--out_log_dir", fitted, "--round", "0",
                             "--views_dirs", views, "--validation_dirs", validations[0], "--seen_w", "0",
                             "--view_indices", *map(str, local_indices), "--steps", str(args.steps), "--seed", "0"])
        meta_file = Path(fitted) / "meta.json"
        meta = json.loads(meta_file.read_text())
        meta["diagnostic_only"] = True
        meta["historical_input_warning"] = config["historical_input_warning"]
        meta["versions"] = {name: version(name) for name in ("torch", "numpy", "Pillow", "gsplat", "lpips")}
        meta_file.write_text(json.dumps(meta, indent=2))
        diagnose(f"{arm}_after", fitted, views, validations)
    sampling = [json.loads((output / f"{arm}_fit" / "sampled_views_r0.json").read_text())
                for arm in ("real_only_s8", "memory_s8")]
    state["paired_sampling_identical"] = sampling[0] == sampling[1]
    if not state["paired_sampling_identical"]:
        (output / "run_status.json").write_text(json.dumps(state, indent=2))
        raise ValueError("real/generated view sampling differs between paired arms")
    state["completed"] = True
    (output / "run_status.json").write_text(json.dumps(state, indent=2))
    subprocess.run([sys.executable, "scripts/summarize_e11_probe.py", "--probe_dir", str(output)], check=True)
    print(f"[probe] completed -> {output}", flush=True)


if __name__ == "__main__":
    main()
