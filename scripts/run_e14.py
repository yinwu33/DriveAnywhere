"""E14 (docs/EXPERIMENTS.md): view-gated turning completion of one scene, from E5c.

Stages, each a CLI in its own venv, logged to <output_dir>/logs/<stage>.log; run_status.json records every command,
exit code and time, and the run stops at the first failure:
    gate               make_view_gate.py: observation-cone gate of the E5c Gaussians farther than gate_mesh_distance
                       from the mesh (D-A3); every novel-view render below goes through it
    r<k>_render        render_views.py: the current model (E5c, then E14) along every frame_stride-th FRONT frame,
                       turned by moves[k] over ramp_frames views; hole masks with the earlier rounds as 3D memory
    r<k>_cache_preview / r<k>_generate   gen3c_fill.py (consistent cache: real FRONT warps first, then validated
                       generated memory of earlier rounds)
    r<k>_depth         depth_views.py: MapAnything jointly over the round's frames, scene-global scale (D-B1)
    r<k>_validate      validate_generated_views.py (translated references, as E13)
    r<k>_train         train_fill.py --view_gate --no_refine --unknown_w --round_affine
    r<k>_after         diagnose_distillation.py on the round's own views
    eval_front / eval_cross_camera / common_rig   FRONT held-out, the four side cameras over E5c's unseen pixels
                       (evaluation reads GT), fixed common cameras against E5c and the compare runs
Output: <output_dir>/ view_gate.pt, views/r<k>, validation/r<k>, model/ (the E14 run: checkpoint, config with
view_gate), r<k>_after/, renders/common/, logs/, config.yaml, meta.json, run_status.json.

Example (main venv, clean committed checkout):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.1 \
        PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH .venvs/main/bin/python -u scripts/run_e14.py \
        --config configs/dashrecon/E14_gated.yaml
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
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    commit = git_commit()
    if commit.endswith("-dirty") or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("start in a clean committed checkout")
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    OmegaConf.save(OmegaConf.create(cfg), out / "config.yaml")
    state = {"completed": False, "started": time.time(), "stages": [], "dashrecon_commit": commit}
    meta = {"exp": "E14", "dashrecon_commit": commit, "generative": True, "uses_oracle": False,
            "oracle_note": "FRONT-only E5c and its dashrecon products; the evaluation stages read GT side cameras",
            "generator_weights_frozen": True,
            "versions": {name: version(name) for name in ["torch", "numpy", "scipy", "Pillow", "gsplat", "lpips"]},
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.config, Path(cfg["init_log_dir"]) / "config.yaml",
                                       Path(cfg["init_log_dir"]) / "checkpoint_final.pth", Path(cfg["mesh_ply"])]}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    def run(name: str, environment: str, arguments: list[str]) -> None:
        """Run one CLI stage; record its command, time and exit code; stop the run if it fails."""
        command = [f".venvs/{environment}/bin/python", "-u", *arguments]
        entry = {"name": name, "command": command, "started": time.time(), "log": str(out / "logs" / f"{name}.log")}
        state["stages"].append(entry)
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        print(f"[E14] START {name} -> {entry['log']} ({time.strftime('%H:%M')})", flush=True)
        with open(entry["log"], "w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, elapsed_s=time.time() - entry["started"])
        (out / "run_status.json").write_text(json.dumps(state, indent=2))
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit {process.returncode}: {entry['log']}")
        print(f"[E14] DONE {name} {entry['elapsed_s'] / 60:.1f} min", flush=True)

    gate = str(out / "view_gate.pt")
    run("gate", "main", ["scripts/make_view_gate.py", "--log_dir", cfg["init_log_dir"], "--mesh_ply", cfg["mesh_ply"],
        "--mesh_distance", str(cfg["gate_mesh_distance"]), "--margin", str(cfg["gate_margin_deg"]),
        "--fade", str(cfg["gate_fade_deg"]), "--cone_min_weight", str(cfg["gate_cone_min_weight"]), "--out", gate])
    model = str(out / "model")
    views_dirs, validations = [], []
    for k, move in enumerate(cfg["moves"]):
        source = cfg["init_log_dir"] if k == 0 else model
        views, validation = str(out / "views" / f"r{k}"), str(out / "validation" / f"r{k}")
        render = ["scripts/render_views.py", "--log_dir", source, "--move", move, "--out_dir", views,
                  "--frame_stride", str(cfg["frame_stride"]), "--ramp_frames", str(cfg["ramp_frames"]),
                  "--render_hw", *map(str, cfg["render_hw"]), "--src_offsets", *map(str, cfg["src_offsets"]),
                  "--memory_dirs", *views_dirs]
        if k == 0:
            render += ["--view_gate", gate]
        run(f"r{k}_render", "main", render)
        generation = ["scripts/gen3c_fill.py", "--views_dir", views, "--cache_policy", "consistent",
                      "--num_steps", str(cfg["generation_steps"]), "--seed", str(cfg["generation_seed"]),
                      "--memory_candidate_policy", "pose", "--memory_radius", str(cfg["memory_radius"]),
                      "--max_memory_candidates", str(cfg["max_memory_candidates"])]
        if views_dirs:
            generation += ["--memory_dirs", *views_dirs, "--validation_dirs", *validations]
        run(f"r{k}_cache_preview", "gen3c", [*generation, "--buffers_only"])
        run(f"r{k}_generate", "gen3c", generation)
        run(f"r{k}_depth", "mapanything", ["scripts/depth_views.py", "--views_dir", views, "--backend", "mapanything",
            "--model_id", cfg["depth_model_id"], "--alignment_policy", "scene_global"])
        verify = ["scripts/validate_generated_views.py", "--views_dir", views, "--out_dir", validation,
                  "--reference_policy", "translated", "--min_baseline", str(cfg["min_baseline_scene_units"]),
                  "--min_parallax_degrees", str(cfg["min_parallax_degrees"]), "--max_references", str(cfg["max_references"]),
                  "--depth_tol", str(cfg["depth_tolerance"]), "--rgb_tol", str(cfg["rgb_tolerance"]),
                  "--min_support", str(cfg["min_support"]), "--max_conflict_fraction", str(cfg["max_conflict_fraction"])]
        if views_dirs:
            verify += ["--memory_dirs", *views_dirs, "--memory_validation_dirs", *validations]
        run(f"r{k}_validate", "main", verify)
        views_dirs.append(views)
        validations.append(validation)
        train = ["scripts/train_fill.py", "--scene_id", cfg["scene_id"], "--exp", "E14", "--init_log_dir", source,
                 "--out_log_dir", model, "--round", str(k), "--views_dirs", *views_dirs, "--validation_dirs", *validations,
                 "--seen_w", "0", "--steps", str(cfg["optimization_steps"]), "--seed", str(cfg["optimization_seed"]),
                 "--spawn_stride", str(cfg["spawn_stride"]), "--pixel_stride", str(cfg["pixel_stride"]),
                 "--no_refine", "--unknown_w", str(cfg["unknown_w"]), "--round_affine",
                 "--round_affine_reg", str(cfg["round_affine_reg"])]
        if k == 0:
            train += ["--view_gate", gate]
        run(f"r{k}_train", "main", train)
        run(f"r{k}_after", "main", ["scripts/diagnose_distillation.py", "--log_dir", model, "--views_dir", views,
            "--out_dir", str(out / f"r{k}_after"), "--label", f"r{k}_after",
            "--view_indices", *map(str, cfg["diagnostic_indices"]), "--novel_yaw", "7", "--validation_dirs", validation])
    run("eval_front", "main", ["scripts/eval_front_heldout.py", "--log_dir", model, "--example_frames", "50", "100", "150"])
    run("eval_cross_camera", "main", ["scripts/eval_cross_camera.py", "--log_dir", model, "--gt_root", cfg["gt_root"],
        "--frame_stride", "5", "--alpha", "0.5", "--example_frames", "50", "100", "150", "--cams", "1", "2", "3", "4",
        "--unseen_dir", cfg["unseen_dir"], "--out_subdir", "cross_camera_p9"])
    run("common_rig", "main", ["scripts/render_sweep_comparison.py", "--init_log_dir", cfg["init_log_dir"],
        "--log_dirs", *cfg["compare_log_dirs"], model, "--labels", "E5c", *cfg["compare_labels"], "E14_gen",
        "--frames", *map(str, cfg["rig_frames"]), "--yaws", *map(str, cfg["rig_yaws"]),
        "--rights", *map(str, cfg["rig_rights_scene_units"]), "--out_dir", str(out / "renders" / "common")])
    state.update(completed=True, runtime_s=time.time() - state["started"])
    (out / "run_status.json").write_text(json.dumps(state, indent=2))
    meta["runtime_s"] = state["runtime_s"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[E14] complete -> {out} ({state['runtime_s'] / 3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
