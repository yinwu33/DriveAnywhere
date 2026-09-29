"""E17b / E17c (docs/EXPERIMENTS.md): against E17a's mesh arm (same inputs, depth and budget),
    E17b  --protect_original: generated views no longer update E5c's Gaussians (D-A5: E14's side supervision altered
          them, and they looked broken from behind);
    E17c  --protect_original + --owner_dirs: each imagined surface point spawns and is supervised from its best
          generated view only (best_view_ownership.py), instead of every view's own imagination (E17a: the overlap
          renders as fine noise).
Stages via dashrecon.stage_runner: ownership, train_b, train_c, per arm distillation diagnostics on rounds 0 and 2,
FRONT held-out and side cameras over E5c's unseen pixels (evaluation reads GT), and one common camera grid with E5c,
E14, E17a mesh, E17b, E17c. Output in <output_dir>: ownership/r<k>/, b/, c/ (runs), renders/common/, logs/.

Example (main venv, clean committed checkout):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.1 \
        PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH .venvs/main/bin/python -u scripts/run_e17bc.py \
        --config configs/dashrecon/E17bc_owner.yaml
"""
import argparse
import json
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
    src, cons, out = Path(cfg["source_dir"]), Path(cfg["consolidated_dir"]), Path(cfg["output_dir"])
    assert json.loads((src / "run_status.json").read_text())["completed"], f"{src} has not finished"
    assert json.loads((cons / "run_status.json").read_text())["completed"], f"{cons} has not finished"
    rounds = range(cfg["rounds"])
    views = [str(src / "views" / f"r{k}") for k in rounds]
    validations = [str(src / "validation" / f"r{k}") for k in rounds]
    depths = [str(cons / "mesh_depth" / f"r{k}") for k in rounds]
    owners = [str(out / "ownership" / f"r{k}") for k in rounds]
    runner = StageRunner(out, cfg, commit, args.resume, "E17bc")
    runner.run("ownership", "main", ["scripts/best_view_ownership.py", "--views_dirs", *views, "--validation_dirs", *validations,
               "--depth_dirs", *depths, "--unknown_w", str(cfg["unknown_w"]), "--frame_window", str(cfg["owner_frame_window"]),
               "--depth_tol", str(cfg["owner_depth_tol"]), "--out_root", str(out / "ownership")])
    common = ["scripts/train_fill.py", "--scene_id", cfg["scene_id"], "--init_log_dir", cfg["init_log_dir"], "--round", "0",
              "--views_dirs", *views, "--validation_dirs", *validations, "--seen_w", "0",
              "--steps", str(cfg["optimization_steps"]), "--seed", str(cfg["optimization_seed"]),
              "--spawn_stride", str(cfg["spawn_stride"]), "--pixel_stride", str(cfg["pixel_stride"]),
              "--view_gate", str(src / "view_gate.pt"), "--no_refine", "--unknown_w", str(cfg["unknown_w"]),
              "--round_affine", "--round_affine_reg", str(cfg["round_affine_reg"]), "--spawn_all",
              "--depth_dirs", *depths, "--protect_original"]
    arms = {"b": [], "c": ["--owner_dirs", *owners]}
    for arm, extra in arms.items():
        runner.run(f"train_{arm}", "main", [*common, "--exp", f"E17{arm}", "--out_log_dir", str(out / arm), *extra])
    for arm in arms:
        model = str(out / arm)
        for k in (0, 2):
            runner.run(f"{arm}_after_r{k}", "main", ["scripts/diagnose_distillation.py", "--log_dir", model,
                       "--views_dir", views[k], "--out_dir", str(out / f"{arm}_after_r{k}"), "--label", f"{arm}_r{k}",
                       "--view_indices", *map(str, cfg["diagnostic_indices"]), "--novel_yaw", "7",
                       "--validation_dirs", validations[k]])
        runner.run(f"{arm}_front", "main", ["scripts/eval_front_heldout.py", "--log_dir", model, "--example_frames", "50", "100", "150"])
        runner.run(f"{arm}_xcam", "main", ["scripts/eval_cross_camera.py", "--log_dir", model, "--gt_root", cfg["gt_root"],
                   "--frame_stride", "5", "--alpha", "0.5", "--example_frames", "50", "100", "150", "--cams", "1", "2", "3", "4",
                   "--unseen_dir", cfg["unseen_dir"], "--out_subdir", "cross_camera_p9"])
    runner.run("common_rig", "main", ["scripts/render_sweep_comparison.py", "--init_log_dir", cfg["init_log_dir"],
               "--log_dirs", str(src / "model"), str(cons / "mesh"), str(out / "b"), str(out / "c"),
               "--labels", "E5c", "E14_gen", "E17a_mesh_gen", "E17b_gen", "E17c_gen",
               "--frames", *map(str, cfg["rig_frames"]), "--yaws", *map(str, cfg["rig_yaws"]),
               "--rights", *map(str, cfg["rig_rights_scene_units"]), "--out_dir", str(out / "renders" / "common")])
    runner.complete()
    print(f"[E17bc] complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
