"""E17a (docs/EXPERIMENTS.md): does one consolidated ("imagined") geometry for all rounds make the distilled scene
sharper and more consistent than each round's own depth?

Uses the finished E14 (views, validation, view gate; nothing of it is modified) and runs, with dashrecon.stage_runner:
    fuse          fuse_generated.py: E5c's fused cloud + the generated pixels E14's validation accepted, all rounds
    mesh          run_mesh.py (nksr venv): the imagined NKSR mesh
    mesh_depth    render_mesh_depth.py: its depth at every generated camera of every round
    train_ma      train_fill.py from E5c, all E14 rounds at once, each round's own MapAnything depth   (arm "ma")
    train_mesh    the same with the imagined mesh's depth (--depth_dirs)                                 (arm "mesh")
                  both: E14's view gate, --no_refine, --unknown_w, --round_affine, --spawn_all, same seed / steps
    <arm>_after / <arm>_front / <arm>_xcam   distillation diagnostics on round 0 and 2 views, FRONT held-out,
                  side cameras over E5c's unseen pixels (evaluation reads GT)
    common_rig    render_sweep_comparison.py: E5c | E14 | ma | mesh on one camera grid
Output: <output_dir>/ imagined_fusion/ (points_fused.ply, mesh_nksr.ply), mesh_depth/r<k>/, ma/, mesh/ (runs),
renders/common/, logs/, config.yaml, run_status.json.

Example (main venv, clean committed checkout):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.1 \
        PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH .venvs/main/bin/python -u scripts/run_e17a.py \
        --config configs/dashrecon/E17a_consolidated.yaml
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
    src, out = Path(cfg["source_dir"]), Path(cfg["output_dir"])
    assert json.loads((src / "run_status.json").read_text())["completed"], f"{src} has not finished"
    views = [str(src / "views" / f"r{k}") for k in range(cfg["rounds"])]
    validations = [str(src / "validation" / f"r{k}") for k in range(cfg["rounds"])]
    runner = StageRunner(out, cfg, commit, args.resume, "E17a")
    fusion = str(out / "imagined_fusion")
    runner.run("fuse", "main", ["scripts/fuse_generated.py", "--real_fusion_dir", cfg["real_fusion_dir"],
               "--views_dirs", *views, "--validation_dirs", *validations, "--pixel_stride", str(cfg["fuse_pixel_stride"]),
               "--voxel", str(cfg["fuse_voxel"]), "--out_dir", fusion])
    runner.run("mesh", "nksr", ["scripts/run_mesh.py", "--fusion_dir", fusion, "--detail_level", str(cfg["mesh_detail_level"]),
               "--mise_iter", str(cfg["mesh_mise_iter"]), "--solver_tol", str(cfg["mesh_solver_tol"]),
               "--coverage_radius", str(cfg["mesh_coverage_radius"])])
    runner.run("mesh_depth", "main", ["scripts/render_mesh_depth.py", "--mesh_ply", f"{fusion}/mesh_nksr.ply",
               "--views_dirs", *views, "--validation_dirs", *validations, "--near", str(cfg["mesh_near"]),
               "--far", str(cfg["mesh_far"]), "--out_root", str(out / "mesh_depth")])
    common = ["scripts/train_fill.py", "--scene_id", cfg["scene_id"], "--init_log_dir", cfg["init_log_dir"], "--round", "0",
              "--views_dirs", *views, "--validation_dirs", *validations, "--seen_w", "0",
              "--steps", str(cfg["optimization_steps"]), "--seed", str(cfg["optimization_seed"]),
              "--spawn_stride", str(cfg["spawn_stride"]), "--pixel_stride", str(cfg["pixel_stride"]),
              "--view_gate", str(src / "view_gate.pt"), "--no_refine", "--unknown_w", str(cfg["unknown_w"]),
              "--round_affine", "--round_affine_reg", str(cfg["round_affine_reg"]), "--spawn_all"]
    arms = {"ma": [], "mesh": ["--depth_dirs", *[str(out / "mesh_depth" / f"r{k}") for k in range(cfg["rounds"])]]}
    for arm, extra in arms.items():
        runner.run(f"train_{arm}", "main", [*common, "--exp", f"E17a_{arm}", "--out_log_dir", str(out / arm), *extra])
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
               "--log_dirs", str(src / "model"), str(out / "ma"), str(out / "mesh"), "--labels", "E5c", "E14_gen", "E17a_ma_gen",
               "E17a_mesh_gen", "--frames", *map(str, cfg["rig_frames"]), "--yaws", *map(str, cfg["rig_yaws"]),
               "--rights", *map(str, cfg["rig_rights_scene_units"]), "--out_dir", str(out / "renders" / "common")])
    runner.complete()
    print(f"[E17a] complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
