"""E19 (docs/EXPERIMENTS.md): NVIDIA Fixer progressive distillation on the turning views of a Phase 9 run.

train_fixer.py continues --init_log_dir (E20: +-45 degree completion from E5f) with Fixer-restored targets at the
config's yaws (no sideways move), then the evaluation every Phase 9 run gets: FRONT held-out, the four side cameras
over E5c's unseen pixels (every 5th frame), D-E1's side renders (every 2nd frame) with eval_realism.py against the
realism_compare runs (KID / FID, classes, sharpness, flat fraction), and the fixed common cameras.
Stages via dashrecon.stage_runner. EVALUATION stages read GT side cameras.

Example (main venv, clean committed checkout):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python -u scripts/run_e19.py --config configs/dashrecon/E19_fixer_turns.yaml
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
    runner = StageRunner(out, cfg, commit, args.resume, cfg["exp"])
    model = str(out / "model")
    runner.run("train", "main", ["scripts/train_fixer.py", "--scene_id", cfg["scene_id"], "--output_root", cfg["output_root"],
               "--init_log_dir", cfg["init_log_dir"], "--out_log_dir", model, "--exp", cfg["exp"],
               "--yaw_choices", *map(str, cfg["yaw_choices"]), "--off_lo_start", "0", "--off_hi_start", "0",
               "--off_lo_end", "0", "--off_hi_end", "0", "--steps", str(cfg["steps"]), "--seed", str(cfg["seed"])])
    runner.run("eval_front", "main", ["scripts/eval_front_heldout.py", "--log_dir", model, "--example_frames", "50", "100", "150"])
    runner.run("eval_cross_camera", "main", ["scripts/eval_cross_camera.py", "--log_dir", model, "--gt_root", cfg["gt_root"],
               "--frame_stride", "5", "--alpha", "0.5", "--example_frames", "50", "100", "150", "--cams", "1", "2", "3", "4",
               "--unseen_dir", cfg["unseen_dir"], "--out_subdir", "cross_camera_p9"])
    runner.run("eval_realism_render", "main", ["scripts/eval_cross_camera.py", "--log_dir", model, "--gt_root", cfg["gt_root"],
               "--frame_stride", "2", "--alpha", "0.5", "--example_frames", "--cams", "1", "2", "3", "4",
               "--out_subdir", "realism2", "--save_renders"])
    runner.run("eval_realism", "main", ["scripts/eval_realism.py", "--runs",
               *[f"{label}={log_dir}/realism2" for label, log_dir in cfg["realism_compare"].items()], f"{cfg['exp']}={model}/realism2",
               "--seg_model_id", cfg["seg_model_id"], "--out", str(out / "realism.json")])
    runner.run("common_rig", "main", ["scripts/render_sweep_comparison.py", "--init_log_dir", cfg["rig_init_log_dir"],
               "--log_dirs", *cfg["rig_compare"].values(), model, "--labels", cfg["rig_init_label"], *cfg["rig_compare"].keys(),
               f"{cfg['exp']}_gen", "--frames", *map(str, cfg["rig_frames"]), "--yaws", *map(str, cfg["rig_yaws"]),
               "--rights", *map(str, cfg["rig_rights_scene_units"]), "--out_dir", str(out / "renders" / "common")])
    runner.complete()


if __name__ == "__main__":
    main()
