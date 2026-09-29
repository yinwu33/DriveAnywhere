"""D-E3 (docs/EXPERIMENTS.md): side-view realism of a round-by-round run after each of its rounds.

D-E2 found that E14's 90 degree rounds lose realism while its 45 degree rounds keep it. Before E20 (only the 45 degree
rounds, from E5f) this measures the same protocol on E14's own checkpoints after round 0, 1 and 2
(<model>/rounds/ckpt_r<k>.pth), each rendered through E14's view gate as a render-time variant (virtual_run.py
--checkpoint, nothing retrained). Per checkpoint: FRONT held-out, the four side cameras over E5c's unseen pixels
(every 5th frame, as E14) and D-E1's side renders (every 2nd frame); then eval_realism.py against the config's
realism_compare runs. EVALUATION: reads GT side cameras.

Example (main venv, clean committed checkout):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python -u scripts/run_de3.py --config configs/dashrecon/DE3_rounds.yaml
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
    runner = StageRunner(out, cfg, commit, args.resume, "D-E3")
    realism = {label: f"{log_dir}/realism2" for label, log_dir in cfg["realism_compare"].items()}
    for label, checkpoint in cfg["checkpoints"].items():
        run = str(out / label)
        runner.run(f"{label}_virtual", "main", ["scripts/virtual_run.py", "--log_dir", cfg["log_dir"], "--gate", cfg["gate"],
                   "--checkpoint", checkpoint, "--out_dir", run])
        runner.run(f"{label}_front", "main", ["scripts/eval_front_heldout.py", "--log_dir", run, "--example_frames", "50", "100", "150"])
        runner.run(f"{label}_xcam", "main", ["scripts/eval_cross_camera.py", "--log_dir", run, "--gt_root", cfg["gt_root"],
                   "--frame_stride", "5", "--alpha", "0.5", "--example_frames", "50", "100", "150", "--cams", "1", "2", "3", "4",
                   "--unseen_dir", cfg["unseen_dir"], "--out_subdir", "cross_camera_p9"])
        runner.run(f"{label}_realism_render", "main", ["scripts/eval_cross_camera.py", "--log_dir", run, "--gt_root", cfg["gt_root"],
                   "--frame_stride", "2", "--alpha", "0.5", "--example_frames", "--cams", "1", "2", "3", "4",
                   "--out_subdir", "realism2", "--save_renders"])
        realism[label] = f"{run}/realism2"
    runner.run("realism", "main", ["scripts/eval_realism.py", "--runs", *[f"{k}={v}" for k, v in realism.items()],
               "--seg_model_id", cfg["seg_model_id"], "--out", str(out / "realism.json")])
    runner.complete()


if __name__ == "__main__":
    main()
