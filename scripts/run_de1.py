"""D-E1 (docs/EXPERIMENTS.md): side-camera renders of every listed model (eval_cross_camera.py --save_renders; no
unseen-region rescoring, whose saved E5c masks exist only every 5th frame) and their distribution-level realism
(eval_realism.py).
EVALUATION: reads GT side images. Stages via dashrecon.stage_runner; output in <output_dir> (summary.json) and in each
model's <log_dir>/<out_subdir>.

Example (main venv, clean committed checkout):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python -u scripts/run_de1.py --config configs/dashrecon/DE1_realism.yaml
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
    runner = StageRunner(out, cfg, commit, args.resume, "D-E1")
    for label, log_dir in cfg["runs"].items():
        runner.run(f"render_{label}", "main", ["scripts/eval_cross_camera.py", "--log_dir", log_dir, "--gt_root", cfg["gt_root"],
                   "--frame_stride", str(cfg["frame_stride"]), "--alpha", "0.5", "--example_frames", "--cams", "1", "2", "3", "4",
                   "--out_subdir", cfg["out_subdir"], "--save_renders"])
    runner.run("realism", "main", ["scripts/eval_realism.py", "--runs",
               *[f"{label}={log_dir}/{cfg['out_subdir']}" for label, log_dir in cfg["runs"].items()],
               "--seg_model_id", cfg["seg_model_id"], "--out", str(out / "summary.json")])
    runner.complete()


if __name__ == "__main__":
    main()
