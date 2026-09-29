"""D-E2 (docs/EXPERIMENTS.md): where does the side-view realism get lost, in the generator or in the distillation?

For every turning round of a Phase 9 run (config: round -> the real camera that looks about the same way), the fully
turned views at three stages are compared with that real camera's images of the same frames (eval_realism.py: KID,
FID, SegFormer class distribution, sharpness; EVALUATION: GT images as a distribution reference only):
    pre    the render the generator started from (views/r<k>/rgb: E5c, or the run after the earlier rounds, gated);
    gen    the generated frames (views/r<k>/filled);
    final  the finished run rendered at the same cameras (render_at_views.py).
Stages via dashrecon.stage_runner; output in <output_dir>/r<k>/ (links and realism.json).

Example (main venv, clean committed checkout):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 HF_HUB_OFFLINE=1 \
        .venvs/main/bin/python -u scripts/run_de2.py --config configs/dashrecon/DE2_stages.yaml
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
    src, out = Path(cfg["source_dir"]), Path(cfg["output_dir"])
    runner = StageRunner(out, cfg, commit, args.resume, "D-E2")
    for k, cam in cfg["rounds"].items():
        views, rd = str(src / "views" / f"r{k}"), out / f"r{k}"
        final = str(rd / "final_render")
        runner.run(f"r{k}_final_render", "main", ["scripts/render_at_views.py", "--log_dir", cfg["final_log_dir"],
                   "--views_dir", views, "--out_dir", final])
        stages = {"pre": (views, "rgb"), "gen": (views, "filled"), "final": (final, "rgb")}
        for stage, (vd, source) in stages.items():
            runner.run(f"r{k}_link_{stage}", "main", ["scripts/link_generated_for_realism.py", "--views_dir", vd, "--cam", str(cam),
                       "--refs_dir", cfg["refs_dir"], "--out_dir", str(rd / stage), "--source", source])
        runner.run(f"r{k}_realism", "main", ["scripts/eval_realism.py", "--runs", *[f"{s}={rd / s}" for s in stages],
                   "--seg_model_id", cfg["seg_model_id"], "--out", str(rd / "realism.json")])
    runner.complete()


if __name__ == "__main__":
    main()
