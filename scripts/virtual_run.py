"""A render-time variant of a trained run: same checkpoint, another view gate (D-A4 in docs/EXPERIMENTS.md).

Writes --out_dir (must not exist) with
    view_gate.pt     --gate with the requested changes (--double_sided);
    config.yaml      the run's config with view_gate pointing to it;
    checkpoint_final.pth  a link to the run's checkpoint (nothing is retrained or copied);
    meta.json        what was changed, from which run and gate.
Every script that loads a run from its config (render_sweep_comparison.py, eval_cross_camera.py, view_gs.py, ...) then
renders the same Gaussians through the new gate. Runs trained without a gate can be given one this way too.

Example:
    .venvs/main/bin/python scripts/virtual_run.py --log_dir results/E14/val056/model --gate results/E14/val056/view_gate.pt \
        --double_sided --out_dir results/_diagnostics/DA4/val056/E14_dgate
"""
import argparse
import json
import os
import sys

from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.floaters import ViewGate  # noqa: E402
from dashrecon.gen.novel import view_gate_path  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--gate", required=True, help="view gate to start from (the run's own, or one for a run without)")
    parser.add_argument("--double_sided", action="store_true")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=False)
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    gate = ViewGate.load(args.gate, torch.device("cpu"))
    gate.double_sided = args.double_sided
    out_gate = os.path.join(args.out_dir, "view_gate.pt")
    gate.save(out_gate, {"derived_from": args.gate, "double_sided": args.double_sided})
    original = view_gate_path(cfg)
    cfg.view_gate = out_gate
    OmegaConf.save(cfg, os.path.join(args.out_dir, "config.yaml"))
    os.symlink(os.path.abspath(os.path.join(args.log_dir, "checkpoint_final.pth")), os.path.join(args.out_dir, "checkpoint_final.pth"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"kind": "render-time variant of a trained run (no training)", "log_dir": args.log_dir,
                   "original_view_gate": original, "gate_from": args.gate, "double_sided": args.double_sided,
                   "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[virtual_run] {args.out_dir}: {args.log_dir} through {out_gate} (double_sided={args.double_sided})", flush=True)


if __name__ == "__main__":
    main()
