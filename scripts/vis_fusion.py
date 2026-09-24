"""Qualitative Phase 5 visualisation data for the web viewer.

Reads (via dashrecon.io) ``points_fused.ply``, ``diagnostics/consistency_rejected.ply`` and ``meta.json`` of
one fusion directory and writes ``<fusion_dir>/vis/fusion_view.js`` (registers ``window.FUSION[<scene_id>]``):
a seeded random sample of the fused cloud (uint16-quantised positions, uint8 colours and labels, int8
normals), a sample of consistency-rejected points, and the per-step counts.

Example (main venv):
    .venvs/main/bin/python scripts/vis_fusion.py --scene_id val056 \
        --fusion_dir data/dashrecon/val056/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer \
        --max_points 250000 --rejected_points 100000 --seed 0
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quantise(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint16 positions relative to the bounding box."""
    lo, hi = xyz.min(0), xyz.max(0)
    return np.round((xyz - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype("<u2"), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--max_points", type=int, required=True)
    parser.add_argument("--rejected_points", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    meta = io.read_meta(args.fusion_dir)
    fused = io.read_ply(os.path.join(args.fusion_dir, "points_fused.ply"))
    pick = rng.choice(len(fused["xyz"]), size=min(args.max_points, len(fused["xyz"])), replace=False)
    q, lo, hi = quantise(fused["xyz"][pick])
    nrm = np.clip(np.rint(fused["normals"][pick] * 127), -127, 127).astype(np.int8)
    rej = io.read_ply(os.path.join(args.fusion_dir, "diagnostics", "consistency_rejected.ply"))["xyz"]
    rpick = rng.choice(len(rej), size=min(args.rejected_points, len(rej)), replace=False)
    rq, rlo, rhi = quantise(rej[rpick])

    payload = {
        "id": args.scene_id,
        "n": int(len(pick)),
        "n_total": int(len(fused["xyz"])),
        "lo": lo.tolist(), "hi": hi.tolist(),
        "pos": b64(q), "col": b64(fused["rgb"][pick]), "lab": b64(fused["labels"][pick]), "nrm": b64(nrm),
        "label_names": meta["label_names"],
        "rejected": {"n": int(len(rpick)), "n_total": meta["rejected_by_consistency"],
                     "lo": rlo.tolist(), "hi": rhi.tolist(), "pos": b64(rq)},
    }
    out_dir = os.path.join(args.fusion_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fusion_view.js"), "w") as f:
        f.write(f"window.FUSION = window.FUSION || {{}};\nwindow.FUSION[{json.dumps(args.scene_id)}] = {json.dumps(payload)};\n")
    with open(os.path.join(out_dir, "vis_params.json"), "w") as f:
        json.dump({**vars(args), "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[vis_fusion] {args.scene_id}: {len(pick):,} of {len(fused['xyz']):,} fused points, "
          f"{len(rpick):,} rejected -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
