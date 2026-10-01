"""Export a trained run's Background Gaussians for the browser splat renderer (dashrecon/viewer).

Reads <log_dir>/checkpoint_final.pth (drivestudio VanillaGaussians state: _means, _scales (log), _quats,
_features_dc, _opacities (logit)) and keeps the ``max_splats`` Gaussians with the largest
opacity x geometric-mean scale among those with opacity >= ``min_opacity`` and a largest axis below the
``max_scale_pct`` percentile (the few huge Gaussians are mostly unconstrained backdrop that smears over
novel views). Per Gaussian 20 bytes:
    position   uint16 x3, quantised to the kept bounding box
    colour     uint8 x4, RGB from the degree-0 SH coefficient (0.5 + C0 * dc, view-independent) + opacity
    log-scale  float16 x3
    rotation   int8 x4, unit quaternion (w, x, y, z) * 127, gsplat convention
Higher SH bands (view-dependent colour), the Sky environment map and the per-image exposure (Affine) are not
exported. The experiment id is --exp, or else the name of <log_dir>'s parent directory (results/<exp>/<scene_id>).
Runs with a view gate in their config (E14-style Phase 9 runs, D-A3) also get, per kept Gaussian, the gate's observation
axis (int8 x3) and cone half-angle (uint8, degrees) with a flag (1 gated, 0 not), plus margin / fade / double_sided, so
the browser fades them exactly as dashrecon.gen.floaters.cone_gate does for novel views; gated Gaussians that no
training view observed (factor 0 from every novel view) are not exported.
Runs with per-traversal appearance (MT1app, dashrecon.train.traversal.TraversalGaussians) take --traversal K: the colour
is then that traversal's (shared DC + its centred residual), written to splat_view_traversal<K>.js /
splat_params_traversal<K>.json.
Output: <log_dir>/renders/splat_view.js (registers ``window.SPLAT["<scene_id>.<exp>"]``) + splat_params.json.

Example (main venv):
    .venvs/main/bin/python scripts/export_splats.py --log_dir results/E5/val056 --scene_id val056 \
        --max_splats 80000 --min_opacity 0.05 --max_scale_pct 99.5
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.gen.floaters import ViewGate  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

SH_C0 = 0.28209479177387814


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--max_splats", type=int, required=True)
    parser.add_argument("--min_opacity", type=float, required=True)
    parser.add_argument("--max_scale_pct", type=float, required=True, help="drop Gaussians whose largest axis is above this percentile")
    parser.add_argument("--exp", default=None, help="experiment id (default: <log_dir>'s parent directory name)")
    parser.add_argument("--traversal", type=int, default=None, help="per-traversal appearance runs: export this traversal's colours")
    args = parser.parse_args()

    ckpt = torch.load(os.path.join(args.log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    bg = ckpt["models"]["Background"]
    means = bg["_means"].float().numpy()
    log_scales = bg["_scales"].float().numpy()
    quats = bg["_quats"].float().numpy()
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacity = 1.0 / (1.0 + np.exp(-bg["_opacities"].float().numpy()[:, 0]))
    dc = bg["_features_dc"].float().numpy()
    assert (args.traversal is None) == ("_features_dc_tr" not in bg), "--traversal is for (and required by) per-traversal appearance runs"
    if args.traversal is not None:
        res = bg["_features_dc_tr"].float().numpy()
        dc = dc + (res - res.mean(axis=1, keepdims=True))[:, args.traversal]
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    assert log_scales.shape[1] == 3 and rgb.shape[1] == 3, (log_scales.shape, rgb.shape)

    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    gate = ViewGate.load(cfg.view_gate, torch.device("cpu")) if "view_gate" in cfg else None
    gated = np.zeros(len(means), bool)
    visible = np.ones(len(means), bool)
    if gate is not None:
        n_gate = gate.selected.shape[0]
        gated[:n_gate] = gate.selected.numpy()
        visible[:n_gate] = ~gate.selected.numpy() | gate.observed.numpy()
    candidates = np.flatnonzero((opacity >= args.min_opacity) & visible)
    max_axis = log_scales[candidates].max(axis=1)
    candidates = candidates[max_axis <= np.percentile(max_axis, args.max_scale_pct)]
    importance = opacity[candidates] * np.exp(log_scales[candidates].mean(axis=1))
    keep = candidates[np.argsort(-importance)[: args.max_splats]]

    xyz = means[keep]
    lo, hi = xyz.min(0), xyz.max(0)
    pos = np.round((xyz - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype("<u2")
    col = np.concatenate([rgb[keep], opacity[keep, None]], axis=1)
    col = np.clip(np.rint(col * 255), 0, 255).astype(np.uint8)
    scl = log_scales[keep].astype("<f2")
    q = quats[keep]
    q = np.where(q[:, :1] < 0, -q, q)
    rot = np.clip(np.rint(q * 127), -127, 127).astype(np.int8)

    exp = os.path.basename(os.path.dirname(os.path.normpath(args.log_dir))) if args.exp is None else args.exp
    payload = {"id": args.scene_id, "n": int(len(keep)), "n_total": int(len(means)), "lo": lo.tolist(), "hi": hi.tolist(),
               "pos": b64(pos), "col": b64(col), "scl": b64(scl), "rot": b64(rot), "exp": exp, "gate": None}
    if gate is not None:
        n_gate = gate.selected.shape[0]
        in_gate = keep < n_gate
        axis = np.zeros((len(keep), 3), np.float32)
        half = np.zeros(len(keep), np.float32)
        axis[in_gate] = gate.axis.numpy()[keep[in_gate]]
        half[in_gate] = gate.half_angle.numpy()[keep[in_gate]]
        flag = gated[keep].astype(np.uint8)
        payload["gate"] = {"margin": gate.margin, "fade": gate.fade, "double_sided": gate.double_sided, "path": str(cfg.view_gate),
                           "gated_kept": int(flag.sum()),
                           "ax": b64(np.clip(np.rint(axis * 127), -127, 127).astype(np.int8)),
                           "par": b64(np.stack([np.clip(np.rint(half), 0, 180).astype(np.uint8), flag], axis=1))}
    out_dir = os.path.join(args.log_dir, "renders")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if args.traversal is None else f"_traversal{args.traversal}"
    with open(os.path.join(out_dir, f"splat_view{suffix}.js"), "w") as f:
        f.write(f"window.SPLAT = window.SPLAT || {{}};\nwindow.SPLAT[{json.dumps(args.scene_id + '.' + exp)}] = {json.dumps(payload)};\n")
    io.write_json(os.path.join(out_dir, f"splat_params{suffix}.json"), {**vars(args), "exp": exp, "kept": int(len(keep)), "total": int(len(means)),
                                                                  "view_gate": None if gate is None else str(cfg.view_gate),
                                                                  "gated_kept": 0 if gate is None else payload["gate"]["gated_kept"],
                                                                  "dashrecon_commit": git_commit()})
    print(f"[export_splats] {args.log_dir}: {len(keep):,} of {len(means):,} Gaussians "
          f"(opacity >= {args.min_opacity}: {len(candidates):,})", flush=True)


if __name__ == "__main__":
    main()
