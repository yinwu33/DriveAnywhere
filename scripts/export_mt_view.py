"""MT1 (docs/EXPERIMENTS.md): cameras and skies of a combined multi-traversal scene for the browser 3D page.

Writes <out_dir>/<scene_id>.view.js, registering ``window.MTVIEW`` with
    traversals   per member of the combined pose dir: name, virtual frame range, the estimated FRONT camera-to-world
                 poses (3 x 4, OpenCV, the poses the runs were trained with before CamPose refinement) and vertical
                 field of view;
    skies        per --runs entry LABEL=LOG_DIR[@K]: the run's Sky model sampled on an equirectangular grid (azimuth
                 -180..180 deg around world z, elevation -90..90 deg; world x forward, y left, z up), as a PNG data URI.
                 ``@K`` picks traversal K of a per-traversal sky (dashrecon.train.traversal.TraversalEnvLight);
                 without it such a sky is rendered as the mean of its traversals, as the runs do for appearance -1.
The sky lookup is EnvLight.forward (models/modules.py): nvdiffrast cube texture, directions in OpenGL axes.

Example (main venv):
    .venvs/main/bin/python scripts/export_mt_view.py --pose_dir data/dashrecon/mt1/pose-glomap-mt1_depth-mapanything \
        --runs MT1A=results/MT1A/mt1 MT1=results/MT1/mt1 MT1app-A=results/MT1app/mt1@0 MT1app-B=results/MT1app/mt1@1 \
        --scene_id mt1 --sky_hw 256 512 --out_dir scratch/mt1_3d
"""
import argparse
import base64
import io as pyio
import json
import os
import sys

import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

TO_OPENGL = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def sky_panorama(base: torch.Tensor, h: int, w: int) -> np.ndarray:
    """(h, w, 3) uint8 equirectangular image of a (6, R, R, 3) cube map; row 0 = elevation +90, column 0 = azimuth -180."""
    el = torch.deg2rad(90.0 - (torch.arange(h) + 0.5) * 180.0 / h)
    az = torch.deg2rad(-180.0 + (torch.arange(w) + 0.5) * 360.0 / w)
    el, az = torch.meshgrid(el, az, indexing="ij")
    d = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1)
    l = (d.reshape(-1, 3) @ TO_OPENGL.T).reshape(1, h, w, 3).cuda().contiguous()
    rgb = dr.texture(base[None].cuda().contiguous(), l, filter_mode="linear", boundary_mode="cube")[0]
    return (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()


def data_uri(img: np.ndarray) -> str:
    buf = pyio.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="LABEL=LOG_DIR or LABEL=LOG_DIR@TRAVERSAL")
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--sky_hw", type=int, nargs=2, required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    meta = io.read_meta(args.pose_dir)
    poses, ks = io.read_poses(args.pose_dir), io.read_intrinsics(args.pose_dir)
    h_img = 2 * ks[0, 1, 2] + 1  # principal point at the image centre (pixel centres at integers)
    traversals = []
    for m in meta["members"]:
        a, b = meta["traversals"][m]["virtual_frames"]
        traversals.append({"name": m, "frames": [a, b], "c2w": np.round(poses[a:b, :3, :4], 5).tolist(),
                           "vfov_deg": float(np.degrees(2 * np.arctan(0.5 * h_img / ks[a, 1, 1])))})
    skies = {}
    for spec in args.runs:
        label, rest = spec.split("=", 1)
        log_dir, k = (rest.split("@") + [None])[:2]
        base = torch.load(os.path.join(log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)["models"]["Sky"]["base"]
        assert base.ndim == 5 or k is None, f"{spec}: @traversal needs a per-traversal sky, got {tuple(base.shape)}"
        cube = base.mean(dim=0) if base.ndim == 5 and k is None else (base[int(k)] if base.ndim == 5 else base)
        assert cube.ndim == 4 and cube.shape[0] == 6, (spec, tuple(base.shape))
        skies[label] = data_uri(sky_panorama(cube.float(), *args.sky_hw))
        print(f"[export_mt_view] sky {label}: {spec}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.scene_id}.view.js")
    with open(out, "w") as f:
        f.write(f"window.MTVIEW = {json.dumps({'scene_id': args.scene_id, 'traversals': traversals, 'skies': skies, 'pose_dir': args.pose_dir, 'dashrecon_commit': git_commit()})};\n")
    print(f"[export_mt_view] {out}: {len(traversals)} traversals, {len(skies)} skies", flush=True)


if __name__ == "__main__":
    main()
