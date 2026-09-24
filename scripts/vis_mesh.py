"""Web-viewer data for the Phase 7 NKSR mesh.

Reads ``<fusion_dir>/mesh_nksr.ply`` and ``mesh_nksr.json`` via dashrecon.io, decimates the mesh for display
with Open3D quadric decimation (vertex colours are carried along), and writes
``<fusion_dir>/vis/mesh_view.js`` (registers ``window.MESH[<scene_id>]``: uint16-quantised vertices,
uint32 faces, uint8 colours, and the full-mesh statistics). The full-resolution mesh stays on disk.

Example (main venv):
    .venvs/main/bin/python scripts/vis_mesh.py --scene_id val056 \
        --fusion_dir data/dashrecon/val056/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer \
        --target_faces 150000
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


def main() -> None:
    import open3d as o3d

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--target_faces", type=int, required=True)
    args = parser.parse_args()

    mesh = io.read_mesh_ply(os.path.join(args.fusion_dir, "mesh_nksr.ply"))
    info = io.read_json(os.path.join(args.fusion_dir, "mesh_nksr.json"))
    tm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh["vertices"].astype(np.float64)),
                                   o3d.utility.Vector3iVector(mesh["faces"]))
    tm.vertex_colors = o3d.utility.Vector3dVector(mesh["rgb"].astype(np.float64) / 255.0)
    if len(mesh["faces"]) > args.target_faces:
        tm = tm.simplify_quadric_decimation(target_number_of_triangles=args.target_faces)
    tm.remove_unreferenced_vertices()
    v = np.asarray(tm.vertices)
    f = np.asarray(tm.triangles).astype("<u4")
    c = np.clip(np.rint(np.asarray(tm.vertex_colors) * 255), 0, 255).astype(np.uint8)
    lo, hi = v.min(0), v.max(0)
    q = np.round((v - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype("<u2")

    payload = {
        "id": args.scene_id,
        "nv": int(len(v)), "nf": int(len(f)),
        "lo": lo.tolist(), "hi": hi.tolist(),
        "pos": b64(q), "idx": b64(f), "col": b64(c),
        "full": info["stats"], "runtime_s": info["runtime_s"], "peak_vram_gb": info["peak_vram_gb"],
        "params": info["params"], "model_voxel_size": info["model_voxel_size"],
    }
    out_dir = os.path.join(args.fusion_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mesh_view.js"), "w") as fh:
        fh.write(f"window.MESH = window.MESH || {{}};\nwindow.MESH[{json.dumps(args.scene_id)}] = {json.dumps(payload)};\n")
    io.write_json(os.path.join(out_dir, "mesh_vis_params.json"), {**vars(args), "dashrecon_commit": git_commit()})
    print(f"[vis_mesh] {args.scene_id}: {len(f):,} of {info['stats']['faces']:,} faces -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
