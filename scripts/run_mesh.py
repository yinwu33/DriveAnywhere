"""Phase 7 entry point: NKSR mesh from the Phase 5 fused point cloud (DECISIONS D8).

Reads ``<fusion_dir>/points_fused.ply`` (xyz, oriented normals, rgb, road label) via dashrecon.io and writes
``<fusion_dir>/mesh_nksr.ply`` (AGENTS.md section 5) plus ``<fusion_dir>/mesh_nksr.json`` with parameters,
runtime, peak VRAM and two geometry checks that need no GT:
    road_coverage   fraction of fused road points within ``coverage_radius`` of a mesh vertex
                    (AGENTS Phase 7 acceptance: the mesh covers the main road surface)
    boundary edges  count and total length of edges used by one triangle (mesh borders + hole rims)

Example (nksr venv):
    PATH=.venvs/nksr/bin:/usr/local/cuda-12.1/bin:$PATH .venvs/nksr/bin/python scripts/run_mesh.py \
        --fusion_dir data/dashrecon/val056/pose-mapanything_depth-mapanything__mask-gsam2_sky-segformer \
        --detail_level 0.5 --mise_iter 1 --solver_tol 1e-4 --coverage_radius 0.15
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.mesh.nksr_recon import boundary_edges, reconstruct_mesh  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--detail_level", type=float, required=True)
    parser.add_argument("--mise_iter", type=int, required=True)
    parser.add_argument("--solver_tol", type=float, required=True)
    parser.add_argument("--coverage_radius", type=float, required=True, help="scene units")
    args = parser.parse_args()
    commit = git_commit()

    fused = io.read_ply(os.path.join(args.fusion_dir, "points_fused.ply"))
    fusion_meta = io.read_meta(args.fusion_dir)
    road_code = fusion_meta["label_names"].index("road")
    v, f, c, info = reconstruct_mesh(fused["xyz"], fused["normals"], fused["rgb"], args.detail_level,
                                     args.mise_iter, args.solver_tol)
    io.write_mesh_ply(os.path.join(args.fusion_dir, "mesh_nksr.ply"), v, f, c)

    road = fused["xyz"][fused["labels"] == road_code]
    dist, _ = cKDTree(v).query(road, distance_upper_bound=args.coverage_radius)
    be = boundary_edges(f)
    be_len = np.linalg.norm(v[be[:, 0]] - v[be[:, 1]], axis=1)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    stats = {
        "vertices": int(len(v)),
        "faces": int(len(f)),
        "surface_area": float(area),
        "road_points": int(len(road)),
        "road_coverage": float(np.isfinite(dist).mean()),
        "boundary_edges": int(len(be)),
        "boundary_length": float(be_len.sum()),
    }
    io.write_json(os.path.join(args.fusion_dir, "mesh_nksr.json"), {
        "params": vars(args),
        "input_points": int(len(fused["xyz"])),
        "stats": stats,
        **info,
        "units": "pose-run units (same as points_fused.ply)",
        "uses_oracle": False,
        "dashrecon_commit": commit,
    })
    print(f"[run_mesh] {os.path.basename(os.path.dirname(os.path.normpath(args.fusion_dir)))}: "
          f"{stats['vertices']:,} vertices, {stats['faces']:,} faces, road coverage {stats['road_coverage']:.3f}, "
          f"{stats['boundary_edges']:,} boundary edges; voxel {info['model_voxel_size']}; "
          f"{info['runtime_s']:.0f}s, peak {info['peak_vram_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
