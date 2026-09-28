"""Validate generated RGB-D against nearby generated views (self-consistency only).

Writes confidence/<k>.npy, support/<k>.png and validation.json, leaving inputs
untouched. Only original hole pixels can become generation memory. No overlap is
unknown; two agreeing views and no material conflicts are required by default.
Run in any project venv with numpy, Pillow, OpenCV; no GPU or GT needed.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.consistency import RGBDView, accepted_confidence, reprojection_evidence
from dashrecon.provenance import git_commit
from dashrecon.gen.trajectory import memory_candidate_indices, translated_references


def validate(views_dir: Path, out_dir: Path, offsets: list[int], depth_tol: float,
             rgb_tol: float, min_support: int, max_conflict_fraction: float,
             reference_policy: str = "offsets", min_baseline: float = 1.,
             max_references: int = 4, min_parallax_degrees: float = 0.,
             memory_dirs: list[Path] | None = None,
             memory_validation_dirs: list[Path] | None = None) -> dict:
    """Create full-resolution evidence and record per-frame acceptance statistics."""
    if views_dir.resolve() == out_dir.resolve():
        raise ValueError("validation must have its own output directory")
    if not offsets or any(d == 0 for d in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("offsets must be distinct and exclude self (0)")
    out_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("confidence", "support"):
        (out_dir / sub).mkdir()
    cams = json.loads((views_dir / "cams.json").read_text())["cams"]
    memory_dirs = [] if memory_dirs is None else memory_dirs
    memory_validation_dirs = [] if memory_validation_dirs is None else memory_validation_dirs
    if len(memory_dirs) != len(memory_validation_dirs):
        raise ValueError("one validation directory per generated memory required")
    memories = []
    positive_indices = []
    for directory, validation in zip(memory_dirs, memory_validation_dirs):
        report = json.loads((validation / "validation.json").read_text())
        if Path(report["views_dir"]).resolve() != directory.resolve():
            raise ValueError(f"memory validation does not belong to {directory}")
        cameras = json.loads((directory / "cams.json").read_text())["cams"]
        if len(report["frames"]) != len(cameras) or [r["k"] for r in report["frames"]] != list(range(len(cameras))):
            raise ValueError("validation rows must correspond to memory cameras")
        positive_indices.append([r["k"] for r in report["frames"] if r["accepted_pixels"] > 0])
        memories.append((directory, validation, cameras))

    @lru_cache(maxsize=12)
    def load_memory(d: int, j: int) -> RGBDView:
        directory, validation, cameras = memories[d]
        c = cameras[j]
        rgb = np.asarray(Image.open(directory / "filled" / f"{j:03d}.png").convert("RGB"), dtype=np.float32) / 255
        depth = np.load(directory / "filled_depth" / f"{j:03d}.npy").astype(np.float32)
        conf = np.load(validation / "confidence" / f"{j:03d}.npy")
        hole = np.asarray(Image.open(directory / "mask" / f"{j:03d}.png")) > 127
        valid = hole & (conf > 0) & np.isfinite(depth) & (depth > 0)
        return RGBDView(rgb, depth, np.array(c["K"]), np.array(c["c2w"]), valid)

    @lru_cache(maxsize=12)
    def load(k: int) -> RGBDView:
        c = cams[k]
        rgb = np.asarray(Image.open(views_dir / "filled" / f"{k:03d}.png").convert("RGB"), dtype=np.float32) / 255
        depth = np.load(views_dir / "filled_depth" / f"{k:03d}.npy").astype(np.float32)
        if rgb.shape[:2] != depth.shape or list(depth.shape) != c["hw"]:
            raise ValueError(f"RGB/depth/camera sizes differ in view {k}")
        valid = np.isfinite(depth) & (depth > 0)
        return RGBDView(rgb, depth, np.array(c["K"]), np.array(c["c2w"]), valid)

    rows = []
    start = time.time()
    for k in range(len(cams)):
        query = load(k)
        support = np.zeros(query.depth.shape, np.uint16)
        conflict = np.zeros_like(support)
        used = []
        if reference_policy == "translated":
            references = translated_references(cams, k, min_baseline, max_references)
        elif reference_policy == "offsets":
            references = [k + d for d in offsets if 0 <= k + d < len(cams)]
        else:
            raise ValueError(f"unknown reference policy: {reference_policy}")
        for j in references:
            if np.allclose(cams[k]["c2w"], cams[j]["c2w"], atol=1e-6):
                continue  # duplicate cameras are not independent geometric support
            agree, disagree = reprojection_evidence(query, load(j), depth_tol, rgb_tol, min_parallax_degrees)
            support += agree
            conflict += disagree
            used.append(j)
        memory_support = np.zeros_like(support)
        memory_conflict = np.zeros_like(support)
        memory_used = []
        for d, (directory, _, mcams) in enumerate(memories):
            for j in memory_candidate_indices(mcams, cams[k], 16, max_references, "pose", positive_indices[d]):
                agree, disagree = reprojection_evidence(query, load_memory(d, j), depth_tol, rgb_tol)
                memory_support += agree
                memory_conflict += disagree
                memory_used.append({"views_dir": str(directory), "k": j})
        hole = np.asarray(Image.open(views_dir / "mask" / f"{k:03d}.png")) > 127
        conf = accepted_confidence(support, conflict, min_support, max_conflict_fraction) * hole * (memory_conflict == 0)
        np.save(out_dir / "confidence" / f"{k:03d}.npy", conf.astype(np.float16))
        panel = np.zeros((*hole.shape, 3), np.uint8)
        panel[hole] = (100, 100, 100)  # unknown
        panel[hole & ((conflict + memory_conflict) > 0)] = (220, 60, 50)
        panel[conf > 0] = (40, 190, 110)
        Image.fromarray(panel).save(out_dir / "support" / f"{k:03d}.png")
        rows.append({"k": k, "frame": cams[k]["frame"], "references": used,
                     "memory_references": memory_used,
                     "memory_supported_pixels": int((hole & (memory_support > 0)).sum()),
                     "memory_conflict_pixels": int((hole & (memory_conflict > 0)).sum()),
                     "hole_pixels": int(hole.sum()), "accepted_pixels": int((conf > 0).sum()),
                     "conflict_pixels": int((hole & (conflict > 0)).sum()),
                     "unknown_pixels": int((hole & ((support + conflict + memory_support + memory_conflict) == 0)).sum()),
                     "accepted_fraction": float((conf > 0).sum() / max(1, hole.sum()))})
        if k % 10 == 0:
            print(f"[validate] {k}/{len(cams)} accepted {rows[-1]['accepted_fraction']:.3f}", flush=True)
    holes = sum(r["hole_pixels"] for r in rows)
    result = {"views_dir": str(views_dir.resolve()), "kind": "generated_self_consistency_not_ground_truth",
              "params": {"offsets": offsets, "reference_policy": reference_policy,
                         "min_baseline_scene_units": min_baseline, "max_references": max_references,
                         "min_parallax_degrees": min_parallax_degrees, "depth_tol": depth_tol, "rgb_tol": rgb_tol,
                         "memory_dirs": [str(p) for p in memory_dirs],
                         "positive_memory_frame_indices": positive_indices,
                         "memory_conflict_policy": "reject contradicted validated memory; memory not counted as new independent support",
                         "min_support": min_support, "max_conflict_fraction": max_conflict_fraction},
              "accepted_fraction": sum(r["accepted_pixels"] for r in rows) / max(1, holes),
              "unknown_fraction": sum(r["unknown_pixels"] for r in rows) / max(1, holes),
              "frames": rows, "runtime_s": time.time() - start, "dashrecon_commit": git_commit()}
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[-3, -1, 1, 3])
    parser.add_argument("--depth_tol", type=float, default=0.10)
    parser.add_argument("--rgb_tol", type=float, default=0.12)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--max_conflict_fraction", type=float, default=0.25)
    parser.add_argument("--reference_policy", choices=["offsets", "translated"], default="offsets")
    parser.add_argument("--min_baseline", type=float, default=1., help="minimum separation of query/reference centres and between reference centres, in scene units")
    parser.add_argument("--max_references", type=int, default=4)
    parser.add_argument("--min_parallax_degrees", type=float, default=0.)
    parser.add_argument("--memory_dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--memory_validation_dirs", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    result = validate(**vars(args))
    print(f"[validate] accepted {result['accepted_fraction']:.3f}, unknown {result['unknown_fraction']:.3f}")


if __name__ == "__main__":
    main()
