"""E41 (docs/EXPERIMENTS.md): keep only the moving objects in the dynamic mask, without GT.

The Phase 4 dynamic mask (Grounded-SAM-2 on "car / truck / bus / motorcycle / bicycle / person") covers every vehicle,
parked ones included, and the static runs exclude those pixels from every loss, from the fused point cloud and from
the mesh: on val056, which has no moving vehicle, the parked cars are never supervised and render as blurs (user,
10-04). Here every connected region of the mask in every frame is tested for multi-view depth consistency, the same test
as the fusion (dashrecon.fusion.consistency): its pixels are back-projected with the estimated depth and pose and
checked against the depth of the training frames --offsets away on each side (frames, not neighbours: in slow traffic a
moving car moves only a few per cent of its distance within a second, so close frames cannot tell it from a parked one;
val041). A pixel is consistent with at least --consistency_min of them; a region whose consistent share is at least --static_share (of at least --min_tested tested
pixels) is static and leaves the mask; a moving object does not reproject onto itself in the other frames and stays.
Regions with too few tested pixels (far, tiny, low confidence) stay masked. People and two-wheelers are never part of
the static background: the dynamic mask of --always_dir (Grounded-SAM-2 with "person / bicycle / motorcycle" only) is
added back after the test (on val056 two pedestrians walking beside a parked van share its region and were released).

With --offset_step the offsets are frames at that camera step (estimated pose units per frame; the Waymo dev scenes at
10 Hz move about 0.5) and are rescaled to this video's median step, at least 1 frame (G2b: a Mapillary sequence is
sampled every 2-3 m, so 5-20 frames span 10-60 m and close parked cars leave the view or hide behind each other); the
output tag is then gsam2mvd.

Output: <out_root>/<scene_id>/<mask tag with gsam2 -> gsam2mv, or gsam2mvd with --offset_step>/ with mask_dynamic (moving only), the input's sky and
road masks (copied), frames.txt and meta.json (parameters, per-frame regions kept / released). GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/refine_dynamic_masks.py --scene_id val056 \\
        --mask_dir data/dashrecon/val056/mask-gsam2_sky-segformer_img-glomap \\
        --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything --out_root data/dashrecon \\
        --test_stride 10 --conf_percentile 0 --max_depth 60 --offsets 5 10 15 20 --consistency_rel 0.05 \\
        --consistency_min 2 --static_share 0.5 --min_tested 30 \\
        --always_dir data/dashrecon/val056/mask-gsam2_sky-segformer_img-glomap-vru
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.fusion.backproject import backproject, project  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene, train_frame_mask  # noqa: E402


def labels_to_depth_grid(labels: np.ndarray, grid: dict) -> np.ndarray:
    """Nearest resample of an integer label image onto the depth grid (as io.mask_to_depth_grid)."""
    rh, rw = grid["resized_hw"]
    top, left = grid["crop_top_left"]
    th, tw = grid["depth_hw"]
    small = np.asarray(Image.fromarray(labels.astype(np.int32)).resize((rw, rh), resample=Image.NEAREST))
    return small[top:top + th, left:left + tw]


def offset_counts(depths: np.ndarray, ks: np.ndarray, poses: np.ndarray, valid: np.ndarray, i: int, others: list,
                  rel_thresh: float) -> np.ndarray:
    """As dashrecon.fusion.consistency.consistency_counts, against the frames ``others`` only."""
    h, w = depths.shape[1:]
    pts = backproject(depths[i], ks[i], poses[i], valid)
    counts = np.zeros(len(pts), dtype=np.int32)
    for j in others:
        u, v, z = project(pts, ks[j], poses[j])
        ui, vi = np.rint(u), np.rint(v)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dj = np.zeros(len(pts))
        dj[inside] = depths[j][vi[inside].astype(np.int64), ui[inside].astype(np.int64)]
        ok = inside & (dj > 0)
        ok[ok] = np.abs(z[ok] - dj[ok]) / dj[ok] < rel_thresh
        counts += ok
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--test_stride", type=int, required=True)
    parser.add_argument("--conf_percentile", type=float, required=True)
    parser.add_argument("--max_depth", type=float, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True, help="frame offsets checked on each side")
    parser.add_argument("--offset_step", type=float, default=None,
                        help="camera step (pose units per frame) the --offsets are given at; rescaled to this video's step")
    parser.add_argument("--consistency_rel", type=float, required=True)
    parser.add_argument("--consistency_min", type=int, required=True)
    parser.add_argument("--static_share", type=float, required=True)
    parser.add_argument("--min_tested", type=int, required=True)
    parser.add_argument("--always_dir", required=True, help="mask dir whose dynamic mask always stays (people, two-wheelers)")
    args = parser.parse_args()
    commit = git_commit()
    t0 = time.time()
    scene = get_scene(args.scene_id)
    tag = os.path.basename(os.path.normpath(args.mask_dir))
    assert tag.startswith("mask-gsam2_"), tag
    out_tag = "mask-gsam2mvd_" if args.offset_step else "mask-gsam2mv_"
    out_dir = io.scene_dir(args.out_root, args.scene_id, tag.replace("mask-gsam2_", out_tag, 1))
    os.makedirs(out_dir, exist_ok=True)

    frames = io.read_frames(args.pose_dir)
    np.testing.assert_array_equal(io.read_frames(args.mask_dir), frames)
    grid = io.read_meta(args.pose_dir)["depth_grid"]
    poses_all = io.read_poses(args.pose_dir)
    k_all = io.depth_intrinsics(io.read_intrinsics(args.pose_dir), grid)
    train = train_frame_mask(frames, scene.start_timestep, args.test_stride)
    depths_all = np.stack([io.read_depth(args.pose_dir, int(t)) for t in frames])
    confs_all = np.stack([io.read_depth_conf(args.pose_dir, int(t)) for t in frames])
    conf_thr = float(np.percentile(confs_all[train][depths_all[train] > 0], args.conf_percentile))
    # neighbours are training frames only (as the fusion); a test frame is checked against them too
    tr_idx = np.nonzero(train)[0]
    median_step = float(np.median(np.linalg.norm(np.diff(poses_all[:, :3, 3], axis=0), axis=1)))
    offsets = list(args.offsets)
    if args.offset_step:
        offsets = sorted({max(1, int(round(o * args.offset_step / median_step))) for o in args.offsets})
        print(f"[refine_dynamic_masks] median camera step {median_step:.3f}: offsets {args.offsets} -> {offsets} frames", flush=True)

    per_frame, kept_px, released_px = [], 0, 0
    for i, t in enumerate(frames):
        t = int(t)
        dyn = io.read_mask(args.mask_dir, "dynamic", t)
        labels, n_reg = ndimage.label(dyn)
        small = labels_to_depth_grid(labels, grid)
        testable = (small > 0) & (depths_all[i] > 0) & (confs_all[i] >= conf_thr) & (depths_all[i] < args.max_depth)
        # the nearest training frame at each offset (test frames are never neighbours)
        others = sorted({int(tr_idx[np.argmin(np.abs(tr_idx - (i + s * o)))]) for o in offsets for s in (-1, 1)
                         if 0 <= i + s * o < len(frames)} - {i})
        counts = offset_counts(depths_all, k_all, poses_all, testable, i, others, args.consistency_rel)
        ok = counts >= args.consistency_min
        lab_t = small[testable]
        tested = np.bincount(lab_t, minlength=n_reg + 1)
        consistent = np.bincount(lab_t[ok], minlength=n_reg + 1)
        moving = dyn.copy()
        regions = []
        for r in range(1, n_reg + 1):
            share = consistent[r] / tested[r] if tested[r] else 0.0
            static = tested[r] >= args.min_tested and share >= args.static_share
            area = int((labels == r).sum())
            if static:
                moving[labels == r] = False
                released_px += area
            else:
                kept_px += area
            regions.append({"area_px": area, "tested": int(tested[r]), "consistent_share": round(float(share), 3), "static": bool(static)})
        moving |= io.read_mask(args.always_dir, "dynamic", t)
        io.write_mask(out_dir, "dynamic", t, moving)
        per_frame.append({"frame": t, "train": bool(train[i]), "regions": regions})
        if i % 20 == 0:
            n_static = sum(r["static"] for r in regions)
            print(f"[refine_dynamic_masks] frame {t}: {n_reg} regions, {n_static} static ({time.time() - t0:.0f} s)", flush=True)

    for kind in ("sky", "road"):
        dst = os.path.join(out_dir, f"mask_{kind}")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(args.mask_dir, f"mask_{kind}"), dst)
    io.write_frames(out_dir, frames)
    src_meta = io.read_meta(args.mask_dir)
    meta = {k: src_meta[k] for k in ("scene_id", "segment", "image_hw", "image_source", "semantic", "uses_oracle", "uses_calibration")}
    meta.update({"dynamic": {"source": args.mask_dir, "source_dynamic": src_meta["dynamic"],
                             "always": args.always_dir, "always_dynamic": io.read_meta(args.always_dir)["dynamic"], "refinement": "multi-view depth consistency per region",
                             "pose_dir": args.pose_dir, "conf_threshold": conf_thr, "median_step": median_step, "offsets_used": offsets, **{k: v for k, v in vars(args).items() if k not in ("mask_dir", "pose_dir", "out_root")}},
                 "summary": {"released_px": released_px, "kept_px": kept_px, "released_share": released_px / max(1, released_px + kept_px)},
                 "oracle_note": "estimated depth / poses and the Grounded-SAM-2 mask only; no GT boxes",
                 "runtime_s": time.time() - t0, "dashrecon_commit": commit, "per_frame": per_frame})
    io.write_meta(out_dir, meta)
    print(f"[refine_dynamic_masks] {args.scene_id}: released {released_px / max(1, released_px + kept_px):.1%} of the masked pixels as static "
          f"-> {out_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
