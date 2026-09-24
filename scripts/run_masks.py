"""Phase 4 entry point: dynamic, sky and road masks for one scene from FRONT images only.

Reads only ``<processed_root>/<scene_idx:03d>/images/<t:03d>_0.jpg`` (no GT boxes: AGENTS.md
section 10) and writes, via ``dashrecon.io``, to ``<out_root>/<scene_id>/mask-gsam2_sky-segformer/``:
    frames.txt, mask_dynamic/, mask_sky/, mask_road/ (uint8 PNG, 255 = positive, original resolution),
    meta.json (models, parameters, runtime, peak VRAM, per-frame masked-pixel ratios and detections).

Masks do not depend on the pose backend, so they live in their own backend-tag directory
(DECISIONS F).

Example (masks venv):
    .venvs/masks/bin/python scripts/run_masks.py --scene_id val041 \
        --processed_root data/waymo/processed/validation --out_root data/dashrecon \
        --detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large \
        --box_threshold 0.25 --text_threshold 0.25 --dilate_px 5 \
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_hw 1024 1536
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.masks.sam_text import DEFAULT_PROMPTS, GroundedSam2Backend  # noqa: E402
from dashrecon.masks.segformer import SegformerSkyRoadBackend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import front_image_paths, get_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--detector_id", required=True)
    parser.add_argument("--sam_id", required=True)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--box_threshold", type=float, required=True)
    parser.add_argument("--text_threshold", type=float, required=True)
    parser.add_argument("--dilate_px", type=int, required=True)
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--seg_input_hw", type=int, nargs=2, required=True)
    args = parser.parse_args()

    commit = git_commit()  # at start: later edits must not change what this run records
    scene = get_scene(args.scene_id)
    frames, paths = front_image_paths(
        os.path.join(args.processed_root, f"{scene.scene_idx:03d}"), scene.start_timestep, scene.end_timestep
    )
    dyn = GroundedSam2Backend(args.detector_id, args.sam_id, tuple(args.prompts), args.box_threshold,
                              args.text_threshold, args.dilate_px)
    seg = SegformerSkyRoadBackend(args.seg_model_id, tuple(args.seg_input_hw))
    out_dir = io.scene_dir(args.out_root, args.scene_id, f"mask-{dyn.name}_sky-{seg.name}")
    print(f"[run_masks] {args.scene_id}: {len(paths)} FRONT frames -> {out_dir}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    per_frame = []
    image_hw = None
    for t, path in zip(frames, paths):
        image = Image.open(path).convert("RGB")
        hw = [image.size[1], image.size[0]]
        assert image_hw is None or hw == image_hw, (hw, image_hw)
        image_hw = hw
        dyn_masks, dyn_info = dyn.predict(image)
        seg_masks, _ = seg.predict(image)
        masks = {**dyn_masks, **seg_masks}
        assert set(masks) == set(io.MASK_KINDS), set(masks)
        for kind, m in masks.items():
            io.write_mask(out_dir, kind, int(t), m)
        per_frame.append({"frame": int(t), **{kind: float(masks[kind].mean()) for kind in io.MASK_KINDS}, **dyn_info})
    runtime = time.time() - t0
    io.write_frames(out_dir, frames)

    summary = {kind: {"mean": float(np.mean([f[kind] for f in per_frame])), "max": float(np.max([f[kind] for f in per_frame]))}
               for kind in io.MASK_KINDS}
    summary["num_boxes_mean"] = float(np.mean([f["num_boxes"] for f in per_frame]))
    io.write_meta(out_dir, {
        "scene_id": args.scene_id,
        "segment": scene.segment,
        "image_hw": image_hw,
        "dynamic": dyn.meta(),
        "semantic": seg.meta(),
        "runtime_s": runtime,
        "seconds_per_frame": runtime / len(paths),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "uses_oracle": False,
        "uses_calibration": False,
        "oracle_note": "FRONT images only; no GT boxes / poses / calibration read",
        "dashrecon_commit": commit,
        "summary": summary,
        "per_frame": per_frame,
    })
    print(f"[run_masks] {args.scene_id}: {runtime:.0f}s ({runtime / len(paths):.2f}s/frame); mean masked "
          + ", ".join(f"{k} {summary[k]['mean']:.3f}" for k in io.MASK_KINDS)
          + f"; {summary['num_boxes_mean']:.1f} boxes/frame", flush=True)


if __name__ == "__main__":
    main()
