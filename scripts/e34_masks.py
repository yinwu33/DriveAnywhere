"""E34 (docs/E34_DATA.md): Phase 4 masks for every camera of one training segment, on the original images.

Same models and parameters as scripts/run_masks.py (AGENTS.md section 13.4): Grounding DINO + SAM 2.1 dynamic
masks (dashrecon.e34.masks.GroundedSam2Gpu: mask upsampling and dilation on the GPU, bit-identical to the original
backend in fp32), SegFormer-B5 Cityscapes sky / road. Matmuls and convolutions run in TF32 (--tf32; checked on 30
images of wt683: 3e-5 of the dynamic-mask pixels and 5e-6 of the sky pixels change, SegFormer 2.7x faster). SegFormer runs at --seg_input_width wide with the camera's own aspect ratio
(FRONT 1920x1280 -> 1536x1024, the run_masks.py size; SIDE 1920x886 -> 1536x709). Cameras and frames:
    FRONT (0)                    every frame (the FRONT-only reconstruction and its self-calibration need them all);
    --target_cams (1..4)         every --target_stride-th frame, from frame 0: the frames that enter the joint
                                 self-calibration and become training-pair targets.
Reads only <raw_root>/<idx:03d>/images/<t:03d>_<cam>.jpg. Writes, via dashrecon.io,
    <out_dir>/cam<c>/frames.txt, mask_dynamic/, mask_sky/, mask_road/   (255 = positive, original resolution)
    <out_dir>/meta.json   models, parameters, runtime, peak VRAM, per-camera means and per-frame ratios.

Example (masks venv):
    .venvs/masks/bin/python scripts/e34_masks.py --scene_id wt646 --raw_root data/e34/raw \
        --out_dir data/e34/work/wt646/masks_orig --target_cams 1 2 3 4 --target_stride 2 \
        --detector_id IDEA-Research/grounding-dino-base --sam_id facebook/sam2.1-hiera-large \
        --box_threshold 0.25 --text_threshold 0.25 --dilate_px 5 \
        --seg_model_id nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --seg_input_width 1536 --tf32
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
from dashrecon.e34.layout import camera_frames, cam_dir, raw_image  # noqa: E402
from dashrecon.e34.masks import GroundedSam2Gpu  # noqa: E402
from dashrecon.masks.sam_text import DEFAULT_PROMPTS  # noqa: E402
from dashrecon.masks.segformer import SegformerSkyRoadBackend  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import get_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_cams", type=int, nargs="+", required=True)
    parser.add_argument("--target_stride", type=int, required=True)
    parser.add_argument("--detector_id", required=True)
    parser.add_argument("--sam_id", required=True)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--box_threshold", type=float, required=True)
    parser.add_argument("--text_threshold", type=float, required=True)
    parser.add_argument("--dilate_px", type=int, required=True)
    parser.add_argument("--seg_model_id", required=True)
    parser.add_argument("--seg_input_width", type=int, required=True)
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls / convolutions")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    commit = git_commit()
    scene = get_scene(args.scene_id)
    assert not os.path.exists(args.out_dir), f"{args.out_dir} exists"
    frames = camera_frames(args.raw_root, scene.scene_idx, args.target_cams, args.target_stride)

    dyn = GroundedSam2Gpu(args.detector_id, args.sam_id, tuple(args.prompts), args.box_threshold,
                              args.text_threshold, args.dilate_px)
    segs = {}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    per_cam = {}
    for cam, cam_frames in frames.items():
        out_dir = cam_dir(args.out_dir, cam)
        rows = []
        for t in cam_frames:
            image = Image.open(raw_image(args.raw_root, scene.scene_idx, int(t), cam)).convert("RGB")
            w, h = image.size
            hw = (int(round(args.seg_input_width * h / w)), args.seg_input_width)
            if hw not in segs:
                segs[hw] = SegformerSkyRoadBackend(args.seg_model_id, hw)
            dyn_masks, dyn_info = dyn.predict(image)
            seg_masks, _ = segs[hw].predict(image)
            masks = {**dyn_masks, **seg_masks}
            assert set(masks) == set(io.MASK_KINDS), set(masks)
            for kind, m in masks.items():
                io.write_mask(out_dir, kind, int(t), m)
            rows.append({"frame": int(t), **{kind: float(masks[kind].mean()) for kind in io.MASK_KINDS},
                         "num_boxes": dyn_info["num_boxes"]})
        io.write_frames(out_dir, cam_frames)
        per_cam[str(cam)] = {"image_hw": [h, w], "seg_input_hw": list(hw), "num_frames": len(cam_frames),
                             **{f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in io.MASK_KINDS},
                             "per_frame": rows}
        print(f"[e34_masks] {args.scene_id} cam {cam}: {len(cam_frames)} frames, dynamic "
              f"{per_cam[str(cam)]['dynamic_mean']:.3f}, sky {per_cam[str(cam)]['sky_mean']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    runtime = time.time() - t0
    n_img = sum(len(v) for v in frames.values())
    io.write_meta(args.out_dir, {
        "scene_id": args.scene_id, "segment": scene.segment, "raw_root": args.raw_root,
        "cameras": {str(c): {"frames": [int(v[0]), int(v[-1])], "stride": int(v[1] - v[0])} for c, v in frames.items()},
        "dynamic": {**dyn.meta(), "implementation": "dashrecon.e34.masks.GroundedSam2Gpu"}, "tf32": args.tf32,
        "semantic": {str(hw): s.meta() for hw, s in segs.items()},
        "runtime_s": runtime, "seconds_per_image": runtime / n_img,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "uses_oracle": False, "uses_calibration": False,
        "oracle_note": "camera images only; no GT boxes / poses / calibration read",
        "dashrecon_commit": commit, "per_camera": per_cam,
    })
    print(f"[e34_masks] {args.scene_id}: {n_img} images in {runtime:.0f} s ({runtime / n_img:.2f} s/image) -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
