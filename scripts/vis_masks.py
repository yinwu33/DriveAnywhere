"""Qualitative Phase 4 visualisation: mask overlays on FRONT frames and per-frame masked-pixel ratios.

Reads the Phase 4 products of one scene (via ``dashrecon.io``) and the FRONT images, and writes
``<mask_dir>/vis/masks_view.js`` (registers ``window.MASKS[<scene_id>]`` for the web viewer) plus
``vis_params.json``. Overlay colours are the viewer's validated categorical slots 1-3
(light-mode steps): sky blue, dynamic orange, road aqua.

Example (masks venv):
    .venvs/masks/bin/python scripts/vis_masks.py --scene_id val041 \
        --mask_dir data/dashrecon/val041/mask-gsam2_sky-segformer \
        --processed_root data/waymo/processed/validation --thumb_stride 10 --thumb_width 640
"""
import argparse
import base64
import io as pyio
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import FRONT_CAM_ID, get_scene  # noqa: E402

OVERLAY_RGB = {"road": (0x1B, 0xAF, 0x7A), "sky": (0x2A, 0x78, 0xD6), "dynamic": (0xEB, 0x68, 0x34)}
OVERLAY_ORDER = ("road", "sky", "dynamic")  # later kinds are painted on top
OVERLAY_ALPHA = 0.5


def overlay(image: Image.Image, masks: dict[str, np.ndarray], width: int) -> str:
    """Downscaled JPEG (data URI) of the image with semi-transparent mask fills."""
    height = round(image.size[1] * width / image.size[0])
    rgb = np.asarray(image.resize((width, height), resample=Image.BILINEAR)).astype(np.float32)
    for kind in OVERLAY_ORDER:
        small = np.asarray(Image.fromarray(masks[kind].astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) == 255
        rgb[small] = (1 - OVERLAY_ALPHA) * rgb[small] + OVERLAY_ALPHA * np.array(OVERLAY_RGB[kind], dtype=np.float32)
    buf = pyio.BytesIO()
    Image.fromarray(rgb.round().astype(np.uint8)).save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--thumb_stride", type=int, required=True)
    parser.add_argument("--thumb_width", type=int, required=True)
    args = parser.parse_args()

    scene = get_scene(args.scene_id)
    frames = io.read_frames(args.mask_dir)
    meta = io.read_meta(args.mask_dir)
    img_dir = os.path.join(args.processed_root, f"{scene.scene_idx:03d}", "images")
    per_frame = meta["per_frame"]
    assert [f["frame"] for f in per_frame] == frames.tolist()

    sel = frames[:: args.thumb_stride]
    thumbs = []
    for t in sel:
        image = Image.open(os.path.join(img_dir, f"{int(t):03d}_{FRONT_CAM_ID}.jpg")).convert("RGB")
        masks = {kind: io.read_mask(args.mask_dir, kind, int(t)) for kind in io.MASK_KINDS}
        thumbs.append(overlay(image, masks, args.thumb_width))

    payload = {
        "id": args.scene_id,
        "thumb_frames": sel.tolist(),
        "thumbs": thumbs,
        "frames": frames.tolist(),
        "ratio": {kind: [round(f[kind], 5) for f in per_frame] for kind in io.MASK_KINDS},
        "boxes": [f["num_boxes"] for f in per_frame],
        "summary": meta["summary"],
        "seconds_per_frame": meta["seconds_per_frame"],
        "peak_vram_gb": meta["peak_vram_gb"],
    }
    out_dir = os.path.join(args.mask_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "masks_view.js"), "w") as f:
        f.write(f"window.MASKS = window.MASKS || {{}};\nwindow.MASKS[{json.dumps(args.scene_id)}] = {json.dumps(payload)};\n")
    with open(os.path.join(out_dir, "vis_params.json"), "w") as f:
        json.dump({**vars(args), "overlay_rgb": OVERLAY_RGB, "overlay_alpha": OVERLAY_ALPHA,
                   "dashrecon_commit": git_commit()}, f, indent=2)
    print(f"[vis_masks] {args.scene_id}: {len(thumbs)} overlays -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
