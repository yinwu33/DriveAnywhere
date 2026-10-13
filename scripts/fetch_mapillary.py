"""Download a Mapillary image sequence as a dashrecon scene (user 10-09: an online street dashcam video through the E54s
pipeline; AGENTS.md section 14). Mapillary images are CC BY-SA 4.0 (attribution: the sequence's contributor and
Mapillary); faces and licence plates are blurred by Mapillary.

The token is read from MAPILLARY_ACCESS_TOKEN in the environment or in --env_file (never printed). Images
--start .. --start + --count of --sequence (in Mapillary's sequence order) are downloaded at --thumb resolution,
optionally cut at the bottom / top by --crop_bottom / --crop_top (fractions of the height: the bonnet / dashboard of a
windscreen-mounted camera, a dashcam's burned-in timestamp or logo; static in every frame), centre-cropped to the 3:2 aspect of the pipeline's camera loader (DATASETS_CONFIG waymo FRONT, 1920 x 1280; the crop
removes the bonnet / sky strips of 16:9 or 4:3 frames) and resized to --size, then written as
<out_root>/<scene_idx:03d>/images/<t:03d>_0.jpg (the layout of the Waymo processed scenes, FRONT camera 0).
The pipeline itself uses nothing else: Mapillary's camera parameters, GPS and computed poses go to mapillary.json
only (flagged "not used by the reconstruction"; GPS may set the viewer's metre scale, like GT on Waymo).

Example (main venv):
    .venvs/main/bin/python scripts/fetch_mapillary.py --sequence <id> --start 0 --count 200 --scene_idx 950 \\
        --out_root data/mapillary/processed --env_file .env
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

API = "https://graph.mapillary.com"
FIELDS = "id,captured_at,camera_type,make,model,width,height,geometry,computed_geometry,compass_angle,camera_parameters,creator,is_pano"


def token(env_file: str) -> str:
    tok = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not tok and env_file and os.path.exists(env_file):
        for line in open(env_file):
            if line.strip().startswith("MAPILLARY_ACCESS_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert tok, "MAPILLARY_ACCESS_TOKEN not found"
    return tok


def get(tok: str, path: str, **params):
    params["access_token"] = tok
    for k in range(5):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"Mapillary request failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--scene_idx", type=int, required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--thumb", default="thumb_2048_url", choices=["thumb_1024_url", "thumb_2048_url", "thumb_original_url"])
    parser.add_argument("--size", type=int, nargs=2, default=[1920, 1280], help="output width height (3:2)")
    parser.add_argument("--crop_bottom", type=float, default=0.0, help="fraction of the height cut at the bottom first")
    parser.add_argument("--crop_top", type=float, default=0.0, help="fraction of the height cut at the top first")
    args = parser.parse_args()
    t0 = time.time()
    tok = token(args.env_file)
    out = os.path.join(args.out_root, f"{args.scene_idx:03d}")
    os.makedirs(os.path.join(out, "images"), exist_ok=False)
    ids = [d["id"] for d in get(tok, "image_ids", sequence_id=args.sequence)["data"]][args.start:args.start + args.count]
    assert len(ids) == args.count, f"sequence has only {len(ids)} images from {args.start}"
    w_out, h_out = args.size
    records = []
    for t, i in enumerate(ids):
        meta = get(tok, i, fields=FIELDS + "," + args.thumb)
        img = Image.open(io.BytesIO(requests.get(meta[args.thumb], timeout=120).content)).convert("RGB")
        w, h0 = img.size
        top = int(round(h0 * args.crop_top))
        h = int(round(h0 * (1.0 - args.crop_bottom))) - top
        if w / h > w_out / h_out:  # too wide: crop the sides
            cw, ch = int(round(h * w_out / h_out)), h
        else:  # too tall: crop top and bottom
            cw, ch = w, int(round(w * h_out / w_out))
        box = ((w - cw) // 2, top + (h - ch) // 2, (w - cw) // 2 + cw, top + (h - ch) // 2 + ch)
        img.crop(box).resize((w_out, h_out), Image.LANCZOS).save(os.path.join(out, "images", f"{t:03d}_0.jpg"), quality=95)
        meta.pop(args.thumb)
        meta.update({"frame": t, "downloaded_size": [w, h0], "crop_box": list(box)})
        records.append(meta)
        if t % 25 == 0:
            print(f"[fetch_mapillary] {t}/{len(ids)} ({time.time() - t0:.0f} s)", flush=True)
    creators = sorted({(r.get("creator") or {}).get("username", "unknown") for r in records})
    with open(os.path.join(out, "mapillary.json"), "w") as f:
        json.dump({"sequence": args.sequence, "start": args.start, "count": args.count, "image_ids": ids,
                   "license": "CC BY-SA 4.0", "attribution": f"Mapillary contributor(s) {', '.join(creators)}, CC BY-SA 4.0, https://www.mapillary.com",
                   "not_used_by_reconstruction": "camera_parameters, geometry, computed_geometry, compass_angle (GPS may set the viewer's metre scale only)",
                   "output_size": [w_out, h_out], "images": records, "params": vars(args), "dashrecon_commit": git_commit(),
                   "runtime_s": time.time() - t0}, f, indent=1)
    print(f"[fetch_mapillary] {len(ids)} images -> {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
