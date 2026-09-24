"""Assemble the web viewer (DashRecon Pose Review: Phase 3 point clouds, Phase 4 masks) from pipeline outputs.

Inputs:
    dashrecon/viewer/index.html                          template (committed)
    <scene_root>/<scene_id>/<tag>/vis/viewer_data.js     from scripts/vis_pose.py
    <scene_root>/<scene_id>/<tag>/vis/vis_params.json    from scripts/vis_pose.py
    --diagnostics JSON                                   from scripts/diagnose_pose.py
    <scene_root>/<scene_id>/<mask_tag>/vis/masks_view.js from scripts/vis_masks.py (optional, Phase 4)
Output directory:
    index.html, config.js (window.VIEWER_CONFIG), scenes/<scene_id>.js, scenes/<scene_id>.masks.js

Open <out_dir>/index.html in a browser (or `python -m http.server` inside it). three.js loads from CDN.
The same directory can be published as a Claude Artifact (index.html + config.js + scenes/*.js).

Example:
    python3 scripts/build_viewer.py --scene_root data/dashrecon --tag pose-mapanything_depth-mapanything \
        --diagnostics data/dashrecon/diagnostics/phase3_pose.json --mask_tag mask-gsam2_sky-segformer \
        --out_dir data/dashrecon/viewer/review
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.provenance import REPO_ROOT, git_commit  # noqa: E402
from dashrecon.scenes import DEV_SCENES  # noqa: E402

TEMPLATE = os.path.join(REPO_ROOT, "dashrecon", "viewer", "index.html")
SHARED_VIS_KEYS = ("frame_stride", "conf_percentile", "max_depth", "max_points", "seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--mask_tag", default=None, help="Phase 4 mask backend tag; adds the mask panel")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.diagnostics) as f:
        diagnostics = json.load(f)
    assert diagnostics["tag"] == args.tag, (diagnostics["tag"], args.tag)

    os.makedirs(os.path.join(args.out_dir, "scenes"), exist_ok=True)
    vis = None
    for sc in DEV_SCENES:
        vis_dir = os.path.join(io.scene_dir(args.scene_root, sc.scene_id, args.tag), "vis")
        shutil.copyfile(os.path.join(vis_dir, "viewer_data.js"), os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.js"))
        with open(os.path.join(vis_dir, "vis_params.json")) as f:
            params = json.load(f)
        shared = {k: params[k] for k in SHARED_VIS_KEYS}
        assert vis is None or vis == shared, f"{sc.scene_id} was visualised with different parameters: {shared} vs {vis}"
        vis = shared

    masks = None
    if args.mask_tag is not None:
        masks = {"tag": args.mask_tag, "scenes": {}}
        for sc in DEV_SCENES:
            mask_dir = io.scene_dir(args.scene_root, sc.scene_id, args.mask_tag)
            meta = io.read_meta(mask_dir)
            shutil.copyfile(os.path.join(mask_dir, "vis", "masks_view.js"),
                            os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.masks.js"))
            masks["dynamic"], masks["semantic"] = meta["dynamic"], meta["semantic"]
            masks["scenes"][sc.scene_id] = {"summary": meta["summary"], "seconds_per_frame": meta["seconds_per_frame"],
                                            "peak_vram_gb": meta["peak_vram_gb"], "dashrecon_commit": meta["dashrecon_commit"]}
            with open(os.path.join(io.scene_dir(args.scene_root, sc.scene_id, args.tag), "vis", "vis_params.json")) as f:
                assert json.load(f)["mask_dir"] is not None, f"{sc.scene_id}: rerun vis_pose.py with --mask_dir for class labels"

    config = {
        "tag": args.tag,
        "built_from_commit": git_commit(),
        "vis": vis,
        "scenes": [{"id": s.scene_id, "category": s.category, "note": s.note} for s in DEV_SCENES],
        "diagnostics": diagnostics,
        "masks": masks,
    }
    with open(os.path.join(args.out_dir, "config.js"), "w") as f:
        f.write(f"window.VIEWER_CONFIG = {json.dumps(config, indent=1)};\n")
    shutil.copyfile(TEMPLATE, os.path.join(args.out_dir, "index.html"))
    print(f"[build_viewer] wrote {args.out_dir} ({len(DEV_SCENES)} scenes)", flush=True)


if __name__ == "__main__":
    main()
