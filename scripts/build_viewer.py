"""Assemble the web viewer (DashRecon Pose Review: Phase 3 point clouds, Phase 4 masks) from pipeline outputs.

Inputs:
    dashrecon/viewer/index.html                          template (committed)
    <scene_root>/<scene_id>/<tag>/vis/viewer_data.js     from scripts/vis_pose.py
    <scene_root>/<scene_id>/<tag>/vis/vis_params.json    from scripts/vis_pose.py
    --diagnostics JSON                                   from scripts/diagnose_pose.py
    <scene_root>/<scene_id>/<mask_tag>/vis/masks_view.js from scripts/vis_masks.py (optional, Phase 4)
    <scene_root>/<scene_id>/<tag>__<mask_tag>/vis/fusion_view.js from scripts/vis_fusion.py (optional, Phase 5)
    <scene_root>/<scene_id>/<tag>__<mask_tag>/vis/mesh_view.js   from scripts/vis_mesh.py (optional, Phase 7 mesh)
    <training_dir>/{summary.json, <scene>_<t>.jpg}                from scripts/vis_training.py (optional, Phase 6)
    <results_root>/<splat_exp>/<scene>/renders/splat_view.js      from scripts/export_splats.py (optional, Phase 6 3DGS)
Output directory:
    index.html, config.js (window.VIEWER_CONFIG), scenes/<scene_id>.js, scenes/<scene_id>.masks.js,
    scenes/<scene_id>.fusion.js, scenes/<scene_id>.mesh.js, scenes/<scene_id>.splat.js, training/<scene>_<t>.jpg

Open <out_dir>/index.html in a browser (or `python -m http.server` inside it). three.js loads from CDN.
The same directory can be published as a Claude Artifact (index.html + config.js + scenes/*.js).

Example:
    python3 scripts/build_viewer.py --scene_root data/dashrecon --tag pose-mapanything_depth-mapanything \
        --diagnostics data/dashrecon/diagnostics/phase3_pose.json --mask_tag mask-gsam2_sky-segformer --fusion --mesh --training_dir results/_vis \
        --results_root results --splat_exp E5 \
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
    parser.add_argument("--fusion", action="store_true", help="add Phase 5 fusion (<tag>__<mask_tag>); needs --mask_tag")
    parser.add_argument("--mesh", action="store_true", help="add the Phase 7 NKSR mesh; needs --fusion")
    parser.add_argument("--training_dir", default=None, help="Phase 6 comparison sheets + summary.json (vis_training.py)")
    parser.add_argument("--results_root", default=None, help="Phase 6 results (for --splat_exp)")
    parser.add_argument("--splat_exp", default=None, help="experiment whose Gaussians the 3DGS mode shows, e.g. E5")
    parser.add_argument("--partial", action="store_true", help="allow scenes without splats while training runs")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    assert args.splat_exp is None or args.results_root is not None, "--splat_exp needs --results_root"
    assert not args.fusion or args.mask_tag is not None, "--fusion needs --mask_tag"
    assert not args.mesh or args.fusion, "--mesh needs --fusion"

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

    fusion = None
    if args.fusion:
        fusion_tag = f"{args.tag}__{args.mask_tag}"
        fusion = {"tag": fusion_tag, "scenes": {}}
        for sc in DEV_SCENES:
            fdir = io.scene_dir(args.scene_root, sc.scene_id, fusion_tag)
            meta = io.read_meta(fdir)
            shutil.copyfile(os.path.join(fdir, "vis", "fusion_view.js"),
                            os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.fusion.js"))
            fusion["params"] = {k: meta["params"][k] for k in (
                "test_stride", "conf_percentile", "max_depth", "consistency_k", "consistency_rel", "consistency_min",
                "voxel", "outlier_nb", "outlier_std", "road_cell", "road_sigma", "road_max_dz", "normal_knn", "skip")}
            fusion["scenes"][sc.scene_id] = {k: meta[k] for k in (
                "counts", "road", "runtime_s", "num_frames_used", "num_frames_total", "conf_threshold",
                "rejected_by_consistency", "dashrecon_commit")}

    mesh = None
    if args.mesh:
        mesh = {"scenes": {}}
        for sc in DEV_SCENES:
            fdir = io.scene_dir(args.scene_root, sc.scene_id, f"{args.tag}__{args.mask_tag}")
            info = io.read_json(os.path.join(fdir, "mesh_nksr.json"))
            shutil.copyfile(os.path.join(fdir, "vis", "mesh_view.js"), os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.mesh.js"))
            mesh["params"] = info["params"]
            mesh["model_voxel_size"] = info["model_voxel_size"]
            mesh["scenes"][sc.scene_id] = {"stats": info["stats"], "runtime_s": info["runtime_s"],
                                           "peak_vram_gb": info["peak_vram_gb"], "input_points": info["input_points"],
                                           "dashrecon_commit": info["dashrecon_commit"]}

    training = None
    if args.training_dir is not None:
        with open(os.path.join(args.training_dir, "summary.json")) as f:
            training = json.load(f)
        os.makedirs(os.path.join(args.out_dir, "training"), exist_ok=True)
        for sc in DEV_SCENES:
            for name in training["scenes"][sc.scene_id]["sheets"]:
                shutil.copyfile(os.path.join(args.training_dir, name), os.path.join(args.out_dir, "training", name))

    splats = None
    if args.splat_exp is not None:
        splats = {"exp": args.splat_exp, "scenes": {}}
        for sc in DEV_SCENES:
            src = os.path.join(args.results_root, args.splat_exp, sc.scene_id, "renders", "splat_view.js")
            if not os.path.exists(src):
                assert args.partial, f"missing {src} (use --partial while training runs)"
                continue
            shutil.copyfile(src, os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.splat.js"))
            with open(os.path.join(os.path.dirname(src), "splat_params.json")) as f:
                sp = json.load(f)
            splats["scenes"][sc.scene_id] = {"kept": sp["kept"], "total": sp["total"], "dashrecon_commit": sp["dashrecon_commit"]}

    config = {
        "tag": args.tag,
        "built_from_commit": git_commit(),
        "vis": vis,
        "scenes": [{"id": s.scene_id, "category": s.category, "note": s.note} for s in DEV_SCENES],
        "diagnostics": diagnostics,
        "masks": masks,
        "fusion": fusion,
        "mesh": mesh,
        "training": training,
        "splats": splats,
    }
    with open(os.path.join(args.out_dir, "config.js"), "w") as f:
        f.write(f"window.VIEWER_CONFIG = {json.dumps(config, indent=1)};\n")
    shutil.copyfile(TEMPLATE, os.path.join(args.out_dir, "index.html"))
    print(f"[build_viewer] wrote {args.out_dir} ({len(DEV_SCENES)} scenes)", flush=True)


if __name__ == "__main__":
    main()
