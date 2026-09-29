"""Assemble the web viewer (DashRecon Pose Review: Phase 3 point clouds, Phase 4 masks) from pipeline outputs.

Inputs:
    dashrecon/viewer/index.html                          template (committed)
    <scene_root>/<scene_id>/<tag>/vis/viewer_data.js     from scripts/vis_pose.py
    <scene_root>/<scene_id>/<tag>/vis/vis_params.json    from scripts/vis_pose.py
    --diagnostics JSON                                   from scripts/diagnose_pose.py
    <scene_root>/<scene_id>/<mask_tag>/vis/masks_view.js from scripts/vis_masks.py (optional, Phase 4)
    <scene_root>/<scene_id>/<tag>__<mask_tag>/vis/fusion_view.js from scripts/vis_fusion.py (optional, Phase 5)
    <scene_root>/<scene_id>/<tag>__<mask_tag>/vis/mesh_view.js   from scripts/vis_mesh.py (optional, Phase 7 mesh)
    <training_dir>/{summary.json, <scene>_<t>.jpg, <scene>_gen_*.jpg}   from scripts/vis_training.py (optional, Phase 6 / 8)
    <results_root>/<exp>/<scene>/renders/splat_view.js            from scripts/export_splats.py, for each of --splat_exps
                                                                  (optional, Phase 6 / 8 3DGS); --splat_run_dirs EXP=DIR
                                                                  takes <DIR>/renders/splat_view.js instead (Phase 9 runs keep
                                                                  their model in results/<exp>/<scene>/model; one scene only)
--scenes restricts the page to some of the development scenes (e.g. val056 while Phase 9 iterates on it).
Output directory:
    index.html, config.js (window.VIEWER_CONFIG), scenes/<scene_id>.js, scenes/<scene_id>.masks.js,
    scenes/<scene_id>.fusion.js, scenes/<scene_id>.mesh.js, scenes/<scene_id>.<exp>.splat.js, training/<scene>_<t>.jpg, training/<scene>_gen_*.jpg

Open <out_dir>/index.html in a browser (or `python -m http.server` inside it). three.js loads from CDN.
The same directory can be published as a Claude Artifact (index.html + config.js + scenes/*.js).

Example:
    python3 scripts/build_viewer.py --scene_root data/dashrecon --tag pose-mapanything_depth-mapanything \
        --diagnostics data/dashrecon/diagnostics/phase3_pose.json --page_title "DashRecon Pose Review" \
        --mask_tag mask-gsam2_sky-segformer --fusion --mesh --training_dir results/_vis \
        --results_root results --splat_exps E5 E6 \
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
    parser.add_argument("--page_title", required=True, help="<title> of the built page, e.g. DashRecon Pose Review")
    parser.add_argument("--mask_tag", default=None, help="Phase 4 mask backend tag; adds the mask panel")
    parser.add_argument("--fusion", action="store_true", help="add Phase 5 fusion (<tag>__<mask_tag>); needs --mask_tag")
    parser.add_argument("--mesh", action="store_true", help="add the Phase 7 NKSR mesh; needs --fusion")
    parser.add_argument("--training_dir", default=None, help="Phase 6 comparison sheets + summary.json (vis_training.py)")
    parser.add_argument("--results_root", default=None, help="Phase 6 / 8 results (for --splat_exps)")
    parser.add_argument("--splat_exps", nargs="+", default=None, help="experiments whose Gaussians the 3DGS mode can show, e.g. E5 E6")
    parser.add_argument("--partial", action="store_true", help="allow scenes without splats while training runs")
    parser.add_argument("--splat_run_dirs", nargs="*", default=[], help="EXP=DIR: that experiment's run directory (one scene only)")
    parser.add_argument("--scenes", nargs="+", default=None, help="scene ids to include (default: all development scenes)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    assert args.splat_exps is None or args.results_root is not None, "--splat_exps needs --results_root"
    assert not args.fusion or args.mask_tag is not None, "--fusion needs --mask_tag"
    assert not args.mesh or args.fusion, "--mesh needs --fusion"
    scenes = DEV_SCENES if args.scenes is None else [sc for sc in DEV_SCENES if sc.scene_id in args.scenes]
    assert args.scenes is None or len(scenes) == len(args.scenes), f"unknown scene in {args.scenes}"
    run_dirs = dict(r.split("=", 1) for r in args.splat_run_dirs)
    assert not run_dirs or len(scenes) == 1, "--splat_run_dirs needs exactly one scene"
    assert set(run_dirs) <= set(args.splat_exps or []), "--splat_run_dirs names an experiment not in --splat_exps"

    with open(args.diagnostics) as f:
        diagnostics = json.load(f)
    assert diagnostics["tag"] == args.tag, (diagnostics["tag"], args.tag)

    os.makedirs(os.path.join(args.out_dir, "scenes"), exist_ok=True)
    vis = None
    for sc in scenes:
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
        for sc in scenes:
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
        for sc in scenes:
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
        for sc in scenes:
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
        for sc in scenes:
            scene = training["scenes"][sc.scene_id]
            for name in scene["sheets"] + [g["file"] for g in scene.get("gen_examples", [])]:
                shutil.copyfile(os.path.join(args.training_dir, name), os.path.join(args.out_dir, "training", name))

    splats = None
    if args.splat_exps is not None:
        splats = {"exps": args.splat_exps, "generative": {}, "scenes": {sc.scene_id: {} for sc in scenes}}
        for exp in args.splat_exps:
            for sc in scenes:
                run_dir = run_dirs[exp] if exp in run_dirs else os.path.join(args.results_root, exp, sc.scene_id)
                src = os.path.join(run_dir, "renders", "splat_view.js")
                if not os.path.exists(src):
                    assert args.partial, f"missing {src} (use --partial while training runs)"
                    continue
                with open(os.path.join(run_dir, "meta.json")) as f:
                    meta = json.load(f)
                generative = "generative" in meta  # only the Phase 8 scripts write the flag, and always as true
                assert not generative or meta["generative"] is True, run_dir
                assert splats["generative"].setdefault(exp, generative) == generative, f"{exp}: mixed generative flags"
                with open(os.path.join(os.path.dirname(src), "splat_params.json")) as f:
                    sp = json.load(f)
                assert sp["exp"] == exp, (src, sp["exp"])
                shutil.copyfile(src, os.path.join(args.out_dir, "scenes", f"{sc.scene_id}.{exp}.splat.js"))
                splats["scenes"][sc.scene_id][exp] = {"kept": sp["kept"], "total": sp["total"], "dashrecon_commit": sp["dashrecon_commit"]}

    # camera source of the pose products: runs with a self-calibration (DECISIONS D13) record camera_dir
    pose_metas = [io.read_meta(io.scene_dir(args.scene_root, sc.scene_id, args.tag)) for sc in scenes]
    calibrated = {"camera_dir" in m for m in pose_metas}
    assert len(calibrated) == 1, "all scenes must come from the same camera pipeline"
    camera = ({"source": "glomap", "distortion_k1_k2": {sc.scene_id: m["distortion_k1_k2"] for sc, m in zip(scenes, pose_metas)}}
              if calibrated.pop() else {"source": "mapanything"})

    config = {
        "tag": args.tag,
        "camera": camera,
        "built_from_commit": git_commit(),
        "vis": vis,
        "scenes": [{"id": s.scene_id, "category": s.category, "note": s.note} for s in scenes],
        "diagnostics": diagnostics,
        "masks": masks,
        "fusion": fusion,
        "mesh": mesh,
        "training": training,
        "splats": splats,
    }
    with open(os.path.join(args.out_dir, "config.js"), "w") as f:
        f.write(f"window.VIEWER_CONFIG = {json.dumps(config, indent=1)};\n")
    with open(TEMPLATE) as f:
        page = f.read()
    default_title = "<title>DashRecon Pose Review</title>"
    assert page.count(default_title) == 1, "template title changed"
    with open(os.path.join(args.out_dir, "index.html"), "w") as f:
        f.write(page.replace(default_title, f"<title>{args.page_title}</title>"))
    print(f"[build_viewer] wrote {args.out_dir} ({len(scenes)} scenes)", flush=True)


if __name__ == "__main__":
    main()
