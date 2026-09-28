"""Report E13 coverage, self-consistency and persistent 3D fitting separately."""
import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    """Read one recorded stage result without default or invented values."""
    return json.loads(path.read_text())


def main() -> None:
    """Write a report; execution success is not visual or geometric acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.probe_dir
    state, meta = read(out / "run_status.json"), read(out / "meta.json")
    if not state["stages"] or state["stages"][-1]["name"] != "right_back_retention" or any(s["exit_code"] != 0 for s in state["stages"]):
        raise ValueError("cannot summarize an incomplete experiment")
    rounds = []
    for index in [0, 1]:
        views = out / "views" / f"r{index}"
        cameras = read(views / "cams.json")
        depth = read(views / "depth.json")
        validation = read(out / "validation" / f"r{index}" / "validation.json")
        cache = read(views / "cache.json")
        rounds.append({"round": index, "mean_input_hole_fraction": sum(cameras["hole_frac"]) / len(cameras["hole_frac"]),
                       "accepted_hole_fraction": validation["accepted_fraction"], "unknown_hole_fraction": validation["unknown_fraction"],
                       "memory_cache_coverage": cache["memory_coverage"], "scene_scale": depth["alignment"]["scene_scale"],
                       "memory_conflict_pixels": sum(r["memory_conflict_pixels"] for r in validation["frames"]),
                       "before": read(out / f"r{index}_before" / "metrics.json"),
                       "after": read(out / f"r{index}_after" / "metrics.json")})
    model_meta = read(out / "model" / "meta.json")
    metrics = {"exp": "E13", "scene_id": "val056", "kind": "surrounding rig self-consistency and fitting, not GT quality",
               "rounds": rounds, "front_history_protocol": read(out / "model" / "metrics.json"),
               "right_back_retention": read(out / "right_back_retention" / "metrics.json"),
               "geometry_quality_accepted": False, "visual_quality_accepted": False,
               "ground_truth_metrics_computed": False}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = ["# E13 surrounding-camera completion", "",
             "FRONT-only; all newly completed content is generated. No hidden cameras, GT or HUGSIM inputs.", "",
             "Eight directions around three estimated camera centres: front, front-left/right, sides, back-left/right and back.",
             "The two 121-frame clips cover 0→+180° and 0→−180°. The second clip uses validated RGB-D from the first;",
             "each clip updates the same persistent 3D scene, retaining earlier targets and real FRONT anchors.", "",
             "Scene-global MoGe depth scale is anchored to supported estimated geometry and reused in round 2.",
             "Unseen depth is a candidate prior; only translated-view support and no validated-memory contradiction permit supervision.",
             "Unknown areas remain unknown. Completion/execution does not imply valid simulation geometry.", "",
             "| Round | Input holes | Accepted within holes | Unknown within holes | Memory in generator | Scene scale |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rounds:
        lines.append(f"| {r['round']} | {100*r['mean_input_hole_fraction']:.2f}% | {100*r['accepted_hole_fraction']:.2f}% | "
                     f"{100*r['unknown_hole_fraction']:.2f}% | {100*r['memory_cache_coverage']:.2f}% | {r['scene_scale']:.4f} |")
    lines += ["", "Input holes differ by direction and previous model; these averages do not measure post-fit hole reduction.", "",
              "| Round | Spawned Gaussians | Final count | Optimization seconds |",
              "|---|---:|---:|---:|"]
    for r in model_meta["rounds"]:
        lines.append(f"| {r['round']} | {r['spawned']} | {r['gaussians_after']} | {r['runtime_s']:.1f} |")
    lines += ["", "Direct generation / same-camera 3D / untrained yaw +7° / untrained translation +0.5 are retained separately:", "",
              "![Right/back diagnostic](r0_after/036.png)", "", "![Left/back diagnostic](r1_after/036.png)", "",
              "The common camera grid uses 3 positions × 9 yaws (±180 are the same rear direction) × 2 translations:",
              "E5c | E13 right/back | E13 full surround (gen). All share identical estimated reference cameras.", "",
              "![Back at frame 104](renders/common/034_compare.png)", "",
              "[All common cameras](renders/common/comparison.mp4)", "",
              "First-half retention after the second update is recorded in right_back_retention/.", "",
              f"Frozen code: `{meta['dashrecon_commit']}`; runtime {state['runtime_s']/60:.2f} minutes; "
              f"generator Torch allocated peak {meta['peak_vram_gb']:.2f} GiB.", "",
              "Visual verdict is pending inspection; no GT geometry, fair hidden-camera score or simulation-asset acceptance is claimed."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
