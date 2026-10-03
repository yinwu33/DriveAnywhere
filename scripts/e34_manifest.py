"""E34 (docs/E34_DATA.md): list every finished pair in data/e34/manifest.json and in Fixer's training-data format.

Scans <pairs_root>/<scene_id>/meta.json of the segments in --segments (data/e34/segments.json, processing order) and
collects the pairs of every accepted segment: per pair the segment, camera, timestep, type ("points" / "gs"), input,
target and loss mask (255 = leave out: dynamic object or outside the original image).

Split by segment (no segment in both): --test_every-th accepted segment in processing order is "test", the rest "train".

Writes
    <pairs_root>/../manifest.json       every pair (paths relative to the manifest's directory), split, per-segment
                                        summary, counts per type and camera
    <pairs_root>/../fixer_<type>.json   for every type: {"train": {id: {"image", "target_image", "prompt", "mask"}},
                                        "test": {...}} with absolute paths, the layout of Fixer's PairedDatasetV2
                                        (.venvs/src/Fixer/src/utils/training_utils.py; "mask" is an extra key that the
                                        stock loader ignores)

Example (any venv):
    python scripts/e34_manifest.py --segments data/e34/segments.json --pairs_root data/e34/pairs --test_every 20 \
        --prompt "remove degradation"
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.e34.layout import CAM_NAMES  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

TYPES = ("points", "gs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--test_every", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(args.pairs_root))
    selected = io.read_json(args.segments)["selected"]
    pairs, seg_rows, accepted, timings = [], [], 0, {}
    for s in selected:
        d = os.path.join(args.pairs_root, s["scene_id"])
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = io.read_meta(d)
        for stage, sec in meta["timing_s"].items():
            timings.setdefault(stage, []).append(sec)
        row = {"scene_id": s["scene_id"], "idx": s["idx"], "location": s["location"], "accepted": meta["accepted"],
               "reasons": meta["reasons"]}
        if meta["accepted"]:
            split = "test" if accepted % args.test_every == args.test_every - 1 else "train"
            accepted += 1
            row.update({"split": split, "pair_counts": meta["pair_counts"]})
            targets = {(r["cam"], r["t"]): r for r in io.read_json(os.path.join(d, "targets.json"))["targets"]}
            for typ in TYPES:
                f = os.path.join(d, f"{typ}.json")
                if not os.path.exists(f):
                    continue
                for r in io.read_json(f)["pairs"]:
                    tg = targets[(r["cam"], r["t"])]
                    rel = lambda p: os.path.relpath(os.path.join(d, p), root)  # noqa: E731
                    pairs.append({"id": f"{s['scene_id']}_{r['t']:03d}_{r['cam']}_{typ}", "segment": s["scene_id"],
                                  "location": s["location"], "camera": CAM_NAMES[r["cam"]], "cam": r["cam"], "t": r["t"],
                                  "type": typ, "split": split, "input": rel(r["input"]), "target": rel(tg["target"]),
                                  "mask": rel(tg["mask"]), "masked_frac": tg["masked_frac"], "coverage": r["coverage"]})
        seg_rows.append(row)
    counts = {}
    for p in pairs:
        key = f"{p['type']}/{p['split']}"
        counts.setdefault(key, {}).setdefault(p["camera"], 0)
        counts[key][p["camera"]] += 1
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"root": root, "paths": "relative to root", "prompt": args.prompt,
                   "split_rule": f"every {args.test_every}-th accepted segment (processing order) is test",
                   "mask_meaning": "255 = excluded from the loss (dynamic object or outside the original image)",
                   "counts": counts,
                   "timing_s": {k: {"segments": len(v), "median": float(np.median(v)), "total": float(np.sum(v))}
                                for k, v in timings.items()},
                   "segments": seg_rows, "pairs": pairs, "dashrecon_commit": git_commit()}, f, indent=1)
    for typ in TYPES:
        sel = [p for p in pairs if p["type"] == typ]
        if not sel:
            continue
        out = {"train": {}, "test": {}}
        for p in sel:
            out[p["split"]][p["id"]] = {"image": os.path.join(root, p["input"]), "target_image": os.path.join(root, p["target"]),
                                        "prompt": args.prompt, "mask": os.path.join(root, p["mask"])}
        with open(os.path.join(root, f"fixer_{typ}.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"[e34_manifest] {len(seg_rows)} segments ({accepted} accepted), {len(pairs)} pairs: "
          + "; ".join(f"{k} {sum(v.values())}" for k, v in sorted(counts.items())) + f" -> {root}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
