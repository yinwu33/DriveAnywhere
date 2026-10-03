"""E34 (docs/E34_DATA.md): pick the Waymo training segments that become repair-model training pairs.

Candidates: --candidates (data/waymo_train_day_moving.txt: "<training idx> <segment> <location> <driven m>", daytime
segments with a driven path of at least 60 m). Excluded:
    - every segment of any pair in --revisits (data/dashrecon/scene_selection/revisits.json, D-M1: segments that share
      road with another segment, which covers the MT1 sites mt1a / mt1b and the training segments that pass the dev
      scenes val056 / val087);
    - the dev scenes (dashrecon.scenes.DEV_SCENES, validation split): checked to be absent from the candidates.
Selection: --num segments split across locations in proportion to the remaining candidates (largest remainder), drawn
per location with numpy's default_rng(--seed), then shuffled with the same generator so that any prefix of the list
(e.g. the first 20 accepted segments of the expensive pair type) is spread across locations too.

Writes --out (data/e34/segments.json): candidates count, excluded segments with the reason, the selected list in
processing order (training idx, scene id wt<idx:03d>, segment, location, driven m), parameters.

Example (any venv):
    .venvs/main/bin/python scripts/e34_select.py --candidates data/waymo_train_day_moving.txt \
        --revisits data/dashrecon/scene_selection/revisits.json --num 100 --seed 0 --out data/e34/segments.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402
from dashrecon.scenes import DEV_SCENES, training_scene_id  # noqa: E402


def proportional_counts(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across groups in proportion to their sizes (largest remainder, ties by group name)."""
    n = sum(sizes.values())
    assert 0 < total <= n, (total, n)
    exact = {k: total * v / n for k, v in sizes.items()}
    counts = {k: int(np.floor(x)) for k, x in exact.items()}
    left = total - sum(counts.values())
    order = sorted(sizes, key=lambda k: (-(exact[k] - counts[k]), k))
    for k in order[:left]:
        counts[k] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--revisits", required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for line in open(args.candidates):
        idx, seg, loc, path_m = line.split()
        rows.append({"idx": int(idx), "scene_id": training_scene_id(int(idx)), "segment": seg, "location": loc,
                     "driven_m": int(path_m)})
    revisits = json.load(open(args.revisits))
    in_pairs = {}
    for p in revisits["pairs"]:
        for side, other in (("a", "b"), ("b", "a")):
            name = p[side]["segment"].removesuffix(".tfrecord")
            in_pairs.setdefault(name, []).append(
                {"split": p[side]["split"], "other": p[other]["segment"].removesuffix(".tfrecord"),
                 "other_split": p[other]["split"], "shared_m": p["shared_m"]})
    dev = {s.segment for s in DEV_SCENES}
    assert not dev & {r["segment"] for r in rows}, "a dev scene is among the training candidates"
    excluded = [{**r, "reason": "shares road with another segment (revisits.json)", "pairs": in_pairs[r["segment"]]}
                for r in rows if r["segment"] in in_pairs]
    kept = [r for r in rows if r["segment"] not in in_pairs]
    by_loc = {}
    for r in kept:
        by_loc.setdefault(r["location"], []).append(r)
    counts = proportional_counts({k: len(v) for k, v in by_loc.items()}, args.num)
    rng = np.random.default_rng(args.seed)
    selected = []
    for loc in sorted(by_loc):
        group = by_loc[loc]
        pick = rng.choice(len(group), size=counts[loc], replace=False)
        selected += [group[i] for i in sorted(pick)]
    selected = [selected[i] for i in rng.permutation(len(selected))]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "params": vars(args),
            "num_candidates": len(rows),
            "revisit_segments_total": len(in_pairs),
            "revisit_segments": sorted(in_pairs),
            "excluded": excluded,
            "dev_scenes_never_used": sorted(dev),
            "num_after_exclusion": len(kept),
            "per_location": {loc: {"candidates": len(by_loc[loc]), "selected": counts[loc]} for loc in sorted(by_loc)},
            "selected": selected,
            "dashrecon_commit": git_commit(),
        }, f, indent=2)
    print(f"[e34_select] {len(rows)} candidates, {len(excluded)} excluded (revisits), {len(kept)} kept; selected "
          f"{len(selected)}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
