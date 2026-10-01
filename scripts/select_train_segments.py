"""Select the Waymo training segments to preprocess for the repair-model training data (docs/MIGRATION.md).

Reads the D-M1 census (scripts/find_revisits.py -> positions.json: GT vehicle positions every --sample_every frames,
time of day, weather, location; dataset selection only, never read by a reconstruction) and keeps the segments that
the FRONT-only pipeline can reconstruct: daytime (time_of_day == --time_of_day) and a driven path of at least
--min_path_m metres (GLOMAP needs the car to move). The path is the polyline through the sampled positions, so it
slightly underestimates the true path.

Output: one line per selected segment, "<scene_idx> <segment> <location> <path_m>", where scene_idx is the line number
in --file_list (drivestudio's --scene_ids for datasets/preprocess.py).

Example:
    .venvs/main/bin/python scripts/select_train_segments.py --positions data/dashrecon/scene_selection/positions.json \
        --file_list data/waymo_train_list.txt --time_of_day Day --min_path_m 60 --out data/waymo_train_day_moving.txt
"""
import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--time_of_day", required=True)
    parser.add_argument("--min_path_m", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    census = json.load(open(args.positions))
    by_name = {s["segment"].removesuffix(".tfrecord"): s for s in census["segments"] if s["split"] == "training"}
    names = [line.split()[0].removesuffix(".tfrecord") for line in open(args.file_list) if line.strip()]
    assert set(names) == set(by_name), "the census and the file list cover different segments"
    rows = []
    for idx, name in enumerate(names):
        s = by_name[name]
        p = np.array(s["positions"])[:, :2]
        path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if s["time_of_day"] == args.time_of_day and path >= args.min_path_m:
            rows.append(f"{idx} {name} {s['location']} {path:.0f}")
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[select_train_segments] {len(rows)} of {len(names)} segments -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
