"""D-M1 (docs/EXPERIMENTS.md): which Waymo perception segments drive the same road as another segment?

Dataset exploration, not part of any reconstruction: reads the GT vehicle poses of the raw tfrecords (waymo venv).
The poses of all segments of one location share a global frame (09-29: segments recorded months apart overlap over
hundreds of metres with matching headings), so revisits show up as nearby trajectories.
    1. every --sample_every-th frame of every tfrecord in <raw_dir>/<split>: vehicle position, location, time of day,
       weather, first timestamp -> --positions JSON (reused when it exists);
    2. trajectories resampled every 1 m; a pair of segments of the same location shares a metre of road when a point
       of one lies within --radius of the other; heading difference < 30 degrees counts as the same direction,
       > 150 degrees as the opposite one; pairs sharing at least --min_shared metres -> --out JSON.

Example (waymo venv, CPU):
    CUDA_VISIBLE_DEVICES="" .venvs/waymo/bin/python scripts/find_revisits.py --raw_dir data/data --splits validation training \
        --positions data/dashrecon/scene_selection/positions.json --out data/dashrecon/scene_selection/revisits.json
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np


def scan(raw_dir: str, splits: list, sample_every: int) -> list:
    """Sampled GT vehicle positions and context of every segment."""
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    rows = []
    for split in splits:
        for path in sorted(glob.glob(os.path.join(raw_dir, split, "*.tfrecord"))):
            positions, context = [], None
            for i, record in enumerate(tf.data.TFRecordDataset(path, compression_type="")):
                if i % sample_every:
                    continue
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(record.numpy()))
                positions.append(np.array(frame.pose.transform).reshape(4, 4)[:3, 3].tolist())
                if context is None:
                    s = frame.context.stats
                    context = {"location": s.location, "time_of_day": s.time_of_day, "weather": s.weather,
                               "t0_us": frame.timestamp_micros}
            rows.append({"split": split, "segment": os.path.basename(path), "positions": positions, **context})
            print(f"[find_revisits] {len(rows)} {split} {context['location']}", flush=True)
    return rows


def resample(points: np.ndarray) -> np.ndarray:
    """Polyline through the xy positions, one point per metre."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.linalg.norm(b - a)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--min_shared", type=int, default=30)
    args = parser.parse_args()
    if not os.path.exists(args.positions):
        rows = scan(args.raw_dir, args.splits, args.sample_every)
        with open(args.positions, "w") as f:
            json.dump({"sample_every": args.sample_every, "kind": "GT vehicle poses, dataset exploration only", "segments": rows}, f)
    segments = json.load(open(args.positions))["segments"]
    tracks = [resample(np.array(s["positions"])[:, :2]) for s in segments]
    headings = [np.arctan2(*np.gradient(t, axis=0)[:, ::-1].T) for t in tracks]
    pairs = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i]["location"] != segments[j]["location"] or np.linalg.norm(tracks[i].mean(0) - tracks[j].mean(0)) > 1500:
                continue
            d = np.linalg.norm(tracks[i][:, None] - tracks[j][None], axis=-1)
            near = d.min(1) < args.radius
            if near.sum() < args.min_shared:
                continue
            dh = np.abs(np.angle(np.exp(1j * (headings[i][near] - headings[j][d[near].argmin(1)]))))
            date = lambda us: datetime.datetime.fromtimestamp(us / 1e6, datetime.timezone.utc).strftime("%Y-%m-%d")
            pairs.append({"shared_m": int(near.sum()), "same_direction_m": int((dh < np.radians(30)).sum()),
                          "opposite_m": int((dh > np.radians(150)).sum()), "location": segments[i]["location"],
                          "a": {k: segments[i][k] for k in ("split", "segment", "time_of_day", "weather")} | {"date": date(segments[i]["t0_us"])},
                          "b": {k: segments[j][k] for k in ("split", "segment", "time_of_day", "weather")} | {"date": date(segments[j]["t0_us"])}})
    pairs.sort(key=lambda p: -p["shared_m"])
    with open(args.out, "w") as f:
        json.dump({"radius_m": args.radius, "min_shared_m": args.min_shared, "segments": len(segments), "pairs": pairs,
                   "kind": "GT vehicle poses, dataset exploration only"}, f, indent=1)
    print(f"[find_revisits] {len(segments)} segments, {len(pairs)} pairs sharing >= {args.min_shared} m -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
