"""E34 (docs/E34_DATA.md): stream one preprocessed Waymo training segment from OneDrive and unpack it, no tar staged.

    rclone cat --tpslimit <n> <remote>/<idx:03d>.tar | tar -x  (images/, extrinsics/, intrinsics/, ego_pose/ only)

Images go to <raw_root>/<idx:03d>/images/<t:03d>_<cam>.jpg (all 5 cameras); the GT calibration and ego poses are moved
to <gt_root>/<idx:03d>/{extrinsics,intrinsics,ego_pose}/, a separate tree that only the evaluation-side diagnostics
(scripts/e34_diag_calib.py) read: the pair pipeline is given <raw_root> and never sees them (AGENTS.md section 10).
GT box masks, objects and sky masks of the tar are not unpacked. Checks that every camera has the same frames 0..N-1.

Example:
    python scripts/e34_fetch.py --idx 646 --remote onedrive:/Projects/P05_DriveAnywhere/waymo_processed/training \
        --raw_root data/e34/raw --gt_root data/e34/gt --tpslimit 8
"""
import argparse
import json
import os
import shutil
import subprocess
import time

NUM_CAMS = 5
GT_DIRS = ("extrinsics", "intrinsics", "ego_pose")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--tpslimit", type=int, required=True)
    args = parser.parse_args()
    name = f"{args.idx:03d}"
    raw_dir, gt_dir = os.path.join(args.raw_root, name), os.path.join(args.gt_root, name)
    assert not os.path.exists(raw_dir), f"{raw_dir} exists"
    assert not os.path.exists(gt_dir), f"{gt_dir} exists"
    os.makedirs(args.raw_root, exist_ok=True)
    os.makedirs(gt_dir)
    t0 = time.time()
    cat = subprocess.Popen(["rclone", "cat", "--tpslimit", str(args.tpslimit), f"{args.remote}/{name}.tar"],
                           stdout=subprocess.PIPE)
    members = [f"{name}/images/*"] + [f"{name}/{d}/*" for d in GT_DIRS]
    tar = subprocess.run(["tar", "-x", "-C", args.raw_root, "--wildcards", *members], stdin=cat.stdout)
    cat.stdout.close()
    assert cat.wait() == 0, f"rclone cat {name}.tar failed with {cat.returncode}"
    assert tar.returncode == 0, f"tar failed with {tar.returncode}"
    for d in GT_DIRS:
        shutil.move(os.path.join(raw_dir, d), os.path.join(gt_dir, d))
    assert sorted(os.listdir(raw_dir)) == ["images"], os.listdir(raw_dir)
    files = os.listdir(os.path.join(raw_dir, "images"))
    per_cam = {c: sorted(int(f.split("_")[0]) for f in files if f.endswith(f"_{c}.jpg")) for c in range(NUM_CAMS)}
    assert len(files) == sum(len(v) for v in per_cam.values()), "unexpected file names"
    n = len(per_cam[0])
    for c in range(NUM_CAMS):
        assert per_cam[c] == list(range(n)), f"camera {c}: frames {per_cam[c][:3]}..{per_cam[c][-3:]} ({len(per_cam[c])}), FRONT has {n}"
    size = sum(os.path.getsize(os.path.join(raw_dir, "images", f)) for f in files)
    info = {"idx": args.idx, "frames": n, "images": len(files), "bytes": size, "seconds": time.time() - t0,
            "remote": f"{args.remote}/{name}.tar"}
    with open(os.path.join(raw_dir, "fetch.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[e34_fetch] {name}: {n} frames x {NUM_CAMS} cameras, {size / 1e6:.0f} MB in {info['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
