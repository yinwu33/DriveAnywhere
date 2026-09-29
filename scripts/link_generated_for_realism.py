"""Pair generated frames with the real camera that looks the same way, for scripts/eval_realism.py (D-C1, D-E1).

A turning round of Phase 9 (render_views.py --move yaw=Y) looks roughly where one real side camera looks (yaw 45:
FRONT_RIGHT = 2, -45: FRONT_LEFT = 1, 90: SIDE_RIGHT = 4, -90: SIDE_LEFT = 3). For every view of --views_dir that is
fully turned (cams.json ramp == 1) this links renders/<t:03d>_<cam>.png to its generated frame (filled/<k>.png; with
--source rgb the render the generator started from, rgb/<k>.png, for stage-wise comparisons, D-E2) and
refs/<t:03d>_<cam>.png to the real image of the same frame in --refs_dir (refs/ of an eval_cross_camera.py
--save_renders run: undistorted, render grid). Only the distribution is compared (EVALUATION: the references are GT
images); the generated frame and the real camera do not share a viewpoint exactly.

Example:
    .venvs/main/bin/python scripts/link_generated_for_realism.py --views_dir results/E14/val056/views/r0 --cam 2 \
        --refs_dir results/E5c/val056/realism/refs --out_dir results/_diagnostics/DC1/val056/E14_r0_realism
"""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--cam", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--refs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--source", choices=["filled", "rgb"], default="filled")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_dir, "renders"), exist_ok=False)
    os.makedirs(os.path.join(args.out_dir, "refs"))
    cams = json.load(open(os.path.join(args.views_dir, "cams.json")))["cams"]
    linked = []
    for k, c in enumerate(cams):
        if c["ramp"] != 1.0:
            continue
        name = f"{c['frame']:03d}_{args.cam}.png"
        ref = os.path.abspath(os.path.join(args.refs_dir, name))
        assert os.path.exists(ref), f"no reference for frame {c['frame']}: {ref} (render the references at the views' frame stride)"
        os.symlink(os.path.abspath(os.path.join(args.views_dir, args.source, f"{k:03d}.png")), os.path.join(args.out_dir, "renders", name))
        os.symlink(ref, os.path.join(args.out_dir, "refs", name))
        linked.append(c["frame"])
    with open(os.path.join(args.out_dir, "links.json"), "w") as f:
        json.dump({"views_dir": args.views_dir, "source": args.source, "cam": args.cam, "refs_dir": args.refs_dir, "frames": linked,
                   "kind": "generated frames vs the real camera looking the same way; distribution comparison only"}, f, indent=2)
    print(f"[link_generated_for_realism] {len(linked)} fully turned views of {args.views_dir} paired with camera {args.cam}", flush=True)


if __name__ == "__main__":
    main()
