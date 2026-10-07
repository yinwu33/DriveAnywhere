"""Demo fly-through video of a trained run (user 10-07: "把 E54s 制作一个视频，期间 camera 要动一动，表示出来这是一个
3D 场景，做 demo show").

The camera drives along the run's own trained FRONT path (CamPose-refined poses; positions and rotations smoothed over
+-``--smooth`` frames and interpolated between frames: slerp), from frame --start to --end, and is moved in its own
frame (dashrecon.gen.views.ViewMove) by a fixed choreography of --duration seconds (times scaled from 32 s):
    0-4 s    straight ahead;
    4-10 s   weave: right = A sin, yaw = -0.45 * Y sin on the right half only (turning towards the road centre while
             moving out: parallax; turning right on the left half shows the unobserved far side of the cross street);
    10-17 s  crane: up to U and pitch down to -P (raised cosine), slower;
    17-23 s  weave the other way, small turns (0.3 Y);
    23-29 s  slow pan: 0.8 Y to the right, back, 0.44 Y to the left, back;
    29-32 s  straight ahead.
A = --lateral, Y = --yaw, U = --up, P = --pitch (scene units / degrees); defaults stay inside DECISIONS Z (lateral +-3 m,
yaw +-45 deg, up 0-2 m), where the reconstruction is valid. Novel-view rendering as everywhere else (render_at: the
run's sky, mean Affine). Output: --out (mp4, H.264, --fps); with --cameras_out the cameras as JSON (time, path frame,
c2w, K), which scripts/simulate_traffic.py --cameras renders with traffic. GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/render_demo.py --log_dir results/E54s/val056 --out results/E54s/val056/demo.mp4
"""
import argparse
import math
import os
import sys
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at  # noqa: E402


def choreography(tau: float, total: float, a: float, y: float, u: float, p: float) -> tuple:
    """(speed factor, ViewMove) at time tau of a clip of length total (segments defined on 32 s). Placed for val056
    E54s (10-07 check of candidate views): big turns only late in the drive, where both sides were seen; the cross
    street (frames ~70-100) only from above."""
    t = tau * 32.0 / total

    def bump(t0, t1):  # raised cosine 0 -> 1 -> 0 over [t0, t1]
        return 0.5 - 0.5 * math.cos(2 * math.pi * (t - t0) / (t1 - t0)) if t0 <= t <= t1 else 0.0

    def wave(t0, t1):  # one sine period over [t0, t1]
        return math.sin(2 * math.pi * (t - t0) / (t1 - t0)) if t0 <= t <= t1 else 0.0

    def pan(t0, t1):  # right, back, a smaller left, back
        if not t0 <= t <= t1:
            return 0.0
        x = (t - t0) / (t1 - t0)
        return math.sin(2 * math.pi * x) * (1.0 if x < 0.5 else 0.55)
    speed = 1.0 - 0.45 * bump(9, 18) - 0.6 * bump(22, 30)
    m = ViewMove(right=a * wave(4, 10) - 0.6 * a * wave(17, 23) + 0.3 * a * pan(23, 29),
                 up=u * bump(10, 17),
                 yaw=-0.45 * y * max(wave(4, 10), 0.0) + 0.3 * y * wave(17, 23) + 0.8 * y * pan(23, 29),
                 pitch=-p * bump(10, 17))
    return speed, m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=32.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=190)
    parser.add_argument("--smooth", type=int, default=4)
    parser.add_argument("--lateral", type=float, default=1.1, help="scene units (val056: 1 unit is about 2.3 m)")
    parser.add_argument("--yaw", type=float, default=22.0)
    parser.add_argument("--up", type=float, default=0.75)
    parser.add_argument("--pitch", type=float, default=10.0)
    parser.add_argument("--hw", type=int, nargs=2, default=[720, 1280])
    parser.add_argument("--cameras_out", help="also write the cameras (JSON; scripts/simulate_traffic.py --cameras)")
    parser.add_argument("--cameras_only", action="store_true", help="write --cameras_out without rendering")
    args = parser.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset
    ds = DrivingDataset(data_cfg=cfg.data)
    tr = build_trainer(cfg, ds, dev)
    tr.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    tr.set_eval()

    # trained camera path, smoothed
    n = ds.num_img_timesteps if hasattr(ds, "num_img_timesteps") else len(ds.full_image_set)
    c2ws = []
    for f in range(n):
        ii, ci = ds.full_image_set.get_image(front_image_index(ds, f), 1)
        c2ws.append(refined_c2w(tr, to_device(ii, dev), to_device(ci, dev)).cpu().numpy())
    c2ws = np.stack(c2ws)
    pos = c2ws[:, :3, 3]
    quat = Rotation.from_matrix(c2ws[:, :3, :3]).as_quat()
    quat *= np.sign((quat * quat[:1]).sum(1, keepdims=True))  # one hemisphere
    k = np.exp(-0.5 * (np.arange(-args.smooth, args.smooth + 1) / max(args.smooth / 2, 1e-6)) ** 2)
    k /= k.sum()
    pad = lambda x: np.concatenate([np.repeat(x[:1], args.smooth, 0), x, np.repeat(x[-1:], args.smooth, 0)])  # noqa: E731
    pos_s = np.stack([np.convolve(pad(pos[:, j]), k, mode="valid") for j in range(3)], 1)
    quat_s = np.stack([np.convolve(pad(quat[:, j]), k, mode="valid") for j in range(4)], 1)
    rots = Rotation.from_quat(quat_s / np.linalg.norm(quat_s, axis=1, keepdims=True))
    slerp = Slerp(np.arange(n), rots)

    # time -> path position (frames), with the choreography's speed profile
    n_out = int(round(args.duration * args.fps))
    taus = np.arange(n_out) / args.fps
    speed = np.array([choreography(tau, args.duration, args.lateral, args.yaw, args.up, args.pitch)[0] for tau in taus])
    prog = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    fpos = args.start + (args.end - args.start) * prog / prog[-1]

    h, w = args.hw
    cams = []
    writer = None if args.cameras_only else imageio.get_writer(args.out, fps=args.fps, quality=9, macro_block_size=16)
    with torch.no_grad():
        for j, (tau, f) in enumerate(zip(taus, fpos)):
            base = np.eye(4)
            base[:3, :3] = slerp([f]).as_matrix()[0]
            base[:3, 3] = [np.interp(f, np.arange(n), pos_s[:, c]) for c in range(3)]
            _, mv = choreography(tau, args.duration, args.lateral, args.yaw, args.up, args.pitch)
            fi = int(round(f))
            ii, ci = ds.full_image_set.get_image(front_image_index(ds, fi), 1)
            ii, ci = to_device(ii, dev), to_device(ci, dev)
            h0, w0 = int(ci["height"]), int(ci["width"])
            kk = ci["intrinsics"].clone()
            kk[:2] *= w / w0
            kk[1, 2] -= (h0 * w / w0 - h) / 2
            c2w = moved_c2w(torch.tensor(base, dtype=torch.float32, device=dev), mv)
            cams.append({"time": float(tau), "path_frame": float(f), "frame": fi, "c2w": c2w.cpu().tolist(), "K": kk.cpu().tolist(), "hw": [h, w]})
            if writer is None:
                continue
            out = render_at(tr, ii, ci, c2w, kk, (h, w))
            writer.append_data((out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            if j % 150 == 0:
                print(f"[render_demo] {j}/{n_out} (frame {f:.1f}, {mv}) {time.time() - t0:.0f} s", flush=True)
    if writer is not None:
        writer.close()
    if args.cameras_out:
        import json
        with open(args.cameras_out, "w") as fh:
            json.dump({"log_dir": args.log_dir, "fps": args.fps, "params": vars(args), "cameras": cams}, fh)
    print(f"[render_demo] {n_out} frames -> {args.out if writer is not None else args.cameras_out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
