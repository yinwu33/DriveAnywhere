"""E56 (docs/EXPERIMENTS.md; user 10-07: "基于目前最好的结果（E54s）放置行驶车辆，做仿真"): synthetic traffic of generated
3D vehicle assets driving on the road of a reconstructed scene, rendered along ego camera paths.

Scene: a trained run (--log_dir, e.g. E54s), rendered as is (its parked cars are its own Gaussians).
Lanes: the ego path (the estimated FRONT positions of --pose_dir, smoothed with a spline) offset sideways by each
--lanes entry "<offset>:<same|oncoming>" (scene units, + = left of the direction of travel). A lane sample counts only
where it lies in the road support of --planes (scripts/fit_planes.py, ground class road) and at least --parked_clear
from every parked vehicle of --parked (scripts/vehicle_instances.py run on the same pose backend): vehicles are drawn
only there, so they drive on the road only; z is the road height field.
Vehicles: generated 3D assets of --assets_dir (scripts/vehicle_assets_sam3d.py, SAM 3D Objects: GENERATED CONTENT)
whose fit in --asset_fits (scripts/vehicle_insert.py meta.json) reached --min_iou; scale = the fitted scale x
--world_scale (the fits were made in another pose backend's world: ratio of the two ego path lengths, no GT). The
asset's longest ground axis is its length; its front is the end with fewer red Gaussians (tail lights; --flip overrides
assets where that fails); Gaussians outside its 1-99 % box grown by --box_pad are dropped. Per oncoming
lane --oncoming_cars vehicles at speeds drawn from --oncoming_speed (scene units per frame), spaced --spacing apart
(same speed within a lane: no overtaking); in the ego lane one lead vehicle --lead_gap ahead of the ego camera,
following it. Each vehicle carries a contact shadow (flat dark discs over its footprint, as scripts/vehicle_insert.py).
Asset colours go through the inverse of the run's mean Affine colour model (vehicle_insert --affine_inverse).
Rendering: per frame and per --ego_moves entry (dashrecon.gen.views.ViewMove of the frame's trained camera, e.g.
"right=0", "right=1.29" = 3 m left, "yaw=-30"), the vehicles' Gaussians at their poses at that frame are appended to
the run's Background model and the frame is rendered at --hw with the run's own sky.
With --cameras (a camera path of scripts/render_demo.py --cameras_out) that path is rendered instead, as one video
camera_path.mp4 at its frame rate; traffic time runs uniformly over the clip (--clock uniform: the other vehicles keep
their speed when the camera slows down) and the lead vehicle stays --lead_gap ahead of the camera's path position.
Output --out_dir: traffic.json (lanes, vehicles, per-frame poses), <move>.mp4 (10 fps), <move>_sheet.jpg (6 frames),
camera_path.mp4 (with --cameras), meta.json (generated content: every vehicle). GT is not read.

Example (main venv):
    PYTHONPATH=. .venvs/main/bin/python scripts/simulate_traffic.py --log_dir results/E54s/val056 \\
        --pose_dir data/dashrecon/val056/pose-glomapf_depth-mapanything \\
        --planes data/dashrecon/val056/pose-glomapf_depth-mapanything__mask-gsam2mv_sky-segformer_img-glomapf/planes.npz \\
        --parked results/E56/val056/parked/instances.json --assets_dir results/E43/val056/assets_sam3d \\
        --asset_fits results/E43/val056/model/meta.json --world_scale 1.0116 --lanes 0:same 2.0:oncoming \\
        --oncoming_cars 6 --oncoming_speed 0.4 0.6 --spacing 6 --lead_gap 8 --ego_moves right=0 right=1.29 yaw=-30 \\
        --out_dir results/E56/val056/sim --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy.interpolate import UnivariateSpline
from torch.nn import Parameter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dashrecon import io  # noqa: E402
from dashrecon.gen.novel import build_trainer, refined_c2w, to_device  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w, render_at  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402
from vehicle_insert import C0, canonical, quat_from_matrix, quat_mul, read_gs_ply  # noqa: E402

PARAMS = ("_means", "_scales", "_quats", "_features_dc", "_features_rest", "_opacities")


class Road:
    """The road height field of scripts/fit_planes.py: height and support lookup, nearest supported cell."""

    def __init__(self, path: str) -> None:
        z = np.load(path)
        names = [str(x) for x in z["ground_classes"]]
        g = names.index("road")
        self.cell = float(z["cell"])
        self.origin = z[f"ground{g}_origin"]
        self.height = z[f"ground{g}_height"]
        self.support = z[f"ground{g}_support"]
        self.sup_ij = np.argwhere(self.support)
        self.sup_xy = (self.sup_ij + 0.5) * self.cell + self.origin

    def ij(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - self.origin) / self.cell).astype(np.int64)

    def on_road(self, xy: np.ndarray) -> np.ndarray:
        ij = self.ij(xy)
        h, w = self.support.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < h) & (ij[:, 1] < w)
        out = np.zeros(len(xy), bool)
        out[ok] = self.support[ij[ok, 0], ij[ok, 1]]
        return out

    def snap(self, xy: np.ndarray) -> tuple:
        """(xy moved into the support where outside, z of the height field there)."""
        xy = xy.copy()
        off = ~self.on_road(xy)
        for k in np.nonzero(off)[0]:
            xy[k] = self.sup_xy[np.argmin(((self.sup_xy - xy[k]) ** 2).sum(1))]
        ij = self.ij(xy).clip(0, np.array(self.height.shape) - 1)
        return xy, self.height[ij[:, 0], ij[:, 1]]


def front_sign(asset: dict, r_can: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """+1 if the asset's front is at canonical +x, -1 if at -x: the rear carries the red tail lights."""
    op = 1 / (1 + np.exp(-asset["opacity"]))
    p = asset["xyz"] @ r_can.T
    col = np.clip(asset["dc"] * C0 + 0.5, 0, 1)
    red = (op > 0.3) & (col[:, 0] > 0.35) & (col[:, 0] - np.maximum(col[:, 1], col[:, 2]) > 0.15)
    x = (p[:, 0] - (lo[0] + hi[0]) / 2) / (hi[0] - lo[0])
    return 1.0 if (red & (x < -0.3)).sum() >= (red & (x > 0.3)).sum() else -1.0


class Lane:
    def __init__(self, s: np.ndarray, p: np.ndarray, tangent: np.ndarray, ok: np.ndarray, oncoming: bool, offset: float) -> None:
        self.s, self.p, self.tangent, self.ok, self.oncoming, self.offset = s, p, tangent, ok, oncoming, offset

    def at(self, s: float):
        """(xy, heading of travel, drivable) at arc length s, or None outside the lane."""
        if s < self.s[0] or s > self.s[-1]:
            return None
        k = int(np.clip(np.searchsorted(self.s, s), 1, len(self.s) - 1))
        a = (s - self.s[k - 1]) / max(self.s[k] - self.s[k - 1], 1e-9)
        xy = (1 - a) * self.p[k - 1] + a * self.p[k]
        t = self.tangent[k] * (-1 if self.oncoming else 1)
        return xy, math.atan2(t[1], t[0]), bool(self.ok[k - 1] and self.ok[k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--log_dir", "--pose_dir", "--planes", "--parked", "--assets_dir", "--asset_fits", "--out_dir"):
        parser.add_argument(a, required=True)
    parser.add_argument("--world_scale", type=float, required=True)
    parser.add_argument("--lanes", nargs="+", required=True)
    parser.add_argument("--oncoming_cars", type=int, required=True)
    parser.add_argument("--oncoming_speed", type=float, nargs=2, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--lead_gap", type=float, required=True, help="0: no lead vehicle")
    parser.add_argument("--ego_moves", nargs="*", default=[])
    parser.add_argument("--cameras", help="camera path JSON of scripts/render_demo.py --cameras_out: render it (one video) instead of --ego_moves")
    parser.add_argument("--clock", choices=["uniform", "path"], default="uniform",
                        help="--cameras: traffic time runs uniformly over the clip (other vehicles keep their speed when the camera slows) or with the camera's path progress")
    parser.add_argument("--min_iou", type=float, default=0.7)
    parser.add_argument("--flip", type=int, nargs="*", default=[], help="assets whose front the tail-light test gets wrong (checked by eye)")
    parser.add_argument("--box_pad", type=float, default=0.08, help="drop asset Gaussians outside the 1-99 %% box grown by this share")
    parser.add_argument("--parked_clear", type=float, default=1.3)
    parser.add_argument("--shadow_opacity", type=float, default=0.6)
    parser.add_argument("--shadow_fade", type=float, default=0.15)
    parser.add_argument("--shadow_step", type=float, default=0.03)
    parser.add_argument("--hw", type=int, nargs=2, default=[704, 1280])
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=False)
    commit = git_commit()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")

    # lanes from the ego path
    poses = io.read_poses(args.pose_dir)
    xy = poses[:, :2, 3]
    tf = np.arange(len(xy), dtype=float)
    sx, sy = UnivariateSpline(tf, xy[:, 0], s=len(tf) * 1e-3), UnivariateSpline(tf, xy[:, 1], s=len(tf) * 1e-3)
    tt = np.linspace(0, len(xy) - 1, 4000)
    path = np.stack([sx(tt), sy(tt)], 1)
    tan = np.stack([sx.derivative()(tt), sy.derivative()(tt)], 1)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    left = np.stack([-tan[:, 1], tan[:, 0]], 1)
    road = Road(args.planes)
    parked = np.array([v["centre"][:2] for v in json.load(open(args.parked))["instances"]])
    lanes = []
    for spec in args.lanes:
        off, kind = spec.split(":")
        p = path + float(off) * left
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        ok = road.on_road(p) & (np.min(np.linalg.norm(p[:, None] - parked[None], axis=-1), axis=1) > args.parked_clear)
        lanes.append(Lane(s, p, tan, ok, kind == "oncoming", float(off)))
        print(f"[simulate_traffic] lane {spec}: length {s[-1]:.1f}, drivable {ok.mean():.2f}", flush=True)
    # ego arc length per frame (on the ego path itself)
    s_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s_ego = np.interp(tf, tt, s_path)

    # assets
    fits = json.load(open(args.asset_fits))
    pool = []
    for v in fits["vehicles"]:
        if not v.get("inserted") or v["iou"] < args.min_iou:
            continue
        asset = read_gs_ply(os.path.join(args.assets_dir, f"asset_{v['id']}.ply"))
        r_can, lo, hi = canonical(asset, v["asset_up"])
        sign = front_sign(asset, r_can, lo, hi) * (-1.0 if v["id"] in args.flip else 1.0)
        # stray Gaussians outside the vehicle (SAM 3D sometimes keeps a faint background layer)
        pc = asset["xyz"] @ r_can.T
        pad = args.box_pad * (hi - lo)
        inside = np.all((pc >= lo - pad) & (pc <= hi + pad), axis=1)
        asset = {k: x[inside] for k, x in asset.items()}
        flip = np.diag([sign, sign, 1.0])  # 180 degrees about z when the front is at -x
        r_can = flip @ r_can
        off = np.array([(lo[0] + hi[0]) / 2 * sign, (lo[1] + hi[1]) / 2 * sign, lo[2]])
        s = v["scale"] * args.world_scale
        length, width, height = (hi - lo) * s
        pool.append({"id": v["id"], "asset": asset, "r_can": r_can, "off": off, "s": s, "size": [float(length), float(width), float(height)],
                     "front_sign": sign, "iou": v["iou"], "cropped": int((~inside).sum())})
    print(f"[simulate_traffic] {len(pool)} assets (fit IoU >= {args.min_iou}): " +
          ", ".join(f"{a['id']} {a['size'][0]:.2f}x{a['size'][1]:.2f}" for a in pool), flush=True)

    # vehicles
    vehicles = []
    order = rng.permutation(len(pool))
    k_asset = 0
    for lane in lanes:
        if lane.oncoming:
            v = float(rng.uniform(*args.oncoming_speed))
            # the vehicles pass the ego camera at frames spread over the clip (at least --spacing apart along the lane)
            v_ego = (s_ego[-1] - s_ego[0]) / (len(tf) - 1)
            meet = np.linspace(0.08, 0.92, args.oncoming_cars) * (len(tf) - 1) + rng.uniform(-3, 3, args.oncoming_cars)
            s0 = s_ego[0] + meet * (v + v_ego)
            s0 = np.maximum.accumulate(np.maximum(s0, s0[0] + np.arange(args.oncoming_cars) * args.spacing))
            for k in range(args.oncoming_cars):
                vehicles.append({"lane": lane.offset, "oncoming": True, "speed": v, "s0": float(s0[k]), "asset": pool[order[k_asset % len(pool)]]["id"]})
                k_asset += 1
        elif args.lead_gap > 0 and abs(lane.offset) < 1e-6:
            vehicles.append({"lane": lane.offset, "oncoming": False, "lead_gap": args.lead_gap, "asset": pool[order[k_asset % len(pool)]]["id"]})
            k_asset += 1
    pool_by_id = {a["id"]: a for a in pool}
    lane_by_off = {lane.offset: lane for lane in lanes}

    def poses_at(f: float, f_ego: float = None) -> list:
        """Vehicles at traffic time f (frames, may be fractional); the lead vehicle follows the ego at path frame f_ego."""
        f_ego = f if f_ego is None else f_ego
        out = []
        for j, veh in enumerate(vehicles):
            lane = lane_by_off[veh["lane"]]
            s = float(np.interp(f_ego, tf, s_ego)) + veh["lead_gap"] if "lead_gap" in veh else veh["s0"] - veh["speed"] * f
            r = lane.at(s)
            if r is None or not r[2]:
                continue
            (x, y), yaw, _ = r
            z = float(road.snap(np.array([[x, y]]))[1][0])
            out.append((j, np.array([x, y, z]), yaw))
        return out

    # the run
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    from datasets.driving_dataset import DrivingDataset
    ds = DrivingDataset(data_cfg=cfg.data)
    tr = build_trainer(cfg, ds, dev)
    tr.resume_from_checkpoint(ckpt_path=os.path.join(args.log_dir, "checkpoint_final.pth"), load_only_model=True)
    tr.set_eval()
    bg = tr.models["Background"]
    base = {k: getattr(bg, k).data.clone() for k in PARAMS}
    aff = None
    if "Affine" in tr.models:
        st_a = {k: v.detach().cpu() for k, v in tr.models["Affine"].state_dict().items()}
        e = st_a["embedding.weight"].mean(0, keepdim=True)
        h = torch.relu(e @ st_a["decoder.0.weight"].T + st_a["decoder.0.bias"])
        d = (h @ st_a["decoder.2.weight"].T + st_a["decoder.2.bias"]).view(3, 4).double().numpy()
        aff = (np.linalg.inv(np.eye(3) + d[:, :3]), d[:, 3])

    # per-asset canonical Gaussians on the GPU (centred footprint, bottom at 0, front at +x, scaled) + shadow
    canon = {}
    for a in pool:
        g = a["asset"]
        dc = g["dc"]
        if aff is not None:
            dc = ((dc * C0 + 0.5 - aff[1]) @ aff[0].T - 0.5) / C0
        q_can = quat_from_matrix(a["r_can"])
        rq = g["rot"] / np.linalg.norm(g["rot"], axis=1, keepdims=True)
        blk = {"xyz": (g["xyz"] @ a["r_can"].T - a["off"]) * a["s"], "scale": g["scale"] + math.log(a["s"]), "quat": quat_mul(q_can, rq),
               "dc": dc, "opacity": g["opacity"][:, None]}
        L, W = a["size"][0], a["size"][1]
        hl, hw = L / 2 + args.shadow_fade, W / 2 + args.shadow_fade
        A, B = np.meshgrid(np.arange(-hl, hl + 1e-9, args.shadow_step), np.arange(-hw, hw + 1e-9, args.shadow_step), indexing="ij")
        dist_out = np.maximum(np.abs(A) - L / 2, 0) + np.maximum(np.abs(B) - W / 2, 0)
        alpha = args.shadow_opacity * np.clip(1 - dist_out / args.shadow_fade, 0, 1)
        keep = alpha > 0.02
        al = np.clip(alpha[keep], 1e-4, 1 - 1e-4)
        n_s = int(keep.sum())
        sh = {"xyz": np.stack([A[keep], B[keep], np.full(n_s, 0.01)], 1),
              "scale": np.log(np.tile([args.shadow_step * 0.8, args.shadow_step * 0.8, args.shadow_step * 0.05], (n_s, 1))),
              "quat": np.tile([1.0, 0, 0, 0], (n_s, 1)), "dc": np.full((n_s, 3), (0.02 - 0.5) / C0), "opacity": np.log(al / (1 - al))[:, None]}
        canon[a["id"]] = {k: torch.tensor(np.concatenate([blk[k], sh[k]]), dtype=torch.float32, device=dev) for k in blk}

    def quat_z(yaw: float) -> torch.Tensor:
        return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=torch.float32, device=dev)

    def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

    def set_traffic(f, f_ego=None) -> int:
        """Append the vehicles at traffic time f (lead vehicle at ego path frame f_ego) to the Background model (f None: none)."""
        extra = []
        for j, ctr, yaw in ([] if f is None else poses_at(f, f_ego)):
            c = canon[vehicles[j]["asset"]]
            cy, sy_ = math.cos(yaw), math.sin(yaw)
            rz = torch.tensor([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]], dtype=torch.float32, device=dev)
            extra.append({"_means": c["xyz"] @ rz.T + torch.tensor(ctr, dtype=torch.float32, device=dev), "_scales": c["scale"],
                          "_quats": qmul(quat_z(yaw), c["quat"]), "_features_dc": c["dc"], "_opacities": c["opacity"]})
        n_extra = 0
        for k in PARAMS:
            if extra:
                if k == "_features_rest":
                    n = sum(len(x["_means"]) for x in extra)
                    add = torch.zeros((n,) + tuple(base[k].shape[1:]), device=dev)
                else:
                    add = torch.cat([x[k] for x in extra])
                setattr(bg, k, Parameter(torch.cat([base[k], add]), requires_grad=False))
                n_extra = len(add)
            else:
                setattr(bg, k, Parameter(base[k], requires_grad=False))
        return n_extra

    hw = tuple(args.hw)
    traffic = {"lanes": [{"offset": lane.offset, "oncoming": lane.oncoming, "drivable": float(lane.ok.mean())} for lane in lanes],
               "vehicles": vehicles, "assets": {a["id"]: {"size": a["size"], "front_sign": a["front_sign"], "fit_iou": a["iou"], "gaussians_cropped": a["cropped"]} for a in pool},
               "frames": {}}
    frames = list(range(0, len(tf), args.frame_stride))
    for f in frames:
        traffic["frames"][f] = [{"vehicle": j, "centre": ctr.tolist(), "yaw": yaw} for j, ctr, yaw in poses_at(f)]
    stats = {}
    if args.cameras:
        cam_path = json.load(open(args.cameras))
        cams = cam_path["cameras"]
        p0, p1, t_end = cams[0]["path_frame"], cams[-1]["path_frame"], max(cams[-1]["time"], 1e-6)
        writer = imageio.get_writer(os.path.join(args.out_dir, "camera_path.mp4"), fps=cam_path["fps"], quality=9, macro_block_size=16)
        n_vis = []
        with torch.no_grad():
            for j, c in enumerate(cams):
                clock = p0 + (p1 - p0) * c["time"] / t_end if args.clock == "uniform" else c["path_frame"]
                set_traffic(clock, c["path_frame"])
                n_vis.append(len(poses_at(clock, c["path_frame"])))
                ii, ci = ds.full_image_set.get_image(front_image_index(ds, c["frame"]), 1)
                ii, ci = to_device(ii, dev), to_device(ci, dev)
                out = render_at(tr, ii, ci, torch.tensor(c["c2w"], dtype=torch.float32, device=dev),
                                torch.tensor(c["K"], dtype=torch.float32, device=dev), tuple(c["hw"]))
                writer.append_data((out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
                if j % 150 == 0:
                    print(f"[simulate_traffic] camera path {j}/{len(cams)}: {n_vis[-1]} vehicles ({time.time() - t0:.0f} s)", flush=True)
        writer.close()
        set_traffic(None)
        stats["camera_path"] = {"frames": len(cams), "clock": args.clock, "vehicles_on_road_per_frame_mean": float(np.mean(n_vis))}
    for spec in args.ego_moves:
        mv = ViewMove.parse(spec)
        name = spec.replace("=", "").replace(",", "_")
        writer = imageio.get_writer(os.path.join(args.out_dir, f"{name}.mp4"), fps=10 / args.frame_stride, macro_block_size=1, quality=8)
        sheet_frames = set(frames[int(i)] for i in np.linspace(0, len(frames) - 1, 6))
        tiles, n_vis = [], []
        with torch.no_grad():
            for f in frames:
                n_extra = set_traffic(f)
                ii, ci = ds.full_image_set.get_image(front_image_index(ds, f), 1)
                ii, ci = to_device(ii, dev), to_device(ci, dev)
                base_c2w = refined_c2w(tr, ii, ci)
                h0, w0 = int(ci["height"]), int(ci["width"])
                k = ci["intrinsics"].clone()
                k[:2] *= hw[1] / w0
                k[1, 2] -= (h0 * hw[1] / w0 - hw[0]) / 2
                out = render_at(tr, ii, ci, moved_c2w(base_c2w, mv), k, hw)
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                writer.append_data(rgb)
                n_vis.append(len(traffic["frames"][f]))
                if f in sheet_frames:
                    im = Image.fromarray(rgb).resize((hw[1] // 2, hw[0] // 2))
                    ImageDraw.Draw(im).text((6, 4), f"{spec} | frame {f} | generated vehicles", fill=(255, 255, 0))
                    tiles.append(np.asarray(im))
        writer.close()
        set_traffic(None)
        rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out_dir, f"{name}_sheet.jpg"), quality=88)
        stats[spec] = {"frames": len(frames), "vehicles_on_road_per_frame_mean": float(np.mean(n_vis))}
        print(f"[simulate_traffic] {spec}: {len(frames)} frames, {np.mean(n_vis):.1f} vehicles per frame ({time.time() - t0:.0f} s)", flush=True)
    for name in PARAMS:
        setattr(bg, name, Parameter(base[name], requires_grad=False))
    with open(os.path.join(args.out_dir, "traffic.json"), "w") as fh:
        json.dump(traffic, fh)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump({"kind": "synthetic traffic of generated vehicle assets in a reconstructed scene (simulation)", "params": vars(args),
                   "generative": True, "generative_note": "every vehicle is a SAM 3D Objects asset generated from one real frame of a parked vehicle of this scene",
                   "uses_oracle": False, "oracle_note": "estimated poses, planes, parked-vehicle tracks and assets only; no GT read",
                   "stats": stats, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, fh, indent=1)
    print(f"[simulate_traffic] -> {args.out_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
