"""E43 step 3 (docs/EXPERIMENTS.md): generated vehicle assets placed into a trained run in place of its parked vehicles.

Per vehicle of --instances (scripts/vehicle_instances.py) with an asset (a 3DGS PLY: x y z, f_dc_*, opacity logit,
scale_* log, rot_* wxyz; from TRELLIS or SAM 3D Objects, step 2) the asset is fitted to the vehicle with a similarity
transform about world z (up): its up axis (--asset_up) to z, its longest ground axis along the vehicle's length axis,
its bottom on the vehicle's ground height (with --asset_up auto, each axis is tried as up and the best fit kept), and yaw (full circle in 6 degree steps, then a fine search), scale and ground
position chosen by the mean silhouette IoU against the vehicle's own instance masks (--inst_dir) in up to --fit_frames
of its detections (asset centres projected with the loader's poses and intrinsics at
1/--fit_down resolution, splatted as small discs), using its --fit_frames largest detections. The vehicle's instance masks only show what is
not occluded; the same occluders affect every candidate, so the ranking holds.

The run's Gaussians inside the fitted box (asset extent x --box_margin, up to the asset's height x --box_margin) are
removed: they are the vehicle's own blurred Gaussians (E35: vehicles never supervised, then repainted by the
generators, which also paint them a little off the real car: --box_margin around 1.4). With --complete_unseen (E47, on a
run whose vehicles were supervised, E41 / E45) the run's Gaussians on the vehicle's observed surface stay (in its
--fit_frames largest detections: inside its instance mask within --obs_tol scene units of the estimated depth of --pose_dir), the asset contributes only the
Gaussians behind that surface in every frame (its far side), and both lose what lies in front of it (free space). With --clean_votes n, the run's Gaussians within --clean_radius
that land inside the vehicle's instance mask, between --clean_front in front of the fitted asset and --clean_back behind its far side,
in at least n of the vehicle's own frames are removed too: the real frames see only the vehicle there, so anything else
is a remnant (E35 paints its invented cars a little off the real ones, outside any box). Two fitted vehicles whose footprints
overlap by more than 30 % (of either) are one vehicle tracked twice; the better fit is kept. The fit score is the mean
IoU minus --yaw_prior x |sin| of the heading off the observed length axis (parked cars are mostly parallel to it; the
silhouettes of far views are nearly symmetric). The asset's Gaussians are appended (colour band 0 from the asset, higher bands zero, opacity and scales
as stored, scales x the fitted scale), with a contact shadow under it: flat black discs on the ground over the
footprint, opacity --shadow_opacity in the middle fading to 0 --shadow_fade beyond the footprint edge (an asset has no
shadow of its own and looks afloat without one; 0 disables). The run's view gate is restricted to the kept Gaussians
(appended ones are never gated). All appended Gaussians are generated content: meta.json lists their index ranges.

Output --out_log_dir: checkpoint_final.pth, config.yaml, view_gate.pt (if the run has one), meta.json (per vehicle the
fitted pose, scale, IoU, removed and appended counts), fit/<id>.jpg (asset silhouette over the mask in the fitted
frames).

Example (main venv):
    PATH=$PWD/.venvs/main/bin:$PATH CUDA_HOME=$HOME/cuda-12.6 PYTHONPATH=. .venvs/main/bin/python scripts/vehicle_insert.py \\
        --log_dir results/E35/val056/model --instances results/E43/val056/instances/instances.json \\
        --inst_dir data/dashrecon/val056/inst-gsam2_img-glomap --pose_dir data/dashrecon/val056/pose-glomap_depth-mapanything \\
        --assets_dir results/E43/val056/assets_sam3d --asset_up y --out_log_dir results/E43/val056/model --exp E43 \\
        --fit_frames 12 --fit_down 4 --box_margin 1.15 --shadow_opacity 0.6 --shadow_fade 0.15
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon import io  # noqa: E402
from dashrecon.gen.floaters import ViewGate  # noqa: E402
from dashrecon.provenance import git_commit  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}
C0 = 0.28209479177387814


def read_gs_ply(path: str) -> dict:
    v = PlyData.read(path)["vertex"].data
    g = lambda *k: np.stack([np.asarray(v[x], np.float64) for x in k], 1)  # noqa: E731
    return {"xyz": g("x", "y", "z"), "dc": g("f_dc_0", "f_dc_1", "f_dc_2"), "opacity": g("opacity")[:, 0],
            "scale": g("scale_0", "scale_1", "scale_2"), "rot": g("rot_0", "rot_1", "rot_2", "rot_3")}


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return np.array([w, x, y, z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions, broadcast over rows of b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def canonical(asset: dict, up: str) -> tuple:
    """Rotation taking the asset to a frame with x = its longest ground axis, z = up; its extents and footprint centre."""
    u = AXES[up]
    ground = [a for a in range(3) if a != u]
    pts = asset["xyz"][1 / (1 + np.exp(-asset["opacity"])) > 0.5]
    g2 = pts[:, ground] - pts[:, ground].mean(0)
    evals, evecs = np.linalg.eigh(np.cov(g2.T))
    e = evecs[:, np.argmax(evals)]
    r = np.zeros((3, 3))
    r[0, ground] = e  # new x
    r[1, ground] = [-e[1], e[0]]  # new y
    r[2, u] = 1.0  # new z
    if np.linalg.det(r) < 0:
        r[1] *= -1
    c = pts @ r.T
    lo, hi = np.percentile(c, 1, 0), np.percentile(c, 99, 0)
    return r, lo, hi


def observation_state(pts: np.ndarray, dets, inst_dir: str, depth_dir: str, poses: np.ndarray, k_depth: np.ndarray, grid: dict,
                      fidx: dict, tol: float, dev: torch.device):
    """Per point: on the observed surface of the vehicle in at least one of the frames ``dets`` (inside its instance mask,
    within tol scene units of the estimated depth), and in front of that surface (free space) in at least one."""
    on = torch.zeros(len(pts), dtype=torch.bool, device=dev)
    front = torch.zeros(len(pts), dtype=torch.bool, device=dev)
    p = torch.tensor(pts, dtype=torch.float32, device=dev)
    th, tw = grid["depth_hw"]
    for f, i in dets:
        m = io.mask_to_depth_grid(np.asarray(Image.open(os.path.join(inst_dir, "inst", f"{f:06d}.png"))) == i, grid)
        if not m.any():
            continue
        mt = torch.from_numpy(m).to(dev)
        d = torch.tensor(io.read_depth(depth_dir, int(f)), dtype=torch.float32, device=dev)
        w2c = torch.tensor(np.linalg.inv(poses[fidx[f]]), dtype=torch.float32, device=dev)
        k = torch.tensor(k_depth[fidx[f]], dtype=torch.float32, device=dev)
        pc = p @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = (k[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + k[0, 2]).round().long()
        v = (k[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + k[1, 2]).round().long()
        ok = (z > 0.05) & (u >= 0) & (u < tw) & (v >= 0) & (v < th)
        idx = torch.nonzero(ok)[:, 0]
        sel = mt[v[idx], u[idx]] & (d[v[idx], u[idx]] > 0)
        idx = idx[sel]
        dz = d[v[idx], u[idx]]
        diff = z[idx] - dz
        on[idx[diff.abs() < tol]] = True
        front[idx[diff < -tol]] = True
    return on.cpu().numpy(), front.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--log_dir", "--instances", "--inst_dir", "--pose_dir", "--assets_dir", "--out_log_dir", "--exp"):
        parser.add_argument(a, required=True)
    parser.add_argument("--asset_up", choices=list(AXES) + ["auto"], required=True, help="auto: the axis with the best fit")
    parser.add_argument("--fit_frames", type=int, default=12)
    parser.add_argument("--fit_down", type=int, default=4)
    parser.add_argument("--fit_points", type=int, default=6000)
    parser.add_argument("--box_margin", type=float, default=1.15)
    parser.add_argument("--box_height_margin", type=float, default=1.3)
    parser.add_argument("--shadow_opacity", type=float, default=0.6)
    parser.add_argument("--shadow_fade", type=float, default=0.15)
    parser.add_argument("--shadow_step", type=float, default=0.03)
    parser.add_argument("--clean_votes", type=int, default=0, help="remove the run's Gaussians on the vehicle in this many of its frames (0 off)")
    parser.add_argument("--clean_radius", type=float, default=4.0)
    parser.add_argument("--clean_front", type=float, default=1.0, help="how far in front of the vehicle (scene units) counts as on it")
    parser.add_argument("--clean_back", type=float, default=0.05, help="how far behind the vehicle's far side counts as on it")
    parser.add_argument("--affine_inverse", action="store_true", help="put asset colours through the inverse of the run's mean Affine")
    parser.add_argument("--complete_unseen", action="store_true",
                        help="keep the run's vehicle where its frames saw it; add the asset only behind the observed surface (E47)")
    parser.add_argument("--obs_tol", type=float, default=0.1, help="depth tolerance of the observed surface (scene units)")
    parser.add_argument("--yaw_prior", type=float, default=0.05, help="IoU cost of a heading 90 degrees off the observed axis")
    parser.add_argument("--min_iou", type=float, default=0.3, help="vehicles whose best fit is below this keep their Gaussians")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_log_dir, "fit"), exist_ok=False)
    commit = git_commit()
    t0 = time.time()
    dev = torch.device("cuda")
    meta_in = json.load(open(args.instances))
    frames = io.read_frames(args.pose_dir)
    poses = io.read_poses(args.pose_dir)
    k_img = io.read_intrinsics(args.pose_dir)
    if k_img.ndim == 2:
        k_img = np.tile(k_img, (len(frames), 1, 1))
    fidx = {int(t): i for i, t in enumerate(frames)}

    state = torch.load(os.path.join(args.log_dir, "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    bg = state["models"]["Background"]
    # the run renders through its Affine colour model (identity + an MLP of a per-image embedding; novel views use the
    # mean embedding): asset colours are real pixel colours, so they get its inverse
    aff = None
    if args.affine_inverse and "Affine" in state["models"]:
        st_a = state["models"]["Affine"]
        e = st_a["embedding.weight"].mean(0, keepdim=True)
        h = torch.relu(e @ st_a["decoder.0.weight"].T + st_a["decoder.0.bias"])
        d = (h @ st_a["decoder.2.weight"].T + st_a["decoder.2.bias"]).view(3, 4).double().numpy()
        aff = (np.linalg.inv(np.eye(3) + d[:, :3]), d[:, 3])
    means = bg["_means"].double().numpy()
    remove = np.zeros(len(means), bool)
    add = {k: [] for k in bg}
    report, cands = [], []
    for inst in meta_in["instances"]:
        paths = glob.glob(os.path.join(args.assets_dir, f"asset_{inst['id']}.ply"))
        if not paths:
            continue
        asset = read_gs_ply(paths[0])
        # fit frames: the vehicle's largest detections (close views constrain yaw and scale; a view cut by the image
        # border is fine, the projected silhouette is cut the same way)
        ok = []
        for f, i in inst["detections"]:
            m = np.asarray(Image.open(os.path.join(args.inst_dir, "inst", f"{f:06d}.png"))) == i
            if m.any():
                ok.append((f, i, m))
        ok = sorted(ok, key=lambda x: -int(x[2].sum()))[:args.fit_frames]
        ok.sort(key=lambda x: x[0])
        if not ok:
            continue
        h, w = ok[0][2].shape
        hd, wd = h // args.fit_down, w // args.fit_down
        masks = torch.stack([F.avg_pool2d(torch.from_numpy(m).float()[None, None], args.fit_down)[0, 0] > 0.5 for _, _, m in ok]).to(dev)
        ks = torch.tensor(np.stack([k_img[fidx[f]] for f, _, _ in ok]), dtype=torch.float32, device=dev)
        ks[:, :2] /= args.fit_down
        w2c = torch.tensor(np.stack([np.linalg.inv(poses[fidx[f]]) for f, _, _ in ok]), dtype=torch.float32, device=dev)

        ax = np.array(inst["axis"][:2])
        perp = np.array([-ax[1], ax[0]])
        c0 = np.array(inst["centre"][:2])

        def fit_up(up):
            """Best (IoU, yaw, scale, shift along, shift across) with the asset's up axis ``up``."""
            r_can, lo, hi = canonical(asset, up)
            length, width, height = hi - lo
            # canonical asset points (opaque centres), footprint centred, bottom at 0
            op = 1 / (1 + np.exp(-asset["opacity"]))
            pts = (asset["xyz"][op > 0.5] @ r_can.T)
            pts = pts - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            sel = np.random.default_rng(0).choice(len(pts), min(args.fit_points, len(pts)), replace=False)
            p_can = torch.tensor(pts[sel], dtype=torch.float32, device=dev)
            s_len = inst["extent"][0] / length
            s_h = inst["extent"][2] / height

            def score(yaw, s, da, dp):
                """Mean silhouette IoU of candidates (tensors of equal length)."""
                n = yaw.shape[0]
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                a = torch.tensor(ax, dtype=torch.float32, device=dev)
                b = torch.tensor(perp, dtype=torch.float32, device=dev)
                # world xy of the canonical x / y axes: rotate (ax, perp) by yaw
                ex = cy[:, None] * a + sy[:, None] * b
                ey = -sy[:, None] * a + cy[:, None] * b
                ctr = torch.tensor(c0, dtype=torch.float32, device=dev) + da[:, None] * a + dp[:, None] * b
                xy = ctr[:, None] + s[:, None, None] * (p_can[None, :, 0:1] * ex[:, None] + p_can[None, :, 1:2] * ey[:, None])
                z = inst["ground_z"] + s[:, None] * p_can[None, :, 2]
                world = torch.cat([xy, z[..., None]], -1)  # (n, P, 3)
                ious = torch.zeros(n, device=dev)
                for j in range(len(ok)):
                    pc = world @ w2c[j, :3, :3].T + w2c[j, :3, 3]
                    zc = pc[..., 2].clamp(min=1e-3)
                    u = (ks[j, 0, 0] * pc[..., 0] / zc + ks[j, 0, 2]).round().long()
                    v = (ks[j, 1, 1] * pc[..., 1] / zc + ks[j, 1, 2]).round().long()
                    inside = (pc[..., 2] > 0.05) & (u >= 0) & (u < wd) & (v >= 0) & (v < hd)
                    occ = torch.zeros(n, hd * wd, device=dev)
                    idx = (v.clamp(0, hd - 1) * wd + u.clamp(0, wd - 1))
                    occ.scatter_(1, idx, inside.float())
                    occ = F.max_pool2d(occ.view(n, 1, hd, wd), 5, 1, 2).view(n, -1) > 0
                    m = masks[j].view(1, -1)
                    inter = (occ & m).sum(1).float()
                    union = (occ | m).sum(1).float().clamp(min=1)
                    ious += inter / union
                # prior: parallel to the observed length axis (either way) unless the silhouettes clearly say otherwise
                return ious / len(ok) - args.yaw_prior * torch.sin(yaw).abs()

            best = None
            s_grid = torch.tensor([float(s_len * f) for f in (0.85, 0.92, 1.0, 1.08, 1.16)] + [float(s_h)], dtype=torch.float32, device=dev)
            for base_yaw in (0.0,):  # full circle: the observed points give only a biased axis (one side seen)
                yaws = torch.tensor([math.radians(d) for d in range(0, 360, 6)], dtype=torch.float32, device=dev)
                da = torch.linspace(-0.3, 0.3, 7, device=dev)
                dp = torch.linspace(-0.2, 0.5, 8, device=dev) * float(np.sign((c0 - poses[fidx[ok[0][0]]][:2, 3]) @ perp) or 1.0)
                grid = torch.stack(torch.meshgrid(yaws, s_grid, da, dp, indexing="ij"), -1).view(-1, 4)
                for chunk in grid.split(256):
                    sc = score(chunk[:, 0], chunk[:, 1], chunk[:, 2], chunk[:, 3])
                    k = int(sc.argmax())
                    if best is None or float(sc[k]) > best[0]:
                        best = (float(sc[k]), *[float(x) for x in chunk[k]])
            # local refinement
            for step in (0.5, 0.25):
                yaw0, s0, da0, dp0 = best[1:]
                cand = torch.tensor([[yaw0 + math.radians(4 * step * i), s0 * (1 + 0.04 * step * j), da0 + 0.06 * step * a, dp0 + 0.06 * step * b]
                                     for i in (-1, 0, 1) for j in (-1, 0, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)], device=dev)
                sc = score(cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3])
                k = int(sc.argmax())
                if float(sc[k]) > best[0]:
                    best = (float(sc[k]), *[float(x) for x in cand[k]])
            return best, r_can, lo, hi, pts

        fits = {up: fit_up(up) for up in (AXES if args.asset_up == "auto" else [args.asset_up])}
        up_best = max(fits, key=lambda u: fits[u][0][0])
        best, r_can, lo, hi, pts = fits[up_best]
        length, width, height = hi - lo
        _, yaw, s, da, dp = best
        iou = float(best[0] + args.yaw_prior * abs(math.sin(yaw)))  # report the IoU itself
        rec = {"id": inst["id"], "iou": iou, "asset_up": up_best, "iou_per_up": {u: fits[u][0][0] for u in fits}, "yaw_deg": math.degrees(yaw), "scale": s, "shift_along": da, "shift_perp": dp,
               "fit_frames": [f for f, _, _ in ok], "asset_extent": [float(length), float(width), float(height)]}
        print(f"[vehicle_insert] vehicle {inst['id']}: IoU {iou:.3f}, scale {s:.4f}, yaw {math.degrees(yaw):.1f} deg "
              f"({time.time() - t0:.0f} s)", flush=True)
        if iou < args.min_iou:
            rec["inserted"] = False
            report.append(rec)
            continue

        # world transform of canonical coordinates
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex, ey = cy * ax + sy * perp, -sy * ax + cy * perp
        r_w = np.array([[ex[0], ey[0], 0.0], [ex[1], ey[1], 0.0], [0.0, 0.0, 1.0]])
        ctr = np.array([*(c0 + da * ax + dp * perp), inst["ground_z"]])
        off = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        rot = r_w @ r_can  # asset -> world rotation
        xyz = ((asset["xyz"] @ r_can.T - off) * s) @ r_w.T + ctr
        # remove the run's Gaussians inside the box
        loc = (means - ctr) @ r_w  # canonical-axis coordinates (scaled)
        half = np.array([length, width]) * s * args.box_margin / 2
        inside = (np.abs(loc[:, 0]) < half[0]) & (np.abs(loc[:, 1]) < half[1]) & (loc[:, 2] > -0.05) & (loc[:, 2] < height * s * args.box_height_margin)
        q = quat_from_matrix(rot)
        rq = asset["rot"] / np.linalg.norm(asset["rot"], axis=1, keepdims=True)
        n_a = len(xyz)
        keep_a = np.ones(n_a, bool)
        if args.complete_unseen:
            # the run's own vehicle (supervised, E41 / E45) stays where the frames saw it; the asset supplies only what lies
            # behind the observed surface in every frame; anything in front of it (free space) goes, scene or asset
            grid_d = io.read_meta(args.pose_dir)["depth_grid"]
            k_d = io.depth_intrinsics(k_img, grid_d)
            obs_dets = [(f, i) for f, i, _ in ok]  # the vehicle's largest (closest) detections: depth is reliable there
            on_a, front_a = observation_state(xyz, obs_dets, args.inst_dir, args.pose_dir, poses, k_d, grid_d, fidx, args.obs_tol, dev)
            keep_a = ~on_a & ~front_a
            in_idx = np.nonzero(inside)[0]
            on_s, _ = observation_state(means[in_idx], obs_dets, args.inst_dir, args.pose_dir, poses, k_d, grid_d, fidx, args.obs_tol, dev)
            inside = inside.copy()
            inside[in_idx[on_s]] = False  # observed scene Gaussians stay
            rec_unseen = {"asset_kept": int(keep_a.sum()), "asset_on_surface": int(on_a.sum()), "asset_free_space": int(front_a.sum()),
                          "scene_inside_kept_observed": int(on_s.sum())}
        dc_a = asset["dc"]
        if aff is not None:
            col = (dc_a * C0 + 0.5 - aff[1]) @ aff[0].T
            dc_a = (col - 0.5) / C0
        parts = {"_means": xyz, "_scales": asset["scale"] + math.log(s), "_quats": quat_mul(q, rq), "_features_dc": dc_a,
                 "_opacities": asset["opacity"][:, None]}
        parts = {k: v[keep_a] for k, v in parts.items()}
        # contact shadow: flat black discs over the footprint (and the fade ring)
        sh = []
        if args.shadow_opacity > 0:
            hl, hw = length * s / 2 + args.shadow_fade, width * s / 2 + args.shadow_fade
            ga = np.arange(-hl, hl + 1e-9, args.shadow_step)
            gb = np.arange(-hw, hw + 1e-9, args.shadow_step)
            A, B = np.meshgrid(ga, gb, indexing="ij")
            dist_out = np.maximum(np.abs(A) - length * s / 2, 0) + np.maximum(np.abs(B) - width * s / 2, 0)
            alpha = args.shadow_opacity * np.clip(1 - dist_out / args.shadow_fade, 0, 1)
            keep = alpha > 0.02
            pos = ctr + A[keep][:, None] * np.array([*ex, 0.0]) + B[keep][:, None] * np.array([*ey, 0.0]) + np.array([0, 0, 0.01])
            a_ = np.clip(alpha[keep], 1e-4, 1 - 1e-4)
            sh = {"_means": pos, "_scales": np.log(np.tile([args.shadow_step * 0.8, args.shadow_step * 0.8, args.shadow_step * 0.05], (len(pos), 1))),
                  "_quats": np.tile(quat_from_matrix(r_w), (len(pos), 1)), "_features_dc": np.full((len(pos), 3), (0.02 - 0.5) / C0),
                  "_opacities": np.log(a_ / (1 - a_))[:, None]}
        cands.append({"rec": rec, "blocks": [parts] + ([sh] if len(sh) else []), "inside": inside, "ctr": ctr, "r_w": r_w,
                      "half": np.array([length, width]) * s / 2, "pts_world": (pts[::max(1, len(pts) // 5000)] * s) @ r_w.T + ctr,
                      "dets": [(f, i) for f, i in inst["detections"]]})
        if args.complete_unseen:
            rec["unseen_completion"] = rec_unseen
        rec.update({"inserted": True, "removed": int(inside.sum()), "asset_gaussians": int(keep_a.sum()), "shadow_gaussians": int(len(sh["_means"])) if len(sh) else 0,
                    "centre": ctr.tolist(), "rotation": rot.tolist()})
        report.append(rec)

        # fit check image: asset silhouette (red) over the instance mask (white) in the fit frames
        with torch.no_grad():
            tiles = []
            for j in range(min(4, len(ok))):
                pc = torch.tensor((pts * s) @ r_w.T + ctr, dtype=torch.float32, device=dev) @ w2c[j, :3, :3].T + w2c[j, :3, 3]
                zc = pc[:, 2].clamp(min=1e-3)
                u = (ks[j, 0, 0] * pc[:, 0] / zc + ks[j, 0, 2]).round().long()
                v = (ks[j, 1, 1] * pc[:, 1] / zc + ks[j, 1, 2]).round().long()
                img = np.zeros((hd, wd, 3), np.uint8)
                img[masks[j].cpu().numpy()] = 255
                okp = ((pc[:, 2] > 0.05) & (u >= 0) & (u < wd) & (v >= 0) & (v < hd)).cpu().numpy()
                img[v.cpu().numpy()[okp], u.cpu().numpy()[okp]] = [255, 0, 0]
                tiles.append(img)
            Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(args.out_log_dir, "fit", f"{inst['id']}.jpg"))

    # one vehicle per place: a fitted vehicle whose centre lies in the footprint of a better-fitting one (or the other
    # way round) is a duplicate track of the same vehicle
    accepted = []
    for c in sorted(cands, key=lambda c: -c["rec"]["iou"]):
        def overlap(a, b):
            """Share of b's footprint inside a's footprint (grid samples)."""
            g = np.stack(np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 5), indexing="ij"), -1).reshape(-1, 2) * b["half"]
            pts = b["ctr"][:2] + g @ b["r_w"][:2, :2].T
            loc = (pts - a["ctr"][:2]) @ a["r_w"][:2, :2]
            return float(((np.abs(loc[:, 0]) < a["half"][0]) & (np.abs(loc[:, 1]) < a["half"][1])).mean())
        dup = [a["rec"]["id"] for a in accepted if max(overlap(a, c), overlap(c, a)) > 0.3]
        if dup:
            c["rec"].update({"inserted": False, "duplicate_of": dup[0]})
            continue
        accepted.append(c)
        remove |= c["inside"]
        if args.clean_votes > 0:  # the run's Gaussians the vehicle's own frames see on the vehicle (or just in front)
            near = np.nonzero(~remove & (np.linalg.norm(means - c["ctr"], axis=1) < args.clean_radius))[0]
            votes = np.zeros(len(near), np.int32)
            mt = torch.tensor(means[near], dtype=torch.float32, device=dev)
            pw = torch.tensor(c["pts_world"], dtype=torch.float32, device=dev)
            for f, i in c["dets"]:
                m = torch.from_numpy(np.asarray(Image.open(os.path.join(args.inst_dir, "inst", f"{f:06d}.png"))) == i).to(dev)
                k = torch.tensor(k_img[fidx[f]], dtype=torch.float32, device=dev)
                w2c_f = torch.tensor(np.linalg.inv(poses[fidx[f]]), dtype=torch.float32, device=dev)
                za = (pw @ w2c_f[:3, :3].T + w2c_f[:3, 3])[:, 2]
                pc = mt @ w2c_f[:3, :3].T + w2c_f[:3, 3]
                z = pc[:, 2]
                u = (k[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + k[0, 2]).round().long()
                v = (k[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + k[1, 2]).round().long()
                ok_ = (z > 0.05) & (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
                hit = torch.zeros_like(ok_)
                hit[ok_] = m[v[ok_], u[ok_]]
                hit &= (z > za.min() - args.clean_front) & (z < za.max() + args.clean_back)
                votes += hit.cpu().numpy().astype(np.int32)
            gone = near[votes >= args.clean_votes]
            remove[gone] = True
            c["rec"]["removed_by_views"] = int(len(gone))
        for blk in c["blocks"]:
            n_b = len(blk["_means"])
            for k in add:
                if k == "_features_rest":
                    add[k].append(torch.zeros((n_b,) + tuple(bg[k].shape[1:])))
                else:
                    add[k].append(torch.from_numpy(np.ascontiguousarray(blk[k])).float())
    keep = np.nonzero(~remove)[0]
    kt = torch.from_numpy(keep)
    n_keep = len(keep)
    ranges, start = [], n_keep
    for blk_m in add["_means"]:
        ranges.append([start, start + len(blk_m)])
        start += len(blk_m)
    for c in cands:
        if c["rec"].get("inserted") and c not in accepted:
            c["rec"]["inserted"] = False
    for k in bg:
        bg[k] = torch.cat([bg[k][kt]] + add[k], 0) if add[k] else bg[k][kt]
    torch.save(state, os.path.join(args.out_log_dir, "checkpoint_final.pth"))
    cfg = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    if "view_gate" in cfg:
        gate = ViewGate.load(cfg.view_gate, torch.device("cpu"))
        gk = kt[kt < gate.n]
        ViewGate(axis=gate.axis[gk], half_angle=gate.half_angle[gk], observed=gate.observed[gk], selected=gate.selected[gk],
                 margin=gate.margin, fade=gate.fade, double_sided=gate.double_sided).save(
            os.path.join(args.out_log_dir, "view_gate.pt"), {"source": cfg.view_gate, "restricted_to": "Gaussians kept by vehicle_insert.py"})
        cfg.view_gate = os.path.join(args.out_log_dir, "view_gate.pt")
    OmegaConf.save(cfg, os.path.join(args.out_log_dir, "config.yaml"))
    src_meta = json.load(open(os.path.join(args.log_dir, "meta.json")))
    with open(os.path.join(args.out_log_dir, "meta.json"), "w") as f:
        json.dump({"exp": args.exp, "kind": "parked vehicles replaced by fitted generated assets", "source_run": os.path.abspath(args.log_dir),
                   **{k: v for k, v in vars(args).items()}, "gaussians_before": int(len(means)), "removed": int(remove.sum()),
                   "generated_ranges": ranges, "vehicles": report, "generative": True,
                   "generative_note": "vehicle assets and contact shadows are generated; " + ("source run generative" if src_meta["generative"] else "source run not generative"),
                   "uses_oracle": False, "runtime_s": time.time() - t0, "dashrecon_commit": commit}, f, indent=2)
    print(f"[vehicle_insert] {sum(r.get('inserted', False) for r in report)} vehicles inserted, {int(remove.sum()):,} Gaussians removed "
          f"-> {args.out_log_dir} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
