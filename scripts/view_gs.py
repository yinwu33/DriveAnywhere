"""Interactive viewer for trained runs: drivestudio's viser + nerfview viewer, loading checkpoints after training.

Serves a web page on --port of this machine. From another machine forward the port first, e.g.
    ssh -L 8080:localhost:8080 <user>@<this host>
then open http://localhost:8080 in a browser.

Differences from drivestudio's built-in viewer (tools/train.py --enable_viewer, Background only, training only):
    - loads finished runs (config.yaml + checkpoint_final.pth of each --log_dirs entry);
    - renders all Background Gaussians with their view-dependent (SH) colour over the Sky model (EnvLight);
      no exposure model (Affine), no dynamic objects (these runs have none);
    - GUI: "run" switches between the given runs (the camera goes back to the chosen frame and move, since runs of
      different camera pipelines have different world frames); "frame" puts the camera at that FRONT camera (the pose
      the run was trained with, with its estimated vertical field of view), moved by "right" / "up" (scene units, see
      DECISIONS L for how they relate to metres) and turned by "yaw" / "pitch" (degrees) as the Phase 9 views
      (dashrecon.gen.views.ViewMove, e.g. E8 round 0 = right 1.5, yaw 15); "go to frame" re-applies it.
Mouse: drag to orbit, right-drag to pan, scroll to move (viser controls); world z is up.

Example (main venv):
    PATH=$PWD/.venvs/main/bin:/usr/local/cuda-12.1/bin:$PATH CUDA_HOME=/usr/local/cuda-12.1 \
        .venvs/main/bin/python scripts/view_gs.py --log_dirs results/E5/val056 results/E6/val056 --port 8080
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.gen.novel import build_trainer  # noqa: E402
from dashrecon.gen.views import ViewMove, front_image_index, moved_c2w  # noqa: E402


def rotmat_to_wxyz(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = np.zeros(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
        return q / np.linalg.norm(q)
    q = np.array(q)
    return q / np.linalg.norm(q)


class Run:
    """One trained run: trainer with weights, and its FRONT cameras."""

    def __init__(self, log_dir: str, device: torch.device) -> None:
        from datasets.driving_dataset import DrivingDataset

        cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
        dataset = DrivingDataset(data_cfg=cfg.data)
        self.trainer = build_trainer(cfg, dataset, device)
        self.trainer.resume_from_checkpoint(ckpt_path=os.path.join(log_dir, "checkpoint_final.pth"), load_only_model=True)
        self.trainer.set_eval()
        full = dataset.full_image_set
        self.c2w, self.vfov = [], []
        for i in range(dataset.num_img_timesteps):
            _, ci = full.get_image(front_image_index(dataset, i), 1)
            self.c2w.append(ci["camera_to_world"].cpu().numpy().astype(np.float64))
            self.vfov.append(float(2 * np.arctan(0.5 * float(ci["height"]) / float(ci["intrinsics"][1, 1]))))
        self.frames = np.arange(dataset.start_timestep, dataset.end_timestep)
        self.label = os.path.relpath(log_dir)
        del dataset


@torch.no_grad()
def render(run: Run, camera_state, img_wh) -> np.ndarray:
    """Background Gaussians (view-dependent colour) over the Sky model, uint8 (H, W, 3)."""
    from datasets.base.pixel_source import get_rays
    from gsplat.rendering import rasterization
    from models.gaussians.basics import dataclass_camera

    trainer, device = run.trainer, run.trainer.device
    w, h = img_wh
    c2w = torch.from_numpy(camera_state.c2w).float().to(device)
    k = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
    cam = dataclass_camera(camtoworlds=c2w, camtoworlds_gt=c2w, Ks=k, H=h, W=w)
    gs = trainer.models["Background"].get_gaussians(cam)
    colors, alphas, _ = rasterization(
        means=gs["_means"], quats=gs["_quats"], scales=gs["_scales"], opacities=gs["_opacities"].squeeze(),
        colors=gs["_rgbs"], viewmats=torch.linalg.inv(c2w)[None], Ks=k[None], width=w, height=h,
        packed=trainer.render_cfg.packed, rasterize_mode="antialiased" if trainer.render_cfg.antialiased else "classic",
        near_plane=trainer.render_cfg.near_plane, far_plane=trainer.render_cfg.far_plane,
    )
    x, y = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="xy")
    _, viewdirs, _ = get_rays(x.flatten(), y.flatten(), c2w, k)
    sky = trainer.models["Sky"]({"viewdirs": viewdirs.reshape(h, w, 3)})
    rgb = colors[0] + sky.reshape(h, w, 3) * (1.0 - alphas[0])
    return (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dirs", nargs="+", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import nerfview
    import viser

    device = torch.device("cuda")
    runs = [Run(d, device) for d in args.log_dirs]
    for r in runs[1:]:
        assert len(r.frames) == len(runs[0].frames), "runs must cover the same frames"
    state = {"run": runs[0]}
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = nerfview.Viewer(server=server, render_fn=lambda cs, wh: render(state["run"], cs, wh), mode="rendering")

    run_choice = server.gui.add_dropdown("run", options=[r.label for r in runs], initial_value=runs[0].label)
    frame = server.gui.add_slider("frame", min=0, max=len(runs[0].frames) - 1, step=1, initial_value=0)
    move = {"right": server.gui.add_slider("right", min=-4.0, max=4.0, step=0.25, initial_value=0.0),
            "up": server.gui.add_slider("up", min=-1.0, max=4.0, step=0.25, initial_value=0.0),
            "yaw": server.gui.add_slider("yaw", min=-180.0, max=180.0, step=5.0, initial_value=0.0),
            "pitch": server.gui.add_slider("pitch", min=-45.0, max=45.0, step=5.0, initial_value=0.0)}
    go = server.gui.add_button("go to frame")

    def place(client) -> None:
        run = state["run"]
        c2w = moved_c2w(torch.from_numpy(run.c2w[int(frame.value)]), ViewMove(**{k: float(v.value) for k, v in move.items()})).numpy()
        with client.atomic():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.wxyz = rotmat_to_wxyz(c2w[:3, :3])
            client.camera.position = c2w[:3, 3]
            client.camera.fov = run.vfov[int(frame.value)]

    def place_all(_=None) -> None:
        for client in server.get_clients().values():
            place(client)

    @run_choice.on_update
    def _(_) -> None:
        # runs from different camera pipelines live in different world frames: keep the frame, not the world pose
        state["run"] = next(r for r in runs if r.label == run_choice.value)
        place_all()
        viewer.rerender(None)

    frame.on_update(place_all)
    for v in move.values():
        v.on_update(place_all)
    go.on_click(place_all)
    server.on_client_connect(place)
    print(f"[view_gs] serving {len(runs)} run(s) on port {args.port}: " + ", ".join(r.label for r in runs), flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
