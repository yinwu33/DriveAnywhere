"""Novel-view helpers shared by the Phase 8 scripts (scripts/train_ggds.py, scripts/postprocess_frames.py)."""
import math
from typing import Dict, Tuple

import torch

from dashrecon.gen.ggds import SDXLRefiner, disparity_image


def build_trainer(cfg, dataset, device: torch.device):
    """drivestudio trainer for a run config, built as in tools/train.py (weights still need to be loaded)."""
    from utils.misc import import_str

    return import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=device,
    )


def view_gate_path(cfg):
    """The view gate a run was trained with (config key ``view_gate``, written by scripts/train_fill.py --view_gate),
    or None for runs without one."""
    return cfg.view_gate if "view_gate" in cfg else None


def attach_view_gate(trainer, path: str) -> None:
    """Load a dashrecon.gen.floaters.ViewGate and make the trainer apply it to every novel view it renders."""
    from dashrecon.gen.floaters import ViewGate

    assert set(trainer.gaussian_classes.keys()) == {"Background"}, trainer.gaussian_classes
    gate = ViewGate.load(path, trainer.device)
    n = trainer.models["Background"]._means.shape[0]
    assert gate.n <= n, f"gate covers {gate.n} Gaussians, the model has {n}"
    trainer.view_gate = gate


def to_device(infos: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in infos.items()}


def shifted_c2w(c2w: torch.Tensor, offset: float, yaw_deg: float) -> torch.Tensor:
    """Camera moved ``offset`` scene units along its own x axis (OpenCV: right) and turned by ``yaw_deg`` about its
    own y axis (OpenCV: down), so positive values move and turn to the right."""
    a = math.radians(yaw_deg)
    ry = torch.tensor([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]],
                      dtype=c2w.dtype, device=c2w.device)
    out = c2w.clone()
    out[:3, 3] = c2w[:3, 3] + offset * c2w[:3, 0]
    out[:3, :3] = c2w[:3, :3] @ ry
    return out


def refined_c2w(trainer, image_infos: Dict, cam_infos: Dict) -> torch.Tensor:
    """The frame's camera after the learned pose refinement (CamPose), as used when rendering that frame."""
    with torch.no_grad():
        return trainer.models["CamPose"](cam_infos["camera_to_world"], image_infos["img_idx"].flatten()[0]).clone()


def mesh_losses(outputs: Dict, keep: torch.Tensor, mesh_cfg) -> Dict[str, torch.Tensor]:
    """The mesh depth / normal losses of DashreconTrainer.compute_losses, over the pixels where ``keep`` [H, W] is 1."""
    d_gs = outputs["depth"][..., 0]
    m_depth = keep * (d_gs > 1e-4).float()
    inv_err = (1.0 / d_gs.clamp(min=1e-4) - 1.0 / outputs["mesh_depth"].clamp(min=1e-4)).abs()
    m_n = keep * (outputs["gs_normal_alpha"] > mesh_cfg.min_alpha).float()
    cos = (outputs["gs_normal"] * outputs["mesh_normal"]).sum(-1)
    return {
        "mesh_depth_loss": mesh_cfg.depth_w * (inv_err * m_depth).sum() / m_depth.sum().clamp(min=1.0),
        "mesh_normal_loss": mesh_cfg.normal_w * ((1.0 - cos) * m_n).sum() / m_n.sum().clamp(min=1.0),
    }


@torch.no_grad()
def render_and_refine(trainer, refiner: SDXLRefiner, image_infos: Dict, cam_infos: Dict, t: float, max_pct: float,
                      novel_view: bool) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """Eval-mode render of one view, the mesh disparity control image and the refined image [H, W, 3]."""
    trainer.set_eval()
    outputs = trainer(image_infos, cam_infos, novel_view=novel_view)
    maps = trainer._mesh_maps(trainer._last_cam)
    control = disparity_image(maps["mesh_depth"], maps["mesh_valid"], max_pct)
    return outputs, control, refiner.refine(outputs["rgb"].clamp(0, 1), control, t)
