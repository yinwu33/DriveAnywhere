"""MapAnything pose + pointmap backend (runs in the ``.venvs/mapanything`` venv).

API confirmed against facebookresearch/map-anything @ 3d10cf7 (v1.1.4):
    - ``mapanything.utils.image.load_images`` (images only) and ``preprocess_inputs`` (images with optional
      ``intrinsics`` (3,3), ``camera_poses`` (4,4) OpenCV cam2world in any world frame, ``is_metric_scale``)
      both resize with ``crop_resize_if_necessary``: a full-extent resize to floor(size * max(target / size))
      followed by a centred crop to the target resolution picked by ``find_closest_aspect_ratio``
      (1920x1280 -> 518x345 -> crop to 518x336); ``preprocess_inputs`` moves the intrinsics with the image;
    - ``MapAnything.infer`` returns per view ``intrinsics`` (pixel centres at integer coordinates,
      see ``recover_pinhole_intrinsics_from_ray_directions``), ``camera_poses`` (OpenCV cam2world,
      world = first view, or the input world when poses are given), ``depth_z``, ``conf``, ``mask`` and
      ``metric_scaling_factor``. Given poses condition the prediction but are not reproduced exactly.
"""
import time

import numpy as np
import torch
from PIL import Image

from dashrecon.io import depth_intrinsics, image_intrinsics_from_depth
from dashrecon.pose.base import PoseBackend, PoseResult

MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
INFER_PARAMS = {"memory_efficient_inference": True, "use_amp": True, "amp_dtype": "bf16", "apply_mask": True,
                "mask_edges": True, "apply_confidence_mask": False}


class MapAnythingBackend(PoseBackend):
    """Metric reconstruction with MapAnything, from images alone or with known intrinsics and poses."""

    name = "mapanything"

    def __init__(self, model_id: str, max_views: int, resolution_set: int) -> None:
        """
        Args:
            model_id: Hugging Face model id, e.g. ``facebook/map-anything-apache``.
            max_views: largest sequence handled in one inference call; longer sequences need chunked
                inference (Phase 3 task 3), which is not implemented yet and raises.
            resolution_set: MapAnything resolution set (518 for MapAnything).
        """
        from mapanything.models import MapAnything

        self.model_id = model_id
        self.max_views = max_views
        self.resolution_set = resolution_set
        self.device = torch.device("cuda")
        self.model = MapAnything.from_pretrained(model_id).to(self.device).eval()

    def _depth_grid(self, image_hw: tuple[int, int], depth_hw: tuple[int, int]) -> dict:
        """Replicate ``crop_resize_if_necessary`` to describe the depth grid."""
        h, w = image_hw
        th, tw = depth_hw
        scale = max(tw / w, th / h) + 1e-8
        rh, rw = int(np.floor(h * scale)), int(np.floor(w * scale))
        return {
            "image_hw": [h, w],
            "resized_hw": [rh, rw],
            "crop_top_left": [(rh - th) // 2, (rw - tw) // 2],
            "depth_hw": [th, tw],
        }

    def _views(self, image_paths: list[str], intrinsics: np.ndarray | None, poses_c2w: np.ndarray | None) -> list:
        from mapanything.utils.image import load_images, preprocess_inputs

        if intrinsics is None:
            assert poses_c2w is None, "known poses need known intrinsics"
            return load_images(image_paths, resolution_set=self.resolution_set)
        assert intrinsics.shape == (3, 3), f"expected one shared (3,3) intrinsics, got {intrinsics.shape}"
        views = []
        for i, p in enumerate(image_paths):
            v = {"img": np.asarray(Image.open(p).convert("RGB")), "intrinsics": torch.from_numpy(intrinsics).float()}
            if poses_c2w is not None:
                v["camera_poses"] = torch.from_numpy(poses_c2w[i]).float()
                v["is_metric_scale"] = torch.tensor([False])
            views.append(v)
        return preprocess_inputs(views, resolution_set=self.resolution_set)

    def estimate(self, image_paths: list[str], intrinsics: np.ndarray | None, poses_c2w: np.ndarray | None) -> PoseResult:
        if len(image_paths) > self.max_views:
            raise NotImplementedError(
                f"{len(image_paths)} views > max_views={self.max_views}: chunked inference "
                "(AGENTS Phase 3 task 3) is not implemented yet"
            )
        assert poses_c2w is None or len(poses_c2w) == len(image_paths), (len(poses_c2w), len(image_paths))
        sizes = {Image.open(p).size for p in image_paths}
        assert len(sizes) == 1, f"images must share one size, got {sizes}"
        (w, h), = sizes

        views = self._views(image_paths, intrinsics, poses_c2w)
        th, tw = (int(x) for x in views[0]["true_shape"][0])
        depth_grid = self._depth_grid((h, w), (th, tw))

        torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        with torch.no_grad():
            preds = self.model.infer(views, **INFER_PARAMS)
        torch.cuda.synchronize(self.device)
        runtime = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3

        k_depth = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
        poses = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
        mask = np.stack([p["mask"][0, ..., 0].cpu().numpy() for p in preds]).astype(bool)
        depth = np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in preds])
        conf = np.stack([p["conf"][0].float().cpu().numpy() for p in preds])
        assert depth.shape[1:] == (th, tw), (depth.shape, th, tw)
        depth = np.where(mask & (depth > 0), depth, 0.0).astype(np.float32)
        scaling = np.stack([p["metric_scaling_factor"][0].float().cpu().numpy() for p in preds])
        if intrinsics is not None:
            # the given calibration must come back unchanged on the depth grid (same resize/crop as ours)
            err = np.abs(k_depth - depth_intrinsics(intrinsics, depth_grid)[None]).max()
            assert err < 0.5, f"MapAnything intrinsics differ from the given ones by {err:.2f} px on the depth grid"
            k_image = np.broadcast_to(intrinsics, (len(image_paths), 3, 3)).copy()
        else:
            k_image = image_intrinsics_from_depth(k_depth.astype(np.float64), depth_grid)

        return PoseResult(
            intrinsics=k_image,
            poses_c2w=poses.astype(np.float64),
            depth=depth,
            depth_conf=conf.astype(np.float32),
            depth_grid=depth_grid,
            meta={
                "backend": self.name,
                "model_id": self.model_id,
                "code_commit": MAPANYTHING_COMMIT,
                "torch": torch.__version__,
                "infer_params": {**INFER_PARAMS, "resolution_set": self.resolution_set},
                "inputs": ["images"] + (["intrinsics"] if intrinsics is not None else [])
                          + (["camera_poses (non-metric)"] if poses_c2w is not None else []),
                "num_views": len(image_paths),
                "runtime_s": runtime,
                "peak_vram_gb": peak_gb,
                "metric_scaling_factor": float(scaling.mean()),
                "valid_depth_fraction": float((depth > 0).mean()),
            },
        )
