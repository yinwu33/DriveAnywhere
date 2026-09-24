"""drivestudio pixel source fed by dashrecon products (AGENTS.md Phase 6; selected with data.pixel_source.type).

Reads only:
    <data_path>/images/<t:03d>_0.jpg                  FRONT images (same files as upstream)
    pixel_source.pose_dir  (Phase 3, dashrecon.io)    intrinsics, poses_c2w, depth, depth_conf, meta
    pixel_source.mask_dir  (Phase 4, dashrecon.io)    mask_dynamic / mask_sky
No GT pose, calibration, LiDAR, boxes or GT-derived masks are read (AGENTS.md section 10); nothing is
undistorted with calibration (DECISIONS D3).

Conventions: dashrecon intrinsics put pixel centres at integer coordinates (OpenCV); drivestudio and gsplat
put them at +0.5 (``get_rays`` adds 0.5; upstream Waymo intrinsics are scaled as corner-origin values), so
cx, cy get +0.5 before scaling to the load size.

Extra image_infos key: ``est_depth_map`` (z-depth on the load grid, 0 = no supervision), the estimated depth
restricted to static, non-sky pixels with confidence >= the training-frame percentile and depth < max_depth
(same rule as Phase 5); used through the configurable depth-loss key (upstream patch P4).
"""
import os

import numpy as np
import torch
from PIL import Image

from dashrecon import io
from dashrecon.scenes import train_frame_mask
from datasets.base.pixel_source import CameraData, sparse_lidar_map_downsampler
from datasets.waymo.waymo_sourceloader import WaymoPixelSource


class DashreconCameraData(CameraData):
    """One camera whose calibration, masks and depth come from dashrecon products."""

    # loaded after CameraData.__init__ (which already calls self.to); same pattern as upstream lidar_depth_maps
    est_depth_maps = None

    def __init__(self, pose_dir: str, mask_dir: str, conf_percentile: float, max_depth: float, test_stride: int, **kwargs) -> None:
        self.pose_dir, self.mask_dir = pose_dir, mask_dir
        self.conf_percentile, self.max_depth, self.test_stride = conf_percentile, max_depth, test_stride
        assert not kwargs["undistort"], "undistortion would use Waymo calibration (DECISIONS D3)"
        super().__init__(**kwargs)
        self.load_est_depth()

    def _frames(self) -> np.ndarray:
        frames = np.arange(self.start_timestep, self.end_timestep)
        np.testing.assert_array_equal(io.read_frames(self.pose_dir)[self.start_timestep:self.end_timestep], frames)
        np.testing.assert_array_equal(io.read_frames(self.mask_dir)[self.start_timestep:self.end_timestep], frames)
        return frames

    def create_all_filelist(self) -> None:
        self.img_filepaths = np.array([
            os.path.join(self.data_path, "images", f"{t:03d}_{self.cam_id}.jpg") for t in self._frames()
        ])

    def load_calibrations(self) -> None:
        frames = self._frames()
        k_all = io.read_intrinsics(self.pose_dir)
        assert k_all.ndim == 3, "expected per-frame (N,3,3) intrinsics from the pose backend"
        k = k_all[frames].astype(np.float64)
        sx = self.load_size[1] / self.original_size[1]
        sy = self.load_size[0] / self.original_size[0]
        ks = np.zeros_like(k)
        ks[:, 0, 0] = k[:, 0, 0] * sx
        ks[:, 1, 1] = k[:, 1, 1] * sy
        ks[:, 0, 2] = (k[:, 0, 2] + 0.5) * sx
        ks[:, 1, 2] = (k[:, 1, 2] + 0.5) * sy
        ks[:, 2, 2] = 1.0
        self.intrinsics = torch.from_numpy(ks).float()
        self.distortions = torch.zeros(len(frames), 5)
        self.cam_to_worlds = torch.from_numpy(io.read_poses(self.pose_dir)[frames]).float()

    def _load_mask_kind(self, kind: str) -> torch.Tensor:
        masks = []
        for t in self._frames():
            m = io.read_mask(self.mask_dir, kind, int(t))
            m = Image.fromarray(m.astype(np.uint8) * 255).resize((self.load_size[1], self.load_size[0]), Image.NEAREST)
            masks.append(np.asarray(m) == 255)
        return torch.from_numpy(np.stack(masks)).float()

    def load_dynamic_masks(self) -> None:
        self.dynamic_masks = self._load_mask_kind("dynamic")
        self.human_masks = None
        self.vehicle_masks = None

    def load_sky_masks(self) -> None:
        self.sky_masks = self._load_mask_kind("sky")

    def load_est_depth(self) -> None:
        """Estimated z-depth resampled (nearest) from the depth grid onto the load grid."""
        frames = self._frames()
        grid = io.read_meta(self.pose_dir)["depth_grid"]
        (h_img, w_img), (h_res, w_res) = grid["image_hw"], grid["resized_hw"]
        top, left = grid["crop_top_left"]
        th, tw = grid["depth_hw"]
        lh, lw = self.load_size
        # load-grid pixel centres (+0.5 convention) -> original image (integer-centre) -> depth grid
        u_img = (np.arange(lw) + 0.5) * (w_img / lw) - 0.5
        v_img = (np.arange(lh) + 0.5) * (h_img / lh) - 0.5
        u_d = np.rint((u_img + 0.5) * (w_res / w_img) - 0.5 - left).astype(np.int64)
        v_d = np.rint((v_img + 0.5) * (h_res / h_img) - 0.5 - top).astype(np.int64)
        u_ok, v_ok = (u_d >= 0) & (u_d < tw), (v_d >= 0) & (v_d < th)
        train = train_frame_mask(frames, self.start_timestep, self.test_stride)
        confs = [io.read_depth_conf(self.pose_dir, int(t)) for t in frames]
        depths = [io.read_depth(self.pose_dir, int(t)) for t in frames]
        thr = float(np.percentile(np.concatenate([c[d > 0] for c, d, tr in zip(confs, depths, train) if tr]), self.conf_percentile))
        out = np.zeros((len(frames), lh, lw), dtype=np.float32)
        for i, t in enumerate(frames):
            dyn = io.mask_to_depth_grid(io.read_mask(self.mask_dir, "dynamic", int(t)), grid)
            sky = io.mask_to_depth_grid(io.read_mask(self.mask_dir, "sky", int(t)), grid)
            d = depths[i]
            valid = (d > 0) & (d < self.max_depth) & (confs[i] >= thr) & ~dyn & ~sky
            dv = np.where(valid, d, 0.0)
            sampled = dv[np.clip(v_d, 0, th - 1)][:, np.clip(u_d, 0, tw - 1)]
            sampled[~v_ok, :] = 0.0
            sampled[:, ~u_ok] = 0.0
            out[i] = sampled
        self.est_depth_maps = torch.from_numpy(out).to(self.device)
        self.est_depth_conf_threshold = thr

    def to(self, device: torch.device) -> None:
        super().to(device)
        if self.est_depth_maps is not None:
            self.est_depth_maps = self.est_depth_maps.to(device)

    def get_image(self, frame_idx: int):
        image_infos, cam_infos = super().get_image(frame_idx)
        d = self.est_depth_maps[frame_idx]
        if self.downscale_factor != 1.0:
            d = sparse_lidar_map_downsampler(d, self.downscale_factor)
        image_infos["est_depth_map"] = d
        return image_infos, cam_infos


class DashreconPixelSource(WaymoPixelSource):
    """WaymoPixelSource whose cameras are DashreconCameraData (FRONT only)."""

    def load_cameras(self) -> None:
        assert list(self.camera_list) == [0], "dashrecon trains on the FRONT camera only (AGENTS.md section 2)"
        assert not self.data_cfg.load_objects, "GT boxes must not be loaded (AGENTS.md section 10)"
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()
        for idx, cam_id in enumerate(self.camera_list):
            camera = DashreconCameraData(
                pose_dir=self.data_cfg.pose_dir,
                mask_dir=self.data_cfg.mask_dir,
                conf_percentile=self.data_cfg.depth_conf_percentile,
                max_depth=self.data_cfg.depth_max,
                test_stride=self.data_cfg.test_image_stride,
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            self.camera_data[cam_id] = camera
