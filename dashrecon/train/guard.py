"""GT-isolation checks for non-oracle training (AGENTS.md sections 2, 6 task 3 and 10)."""
from omegaconf import OmegaConf

ALLOWED_MODELS = {"Background", "Sky", "Affine", "CamPose"}
DASHRECON_ROOT = "data/dashrecon/"
# where FRONT images may come from: the original processed split, or the undistorted FRONT-only tree of a
# self-calibration (DECISIONS D13; calib-glomap-mt1: the joint one of MT1, whose combined scene links its members'
# undistorted FRONT images), which holds no other file
IMAGE_ROOTS = ("data/waymo/processed/validation", "data/dashrecon/_undistorted/calib-glomap",
               "data/dashrecon/_undistorted/calib-glomap-mt1")


def assert_non_oracle(cfg: OmegaConf) -> None:
    """Fail loudly if a merged drivestudio config could read GT pose, LiDAR, boxes, calibration or GT masks."""
    d, ps = cfg.data, cfg.data.pixel_source
    assert d.lidar_source.load_lidar is False, "LiDAR must not be loaded"
    assert d.frames_from_images is True, "frame count must come from images, not ego_pose/"
    assert ps.type == "dashrecon.train.pixel_source.DashreconPixelSource", ps.type
    assert list(ps.cameras) == [0], f"FRONT camera only, got {list(ps.cameras)}"
    assert ps.load_objects is False and ps.load_smpl is False, "GT boxes must not be loaded"
    assert ps.undistort is False, "undistortion would use Waymo calibration"
    assert d.data_root in IMAGE_ROOTS, f"images must come from {IMAGE_ROOTS}, got {d.data_root}"
    assert set(cfg.model.keys()) <= ALLOWED_MODELS, f"only static background models allowed: {set(cfg.model.keys())}"
    init = cfg.model.Background.init
    assert "from_lidar" not in init, "LiDAR initialisation is oracle"
    assert cfg.trainer.losses.depth.key == "est_depth_map", "depth supervision must use the estimated depth"
    paths = [ps.pose_dir, ps.mask_dir, init.from_dashrecon.path]
    if "mesh" in cfg.trainer.losses:
        paths.append(cfg.trainer.losses.mesh.path)
    for p in paths:
        assert str(p).startswith(DASHRECON_ROOT), f"non-oracle inputs must be dashrecon products: {p}"
