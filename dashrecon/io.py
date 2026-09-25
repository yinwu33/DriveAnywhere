"""Readers/writers for the per-scene intermediate products (AGENTS.md section 5).

This module is the only entry point for reading and writing ``data/dashrecon/<scene_id>/<backend_tag>/``
(including the self-calibration directories ``calib-<method>/``: camera.json, frames.txt, poses_c2w.npy,
sparse_obs.npz, meta.json).
It must stay importable from every venv, so it only depends on numpy, PIL and the standard library.

Conventions:
    - lengths in metres; camera frame is OpenCV (x right, y down, z forward);
    - poses are camera-to-world 4x4 float64;
    - intrinsics are for the ORIGINAL image pixel grid, with pixel centres at integer coordinates;
    - depth maps may live on a different (resized + cropped) pixel grid; ``meta.json["depth_grid"]``
      describes that grid and :func:`depth_intrinsics` converts image intrinsics to it.
"""
import json
import os

import numpy as np
from PIL import Image

MASK_KINDS = ("dynamic", "sky", "road")


def scene_dir(root: str, scene_id: str, backend_tag: str) -> str:
    """Directory holding one scene's products for one backend combination."""
    return os.path.join(root, scene_id, backend_tag)


def _frame_file(out_dir: str, sub: str, idx: int, ext: str) -> str:
    return os.path.join(out_dir, sub, f"{idx:06d}.{ext}")


def write_frames(out_dir: str, frames: np.ndarray) -> None:
    """Write the frame indices (drivestudio timestep indices), one per line."""
    assert frames.ndim == 1 and np.issubdtype(frames.dtype, np.integer), frames.dtype
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "frames.txt"), frames, fmt="%d")


def read_frames(out_dir: str) -> np.ndarray:
    """Read the frame indices written by :func:`write_frames`."""
    return np.loadtxt(os.path.join(out_dir, "frames.txt"), dtype=np.int64, ndmin=1)


def write_intrinsics(out_dir: str, intrinsics: np.ndarray) -> None:
    """Write (3,3) shared or (N,3,3) per-frame intrinsics as float64."""
    assert intrinsics.shape[-2:] == (3, 3) and intrinsics.ndim in (2, 3), intrinsics.shape
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics.astype(np.float64))


def read_intrinsics(out_dir: str) -> np.ndarray:
    """Read intrinsics, (3,3) or (N,3,3) float64."""
    return np.load(os.path.join(out_dir, "intrinsics.npy"))


def write_poses(out_dir: str, poses_c2w: np.ndarray) -> None:
    """Write (N,4,4) camera-to-world poses as float64."""
    assert poses_c2w.ndim == 3 and poses_c2w.shape[1:] == (4, 4), poses_c2w.shape
    np.save(os.path.join(out_dir, "poses_c2w.npy"), poses_c2w.astype(np.float64))


def read_poses(out_dir: str) -> np.ndarray:
    """Read (N,4,4) camera-to-world poses."""
    return np.load(os.path.join(out_dir, "poses_c2w.npy"))


def write_depth(out_dir: str, idx: int, depth: np.ndarray) -> None:
    """Write one (H,W) z-depth map in metres; invalid pixels must be 0."""
    assert depth.ndim == 2, depth.shape
    path = _frame_file(out_dir, "depth", idx, "npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, depth.astype(np.float32))


def read_depth(out_dir: str, idx: int) -> np.ndarray:
    """Read one (H,W) float32 z-depth map."""
    return np.load(_frame_file(out_dir, "depth", idx, "npy"))


def write_depth_conf(out_dir: str, idx: int, conf: np.ndarray) -> None:
    """Write one (H,W) depth confidence map (same grid as the depth map)."""
    assert conf.ndim == 2, conf.shape
    path = _frame_file(out_dir, "depth_conf", idx, "npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, conf.astype(np.float32))


def read_depth_conf(out_dir: str, idx: int) -> np.ndarray:
    """Read one (H,W) float32 depth confidence map."""
    return np.load(_frame_file(out_dir, "depth_conf", idx, "npy"))


def write_mask(out_dir: str, kind: str, idx: int, mask: np.ndarray) -> None:
    """Write a boolean mask as uint8 PNG (255 = positive) to ``mask_<kind>/``."""
    assert kind in MASK_KINDS, kind
    assert mask.ndim == 2 and mask.dtype == bool, (mask.shape, mask.dtype)
    path = _frame_file(out_dir, f"mask_{kind}", idx, "png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def read_mask(out_dir: str, kind: str, idx: int) -> np.ndarray:
    """Read a mask written by :func:`write_mask` as a boolean array."""
    assert kind in MASK_KINDS, kind
    arr = np.asarray(Image.open(_frame_file(out_dir, f"mask_{kind}", idx, "png")))
    assert set(np.unique(arr)) <= {0, 255}, "mask must be binary 0/255"
    return arr == 255


def write_meta(out_dir: str, meta: dict) -> None:
    """Write ``meta.json`` (backend name, version, parameters, runtime, peak VRAM, ...)."""
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def read_meta(out_dir: str) -> dict:
    """Read ``meta.json``."""
    with open(os.path.join(out_dir, "meta.json")) as f:
        return json.load(f)


def write_camera(calib_dir: str, camera: dict) -> None:
    """Write ``camera.json`` of a self-calibration run (DECISIONS D13).

    Required keys: ``image_hw`` [h, w]; ``K`` (3x3 list) of the undistorted pinhole images, pixel centres at
    integer coordinates; ``dist`` [k1, k2] OpenCV radial coefficients of the original images (K is also their
    pinhole part); ``model`` (the COLMAP camera model the parameters came from).
    """
    assert set(camera) >= {"image_hw", "K", "dist", "model"}, sorted(camera)
    assert np.asarray(camera["K"]).shape == (3, 3) and len(camera["dist"]) == 2, (camera["K"], camera["dist"])
    os.makedirs(calib_dir, exist_ok=True)
    with open(os.path.join(calib_dir, "camera.json"), "w") as f:
        json.dump(camera, f, indent=2)


def read_camera(calib_dir: str) -> dict:
    """Read ``camera.json`` written by :func:`write_camera`; ``K`` and ``dist`` come back as float64 arrays."""
    with open(os.path.join(calib_dir, "camera.json")) as f:
        camera = json.load(f)
    camera["K"] = np.asarray(camera["K"], dtype=np.float64)
    camera["dist"] = np.asarray(camera["dist"], dtype=np.float64)
    return camera


def write_sparse_obs(calib_dir: str, frame: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> None:
    """Write the SfM point observations used for scale alignment: frame index, undistorted pixel (u, v) with
    integer pixel centres, and camera-frame z in SfM units (``sparse_obs.npz``)."""
    assert frame.shape == u.shape == v.shape == z.shape and frame.ndim == 1, (frame.shape, u.shape, v.shape, z.shape)
    np.savez(os.path.join(calib_dir, "sparse_obs.npz"), frame=frame.astype(np.int64), u=u.astype(np.float32),
             v=v.astype(np.float32), z=z.astype(np.float32))


def read_sparse_obs(calib_dir: str) -> dict[str, np.ndarray]:
    """Read ``sparse_obs.npz`` written by :func:`write_sparse_obs`."""
    with np.load(os.path.join(calib_dir, "sparse_obs.npz")) as data:
        return {k: data[k] for k in ("frame", "u", "v", "z")}


def image_to_depth_grid_coords(u: np.ndarray, v: np.ndarray, depth_grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Original-image pixel coordinates (integer centres) to depth-grid coordinates, as in :func:`depth_intrinsics`."""
    (h_img, w_img), (h_res, w_res) = depth_grid["image_hw"], depth_grid["resized_hw"]
    top, left = depth_grid["crop_top_left"]
    return (u + 0.5) * (w_res / w_img) - 0.5 - left, (v + 0.5) * (h_res / h_img) - 0.5 - top


def depth_intrinsics(intrinsics: np.ndarray, depth_grid: dict) -> np.ndarray:
    """Convert original-image intrinsics to the depth-map pixel grid.

    The depth grid is the original image resized to ``resized_hw`` (full-extent resize) and then
    cropped at ``crop_top_left``; pixel centres are at integer coordinates on both grids, so
    ``u_depth = (u_img + 0.5) * sx - 0.5 - left``.

    Args:
        intrinsics: (3,3) or (N,3,3) intrinsics on the original image grid.
        depth_grid: ``meta["depth_grid"]`` with keys image_hw, resized_hw, crop_top_left, depth_hw.

    Returns:
        Intrinsics of the same shape on the depth grid.
    """
    (h_img, w_img), (h_res, w_res) = depth_grid["image_hw"], depth_grid["resized_hw"]
    top, left = depth_grid["crop_top_left"]
    sx, sy = w_res / w_img, h_res / h_img
    k = intrinsics.astype(np.float64).copy()
    k[..., 0, 0] *= sx
    k[..., 1, 1] *= sy
    k[..., 0, 2] = (k[..., 0, 2] + 0.5) * sx - 0.5 - left
    k[..., 1, 2] = (k[..., 1, 2] + 0.5) * sy - 0.5 - top
    return k


def image_intrinsics_from_depth(intrinsics_depth: np.ndarray, depth_grid: dict) -> np.ndarray:
    """Inverse of :func:`depth_intrinsics`: depth-grid intrinsics to original-image intrinsics."""
    (h_img, w_img), (h_res, w_res) = depth_grid["image_hw"], depth_grid["resized_hw"]
    top, left = depth_grid["crop_top_left"]
    sx, sy = w_res / w_img, h_res / h_img
    k = intrinsics_depth.astype(np.float64).copy()
    k[..., 0, 0] /= sx
    k[..., 1, 1] /= sy
    k[..., 0, 2] = (k[..., 0, 2] + 0.5 + left) / sx - 0.5
    k[..., 1, 2] = (k[..., 1, 2] + 0.5 + top) / sy - 0.5
    return k


def mask_to_depth_grid(mask: np.ndarray, depth_grid: dict) -> np.ndarray:
    """Resample an original-resolution boolean mask onto the depth grid (nearest resize, then crop).

    Args:
        mask: (H,W) bool mask on the original image grid.
        depth_grid: ``meta["depth_grid"]`` of the depth maps.

    Returns:
        (h,w) bool mask on the depth grid.
    """
    assert mask.dtype == bool and list(mask.shape) == depth_grid["image_hw"], (mask.dtype, mask.shape)
    rh, rw = depth_grid["resized_hw"]
    top, left = depth_grid["crop_top_left"]
    th, tw = depth_grid["depth_hw"]
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((rw, rh), resample=Image.NEAREST))
    return small[top : top + th, left : left + tw] == 255


def image_to_depth_grid(path: str, depth_grid: dict) -> np.ndarray:
    """Load an RGB image and resize + crop it exactly like MapAnything's ``crop_resize_if_necessary``.

    Returns:
        (h,w,3) uint8 image on the depth grid.
    """
    img = Image.open(path).convert("RGB")
    assert [img.size[1], img.size[0]] == depth_grid["image_hw"], (img.size, depth_grid["image_hw"])
    rh, rw = depth_grid["resized_hw"]
    top, left = depth_grid["crop_top_left"]
    th, tw = depth_grid["depth_hw"]
    img = img.resize((rw, rh), resample=Image.LANCZOS)
    return np.asarray(img)[top : top + th, left : left + tw]


PLY_TYPES = {"<f4": "float", "u1": "uchar"}


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray | None = None,
              labels: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY: float32 xyz, optional float32 normals, uint8 rgb, optional uint8 label.

    Args:
        path: output file.
        xyz: (N,3) positions.
        rgb: (N,3) uint8 colours.
        normals: optional (N,3) normals.
        labels: optional (N,) uint8 per-point class codes (meaning documented by the writer's meta.json).
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3, xyz.shape
    assert rgb.shape == xyz.shape and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        assert normals.shape == xyz.shape, normals.shape
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if labels is not None:
        assert labels.shape == (len(xyz),) and labels.dtype == np.uint8, (labels.shape, labels.dtype)
        fields += [("label", "u1")]
    data = np.empty(len(xyz), dtype=fields)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        data["nx"], data["ny"], data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if labels is not None:
        data["label"] = labels
    ply_types = PLY_TYPES
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {ply_types[t]} {n}" for n, t in fields]
    header += ["end_header"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: str) -> dict[str, np.ndarray]:
    """Read a PLY written by :func:`write_ply`.

    Returns:
        Dict with ``xyz`` (N,3) float32, ``rgb`` (N,3) uint8 and, when present, ``normals`` (N,3) float32
        and ``labels`` (N,) uint8.
    """
    inv_types = {v: k for k, v in PLY_TYPES.items()}
    with open(path, "rb") as f:
        assert f.readline() == b"ply\n"
        assert f.readline() == b"format binary_little_endian 1.0\n"
        n, fields = None, []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "element":
                assert parts[1] == "vertex", line
                n = int(parts[2])
            else:
                assert parts[0] == "property", line
                fields.append((parts[2], inv_types[parts[1]]))
        data = np.frombuffer(f.read(), dtype=fields, count=n)
    names = [name for name, _ in fields]
    out = {"xyz": np.stack([data["x"], data["y"], data["z"]], axis=1),
           "rgb": np.stack([data["red"], data["green"], data["blue"]], axis=1)}
    if "nx" in names:
        out["normals"] = np.stack([data["nx"], data["ny"], data["nz"]], axis=1)
    if "label" in names:
        out["labels"] = data["label"].copy()
    return out


def write_json(path: str, obj: dict) -> None:
    """Write an auxiliary JSON file (e.g. per-product metadata next to meta.json)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> dict:
    """Read a JSON file written by :func:`write_json`."""
    with open(path) as f:
        return json.load(f)


def write_mesh_ply(path: str, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian triangle-mesh PLY (float32 xyz + uint8 rgb per vertex, int32 faces).

    Args:
        vertices: (V,3). faces: (F,3) vertex indices. rgb: (V,3) uint8.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3, vertices.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape
    assert rgb.shape == vertices.shape and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
    assert faces.min() >= 0 and faces.max() < len(vertices), "face index out of range"
    vdata = np.empty(len(vertices), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vdata["x"], vdata["y"], vdata["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vdata["red"], vdata["green"], vdata["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    fdata = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    fdata["n"] = 3
    fdata["v"] = faces
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue",
              f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vdata.tobytes())
        f.write(fdata.tobytes())


def read_mesh_ply(path: str) -> dict[str, np.ndarray]:
    """Read a mesh written by :func:`write_mesh_ply`: ``vertices`` (V,3), ``faces`` (F,3), ``rgb`` (V,3)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        assert header[1] == "format binary_little_endian 1.0", header[1]
        n_v = int(header[2].split()[2])
        n_f = int(header[9].split()[2])
        vdt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        vdata = np.frombuffer(f.read(n_v * 15), dtype=vdt, count=n_v)
        fdata = np.frombuffer(f.read(n_f * 13), dtype=[("n", "u1"), ("v", "<i4", (3,))], count=n_f)
    assert (fdata["n"] == 3).all()
    return {"vertices": np.stack([vdata["x"], vdata["y"], vdata["z"]], 1),
            "faces": fdata["v"].copy(),
            "rgb": np.stack([vdata["red"], vdata["green"], vdata["blue"]], 1)}
