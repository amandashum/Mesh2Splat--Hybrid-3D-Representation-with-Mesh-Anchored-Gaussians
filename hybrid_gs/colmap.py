from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from hybrid_gs.camera import Camera


@dataclass
class ColmapView:
    # One training image plus its recovered camera model.
    name: str
    image_path: Path
    camera: Camera
    target: torch.Tensor


@dataclass
class ColmapPointCloud:
    # Sparse scene prior exported by COLMAP, enriched with confidence proxies.
    xyz: torch.Tensor
    rgb: torch.Tensor
    error: torch.Tensor
    track_length: torch.Tensor


def _iter_data_lines(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line


def _qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    # COLMAP stores rotations as quaternions in images.txt.
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def _parse_camera_params(model: str, params: list[float]) -> tuple[float, float, float, float]:
    # Support the common undistorted COLMAP camera models used in exported text scenes.
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"}:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    raise ValueError(
        f"Unsupported COLMAP camera model '{model}'. "
        "Export an undistorted model or extend the parser for this model."
    )


def load_colmap_text_dataset(
    model_dir: str | Path,
    image_dir: str | Path,
    device: torch.device,
    max_views: int | None = None,
    resize_long_edge: int | None = None,
) -> list[ColmapView]:
    # Load real images and camera poses so training can use true multi-view supervision.
    model_path = Path(model_dir)
    image_root = Path(image_dir)
    cameras_path = model_path / "cameras.txt"
    images_path = model_path / "images.txt"
    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"Expected cameras.txt and images.txt in {model_path}. "
            "Use COLMAP's model_converter to export the sparse model to TXT."
        )

    cameras: dict[int, dict[str, float | int | str]] = {}
    for line in _iter_data_lines(cameras_path):
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(item) for item in parts[4:]]
        fx, fy, cx, cy = _parse_camera_params(model, params)
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    image_lines = list(_iter_data_lines(images_path))
    views: list[ColmapView] = []
    for index in range(0, len(image_lines), 2):
        parts = image_lines[index].split()
        image_id = int(parts[0])
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float32)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        name = parts[9]

        camera_data = cameras[camera_id]
        image_path = image_root / name
        if not image_path.exists():
            raise FileNotFoundError(f"COLMAP image '{name}' was not found under {image_root}.")

        width = int(camera_data["width"])
        height = int(camera_data["height"])
        fx = float(camera_data["fx"])
        fy = float(camera_data["fy"])
        cx = float(camera_data["cx"])
        cy = float(camera_data["cy"])

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        if (image_width, image_height) != (width, height):
            image = image.resize((width, height), Resampling.BILINEAR)

        if resize_long_edge and max(width, height) > resize_long_edge:
            scale = resize_long_edge / max(width, height)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            image = image.resize((resized_width, resized_height), Resampling.BILINEAR)
            fx *= scale
            fy *= scale
            cx *= scale
            cy *= scale
            width = resized_width
            height = resized_height

        target = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).to(device)
        rotation = torch.from_numpy(_qvec_to_rotation_matrix(qvec)).to(device=device, dtype=torch.float32)
        translation = torch.from_numpy(tvec).to(device=device, dtype=torch.float32)
        camera = Camera(
            rotation=rotation,
            translation=translation,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        views.append(ColmapView(name=name, image_path=image_path, camera=camera, target=target))

    views.sort(key=lambda item: item.name)
    if max_views is not None and max_views > 0 and len(views) > max_views:
        indices = np.linspace(0, len(views) - 1, num=max_views, dtype=int)
        views = [views[index] for index in indices.tolist()]
    return views


def load_colmap_points3d(
    model_dir: str | Path,
    device: torch.device,
    max_points: int | None = None,
) -> ColmapPointCloud:
    # Load sparse 3D points as a lightweight scene prior when no mesh is available.
    model_path = Path(model_dir)
    points_path = model_path / "points3D.txt"
    if not points_path.exists():
        raise FileNotFoundError(
            f"Expected points3D.txt in {model_path}. "
            "Export the sparse model to TXT after running COLMAP."
        )

    xyz_list: list[list[float]] = []
    rgb_list: list[list[float]] = []
    error_list: list[float] = []
    track_length_list: list[float] = []

    for line in _iter_data_lines(points_path):
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb_list.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
        error_list.append(float(parts[7]))
        track_items = parts[8:]
        track_length_list.append(float(len(track_items) // 2))

    if not xyz_list:
        raise ValueError(f"No sparse points were found in {points_path}.")

    xyz = torch.tensor(xyz_list, dtype=torch.float32, device=device)
    rgb = torch.tensor(rgb_list, dtype=torch.float32, device=device)
    error = torch.tensor(error_list, dtype=torch.float32, device=device)
    track_length = torch.tensor(track_length_list, dtype=torch.float32, device=device)

    centered = xyz - xyz.mean(dim=0, keepdim=True)
    scale = centered.norm(dim=-1).amax().clamp_min(1e-6)
    xyz = centered / scale

    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        weights = (track_length / track_length.sum().clamp_min(1e-8)).detach().cpu().numpy()
        choice = np.random.choice(xyz.shape[0], size=max_points, replace=False, p=weights)
        indices = torch.tensor(choice, device=device, dtype=torch.long)
        xyz = xyz[indices]
        rgb = rgb[indices]
        error = error[indices]
        track_length = track_length[indices]

    return ColmapPointCloud(
        xyz=xyz,
        rgb=rgb,
        error=error,
        track_length=track_length,
    )
