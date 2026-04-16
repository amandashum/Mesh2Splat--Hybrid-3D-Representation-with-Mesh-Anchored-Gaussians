from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class Camera:
    rotation: torch.Tensor
    translation: torch.Tensor
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        return points @ self.rotation.T + self.translation.unsqueeze(0)

    @property
    def image_size(self) -> int:
        if self.width != self.height:
            raise ValueError("image_size is only defined for square cameras.")
        return self.width


def look_at_camera(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
    width: int,
    height: int,
    fov_degrees: float,
) -> Camera:
    forward = _normalize(target - eye)
    right = _normalize(torch.cross(forward, up, dim=0))
    true_up = _normalize(torch.cross(right, forward, dim=0))
    rotation = torch.stack((right, true_up, forward), dim=0)
    translation = -rotation @ eye
    fx = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    fy = fx
    cx = 0.5 * width
    cy = 0.5 * height
    return Camera(
        rotation=rotation,
        translation=translation,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def orbit_cameras(
    num_views: int,
    radius: float,
    elevation_degrees: float,
    image_width: int,
    image_height: int,
    fov_degrees: float,
    device: torch.device,
) -> list[Camera]:
    elevation = math.radians(elevation_degrees)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    cameras: list[Camera] = []

    for index in range(num_views):
        azimuth = (2.0 * math.pi * index) / max(num_views, 1)
        eye = torch.tensor(
            [
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.sin(elevation),
                radius * math.cos(elevation) * math.sin(azimuth),
            ],
            dtype=torch.float32,
            device=device,
        )
        cameras.append(
            look_at_camera(
                eye=eye,
                target=target,
                up=up,
                width=image_width,
                height=image_height,
                fov_degrees=fov_degrees,
            )
        )

    return cameras
