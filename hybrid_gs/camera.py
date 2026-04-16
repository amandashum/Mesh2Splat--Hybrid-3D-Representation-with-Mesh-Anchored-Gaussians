from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Camera basis construction and normal estimation both rely on normalized
    # direction vectors; clamp the denominator so degenerate vectors do not
    # produce NaNs during setup.
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class Camera:
    # Store the world-to-camera transform directly so both synthetic and COLMAP
    # cameras can share the same renderer interface. Intrinsics are stored in
    # the usual pinhole form (fx, fy, cx, cy) so the renderer can project
    # Gaussians without caring where the camera came from.
    rotation: torch.Tensor
    translation: torch.Tensor
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        # Project world-space points into the camera coordinate system expected
        # by the renderer. `rotation` and `translation` are already world-to-
        # camera quantities, so this is just an affine transform.
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
    # Build a simple pinhole camera from eye/target/up parameters for synthetic
    # orbit views. This is only used when no COLMAP reconstruction is
    # available, so the code favors clarity over completeness.
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
    # Synthetic fallback cameras used when no real COLMAP scene is provided.
    # These cameras circle the origin at a fixed elevation so the baseline can
    # still train in a self-contained mode.
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
