from __future__ import annotations

import torch

from hybrid_gs.camera import Camera
from hybrid_gs.gaussians import GaussianState


def render_gaussians(
    state: GaussianState,
    camera: Camera,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    near_plane: float = 0.05,
    tile_size: int | None = None,
    return_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    device = state.means.device
    image_width = camera.width
    image_height = camera.height
    background_color = torch.tensor(background, device=device, dtype=torch.float32).view(1, 1, 3)
    image = torch.ones((image_height, image_width, 3), device=device, dtype=torch.float32) * background_color
    alpha_image = torch.zeros((image_height, image_width), device=device, dtype=torch.float32)

    camera_points = camera.world_to_camera(state.means)
    depth = camera_points[:, 2]
    valid = depth > near_plane
    if not torch.any(valid):
        return (image, alpha_image) if return_alpha else image

    points = camera_points[valid]
    colors = state.colors[valid]
    opacity = state.opacity[valid]
    scales = state.scales[valid]

    order = torch.argsort(points[:, 2], descending=True)
    points = points[order]
    colors = colors[order]
    opacity = opacity[order]
    scales = scales[order]

    projected_x = camera.fx * (points[:, 0] / points[:, 2]) + camera.cx
    projected_y = camera.cy - camera.fy * (points[:, 1] / points[:, 2])
    sigma = (0.5 * (camera.fx + camera.fy) * scales.mean(dim=-1) / points[:, 2]).clamp(
        0.7,
        max(image_width, image_height) * 0.2,
    )

    effective_tile_size = tile_size if tile_size and tile_size > 0 else max(image_width, image_height)
    image_rows: list[torch.Tensor] = []
    alpha_rows: list[torch.Tensor] = []
    for top in range(0, image_height, effective_tile_size):
        bottom = min(top + effective_tile_size, image_height)
        tile_ys = torch.arange(top, bottom, device=device, dtype=torch.float32)
        row_images: list[torch.Tensor] = []
        row_alphas: list[torch.Tensor] = []
        for left in range(0, image_width, effective_tile_size):
            right = min(left + effective_tile_size, image_width)
            tile_xs = torch.arange(left, right, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(tile_ys, tile_xs, indexing="ij")

            tile_image = torch.ones((bottom - top, right - left, 3), device=device, dtype=torch.float32) * background_color
            tile_alpha = torch.zeros((bottom - top, right - left), device=device, dtype=torch.float32)

            for index in range(points.shape[0]):
                dx = grid_x - projected_x[index]
                dy = grid_y - projected_y[index]
                dist2 = (dx * dx + dy * dy) / (sigma[index] * sigma[index] + 1e-6)
                alpha = opacity[index, 0] * torch.exp(-0.5 * dist2)
                alpha = alpha.clamp(0.0, 0.98).unsqueeze(-1)
                tile_image = tile_image * (1.0 - alpha) + colors[index].view(1, 1, 3) * alpha
                tile_alpha = tile_alpha + (1.0 - tile_alpha) * alpha[..., 0]

            row_images.append(tile_image)
            row_alphas.append(tile_alpha)

        image_rows.append(torch.cat(row_images, dim=1))
        alpha_rows.append(torch.cat(row_alphas, dim=1))

    image = torch.cat(image_rows, dim=0)
    alpha_image = torch.cat(alpha_rows, dim=0)

    image = image.clamp(0.0, 1.0)
    alpha_image = alpha_image.clamp(0.0, 1.0)
    return (image, alpha_image) if return_alpha else image
