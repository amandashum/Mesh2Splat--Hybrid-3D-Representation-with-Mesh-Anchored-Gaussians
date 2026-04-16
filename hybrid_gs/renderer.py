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
    support_scale: float = 2.0,
    alpha_threshold: float = 1e-3,
    return_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # This is an intentionally simple differentiable splat renderer. It is
    # useful for experiments, but it is much less sophisticated than optimized
    # 3DGS renderers from the literature: Gaussians are isotropic in screen
    # space, compositing is brute-force, and visibility handling is simple.
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
        # If nothing projects in front of the camera, return the background.
        return (image, alpha_image) if return_alpha else image

    points = camera_points[valid]
    colors = state.colors[valid]
    opacity = state.opacity[valid]
    scales = state.scales[valid]

    order = torch.argsort(points[:, 2], descending=True)

    # Sort back-to-front so the simple alpha compositing behaves sensibly.
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

    # Use a controllable screen-space footprint so each tile only considers
    # splats whose projected support overlaps that tile. Smaller support windows
    # make the renderer more focused and reduce wasted work on far-away splats.
    footprint_radius = support_scale * sigma
    min_x = projected_x - footprint_radius
    max_x = projected_x + footprint_radius
    min_y = projected_y - footprint_radius
    max_y = projected_y + footprint_radius

    effective_tile_size = tile_size if tile_size and tile_size > 0 else max(image_width, image_height)
    
    # Tiling lowers peak memory use. Per-tile culling avoids looping over
    # splats that cannot affect a tile, which is especially important for
    # scene-mode runs with many real views.
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

            tile_overlap = (
                (max_x >= left)
                & (min_x <= (right - 1))
                & (max_y >= top)
                & (min_y <= (bottom - 1))
            )
            if not torch.any(tile_overlap):
                # Most splats will miss most tiles; skip those tiles cheaply.
                row_images.append(tile_image)
                row_alphas.append(tile_alpha)
                continue

            tile_projected_x = projected_x[tile_overlap]
            tile_projected_y = projected_y[tile_overlap]
            tile_sigma = sigma[tile_overlap]
            tile_colors = colors[tile_overlap]
            tile_opacity = opacity[tile_overlap]
            tile_min_x = min_x[tile_overlap]
            tile_max_x = max_x[tile_overlap]
            tile_min_y = min_y[tile_overlap]
            tile_max_y = max_y[tile_overlap]

            for index in range(tile_projected_x.shape[0]):
                # Skip splats whose clipped support barely touches the tile. This
                # keeps the renderer focused on the current window rather than
                # spending work on effectively invisible tails.
                clipped_left = torch.maximum(tile_min_x[index], torch.tensor(left, device=device, dtype=torch.float32))
                clipped_right = torch.minimum(tile_max_x[index], torch.tensor(right - 1, device=device, dtype=torch.float32))
                clipped_top = torch.maximum(tile_min_y[index], torch.tensor(top, device=device, dtype=torch.float32))
                clipped_bottom = torch.minimum(tile_max_y[index], torch.tensor(bottom - 1, device=device, dtype=torch.float32))
                if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
                    continue

                dx = grid_x - tile_projected_x[index]
                dy = grid_y - tile_projected_y[index]
                dist2 = (dx * dx + dy * dy) / (tile_sigma[index] * tile_sigma[index] + 1e-6)
                alpha = tile_opacity[index, 0] * torch.exp(-0.5 * dist2)
                if torch.amax(alpha) < alpha_threshold:
                    continue
                alpha = alpha.clamp(0.0, 0.98).unsqueeze(-1)
                tile_image = tile_image * (1.0 - alpha) + tile_colors[index].view(1, 1, 3) * alpha
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
