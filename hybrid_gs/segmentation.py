from __future__ import annotations

import torch


def _shift_mask(mask: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    # Lightweight morphology is enough for the current segmentation pass.
    # Using tensor shifts keeps the implementation self-contained and fast.
    height, width = mask.shape
    shifted = torch.zeros_like(mask)

    src_top = max(-dy, 0)
    src_bottom = min(height - dy, height)
    dst_top = max(dy, 0)
    dst_bottom = min(height + dy, height)

    src_left = max(-dx, 0)
    src_right = min(width - dx, width)
    dst_left = max(dx, 0)
    dst_right = min(width + dx, width)

    if src_top >= src_bottom or src_left >= src_right:
        return shifted

    shifted[dst_top:dst_bottom, dst_left:dst_right] = mask[src_top:src_bottom, src_left:src_right]
    return shifted


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    # Dilation expands the known-surface silhouette into a nearby continuation
    # band. That band is where completion is allowed to search for missing
    # surfaces without spreading across the full image.
    if radius <= 0:
        return mask

    result = mask.bool()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            result = result | _shift_mask(mask, dy, dx).bool()
    return result.float()


def build_scene_structure_masks(
    target: torch.Tensor,
    prior_render: torch.Tensor,
    prior_alpha: torch.Tensor,
    *,
    dilation_radius: int = 10,
) -> dict[str, torch.Tensor]:
    # This is a scene-general structure pass, not a semantic classifier. It
    # uses three cues:
    # - `prior_alpha` tells us where the current geometry already explains a
    #   surface
    # - `prior_render` vs `target` highlights pixels where the image disagrees
    #   with the current geometry, which often means occluders or missing
    #   surfaces
    # - simple brightness/saturation heuristics identify likely empty
    #   background areas
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("build_scene_structure_masks expects an HxWx3 RGB target tensor.")

    # These three per-pixel summaries are the only cues the current heuristic
    # segmentation pass uses. Keeping them explicit makes it easier to tune the
    # completion masks later or replace them with a learned segmenter such as
    # SAM without changing the rest of the pipeline contract.
    brightness = target.mean(dim=-1)
    saturation = target.amax(dim=-1) - target.amin(dim=-1)
    residual = torch.mean(torch.abs(target - prior_render), dim=-1)

    surface_core = (prior_alpha > 0.08).float()
    surface_context = dilate_mask(surface_core, dilation_radius)
    near_surface_band = torch.clamp(surface_context - surface_core, 0.0, 1.0)

    # Candidate occluders are near the known surface and disagree strongly with
    # the current geometry. This catches things like bushes in front of a wall
    # without forcing a building-specific label set.
    occluder = (
        (surface_context > 0.0)
        & (surface_core < 0.5)
        & (residual > 0.12)
        & (saturation > 0.10)
    ).float()

    # Clear background is where the image is bright/flat and well away from the
    # current surface. This is deliberately conservative because false
    # background suppression is worse than leaving some pixels unconstrained.
    far_from_surface = 1.0 - dilate_mask(surface_context, max(dilation_radius // 2, 2))
    clear_background = (
        (far_from_surface > 0.0)
        & (brightness > 0.48)
        & (saturation < 0.22)
    ).float()

    # A lower-image support band helps suppress floating completion under the
    # main structure without assuming the scene is specifically a building.
    height = target.shape[0]
    row_coords = torch.linspace(0.0, 1.0, steps=height, device=target.device).view(height, 1)
    lower_support = ((row_coords > 0.80).expand_as(brightness)).float() * far_from_surface

    # `completion_allowed` is the broad region where completion is permitted to
    # add geometry. `completion_focus` is narrower: it emphasizes where the
    # current surface should plausibly continue, especially around near-surface
    # occluders.
    completion_allowed = torch.clamp(surface_context * (1.0 - clear_background), 0.0, 1.0)
    completion_focus = torch.clamp(near_surface_band + 0.75 * occluder - lower_support, 0.0, 1.0)

    return {
        "surface_core": surface_core,
        "surface_context": surface_context,
        "near_surface_band": near_surface_band,
        "occluder": occluder,
        "clear_background": clear_background,
        "lower_support": lower_support,
        "completion_allowed": completion_allowed,
        "completion_focus": completion_focus,
    }
