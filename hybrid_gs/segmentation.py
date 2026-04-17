from __future__ import annotations

import torch


def _shift_mask(mask: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    # Small morphological operators are enough for the current heuristic
    # segmentation pass. Implement them with tensor shifts to avoid new
    # dependencies just for dilation/erosion.
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
    # Dilating the mesh-supported silhouette creates a candidate region around
    # the building where completion is allowed to explore. This is the main
    # mechanism that lets completion fill nearby gaps without spreading across
    # the whole image.
    if radius <= 0:
        return mask

    result = mask.bool()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            result = result | _shift_mask(mask, dy, dx).bool()
    return result.float()


def build_semantic_masks(
    target: torch.Tensor,
    mesh_alpha: torch.Tensor,
    *,
    dilation_radius: int = 10,
) -> dict[str, torch.Tensor]:
    # This is a pragmatic segmentation pass, not a learned semantic model.
    # It combines:
    # - the current geometry silhouette (`mesh_alpha`) as "where the building
    #   already is"
    # - simple color cues for vegetation and sky
    # The result is used to steer completion toward plausible building regions
    # and away from obvious non-building areas.
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("build_semantic_masks expects an HxWx3 RGB target tensor.")

    red = target[..., 0]
    green = target[..., 1]
    blue = target[..., 2]
    brightness = target.mean(dim=-1)

    building_core = (mesh_alpha > 0.08).float()
    building_context = dilate_mask(building_core, dilation_radius)
    immediate_context = dilate_mask(building_core, max(dilation_radius // 2, 2))

    vegetation = (
        (green > red + 0.05)
        & (green > blue + 0.02)
        & (green > 0.18)
    ).float()
    sky = (
        (blue > red + 0.06)
        & (blue > green + 0.03)
        & (brightness > 0.45)
    ).float()

    # Ground is approximated very conservatively: only the lower image band and
    # only away from the building silhouette. This avoids suppressing the
    # lower facade when the camera points slightly upward.
    height = target.shape[0]
    row_coords = torch.linspace(0.0, 1.0, steps=height, device=target.device).view(height, 1)
    ground_band = (row_coords > 0.78).expand_as(red).float()
    ground = ground_band * (1.0 - building_context)

    # Completion is allowed near the building, even around bushes, but should
    # avoid obvious sky and distant vegetation/ground. Near-building vegetation
    # is downweighted rather than removed completely because some desired fills
    # sit behind bushes.
    far_vegetation = vegetation * (1.0 - immediate_context)
    invalid = torch.clamp(sky + ground + far_vegetation, 0.0, 1.0)
    completion_region = torch.clamp(building_context * (1.0 - invalid), 0.0, 1.0)
    completion_band = torch.clamp(building_context - building_core, 0.0, 1.0)

    return {
        "building_core": building_core,
        "building_context": building_context,
        "vegetation": vegetation,
        "sky": sky,
        "ground": ground,
        "completion_region": completion_region,
        "completion_band": completion_band,
    }
