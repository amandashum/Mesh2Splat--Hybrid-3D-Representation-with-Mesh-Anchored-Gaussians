from __future__ import annotations

import torch


def reconstruction_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Simple photometric objective: robust L1 plus a small MSE term. The L1
    # part helps with sharper residuals, while the MSE term smooths training.
    l1 = torch.mean(torch.abs(rendered - target))
    mse = torch.mean((rendered - target) ** 2)
    return l1 + 0.25 * mse


def tether_loss(
    means: torch.Tensor,
    anchors: torch.Tensor,
    normals: torch.Tensor,
    normal_weight: float = 0.25,
) -> torch.Tensor:
    # Penalize anchored splats for drifting too far from their structural
    # support. Tangential drift is punished, but movement along the surface
    # normal is also regularized so anchors remain tied to the prior.
    offsets = means - anchors
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    return tangent_offsets.pow(2).sum(dim=-1).mean() + normal_weight * normal_offsets.pow(2).sum(dim=-1).mean()


def completion_smoothness_loss(
    means: torch.Tensor,
    seeds: torch.Tensor,
    normals: torch.Tensor,
    radial_weight: float = 0.05,
) -> torch.Tensor:
    # Completion splats are weakly regularized so they can move, but not
    # arbitrarily. They should explore uncertain space without immediately
    # collapsing into noise.
    offsets = means - seeds
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    return radial_weight * tangent_offsets.pow(2).sum(dim=-1).mean() + normal_offsets.pow(2).sum(dim=-1).mean()


def detail_tether_loss(
    means: torch.Tensor,
    anchors: torch.Tensor,
    normals: torch.Tensor,
    tangent_weight: float = 0.20,
    normal_weight: float = 1.0,
) -> torch.Tensor:
    # Detail splats should stay near the prior while still modeling residual
    # appearance variation. They get more freedom tangentially than the
    # anchored branch, but still remain local.
    offsets = means - anchors
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    return (
        tangent_weight * tangent_offsets.pow(2).sum(dim=-1).mean()
        + normal_weight * normal_offsets.pow(2).sum(dim=-1).mean()
    )


def appearance_guidance_loss(colors: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    # Lightweight proxy for a stronger semantic appearance prior. Each splat is
    # softly encouraged to stay near at least one palette color.
    distances = ((colors.unsqueeze(1) - palette.unsqueeze(0)) ** 2).sum(dim=-1)
    return distances.min(dim=1).values.mean()


def scale_regularization(scales: torch.Tensor) -> torch.Tensor:
    # Prevent splats from expanding indefinitely.
    return scales.mean()


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    # Encourage opacity values away from the ambiguous 0.5 region.
    return torch.mean(opacity * (1.0 - opacity))


def completion_region_loss(
    completion_alpha: torch.Tensor,
    allowed_region: torch.Tensor,
    focus_region: torch.Tensor,
    outside_weight: float = 1.0,
    inside_weight: float = 0.25,
) -> torch.Tensor:
    # A purely photometric completion branch will happily grow into roofs,
    # sky, ground, or vegetation if that helps the image loss a bit. This
    # region loss keeps completion concentrated near plausible building gaps.
    outside_penalty = completion_alpha * (1.0 - allowed_region)
    inside_penalty = (1.0 - completion_alpha) * focus_region
    return outside_weight * outside_penalty.mean() + inside_weight * inside_penalty.mean()
