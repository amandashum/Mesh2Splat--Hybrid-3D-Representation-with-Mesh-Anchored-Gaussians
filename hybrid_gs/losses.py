from __future__ import annotations

import torch


def reconstruction_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Simple photometric objective: robust L1 plus a small MSE term.
    l1 = torch.mean(torch.abs(rendered - target))
    mse = torch.mean((rendered - target) ** 2)
    return l1 + 0.25 * mse


def tether_loss(
    means: torch.Tensor,
    anchors: torch.Tensor,
    normals: torch.Tensor,
    normal_weight: float = 0.25,
) -> torch.Tensor:
    # Penalize anchored splats for drifting too far from their structural support.
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
    # Completion splats are weakly regularized so they can move, but not arbitrarily.
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
    # Detail splats should stay near the prior while still modeling residual appearance variation.
    offsets = means - anchors
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    return (
        tangent_weight * tangent_offsets.pow(2).sum(dim=-1).mean()
        + normal_weight * normal_offsets.pow(2).sum(dim=-1).mean()
    )


def appearance_guidance_loss(colors: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    # Lightweight proxy for a stronger semantic appearance prior.
    distances = ((colors.unsqueeze(1) - palette.unsqueeze(0)) ** 2).sum(dim=-1)
    return distances.min(dim=1).values.mean()


def scale_regularization(scales: torch.Tensor) -> torch.Tensor:
    return scales.mean()


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    return torch.mean(opacity * (1.0 - opacity))
