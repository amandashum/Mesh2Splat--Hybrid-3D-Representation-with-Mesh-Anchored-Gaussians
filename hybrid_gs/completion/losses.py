from __future__ import annotations

import torch


def completion_continuity_loss(
    means: torch.Tensor,
    seeds: torch.Tensor,
    normals: torch.Tensor,
    strengths: torch.Tensor,
    neighbor_count: int = 4,
    tangent_weight: float = 0.15,
    continuity_weight: float = 0.35,
) -> torch.Tensor:
    # Completion splats should extend local surfaces, not drift independently.
    # This loss keeps them near the boundary/frontier seeds while also forcing
    # nearby completion splats to move in similar ways.
    if means.shape[0] == 0:
        return torch.zeros((), device=means.device)

    weights = strengths / strengths.sum().clamp_min(1e-8)
    offsets = means - seeds
    # Split motion into normal and tangential components so the branch can move
    # outward from a hole boundary while still discouraging uncontrolled
    # sideways spreading.
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    anchor_term = (
        (normal_offsets.pow(2).sum(dim=-1) + tangent_weight * tangent_offsets.pow(2).sum(dim=-1)) * weights
    ).sum()

    if means.shape[0] == 1:
        # A single completion splat has no neighborhood, so only the local
        # anchor term applies.
        return anchor_term

    k = min(neighbor_count + 1, seeds.shape[0])
    distances = torch.cdist(seeds.detach(), seeds.detach())
    neighbor_indices = torch.topk(distances, k=k, largest=False).indices[:, 1:]
    if neighbor_indices.shape[1] == 0:
        return anchor_term

    # Nearby completion splats should behave like one coherent patch instead of
    # each following its own offset direction. Matching each splat to the mean
    # offset of its nearest neighbors is a cheap proxy for that surface
    # continuation behavior.
    neighbor_offsets = offsets[neighbor_indices]
    neighbor_mean = neighbor_offsets.mean(dim=1)
    continuity_term = ((offsets - neighbor_mean).pow(2).sum(dim=-1) * weights).sum()
    return anchor_term + continuity_weight * continuity_term
