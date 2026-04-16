from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


PROMPT_PALETTES = {
    "car": [[0.88, 0.18, 0.15], [0.14, 0.15, 0.18], [0.82, 0.82, 0.84]],
    "tree": [[0.20, 0.46, 0.19], [0.55, 0.69, 0.32], [0.45, 0.28, 0.14]],
    "stone": [[0.75, 0.73, 0.68], [0.58, 0.57, 0.55], [0.36, 0.35, 0.34]],
    "statue": [[0.76, 0.74, 0.69], [0.61, 0.60, 0.58], [0.31, 0.31, 0.32]],
    "ice": [[0.86, 0.93, 0.98], [0.52, 0.72, 0.90], [0.20, 0.42, 0.67]],
    "building": [[0.76, 0.75, 0.72], [0.58, 0.56, 0.54], [0.26, 0.28, 0.33]],
    "robot": [[0.77, 0.79, 0.82], [0.19, 0.23, 0.29], [0.28, 0.55, 0.86]],
    "default": [[0.79, 0.67, 0.48], [0.43, 0.56, 0.78], [0.26, 0.27, 0.31]],
}


def prompt_palette(prompt: str, device: torch.device) -> torch.Tensor:
    # The palette is a placeholder appearance prior, not a learned generative model.
    lower_prompt = prompt.lower()
    for keyword, palette in PROMPT_PALETTES.items():
        if keyword != "default" and keyword in lower_prompt:
            return torch.tensor(palette, dtype=torch.float32, device=device)
    return torch.tensor(PROMPT_PALETTES["default"], dtype=torch.float32, device=device)


def procedural_colors(points: torch.Tensor, normals: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    # Cheap initialization used when real colors are not available from COLMAP points.
    indices = (
        (points[:, 0] > 0).long()
        + (points[:, 1] > 0).long()
        + 2 * (points[:, 2] > 0).long()
    ) % palette.shape[0]
    base = palette[indices]
    shading = 0.60 + 0.40 * ((normals[:, 1:2] + 1.0) * 0.5)
    tint = 0.12 * ((normals + 1.0) * 0.5)
    return (base * shading + tint).clamp(0.02, 0.98)


@dataclass
class GaussianState:
    means: torch.Tensor
    scales: torch.Tensor
    colors: torch.Tensor
    opacity: torch.Tensor


def concat_states(*states: GaussianState) -> GaussianState:
    return GaussianState(
        means=torch.cat([state.means for state in states], dim=0),
        scales=torch.cat([state.scales for state in states], dim=0),
        colors=torch.cat([state.colors for state in states], dim=0),
        opacity=torch.cat([state.opacity for state in states], dim=0),
    )


class AnchoredGaussianModel(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        normals: torch.Tensor,
        prompt: str,
        colors_override: torch.Tensor | None = None,
        init_scale: float = 0.075,
        jitter: float = 0.03,
    ) -> None:
        super().__init__()
        # Anchored splats stay close to the structural prior and carry the base appearance.
        palette = prompt_palette(prompt, anchors.device)
        colors = colors_override if colors_override is not None else procedural_colors(anchors, normals, palette)
        noise = jitter * torch.randn_like(anchors)

        self.register_buffer("anchor_positions", anchors)
        self.register_buffer("anchor_normals", normals)
        self.register_buffer("palette", palette)

        self.means = nn.Parameter(anchors + 0.5 * noise + 0.5 * jitter * normals)
        self.log_scales = nn.Parameter(torch.full_like(anchors, init_scale).log())
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(0.02, 0.98)))
        self.opacity_logits = nn.Parameter(torch.full((anchors.shape[0], 1), 0.70, device=anchors.device).logit())

    def state(self) -> GaussianState:
        scales = torch.exp(self.log_scales).clamp(0.01, 0.20)
        colors = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits).clamp(0.02, 0.98)
        return GaussianState(means=self.means, scales=scales, colors=colors, opacity=opacity)


class HybridGaussianModel(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        normals: torch.Tensor,
        detail_anchors: torch.Tensor,
        detail_normals: torch.Tensor,
        completion_seeds: torch.Tensor,
        completion_normals: torch.Tensor,
        prompt: str,
        anchor_colors: torch.Tensor | None = None,
        detail_colors: torch.Tensor | None = None,
        completion_colors: torch.Tensor | None = None,
        anchored_init_scale: float = 0.075,
        detail_init_scale: float = 0.040,
        completion_init_scale: float = 0.11,
        anchored_jitter: float = 0.03,
        detail_jitter: float = 0.025,
        completion_jitter: float = 0.09,
        detail_offset: float = 0.02,
        completion_offset: float = 0.10,
    ) -> None:
        super().__init__()
        # The hybrid model separates trusted structure, local detail, and speculative completion.

        self.anchored = AnchoredGaussianModel(
            anchors=anchors,
            normals=normals,
            prompt=prompt,
            colors_override=anchor_colors,
            init_scale=anchored_init_scale,
            jitter=anchored_jitter,
        )

        palette = self.anchored.palette
        detail_count = detail_anchors.shape[0]
        detail_colors = detail_colors if detail_colors is not None else procedural_colors(detail_anchors, detail_normals, palette)
        completion_count = completion_seeds.shape[0]
        completion_colors = completion_colors if completion_colors is not None else procedural_colors(completion_seeds, completion_normals, palette)

        detail_outward = detail_offset * detail_normals
        detail_random_walk = detail_jitter * torch.randn_like(detail_anchors)
        # Detail splats are allowed small deviations around the prior surface.
        self.register_buffer("detail_anchor_positions", detail_anchors)
        self.register_buffer("detail_anchor_normals", detail_normals)

        self.detail_means = nn.Parameter(detail_anchors + detail_outward + detail_random_walk)
        self.detail_log_scales = nn.Parameter(
            torch.full_like(detail_anchors, detail_init_scale).log()
        )
        self.detail_color_logits = nn.Parameter(torch.logit(detail_colors.clamp(0.02, 0.98)))
        self.detail_opacity_logits = nn.Parameter(
            torch.full((detail_count, 1), 0.32, device=anchors.device).logit()
        )

        outward = completion_offset * completion_normals
        random_walk = completion_jitter * torch.randn_like(completion_seeds)
        # Completion splats start farther from the prior because they are meant to explore gaps.
        self.register_buffer("completion_seed_positions", completion_seeds)
        self.register_buffer("completion_seed_normals", completion_normals)

        self.completion_means = nn.Parameter(completion_seeds + outward + random_walk)
        self.completion_log_scales = nn.Parameter(
            torch.full_like(completion_seeds, completion_init_scale).log()
        )
        self.completion_color_logits = nn.Parameter(torch.logit(completion_colors.clamp(0.02, 0.98)))
        self.completion_opacity_logits = nn.Parameter(
            torch.full((completion_count, 1), 0.42, device=anchors.device).logit()
        )

    @property
    def anchor_positions(self) -> torch.Tensor:
        return self.anchored.anchor_positions

    @property
    def anchor_normals(self) -> torch.Tensor:
        return self.anchored.anchor_normals

    @property
    def palette(self) -> torch.Tensor:
        return self.anchored.palette

    def anchored_state(self) -> GaussianState:
        return self.anchored.state()

    def detail_state(self) -> GaussianState:
        scales = torch.exp(self.detail_log_scales).clamp(0.01, 0.12)
        colors = torch.sigmoid(self.detail_color_logits)
        opacity = torch.sigmoid(self.detail_opacity_logits).clamp(0.02, 0.90)
        return GaussianState(
            means=self.detail_means,
            scales=scales,
            colors=colors,
            opacity=opacity,
        )

    def completion_state(self) -> GaussianState:
        scales = torch.exp(self.completion_log_scales).clamp(0.02, 0.28)
        colors = torch.sigmoid(self.completion_color_logits)
        opacity = torch.sigmoid(self.completion_opacity_logits).clamp(0.02, 0.98)
        return GaussianState(
            means=self.completion_means,
            scales=scales,
            colors=colors,
            opacity=opacity,
        )

    def state(self) -> GaussianState:
        return concat_states(
            self.anchored_state(),
            self.detail_state(),
            self.completion_state(),
        )
