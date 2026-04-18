from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling


def _select_primary_mask(
    masks: list[dict],
    image_height: int,
    image_width: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> np.ndarray:
    # SAM can return dozens or hundreds of candidate masks. For this pipeline
    # we need one broad foreground/surface mask that can be used as a semantic
    # gate for completion. The scoring below favors masks that are:
    # - reasonably large, but not almost the whole image
    # - stable / high quality according to SAM's own outputs
    # - near the image center, which is a good default for object-centric and
    #   scene-centric captures alike
    image_area = float(image_height * image_width)
    best_score = None
    best_mask = None

    for candidate in masks:
        segmentation = candidate.get("segmentation")
        if segmentation is None:
            continue

        area_ratio = float(candidate.get("area", float(segmentation.sum()))) / max(image_area, 1.0)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        x, y, width, height = candidate.get("bbox", [0.0, 0.0, float(image_width), float(image_height)])
        center_x = x + width * 0.5
        center_y = y + height * 0.5
        norm_dx = (center_x / max(image_width, 1)) - 0.5
        norm_dy = (center_y / max(image_height, 1)) - 0.5
        center_penalty = norm_dx * norm_dx + norm_dy * norm_dy

        stability = float(candidate.get("stability_score", 1.0))
        predicted_iou = float(candidate.get("predicted_iou", 1.0))
        score = predicted_iou * stability * (0.4 + area_ratio) * (1.0 - min(center_penalty, 0.9))

        if best_score is None or score > best_score:
            best_score = score
            best_mask = segmentation

    if best_mask is None:
        if not masks:
            return np.zeros((image_height, image_width), dtype=np.float32)
        largest = max(masks, key=lambda candidate: float(candidate.get("area", 0.0)))
        best_mask = largest["segmentation"]

    return np.asarray(best_mask, dtype=np.float32)


def generate_sam_masks_for_paths(
    image_paths: list[Path],
    output_shapes: list[tuple[int, int]],
    checkpoint_path: str,
    model_type: str,
    device_name: str,
    *,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95,
) -> list[torch.Tensor]:
    # This module is optional. The import stays local so users who do not want
    # SAM installed can keep using the heuristic segmentation path unchanged.
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:  # pragma: no cover - exercised only in missing optional dependency setups
        raise ImportError(
            "SAM support requires the official `segment_anything` package. "
            "Install Meta's Segment Anything repository and provide --sam-checkpoint."
        ) from exc

    if len(image_paths) != len(output_shapes):
        raise ValueError("generate_sam_masks_for_paths expects one output shape per image path.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device_name)
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks: list[torch.Tensor] = []
    for image_path, (target_height, target_width) in zip(image_paths, output_shapes):
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        generated_masks = mask_generator.generate(image_array)
        selected_mask = _select_primary_mask(
            generated_masks,
            image_height=image_array.shape[0],
            image_width=image_array.shape[1],
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        resized_mask = Image.fromarray((selected_mask * 255.0).astype(np.uint8), mode="L").resize(
            (target_width, target_height),
            Resampling.BILINEAR,
        )
        mask_tensor = torch.from_numpy(np.asarray(resized_mask, dtype=np.float32) / 255.0)
        masks.append(mask_tensor)

    return masks
