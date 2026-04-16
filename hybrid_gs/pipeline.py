from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from hybrid_gs.camera import orbit_cameras
from hybrid_gs.colmap import ColmapPointCloud, load_colmap_points3d, load_colmap_text_dataset
from hybrid_gs.gaussians import GaussianState, HybridGaussianModel, prompt_palette
from interactive_splat_viewer import (
    load_metadata as viewer_load_metadata,
    load_state_from_npz,
    maybe_subsample,
    state_to_figure,
)
from hybrid_gs.losses import (
    appearance_guidance_loss,
    completion_smoothness_loss,
    detail_tether_loss,
    opacity_regularization,
    reconstruction_loss,
    scale_regularization,
    tether_loss,
)
from hybrid_gs.mesh import Mesh, load_obj_mesh, primitive_mesh_from_prompt, sample_completion_regions, sample_surface
from hybrid_gs.renderer import render_gaussians


@dataclass
class HybridConfig:
    prompt: str
    mesh_path: str | None
    colmap_model_dir: str | None
    colmap_image_dir: str | None
    scene_mode: bool
    reference_image_path: str | None
    reference_mask_path: str | None
    out_dir: Path
    num_splats: int
    steps: int
    num_views: int
    image_size: int
    colmap_resize_long_edge: int | None
    render_tile_size: int | None
    render_support_scale: float
    render_alpha_threshold: float
    prompt_viewer: bool
    lr: float
    seed: int
    device: torch.device
    lambda_tether: float = 3.0
    lambda_detail: float = 0.25
    lambda_completion: float = 0.35
    lambda_appearance: float = 0.15
    lambda_scale: float = 0.02
    lambda_opacity: float = 0.01
    lambda_mask: float = 0.20
    num_detail_splats: int = 192
    num_completion_splats: int = 128
    max_sparse_points: int = 12000


def set_seed(seed: int) -> None:
    # Deterministic seeds make small experiments easier to compare.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_image(path: Path, image: torch.Tensor) -> None:
    # Save tensor images in a viewer-friendly format for quick comparisons.
    array = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_gaussian_state(path: Path, state: GaussianState) -> None:
    # Save the learned Gaussian cloud so later tools can inspect or visualize
    # it without retraining.
    np.savez(
        path,
        means=state.means.detach().cpu().numpy(),
        scales=state.scales.detach().cpu().numpy(),
        colors=state.colors.detach().cpu().numpy(),
        opacity=state.opacity.detach().cpu().numpy(),
    )


def load_mesh(cfg: HybridConfig) -> Mesh:
    # Mesh mode either consumes a user-supplied OBJ or falls back to a simple
    # primitive so the repo remains runnable without external assets.
    if cfg.mesh_path:
        return load_obj_mesh(cfg.mesh_path, cfg.device)
    return primitive_mesh_from_prompt(cfg.prompt, cfg.device)


def save_metadata(path: Path, payload: dict[str, int | float | str]) -> None:
    # Keep a small sidecar file describing the branch split for the viewer.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_prompt_to_create_viewer(cfg: HybridConfig) -> None:
    # Optional convenience hook so a training run can immediately produce
    # `viewer.html` without making the user remember a second command.
    if not cfg.prompt_viewer:
        return

    try:
        response = input("Create interactive viewer HTML now? [y/N] ").strip().lower()
    except EOFError:
        return

    if response not in {"y", "yes"}:
        return

    state_path = cfg.out_dir / "gaussian_state.npz"
    metadata_path = cfg.out_dir / "gaussian_metadata.txt"
    output_html = cfg.out_dir / "viewer.html"

    state = load_state_from_npz(state_path)
    state = maybe_subsample(state, max_splats=3000)
    metadata = viewer_load_metadata(metadata_path)
    figure = state_to_figure(
        state=state,
        metadata=metadata,
        mesh_path=cfg.mesh_path,
        title=f"Interactive Viewer: {cfg.out_dir.name}",
        size_scale=38.0,
        min_size=3.0,
        mesh_opacity=0.22,
        show_wireframe=bool(cfg.mesh_path),
    )

    import plotly.io as pio

    pio.write_html(figure, file=str(output_html), auto_open=False, include_plotlyjs=True)
    print(f"Saved interactive viewer to: {output_html}")


def build_proxy_targets(mesh: Mesh, cameras, prompt: str, image_size: int) -> list[torch.Tensor]:
    # Synthetic targets keep the repo self-contained when no real dataset is
    # provided. A denser "teacher" cloud is rendered from orbit cameras and the
    # trainable cloud tries to match those images.
    samples, normals = sample_surface(mesh, 4 * 512)
    palette = prompt_palette(prompt, mesh.vertices.device)
    colors = build_palette_colors(samples, normals, palette)
    teacher = GaussianState(
        means=samples,
        scales=torch.full_like(samples, 0.035),
        colors=colors,
        opacity=torch.full((samples.shape[0], 1), 0.35, device=samples.device),
    )
    return [render_gaussians(teacher, camera) for camera in cameras]


def build_palette_colors(points: torch.Tensor, normals: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    # Smooth palette interpolation gives scene-mode splats a more stable
    # fallback initialization than assigning one hard color per point.
    radial = points.norm(dim=-1, keepdim=True)
    vertical = 0.5 * (normals[:, 1:2] + 1.0)
    blend = (0.55 * radial + 0.45 * vertical).clamp(0.0, 0.999)
    low = palette[0].view(1, 3)
    mid = palette[min(1, palette.shape[0] - 1)].view(1, 3)
    high = palette[min(2, palette.shape[0] - 1)].view(1, 3)
    mixed = torch.where(
        blend < 0.5,
        low + (blend * 2.0) * (mid - low),
        mid + ((blend - 0.5) * 2.0) * (high - mid),
    )
    tint = 0.08 * ((normals + 1.0) * 0.5)
    return (mixed + tint).clamp(0.02, 0.98)


def load_training_views(cfg: HybridConfig, mesh: Mesh) -> tuple[list, list[torch.Tensor], str]:
    # Switch between real COLMAP supervision and synthetic fallback views. The
    # rest of the optimizer only sees `(camera, target)` pairs either way.
    if cfg.colmap_model_dir and cfg.colmap_image_dir:
        views = load_colmap_text_dataset(
            model_dir=cfg.colmap_model_dir,
            image_dir=cfg.colmap_image_dir,
            device=cfg.device,
            max_views=cfg.num_views if cfg.num_views > 0 else None,
            resize_long_edge=cfg.colmap_resize_long_edge,
        )
        if not views:
            raise ValueError("No COLMAP views were loaded.")
        return [view.camera for view in views], [view.target for view in views], "colmap"

    cameras = orbit_cameras(
        num_views=cfg.num_views,
        radius=2.8,
        elevation_degrees=20.0,
        image_width=cfg.image_size,
        image_height=cfg.image_size,
        fov_degrees=45.0,
        device=cfg.device,
    )
    return cameras, build_proxy_targets(mesh, cameras, cfg.prompt, cfg.image_size), "synthetic"


def estimate_point_normals(points: torch.Tensor, cameras: list) -> torch.Tensor:
    # Approximate normals from local neighborhoods so sparse COLMAP points can
    # seed oriented splats. This is heuristic but good enough for a scene prior.
    if points.shape[0] == 0:
        return points.new_zeros((0, 3))

    if cameras:
        # Face normals outward from the nearest recovered camera when possible.
        camera_centers = []
        for camera in cameras:
            center = -camera.rotation.T @ camera.translation
            camera_centers.append(center)
        camera_centers_tensor = torch.stack(camera_centers, dim=0)
        diffs = camera_centers_tensor.unsqueeze(1) - points.unsqueeze(0)
        distances = diffs.norm(dim=-1)
        nearest_view = torch.argmin(distances, dim=0)
        outward = points - camera_centers_tensor[nearest_view]
        outward = outward / outward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        outward = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    if points.shape[0] < 4:
        return outward

    with torch.no_grad():
        # Estimate one PCA normal per point from its nearest neighbors.
        pairwise = torch.cdist(points, points)
        k = min(8, max(points.shape[0] - 1, 1))
        knn = torch.topk(pairwise, k=k + 1, largest=False).indices[:, 1:]
        neighbors = points[knn]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        covariances = centered.transpose(1, 2) @ centered / max(k, 1)
        _, eigenvectors = torch.linalg.eigh(covariances)
        normals = eigenvectors[:, :, 0]
        aligned = torch.where(
            (normals * outward).sum(dim=-1, keepdim=True) < 0.0,
            -normals,
            normals,
        )
        return aligned / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _sample_indices_from_weights(num_items: int, sample_count: int, weights: torch.Tensor) -> torch.Tensor:
    # Shared weighted sampler used to pick anchors/detail/completion seeds from
    # the sparse COLMAP scene prior.
    if sample_count <= 0:
        return torch.zeros((0,), device=weights.device, dtype=torch.long)
    if num_items == 0:
        raise ValueError("Cannot sample from an empty point set.")
    sample_count = min(sample_count, num_items)
    normalized = (weights / weights.sum().clamp_min(1e-8)).detach().cpu().numpy()
    choice = np.random.choice(num_items, size=sample_count, replace=False, p=normalized)
    return torch.tensor(choice, device=weights.device, dtype=torch.long)


def build_scene_prior(
    cfg: HybridConfig,
    cameras: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
    # Turn the sparse COLMAP cloud into anchored/detail/completion seeds for
    # scene reconstruction. This is the heart of scene mode.
    if not cfg.colmap_model_dir:
        raise ValueError("Scene mode requires --colmap-model-dir.")

    point_cloud: ColmapPointCloud = load_colmap_points3d(
        model_dir=cfg.colmap_model_dir,
        device=cfg.device,
        max_points=cfg.max_sparse_points,
    )
    normals = estimate_point_normals(point_cloud.xyz, cameras)
    palette = prompt_palette(cfg.prompt, cfg.device)

    anchor_weights = point_cloud.track_length + 1.0 / (1.0 + point_cloud.error)
    # Reliable, well-supported points become anchored structure.
    anchor_indices = _sample_indices_from_weights(point_cloud.xyz.shape[0], cfg.num_splats, anchor_weights)
    anchors = point_cloud.xyz[anchor_indices]
    anchor_normals = normals[anchor_indices]

    detail_weights = (point_cloud.track_length + 1.0) * (1.0 + point_cloud.error.reciprocal().clamp(max=10.0))
    # Detail points prefer regions that are still trustworthy but potentially richer in variation.
    detail_indices = _sample_indices_from_weights(point_cloud.xyz.shape[0], cfg.num_detail_splats, detail_weights)
    detail_anchors = point_cloud.xyz[detail_indices]
    detail_normals = normals[detail_indices]

    support = point_cloud.track_length
    uncertainty = 1.0 / support.clamp_min(1.0)
    camera_centers = torch.stack([-camera.rotation.T @ camera.translation for camera in cameras], dim=0)
    distances = torch.cdist(point_cloud.xyz, camera_centers).amin(dim=1)
    completion_weights = uncertainty * (0.35 + distances / distances.max().clamp_min(1e-8))
    # Under-supported and far-from-camera points are treated as better
    # candidates for gap completion because they are more likely to lie near
    # missing or weakly observed regions.
    completion_indices = _sample_indices_from_weights(
        point_cloud.xyz.shape[0],
        cfg.num_completion_splats,
        completion_weights,
    )
    completion_base = point_cloud.xyz[completion_indices]
    completion_normals = normals[completion_indices]
    colors = point_cloud.rgb.clamp(0.02, 0.98)
    completion_seed_colors = colors[completion_indices]

    if camera_centers.shape[0] > 0 and completion_base.shape[0] > 0:
        view_deltas = completion_base.unsqueeze(1) - camera_centers.unsqueeze(0)
        nearest_camera = torch.argmin(view_deltas.norm(dim=-1), dim=1)
        view_direction = completion_base - camera_centers[nearest_camera]
        view_direction = view_direction / view_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        view_direction = completion_normals

    completion_normals = completion_normals + 0.6 * view_direction
    # Push completion seeds slightly outward so they do not collapse exactly on
    # top of the sparse support points from the start.
    completion_normals = completion_normals / completion_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    completion_seeds = completion_base + 0.04 * completion_normals

    color_points = point_cloud.xyz
    point_colors = point_cloud.rgb
    anchor_colors = point_colors[anchor_indices]
    detail_colors = point_colors[detail_indices]
    if anchor_colors.numel() == 0:
        anchor_colors = build_palette_colors(anchors, anchor_normals, palette)
    if detail_colors.numel() == 0:
        detail_colors = build_palette_colors(detail_anchors, detail_normals, palette)
    if completion_seed_colors.numel() == 0:
        completion_seed_colors = build_palette_colors(completion_seeds, completion_normals, palette)

    scene_colors = torch.cat((anchor_colors, detail_colors, completion_seed_colors), dim=0)
    return (
        anchors,
        anchor_normals,
        detail_anchors,
        detail_normals,
        completion_seeds,
        completion_normals,
        "colmap_sparse_scene",
        scene_colors,
    )


def sample_detail_anchors(
    anchors: torch.Tensor,
    normals: torch.Tensor,
    num_detail_splats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mesh mode reuses anchored samples for the detail branch by subsampling.
    if num_detail_splats <= 0:
        empty_points = anchors.new_zeros((0, anchors.shape[1]))
        empty_normals = normals.new_zeros((0, normals.shape[1]))
        return empty_points, empty_normals

    replace = anchors.shape[0] < num_detail_splats
    indices = torch.randint(anchors.shape[0], (num_detail_splats,), device=anchors.device) if replace else torch.randperm(
        anchors.shape[0], device=anchors.device
    )[:num_detail_splats]
    return anchors[indices], normals[indices]


def _load_resized_rgb(path: str | Path, image_size: int) -> torch.Tensor:
    # Real image supervision is resized into the same square canvas used by the
    # renderer. This keeps the loss simple at the cost of some distortion.
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _load_resized_mask(path: str | Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((image_size, image_size), Resampling.BILINEAR)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def maybe_load_reference_supervision(cfg: HybridConfig) -> tuple[torch.Tensor, torch.Tensor] | None:
    # Optional real-image supervision: use a single image plus a foreground mask
    # to anchor appearance and silhouette for the front-most camera view.
    if not cfg.reference_image_path:
        return None

    rgb = _load_resized_rgb(cfg.reference_image_path, cfg.image_size).to(cfg.device)
    if cfg.reference_mask_path:
        mask = _load_resized_mask(cfg.reference_mask_path, cfg.image_size).to(cfg.device)
    else:
        mask = torch.ones((cfg.image_size, cfg.image_size), device=cfg.device, dtype=torch.float32)
    return rgb, mask


def build_mesh_prior(
    cfg: HybridConfig,
    cameras: list,
) -> tuple[Mesh | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor | None]:
    # Mesh mode and scene mode share the same Gaussian optimizer; they only
    # differ in how seeds are built.
    if cfg.scene_mode and cfg.colmap_model_dir:
        (
            anchors,
            normals,
            detail_anchors,
            detail_normals,
            completion_seeds,
            completion_normals,
            completion_strategy,
            scene_colors,
        ) = build_scene_prior(cfg, cameras)
        return (
            None,
            anchors,
            normals,
            detail_anchors,
            detail_normals,
            completion_seeds,
            completion_normals,
            completion_strategy,
            scene_colors,
        )

    mesh = load_mesh(cfg)
    anchors, normals = sample_surface(mesh, cfg.num_splats)
    detail_anchors, detail_normals = sample_detail_anchors(anchors, normals, cfg.num_detail_splats)
    completion_seeds, completion_normals, completion_strategy = sample_completion_regions(
        mesh,
        cfg.num_completion_splats,
    )
    return (
        mesh,
        anchors,
        normals,
        detail_anchors,
        detail_normals,
        completion_seeds,
        completion_normals,
        completion_strategy,
        None,
    )


def optimize(cfg: HybridConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Real datasets may skip meshes entirely, so use a primitive only as a
    # synthetic fallback for the target-view loader.
    fallback_mesh = load_mesh(cfg) if not (cfg.scene_mode and cfg.colmap_model_dir) else primitive_mesh_from_prompt(cfg.prompt, cfg.device)
    cameras, targets, view_source = load_training_views(cfg, fallback_mesh)
    (
        mesh,
        anchors,
        normals,
        detail_anchors,
        detail_normals,
        completion_seeds,
        completion_normals,
        completion_strategy,
        scene_colors,
    ) = build_mesh_prior(cfg, cameras)
    model = HybridGaussianModel(
        anchors=anchors,
        normals=normals,
        detail_anchors=detail_anchors,
        detail_normals=detail_normals,
        completion_seeds=completion_seeds,
        completion_normals=completion_normals,
        prompt=cfg.prompt,
        anchor_colors=scene_colors[: anchors.shape[0]] if scene_colors is not None else None,
        detail_colors=scene_colors[anchors.shape[0] : anchors.shape[0] + detail_anchors.shape[0]] if scene_colors is not None else None,
        completion_colors=scene_colors[anchors.shape[0] + detail_anchors.shape[0] :] if scene_colors is not None else None,
    ).to(cfg.device)
    reference_supervision = maybe_load_reference_supervision(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    total_splats = cfg.num_splats + detail_anchors.shape[0] + completion_seeds.shape[0]
    print(
        f"Running hybrid baseline with {cfg.num_splats} anchored splats + "
        f"{detail_anchors.shape[0]} detail splats + "
        f"{completion_seeds.shape[0]} completion splats on {cfg.device}."
    )
    print(
        f"Pipeline: {'scene prior' if cfg.scene_mode else 'mesh prior'} -> anchored splats + detail splats + completion splats "
        f"-> refinement -> multi-view render (completion seeds: {completion_strategy}, supervision: {view_source})"
    )
    start_time = time.perf_counter()
    last_log_time = start_time

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()
        # Rebuild the full Gaussian state every step so all three branches
        # train jointly.
        state = model.state()
        anchored_state = model.anchored_state()
        detail_state = model.detail_state()
        completion_state = model.completion_state()

        reconstruction = torch.zeros((), device=cfg.device)
        mask_loss = torch.zeros((), device=cfg.device)
        for camera, target in zip(cameras, targets):
            rendered = render_gaussians(
                state,
                camera,
                tile_size=cfg.render_tile_size,
                support_scale=cfg.render_support_scale,
                alpha_threshold=cfg.render_alpha_threshold,
            )
            reconstruction = reconstruction + reconstruction_loss(rendered, target)
        reconstruction = reconstruction / len(cameras)

        if reference_supervision is not None:
            # Use the first orbit camera as the "observed" view and add both RGB and
            # silhouette supervision from the actual source image.
            reference_rgb, reference_mask = reference_supervision
            reference_render, reference_alpha = render_gaussians(
                state,
                cameras[0],
                tile_size=cfg.render_tile_size,
                support_scale=cfg.render_support_scale,
                alpha_threshold=cfg.render_alpha_threshold,
                return_alpha=True,
            )
            reconstruction = reconstruction + reconstruction_loss(reference_render, reference_rgb)
            mask_loss = torch.mean(torch.abs(reference_alpha - reference_mask))

        tether = tether_loss(
            anchored_state.means,
            model.anchor_positions,
            model.anchor_normals,
        )
        detail = detail_tether_loss(
            detail_state.means,
            model.detail_anchor_positions,
            model.detail_anchor_normals,
        )
        completion = completion_smoothness_loss(
            completion_state.means,
            model.completion_seed_positions,
            model.completion_seed_normals,
        )
        appearance = appearance_guidance_loss(state.colors, model.palette)
        scale_penalty = scale_regularization(state.scales)
        opacity_penalty = opacity_regularization(state.opacity)

        total = (
            reconstruction
            + cfg.lambda_tether * tether
            + cfg.lambda_detail * detail
            + cfg.lambda_completion * completion
            + cfg.lambda_appearance * appearance
            + cfg.lambda_scale * scale_penalty
            + cfg.lambda_opacity * opacity_penalty
            + cfg.lambda_mask * mask_loss
        )
        total.backward()
        optimizer.step()

        if step == 1 or step % max(cfg.steps // 10, 1) == 0 or step == cfg.steps:
            # Print periodic timing blocks so long scene runs are easier to monitor.
            now = time.perf_counter()
            step_time = now - last_log_time
            total_time = now - start_time
            print(
                f"\n\n[{step:03d}/{cfg.steps}]\n"
                "______________________\n"
                f"time taken: {step_time:.2f}s\n"
                f"total time used: {total_time:.2f}s\n"
                "______________________\n"
                f"total={total.item():.4f} "
                f"recon={reconstruction.item():.4f} "
                f"tether={tether.item():.4f} "
                f"detail={detail.item():.4f} "
                f"completion={completion.item():.4f} "
                f"appearance={appearance.item():.4f} "
                f"mask={mask_loss.item():.4f}"
            )
            last_log_time = now

    final_state = model.state()
    save_gaussian_state(cfg.out_dir / "gaussian_state.npz", final_state)
    save_metadata(
        cfg.out_dir / "gaussian_metadata.txt",
        {
            "anchored_splats": int(anchored_state.means.shape[0]),
            "detail_splats": int(detail_state.means.shape[0]),
            "completion_splats": int(completion_state.means.shape[0]),
            "total_splats": int(total_splats),
            "completion_seed_strategy": completion_strategy,
        },
    )
    for index, (camera, target) in enumerate(zip(cameras, targets)):
        rendered = render_gaussians(
            final_state,
            camera,
            tile_size=cfg.render_tile_size,
            support_scale=cfg.render_support_scale,
            alpha_threshold=cfg.render_alpha_threshold,
        )
        save_image(cfg.out_dir / f"view_{index:02d}_render.png", rendered)
        save_image(cfg.out_dir / f"view_{index:02d}_target.png", target)
    maybe_prompt_to_create_viewer(cfg)


def parse_args() -> HybridConfig:
    parser = argparse.ArgumentParser(description="Hybrid mesh + Gaussian splatting baseline.")
    # The CLI keeps both mesh mode and scene mode in one command so experiments
    # can switch priors without changing scripts.
    parser.add_argument("--prompt", default="stone statue", help="Semantic prompt used for mesh choice and appearance prior.")
    parser.add_argument("--mesh", dest="mesh_path", default=None, help="Optional OBJ mesh path exported from Fantasia3D or another generator.")
    parser.add_argument(
        "--colmap-model-dir",
        default=None,
        help="Optional COLMAP TXT model directory containing cameras.txt and images.txt.",
    )
    parser.add_argument(
        "--colmap-image-dir",
        default=None,
        help="Optional image directory that matches the COLMAP image names.",
    )
    parser.add_argument(
        "--scene-mode",
        action="store_true",
        help="Use the COLMAP sparse point cloud as a scene prior for reconstruction and missing-region completion.",
    )
    parser.add_argument("--reference-image", dest="reference_image_path", default=None, help="Optional real RGB image used to supervise the front view.")
    parser.add_argument("--reference-mask", dest="reference_mask_path", default=None, help="Optional mask aligned with --reference-image for silhouette supervision.")
    parser.add_argument("--out-dir", default="outputs/demo", help="Directory for rendered outputs.")
    parser.add_argument("--num-splats", type=int, default=384, help="Number of anchored Gaussian splats.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--num-views", type=int, default=6, help="Number of training/rendering views. When using COLMAP, views are subsampled evenly.")
    parser.add_argument("--image-size", type=int, default=96, help="Square render resolution.")
    parser.add_argument(
        "--colmap-resize-long-edge",
        type=int,
        default=256,
        help="Optional resize cap for COLMAP images. Set to 0 to keep original resolution.",
    )
    parser.add_argument(
        "--render-tile-size",
        type=int,
        default=96,
        help="Tile size for memory-efficient rendering. Set to 0 to disable tiling.",
    )
    parser.add_argument(
        "--render-support-scale",
        type=float,
        default=2.0,
        help="Projected Gaussian support radius in sigma units. Smaller values keep rendering more focused per tile.",
    )
    parser.add_argument(
        "--render-alpha-threshold",
        type=float,
        default=1e-3,
        help="Skip splats whose peak alpha on a tile is below this threshold.",
    )
    parser.add_argument(
        "--prompt-viewer",
        action="store_true",
        help="After training, ask whether to generate viewer.html in the output directory.",
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--lambda-mask", type=float, default=0.20, help="Weight for optional mask supervision from a real image.")
    parser.add_argument("--lambda-detail", type=float, default=0.25, help="Weight for mesh-local detail splat tethering.")
    parser.add_argument("--lambda-completion", type=float, default=0.35, help="Weight for weak completion-splat geometry regularization.")
    parser.add_argument("--num-detail-splats", type=int, default=192, help="Residual detail splats kept close to the mesh surface.")
    parser.add_argument("--num-completion-splats", type=int, default=128, help="Additional weakly constrained splats used to fill missing regions.")
    parser.add_argument("--max-sparse-points", type=int, default=12000, help="Maximum sparse COLMAP points kept when scene mode is enabled.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bool(args.colmap_model_dir) != bool(args.colmap_image_dir):
        raise SystemExit("--colmap-model-dir and --colmap-image-dir must be provided together.")
    if args.scene_mode and not args.colmap_model_dir:
        raise SystemExit("--scene-mode requires --colmap-model-dir and --colmap-image-dir.")

    colmap_resize_long_edge = args.colmap_resize_long_edge if args.colmap_resize_long_edge > 0 else None
    render_tile_size = args.render_tile_size if args.render_tile_size > 0 else None

    return HybridConfig(
        prompt=args.prompt,
        mesh_path=args.mesh_path,
        colmap_model_dir=args.colmap_model_dir,
        colmap_image_dir=args.colmap_image_dir,
        scene_mode=args.scene_mode,
        reference_image_path=args.reference_image_path,
        reference_mask_path=args.reference_mask_path,
        out_dir=Path(args.out_dir),
        num_splats=args.num_splats,
        steps=args.steps,
        num_views=args.num_views,
        image_size=args.image_size,
        colmap_resize_long_edge=colmap_resize_long_edge,
        render_tile_size=render_tile_size,
        render_support_scale=args.render_support_scale,
        render_alpha_threshold=args.render_alpha_threshold,
        prompt_viewer=args.prompt_viewer,
        lr=args.lr,
        seed=args.seed,
        device=device,
        lambda_mask=args.lambda_mask,
        lambda_detail=args.lambda_detail,
        lambda_completion=args.lambda_completion,
        num_detail_splats=args.num_detail_splats,
        num_completion_splats=args.num_completion_splats,
        max_sparse_points=args.max_sparse_points,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    optimize(cfg)
