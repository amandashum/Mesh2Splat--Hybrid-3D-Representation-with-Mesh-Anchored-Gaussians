from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from hybrid_gs.camera import orbit_cameras
from hybrid_gs.colmap import ColmapPointCloud, load_colmap_points3d, load_colmap_text_dataset
from hybrid_gs.completion import (
    CompletionPrior,
    CompletionMeshArtifacts,
    build_completion_patch_mesh,
    build_mesh_completion_prior,
    build_sparse_completion_prior,
    completion_continuity_loss,
)
from hybrid_gs.gaussians import GaussianState, HybridGaussianModel, concat_states, prompt_palette
from interactive_splat_viewer import (
    load_metadata as viewer_load_metadata,
    load_state_from_npz,
    maybe_subsample,
    save_rendered_viewer_html,
)
from hybrid_gs.losses import (
    appearance_guidance_loss,
    completion_region_loss,
    completion_smoothness_loss,
    detail_tether_loss,
    opacity_regularization,
    reconstruction_loss,
    scale_regularization,
    tether_loss,
)
from hybrid_gs.mesh import Mesh, load_obj_mesh, primitive_mesh_from_prompt, sample_surface
from hybrid_gs.renderer import render_gaussians
from hybrid_gs.segmentation import build_scene_structure_masks


@dataclass
class HybridConfig:
    prompt: str
    mesh_path: str | None
    colmap_bat: str | None
    mesh_workspace: str | None
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
    lambda_completion_continuity: float = 0.20
    lambda_completion_region: float = 0.35
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


def save_mask_image(path: Path, mask: torch.Tensor) -> None:
    # Segmentation/debug masks are saved as grayscale images so completion
    # gating can be inspected directly alongside the rendered outputs.
    array = (mask.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


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


def save_obj_mesh(path: Path, mesh: Mesh) -> None:
    # Save the normalized mesh prior into the output folder so the exact
    # geometry used by training can be opened later without hunting for the
    # original source OBJ. This is intentionally the normalized training mesh,
    # not a verbatim file copy, so the geometry matches the optimizer's scale.
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def save_gaussian_point_cloud(path: Path, state: GaussianState) -> None:
    # MeshLab can open colored PLY point clouds directly, so export the
    # Gaussian centers as a geometry comparison artifact before and after
    # completion. These are still splat centers, not a surfaced mesh, but they
    # are useful when you want to inspect how completion changed the spatial
    # distribution without waiting on the HTML viewer.
    means = state.means.detach().cpu().numpy()
    colors = (state.colors.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    opacity = (state.opacity.detach().cpu().clamp(0.0, 1.0).numpy().reshape(-1) * 255.0).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {means.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property uchar alpha\n")
        handle.write("end_header\n")
        for point, color, alpha in zip(means, colors, opacity):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(alpha)}\n"
            )


def build_completion_mesh_exports(
    mesh: Mesh | None,
    completion_state: GaussianState,
    completion_normals: torch.Tensor,
    strategy: str,
) -> CompletionMeshArtifacts:
    # The completion branch should affect exported geometry, not just rendered
    # images. This helper converts the learned completion splats into a simple
    # patch mesh and merges that patch back into the base mesh when possible.
    return build_completion_patch_mesh(
        base_mesh=mesh,
        completion_state=completion_state,
        strategy=strategy,
        completion_normals=completion_normals,
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


def print_phase_banner(title: str) -> None:
    # Large scene runs are easier to follow when the log clearly marks the
    # structural prior stage and the later completion-learning stage.
    text = title.strip()
    border = "+" * max(len(text) + 12, 23)
    print(f"\n\n{border}\n+     {text}     +\n{border}")


def print_phase_detail(*lines: str) -> None:
    # Follow each export banner with concrete file targets so the user can see
    # exactly what is being produced and where it will appear in the output
    # folder.
    for line in lines:
        print(line)


def maybe_prepare_mesh_prior(cfg: HybridConfig) -> HybridConfig:
    # When the user forgets to supply a mesh prior but does have a COLMAP
    # scene, offer to generate one before training starts. This keeps the run
    # on the mesh-comparison path without forcing a separate manual step.
    #
    # This is the startup flow documented in the README:
    # - print a mesh-check banner
    # - ask whether a mesh prior should be generated
    # - call the COLMAP dense-meshing helper if the user agrees
    # - continue the same run in mesh mode once mesh_prior.obj exists
    if cfg.mesh_path:
        return cfg
    if not cfg.colmap_model_dir or not cfg.colmap_image_dir:
        return cfg

    # Reuse an already-generated mesh prior automatically before prompting the
    # user again. This covers the common case where mesh_prior.obj was created
    # manually in an earlier step and the next run omits --mesh.
    model_dir = Path(cfg.colmap_model_dir)
    default_workspace = model_dir.parent / "mesh_prior"
    candidate_workspaces = []
    if cfg.mesh_workspace:
        candidate_workspaces.append(Path(cfg.mesh_workspace))
    candidate_workspaces.append(default_workspace)
    if model_dir.parent.parent != model_dir.parent:
        # Some datasets unpack as dataset/dataset/{images,sparse} while the mesh
        # prior is written to dataset/mesh_prior. Check that common layout too.
        candidate_workspaces.append(model_dir.parent.parent / "mesh_prior")

    for workspace in candidate_workspaces:
        existing_mesh = workspace / "dense" / "mesh_prior.obj"
        if existing_mesh.exists():
            cfg.mesh_path = str(existing_mesh)
            cfg.scene_mode = False
            print_phase_banner("Check for Mesh Prior to Compare")
            print_phase_detail(f"Found existing mesh prior at path: {existing_mesh}")
            return cfg

    workspace = Path(cfg.mesh_workspace) if cfg.mesh_workspace else default_workspace

    print_phase_banner("Check for Mesh Prior to Compare")
    print_phase_detail("Checking whether there is mesh prior...")

    try:
        response = input("\nNo mesh found, would you like to generate one? y/n ").strip().lower()
    except EOFError:
        return cfg

    if response not in {"y", "yes"}:
        return cfg
    if not cfg.colmap_bat:
        print(">>> cannot generate mesh prior automatically without --colmap-bat <<<")
        return cfg

    # Keep auto-generated dense meshes close to the COLMAP model by default so
    # each dataset keeps its sparse inputs and derived mesh workspace together.
    script_path = Path(__file__).resolve().parent.parent / "tools" / "generate_colmap_mesh.ps1"
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-ColmapBat",
        cfg.colmap_bat,
        "-ImageDir",
        cfg.colmap_image_dir,
        "-ModelDir",
        cfg.colmap_model_dir,
        "-Workspace",
        str(workspace),
        "-Mesher",
        "poisson",
    ]
    subprocess.run(command, check=True)
    generated_mesh = workspace / "dense" / "mesh_prior.obj"
    if not generated_mesh.exists():
        raise FileNotFoundError(f"Expected generated mesh at {generated_mesh}")

    cfg.mesh_path = str(generated_mesh)
    cfg.scene_mode = False
    print(f"Generated mesh prior at {generated_mesh}")
    return cfg


def maybe_prompt_to_create_viewer(cfg: HybridConfig) -> None:
    # Optional convenience hook so a training run can immediately produce
    # `viewer.html` without making the user remember a second command.
    if not cfg.prompt_viewer:
        return

    try:
        response = input("\n\nCreate interactive viewer HTML now? [y/N] ").strip().lower()
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
    save_rendered_viewer_html(
        state=state,
        metadata=metadata,
        output_path=output_html,
        title=f"Interactive Viewer: {cfg.out_dir.name}",
        mesh_path=cfg.mesh_path,
        max_splats=3000,
        num_frames=40,
        width=768,
        height=768,
        fps=10.0,
        fov_degrees=45.0,
        elevation_degrees=18.0,
        radius_scale=2.3,
        supersample=1.5,
        crop_padding=0.0,
        tile_size=cfg.render_tile_size or 96,
        support_scale=cfg.render_support_scale,
        alpha_threshold=cfg.render_alpha_threshold,
        device_name="auto",
    )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CompletionPrior, torch.Tensor]:
    # Turn the sparse COLMAP cloud into anchored/detail/completion seeds for
    # scene reconstruction. This is the heart of scene mode:
    # anchored/detail splats come from reliable sparse points, while the new
    # completion prior targets weakly observed frontier regions.
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

    completion_prior = build_sparse_completion_prior(
        point_cloud=point_cloud,
        normals=normals,
        cameras=cameras,
        num_samples=cfg.num_completion_splats,
    )

    color_points = point_cloud.xyz
    point_colors = point_cloud.rgb
    anchor_colors = point_colors[anchor_indices]
    detail_colors = point_colors[detail_indices]
    completion_seed_colors = point_colors.new_zeros((completion_prior.seeds.shape[0], 3))
    if completion_prior.seeds.shape[0] > 0:
        completion_color_indices = torch.topk(
            torch.cdist(completion_prior.seeds, color_points),
            k=1,
            largest=False,
        ).indices.squeeze(1)
        completion_seed_colors = point_colors[completion_color_indices]
    if anchor_colors.numel() == 0:
        anchor_colors = build_palette_colors(anchors, anchor_normals, palette)
    if detail_colors.numel() == 0:
        detail_colors = build_palette_colors(detail_anchors, detail_normals, palette)
    if completion_seed_colors.numel() == 0:
        completion_seed_colors = build_palette_colors(completion_prior.seeds, completion_prior.normals, palette)

    scene_colors = torch.cat((anchor_colors, detail_colors, completion_seed_colors), dim=0)
    return (
        anchors,
        anchor_normals,
        detail_anchors,
        detail_normals,
        completion_prior,
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
) -> tuple[Mesh | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CompletionPrior, torch.Tensor | None]:
    # Mesh mode and scene mode share the same Gaussian optimizer; they only
    # differ in how seeds are built. Returning `CompletionPrior` instead of raw
    # seed tensors keeps the strategy name and confidence weights attached to
    # the completion branch all the way through optimization and export.
    if cfg.scene_mode and cfg.colmap_model_dir:
        (
            anchors,
            normals,
            detail_anchors,
            detail_normals,
            completion_prior,
            scene_colors,
        ) = build_scene_prior(cfg, cameras)
        return (
            None,
            anchors,
            normals,
            detail_anchors,
            detail_normals,
            completion_prior,
            scene_colors,
        )

    mesh = load_mesh(cfg)
    anchors, normals = sample_surface(mesh, cfg.num_splats)
    detail_anchors, detail_normals = sample_detail_anchors(anchors, normals, cfg.num_detail_splats)
    completion_prior = build_mesh_completion_prior(mesh, cfg.num_completion_splats)
    return (
        mesh,
        anchors,
        normals,
        detail_anchors,
        detail_normals,
        completion_prior,
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
        completion_prior,
        scene_colors,
    ) = build_mesh_prior(cfg, cameras)
    model = HybridGaussianModel(
        anchors=anchors,
        normals=normals,
        detail_anchors=detail_anchors,
        detail_normals=detail_normals,
        completion_seeds=completion_prior.seeds,
        completion_normals=completion_prior.normals,
        prompt=cfg.prompt,
        anchor_colors=scene_colors[: anchors.shape[0]] if scene_colors is not None else None,
        detail_colors=scene_colors[anchors.shape[0] : anchors.shape[0] + detail_anchors.shape[0]] if scene_colors is not None else None,
        completion_colors=scene_colors[anchors.shape[0] + detail_anchors.shape[0] :] if scene_colors is not None else None,
    ).to(cfg.device)
    reference_supervision = maybe_load_reference_supervision(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    total_splats = cfg.num_splats + detail_anchors.shape[0] + completion_prior.seeds.shape[0]
    print(
        f"Running hybrid baseline with {cfg.num_splats} anchored splats + "
        f"{detail_anchors.shape[0]} detail splats + "
        f"{completion_prior.seeds.shape[0]} completion splats on {cfg.device}."
    )
    print(
        f"Pipeline: {'scene prior' if cfg.scene_mode else 'mesh prior'} -> anchored splats + detail splats + completion splats "
        f"-> refinement -> multi-view render (completion seeds: {completion_prior.strategy}, supervision: {view_source})"
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
        completion_region_penalty = torch.zeros((), device=cfg.device)
        mask_loss = torch.zeros((), device=cfg.device)
        for camera, target in zip(cameras, targets):
            # Render the known-surface branches separately so the scene-general
            # structure masks can reason about "what the current geometry
            # already explains" versus "what only the completion branch is
            # trying to add".
            mesh_prior_render, mesh_prior_alpha = render_gaussians(
                concat_states(anchored_state, detail_state),
                camera,
                tile_size=cfg.render_tile_size,
                support_scale=cfg.render_support_scale,
                alpha_threshold=cfg.render_alpha_threshold,
                return_alpha=True,
            )
            completion_render, completion_alpha = render_gaussians(
                completion_state,
                camera,
                tile_size=cfg.render_tile_size,
                support_scale=cfg.render_support_scale,
                alpha_threshold=cfg.render_alpha_threshold,
                return_alpha=True,
            )
            rendered = render_gaussians(
                state,
                camera,
                tile_size=cfg.render_tile_size,
                support_scale=cfg.render_support_scale,
                alpha_threshold=cfg.render_alpha_threshold,
            )
            reconstruction = reconstruction + reconstruction_loss(rendered, target)
            # The scene-structure masks steer completion toward plausible
            # continuation zones without assuming the scene is specifically a
            # building. This is the current semantic/structural gating layer.
            scene_masks = build_scene_structure_masks(target, mesh_prior_render, mesh_prior_alpha)
            completion_region_penalty = completion_region_penalty + completion_region_loss(
                completion_alpha,
                scene_masks["completion_allowed"],
                scene_masks["completion_focus"],
            )
        reconstruction = reconstruction / len(cameras)
        completion_region_penalty = completion_region_penalty / len(cameras)

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
        completion_continuity = completion_continuity_loss(
            completion_state.means,
            model.completion_seed_positions,
            model.completion_seed_normals,
            completion_prior.strengths,
        )
        appearance = appearance_guidance_loss(state.colors, model.palette)
        scale_penalty = scale_regularization(state.scales)
        opacity_penalty = opacity_regularization(state.opacity)

        total = (
            reconstruction
            + cfg.lambda_tether * tether
            + cfg.lambda_detail * detail
            + cfg.lambda_completion * completion
            + cfg.lambda_completion_continuity * completion_continuity
            + cfg.lambda_completion_region * completion_region_penalty
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
                f"continuity={completion_continuity.item():.4f} "
                f"region={completion_region_penalty.item():.4f} "
                f"appearance={appearance.item():.4f} "
                f"mask={mask_loss.item():.4f}"
            )
            last_log_time = now

    final_state = model.state()
    mesh_prior_state = concat_states(model.anchored_state(), model.detail_state())
    # Export both stages explicitly so comparisons do not rely on recomputing
    # states later: mesh-prior-only and the final completion-enhanced result.
    print_phase_banner("Mesh Prior")
    print_phase_detail(
        f"writing mesh prior geometry to {cfg.out_dir / 'mesh_prior.obj'}" if mesh is not None else "no explicit mesh prior OBJ to write in scene mode",
        f"writing mesh-prior point cloud to {cfg.out_dir / 'mesh_prior_cloud.ply'}",
        f"writing mesh-prior state to {cfg.out_dir / 'mesh_prior_state.npz'}",
        "writing mesh-prior renders to view_XX_mesh_prior.png",
    )
    if mesh is not None:
        save_obj_mesh(cfg.out_dir / "mesh_prior.obj", mesh)
    save_gaussian_point_cloud(cfg.out_dir / "mesh_prior_cloud.ply", mesh_prior_state)
    save_gaussian_state(cfg.out_dir / "mesh_prior_state.npz", mesh_prior_state)
    save_metadata(
        cfg.out_dir / "gaussian_metadata.txt",
        {
            "anchored_splats": int(anchored_state.means.shape[0]),
            "detail_splats": int(detail_state.means.shape[0]),
            "completion_splats": int(completion_state.means.shape[0]),
            "total_splats": int(total_splats),
            "completion_seed_strategy": completion_prior.strategy,
        },
    )
    for index, (camera, target) in enumerate(zip(cameras, targets)):
        mesh_prior_render, mesh_prior_alpha = render_gaussians(
            mesh_prior_state,
            camera,
            tile_size=cfg.render_tile_size,
            support_scale=cfg.render_support_scale,
            alpha_threshold=cfg.render_alpha_threshold,
            return_alpha=True,
        )
        scene_masks = build_scene_structure_masks(target, mesh_prior_render, mesh_prior_alpha)
        rendered = render_gaussians(
            final_state,
            camera,
            tile_size=cfg.render_tile_size,
            support_scale=cfg.render_support_scale,
            alpha_threshold=cfg.render_alpha_threshold,
        )
        save_image(cfg.out_dir / f"view_{index:02d}_mesh_prior.png", mesh_prior_render)
        save_image(cfg.out_dir / f"view_{index:02d}_target.png", target)
        # Export the masks per view so the user can inspect whether completion
        # is being constrained for the right reason: near-surface continuation,
        # occlusion, or obvious background suppression.
        save_mask_image(cfg.out_dir / f"view_{index:02d}_completion_region_mask.png", scene_masks["completion_allowed"])
        save_mask_image(cfg.out_dir / f"view_{index:02d}_surface_core_mask.png", scene_masks["surface_core"])
        save_mask_image(cfg.out_dir / f"view_{index:02d}_occluder_mask.png", scene_masks["occluder"])

    print_phase_banner("Completion Using Splats")
    completion_mesh_artifacts = build_completion_mesh_exports(
        mesh=mesh,
        completion_state=completion_state,
        completion_normals=model.completion_seed_normals,
        strategy=completion_prior.strategy,
    )
    print_phase_detail(
        f"writing completion point cloud to {cfg.out_dir / 'with_completion_cloud.ply'}",
        (
            f"writing completion patch mesh to {cfg.out_dir / 'completion_patch.obj'} "
            f"and {cfg.out_dir / 'missing_mesh_parts.obj'} "
            f"({completion_mesh_artifacts.patched_edge_count} patched edges)"
        )
        if completion_mesh_artifacts.patch_mesh is not None
        else ">>> no completion patch mesh was produced <<<",
        (
            f"writing merged mesh to {cfg.out_dir / 'mesh_with_completion.obj'} "
            f"and {cfg.out_dir / 'merged_mesh_with_splats.obj'} "
            f"({completion_mesh_artifacts.selected_completion_count} completion points used)"
        )
        if completion_mesh_artifacts.merged_mesh is not None
        else ">>> no merged completion mesh was produced <<<",
        f"writing completion state to {cfg.out_dir / 'completion_state.npz'}",
        f"writing final compatible state to {cfg.out_dir / 'gaussian_state.npz'}",
        "writing completion renders to view_XX_with_completion.png and view_XX_render.png",
    )
    save_gaussian_point_cloud(cfg.out_dir / "with_completion_cloud.ply", final_state)
    if completion_mesh_artifacts.patch_mesh is not None:
        save_obj_mesh(cfg.out_dir / "completion_patch.obj", completion_mesh_artifacts.patch_mesh)
        save_obj_mesh(cfg.out_dir / "missing_mesh_parts.obj", completion_mesh_artifacts.patch_mesh)
    if completion_mesh_artifacts.merged_mesh is not None:
        save_obj_mesh(cfg.out_dir / "mesh_with_completion.obj", completion_mesh_artifacts.merged_mesh)
        save_obj_mesh(cfg.out_dir / "merged_mesh_with_splats.obj", completion_mesh_artifacts.merged_mesh)
    save_gaussian_state(cfg.out_dir / "gaussian_state.npz", final_state)
    save_gaussian_state(cfg.out_dir / "completion_state.npz", final_state)
    for index, (camera, _) in enumerate(zip(cameras, targets)):
        rendered = render_gaussians(
            final_state,
            camera,
            tile_size=cfg.render_tile_size,
            support_scale=cfg.render_support_scale,
            alpha_threshold=cfg.render_alpha_threshold,
        )
        save_image(cfg.out_dir / f"view_{index:02d}_with_completion.png", rendered)
        save_image(cfg.out_dir / f"view_{index:02d}_render.png", rendered)
    maybe_prompt_to_create_viewer(cfg)


def parse_args() -> HybridConfig:
    parser = argparse.ArgumentParser(description="Hybrid mesh + Gaussian splatting baseline.")
    # The CLI keeps both mesh mode and scene mode in one command so experiments
    # can switch priors without changing scripts.
    parser.add_argument("--prompt", default="stone statue", help="Semantic prompt used for mesh choice and appearance prior.")
    parser.add_argument("--mesh", dest="mesh_path", default=None, help="Optional OBJ mesh path exported from Fantasia3D or another generator.")
    # These options support the auto mesh-prior path: start from COLMAP
    # cameras/images, detect that no mesh was supplied, optionally generate one,
    # then continue the same run in mesh mode.
    parser.add_argument(
        "--colmap-bat",
        default=None,
        help="Optional COLMAP.bat path used when auto-generating a missing mesh prior before the run.",
    )
    parser.add_argument(
        "--mesh-workspace",
        default=None,
        help="Optional workspace directory used when auto-generating a missing mesh prior.",
    )
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
    parser.add_argument(
        "--lambda-completion-continuity",
        type=float,
        default=0.20,
        help="Weight for the completion continuity loss that keeps hole-filling splats coherent.",
    )
    parser.add_argument(
        "--lambda-completion-region",
        type=float,
        default=0.35,
        help="Weight for the structure-aware completion region loss that keeps completion near plausible building gaps.",
    )
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
        colmap_bat=args.colmap_bat,
        mesh_workspace=args.mesh_workspace,
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
        lambda_completion_continuity=args.lambda_completion_continuity,
        lambda_completion_region=args.lambda_completion_region,
        num_detail_splats=args.num_detail_splats,
        num_completion_splats=args.num_completion_splats,
        max_sparse_points=args.max_sparse_points,
    )


def main() -> None:
    # Resolve any optional mesh generation before the random seeds, priors, and
    # optimizer state are created so the rest of the pipeline can treat a
    # manual mesh path and an auto-generated mesh path identically.
    cfg = parse_args()
    cfg = maybe_prepare_mesh_prior(cfg)
    set_seed(cfg.seed)
    optimize(cfg)
