from __future__ import annotations

from dataclasses import dataclass

import torch

from hybrid_gs.colmap import ColmapPointCloud
from hybrid_gs.mesh import Mesh, sample_surface


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Completion logic constantly mixes normals, view directions, and bridge
    # directions. Keeping normalization in one helper makes those blends less
    # error-prone and keeps zero-length edge cases from exploding.
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class CompletionPrior:
    # The completion branch should know both where it starts and how confident
    # that initialization is. `strengths` is used to weight the continuity loss
    # so strong hole-boundary seeds matter more than generic fallback samples.
    seeds: torch.Tensor
    normals: torch.Tensor
    strengths: torch.Tensor
    strategy: str


def _mesh_face_normals(mesh: Mesh) -> torch.Tensor:
    # Completion seeds near a hole should inherit the local supporting face
    # orientation, so compute one stable normal per triangle once here.
    triangles = mesh.vertices[mesh.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    return _normalize(torch.cross(edges_a, edges_b, dim=-1))


def _boundary_edge_data(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Boundary edges are the clearest signal that the mesh is missing surface.
    # The output includes:
    # - the unique open edges themselves
    # - one supporting face id per open edge
    # - the edge lengths, which become sampling weights so large holes attract
    #   more completion splats than tiny cracks
    directed_edges = torch.cat(
        (
            mesh.faces[:, [0, 1]],
            mesh.faces[:, [1, 2]],
            mesh.faces[:, [2, 0]],
        ),
        dim=0,
    )
    edge_face_ids = torch.arange(mesh.faces.shape[0], device=mesh.vertices.device).repeat(3)
    undirected_edges = torch.sort(directed_edges, dim=1).values
    unique_edges, inverse_indices, counts = torch.unique(
        undirected_edges,
        dim=0,
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    boundary_mask = counts == 1
    if not torch.any(boundary_mask):
        empty_long = torch.zeros((0, 2), device=mesh.vertices.device, dtype=torch.long)
        empty_face_ids = torch.zeros((0,), device=mesh.vertices.device, dtype=torch.long)
        empty_lengths = torch.zeros((0,), device=mesh.vertices.device, dtype=torch.float32)
        return empty_long, empty_face_ids, empty_lengths

    boundary_unique_indices = torch.nonzero(boundary_mask, as_tuple=False).squeeze(1)
    boundary_edges = unique_edges[boundary_unique_indices]
    boundary_face_ids = torch.zeros_like(boundary_unique_indices)
    for index, unique_index in enumerate(boundary_unique_indices):
        face_slot = torch.nonzero(inverse_indices == unique_index, as_tuple=False)[0, 0]
        boundary_face_ids[index] = edge_face_ids[face_slot]

    edge_lengths = (mesh.vertices[boundary_edges[:, 1]] - mesh.vertices[boundary_edges[:, 0]]).norm(dim=-1)
    return boundary_edges, boundary_face_ids, edge_lengths


def build_mesh_completion_prior(
    mesh: Mesh,
    num_samples: int,
    bridge_offset: float = 0.035,
) -> CompletionPrior:
    # Completion splats should learn from actual gaps in the mesh. When the
    # mesh has open boundaries, seed splats just inside plausible surface
    # continuation directions instead of sampling the whole object uniformly.
    if num_samples <= 0:
        # Keeping the disabled case explicit lets the rest of the optimizer run
        # unchanged when the user sets `--num-completion-splats 0`.
        empty = mesh.vertices.new_zeros((0, 3))
        empty_strengths = mesh.vertices.new_zeros((0,))
        return CompletionPrior(empty, empty, empty_strengths, "disabled")

    boundary_edges, boundary_face_ids, edge_lengths = _boundary_edge_data(mesh)
    if boundary_edges.shape[0] == 0:
        # Closed meshes do not expose obvious hole boundaries, so fall back to
        # surface samples. The continuity loss will then behave more like a
        # mild surface-refinement prior than true hole bridging.
        fallback_points, fallback_normals = sample_surface(mesh, num_samples)
        fallback_strengths = torch.full((num_samples,), 0.35, device=mesh.vertices.device, dtype=torch.float32)
        return CompletionPrior(fallback_points, fallback_normals, fallback_strengths, "surface_fallback")

    face_normals = _mesh_face_normals(mesh)
    probabilities = edge_lengths / edge_lengths.sum().clamp_min(1e-8)
    sampled_edges = torch.multinomial(probabilities, num_samples, replacement=True)
    edges = boundary_edges[sampled_edges]
    face_ids = boundary_face_ids[sampled_edges]

    vertices_a = mesh.vertices[edges[:, 0]]
    vertices_b = mesh.vertices[edges[:, 1]]
    edge_points = torch.lerp(vertices_a, vertices_b, torch.rand((num_samples, 1), device=mesh.vertices.device))
    edge_tangents = _normalize(vertices_b - vertices_a)
    supporting_normals = face_normals[face_ids]

    # Approximate the missing-side bridge direction as the in-plane vector
    # orthogonal to the boundary edge. The sign is chosen to point away from
    # the mesh center so seeds start outside the known surface.
    bridge_directions = _normalize(torch.cross(edge_tangents, supporting_normals, dim=-1))
    center = mesh.vertices.mean(dim=0, keepdim=True)
    sign = torch.sign(((edge_points - center) * bridge_directions).sum(dim=-1, keepdim=True)).clamp_min(0.0) * 2.0 - 1.0
    bridge_directions = bridge_directions * sign

    seeds = edge_points + bridge_offset * bridge_directions
    normals = _normalize(0.7 * supporting_normals + 0.3 * bridge_directions)
    strengths = edge_lengths[sampled_edges]
    strengths = strengths / strengths.amax().clamp_min(1e-8)
    # `mesh_boundary_bridge` is the key mode you want to see in logs when
    # testing hole filling from a real mesh prior.
    return CompletionPrior(seeds, normals, strengths, "mesh_boundary_bridge")


def build_sparse_completion_prior(
    point_cloud: ColmapPointCloud,
    normals: torch.Tensor,
    cameras: list,
    num_samples: int,
) -> CompletionPrior:
    # Scene mode has no clean mesh boundaries, so use sparse-view confidence.
    # The weakest, most distant sparse points are treated as likely frontier
    # regions where completion splats should explore missing structure.
    if num_samples <= 0:
        # Scene-mode completion can also be disabled without needing a
        # different training path.
        empty = point_cloud.xyz.new_zeros((0, 3))
        empty_strengths = point_cloud.xyz.new_zeros((0,))
        return CompletionPrior(empty, empty, empty_strengths, "disabled")

    support = point_cloud.track_length
    uncertainty = 1.0 / support.clamp_min(1.0)
    if cameras:
        camera_centers = torch.stack([-camera.rotation.T @ camera.translation for camera in cameras], dim=0)
        distances = torch.cdist(point_cloud.xyz, camera_centers).amin(dim=1)
        distance_term = distances / distances.amax().clamp_min(1e-8)
    else:
        camera_centers = point_cloud.xyz.new_zeros((0, 3))
        distance_term = torch.ones_like(uncertainty)

    completion_weights = uncertainty * (0.35 + distance_term)
    # The sparse cloud itself is still the source of truth here; completion
    # seeds are only nudged outward from weakly supported frontier points.
    completion_weights = completion_weights / completion_weights.sum().clamp_min(1e-8)
    chosen = torch.multinomial(completion_weights, num_samples, replacement=point_cloud.xyz.shape[0] < num_samples)
    base = point_cloud.xyz[chosen]
    chosen_normals = normals[chosen]
    strengths = completion_weights[chosen]

    if camera_centers.shape[0] > 0:
        view_deltas = base.unsqueeze(1) - camera_centers.unsqueeze(0)
        nearest_camera = torch.argmin(view_deltas.norm(dim=-1), dim=1)
        view_direction = base - camera_centers[nearest_camera]
        view_direction = _normalize(view_direction)
    else:
        view_direction = chosen_normals

    completion_normals = _normalize(chosen_normals + 0.6 * view_direction)
    seeds = base + 0.04 * completion_normals
    strengths = strengths / strengths.amax().clamp_min(1e-8)
    # This strategy name is written into metadata so comparisons can tell
    # whether the completion branch came from a mesh-hole prior or a sparse
    # COLMAP frontier prior.
    return CompletionPrior(seeds, completion_normals, strengths, "colmap_sparse_frontier")
