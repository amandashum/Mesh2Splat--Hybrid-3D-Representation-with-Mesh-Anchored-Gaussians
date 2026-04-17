from __future__ import annotations

from dataclasses import dataclass

import torch

from hybrid_gs.gaussians import GaussianState
from hybrid_gs.mesh import Mesh

from .seeding import _boundary_edge_data


@dataclass
class CompletionMeshArtifacts:
    # The completion branch can now export two explicit mesh artifacts:
    # - a patch mesh made only from selected completion triangles
    # - a merged mesh that appends that patch to the original mesh prior
    patch_mesh: Mesh | None
    merged_mesh: Mesh | None
    selected_completion_count: int
    patched_edge_count: int
    strategy: str


def _empty_artifacts(strategy: str) -> CompletionMeshArtifacts:
    return CompletionMeshArtifacts(
        patch_mesh=None,
        merged_mesh=None,
        selected_completion_count=0,
        patched_edge_count=0,
        strategy=strategy,
    )


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _mesh_face_normals(mesh: Mesh) -> torch.Tensor:
    triangles = mesh.vertices[mesh.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    return _normalize(torch.cross(edges_a, edges_b, dim=-1))


def build_completion_patch_mesh(
    base_mesh: Mesh | None,
    completion_state: GaussianState,
    *,
    strategy: str,
    completion_normals: torch.Tensor | None = None,
    opacity_threshold: float = 0.20,
    max_edge_distance: float = 0.18,
    max_vertical_offset: float = 0.12,
    max_normal_deviation: float = 0.55,
    blend_with_edge_midpoint: float = 0.65,
) -> CompletionMeshArtifacts:
    # Turning splats into a full watertight learned mesh would require a much
    # stronger meshing stage. This helper implements a narrower step: select
    # completion splats that appear to bridge open boundary edges, then create
    # one triangle patch per supported boundary edge and merge those patches
    # into the original mesh.
    if base_mesh is None or completion_state.means.shape[0] == 0:
        return _empty_artifacts(strategy)

    boundary_edges, boundary_face_ids, edge_lengths = _boundary_edge_data(base_mesh)
    if boundary_edges.shape[0] == 0:
        return _empty_artifacts(strategy)

    completion_points = completion_state.means
    completion_opacity = completion_state.opacity.reshape(-1)
    valid_mask = completion_opacity >= opacity_threshold
    if not torch.any(valid_mask):
        return _empty_artifacts(strategy)

    candidate_points = completion_points[valid_mask]
    candidate_normals = completion_normals[valid_mask] if completion_normals is not None else None
    edge_vertices_a = base_mesh.vertices[boundary_edges[:, 0]]
    edge_vertices_b = base_mesh.vertices[boundary_edges[:, 1]]
    edge_midpoints = 0.5 * (edge_vertices_a + edge_vertices_b)
    face_normals = _mesh_face_normals(base_mesh)[boundary_face_ids]

    # Favor side-wall gap filling over large roof or floor closure. In practice
    # this means preferring boundary edges whose supporting faces are closer to
    # vertical walls than to horizontal roof/ground surfaces.
    wall_like_edges = face_normals[:, 1].abs() <= max_normal_deviation
    if torch.any(wall_like_edges):
        boundary_edges = boundary_edges[wall_like_edges]
        boundary_face_ids = boundary_face_ids[wall_like_edges]
        edge_lengths = edge_lengths[wall_like_edges]
        edge_vertices_a = edge_vertices_a[wall_like_edges]
        edge_vertices_b = edge_vertices_b[wall_like_edges]
        edge_midpoints = edge_midpoints[wall_like_edges]
        face_normals = face_normals[wall_like_edges]
    if boundary_edges.shape[0] == 0:
        return _empty_artifacts(strategy)

    distances = torch.cdist(edge_midpoints, candidate_points)
    nearest_distance, nearest_candidate = distances.min(dim=1)
    max_distances = edge_lengths * 1.5 + max_edge_distance
    supported_edges = nearest_distance <= max_distances
    chosen_points = candidate_points[nearest_candidate]

    # Reject completion points that jump too far vertically away from the hole
    # they are supposed to patch. This removes many of the long roof-to-ground
    # spikes seen in the earlier naive fan triangulation.
    vertical_offsets = (chosen_points[:, 1] - edge_midpoints[:, 1]).abs()
    supported_edges = supported_edges & (vertical_offsets <= (edge_lengths * 0.75 + max_vertical_offset))

    if candidate_normals is not None:
        chosen_normals = _normalize(candidate_normals[nearest_candidate])
        normal_alignment = (chosen_normals * face_normals).sum(dim=-1).abs()
        supported_edges = supported_edges & (normal_alignment >= 0.25)

    if not torch.any(supported_edges):
        return _empty_artifacts(strategy)

    chosen_edges = boundary_edges[supported_edges]
    chosen_candidate_indices = nearest_candidate[supported_edges]
    chosen_midpoints = edge_midpoints[supported_edges]
    chosen_points = candidate_points[chosen_candidate_indices]
    chosen_face_normals = face_normals[supported_edges]

    # Pull each learned completion point back toward the boundary midpoint and
    # clamp its excursion along the supporting face normal. This produces a
    # smoother local patch instead of letting one distant splat create a long
    # skinny spike.
    offset_vectors = chosen_points - chosen_midpoints
    normal_offsets = (offset_vectors * chosen_face_normals).sum(dim=-1, keepdim=True)
    clamped_normal_offsets = normal_offsets.clamp(min=-0.10, max=0.10)
    tangent_offsets = offset_vectors - normal_offsets * chosen_face_normals
    tangent_offsets = tangent_offsets.clamp(min=-0.08, max=0.08)
    smoothed_points = chosen_midpoints + (1.0 - blend_with_edge_midpoint) * (
        clamped_normal_offsets * chosen_face_normals + tangent_offsets
    )

    patch_vertices = smoothed_points
    patch_faces = []
    for edge_index, edge in enumerate(chosen_edges):
        new_vertex_index = base_mesh.vertices.shape[0] + edge_index
        patch_faces.append([int(edge[0].item()), int(edge[1].item()), new_vertex_index])

    if not patch_faces:
        return _empty_artifacts(strategy)

    merged_vertices = torch.cat((base_mesh.vertices, patch_vertices), dim=0)
    merged_faces = torch.cat(
        (
            base_mesh.faces,
            torch.tensor(patch_faces, dtype=torch.long, device=base_mesh.vertices.device),
        ),
        dim=0,
    )
    patch_local_faces = torch.tensor(
        [[int(edge[0].item()), int(edge[1].item()), base_mesh.vertices.shape[0] + index] for index, edge in enumerate(chosen_edges)],
        dtype=torch.long,
        device=base_mesh.vertices.device,
    )
    # The patch mesh itself is stored in the merged/global index space so OBJ
    # export can be shared with the same helper used for ordinary meshes.
    patch_mesh = Mesh(vertices=merged_vertices, faces=patch_local_faces)
    merged_mesh = Mesh(vertices=merged_vertices, faces=merged_faces)
    return CompletionMeshArtifacts(
        patch_mesh=patch_mesh,
        merged_mesh=merged_mesh,
        selected_completion_count=int(chosen_points.shape[0]),
        patched_edge_count=int(chosen_edges.shape[0]),
        strategy=strategy,
    )
