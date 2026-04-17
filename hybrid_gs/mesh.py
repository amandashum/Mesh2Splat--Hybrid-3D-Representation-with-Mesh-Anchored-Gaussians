from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class Mesh:
    vertices: torch.Tensor
    faces: torch.Tensor

    def normalized(self) -> "Mesh":
        # Keep different meshes in a consistent scale range for renderer and
        # optimization stability. Without this, learning rates and splat sizes
        # would need retuning per mesh.
        centered = self.vertices - self.vertices.mean(dim=0, keepdim=True)
        scale = centered.norm(dim=-1).amax().clamp_min(1e-6)
        return Mesh(centered / scale, self.faces)


def load_obj_mesh(path: str | Path, device: torch.device) -> Mesh:
    # Minimal OBJ loader: enough for triangle meshes exported by common
    # reconstruction tools. The code intentionally ignores materials and other
    # rich OBJ features because the hybrid baseline only needs geometry.
    vertices = []
    faces = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                items = line.split()[1:]
                indices = [int(item.split("/")[0]) - 1 for item in items]
                for face_index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[face_index], indices[face_index + 1]])

    if not vertices or not faces:
        raise ValueError(f"OBJ file at {path} did not contain vertices and triangle faces.")

    mesh = Mesh(
        vertices=torch.tensor(vertices, dtype=torch.float32, device=device),
        faces=torch.tensor(faces, dtype=torch.long, device=device),
    )
    return mesh.normalized()


def create_cube_mesh(device: torch.device) -> Mesh:
    # Primitive fallback used when no real mesh is available.
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
            [1, 5, 6],
            [1, 6, 2],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=torch.long,
        device=device,
    )
    return Mesh(vertices=vertices, faces=faces).normalized()


def create_uv_sphere_mesh(device: torch.device, lat_steps: int = 16, lon_steps: int = 24) -> Mesh:
    # Another primitive fallback, useful for toy experiments.
    vertices: list[list[float]] = []
    for lat in range(lat_steps + 1):
        theta = math.pi * lat / lat_steps
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for lon in range(lon_steps):
            phi = 2.0 * math.pi * lon / lon_steps
            vertices.append(
                [
                    sin_theta * math.cos(phi),
                    cos_theta,
                    sin_theta * math.sin(phi),
                ]
            )

    faces: list[list[int]] = []
    for lat in range(lat_steps):
        for lon in range(lon_steps):
            next_lon = (lon + 1) % lon_steps
            top_left = lat * lon_steps + lon
            top_right = lat * lon_steps + next_lon
            bottom_left = (lat + 1) * lon_steps + lon
            bottom_right = (lat + 1) * lon_steps + next_lon
            if lat != 0:
                faces.append([top_left, bottom_left, top_right])
            if lat != lat_steps - 1:
                faces.append([top_right, bottom_left, bottom_right])

    return Mesh(
        vertices=torch.tensor(vertices, dtype=torch.float32, device=device),
        faces=torch.tensor(faces, dtype=torch.long, device=device),
    ).normalized()


def create_cone_mesh(device: torch.device, radial_steps: int = 24) -> Mesh:
    # Simple non-convex-ish primitive for prompts that imply vertical shapes.
    vertices = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    for index in range(radial_steps):
        angle = (2.0 * math.pi * index) / radial_steps
        vertices.append([math.cos(angle), -1.0, math.sin(angle)])

    faces = []
    for index in range(radial_steps):
        next_index = ((index + 1) % radial_steps) + 2
        current = index + 2
        faces.append([0, current, next_index])
        faces.append([1, next_index, current])

    return Mesh(
        vertices=torch.tensor(vertices, dtype=torch.float32, device=device),
        faces=torch.tensor(faces, dtype=torch.long, device=device),
    ).normalized()


def primitive_mesh_from_prompt(prompt: str, device: torch.device) -> Mesh:
    # Synthetic fallback when no explicit geometry prior is available. This is
    # only a convenience baseline, not semantic text-to-3D generation.
    lower_prompt = prompt.lower()
    if any(token in lower_prompt for token in ("sphere", "ball", "planet", "orb")):
        return create_uv_sphere_mesh(device)
    if any(token in lower_prompt for token in ("cone", "tree", "tower", "mountain")):
        return create_cone_mesh(device)
    return create_cube_mesh(device)


def _sample_face_indices(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
    # torch.multinomial cannot sample from more than 2^24 categories in one
    # call. Dense COLMAP Poisson meshes can exceed that, so use a hierarchical
    # chunked draw when needed:
    # 1. sample a chunk based on chunk mass
    # 2. sample a face within that chunk
    max_categories = 1 << 24
    if probabilities.shape[0] < max_categories:
        return torch.multinomial(probabilities, num_samples, replacement=True)

    cpu_probabilities = probabilities.detach().cpu()
    chunk_size = 1 << 20
    chunk_masses = []
    chunk_starts = []
    for start in range(0, cpu_probabilities.shape[0], chunk_size):
        end = min(start + chunk_size, cpu_probabilities.shape[0])
        chunk_starts.append(start)
        chunk_masses.append(cpu_probabilities[start:end].sum())

    chunk_mass_tensor = torch.stack(chunk_masses)
    chunk_choices = torch.multinomial(chunk_mass_tensor / chunk_mass_tensor.sum().clamp_min(1e-8), num_samples, replacement=True)

    sample_indices = torch.empty(num_samples, dtype=torch.long)
    for chunk_id in torch.unique(chunk_choices):
        chunk_id_int = int(chunk_id.item())
        sample_mask = chunk_choices == chunk_id
        count = int(sample_mask.sum().item())
        start = chunk_starts[chunk_id_int]
        end = min(start + chunk_size, cpu_probabilities.shape[0])
        local_probabilities = cpu_probabilities[start:end]
        local_indices = torch.multinomial(
            local_probabilities / local_probabilities.sum().clamp_min(1e-8),
            count,
            replacement=True,
        )
        sample_indices[sample_mask] = local_indices + start

    return sample_indices.to(probabilities.device)


def sample_surface(mesh: Mesh, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Area-weighted triangle sampling gives anchors distributed over the mesh
    # surface without over-focusing on tiny triangles.
    triangles = mesh.vertices[mesh.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = torch.cross(edges_a, edges_b, dim=-1)
    areas = 0.5 * cross.norm(dim=-1)
    probabilities = areas / areas.sum().clamp_min(1e-8)
    face_indices = _sample_face_indices(probabilities, num_samples)

    chosen_triangles = triangles[face_indices]
    chosen_normals = _normalize(cross[face_indices])

    u = torch.rand(num_samples, 1, device=mesh.vertices.device)
    v = torch.rand(num_samples, 1, device=mesh.vertices.device)
    sqrt_u = torch.sqrt(u)
    barycentric = torch.cat((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), dim=1)
    samples = (chosen_triangles * barycentric.unsqueeze(-1)).sum(dim=1)
    return samples, chosen_normals


def sample_completion_regions(
    mesh: Mesh,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    # Open boundaries are the most likely missing regions, so prioritize them
    # when possible. If the mesh is closed, fall back to generic surface seeds.
    triangles = mesh.vertices[mesh.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    face_normals = _normalize(torch.cross(edges_a, edges_b, dim=-1))

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
    if torch.any(boundary_mask):
        boundary_unique_indices = torch.nonzero(boundary_mask, as_tuple=False).squeeze(1)
        chosen_unique_indices = boundary_unique_indices[
            torch.randint(
                low=0,
                high=boundary_unique_indices.shape[0],
                size=(num_samples,),
                device=mesh.vertices.device,
            )
        ]

        sample_edges = unique_edges[chosen_unique_indices]
        boundary_face_ids = torch.zeros_like(boundary_unique_indices)
        for index, unique_index in enumerate(boundary_unique_indices):
            face_slot = torch.nonzero(inverse_indices == unique_index, as_tuple=False)[0, 0]
            boundary_face_ids[index] = edge_face_ids[face_slot]

        face_lookup = {
            int(unique_index.item()): int(face_id.item())
            for unique_index, face_id in zip(boundary_unique_indices, boundary_face_ids)
        }
        sample_face_ids = torch.tensor(
            [face_lookup[int(index.item())] for index in chosen_unique_indices],
            device=mesh.vertices.device,
            dtype=torch.long,
        )

        vertices_a = mesh.vertices[sample_edges[:, 0]]
        vertices_b = mesh.vertices[sample_edges[:, 1]]
        edge_t = torch.rand((num_samples, 1), device=mesh.vertices.device)
        edge_points = torch.lerp(vertices_a, vertices_b, edge_t)
        normals = face_normals[sample_face_ids]
        return edge_points, normals, "boundary_edges"

    fallback_points, fallback_normals = sample_surface(mesh, num_samples)
    return fallback_points, fallback_normals, "surface_fallback"
