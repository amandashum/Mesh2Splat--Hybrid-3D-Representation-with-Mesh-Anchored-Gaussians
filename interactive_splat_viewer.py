from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hybrid_gs.gaussians import procedural_colors, prompt_palette
from hybrid_gs.mesh import load_obj_mesh, primitive_mesh_from_prompt, sample_surface


def _load_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required for the interactive viewer. "
            "Install it with: python -m pip install plotly"
        ) from exc
    return go, pio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive 3D hybrid mesh + Gaussian viewer.")
    parser.add_argument(
        "--state",
        default=None,
        help="Path to a saved gaussian_state.npz file from hybrid_gs.pipeline.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional gaussian_metadata.txt path. Used to split anchored/detail/completion branches.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Optional OBJ mesh path. Used to draw the mesh and to sample a fallback Gaussian cloud if --state is not provided.",
    )
    parser.add_argument(
        "--prompt",
        default="stone statue",
        help="Prompt used for procedural colors when sampling a fallback cloud.",
    )
    parser.add_argument(
        "--num-splats",
        type=int,
        default=384,
        help="Number of sampled fallback splats when loading from a mesh or primitive prompt.",
    )
    parser.add_argument(
        "--max-splats",
        type=int,
        default=3000,
        help="Optional cap for displayed splats. Larger values make the browser heavier.",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=38.0,
        help="Multiplier applied to Gaussian scale to produce marker sizes.",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=3.0,
        help="Minimum displayed marker size in pixels.",
    )
    parser.add_argument(
        "--output-html",
        default="outputs/interactive_splat_viewer.html",
        help="Where to save the interactive HTML viewer.",
    )
    parser.add_argument(
        "--title",
        default="Interactive Hybrid Mesh + Gaussian Viewer",
        help="Title shown in the HTML viewer.",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.22,
        help="Opacity of the optional mesh overlay.",
    )
    parser.add_argument(
        "--show-wireframe",
        action="store_true",
        help="Overlay mesh triangle edges when --mesh is provided.",
    )
    return parser.parse_args()


def load_state_from_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"means", "scales", "colors", "opacity"}
    missing = required.difference(data.files)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"State file {path} is missing required arrays: {missing_text}")

    return {
        "means": np.asarray(data["means"], dtype=np.float32),
        "scales": np.asarray(data["scales"], dtype=np.float32),
        "colors": np.asarray(data["colors"], dtype=np.float32),
        "opacity": np.asarray(data["opacity"], dtype=np.float32),
    }


def build_fallback_state(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    if args.mesh:
        mesh = load_obj_mesh(args.mesh, device)
    else:
        mesh = primitive_mesh_from_prompt(args.prompt, device)

    anchors, normals = sample_surface(mesh, args.num_splats)
    palette = prompt_palette(args.prompt, device)
    colors = procedural_colors(anchors, normals, palette)
    scales = torch.full_like(anchors, 0.05)
    opacity = torch.full((anchors.shape[0], 1), 0.75, dtype=torch.float32, device=device)

    return {
        "means": anchors.cpu().numpy(),
        "scales": scales.cpu().numpy(),
        "colors": colors.cpu().numpy(),
        "opacity": opacity.cpu().numpy(),
    }


def load_metadata(path: str | Path | None) -> dict[str, int | str]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_subsample(state: dict[str, np.ndarray], max_splats: int) -> dict[str, np.ndarray]:
    num_splats = state["means"].shape[0]
    if max_splats <= 0 or num_splats <= max_splats:
        return state

    opacity = state["opacity"].reshape(-1)
    order = np.argsort(opacity)[::-1][:max_splats]
    return {key: value[order] for key, value in state.items()}


def build_wireframe_points(vertices: np.ndarray, faces: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    x_lines: list[float] = []
    y_lines: list[float] = []
    z_lines: list[float] = []

    for face in faces:
        triangle = vertices[face]
        loop = np.concatenate((triangle, triangle[:1]), axis=0)
        for point in loop:
            x_lines.append(float(point[0]))
            y_lines.append(float(point[1]))
            z_lines.append(float(point[2]))
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    return x_lines, y_lines, z_lines


def _make_branch_trace(
    go,
    state: dict[str, np.ndarray],
    label: str,
    size_scale: float,
    min_size: float,
    marker_line_color: str,
):
    means = state["means"]
    scales = state["scales"]
    colors = state["colors"].clip(0.0, 1.0)
    opacity = state["opacity"].reshape(-1).clip(0.05, 1.0)

    if means.shape[0] == 0:
        return None

    marker_sizes = np.clip(scales.mean(axis=1) * size_scale, min_size, None)
    rgba_colors = [
        f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, {alpha:.4f})"
        for rgb, alpha in zip(colors, opacity)
    ]

    return go.Scatter3d(
        x=means[:, 0],
        y=means[:, 1],
        z=means[:, 2],
        mode="markers",
        name=label,
        marker={
            "size": marker_sizes,
            "color": rgba_colors,
            "sizemode": "diameter",
            "line": {"width": 1, "color": marker_line_color},
        },
        customdata=np.concatenate(
            [
                scales.mean(axis=1, keepdims=True),
                opacity[:, None],
                colors,
            ],
            axis=1,
        ),
        hovertemplate=(
            f"{label}<br>"
            "x=%{x:.3f}<br>"
            "y=%{y:.3f}<br>"
            "z=%{z:.3f}<br>"
            "scale=%{customdata[0]:.4f}<br>"
            "opacity=%{customdata[1]:.3f}<br>"
            "color=(%{customdata[2]:.2f}, %{customdata[3]:.2f}, %{customdata[4]:.2f})"
            "<extra></extra>"
        ),
    )


def split_state_by_metadata(
    state: dict[str, np.ndarray],
    metadata: dict[str, int | str],
) -> list[tuple[str, dict[str, np.ndarray], str]]:
    anchored_count = int(metadata.get("anchored_splats", 0))
    detail_count = int(metadata.get("detail_splats", 0))
    completion_count = int(metadata.get("completion_splats", 0))
    total_count = state["means"].shape[0]

    if anchored_count + detail_count + completion_count != total_count:
        return [("Gaussians", state, "#253238")]

    counts = [
        ("Anchored", anchored_count, "#1f4e79"),
        ("Detail", detail_count, "#9c5f19"),
        ("Completion", completion_count, "#7d2e68"),
    ]

    offset = 0
    branches: list[tuple[str, dict[str, np.ndarray], str]] = []
    for label, count, outline in counts:
        branch_state = {key: value[offset : offset + count] for key, value in state.items()}
        branches.append((label, branch_state, outline))
        offset += count
    return branches


def state_to_figure(
    state: dict[str, np.ndarray],
    metadata: dict[str, int | str],
    mesh_path: str | None,
    title: str,
    size_scale: float,
    min_size: float,
    mesh_opacity: float,
    show_wireframe: bool,
):
    go, _ = _load_plotly()

    traces = []
    if mesh_path:
        mesh = load_obj_mesh(mesh_path, torch.device("cpu"))
        vertices = mesh.vertices.cpu().numpy()
        faces = mesh.faces.cpu().numpy()
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#b7c4bf",
                opacity=max(0.0, min(mesh_opacity, 1.0)),
                name="Mesh",
                hoverinfo="skip",
                lighting={
                    "ambient": 0.55,
                    "diffuse": 0.65,
                    "specular": 0.08,
                    "roughness": 0.95,
                    "fresnel": 0.02,
                },
                lightposition={"x": 120, "y": 160, "z": 80},
            )
        )
        if show_wireframe:
            line_x, line_y, line_z = build_wireframe_points(vertices, faces)
            traces.append(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    name="Wireframe",
                    line={"color": "#44525a", "width": 2},
                    hoverinfo="skip",
                )
            )

    for label, branch_state, outline in split_state_by_metadata(state, metadata):
        trace = _make_branch_trace(
            go=go,
            state=branch_state,
            label=label,
            size_scale=size_scale,
            min_size=min_size,
            marker_line_color=outline,
        )
        if trace is not None:
            traces.append(trace)

    figure = go.Figure(data=traces)

    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.85)"},
        scene={
            "aspectmode": "data",
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.1}},
        },
    )
    return figure


def main() -> None:
    args = parse_args()
    if args.state:
        state = load_state_from_npz(args.state)
    else:
        state = build_fallback_state(args)

    state = maybe_subsample(state, args.max_splats)
    metadata = load_metadata(args.metadata)
    figure = state_to_figure(
        state=state,
        metadata=metadata,
        mesh_path=args.mesh,
        title=args.title,
        size_scale=args.size_scale,
        min_size=args.min_size,
        mesh_opacity=args.mesh_opacity,
        show_wireframe=args.show_wireframe,
    )

    _, pio = _load_plotly()
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(figure, file=str(output_path), auto_open=False, include_plotlyjs=True)
    print(f"Saved interactive viewer to: {output_path}")


if __name__ == "__main__":
    main()
