from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hybrid_gs.camera import look_at_camera
from hybrid_gs.gaussians import GaussianState, procedural_colors, prompt_palette
from hybrid_gs.mesh import load_obj_mesh, primitive_mesh_from_prompt, sample_surface
from hybrid_gs.renderer import render_gaussians


def parse_args() -> argparse.Namespace:
    # The viewer now renders a sequence of orbit frames with the same Gaussian
    # renderer used during training. This produces a much more realistic result
    # than plotting Gaussian centers as a debug scatter cloud.
    parser = argparse.ArgumentParser(
        description="Create a rendered HTML viewer for a saved hybrid Gaussian state."
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Path to a saved gaussian_state.npz file from hybrid_gs.pipeline.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional gaussian_metadata.txt path used for branch statistics in the HTML summary.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Optional OBJ mesh path. Used only to sample a fallback Gaussian cloud if --state is not provided.",
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
        help="Optional cap for displayed splats. Larger values make viewer generation slower.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=40,
        help="Number of orbit frames rendered into the HTML viewer.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Rendered viewer frame width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Rendered viewer frame height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback speed for the embedded orbit animation.",
    )
    parser.add_argument(
        "--supersample",
        type=float,
        default=1.5,
        help="Render frames above display resolution, then downsample them for a cleaner viewer image.",
    )
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=45.0,
        help="Field of view used for the viewer orbit cameras.",
    )
    parser.add_argument(
        "--elevation-degrees",
        type=float,
        default=18.0,
        help="Elevation angle of the viewer orbit cameras.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=2.3,
        help="Multiplier applied to the scene extent to choose the orbit radius.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.0,
        help="Extra padding around the visible alpha footprint when auto-cropping viewer frames. Set to 0 to disable cropping.",
    )
    parser.add_argument(
        "--render-tile-size",
        type=int,
        default=96,
        help="Tile size used while rendering viewer frames.",
    )
    parser.add_argument(
        "--render-support-scale",
        type=float,
        default=1.75,
        help="Screen-space support scale passed through to the Gaussian renderer.",
    )
    parser.add_argument(
        "--render-alpha-threshold",
        type=float,
        default=0.002,
        help="Minimum peak alpha needed for a splat to contribute to a tile.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used to render viewer frames.",
    )
    parser.add_argument(
        "--output-html",
        default="outputs/interactive_splat_viewer.html",
        help="Where to save the standalone HTML viewer.",
    )
    parser.add_argument(
        "--title",
        default="Hybrid Gaussian Render Viewer",
        help="Title shown in the HTML viewer.",
    )
    return parser.parse_args()


def load_state_from_npz(path: str | Path) -> dict[str, np.ndarray]:
    # Viewer input is the same saved state emitted by the training pipeline.
    # Keep the required keys explicit so invalid files fail early.
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
    # Allow the viewer to inspect a mesh or primitive even without a trained
    # Gaussian state file.
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
    # Metadata is optional so older runs without the sidecar file still load.
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_subsample(state: dict[str, np.ndarray], max_splats: int) -> dict[str, np.ndarray]:
    # Viewer generation cost scales with the number of splats, so cap very
    # dense results by keeping the most opaque splats.
    num_splats = state["means"].shape[0]
    if max_splats <= 0 or num_splats <= max_splats:
        return state

    opacity = state["opacity"].reshape(-1)
    order = np.argsort(opacity)[::-1][:max_splats]
    return {key: value[order] for key, value in state.items()}


def resolve_device(device_name: str) -> torch.device:
    # Auto mode prefers CUDA when available because rendering several orbit
    # frames is materially faster there, but callers can force CPU when needed.
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA viewer rendering was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def numpy_state_to_torch(state: dict[str, np.ndarray], device: torch.device) -> GaussianState:
    # Convert saved numpy arrays back into the torch container expected by the
    # renderer. This mirrors the state layout saved by the training pipeline.
    return GaussianState(
        means=torch.from_numpy(state["means"]).to(device=device, dtype=torch.float32),
        scales=torch.from_numpy(state["scales"]).to(device=device, dtype=torch.float32),
        colors=torch.from_numpy(state["colors"]).to(device=device, dtype=torch.float32),
        opacity=torch.from_numpy(state["opacity"]).to(device=device, dtype=torch.float32),
    )


def infer_scene_frame(state: dict[str, np.ndarray], radius_scale: float, min_radius: float = 1.2) -> tuple[np.ndarray, float, float]:
    # Orbiting around the bounding-box center keeps off-center scenes visible.
    # Using the box diagonal rather than assuming the scene is at the origin
    # makes the viewer work for both normalized objects and COLMAP scenes.
    means = state["means"]
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(np.linalg.norm(maxs - mins))
    radius = max(extent * radius_scale, min_radius)
    return center.astype(np.float32), radius, extent


def build_orbit_cameras_around_state(
    state: dict[str, np.ndarray],
    num_frames: int,
    width: int,
    height: int,
    fov_degrees: float,
    elevation_degrees: float,
    radius_scale: float,
    device: torch.device,
) -> tuple[list, np.ndarray, float, float]:
    # Render a camera path around the actual learned Gaussian cloud instead of
    # assuming the subject lives exactly at the origin.
    center_np, radius, extent = infer_scene_frame(state, radius_scale=radius_scale)
    center = torch.tensor(center_np, dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    elevation = math.radians(elevation_degrees)
    cameras = []

    for index in range(max(num_frames, 1)):
        azimuth = (2.0 * math.pi * index) / max(num_frames, 1)
        eye = center + torch.tensor(
            [
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.sin(elevation),
                radius * math.cos(elevation) * math.sin(azimuth),
            ],
            dtype=torch.float32,
            device=device,
        )
        cameras.append(
            look_at_camera(
                eye=eye,
                target=center,
                up=up,
                width=width,
                height=height,
                fov_degrees=fov_degrees,
            )
        )

    return cameras, center_np, radius, extent


def crop_rendered_image(
    image: torch.Tensor,
    alpha: torch.Tensor,
    output_width: int,
    output_height: int,
    padding_ratio: float,
    alpha_cutoff: float = 0.02,
) -> torch.Tensor:
    # Crop around the visible alpha footprint so sparse reconstructions no
    # longer sit as tiny blobs in a large empty frame. Padding keeps the crop
    # from feeling claustrophobic and preserves a stable orbit composition.
    if padding_ratio <= 0.0:
        return image

    alpha_mask = alpha > alpha_cutoff
    if not torch.any(alpha_mask):
        return image

    ys, xs = torch.where(alpha_mask)
    top = int(ys.min().item())
    bottom = int(ys.max().item()) + 1
    left = int(xs.min().item())
    right = int(xs.max().item()) + 1

    subject_height = max(bottom - top, 1)
    subject_width = max(right - left, 1)
    pad_y = max(int(subject_height * padding_ratio), 4)
    pad_x = max(int(subject_width * padding_ratio), 4)

    top = max(top - pad_y, 0)
    bottom = min(bottom + pad_y, image.shape[0])
    left = max(left - pad_x, 0)
    right = min(right + pad_x, image.shape[1])

    cropped = image[top:bottom, left:right]
    array = (cropped.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(array).resize((output_width, output_height), Image.Resampling.LANCZOS)
    resized = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(resized)


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    # Convert a rendered frame into a standard uint8 image array before saving
    # it to disk. Keeping this separate avoids duplicating clamp/scale logic.
    return (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def save_frame_image(image: torch.Tensor, path: Path) -> None:
    # Viewer HTML now references external PNG files instead of embedding every
    # frame inline. That keeps the page size manageable so browsers actually
    # finish loading medium and large reconstructions.
    pil_image = Image.fromarray(tensor_to_uint8(image))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, format="PNG", optimize=True)


def relative_posix_path(path: Path, start: Path) -> str:
    # Use relative URLs so the HTML plus sibling frame folder can be moved
    # together without breaking links, and normalize separators for browsers.
    return Path(path.relative_to(start)).as_posix()


def ensure_clean_frame_dir(frame_dir: Path) -> None:
    # Regenerating the viewer should not leave stale frames behind from a
    # previous export with a different frame count.
    if frame_dir.exists():
        for child in frame_dir.iterdir():
            if child.is_file():
                child.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)


def build_mesh_section_html(mesh_preview_html: str) -> str:
    # Keep the optional mesh preview in a dedicated block so the main template
    # stays readable and does not rely on inline Python expressions.
    if not mesh_preview_html:
        return ""
    return (
        "<section class='panel mesh-preview'>"
        "<p class='kicker'>Geometry</p>"
        "<h2 class='section-title'>Mesh Preview</h2>"
        f"{mesh_preview_html}"
        "</section>"
    )


def summarize_branches(metadata: dict[str, int | str], total_splats: int) -> dict[str, int]:
    # Branch counts are still useful even though the rendered viewer no longer
    # plots each branch separately.
    anchored = int(metadata.get("anchored_splats", 0))
    detail = int(metadata.get("detail_splats", 0))
    completion = int(metadata.get("completion_splats", 0))
    if anchored + detail + completion != total_splats:
        return {"gaussians": total_splats}
    return {
        "anchored": anchored,
        "detail": detail,
        "completion": completion,
    }


def maybe_build_mesh_preview_html(mesh_path: str | None) -> str:
    # When a real mesh prior exists, keep that geometry visible in the viewer.
    # The rendered Gaussian orbit remains the primary view, but an adjacent
    # Plotly mesh preview preserves the explicit geometric structure.
    if not mesh_path:
        return ""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return "<p class='hint'>Plotly is not installed, so the mesh geometry preview is unavailable in this HTML.</p>"

    mesh = load_obj_mesh(mesh_path, torch.device("cpu"))
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#a5b6d8",
                opacity=0.92,
                hoverinfo="skip",
                lighting={
                    "ambient": 0.65,
                    "diffuse": 0.7,
                    "specular": 0.06,
                    "roughness": 0.95,
                    "fresnel": 0.02,
                },
                lightposition={"x": 120, "y": 160, "z": 80},
            )
        ]
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
        },
    )
    return pio.to_html(figure, include_plotlyjs=True, full_html=False, config={"displaylogo": False})


def render_viewer_frames(
    state: dict[str, np.ndarray],
    frame_dir: Path,
    html_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    fov_degrees: float,
    elevation_degrees: float,
    radius_scale: float,
    supersample: float,
    tile_size: int | None,
    support_scale: float,
    alpha_threshold: float,
    crop_padding: float,
    device: torch.device,
) -> tuple[list[str], dict[str, float | list[float] | str]]:
    # Use the same Gaussian renderer as training so the viewer reflects what
    # the model actually knows, not a secondary scatter-plot approximation.
    torch_state = numpy_state_to_torch(state, device=device)
    render_width = max(int(round(width * max(supersample, 1.0))), width)
    render_height = max(int(round(height * max(supersample, 1.0))), height)
    cameras, center, radius, extent = build_orbit_cameras_around_state(
        state=state,
        num_frames=num_frames,
        width=render_width,
        height=render_height,
        fov_degrees=fov_degrees,
        elevation_degrees=elevation_degrees,
        radius_scale=radius_scale,
        device=device,
    )

    frames: list[str] = []
    with torch.no_grad():
        for frame_index, camera in enumerate(cameras):
            rendered, alpha = render_gaussians(
                torch_state,
                camera,
                tile_size=tile_size,
                support_scale=support_scale,
                alpha_threshold=alpha_threshold,
                return_alpha=True,
            )
            framed = crop_rendered_image(
                rendered,
                alpha,
                output_width=width,
                output_height=height,
                padding_ratio=crop_padding,
            )
            frame_path = frame_dir / f"frame_{frame_index:03d}.png"
            save_frame_image(framed, frame_path)
            frames.append(relative_posix_path(frame_path, html_dir))

    summary = {
        "center": [round(float(value), 4) for value in center.tolist()],
        "radius": round(float(radius), 4),
        "extent": round(float(extent), 4),
        "device": str(device),
        "resolution": f"{width}x{height}",
        "render_resolution": f"{render_width}x{render_height}",
        "supersample": round(float(supersample), 3),
        "crop_padding": round(float(crop_padding), 4),
    }
    return frames, summary


def build_html_document(
    title: str,
    frames: list[str],
    fps: float,
    render_summary: dict[str, float | list[float] | str],
    branch_summary: dict[str, int],
    mesh_preview_html: str,
) -> str:
    # The HTML viewer is a lightweight orbit player with keyboard controls,
    # frame slider, and a compact stats panel. It favors presentation over
    # debugging because the goal is to inspect the rendered reconstruction.
    frame_json = json.dumps(frames)
    render_summary_json = json.dumps(render_summary, indent=2)
    branch_summary_json = json.dumps(branch_summary, indent=2)
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    mesh_section_html = build_mesh_section_html(mesh_preview_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #ffffff;
      --panel-edge: #d7dfeb;
      --text: #344767;
      --muted: #667895;
      --accent: #3b5bdb;
      --shadow: none;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Aptos", "Helvetica Neue", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px 18px 26px;
    }}
    .header {{
      display: block;
      margin-bottom: 14px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 1.05rem;
      line-height: 1.2;
      font-weight: 600;
      color: #40506c;
    }}
    .header p {{
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 820px;
      font-size: 0.92rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-edge);
      border-radius: 4px;
      box-shadow: var(--shadow);
    }}
    .viewer-panel {{
      padding: 10px;
    }}
    .image-wrap {{
      position: relative;
      overflow: hidden;
      border-radius: 2px;
      background: #ffffff;
      border: 1px solid #d7dfeb;
    }}
    .image-wrap img {{
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 1 / 1;
      object-fit: contain;
    }}
    .badge {{
      position: absolute;
      top: 14px;
      left: 14px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #d7dfeb;
      color: #52637f;
      padding: 6px 10px;
      border-radius: 14px;
      font-size: 0.84rem;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto auto minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
    }}
    button {{
      border: 0;
      border-radius: 4px;
      padding: 8px 12px;
      color: #ffffff;
      background: #5673c8;
      font-weight: 600;
      cursor: pointer;
      box-shadow: none;
    }}
    button.secondary {{
      background: #f4f7fb;
      color: #41557f;
      box-shadow: none;
      border: 1px solid #d7dfeb;
    }}
    input[type=range] {{
      width: 100%;
      accent-color: #5673c8;
    }}
    .stats-panel {{
      padding: 10px;
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .mesh-preview {{
      padding: 10px;
    }}
    .mesh-preview .plotly-graph-div {{
      width: 100% !important;
      height: 340px !important;
    }}
    .kicker {{
      margin: 0;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.68rem;
      font-weight: 600;
    }}
    .section-title {{
      margin: 4px 0 0;
      font-size: 0.98rem;
      font-weight: 600;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .card {{
      padding: 10px;
      border-radius: 4px;
      background: #f7f9fc;
      border: 1px solid #e2e8f2;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card .value {{
      margin-top: 5px;
      font-size: 0.98rem;
      font-weight: 600;
      word-break: break-word;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      border-radius: 4px;
      background: #f7f9fc;
      border: 1px solid #e2e8f2;
      color: #425171;
      overflow: auto;
      font-size: 0.82rem;
      line-height: 1.38;
    }}
    .hint {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 0.86rem;
    }}
    @media (max-width: 1080px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .controls {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .controls .slider-wrap {{
        grid-column: 1 / -1;
      }}
      .card-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div>
        <h1>{safe_title}</h1>
        <p>Rendered orbit preview generated from the saved Gaussian state using the same tile-aware renderer as training.</p>
      </div>
    </div>

    <div class="grid">
      <section class="panel viewer-panel">
        <div class="image-wrap">
          <span class="badge" id="frameLabel">Frame 1 / {len(frames)}</span>
          <img id="viewerFrame" src="{frames[0]}" alt="Rendered Gaussian orbit frame">
        </div>
        <div class="controls">
          <button id="playButton">Play</button>
          <button id="prevButton" class="secondary">Prev</button>
          <button id="nextButton" class="secondary">Next</button>
          <div class="slider-wrap">
            <input id="frameSlider" type="range" min="0" max="{max(len(frames) - 1, 0)}" value="0">
          </div>
          <div id="timecode">1 / {len(frames)}</div>
        </div>
      </section>

      <aside class="panel stats-panel">
        <div>
          <p class="kicker">Viewer Summary</p>
          <h2 class="section-title">Render Settings</h2>
        </div>

        <div class="card-grid">
          <div class="card">
            <div class="label">Frames</div>
            <div class="value">{len(frames)}</div>
          </div>
          <div class="card">
            <div class="label">Playback</div>
            <div class="value">{fps:.1f} fps</div>
          </div>
          <div class="card">
            <div class="label">Device</div>
            <div class="value" id="deviceValue"></div>
          </div>
          <div class="card">
            <div class="label">Resolution</div>
            <div class="value" id="resolutionValue"></div>
          </div>
        </div>

        <div>
          <p class="kicker">Scene Stats</p>
          <pre id="sceneSummary">{render_summary_json}</pre>
        </div>

        <div>
          <p class="kicker">Branch Counts</p>
          <pre id="branchSummary">{branch_summary_json}</pre>
        </div>

        <p class="hint">Arrow keys step through the orbit. Space toggles playback. The viewer keeps a looser default framing so it stays visually closer to the earlier Plotly-style inspection page.</p>
      </aside>
    </div>
    {mesh_section_html}
  </div>

  <script>
    const frames = {frame_json};
    const fps = {float(fps)};
    const renderSummary = {json.dumps(render_summary)};
    document.getElementById("deviceValue").textContent = renderSummary.device;
    document.getElementById("resolutionValue").textContent = renderSummary.resolution;

    const image = document.getElementById("viewerFrame");
    const slider = document.getElementById("frameSlider");
    const playButton = document.getElementById("playButton");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");
    const frameLabel = document.getElementById("frameLabel");
    const timecode = document.getElementById("timecode");

    let currentIndex = 0;
    let timer = null;

    function updateFrame(index) {{
      currentIndex = (index + frames.length) % frames.length;
      image.src = frames[currentIndex];
      slider.value = currentIndex;
      frameLabel.textContent = `Frame ${{currentIndex + 1}} / ${{frames.length}}`;
      timecode.textContent = `${{currentIndex + 1}} / ${{frames.length}}`;
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        window.clearInterval(timer);
        timer = null;
      }}
      playButton.textContent = "Play";
    }}

    function startPlayback() {{
      stopPlayback();
      timer = window.setInterval(() => updateFrame(currentIndex + 1), 1000 / fps);
      playButton.textContent = "Pause";
    }}

    playButton.addEventListener("click", () => {{
      if (timer === null) {{
        startPlayback();
      }} else {{
        stopPlayback();
      }}
    }});

    prevButton.addEventListener("click", () => {{
      stopPlayback();
      updateFrame(currentIndex - 1);
    }});

    nextButton.addEventListener("click", () => {{
      stopPlayback();
      updateFrame(currentIndex + 1);
    }});

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      updateFrame(Number(event.target.value));
    }});

    window.addEventListener("keydown", (event) => {{
      if (event.code === "Space") {{
        event.preventDefault();
        if (timer === null) {{
          startPlayback();
        }} else {{
          stopPlayback();
        }}
      }} else if (event.code === "ArrowRight") {{
        stopPlayback();
        updateFrame(currentIndex + 1);
      }} else if (event.code === "ArrowLeft") {{
        stopPlayback();
        updateFrame(currentIndex - 1);
      }}
    }});
  </script>
</body>
</html>
"""


def save_rendered_viewer_html(
    state: dict[str, np.ndarray],
    metadata: dict[str, int | str],
    output_path: str | Path,
    title: str,
    *,
    mesh_path: str | None = None,
    max_splats: int = 3000,
    num_frames: int = 40,
    width: int = 640,
    height: int = 640,
    fps: float = 10.0,
    fov_degrees: float = 45.0,
    elevation_degrees: float = 18.0,
    radius_scale: float = 2.3,
    supersample: float = 1.5,
    crop_padding: float = 0.0,
    tile_size: int | None = 96,
    support_scale: float = 1.75,
    alpha_threshold: float = 0.002,
    device_name: str = "auto",
) -> Path:
    # Public helper used by both the CLI and the training pipeline's
    # `--prompt-viewer` hook.
    output_path = Path(output_path)
    html_dir = output_path.parent.resolve()
    frame_dir = html_dir / f"{output_path.stem}_frames"
    ensure_clean_frame_dir(frame_dir)
    state = maybe_subsample(state, max_splats=max_splats)
    device = resolve_device(device_name)
    frames, render_summary = render_viewer_frames(
        state=state,
        frame_dir=frame_dir,
        html_dir=html_dir,
        num_frames=num_frames,
        width=width,
        height=height,
        fov_degrees=fov_degrees,
        elevation_degrees=elevation_degrees,
        radius_scale=radius_scale,
        supersample=supersample,
        tile_size=tile_size,
        support_scale=support_scale,
        alpha_threshold=alpha_threshold,
        crop_padding=crop_padding,
        device=device,
    )
    html = build_html_document(
        title=title,
        frames=frames,
        fps=fps,
        render_summary=render_summary,
        branch_summary=summarize_branches(metadata, total_splats=state["means"].shape[0]),
        mesh_preview_html=maybe_build_mesh_preview_html(mesh_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    # CLI entrypoint: load a saved Gaussian state, render an orbit sequence,
    # then bundle the frames into one standalone HTML file.
    args = parse_args()
    if args.state:
        state = load_state_from_npz(args.state)
    else:
        state = build_fallback_state(args)

    metadata = load_metadata(args.metadata)
    output_path = save_rendered_viewer_html(
        state=state,
        metadata=metadata,
        output_path=args.output_html,
        title=args.title,
        mesh_path=args.mesh,
        max_splats=args.max_splats,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        supersample=args.supersample,
        fov_degrees=args.fov_degrees,
        elevation_degrees=args.elevation_degrees,
        radius_scale=args.radius_scale,
        crop_padding=args.crop_padding,
        tile_size=args.render_tile_size,
        support_scale=args.render_support_scale,
        alpha_threshold=args.render_alpha_threshold,
        device_name=args.device,
    )
    print(f"Saved interactive viewer to: {output_path}")


if __name__ == "__main__":
    main()
