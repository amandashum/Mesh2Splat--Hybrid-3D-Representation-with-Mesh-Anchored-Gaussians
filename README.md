# Hybrid Mesh + Gaussian Splatting Baseline

This repository is a reduced implementation of the project proposal in `CMPT 743_ 3D Reconstruction.docx`: a hybrid Gaussian scene/model reconstruction pipeline with three Gaussian branches:

- anchored splats tied to a mesh prior or COLMAP sparse scene prior
- detail splats for local appearance variation
- completion splats for weakly constrained missing or under-observed regions

The repository is intentionally scoped to the core hybrid baseline. Side experiments, generated outputs, local environments, and image-to-3D wrappers have been removed.

## Repository layout

- `main.py`: entrypoint
- `hybrid_gs/`: core mesh, camera, Gaussian, renderer, loss, and optimization code
- `hybrid_gs/completion/`: completion-specific seeding and continuity losses for missing regions
- `hybrid_gs/segmentation.py`: heuristic scene-structure masks that keep completion near plausible surface continuation regions
- `interactive_splat_viewer.py`: optional HTML viewer for saved Gaussian states
- `COLMAP_SETUP.md`: Windows-oriented COLMAP setup notes for multi-view reconstruction
- `tools/run_colmap.ps1`: helper script for a standard COLMAP sparse pipeline
- `data/`: place optional input meshes or reference images here
- `checkpoints/`: reserved for future external model checkpoints

## What the baseline does

1. Load an OBJ mesh, create a primitive mesh, or use a COLMAP sparse scene prior
2. Optionally load real multi-view cameras and targets from a COLMAP text export
3. Build anchor, detail, and completion seeds from the mesh or sparse scene points
   Completion seeding now lives in `hybrid_gs/completion/` so missing-region
   logic can evolve independently of mesh loading and rendering.
4. Initialize anchored, detail, and completion Gaussian branches
5. Optimize Gaussian parameters with reconstruction, tether, completion,
   completion-continuity, completion-region, appearance, scale, and opacity losses
6. Save rendered views plus separate mesh-prior and with-completion artifacts
   for inspection

## Install

```powershell
python -m pip install -r requirements.txt
```

## Reactivate the Environment

If you created the virtual environment in this repository and later open a new
terminal, reactivate it before running training commands.

In Command Prompt:

```cmd
cd /d C:\mesh2splat
.\.venv\Scripts\activate.bat
```

In PowerShell:

```powershell
cd C:\mesh2splat
.\.venv\Scripts\Activate.ps1
```

## Run the baseline

Default run:

```powershell
python .\main.py --steps 200 --out-dir .\outputs\demo
```

Run with a real OBJ mesh:

```powershell
python .\main.py --mesh .\data\example.obj --prompt "stone statue" --steps 300 --out-dir .\outputs\mesh_run
```

Run with an additional front-view image and mask:

```powershell
python .\main.py `
  --mesh .\data\example.obj `
  --reference-image .\data\front.png `
  --reference-mask .\data\front_mask.png `
  --prompt "stone statue" `
  --out-dir .\outputs\image_guided
```

Run with real COLMAP multi-view supervision:

```powershell
python .\main.py `
  --mesh .\data\example.obj `
  --colmap-model-dir .\data\example\colmap\sparse_txt `
  --colmap-image-dir .\data\example\images `
  --num-views 24 `
  --steps 300 `
  --out-dir .\outputs\colmap_run
```

Run in scene reconstruction mode directly from COLMAP sparse points:

```powershell
python .\main.py `
  --scene-mode `
  --prompt "indoor room" `
  --colmap-model-dir .\data\room\colmap\sparse_txt `
  --colmap-image-dir .\data\room\images `
  --num-views 24 `
  --num-splats 2500 `
  --num-detail-splats 1500 `
  --num-completion-splats 1200 `
  --steps 400 `
  --out-dir .\outputs\room_scene
```

## How to test

If you want to test the current completion pipeline on the Gerrard Hall example,
use this full Command Prompt run:

```cmd
cd /d C:\mesh2splat
.\.venv\Scripts\activate.bat
python .\main.py --mesh C:\mesh2splat\data\room\images\gerrard-hall\mesh_prior\dense\mesh_prior.obj --colmap-model-dir C:\mesh2splat\data\room\images\gerrard-hall\sparse --colmap-image-dir C:\mesh2splat\data\room\images\gerrard-hall\images --prompt building --num-views 6 --num-splats 900 --num-detail-splats 550 --num-completion-splats 300 --steps 120 --colmap-resize-long-edge 128 --render-tile-size 48 --render-support-scale 1.75 --render-alpha-threshold 0.002 --lambda-completion-continuity 0.30 --lambda-completion-region 0.45 --prompt-viewer --out-dir C:\mesh2splat\outputs\gerrard_hall_mesh_completion
```

Expected terminal behavior:

- the usual optimization log with `recon=...`, `completion=...`, `continuity=...`, and `region=...`
- a `Mesh Prior` banner near the end that lists the mesh-prior files being written
- a `Completion Using Splats` banner near the end that lists the completion files being written
- if the completion patch can be formed, the completion banner should mention:
  - `missing_mesh_parts.obj`
  - `merged_mesh_with_splats.obj`

Expected output folder:

```text
C:\mesh2splat\outputs\gerrard_hall_mesh_completion
```

Important files to inspect after the run:

- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\mesh_prior.obj`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\mesh_prior_cloud.ply`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\with_completion_cloud.ply`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\mesh_prior_state.npz`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\completion_state.npz`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\gaussian_state.npz`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\view_00_mesh_prior.png`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\view_00_with_completion.png`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\view_00_completion_region_mask.png`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\view_00_surface_core_mask.png`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\view_00_occluder_mask.png`
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\missing_mesh_parts.obj` if a completion patch is produced
- `C:\mesh2splat\outputs\gerrard_hall_mesh_completion\merged_mesh_with_splats.obj` if a merged completion mesh is produced

If you want a side-by-side PNG comparison from that same run folder:

```cmd
python .\tools\compare_renders.py --single-run-dir C:\mesh2splat\outputs\gerrard_hall_mesh_completion --output-dir C:\mesh2splat\outputs\gerrard_hall_mesh_completion_comparison --left-label "Mesh Prior" --right-label "With Completion" --contact-sheet
```

Useful controls:

- `--num-splats`: anchored splat count
- `--num-detail-splats`: detail splat count
- `--num-completion-splats`: completion splat count
- `--colmap-model-dir`: folder with `cameras.txt` and `images.txt`
- `--colmap-image-dir`: folder containing the original or undistorted images
- `--colmap-resize-long-edge`: downscale cap for COLMAP images during training
- `--render-tile-size`: split rendering into smaller image tiles to reduce GPU memory use
- `--render-support-scale`: shrink or expand how far each splat influences a tile in screen space
- `--render-alpha-threshold`: skip splats whose contribution on a tile is negligible
- `--scene-mode`: use COLMAP sparse points as the scene prior instead of an OBJ mesh
- `--max-sparse-points`: cap the sparse COLMAP point cloud used in scene mode
- `--prompt-viewer`: ask at the end of training whether to generate `viewer.html`
- `--lambda-completion-continuity`: strengthen the new hole-filling continuity loss for completion splats
- `--lambda-completion-region`: strengthen the scene-structure mask loss that keeps completion near plausible surface continuation regions
- `--steps`: optimization length
- `--cpu`: force CPU execution

Each run now writes two explicit stages into the output folder:

- `mesh_prior.obj`: the normalized mesh prior actually used by training
- `mesh_prior_cloud.ply`: Gaussian centers before completion is applied
- `with_completion_cloud.ply`: Gaussian centers after completion learning
- `missing_mesh_parts.obj`: completion patch mesh built from learned splats near hole boundaries, when one can be formed
- `merged_mesh_with_splats.obj`: attempted merged mesh that appends the completion patch back onto the mesh prior
- `mesh_prior_state.npz`: saved Gaussian state before completion
- `completion_state.npz`: saved Gaussian state after completion
- `view_XX_mesh_prior.png`: render from the mesh-prior-only state
- `view_XX_with_completion.png`: render from the final completed state
- `view_XX_completion_region_mask.png`: heuristic mask showing where completion is allowed to grow
- `view_XX_surface_core_mask.png`: conservative mask for the current geometry-supported surface silhouette
- `view_XX_occluder_mask.png`: near-surface disagreement mask highlighting likely occluders or missing-surface zones

## Completion package

The new `hybrid_gs/completion/` folder is where missing-region learning lives:

- `seeding.py` finds likely completion regions
  - mesh mode: sample along open boundary edges and push seeds just beyond the missing side
  - scene mode: sample sparse COLMAP frontier points with low support and high uncertainty
- `losses.py` adds a continuity loss so nearby completion splats move like a
  coherent surface extension instead of drifting independently

If you want completion splats to stay more tightly organized while filling mesh
holes, increase:

```powershell
--lambda-completion-continuity 0.30
```

In plain terms, the three Gaussian branches play different roles:

- `anchored`: splats tied closely to the known geometry or reliable sparse scene structure
- `detail`: splats that refine local appearance and small geometric variation near that known structure
- `completion`: splats allowed to extend into weakly observed or missing regions so the model can try to fill holes instead of leaving gaps

So the completion branch is the part of the model that tries to infer missing
surface regions. In mesh mode it now prefers open boundary edges, and in scene
mode it prefers weakly supported sparse frontier points. It is still a
plausibility-based fill mechanism, not a guaranteed recovery of true unseen
geometry.

When the mesh really has open boundaries, watch the log for:

```text
completion seeds: mesh_boundary_bridge
```

That indicates the completion branch is being initialized from mesh-hole
boundaries rather than generic fallback surface samples.

## Scene-structure segmentation

This repository still does not ship a learned semantic segmenter. Instead, it
now uses a scene-general structure pass in `hybrid_gs/segmentation.py`:

- the current mesh-prior silhouette marks the known surface core
- that silhouette is dilated to create a near-surface continuation context
- residual disagreement between the target image and the mesh-prior render is
  used to flag likely occluders or missing-surface zones
- simple brightness and saturation heuristics flag likely clear background
- completion is encouraged to stay near plausible surface continuation regions
  and discouraged from expanding into obvious background or unsupported lower
  image regions

This is meant to make completion more scene-general, for cases like:

- filling surfaces obscured by bushes or clutter
- avoiding giant roof or ground closures
- avoiding random floating completion far from the main structure

## COLMAP workflow

This repository can train against a COLMAP sparse reconstruction exported to TXT. The typical flow is:

1. Capture overlapping object images.
2. Run COLMAP to estimate poses.
3. Export the sparse model to text.
4. For object-centric reconstruction, optionally pass an OBJ mesh into `main.py`.
5. For scene reconstruction, run `main.py --scene-mode` directly on the sparse model and image folder.

See `COLMAP_SETUP.md` for the Windows setup path and the included `tools/run_colmap.ps1` helper.

## Generate a Mesh Prior from a Scene

If you already have a COLMAP sparse model and image folder, you can generate a
dense mesh prior and convert it to OBJ for the existing `--mesh` workflow.

This helper runs:

1. `image_undistorter`
2. `patch_match_stereo`
3. `stereo_fusion`
4. `poisson_mesher` or `delaunay_mesher`
5. `tools/ply_to_obj.py` to write `mesh_prior.obj`

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\generate_colmap_mesh.ps1 `
  -ColmapBat C:\tools\COLMAP\COLMAP.bat `
  -ImageDir .\data\room\images `
  -ModelDir .\data\room\colmap\sparse_txt `
  -Workspace .\data\room\mesh_prior `
  -Mesher poisson
```

After it finishes, the OBJ mesh prior will be at:

```text
.\data\room\mesh_prior\dense\mesh_prior.obj
```

You can then use it in mesh mode:

```powershell
python .\main.py `
  --mesh .\data\room\mesh_prior\dense\mesh_prior.obj `
  --colmap-model-dir .\data\room\colmap\sparse_txt `
  --colmap-image-dir .\data\room\images `
  --num-views 24 `
  --steps 300 `
  --out-dir .\outputs\room_mesh_prior
```

## Renderer Tuning

The renderer now focuses work more tightly inside each image tile:

- `--render-tile-size` controls the tile/window size used during rendering
- `--render-support-scale` controls how many Gaussian standard deviations are treated as "active" support for a tile
- `--render-alpha-threshold` skips splats whose peak alpha inside a tile is too small to matter

If you need to stay within GPU memory while keeping quality reasonable, a practical pattern is:

```powershell
python .\main.py `
  --scene-mode `
  --colmap-model-dir .\data\room\colmap\sparse_txt `
  --colmap-image-dir .\data\room\images `
  --num-views 8 `
  --num-splats 900 `
  --num-detail-splats 600 `
  --num-completion-splats 350 `
  --steps 80 `
  --colmap-resize-long-edge 192 `
  --render-tile-size 64 `
  --render-support-scale 1.75 `
  --render-alpha-threshold 0.002 `
  --out-dir .\outputs\scene_run
```

## Inspect a saved result

```powershell
python .\interactive_splat_viewer.py `
  --state .\outputs\demo\gaussian_state.npz `
  --metadata .\outputs\demo\gaussian_metadata.txt `
  --output-html .\outputs\demo\viewer.html
```

The viewer is now renderer-backed rather than a 3D scatter plot. It renders an
orbit sequence from the saved Gaussian state, then writes a lightweight HTML
player plus a sibling `*_frames/` folder of PNG images. Keep that frame folder
beside the HTML if you move the viewer elsewhere.

Useful viewer controls:

- `--num-frames`: number of orbit frames referenced by the HTML player
- `--width` / `--height`: rendered frame resolution
- `--supersample`: render above display resolution, then downsample for a cleaner viewer image
- `--mesh`: when you have an OBJ mesh prior, include a Plotly geometry panel beside the rendered orbit viewer
- `--radius-scale`: how tightly the orbit camera hugs the reconstruction
- `--crop-padding`: extra padding around the visible alpha footprint after auto-cropping. Use `0` to keep the looser default framing.
- `--render-tile-size`: tile size used while rendering the viewer frames
- `--render-support-scale`: screen-space support used by the viewer renderer
- `--render-alpha-threshold`: skips negligible splat contributions while generating viewer frames

If you want the viewer to stay closer to the older looser framing, keep `--crop-padding 0`.
If you want a tighter-quality crop, use:

```powershell
python .\interactive_splat_viewer.py `
  --state .\outputs\demo\gaussian_state.npz `
  --metadata .\outputs\demo\gaussian_metadata.txt `
  --num-frames 48 `
  --width 768 `
  --height 768 `
  --supersample 1.75 `
  --radius-scale 1.45 `
  --crop-padding 0.12 `
  --render-tile-size 64 `
  --output-html .\outputs\demo\viewer.html
```

If you want to keep the explicit mesh geometry visible in the same HTML:

```powershell
python .\interactive_splat_viewer.py `
  --state .\outputs\mesh_run\gaussian_state.npz `
  --metadata .\outputs\mesh_run\gaussian_metadata.txt `
  --mesh .\data\example.obj `
  --output-html .\outputs\mesh_run\viewer.html
```

## Build side-by-side comparisons

If one run folder already contains both `view_XX_mesh_prior.png` and
`view_XX_with_completion.png`, you can generate side-by-side comparison images
directly:

```powershell
python .\tools\compare_renders.py `
  --single-run-dir .\outputs\mesh_run `
  --output-dir .\outputs\mesh_run_comparison `
  --left-label "Mesh Prior" `
  --right-label "With Completion" `
  --contact-sheet
```

This writes one comparison image per matching view plus a single
`comparison_contact_sheet.png`.

## Current limitations

- The renderer uses simple isotropic screen-space splats, not full anisotropic 3DGS covariances.
- The renderer now uses tile-based culling and support-thresholding for memory, but it is still a simple research renderer rather than an optimized 3DGS implementation.
- The appearance prior is a lightweight prompt palette, not diffusion guidance.
- If COLMAP is not provided, training targets are synthetic orbit renders.
- Scene mode uses COLMAP sparse points plus heuristics for normals and completion seeding; it is not yet a learned scene-completion prior.
- Mesh mode still expects an external OBJ if you want explicit surface geometry.

## Next logical extensions

1. Replace the prompt-palette appearance prior with SDS or another diffusion-based objective.
2. Add real multi-view supervision instead of proxy views.
3. Export mesh-aligned Gaussian states in a more standard training/evaluation format.
4. Extend completion seeding with stronger hole and visibility reasoning.
