# Hybrid Mesh + Gaussian Splatting Baseline

This repository is a reduced implementation of the project proposal in `CMPT 743_ 3D Reconstruction.docx`: a hybrid Gaussian scene/model reconstruction pipeline with three Gaussian branches:

- anchored splats tied to a mesh prior or COLMAP sparse scene prior
- detail splats for local appearance variation
- completion splats for weakly constrained missing or under-observed regions

The repository is intentionally scoped to the core hybrid baseline. Side experiments, generated outputs, local environments, and image-to-3D wrappers have been removed.

## Repository layout

- `main.py`: entrypoint
- `hybrid_gs/`: core mesh, camera, Gaussian, renderer, loss, and optimization code
- `interactive_splat_viewer.py`: optional HTML viewer for saved Gaussian states
- `COLMAP_SETUP.md`: Windows-oriented COLMAP setup notes for multi-view reconstruction
- `tools/run_colmap.ps1`: helper script for a standard COLMAP sparse pipeline
- `data/`: place optional input meshes or reference images here
- `checkpoints/`: reserved for future external model checkpoints

## What the baseline does

1. Load an OBJ mesh, create a primitive mesh, or use a COLMAP sparse scene prior
2. Optionally load real multi-view cameras and targets from a COLMAP text export
3. Build anchor, detail, and completion seeds from the mesh or sparse scene points
4. Initialize anchored, detail, and completion Gaussian branches
5. Optimize Gaussian parameters with reconstruction, tether, completion, appearance, scale, and opacity losses
6. Save rendered views plus a `gaussian_state.npz` snapshot for inspection

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
- `--steps`: optimization length
- `--cpu`: force CPU execution

## COLMAP workflow

This repository can train against a COLMAP sparse reconstruction exported to TXT. The typical flow is:

1. Capture overlapping object images.
2. Run COLMAP to estimate poses.
3. Export the sparse model to text.
4. For object-centric reconstruction, optionally pass an OBJ mesh into `main.py`.
5. For scene reconstruction, run `main.py --scene-mode` directly on the sparse model and image folder.

See `COLMAP_SETUP.md` for the Windows setup path and the included `tools/run_colmap.ps1` helper.

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

If you also have the source mesh:

```powershell
python .\interactive_splat_viewer.py `
  --state .\outputs\mesh_run\gaussian_state.npz `
  --metadata .\outputs\mesh_run\gaussian_metadata.txt `
  --mesh .\data\example.obj `
  --output-html .\outputs\mesh_run\viewer.html `
  --show-wireframe
```

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
