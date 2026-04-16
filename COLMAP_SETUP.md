# COLMAP Setup for This Repository

This project expects a COLMAP reconstruction exported as a text model:

- `cameras.txt`
- `images.txt`

and the corresponding image folder.

The easiest Windows workflow is:

1. Install COLMAP and confirm `COLMAP.bat -h` works.
2. Put your input photos in one folder.
3. Run the PowerShell helper in `tools/run_colmap.ps1`.
4. Train this repository with `--colmap-model-dir` and `--colmap-image-dir`.

## Recommended capture rules

- Take 20 to 80 overlapping images around the object.
- Keep the object centered and visible in adjacent views.
- Avoid strong motion blur.
- Avoid large exposure changes across views.
- Use a static background when possible.

## Windows install

Download the prebuilt Windows archive, extract it, and use `COLMAP.bat` from that folder.

Example:

```powershell
C:\tools\COLMAP\COLMAP.bat -h
```

## Run COLMAP with the helper

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_colmap.ps1 `
  -ColmapBat C:\tools\COLMAP\COLMAP.bat `
  -ImageDir .\data\my_object\images `
  -Workspace .\data\my_object\colmap
```

This creates:

- `database.db`
- `sparse\0\`
- `dense\`
- `sparse_txt\`

The directory `sparse_txt\` is what this repository reads.

## Train the hybrid model with COLMAP views

If you already have a coarse OBJ mesh:

```powershell
python .\main.py `
  --mesh .\data\my_object\mesh.obj `
  --colmap-model-dir .\data\my_object\colmap\sparse_txt `
  --colmap-image-dir .\data\my_object\images `
  --num-views 24 `
  --steps 300 `
  --out-dir .\outputs\my_object_colmap
```

## Reconstruct a scene directly from COLMAP sparse points

If you do not have a clean mesh and want to recreate under-observed parts of a scene, use scene mode:

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

This mode uses:

- sparse COLMAP points as anchored scene splats
- estimated normals from local geometry and camera layout
- low-support sparse points as completion candidates for missing regions

## Notes

- This repository reads COLMAP text exports, not binary `.bin` models.
- For best results, use undistorted images and the exported sparse text model from the COLMAP workspace.
- If you want explicit surface geometry, generate a dense mesh in COLMAP and convert it to OBJ in Blender or MeshLab before passing it with `--mesh`.
- Scene mode is heuristic scene completion, not a learned generative completion model.
