"""Baseline hybrid mesh-plus-Gaussian-splatting package.

The package is organized around a small end-to-end training pipeline:

- `camera.py` defines the camera model used by both synthetic orbit views and
  imported COLMAP reconstructions.
- `mesh.py` provides explicit geometry priors and mesh surface sampling.
- `completion/` isolates hole-region seeding and completion-specific losses.
- `colmap.py` reads real multi-view scenes exported by COLMAP.
- `gaussians.py` defines the anchored/detail/completion Gaussian branches.
- `renderer.py` turns Gaussian states into differentiable image predictions.
- `losses.py` contains the regularizers and reconstruction objectives.
- `pipeline.py` connects everything into one train-and-save workflow.
"""
