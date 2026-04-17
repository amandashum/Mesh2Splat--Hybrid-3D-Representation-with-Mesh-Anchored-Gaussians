"""Completion-specific priors and losses for hybrid Gaussian reconstruction.

This package isolates the missing-region logic from the rest of the pipeline:

- `seeding.py` decides where completion splats should start from.
- `losses.py` regularizes completion splats so they behave like surface
  continuation rather than arbitrary floating blobs.
"""

from .losses import completion_continuity_loss
from .seeding import CompletionPrior, build_mesh_completion_prior, build_sparse_completion_prior

__all__ = [
    "CompletionPrior",
    "build_mesh_completion_prior",
    "build_sparse_completion_prior",
    "completion_continuity_loss",
]
