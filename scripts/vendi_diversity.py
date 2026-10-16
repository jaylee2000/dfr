"""Vendi Score helpers for grouped image features.

The training pipeline lays samples out as contiguous prompt groups.  This
module computes one order-1 Vendi Score per group from an encoder's feature
vectors; callers own distributed aggregation and logging.
"""

import numpy as np
import torch
from vendi_score import vendi


@torch.no_grad()
def grouped_vendi_scores(features, num_samples_per_prompt):
    """Return one cosine-kernel Vendi Score for every contiguous prompt group.

    ``vendi_score`` performs its eigendecomposition with NumPy/SciPy, so the
    detached features are intentionally moved to CPU.  Rows are normalized by
    ``score_X`` before it constructs the positive-semidefinite cosine Gram
    matrix.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        shape = getattr(features, "shape", None)
        raise ValueError(f"features must be a 2D tensor, got shape {shape}")
    group_size = int(num_samples_per_prompt)
    if group_size < 2:
        raise ValueError(
            f"num_samples_per_prompt must be >= 2, got {group_size}"
        )
    total = features.shape[0]
    if total % group_size != 0:
        raise ValueError(
            f"number of features ({total}) must be divisible by "
            f"num_samples_per_prompt ({group_size})"
        )
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")

    groups = (
        features.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .reshape(-1, group_size, features.shape[1])
    )
    scores = np.asarray(
        [vendi.score_X(group, q=1, normalize=True) for group in groups],
        dtype=np.float32,
    )
    return torch.from_numpy(scores).to(features.device)
