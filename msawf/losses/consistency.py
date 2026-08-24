"""Feature consistency objective used by Stage I and Stage III."""

from __future__ import annotations

from torch import Tensor


def consistency_loss(clean_features: Tensor, perturbed_features: Tensor) -> Tensor:
    """Compute ``1/B * sum_i ||z_i - z_tilde_i||_2^2``."""

    if clean_features.shape != perturbed_features.shape:
        raise ValueError(
            "paired features must have the same shape; "
            f"got {tuple(clean_features.shape)} and {tuple(perturbed_features.shape)}"
        )
    if clean_features.ndim < 2:
        raise ValueError("features must include batch and feature dimensions")
    reduce_dims = tuple(range(1, clean_features.ndim))
    return (clean_features - perturbed_features).pow(2).sum(dim=reduce_dims).mean()
