"""Numerically stable implementation of the paper's multi-label BCE."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return batch-and-class mean BCE from logits."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape; "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.ndim != 2:
        raise ValueError("classification tensors must have shape [B,C]")
    if not (((targets == 0) | (targets == 1)).all().item()):
        raise ValueError("targets must contain binary multi-hot values")
    return F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="mean"
    )
