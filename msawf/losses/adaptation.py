"""Stage III feature-alignment and robustness objective primitives."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .classification import classification_loss
from .consistency import consistency_loss


def feature_alignment_loss(clean_features: Tensor, perturbed_features: Tensor) -> Tensor:
    """Use the approved feature ``z`` and batch-mean squared L2 alignment."""

    return consistency_loss(clean_features, perturbed_features)


@dataclass(frozen=True)
class StageThreeLosses:
    clean_classification: Tensor
    perturbed_classification: Tensor
    alignment: Tensor
    robust: Tensor
    total: Tensor


def stage_three_objective(
    *,
    clean_logits: Tensor,
    perturbed_logits: Tensor,
    targets: Tensor,
    clean_features: Tensor,
    perturbed_features: Tensor,
    lambda_align: float,
    lambda_rob: float,
) -> StageThreeLosses:
    """Return every named component of the paper's Stage III objective."""

    if lambda_align < 0 or lambda_rob < 0:
        raise ValueError("loss coefficients must be non-negative")
    clean = classification_loss(clean_logits, targets)
    perturbed = classification_loss(perturbed_logits, targets)
    alignment = feature_alignment_loss(clean_features, perturbed_features)
    robust = perturbed + lambda_align * alignment
    total = clean + lambda_rob * robust
    return StageThreeLosses(
        clean_classification=clean,
        perturbed_classification=perturbed,
        alignment=alignment,
        robust=robust,
        total=total,
    )
