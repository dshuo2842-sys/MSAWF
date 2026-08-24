"""Stage II progressive source-to-target loss primitives."""

from __future__ import annotations

from torch import Tensor


def bridge_alpha(epoch: int, total_epochs: int) -> float:
    """Return the paper's one-indexed linear bridge coefficient."""

    if total_epochs < 2:
        raise ValueError("total_epochs must be at least 2")
    if not 1 <= epoch <= total_epochs:
        raise ValueError(f"epoch must be in [1,{total_epochs}]; got {epoch}")
    return 1.0 - ((epoch - 1) / (total_epochs - 1))


def bridge_loss(source_loss: Tensor, target_loss: Tensor, alpha: float) -> Tensor:
    """Compute ``alpha * L_s + (1-alpha) * L_t``."""

    if source_loss.ndim != 0 or target_loss.ndim != 0:
        raise ValueError("source_loss and target_loss must be scalar tensors")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * source_loss + (1.0 - alpha) * target_loss
