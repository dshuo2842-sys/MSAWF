"""Minimal optimizer engine with no paper-specific objective logic."""

from __future__ import annotations

import torch
from torch import Tensor


class TrainingEngine:
    """Perform exactly one zero-grad/backward/step for a scalar loss."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.global_step = 0

    def step(self, loss: Tensor) -> int:
        if loss.ndim != 0:
            raise ValueError("training loss must be a scalar tensor")
        if not torch.isfinite(loss).item():
            raise ValueError("training loss must be finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return self.global_step
