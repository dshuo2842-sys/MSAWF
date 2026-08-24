"""Fixed-window prefix construction and random multi-prefix sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from msawf.constants import L_MAX, OMEGA, PADDING_VALUE
from msawf.data import validate_trace


@dataclass(frozen=True)
class PrefixOperator:
    """Implement the paper's ``P_l`` operator for one trace or a batch."""

    max_length: int = L_MAX
    prefix_lengths: tuple[int, ...] = OMEGA

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if not self.prefix_lengths:
            raise ValueError("prefix_lengths must not be empty")
        if tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths:
            raise ValueError("prefix_lengths must be unique and increasing")
        if self.prefix_lengths[-1] > self.max_length:
            raise ValueError("prefix lengths must not exceed max_length")

    def __call__(self, trace: Tensor | Sequence[float], prefix_length: int) -> Tensor:
        if prefix_length not in self.prefix_lengths:
            raise ValueError(
                f"prefix_length must be one of {self.prefix_lengths}; got {prefix_length}"
            )
        tensor = trace if isinstance(trace, Tensor) else torch.as_tensor(trace)
        if tensor.ndim not in (1, 2):
            raise ValueError(
                f"trace must have shape [L] or [B,L]; got {tuple(tensor.shape)}"
            )

        rows = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
        for row in rows:
            validate_trace(row, max_length=self.max_length, require_fixed_length=False)

        output_shape = (*tensor.shape[:-1], self.max_length)
        output = torch.full(
            output_shape,
            fill_value=PADDING_VALUE,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        copy_length = min(prefix_length, tensor.shape[-1])
        output[..., :copy_length] = tensor[..., :copy_length]
        return output


@dataclass
class RandomPrefixSampler:
    """Sample one shared observation length for a future training batch."""

    prefix_lengths: tuple[int, ...] = OMEGA
    seed: int = 18
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths:
            raise ValueError("prefix_lengths must be unique and increasing")
        self._rng = random.Random(self.seed)

    def sample(self) -> int:
        return self._rng.choice(self.prefix_lengths)
