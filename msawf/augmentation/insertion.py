"""Insertion-like direction perturbation with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from msawf.constants import L_MAX, PADDING_VALUE
from msawf.data import observed_length, validate_labels, validate_trace


@dataclass(frozen=True)
class InsertionResult:
    """Perturbed view plus labels and origin indices for validation.

    ``origin_indices`` contains an original packet index for genuine packets,
    ``-1`` for inserted dummy packets, and ``-2`` for final padding.
    """

    trace: Tensor
    labels: Tensor
    origin_indices: Tensor

    @property
    def dummy_mask(self) -> Tensor:
        return self.origin_indices == -1

    @property
    def padding_mask(self) -> Tensor:
        return self.origin_indices == -2


class InsertionTransform:
    """Insert at most one dummy immediately before each selected real packet.

    Every genuine position is selected independently with probability ``p``.
    Dummy directions are sampled equiprobably from ``{-1,+1}``. The result is
    truncated or zero-padded to ``max_length``. This makes the implemented
    insertion realization explicit while preserving genuine-packet order.
    """

    def __init__(
        self,
        probability: float,
        *,
        max_length: int = L_MAX,
        seed: int = 18,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.probability = float(probability)
        self.max_length = max_length
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def __call__(
        self,
        trace: Tensor | Sequence[float],
        labels: Tensor | Sequence[float],
        *,
        generator: torch.Generator | None = None,
    ) -> InsertionResult:
        tensor = validate_trace(
            trace, max_length=self.max_length, require_fixed_length=False
        )
        label_tensor = validate_labels(labels)
        length = observed_length(tensor)
        genuine_values = tensor[:length].detach().cpu().tolist()
        rng = generator if generator is not None else self._generator

        insert_mask = torch.rand(length, generator=rng, device="cpu") < self.probability
        dummy_count = int(insert_mask.sum().item())
        dummy_values = (
            torch.randint(0, 2, (dummy_count,), generator=rng, device="cpu") * 2 - 1
        ).tolist()

        values: list[float] = []
        origins: list[int] = []
        dummy_index = 0
        for original_index, value in enumerate(genuine_values):
            if bool(insert_mask[original_index].item()):
                values.append(dummy_values[dummy_index])
                origins.append(-1)
                dummy_index += 1
            values.append(value)
            origins.append(original_index)

        values = values[: self.max_length]
        origins = origins[: self.max_length]
        result_trace = torch.full(
            (self.max_length,),
            fill_value=PADDING_VALUE,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        origin_indices = torch.full(
            (self.max_length,), -2, dtype=torch.long, device=tensor.device
        )
        if values:
            result_trace[: len(values)] = torch.as_tensor(
                values, dtype=tensor.dtype, device=tensor.device
            )
            origin_indices[: len(origins)] = torch.as_tensor(
                origins, dtype=torch.long, device=tensor.device
            )

        validate_trace(result_trace, max_length=self.max_length, require_fixed_length=True)
        return InsertionResult(
            trace=result_trace,
            labels=label_tensor.to(device=tensor.device).clone(),
            origin_indices=origin_indices,
        )
