"""Unified-unmonitored-class open-world evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from msawf.constants import OPEN_WORLD_CLASSES, UNMONITORED_CLASS_INDEX

from .closed_world import ClosedWorldEvaluator
from .core import PredictionEngine
from .prediction import EvaluationBatch, EvaluationResult

OPEN_WORLD_PROTOCOL = "msawf-open-world-v1"


def encode_open_world_target(
    monitored_classes: Iterable[int], *, contains_unmonitored: bool
) -> Tensor:
    labels = torch.zeros(OPEN_WORLD_CLASSES, dtype=torch.float32)
    for class_index in monitored_classes:
        if not 0 <= class_index < UNMONITORED_CLASS_INDEX:
            raise ValueError("monitored class must be in [0,99]")
        labels[class_index] = 1
    if contains_unmonitored:
        labels[UNMONITORED_CLASS_INDEX] = 1
    if not labels.any().item():
        raise ValueError("open-world target must contain a monitored or unmonitored label")
    return labels


class OpenWorldEvaluator(ClosedWorldEvaluator):
    protocol_version = OPEN_WORLD_PROTOCOL

    def __init__(self, engine: PredictionEngine) -> None:
        if engine.expected_output_dim != OPEN_WORLD_CLASSES:
            raise ValueError("open-world evaluation requires 101 classifier outputs")
        self.engine = engine
        from msawf.augmentation import PrefixOperator

        self.prefix_operator = PrefixOperator()

    def evaluate(self, batch: EvaluationBatch, **kwargs: object) -> EvaluationResult:
        if batch.labels.shape[1] != OPEN_WORLD_CLASSES:
            raise ValueError("open-world labels must contain 101 outputs")
        if not torch.all(batch.labels.sum(dim=1) >= 1).item():
            raise ValueError("every open-world record requires at least one true label")
        return super().evaluate(batch, **kwargs)
