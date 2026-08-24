"""Fixed-prefix evaluation with one unchanged model."""

from __future__ import annotations

from dataclasses import dataclass

from msawf.constants import OMEGA

from .closed_world import ClosedWorldEvaluator
from .prediction import EvaluationBatch, EvaluationResult

PREFIX_PROTOCOL = "msawf-fixed-prefix-v1"


@dataclass(frozen=True)
class FixedPrefixResult:
    by_prefix: tuple[tuple[int, EvaluationResult], ...]
    protocol_version: str = PREFIX_PROTOCOL


class FixedPrefixEvaluator:
    def __init__(self, evaluator: ClosedWorldEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(
        self, batch: EvaluationBatch, *, insertion_probability: float = 0.0
    ) -> FixedPrefixResult:
        return FixedPrefixResult(
            by_prefix=tuple(
                (
                    prefix_length,
                    self.evaluator.evaluate(
                        batch,
                        prefix_length=prefix_length,
                        insertion_probability=insertion_probability,
                    ),
                )
                for prefix_length in OMEGA
            )
        )
