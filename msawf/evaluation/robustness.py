"""Insertion-noise robustness evaluation on one immutable query set."""

from __future__ import annotations

from dataclasses import dataclass

from .closed_world import ClosedWorldEvaluator
from .metrics import degradation_rate
from .prediction import EvaluationBatch, EvaluationResult

ROBUSTNESS_PROTOCOL = "msawf-insertion-robustness-v1"
CANONICAL_NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class RobustnessResult:
    by_probability: tuple[tuple[float, EvaluationResult], ...]
    degradation_rate: float
    protocol_version: str = ROBUSTNESS_PROTOCOL


class RobustnessEvaluator:
    def __init__(self, evaluator: ClosedWorldEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, batch: EvaluationBatch) -> RobustnessResult:
        results = tuple(
            (probability, self.evaluator.evaluate(batch, insertion_probability=probability))
            for probability in CANONICAL_NOISE_LEVELS
        )
        metrics = {probability: result.aggregate.metrics for probability, result in results}
        return RobustnessResult(
            by_probability=results,
            degradation_rate=degradation_rate(metrics[0.0]["f1"], metrics[0.3]["f1"]),
        )
