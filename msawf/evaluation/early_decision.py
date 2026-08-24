"""Approved confidence-and-stability early-recognition protocol."""

from __future__ import annotations

from dataclasses import replace

import torch

from msawf.constants import L_MAX, OMEGA

from .closed_world import ClosedWorldEvaluator
from .metrics import average_decision_length, average_observation_ratio, early_decision_rate
from .prediction import (
    EvaluationBatch,
    EvaluationResult,
    build_aggregate_record,
    build_prediction_records,
)

EARLY_DECISION_PROTOCOL = "msawf-early-decision-v1"


class EarlyDecisionEvaluator:
    def __init__(
        self,
        evaluator: ClosedWorldEvaluator,
        *,
        k: int = 5,
        confidence_threshold: float = 0.70,
        stability_threshold: float = 0.70,
    ) -> None:
        if (k, confidence_threshold, stability_threshold) != (5, 0.70, 0.70):
            raise ValueError("canonical early-decision settings are K=5, tau=0.70, delta=0.70")
        self.evaluator = evaluator
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold

    def evaluate(self, batch: EvaluationBatch) -> EvaluationResult:
        prefix_results = [
            self.evaluator.evaluate(batch, prefix_length=prefix_length)
            for prefix_length in OMEGA
        ]
        count = len(batch.trace_ids)
        chosen_probabilities: list[torch.Tensor | None] = [None] * count
        decision_lengths = [L_MAX] * count
        previous_sets: list[set[int] | None] = [None] * count
        for prefix_index, (prefix_length, result) in enumerate(zip(OMEGA, prefix_results)):
            for sample_index, record in enumerate(result.predictions):
                if chosen_probabilities[sample_index] is not None:
                    continue
                probabilities = torch.tensor(record.probabilities, dtype=torch.float32)
                top_values, top_indices = torch.topk(probabilities, k=self.k)
                current_set = set(int(index) for index in top_indices.tolist())
                if prefix_index > 0:
                    previous = previous_sets[sample_index]
                    if previous is None:
                        raise RuntimeError("previous Top-K set is missing")
                    union = current_set | previous
                    stability = len(current_set & previous) / len(union)
                    confidence = float(top_values.mean().item())
                    if confidence >= self.confidence_threshold and stability >= self.stability_threshold:
                        chosen_probabilities[sample_index] = probabilities
                        decision_lengths[sample_index] = prefix_length
                previous_sets[sample_index] = current_set
                if prefix_length == L_MAX and chosen_probabilities[sample_index] is None:
                    chosen_probabilities[sample_index] = probabilities
        if any(probabilities is None for probabilities in chosen_probabilities):
            raise RuntimeError("early-decision evaluation did not produce every prediction")
        selected = torch.stack(
            [probabilities for probabilities in chosen_probabilities if probabilities is not None]
        )
        extra = {
            "adl": average_decision_length(decision_lengths),
            "aor": average_observation_ratio(decision_lengths, max_length=L_MAX),
            "edr": early_decision_rate(decision_lengths, max_length=L_MAX),
        }
        engine = self.evaluator.engine
        base_records = build_prediction_records(
            batch=batch,
            probabilities=selected,
            checkpoint_digest=engine.checkpoint.content_digest,
            config_digest=engine.config_digest,
            condition="early-decision",
            prefix_length=L_MAX,
            decision_lengths=decision_lengths,
            method=engine.checkpoint.method_variant,
            protocol_version=EARLY_DECISION_PROTOCOL,
            result_status=engine.checkpoint.artifact_status,
        )
        records = tuple(
            replace(record, prefix_length=decision_lengths[index])
            for index, record in enumerate(base_records)
        )
        aggregate = build_aggregate_record(
            batch=batch,
            probabilities=selected,
            checkpoint_digest=engine.checkpoint.content_digest,
            config_digest=engine.config_digest,
            condition="early-decision",
            method=engine.checkpoint.method_variant,
            protocol_version=EARLY_DECISION_PROTOCOL,
            result_status=engine.checkpoint.artifact_status,
            extra_metrics=extra,
        )
        return EvaluationResult(records, aggregate)
