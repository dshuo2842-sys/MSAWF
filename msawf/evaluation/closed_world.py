"""Closed-world clean and insertion-noise evaluation."""

from __future__ import annotations

import torch

from msawf.augmentation import InsertionTransform, PrefixOperator
from msawf.constants import CLOSED_WORLD_CLASSES, L_MAX
from msawf.utils import derive_seed, make_generator

from .core import PredictionEngine
from .prediction import (
    EvaluationBatch,
    EvaluationResult,
    build_aggregate_record,
    build_prediction_records,
)

CLOSED_WORLD_PROTOCOL = "msawf-closed-world-v1"


class ClosedWorldEvaluator:
    """Apply one immutable checkpoint to query records without optimization."""

    protocol_version = CLOSED_WORLD_PROTOCOL

    def __init__(self, engine: PredictionEngine) -> None:
        if engine.expected_output_dim != CLOSED_WORLD_CLASSES:
            raise ValueError("closed-world evaluation requires 100 classifier outputs")
        self.engine = engine
        self.prefix_operator = PrefixOperator()

    def _condition_traces(
        self,
        batch: EvaluationBatch,
        *,
        prefix_length: int,
        insertion_probability: float,
    ) -> torch.Tensor:
        if insertion_probability not in (0.0, 0.1, 0.2, 0.3):
            raise ValueError("evaluation insertion probability must be one of 0/0.1/0.2/0.3")
        prefixed = self.prefix_operator(batch.traces, prefix_length)
        if insertion_probability == 0.0:
            return prefixed
        transform = InsertionTransform(insertion_probability)
        rows: list[torch.Tensor] = []
        for index, root_id in enumerate(batch.root_provenance_ids):
            seed = derive_seed(
                "evaluation-insertion-v1",
                self.engine.checkpoint.content_digest,
                batch.manifest_digest,
                batch.split_id,
                root_id,
                prefix_length,
                insertion_probability,
            )
            result = transform(
                prefixed[index], batch.labels[index], generator=make_generator(seed)
            )
            if not torch.equal(result.labels, batch.labels[index]):
                raise RuntimeError("evaluation insertion changed a label")
            rows.append(result.trace)
        return torch.stack(rows)

    def evaluate(
        self,
        batch: EvaluationBatch,
        *,
        prefix_length: int = L_MAX,
        insertion_probability: float = 0.0,
    ) -> EvaluationResult:
        self.engine.validate_batch(batch)
        traces = self._condition_traces(
            batch,
            prefix_length=prefix_length,
            insertion_probability=insertion_probability,
        )
        probabilities = self.engine.infer(traces)
        condition = "clean" if insertion_probability == 0 else f"insertion-p{insertion_probability:g}"
        status = self.engine.checkpoint.artifact_status
        records = build_prediction_records(
            batch=batch,
            probabilities=probabilities,
            checkpoint_digest=self.engine.checkpoint.content_digest,
            config_digest=self.engine.config_digest,
            condition=condition,
            prefix_length=prefix_length,
            method=self.engine.checkpoint.method_variant,
            protocol_version=self.protocol_version,
            result_status=status,
        )
        aggregate = build_aggregate_record(
            batch=batch,
            probabilities=probabilities,
            checkpoint_digest=self.engine.checkpoint.content_digest,
            config_digest=self.engine.config_digest,
            condition=condition,
            method=self.engine.checkpoint.method_variant,
            protocol_version=self.protocol_version,
            result_status=status,
        )
        return EvaluationResult(records, aggregate)
