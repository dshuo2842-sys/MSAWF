"""Versioned immutable evaluation batches, predictions, and aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from msawf.constants import L_MAX
from msawf.data import validate_labels, validate_trace

from .metrics import evaluate_multilabel_metrics, top_k_sets

EVALUATION_RESULT_SCHEMA = "msawf-evaluation-result-v1"
_RESULT_STATUSES = {"measured", "historical", "paper_stated", "synthetic_smoke", "illustrative"}


@dataclass(frozen=True)
class EvaluationBatch:
    traces: Tensor
    labels: Tensor
    trace_ids: tuple[str, ...]
    root_provenance_ids: tuple[str, ...]
    roles: tuple[str, ...]
    split_id: str
    dataset_fingerprint: str
    manifest_digest: str

    def __post_init__(self) -> None:
        if self.traces.ndim != 2 or self.traces.shape[1] != L_MAX:
            raise ValueError(f"evaluation traces must have shape [N,{L_MAX}]")
        if self.labels.ndim != 2 or self.labels.shape[0] != self.traces.shape[0]:
            raise ValueError("evaluation labels must have shape [N,C]")
        count = self.traces.shape[0]
        if count == 0 or {len(self.trace_ids), len(self.root_provenance_ids), len(self.roles)} != {count}:
            raise ValueError("evaluation identity metadata must match a non-empty batch")
        if len(set(self.trace_ids)) != count:
            raise ValueError("evaluation batch contains duplicated trace IDs")
        if set(self.roles) != {"query"}:
            raise PermissionError("evaluators accept query records only")
        if not all((self.split_id, self.dataset_fingerprint, self.manifest_digest)):
            raise ValueError("split, dataset, and manifest identity are required")
        for trace in self.traces:
            validate_trace(trace, max_length=L_MAX, require_fixed_length=True)
        for labels in self.labels:
            validate_labels(labels, num_classes=self.labels.shape[1])

    def to(self, device: torch.device | str) -> "EvaluationBatch":
        return EvaluationBatch(
            traces=self.traces.to(device),
            labels=self.labels.to(device),
            trace_ids=self.trace_ids,
            root_provenance_ids=self.root_provenance_ids,
            roles=self.roles,
            split_id=self.split_id,
            dataset_fingerprint=self.dataset_fingerprint,
            manifest_digest=self.manifest_digest,
        )


@dataclass(frozen=True)
class PredictionRecord:
    trace_id: str
    split_id: str
    checkpoint_digest: str
    config_digest: str
    manifest_digest: str
    condition: str
    prefix_length: int
    decision_length: int | None
    probabilities: tuple[float, ...]
    predicted_set: tuple[int, ...]
    top_k_set: tuple[int, ...]
    true_set: tuple[int, ...]
    method: str
    protocol_version: str
    result_status: str
    schema_version: str = EVALUATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.result_status not in _RESULT_STATUSES:
            raise ValueError("invalid evaluation result status")
        if self.prefix_length <= 0 or self.decision_length is not None and self.decision_length <= 0:
            raise ValueError("observation lengths must be positive")


@dataclass(frozen=True)
class AggregateRecord:
    method: str
    protocol_version: str
    split_id: str
    checkpoint_digest: str
    config_digest: str
    manifest_digest: str
    condition: str
    metric_values: tuple[tuple[str, float], ...]
    split_statistics: tuple[tuple[str, float, float, int], ...]
    sample_count: int
    result_status: str
    schema_version: str = EVALUATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.result_status not in _RESULT_STATUSES:
            raise ValueError("invalid aggregate sample count or status")

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.metric_values)


@dataclass(frozen=True)
class EvaluationResult:
    predictions: tuple[PredictionRecord, ...]
    aggregate: AggregateRecord


def build_prediction_records(
    *,
    batch: EvaluationBatch,
    probabilities: Tensor,
    checkpoint_digest: str,
    config_digest: str,
    condition: str,
    prefix_length: int,
    method: str,
    protocol_version: str,
    result_status: str,
    threshold: float = 0.5,
    k: int = 5,
    decision_lengths: Sequence[int | None] | None = None,
) -> tuple[PredictionRecord, ...]:
    if probabilities.shape != batch.labels.shape:
        raise ValueError("probability shape does not match evaluation labels")
    decisions = tuple(decision_lengths or (None,) * len(batch.trace_ids))
    if len(decisions) != len(batch.trace_ids):
        raise ValueError("decision lengths must match evaluation records")
    top_sets = top_k_sets(probabilities, k=k)
    rows = probabilities.detach().cpu()
    targets = batch.labels.detach().cpu()
    records: list[PredictionRecord] = []
    for index, trace_id in enumerate(batch.trace_ids):
        predicted = tuple(torch.nonzero(rows[index] >= threshold, as_tuple=False).flatten().tolist())
        truth = tuple(torch.nonzero(targets[index] == 1, as_tuple=False).flatten().tolist())
        records.append(
            PredictionRecord(
                trace_id=trace_id,
                split_id=batch.split_id,
                checkpoint_digest=checkpoint_digest,
                config_digest=config_digest,
                manifest_digest=batch.manifest_digest,
                condition=condition,
                prefix_length=prefix_length,
                decision_length=decisions[index],
                probabilities=tuple(float(value) for value in rows[index].tolist()),
                predicted_set=tuple(int(value) for value in predicted),
                top_k_set=top_sets[index],
                true_set=tuple(int(value) for value in truth),
                method=method,
                protocol_version=protocol_version,
                result_status=result_status,
            )
        )
    return tuple(records)


def build_aggregate_record(
    *,
    batch: EvaluationBatch,
    probabilities: Tensor,
    checkpoint_digest: str,
    config_digest: str,
    condition: str,
    method: str,
    protocol_version: str,
    result_status: str,
    extra_metrics: Mapping[str, float] | None = None,
    threshold: float = 0.5,
    k: int = 5,
) -> AggregateRecord:
    metrics = evaluate_multilabel_metrics(probabilities, batch.labels, threshold=threshold, k=k).to_dict(k=k)
    metrics.update(extra_metrics or {})
    return AggregateRecord(
        method=method,
        protocol_version=protocol_version,
        split_id=batch.split_id,
        checkpoint_digest=checkpoint_digest,
        config_digest=config_digest,
        manifest_digest=batch.manifest_digest,
        condition=condition,
        metric_values=tuple(sorted((name, float(value)) for name, value in metrics.items())),
        split_statistics=(),
        sample_count=len(batch.trace_ids),
        result_status=result_status,
    )
