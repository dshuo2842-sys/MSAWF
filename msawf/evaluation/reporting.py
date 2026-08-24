"""Shared method and cumulative-ablation report composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from msawf.methods import get_method_plan

from .metrics import split_mean_std
from .prediction import AggregateRecord


@dataclass(frozen=True)
class MethodComparisonReport:
    entries: tuple[AggregateRecord, ...]
    manifest_digest: str
    split_id: str
    condition: str


def cumulative_ablation_graph(level: int) -> tuple[str, ...]:
    if level not in (0, 1, 2, 3):
        raise ValueError("ablation level must be 0, 1, 2, or 3")
    return ("baseline", "stage1", "stage2", "stage3")[: level + 1]


def build_method_comparison(entries: Sequence[AggregateRecord]) -> MethodComparisonReport:
    if not entries:
        raise ValueError("method comparison requires at least one entry")
    for entry in entries:
        get_method_plan(entry.method)
    manifests = {entry.manifest_digest for entry in entries}
    splits = {entry.split_id for entry in entries}
    conditions = {entry.condition for entry in entries}
    if len(manifests) != 1 or len(splits) != 1 or len(conditions) != 1:
        raise ValueError("method comparison requires one shared manifest, split, and condition")
    return MethodComparisonReport(
        entries=tuple(entries),
        manifest_digest=next(iter(manifests)),
        split_id=next(iter(splits)),
        condition=next(iter(conditions)),
    )


def aggregate_metric_across_splits(
    entries: Sequence[AggregateRecord], metric: str, *, correction: int
) -> tuple[float, float]:
    return split_mean_std([entry.metrics[metric] for entry in entries], correction=correction)
