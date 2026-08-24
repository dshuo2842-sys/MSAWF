"""Tiny deterministic records shared by CPU-only data protocol tests."""

from __future__ import annotations

from itertools import combinations

import torch

from msawf.constants import L_MAX
from msawf.data import TraceRecord


def make_record(
    trace_id: str,
    active_labels: tuple[int, ...],
    *,
    num_classes: int,
    length: int = 8,
    root_provenance_id: str | None = None,
    domain: str = "target-real",
    source_provenance: str = "tiny-fixture-v1",
    phase: int = 0,
) -> TraceRecord:
    if not 0 < length <= L_MAX:
        raise ValueError("fixture length must be within L_max")
    observed = [1 if (index + phase) % 2 == 0 else -1 for index in range(length)]
    trace = torch.tensor(observed + [0] * (L_MAX - length), dtype=torch.float32)
    labels = torch.zeros(num_classes, dtype=torch.float32)
    labels[list(active_labels)] = 1
    return TraceRecord(
        trace=trace,
        labels=labels,
        trace_id=trace_id,
        domain=domain,
        root_provenance_id=root_provenance_id or f"root-{trace_id}",
        source_provenance=source_provenance,
    )


def make_single_tab_records(
    *, class_count: int = 10, traces_per_class: int = 2, length: int = 8
) -> tuple[TraceRecord, ...]:
    return tuple(
        make_record(
            f"single-c{class_id}-r{replica}",
            (class_id,),
            num_classes=class_count,
            length=length,
            phase=class_id + replica,
            domain="source-single-tab",
        )
        for class_id in range(class_count)
        for replica in range(traces_per_class)
    )


def make_pair_records(
    *, class_count: int = 6, replicas_per_pair: int = 2
) -> tuple[TraceRecord, ...]:
    records: list[TraceRecord] = []
    for left, right in combinations(range(class_count), 2):
        for replica in range(replicas_per_pair):
            records.append(
                make_record(
                    f"pair-{left}-{right}-r{replica}",
                    (left, right),
                    num_classes=class_count,
                    phase=left + right + replica,
                )
            )
    return tuple(records)
