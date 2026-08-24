"""Deterministic multi-label K-shot support construction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from msawf.constants import MULTILABEL_KSHOT_PROTOCOL
from msawf.data.manifest import CoverageStatistics, fingerprint_records
from msawf.data.schema import TraceRecord
from msawf.utils.hashing import stable_digest


class KShotProtocolError(ValueError):
    """Raised when candidate records violate the canonical K-shot contract."""


class InfeasibleKShotError(KShotProtocolError):
    """Raised when one or more monitored classes cannot reach K coverage."""

    def __init__(self, insufficient_classes: Sequence[tuple[int, int]], k: int) -> None:
        self.insufficient_classes = tuple(insufficient_classes)
        details = ", ".join(
            f"class {class_id}: {available}/{k}"
            for class_id, available in self.insufficient_classes
        )
        super().__init__(f"K-shot coverage is infeasible ({details})")


@dataclass(frozen=True)
class KShotSelection:
    """Support/query membership and complete selection audit data."""

    split_id: str
    seed: int
    k: int
    support: tuple[TraceRecord, ...]
    query: tuple[TraceRecord, ...]
    statistics: CoverageStatistics

    @property
    def support_trace_ids(self) -> tuple[str, ...]:
        return tuple(record.trace_id for record in self.support)

    @property
    def query_trace_ids(self) -> tuple[str, ...]:
        return tuple(record.trace_id for record in self.query)


def _validate_candidates(
    records: Sequence[TraceRecord],
    monitored_classes: Sequence[int],
    k: int,
    seed: int,
    split_id: str,
    dataset_fingerprint: str,
) -> tuple[tuple[TraceRecord, ...], tuple[int, ...]]:
    canonical_records = tuple(sorted(records, key=lambda record: record.trace_id))
    classes = tuple(monitored_classes)
    if not canonical_records:
        raise KShotProtocolError("target candidate records must not be empty")
    if classes != tuple(sorted(set(classes))) or any(class_id < 0 for class_id in classes):
        raise KShotProtocolError(
            "monitored_classes must be unique, sorted, non-negative indices"
        )
    if not classes:
        raise KShotProtocolError("monitored_classes must not be empty")
    if k <= 0:
        raise KShotProtocolError("K must be positive")
    if seed < 0 or seed >= 2**64:
        raise KShotProtocolError("seed must be an unsigned 64-bit integer")
    if not split_id.strip():
        raise KShotProtocolError("split_id must not be empty")
    trace_ids = [record.trace_id for record in canonical_records]
    root_ids = [record.root_provenance_id for record in canonical_records]
    if len(trace_ids) != len(set(trace_ids)):
        raise KShotProtocolError("target candidates contain duplicated trace_id")
    if len(root_ids) != len(set(root_ids)):
        raise KShotProtocolError(
            "target candidates contain duplicated root provenance; canonical "
            "support/query selection requires one record per root"
        )
    maximum_class = max(classes)
    for record in canonical_records:
        if record.labels.numel() <= maximum_class:
            raise KShotProtocolError(
                f"record {record.trace_id} label dimension excludes a monitored class"
            )
    actual_fingerprint = fingerprint_records(canonical_records)
    if actual_fingerprint != dataset_fingerprint:
        raise KShotProtocolError(
            "dataset fingerprint mismatch: "
            f"expected {dataset_fingerprint}, actual {actual_fingerprint}"
        )
    return canonical_records, classes


def _monitored_labels(
    record: TraceRecord, monitored_set: frozenset[int]
) -> frozenset[int]:
    return frozenset(record.active_label_indices).intersection(monitored_set)


def select_multilabel_kshot(
    records: Sequence[TraceRecord],
    monitored_classes: Sequence[int],
    *,
    k: int,
    seed: int,
    split_id: str,
    dataset_fingerprint: str,
) -> KShotSelection:
    """Apply ``multilabel-kshot-v1`` and return an inclusion-minimal support."""

    canonical_records, classes = _validate_candidates(
        records, monitored_classes, k, seed, split_id, dataset_fingerprint
    )
    monitored_set = frozenset(classes)
    labels_by_id = {
        record.trace_id: _monitored_labels(record, monitored_set)
        for record in canonical_records
    }
    available = {
        class_id: sum(
            class_id in labels_by_id[record.trace_id] for record in canonical_records
        )
        for class_id in classes
    }
    insufficient = tuple(
        (class_id, available[class_id])
        for class_id in classes
        if available[class_id] < k
    )
    if insufficient:
        raise InfeasibleKShotError(insufficient, k)

    deficits = {class_id: k for class_id in classes}
    remaining = list(canonical_records)
    selection_order: list[TraceRecord] = []
    while any(deficit > 0 for deficit in deficits.values()):
        deficient_classes = {
            class_id for class_id, deficit in deficits.items() if deficit > 0
        }
        scarcity_counts = {
            class_id: sum(
                class_id in labels_by_id[record.trace_id] for record in remaining
            )
            for class_id in deficient_classes
        }

        def rank(record: TraceRecord) -> tuple[object, ...]:
            record_labels = labels_by_id[record.trace_id]
            deficient_covered = record_labels.intersection(deficient_classes)
            total_deficit = sum(deficits[class_id] for class_id in deficient_covered)
            scarcity = sum(
                (
                    Fraction(1, scarcity_counts[class_id])
                    for class_id in deficient_covered
                ),
                start=Fraction(0, 1),
            )
            satisfied_labels = sum(
                deficits[class_id] == 0 for class_id in record_labels
            )
            tie_rank = stable_digest(
                "multilabel-kshot-v1/tie-rank",
                dataset_fingerprint,
                seed,
                split_id,
                record.trace_id,
            )
            return (
                -len(deficient_covered),
                -total_deficit,
                -scarcity,
                satisfied_labels,
                tie_rank,
                record.trace_id,
            )

        selected = min(remaining, key=rank)
        covered = labels_by_id[selected.trace_id].intersection(deficient_classes)
        if not covered:
            raise KShotProtocolError(
                "greedy selection stalled despite passing feasibility validation"
            )
        selection_order.append(selected)
        remaining = [
            record for record in remaining if record.trace_id != selected.trace_id
        ]
        for class_id in covered:
            deficits[class_id] -= 1

    coverage = {
        class_id: sum(
            class_id in labels_by_id[record.trace_id] for record in selection_order
        )
        for class_id in classes
    }
    retained_ids = {record.trace_id for record in selection_order}
    pruned_ids: list[str] = []
    for record in reversed(selection_order):
        record_labels = labels_by_id[record.trace_id]
        if all(
            coverage[class_id] - (class_id in record_labels) >= k
            for class_id in classes
        ):
            retained_ids.remove(record.trace_id)
            pruned_ids.append(record.trace_id)
            for class_id in record_labels:
                coverage[class_id] -= 1

    support = tuple(
        record for record in selection_order if record.trace_id in retained_ids
    )
    query = tuple(
        record for record in canonical_records if record.trace_id not in retained_ids
    )
    support_trace_ids = {record.trace_id for record in support}
    query_trace_ids = {record.trace_id for record in query}
    support_roots = {record.root_provenance_id for record in support}
    query_roots = {record.root_provenance_id for record in query}
    if support_trace_ids.intersection(query_trace_ids):
        raise KShotProtocolError("support/query trace_id overlap after selection")
    if support_roots.intersection(query_roots):
        raise KShotProtocolError("support/query root provenance overlap after selection")

    per_class = tuple((class_id, coverage[class_id]) for class_id in classes)
    coverage_values = [count for _, count in per_class]
    statistics = CoverageStatistics(
        support_record_count=len(support),
        query_record_count=len(query),
        per_class_coverage=per_class,
        min_coverage=min(coverage_values),
        max_coverage=max(coverage_values),
        mean_coverage=sum(coverage_values) / len(coverage_values),
        selected_trace_ids=tuple(record.trace_id for record in support),
        selection_order=tuple(record.trace_id for record in selection_order),
        pruned_trace_ids=tuple(pruned_ids),
    )
    return KShotSelection(
        split_id=split_id,
        seed=seed,
        k=k,
        support=support,
        query=query,
        statistics=statistics,
    )


__all__ = [
    "InfeasibleKShotError",
    "KShotProtocolError",
    "KShotSelection",
    "MULTILABEL_KSHOT_PROTOCOL",
    "select_multilabel_kshot",
]
