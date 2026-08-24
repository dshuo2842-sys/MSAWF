"""Canonical ``synthetic-five-tab-v1`` corpus construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from msawf.constants import (
    FIVE_TAB_SIZE,
    L_MAX,
    SYNTHETIC_FIVE_TAB_PROTOCOL,
)
from msawf.data.manifest import (
    Manifest,
    SyntheticManifestEntry,
    fingerprint_records,
    sequence_digest,
)
from msawf.data.schema import TraceRecord
from msawf.utils.hashing import StableRNG, derive_seed, stable_digest


class SyntheticProtocolError(ValueError):
    """Raised when inputs cannot satisfy ``synthetic-five-tab-v1``."""


@dataclass(frozen=True)
class SyntheticSample:
    """One generated tensor plus packet-level constituent provenance."""

    trace: Tensor
    labels: Tensor
    origin_constituent_indices: Tensor
    manifest_entry: SyntheticManifestEntry


@dataclass(frozen=True)
class SyntheticCorpus:
    """Generated samples and their immutable manifest."""

    samples: tuple[SyntheticSample, ...]
    manifest: Manifest
    traces_per_class: int


def _validate_inputs(
    records: Sequence[TraceRecord],
    monitored_classes: Sequence[int],
    dataset_fingerprint: str,
    max_length: int,
) -> tuple[tuple[TraceRecord, ...], tuple[int, ...]]:
    canonical_records = tuple(records)
    classes = tuple(monitored_classes)
    if not canonical_records:
        raise SyntheticProtocolError("single-tab records must not be empty")
    if max_length != L_MAX:
        raise SyntheticProtocolError(f"canonical max_length must be {L_MAX}")
    if len(classes) < FIVE_TAB_SIZE or len(classes) % FIVE_TAB_SIZE != 0:
        raise SyntheticProtocolError(
            "monitored class count must be at least five and divisible by five"
        )
    if classes != tuple(range(len(classes))):
        raise SyntheticProtocolError(
            "monitored classes must be contiguous zero-based class indices"
        )
    trace_ids = [record.trace_id for record in canonical_records]
    root_ids = [record.root_provenance_id for record in canonical_records]
    if len(trace_ids) != len(set(trace_ids)):
        raise SyntheticProtocolError("single-tab records contain duplicated trace_id")
    if len(root_ids) != len(set(root_ids)):
        raise SyntheticProtocolError(
            "single-tab records contain duplicated root provenance"
        )
    for record in canonical_records:
        if record.labels.numel() != len(classes):
            raise SyntheticProtocolError(
                f"record {record.trace_id} label dimension does not match class schema"
            )
        active = record.active_label_indices
        if len(active) != 1 or active[0] not in classes:
            raise SyntheticProtocolError(
                f"record {record.trace_id} must contain one monitored label"
            )
        if record.observed_length <= 0:
            raise SyntheticProtocolError(
                f"record {record.trace_id} has no observed packets"
            )
    actual_fingerprint = fingerprint_records(canonical_records)
    if actual_fingerprint != dataset_fingerprint:
        raise SyntheticProtocolError(
            "dataset fingerprint mismatch: "
            f"expected {dataset_fingerprint}, actual {actual_fingerprint}"
        )
    return canonical_records, classes


def _interleave(
    records: tuple[TraceRecord, ...], sample_seed: int, max_length: int
) -> tuple[Tensor, Tensor, tuple[int, ...], bool]:
    observed_sequences = tuple(
        record.trace[: record.observed_length].to(dtype=torch.float32)
        for record in records
    )
    positions = [0] * len(records)
    original_lengths = tuple(sequence.numel() for sequence in observed_sequences)
    output: list[float] = []
    origins: list[int] = []
    rng = StableRNG(sample_seed)

    while len(output) < max_length:
        remaining = [
            original_lengths[index] - positions[index]
            for index in range(len(records))
        ]
        if sum(remaining) == 0:
            break
        constituent_index = rng.weighted_index(remaining)
        output.append(
            float(observed_sequences[constituent_index][positions[constituent_index]].item())
        )
        origins.append(constituent_index)
        positions[constituent_index] += 1

    observed_count = len(output)
    trace = torch.zeros(max_length, dtype=torch.float32)
    origin_indices = torch.full((max_length,), -1, dtype=torch.int64)
    if observed_count:
        trace[:observed_count] = torch.tensor(output, dtype=torch.float32)
        origin_indices[:observed_count] = torch.tensor(origins, dtype=torch.int64)
    truncated = sum(original_lengths) > max_length
    return trace, origin_indices, tuple(positions), truncated


def build_synthetic_five_tab_corpus(
    records: Sequence[TraceRecord],
    monitored_classes: Sequence[int],
    *,
    base_seed: int,
    dataset_fingerprint: str,
    max_length: int = L_MAX,
) -> SyntheticCorpus:
    """Build the approved balanced, no-reuse synthetic five-tab corpus.

    This deterministic public protocol does not claim timestamp-based temporal
    merging.
    """

    if base_seed < 0:
        raise SyntheticProtocolError("base_seed must be non-negative")
    canonical_records, classes = _validate_inputs(
        records, monitored_classes, dataset_fingerprint, max_length
    )
    by_class: dict[int, list[TraceRecord]] = {class_id: [] for class_id in classes}
    for record in canonical_records:
        by_class[record.active_label_indices[0]].append(record)

    missing = [class_id for class_id, items in by_class.items() if not items]
    if missing:
        raise SyntheticProtocolError(f"monitored classes have no traces: {missing}")
    traces_per_class = min(len(items) for items in by_class.values())
    selected_by_class: dict[int, tuple[TraceRecord, ...]] = {}
    for class_id, items in by_class.items():
        ranked = sorted(
            items,
            key=lambda record: (
                stable_digest(
                    "synthetic-five-tab-v1/trace-rank",
                    base_seed,
                    dataset_fingerprint,
                    class_id,
                    record.trace_id,
                ),
                record.trace_id,
            ),
        )
        selected_by_class[class_id] = tuple(ranked[:traces_per_class])

    samples: list[SyntheticSample] = []
    entries: list[SyntheticManifestEntry] = []
    for round_index in range(traces_per_class):
        class_order = sorted(
            classes,
            key=lambda class_id: (
                stable_digest(
                    "synthetic-five-tab-v1/class-permutation",
                    base_seed,
                    dataset_fingerprint,
                    round_index,
                    class_id,
                ),
                class_id,
            ),
        )
        for group_index in range(0, len(class_order), FIVE_TAB_SIZE):
            grouped_classes = tuple(
                class_order[group_index : group_index + FIVE_TAB_SIZE]
            )
            constituents = tuple(
                selected_by_class[class_id][round_index]
                for class_id in grouped_classes
            )
            constituent_ids = tuple(record.trace_id for record in constituents)
            sample_seed = derive_seed(
                "synthetic-five-tab-v1/sample",
                base_seed,
                dataset_fingerprint,
                round_index,
                group_index // FIVE_TAB_SIZE,
                constituent_ids,
            )
            trace, origins, consumed_lengths, truncated = _interleave(
                constituents, sample_seed, max_length
            )
            labels_tuple = tuple(sorted(grouped_classes))
            labels = torch.zeros(len(classes), dtype=torch.float32)
            labels[list(labels_tuple)] = 1
            trace_id = "synthetic-" + stable_digest(
                "synthetic-five-tab-v1/trace-id",
                dataset_fingerprint,
                sample_seed,
                constituent_ids,
            )[:24]
            original_lengths = tuple(
                record.observed_length for record in constituents
            )
            observed_count = sum(consumed_lengths)
            entry = SyntheticManifestEntry(
                trace_id=trace_id,
                labels=labels_tuple,
                constituent_trace_ids=constituent_ids,
                constituent_root_provenance_ids=tuple(
                    str(record.root_provenance_id) for record in constituents
                ),
                constituent_source_provenance=tuple(
                    str(record.source_provenance) for record in constituents
                ),
                original_lengths=original_lengths,
                consumed_lengths=consumed_lengths,
                sequence_digest=sequence_digest(trace[:observed_count]),
                truncated=truncated,
                sample_seed=sample_seed,
                dataset_fingerprint=dataset_fingerprint,
                protocol_version=SYNTHETIC_FIVE_TAB_PROTOCOL,
            )
            samples.append(
                SyntheticSample(
                    trace=trace,
                    labels=labels,
                    origin_constituent_indices=origins,
                    manifest_entry=entry,
                )
            )
            entries.append(entry)

    manifest = Manifest(
        kind="synthetic",
        protocol_version=SYNTHETIC_FIVE_TAB_PROTOCOL,
        dataset_fingerprint=dataset_fingerprint,
        entries=tuple(entries),
        base_seed=base_seed,
    )
    from msawf.data.leakage import validate_manifest_leakage

    validate_manifest_leakage(manifest)
    return SyntheticCorpus(
        samples=tuple(samples),
        manifest=manifest,
        traces_per_class=traces_per_class,
    )
