"""Canonical five-split generation using immutable shared manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from msawf.constants import (
    CANONICAL_CONSUMER_METHODS,
    CANONICAL_SEED,
    FIVE_SPLIT_PROTOCOL,
)
from msawf.data.kshot import KShotSelection, select_multilabel_kshot
from msawf.data.leakage import (
    LeakageError,
    validate_manifest_leakage,
    validate_pairwise_distinct_support_sets,
)
from msawf.data.manifest import Manifest, TargetSplitManifestEntry
from msawf.data.schema import TraceRecord
from msawf.utils.hashing import derive_five_split_seed


class FiveSplitProtocolError(ValueError):
    """Raised when five canonical distinct splits cannot be generated."""


@dataclass(frozen=True)
class GeneratedSplit:
    """One deterministic selection and its shared immutable manifest."""

    selection: KShotSelection
    manifest: Manifest

    @property
    def manifest_id(self) -> str:
        return self.manifest.digest

    def manifest_id_for_method(self, method: str) -> str:
        """Return the same canonical manifest ID for every approved method."""

        if method not in self.manifest.consumer_methods:
            raise FiveSplitProtocolError(f"method is not an approved consumer: {method}")
        return self.manifest.digest


@dataclass(frozen=True)
class FiveSplitCollection:
    """The five pairwise-distinct canonical target splits."""

    splits: tuple[GeneratedSplit, ...]
    base_seed: int
    dataset_fingerprint: str

    def for_split(self, split_id: str) -> GeneratedSplit:
        for generated in self.splits:
            if generated.selection.split_id == split_id:
                return generated
        raise KeyError(split_id)


def _target_entry(
    record: TraceRecord,
    *,
    split_id: str,
    role: str,
    seed: int,
    dataset_fingerprint: str,
) -> TargetSplitManifestEntry:
    return TargetSplitManifestEntry(
        trace_id=record.trace_id,
        root_provenance_id=str(record.root_provenance_id),
        split_id=split_id,
        role=role,
        labels=record.active_label_indices,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        source_provenance=str(record.source_provenance),
    )


def generate_five_splits(
    records: Sequence[TraceRecord],
    monitored_classes: Sequence[int],
    *,
    k: int,
    dataset_fingerprint: str,
    base_seed: int = CANONICAL_SEED,
) -> FiveSplitCollection:
    """Generate `split-0` through `split-4` independently of training."""

    if base_seed != CANONICAL_SEED:
        raise FiveSplitProtocolError(
            f"canonical five-split base_seed must be {CANONICAL_SEED}"
        )
    generated_splits: list[GeneratedSplit] = []
    for index in range(5):
        split_id = f"split-{index}"
        seed = derive_five_split_seed(base_seed, dataset_fingerprint, split_id)
        selection = select_multilabel_kshot(
            records,
            monitored_classes,
            k=k,
            seed=seed,
            split_id=split_id,
            dataset_fingerprint=dataset_fingerprint,
        )
        support_ids = {record.trace_id for record in selection.support}
        entries = tuple(
            _target_entry(
                record,
                split_id=split_id,
                role="support" if record.trace_id in support_ids else "query",
                seed=seed,
                dataset_fingerprint=dataset_fingerprint,
            )
            for record in sorted(records, key=lambda item: item.trace_id)
        )
        manifest = Manifest(
            kind="target-split",
            protocol_version=FIVE_SPLIT_PROTOCOL,
            dataset_fingerprint=dataset_fingerprint,
            entries=entries,
            base_seed=base_seed,
            split_id=split_id,
            seed=seed,
            k=k,
            consumer_methods=CANONICAL_CONSUMER_METHODS,
            statistics=selection.statistics,
        )
        validate_manifest_leakage(manifest)
        generated_splits.append(
            GeneratedSplit(selection=selection, manifest=manifest)
        )

    manifests = tuple(generated.manifest for generated in generated_splits)
    try:
        validate_pairwise_distinct_support_sets(manifests)
    except LeakageError as exc:
        raise FiveSplitProtocolError(str(exc)) from exc
    return FiveSplitCollection(
        splits=tuple(generated_splits),
        base_seed=base_seed,
        dataset_fingerprint=dataset_fingerprint,
    )


__all__ = [
    "FiveSplitCollection",
    "FiveSplitProtocolError",
    "GeneratedSplit",
    "generate_five_splits",
]
