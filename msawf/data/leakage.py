"""Strict provenance and leakage validation for MSAWF data protocols."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from msawf.data.manifest import (
    Manifest,
    SyntheticManifestEntry,
    TargetSplitManifestEntry,
)


class LeakageError(ValueError):
    """Raised when a manifest contains forbidden identity or provenance overlap."""


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def validate_manifest_leakage(manifest: Manifest) -> None:
    """Reject duplicate entries, fingerprint mismatches, and role leakage."""

    duplicate_entries = _duplicates(entry.trace_id for entry in manifest.entries)
    if duplicate_entries:
        raise LeakageError(f"duplicated manifest trace_id values: {duplicate_entries}")

    mismatched_fingerprints = tuple(
        sorted(
            entry.trace_id
            for entry in manifest.entries
            if entry.dataset_fingerprint != manifest.dataset_fingerprint
        )
    )
    if mismatched_fingerprints:
        raise LeakageError(
            "entry dataset fingerprint mismatch for: "
            f"{mismatched_fingerprints}"
        )

    if manifest.kind == "synthetic":
        _validate_synthetic_manifest(manifest)
    elif manifest.kind == "target-split":
        _validate_target_manifest(manifest)
    else:
        raise LeakageError(f"unsupported manifest kind: {manifest.kind}")


def _validate_synthetic_manifest(manifest: Manifest) -> None:
    entries = tuple(
        entry
        for entry in manifest.entries
        if isinstance(entry, SyntheticManifestEntry)
    )
    if len(entries) != len(manifest.entries):
        raise LeakageError("synthetic manifest contains non-synthetic entries")
    constituent_ids = [
        constituent_id
        for entry in entries
        for constituent_id in entry.constituent_trace_ids
    ]
    duplicate_constituents = _duplicates(constituent_ids)
    if duplicate_constituents:
        raise LeakageError(
            "source constituent reused in canonical synthetic corpus: "
            f"{duplicate_constituents}"
        )
    root_ids = [
        root_id
        for entry in entries
        for root_id in entry.constituent_root_provenance_ids
    ]
    duplicate_roots = _duplicates(root_ids)
    if duplicate_roots:
        raise LeakageError(
            "source root provenance reused in canonical synthetic corpus: "
            f"{duplicate_roots}"
        )


def _validate_target_manifest(manifest: Manifest) -> None:
    entries = tuple(
        entry
        for entry in manifest.entries
        if isinstance(entry, TargetSplitManifestEntry)
    )
    if len(entries) != len(manifest.entries):
        raise LeakageError("target manifest contains non-target entries")
    support = tuple(entry for entry in entries if entry.role == "support")
    query = tuple(entry for entry in entries if entry.role == "query")
    support_trace_ids = {entry.trace_id for entry in support}
    query_trace_ids = {entry.trace_id for entry in query}
    trace_overlap = tuple(sorted(support_trace_ids.intersection(query_trace_ids)))
    if trace_overlap:
        raise LeakageError(f"support/query trace_id overlap: {trace_overlap}")
    support_roots = {entry.root_provenance_id for entry in support}
    query_roots = {entry.root_provenance_id for entry in query}
    root_overlap = tuple(sorted(support_roots.intersection(query_roots)))
    if root_overlap:
        raise LeakageError(
            "support/query root provenance or derived-view overlap: "
            f"{root_overlap}"
        )
    statistics = manifest.statistics
    if statistics is None:
        raise LeakageError("target manifest is missing coverage statistics")
    if len(support) != statistics.support_record_count:
        raise LeakageError("support count does not match manifest statistics")
    if len(query) != statistics.query_record_count:
        raise LeakageError("query count does not match manifest statistics")
    if tuple(entry.trace_id for entry in support) != statistics.selected_trace_ids:
        sorted_support_ids = tuple(sorted(entry.trace_id for entry in support))
        sorted_selected_ids = tuple(sorted(statistics.selected_trace_ids))
        if sorted_support_ids != sorted_selected_ids:
            raise LeakageError("support IDs do not match coverage statistics")


def validate_target_query_against_synthetic_source(
    target_manifest: Manifest, synthetic_manifest: Manifest
) -> None:
    """Reject target query records or ancestors used as source constituents."""

    if target_manifest.kind != "target-split":
        raise LeakageError("target_manifest must have kind target-split")
    if synthetic_manifest.kind != "synthetic":
        raise LeakageError("synthetic_manifest must have kind synthetic")
    target_query = tuple(
        entry
        for entry in target_manifest.entries
        if isinstance(entry, TargetSplitManifestEntry) and entry.role == "query"
    )
    source_entries = tuple(
        entry
        for entry in synthetic_manifest.entries
        if isinstance(entry, SyntheticManifestEntry)
    )
    query_trace_ids = {entry.trace_id for entry in target_query}
    query_trace_ids.update(
        constituent
        for entry in target_query
        for constituent in entry.constituent_trace_ids
    )
    query_root_ids = {entry.root_provenance_id for entry in target_query}
    source_trace_ids = {
        constituent
        for entry in source_entries
        for constituent in entry.constituent_trace_ids
    }
    source_root_ids = {
        root
        for entry in source_entries
        for root in entry.constituent_root_provenance_ids
    }
    trace_overlap = tuple(sorted(query_trace_ids.intersection(source_trace_ids)))
    root_overlap = tuple(sorted(query_root_ids.intersection(source_root_ids)))
    if trace_overlap or root_overlap:
        raise LeakageError(
            "target query provenance is used by the synthetic source; "
            f"trace_ids={trace_overlap}, root_ids={root_overlap}"
        )


def validate_pairwise_distinct_support_sets(manifests: Sequence[Manifest]) -> None:
    """Require every target split to have a distinct support membership set."""

    seen: dict[frozenset[str], str] = {}
    for manifest in manifests:
        if manifest.kind != "target-split":
            raise LeakageError("pairwise split validation requires target manifests")
        support_ids = frozenset(
            entry.trace_id
            for entry in manifest.entries
            if isinstance(entry, TargetSplitManifestEntry) and entry.role == "support"
        )
        if support_ids in seen:
            raise LeakageError(
                "identical support-set collision between "
                f"{seen[support_ids]} and {manifest.split_id}"
            )
        seen[support_ids] = str(manifest.split_id)


__all__ = [
    "LeakageError",
    "validate_manifest_leakage",
    "validate_pairwise_distinct_support_sets",
    "validate_target_query_against_synthetic_source",
]
