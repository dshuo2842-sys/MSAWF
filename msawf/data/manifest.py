"""Versioned, immutable, content-addressed manifests for MSAWF protocols."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from msawf.constants import MANIFEST_SCHEMA_VERSION
from msawf.data.schema import TraceRecord
from msawf.utils.hashing import canonical_json_bytes, sha256_hex


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest violates its schema or integrity contract."""


class ManifestDigestError(ManifestError):
    """Raised when manifest content does not match its recorded digest."""


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must not be empty")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ManifestError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_seed(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**64:
        raise ManifestError(f"{name} must be an unsigned 64-bit integer")


def _text_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ManifestError(f"{name} must contain non-empty strings")
    return result


def _int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise ManifestError(f"{name} must contain integers")
    return result


def sequence_digest(directions: Sequence[int] | Any) -> str:
    """Digest observed directions independently of tensor representation."""

    values = (
        directions.detach().cpu().flatten().tolist()
        if hasattr(directions, "detach")
        else list(directions)
    )
    canonical = [int(value) for value in values]
    if any(value not in (-1, 1) for value in canonical):
        raise ManifestError("observed sequence values must be {-1,+1}")
    return sha256_hex(
        {"schema": "msawf-direction-sequence-v1", "directions": canonical}
    )


def fingerprint_records(records: Sequence[TraceRecord]) -> str:
    """Create a path-independent fingerprint of canonical trace content."""

    ordered = sorted(records, key=lambda record: record.trace_id)
    trace_ids = [record.trace_id for record in ordered]
    if len(trace_ids) != len(set(trace_ids)):
        raise ManifestError("dataset records contain duplicated trace_id values")
    entries = [
        {
            "trace_id": record.trace_id,
            "root_provenance_id": record.root_provenance_id,
            "labels": list(record.active_label_indices),
            "sequence_digest": sequence_digest(
                record.trace[: record.observed_length]
            ),
            "domain": record.domain,
            "source_provenance": record.source_provenance,
        }
        for record in ordered
    ]
    return sha256_hex(
        {"schema": "msawf-dataset-fingerprint-v1", "records": entries}
    )


@dataclass(frozen=True)
class SyntheticManifestEntry:
    """Provenance for one synthetic five-tab sample."""

    trace_id: str
    labels: tuple[int, ...]
    constituent_trace_ids: tuple[str, ...]
    constituent_root_provenance_ids: tuple[str, ...]
    constituent_source_provenance: tuple[str, ...]
    original_lengths: tuple[int, ...]
    consumed_lengths: tuple[int, ...]
    sequence_digest: str
    truncated: bool
    sample_seed: int
    dataset_fingerprint: str
    protocol_version: str

    def __post_init__(self) -> None:
        _require_text(self.trace_id, "trace_id")
        labels = _int_tuple(self.labels, "labels")
        trace_ids = _text_tuple(self.constituent_trace_ids, "constituent_trace_ids")
        roots = _text_tuple(
            self.constituent_root_provenance_ids,
            "constituent_root_provenance_ids",
        )
        sources = _text_tuple(
            self.constituent_source_provenance,
            "constituent_source_provenance",
        )
        original = _int_tuple(self.original_lengths, "original_lengths")
        consumed = _int_tuple(self.consumed_lengths, "consumed_lengths")
        if {len(labels), len(trace_ids), len(roots), len(sources), len(original), len(consumed)} != {5}:
            raise ManifestError("synthetic manifest fields must describe exactly five tabs")
        if labels != tuple(sorted(labels)) or len(set(labels)) != 5:
            raise ManifestError("synthetic labels must be five distinct sorted classes")
        if len(set(trace_ids)) != 5 or len(set(roots)) != 5:
            raise ManifestError("synthetic constituents and roots must be distinct")
        if any(length <= 0 for length in original):
            raise ManifestError("original_lengths must be positive")
        if any(used < 0 or used > total for used, total in zip(consumed, original)):
            raise ManifestError("consumed_lengths must be within original lengths")
        _require_sha256(self.sequence_digest, "sequence_digest")
        if not isinstance(self.truncated, bool):
            raise ManifestError("truncated must be boolean")
        _require_seed(self.sample_seed, "sample_seed")
        _require_sha256(self.dataset_fingerprint, "dataset_fingerprint")
        _require_text(self.protocol_version, "protocol_version")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "constituent_trace_ids", trace_ids)
        object.__setattr__(self, "constituent_root_provenance_ids", roots)
        object.__setattr__(self, "constituent_source_provenance", sources)
        object.__setattr__(self, "original_lengths", original)
        object.__setattr__(self, "consumed_lengths", consumed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "labels": list(self.labels),
            "constituent_trace_ids": list(self.constituent_trace_ids),
            "constituent_root_provenance_ids": list(self.constituent_root_provenance_ids),
            "constituent_source_provenance": list(self.constituent_source_provenance),
            "original_lengths": list(self.original_lengths),
            "consumed_lengths": list(self.consumed_lengths),
            "sequence_digest": self.sequence_digest,
            "truncated": self.truncated,
            "sample_seed": self.sample_seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True)
class TargetSplitManifestEntry:
    """Membership and provenance for one target support/query record."""

    trace_id: str
    root_provenance_id: str
    split_id: str
    role: str
    labels: tuple[int, ...]
    seed: int
    dataset_fingerprint: str
    source_provenance: str
    constituent_trace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.trace_id, "trace_id")
        _require_text(self.root_provenance_id, "root_provenance_id")
        _require_text(self.split_id, "split_id")
        if self.role not in {"support", "query"}:
            raise ManifestError("role must be 'support' or 'query'")
        labels = _int_tuple(self.labels, "labels")
        if labels != tuple(sorted(set(labels))):
            raise ManifestError("labels must be unique and sorted")
        _require_seed(self.seed, "seed")
        _require_sha256(self.dataset_fingerprint, "dataset_fingerprint")
        _require_text(self.source_provenance, "source_provenance")
        constituents = _text_tuple(self.constituent_trace_ids, "constituent_trace_ids")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "constituent_trace_ids", constituents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_provenance_id": self.root_provenance_id,
            "split_id": self.split_id,
            "role": self.role,
            "labels": list(self.labels),
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_provenance": self.source_provenance,
            "constituent_trace_ids": list(self.constituent_trace_ids),
        }


@dataclass(frozen=True)
class CoverageStatistics:
    """Auditable statistics from multi-label support selection."""

    support_record_count: int
    query_record_count: int
    per_class_coverage: tuple[tuple[int, int], ...]
    min_coverage: int
    max_coverage: int
    mean_coverage: float
    selected_trace_ids: tuple[str, ...]
    selection_order: tuple[str, ...]
    pruned_trace_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        coverage = tuple(
            (int(class_id), int(count)) for class_id, count in self.per_class_coverage
        )
        if coverage != tuple(sorted(coverage)) or not coverage:
            raise ManifestError("per_class_coverage must be non-empty and class-sorted")
        if self.support_record_count < 0 or self.query_record_count < 0:
            raise ManifestError("support/query counts must be non-negative")
        counts = [count for _, count in coverage]
        if self.min_coverage != min(counts) or self.max_coverage != max(counts):
            raise ManifestError("coverage extrema do not match per-class coverage")
        if self.mean_coverage != sum(counts) / len(counts):
            raise ManifestError("mean_coverage does not match per-class coverage")
        if self.support_record_count != len(self.selected_trace_ids):
            raise ManifestError("selected_trace_ids count does not match support count")
        object.__setattr__(self, "per_class_coverage", coverage)
        object.__setattr__(self, "selected_trace_ids", tuple(self.selected_trace_ids))
        object.__setattr__(self, "selection_order", tuple(self.selection_order))
        object.__setattr__(self, "pruned_trace_ids", tuple(self.pruned_trace_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_record_count": self.support_record_count,
            "query_record_count": self.query_record_count,
            "per_class_coverage": [
                {"class_id": class_id, "count": count}
                for class_id, count in self.per_class_coverage
            ],
            "min_coverage": self.min_coverage,
            "max_coverage": self.max_coverage,
            "mean_coverage": self.mean_coverage,
            "selected_trace_ids": list(self.selected_trace_ids),
            "selection_order": list(self.selection_order),
            "pruned_trace_ids": list(self.pruned_trace_ids),
        }


ManifestEntry = SyntheticManifestEntry | TargetSplitManifestEntry


@dataclass(frozen=True)
class Manifest:
    """Frozen manifest envelope addressed by its canonical payload digest."""

    kind: str
    protocol_version: str
    dataset_fingerprint: str
    entries: tuple[ManifestEntry, ...]
    base_seed: int
    split_id: str | None = None
    seed: int | None = None
    k: int | None = None
    consumer_methods: tuple[str, ...] = ()
    statistics: CoverageStatistics | None = None
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in {"synthetic", "target-split"}:
            raise ManifestError("manifest kind must be synthetic or target-split")
        _require_text(self.protocol_version, "protocol_version")
        _require_sha256(self.dataset_fingerprint, "dataset_fingerprint")
        _require_seed(self.base_seed, "base_seed")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
        entries = tuple(sorted(tuple(self.entries), key=lambda entry: entry.trace_id))
        if not entries:
            raise ManifestError("manifest entries must not be empty")
        methods = _text_tuple(self.consumer_methods, "consumer_methods")
        if self.kind == "synthetic":
            if any(not isinstance(entry, SyntheticManifestEntry) for entry in entries):
                raise ManifestError("synthetic manifest contains a target entry")
            if any(
                entry.protocol_version != self.protocol_version
                or entry.dataset_fingerprint != self.dataset_fingerprint
                for entry in entries
            ):
                raise ManifestError("synthetic entry protocol/fingerprint mismatch")
            if any(value is not None for value in (self.split_id, self.seed, self.k)):
                raise ManifestError("synthetic manifest must not define split fields")
            if self.statistics is not None:
                raise ManifestError("synthetic manifest must not define support statistics")
        else:
            if any(not isinstance(entry, TargetSplitManifestEntry) for entry in entries):
                raise ManifestError("target manifest contains a synthetic entry")
            if self.split_id is None or self.seed is None or self.k is None:
                raise ManifestError("target manifest requires split_id, seed, and k")
            _require_text(self.split_id, "split_id")
            _require_seed(self.seed, "seed")
            if self.k <= 0:
                raise ManifestError("k must be positive")
            if self.statistics is None:
                raise ManifestError("target manifest requires coverage statistics")
            if any(
                entry.split_id != self.split_id
                or entry.seed != self.seed
                or entry.dataset_fingerprint != self.dataset_fingerprint
                for entry in entries
            ):
                raise ManifestError("target entry split/seed/fingerprint mismatch")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "consumer_methods", methods)

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "base_seed": self.base_seed,
            "split_id": self.split_id,
            "seed": self.seed,
            "k": self.k,
            "consumer_methods": list(self.consumer_methods),
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return sha256_hex(self.payload_dict())

    def document_dict(self) -> dict[str, Any]:
        return {"digest": self.digest, "manifest": self.payload_dict()}


def _statistics_from_dict(raw: Mapping[str, Any]) -> CoverageStatistics:
    return CoverageStatistics(
        support_record_count=int(raw["support_record_count"]),
        query_record_count=int(raw["query_record_count"]),
        per_class_coverage=tuple(
            (int(item["class_id"]), int(item["count"]))
            for item in raw["per_class_coverage"]
        ),
        min_coverage=int(raw["min_coverage"]),
        max_coverage=int(raw["max_coverage"]),
        mean_coverage=float(raw["mean_coverage"]),
        selected_trace_ids=tuple(raw["selected_trace_ids"]),
        selection_order=tuple(raw["selection_order"]),
        pruned_trace_ids=tuple(raw["pruned_trace_ids"]),
    )


def manifest_from_payload(raw: Mapping[str, Any]) -> Manifest:
    """Parse canonical JSON data into frozen schema objects."""

    try:
        kind = str(raw["kind"])
        if kind == "synthetic":
            entries: tuple[ManifestEntry, ...] = tuple(
                SyntheticManifestEntry(
                    trace_id=item["trace_id"],
                    labels=tuple(item["labels"]),
                    constituent_trace_ids=tuple(item["constituent_trace_ids"]),
                    constituent_root_provenance_ids=tuple(item["constituent_root_provenance_ids"]),
                    constituent_source_provenance=tuple(item["constituent_source_provenance"]),
                    original_lengths=tuple(item["original_lengths"]),
                    consumed_lengths=tuple(item["consumed_lengths"]),
                    sequence_digest=item["sequence_digest"],
                    truncated=item["truncated"],
                    sample_seed=int(item["sample_seed"]),
                    dataset_fingerprint=item["dataset_fingerprint"],
                    protocol_version=item["protocol_version"],
                )
                for item in raw["entries"]
            )
        elif kind == "target-split":
            entries = tuple(
                TargetSplitManifestEntry(
                    trace_id=item["trace_id"],
                    root_provenance_id=item["root_provenance_id"],
                    split_id=item["split_id"],
                    role=item["role"],
                    labels=tuple(item["labels"]),
                    seed=int(item["seed"]),
                    dataset_fingerprint=item["dataset_fingerprint"],
                    source_provenance=item["source_provenance"],
                    constituent_trace_ids=tuple(item.get("constituent_trace_ids", ())),
                )
                for item in raw["entries"]
            )
        else:
            raise ManifestError(f"unsupported manifest kind: {kind}")
        statistics_raw = raw.get("statistics")
        return Manifest(
            kind=kind,
            protocol_version=raw["protocol_version"],
            dataset_fingerprint=raw["dataset_fingerprint"],
            entries=entries,
            base_seed=int(raw["base_seed"]),
            split_id=raw.get("split_id"),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            k=int(raw["k"]) if raw.get("k") is not None else None,
            consumer_methods=tuple(raw.get("consumer_methods", ())),
            statistics=(
                _statistics_from_dict(statistics_raw)
                if statistics_raw is not None
                else None
            ),
            schema_version=raw["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"invalid manifest payload: {exc}") from exc


def validate_manifest(manifest: Manifest) -> None:
    """Run strict structure and leakage checks."""

    from msawf.data.leakage import validate_manifest_leakage

    validate_manifest_leakage(manifest)


def load_manifest(path: str | Path, *, expected_digest: str | None = None) -> Manifest:
    """Load and authenticate a manifest without trusting its file path."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest: {exc}") from exc
    if not isinstance(document, Mapping) or set(document) != {"digest", "manifest"}:
        raise ManifestError("manifest document must contain only digest and manifest")
    recorded = document["digest"]
    _require_sha256(recorded, "digest")
    if expected_digest is not None and recorded != expected_digest:
        raise ManifestDigestError(
            f"manifest digest {recorded} does not match expected {expected_digest}"
        )
    payload = document["manifest"]
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest payload must be an object")
    actual = sha256_hex(payload)
    if actual != recorded:
        raise ManifestDigestError(
            f"manifest digest mismatch: recorded {recorded}, actual {actual}"
        )
    manifest = manifest_from_payload(payload)
    if manifest.digest != recorded:
        raise ManifestDigestError("parsed manifest does not reproduce recorded digest")
    validate_manifest(manifest)
    return manifest


def save_content_addressed_manifest(manifest: Manifest, directory: str | Path) -> Path:
    """Create ``<digest>.json`` once and never overwrite manifest content."""

    validate_manifest(manifest)
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    path = target_directory / f"{manifest.digest}.json"
    if path.exists():
        existing = load_manifest(path, expected_digest=manifest.digest)
        if existing != manifest:
            raise ManifestDigestError("existing content-addressed manifest differs")
        return path
    encoded = canonical_json_bytes(manifest.document_dict()).decode("utf-8") + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except FileExistsError:
        existing = load_manifest(path, expected_digest=manifest.digest)
        if existing != manifest:
            raise ManifestDigestError("concurrent manifest creation differs")
    return path
