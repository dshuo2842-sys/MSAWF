import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from msawf.constants import SYNTHETIC_FIVE_TAB_PROTOCOL
from msawf.data import (
    CoverageStatistics,
    LeakageError,
    Manifest,
    ManifestDigestError,
    ManifestError,
    SyntheticManifestEntry,
    TargetSplitManifestEntry,
    load_manifest,
    save_content_addressed_manifest,
    validate_manifest_leakage,
    validate_target_query_against_synthetic_source,
)


FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64
SEQUENCE_DIGEST = "c" * 64


def target_entry(
    trace_id: str,
    role: str,
    *,
    root_id: str | None = None,
    fingerprint: str = FINGERPRINT,
) -> TargetSplitManifestEntry:
    return TargetSplitManifestEntry(
        trace_id=trace_id,
        root_provenance_id=root_id or f"root-{trace_id}",
        split_id="split-0",
        role=role,
        labels=(0,),
        seed=123,
        dataset_fingerprint=fingerprint,
        source_provenance="tiny-target-v1",
    )


def target_manifest(
    support_entries: tuple[TargetSplitManifestEntry, ...],
    query_entries: tuple[TargetSplitManifestEntry, ...],
) -> Manifest:
    statistics = CoverageStatistics(
        support_record_count=len(support_entries),
        query_record_count=len(query_entries),
        per_class_coverage=((0, len(support_entries)),),
        min_coverage=len(support_entries),
        max_coverage=len(support_entries),
        mean_coverage=float(len(support_entries)),
        selected_trace_ids=tuple(entry.trace_id for entry in support_entries),
        selection_order=tuple(entry.trace_id for entry in support_entries),
        pruned_trace_ids=(),
    )
    return Manifest(
        kind="target-split",
        protocol_version="msawf-five-split-v1",
        dataset_fingerprint=FINGERPRINT,
        entries=support_entries + query_entries,
        base_seed=18,
        split_id="split-0",
        seed=123,
        k=1,
        consumer_methods=("MSAWF",),
        statistics=statistics,
    )


def synthetic_entry(
    trace_id: str,
    constituent_ids: tuple[str, ...],
    *,
    class_offset: int = 0,
) -> SyntheticManifestEntry:
    return SyntheticManifestEntry(
        trace_id=trace_id,
        labels=tuple(range(class_offset, class_offset + 5)),
        constituent_trace_ids=constituent_ids,
        constituent_root_provenance_ids=tuple(
            f"root-{constituent_id}" for constituent_id in constituent_ids
        ),
        constituent_source_provenance=("tiny-source-v1",) * 5,
        original_lengths=(1,) * 5,
        consumed_lengths=(1,) * 5,
        sequence_digest=SEQUENCE_DIGEST,
        truncated=False,
        sample_seed=123,
        dataset_fingerprint=FINGERPRINT,
        protocol_version=SYNTHETIC_FIVE_TAB_PROTOCOL,
    )


def synthetic_manifest(entries: tuple[SyntheticManifestEntry, ...]) -> Manifest:
    return Manifest(
        kind="synthetic",
        protocol_version=SYNTHETIC_FIVE_TAB_PROTOCOL,
        dataset_fingerprint=FINGERPRINT,
        entries=entries,
        base_seed=18,
    )


class ManifestAndLeakageTests(unittest.TestCase):
    def test_content_addressed_save_load_and_frozen_schema(self) -> None:
        manifest = target_manifest(
            (target_entry("support", "support"),),
            (target_entry("query", "query"),),
        )
        reordered = Manifest(
            kind=manifest.kind,
            protocol_version=manifest.protocol_version,
            dataset_fingerprint=manifest.dataset_fingerprint,
            entries=tuple(reversed(manifest.entries)),
            base_seed=manifest.base_seed,
            split_id=manifest.split_id,
            seed=manifest.seed,
            k=manifest.k,
            consumer_methods=manifest.consumer_methods,
            statistics=manifest.statistics,
        )
        self.assertEqual(manifest.digest, reordered.digest)
        with self.assertRaises(FrozenInstanceError):
            manifest.kind = "synthetic"
        with tempfile.TemporaryDirectory() as directory:
            first_path = save_content_addressed_manifest(manifest, directory)
            second_path = save_content_addressed_manifest(manifest, directory)
            self.assertEqual(first_path, second_path)
            self.assertEqual(first_path.name, f"{manifest.digest}.json")
            loaded = load_manifest(first_path, expected_digest=manifest.digest)
            self.assertEqual(loaded, manifest)

    def test_manifest_digest_mismatch_is_rejected(self) -> None:
        manifest = target_manifest(
            (target_entry("support", "support"),),
            (target_entry("query", "query"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_content_addressed_manifest(manifest, directory)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["manifest"]["base_seed"] = 19
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ManifestDigestError):
                load_manifest(path)

    def test_duplicated_trace_id_across_support_query_is_rejected(self) -> None:
        manifest = target_manifest(
            (target_entry("duplicate", "support", root_id="root-support"),),
            (target_entry("duplicate", "query", root_id="root-query"),),
        )
        with self.assertRaisesRegex(LeakageError, "duplicated manifest trace_id"):
            validate_manifest_leakage(manifest)

    def test_duplicated_manifest_entry_within_role_is_rejected(self) -> None:
        duplicate = target_entry("duplicate", "support")
        manifest = target_manifest(
            (duplicate, duplicate),
            (target_entry("query", "query"),),
        )
        with self.assertRaisesRegex(LeakageError, "duplicated manifest trace_id"):
            validate_manifest_leakage(manifest)

    def test_root_provenance_and_derived_view_crossing_is_rejected(self) -> None:
        manifest = target_manifest(
            (target_entry("view-support", "support", root_id="shared-root"),),
            (target_entry("view-query", "query", root_id="shared-root"),),
        )
        with self.assertRaisesRegex(LeakageError, "derived-view overlap"):
            validate_manifest_leakage(manifest)

    def test_source_constituent_reuse_is_rejected(self) -> None:
        first = synthetic_entry("synthetic-a", ("a", "b", "c", "d", "e"))
        second = synthetic_entry(
            "synthetic-b", ("a", "f", "g", "h", "i"), class_offset=5
        )
        with self.assertRaisesRegex(LeakageError, "constituent reused"):
            validate_manifest_leakage(synthetic_manifest((first, second)))

    def test_target_query_used_as_source_constituent_is_rejected(self) -> None:
        target = target_manifest(
            (target_entry("support", "support"),),
            (target_entry("query-source", "query"),),
        )
        source = synthetic_manifest(
            (
                synthetic_entry(
                    "synthetic-a",
                    ("query-source", "b", "c", "d", "e"),
                ),
            )
        )
        with self.assertRaisesRegex(LeakageError, "target query provenance"):
            validate_target_query_against_synthetic_source(target, source)

    def test_entry_fingerprint_mismatch_is_rejected(self) -> None:
        mismatched = target_entry(
            "support", "support", fingerprint=OTHER_FINGERPRINT
        )
        statistics = CoverageStatistics(
            support_record_count=1,
            query_record_count=0,
            per_class_coverage=((0, 1),),
            min_coverage=1,
            max_coverage=1,
            mean_coverage=1.0,
            selected_trace_ids=("support",),
            selection_order=("support",),
            pruned_trace_ids=(),
        )
        with self.assertRaisesRegex(ManifestError, "fingerprint mismatch"):
            Manifest(
                kind="target-split",
                protocol_version="msawf-five-split-v1",
                dataset_fingerprint=FINGERPRINT,
                entries=(mismatched,),
                base_seed=18,
                split_id="split-0",
                seed=123,
                k=1,
                statistics=statistics,
            )


if __name__ == "__main__":
    unittest.main()
