"""Canonical data contracts and approved MSAWF public data protocols."""

from .kshot import (
    InfeasibleKShotError,
    KShotProtocolError,
    KShotSelection,
    select_multilabel_kshot,
)
from .leakage import (
    LeakageError,
    validate_manifest_leakage,
    validate_pairwise_distinct_support_sets,
    validate_target_query_against_synthetic_source,
)
from .manifest import (
    CoverageStatistics,
    Manifest,
    ManifestDigestError,
    ManifestError,
    SyntheticManifestEntry,
    TargetSplitManifestEntry,
    fingerprint_records,
    load_manifest,
    save_content_addressed_manifest,
    sequence_digest,
    validate_manifest,
)
from .schema import (
    SchemaError,
    TraceRecord,
    observed_length,
    validate_labels,
    validate_trace,
)
from .splits import (
    FiveSplitCollection,
    FiveSplitProtocolError,
    GeneratedSplit,
    generate_five_splits,
)
from .synthetic import (
    SyntheticCorpus,
    SyntheticProtocolError,
    SyntheticSample,
    build_synthetic_five_tab_corpus,
)

__all__ = [
    "CoverageStatistics",
    "FiveSplitCollection",
    "FiveSplitProtocolError",
    "GeneratedSplit",
    "InfeasibleKShotError",
    "KShotProtocolError",
    "KShotSelection",
    "LeakageError",
    "Manifest",
    "ManifestDigestError",
    "ManifestError",
    "SchemaError",
    "SyntheticCorpus",
    "SyntheticManifestEntry",
    "SyntheticProtocolError",
    "SyntheticSample",
    "TargetSplitManifestEntry",
    "TraceRecord",
    "build_synthetic_five_tab_corpus",
    "fingerprint_records",
    "generate_five_splits",
    "load_manifest",
    "observed_length",
    "save_content_addressed_manifest",
    "select_multilabel_kshot",
    "sequence_digest",
    "validate_labels",
    "validate_manifest",
    "validate_manifest_leakage",
    "validate_pairwise_distinct_support_sets",
    "validate_target_query_against_synthetic_source",
    "validate_trace",
]
