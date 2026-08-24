import tempfile
import unittest

from msawf.data import (
    build_synthetic_five_tab_corpus,
    fingerprint_records,
    generate_five_splits,
    load_manifest,
    save_content_addressed_manifest,
)
from protocol_fixtures import make_pair_records, make_single_tab_records


class ProtocolManifestRoundTripTests(unittest.TestCase):
    def test_synthetic_manifest_round_trip(self) -> None:
        records = make_single_tab_records(
            class_count=5, traces_per_class=1, length=4
        )
        fingerprint = fingerprint_records(records)
        corpus = build_synthetic_five_tab_corpus(
            records,
            tuple(range(5)),
            base_seed=18,
            dataset_fingerprint=fingerprint,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_content_addressed_manifest(corpus.manifest, directory)
            loaded = load_manifest(path, expected_digest=corpus.manifest.digest)
        self.assertEqual(loaded, corpus.manifest)

    def test_generated_target_manifest_round_trip(self) -> None:
        records = make_pair_records(class_count=6, replicas_per_pair=3)
        fingerprint = fingerprint_records(records)
        generated = generate_five_splits(
            records,
            tuple(range(6)),
            k=1,
            dataset_fingerprint=fingerprint,
        ).splits[0]
        with tempfile.TemporaryDirectory() as directory:
            path = save_content_addressed_manifest(generated.manifest, directory)
            loaded = load_manifest(path, expected_digest=generated.manifest.digest)
        self.assertEqual(loaded, generated.manifest)


if __name__ == "__main__":
    unittest.main()
