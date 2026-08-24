import unittest

from msawf.constants import CANONICAL_CONSUMER_METHODS, CANONICAL_SEED
from msawf.data import fingerprint_records, generate_five_splits
from msawf.utils import derive_five_split_seed
from protocol_fixtures import make_pair_records


class FiveSplitProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_pair_records(class_count=6, replicas_per_pair=3)
        self.fingerprint = fingerprint_records(self.records)

    def generate(self):
        return generate_five_splits(
            self.records,
            tuple(range(6)),
            k=1,
            dataset_fingerprint=self.fingerprint,
        )

    def test_exactly_five_deterministic_pairwise_distinct_manifests(self) -> None:
        first = self.generate()
        second = self.generate()
        self.assertEqual(len(first.splits), 5)
        self.assertEqual(
            tuple(item.manifest.digest for item in first.splits),
            tuple(item.manifest.digest for item in second.splits),
        )
        support_sets = [frozenset(item.selection.support_trace_ids) for item in first.splits]
        self.assertEqual(len(set(support_sets)), 5)
        for index, generated in enumerate(first.splits):
            split_id = f"split-{index}"
            self.assertEqual(generated.selection.split_id, split_id)
            self.assertEqual(
                generated.selection.seed,
                derive_five_split_seed(CANONICAL_SEED, self.fingerprint, split_id),
            )

    def test_all_methods_share_one_manifest_id_per_split(self) -> None:
        collection = self.generate()
        for generated in collection.splits:
            ids = {
                generated.manifest_id_for_method(method)
                for method in CANONICAL_CONSUMER_METHODS
            }
            self.assertEqual(ids, {generated.manifest.digest})
            self.assertEqual(
                generated.manifest.consumer_methods, CANONICAL_CONSUMER_METHODS
            )


if __name__ == "__main__":
    unittest.main()
