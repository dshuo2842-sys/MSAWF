import unittest
from collections import Counter

import torch

from msawf.constants import L_MAX, SYNTHETIC_FIVE_TAB_PROTOCOL
from msawf.data import (
    build_synthetic_five_tab_corpus,
    fingerprint_records,
)
from protocol_fixtures import make_single_tab_records


class SyntheticFiveTabProtocolTests(unittest.TestCase):
    def build(self, records):
        class_count = records[0].labels.numel()
        fingerprint = fingerprint_records(records)
        return build_synthetic_five_tab_corpus(
            records,
            tuple(range(class_count)),
            base_seed=18,
            dataset_fingerprint=fingerprint,
        )

    def test_exactly_five_distinct_labels_and_exact_class_balance(self) -> None:
        records = make_single_tab_records(class_count=10, traces_per_class=3)
        corpus = self.build(records)
        self.assertEqual(len(corpus.samples), 10 * 3 // 5)
        label_counts = Counter()
        for sample in corpus.samples:
            active = tuple(torch.nonzero(sample.labels).flatten().tolist())
            self.assertEqual(len(active), 5)
            self.assertEqual(len(set(active)), 5)
            label_counts.update(active)
        self.assertEqual(label_counts, Counter({class_id: 3 for class_id in range(10)}))

    def test_constituent_order_and_provenance_are_preserved(self) -> None:
        records = make_single_tab_records(class_count=5, traces_per_class=1, length=13)
        by_id = {record.trace_id: record for record in records}
        sample = self.build(records).samples[0]
        entry = sample.manifest_entry
        for constituent_index, trace_id in enumerate(entry.constituent_trace_ids):
            original = by_id[trace_id]
            emitted = sample.trace[
                sample.origin_constituent_indices == constituent_index
            ]
            expected = original.trace[: entry.consumed_lengths[constituent_index]]
            self.assertTrue(torch.equal(emitted, expected))
            self.assertEqual(
                entry.constituent_root_provenance_ids[constituent_index],
                original.root_provenance_id,
            )
        self.assertEqual(entry.protocol_version, SYNTHETIC_FIVE_TAB_PROTOCOL)

    def test_canonical_corpus_never_reuses_a_constituent(self) -> None:
        records = make_single_tab_records(class_count=10, traces_per_class=2)
        corpus = self.build(records)
        constituent_ids = [
            trace_id
            for entry in corpus.manifest.entries
            for trace_id in entry.constituent_trace_ids
        ]
        self.assertEqual(len(constituent_ids), len(set(constituent_ids)))

    def test_generation_is_deterministic(self) -> None:
        records = make_single_tab_records(class_count=10, traces_per_class=2)
        first = self.build(records)
        second = self.build(tuple(reversed(records)))
        self.assertEqual(first.manifest.digest, second.manifest.digest)
        self.assertEqual(
            tuple(sample.manifest_entry.trace_id for sample in first.samples),
            tuple(sample.manifest_entry.trace_id for sample in second.samples),
        )
        for left, right in zip(first.samples, second.samples):
            self.assertTrue(torch.equal(left.trace, right.trace))
            self.assertTrue(
                torch.equal(
                    left.origin_constituent_indices,
                    right.origin_constituent_indices,
                )
            )

    def test_trace_is_truncated_at_l_max(self) -> None:
        records = make_single_tab_records(
            class_count=5, traces_per_class=1, length=4_000
        )
        sample = self.build(records).samples[0]
        self.assertEqual(sample.trace.numel(), L_MAX)
        self.assertEqual(sum(sample.manifest_entry.consumed_lengths), L_MAX)
        self.assertTrue(sample.manifest_entry.truncated)
        self.assertTrue(torch.all(sample.trace != 0).item())

    def test_short_trace_is_zero_padded(self) -> None:
        records = make_single_tab_records(class_count=5, traces_per_class=1, length=3)
        sample = self.build(records).samples[0]
        self.assertFalse(sample.manifest_entry.truncated)
        self.assertTrue(torch.all(sample.trace[:15] != 0).item())
        self.assertTrue(torch.all(sample.trace[15:] == 0).item())
        self.assertTrue(torch.all(sample.origin_constituent_indices[15:] == -1).item())


if __name__ == "__main__":
    unittest.main()
