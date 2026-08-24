import unittest
from collections import Counter

from msawf.data import (
    InfeasibleKShotError,
    fingerprint_records,
    select_multilabel_kshot,
)
from protocol_fixtures import make_pair_records, make_record


class MultiLabelKShotProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_pair_records(class_count=6, replicas_per_pair=2)
        self.fingerprint = fingerprint_records(self.records)

    def select(self):
        return select_multilabel_kshot(
            self.records,
            tuple(range(6)),
            k=2,
            seed=1234,
            split_id="split-test",
            dataset_fingerprint=self.fingerprint,
        )

    def test_coverage_statistics_and_disjointness(self) -> None:
        result = self.select()
        coverage = dict(result.statistics.per_class_coverage)
        self.assertTrue(all(value >= 2 for value in coverage.values()))
        self.assertEqual(result.statistics.min_coverage, min(coverage.values()))
        self.assertEqual(result.statistics.max_coverage, max(coverage.values()))
        self.assertEqual(
            result.statistics.mean_coverage,
            sum(coverage.values()) / len(coverage),
        )
        self.assertEqual(result.statistics.support_record_count, len(result.support))
        self.assertEqual(result.statistics.query_record_count, len(result.query))
        self.assertFalse(set(result.support_trace_ids) & set(result.query_trace_ids))
        self.assertFalse(
            {record.root_provenance_id for record in result.support}
            & {record.root_provenance_id for record in result.query}
        )

    def test_selection_is_deterministic_under_input_reordering(self) -> None:
        first = self.select()
        second = select_multilabel_kshot(
            tuple(reversed(self.records)),
            tuple(range(6)),
            k=2,
            seed=1234,
            split_id="split-test",
            dataset_fingerprint=self.fingerprint,
        )
        self.assertEqual(first.support_trace_ids, second.support_trace_ids)
        self.assertEqual(first.statistics, second.statistics)

    def test_reverse_pruning_is_inclusion_minimal(self) -> None:
        result = self.select()
        for removed in result.support:
            coverage = Counter()
            for record in result.support:
                if record.trace_id != removed.trace_id:
                    coverage.update(record.active_label_indices)
            self.assertTrue(any(coverage[class_id] < 2 for class_id in range(6)))

    def test_infeasible_k_lists_insufficient_classes(self) -> None:
        records = (
            make_record("only-zero", (0,), num_classes=2),
            make_record("only-one", (1,), num_classes=2),
        )
        with self.assertRaises(InfeasibleKShotError) as context:
            select_multilabel_kshot(
                records,
                (0, 1),
                k=2,
                seed=18,
                split_id="split-test",
                dataset_fingerprint=fingerprint_records(records),
            )
        self.assertEqual(context.exception.insufficient_classes, ((0, 1), (1, 1)))


if __name__ == "__main__":
    unittest.main()
