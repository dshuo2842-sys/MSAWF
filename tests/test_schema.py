import unittest

import torch

from msawf.constants import L_MAX
from msawf.data import SchemaError, TraceRecord, validate_trace


class TraceSchemaTests(unittest.TestCase):
    def test_valid_direction_and_zero_padding_contract(self) -> None:
        trace = torch.tensor([1, -1, 1] + [0] * (L_MAX - 3))
        validated = validate_trace(trace)
        self.assertEqual(validated.numel(), L_MAX)

    def test_signed_packet_magnitude_is_rejected(self) -> None:
        trace = torch.tensor([1, -1200] + [0] * (L_MAX - 2))
        with self.assertRaisesRegex(SchemaError, "packet directions"):
            validate_trace(trace)

    def test_non_suffix_zero_is_rejected(self) -> None:
        trace = torch.tensor([1, 0, -1] + [0] * (L_MAX - 3))
        with self.assertRaisesRegex(SchemaError, "contiguous suffix"):
            validate_trace(trace)

    def test_fixed_length_is_exactly_l_max(self) -> None:
        with self.assertRaisesRegex(SchemaError, "L_max=15000"):
            validate_trace(torch.tensor([1, -1]))

    def test_trace_record_preserves_provenance_and_labels(self) -> None:
        trace = torch.tensor([1, -1] + [0] * (L_MAX - 2))
        labels = torch.tensor([1, 0, 1])
        record = TraceRecord(trace, labels, "trace-1", "source", "split-0")
        self.assertEqual(record.observed_length, 2)
        self.assertEqual(record.trace_id, "trace-1")
        self.assertTrue(torch.equal(record.labels, labels.float()))


if __name__ == "__main__":
    unittest.main()
