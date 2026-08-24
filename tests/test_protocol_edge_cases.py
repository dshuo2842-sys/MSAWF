import hashlib
import unittest

from msawf.data import (
    FiveSplitProtocolError,
    fingerprint_records,
    generate_five_splits,
)
from msawf.utils import StableRNG, derive_five_split_seed
from protocol_fixtures import make_record


class ProtocolEdgeCaseTests(unittest.TestCase):
    def test_five_split_seed_matches_approved_byte_formula(self) -> None:
        fingerprint = "a" * 64
        split_id = "split-3"
        expected_bytes = hashlib.sha256(
            b"msawf-five-split-v1"
            + (18).to_bytes(8, "big", signed=False)
            + fingerprint.encode("utf-8")
            + split_id.encode("utf-8")
        ).digest()[:8]
        self.assertEqual(
            derive_five_split_seed(18, fingerprint, split_id),
            int.from_bytes(expected_bytes, "big"),
        )

    def test_stable_rng_reproduces_weighted_choices(self) -> None:
        first = StableRNG(18)
        second = StableRNG(18)
        self.assertEqual(
            [first.weighted_index([1, 2, 3, 4, 5]) for _ in range(50)],
            [second.weighted_index([1, 2, 3, 4, 5]) for _ in range(50)],
        )

    def test_identical_support_set_collision_is_a_protocol_failure(self) -> None:
        records = (
            make_record(
                "covers-all",
                (0, 1, 2, 3, 4),
                num_classes=5,
            ),
        )
        with self.assertRaisesRegex(FiveSplitProtocolError, "collision"):
            generate_five_splits(
                records,
                tuple(range(5)),
                k=1,
                dataset_fingerprint=fingerprint_records(records),
            )


if __name__ == "__main__":
    unittest.main()
