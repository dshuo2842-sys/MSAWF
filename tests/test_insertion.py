import unittest

import torch

from msawf.augmentation import InsertionTransform
from msawf.constants import L_MAX


class InsertionTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = torch.tensor([1, -1, -1, 1, -1], dtype=torch.float32)
        self.labels = torch.tensor([1, 0, 1], dtype=torch.float32)

    def test_labels_and_genuine_packet_order_are_preserved(self) -> None:
        result = InsertionTransform(1.0, seed=18)(self.trace, self.labels)
        self.assertEqual(tuple(result.trace.shape), (L_MAX,))
        self.assertTrue(torch.equal(result.labels, self.labels))
        genuine = result.trace[result.origin_indices >= 0]
        self.assertTrue(torch.equal(genuine, self.trace))
        self.assertEqual(int(result.dummy_mask.sum().item()), len(self.trace))

    def test_fixed_seed_produces_identical_augmentation(self) -> None:
        first = InsertionTransform(0.2, seed=18)(self.trace, self.labels)
        second = InsertionTransform(0.2, seed=18)(self.trace, self.labels)
        self.assertTrue(torch.equal(first.trace, second.trace))
        self.assertTrue(torch.equal(first.origin_indices, second.origin_indices))

    def test_zero_probability_only_pads(self) -> None:
        result = InsertionTransform(0.0, seed=18)(self.trace, self.labels)
        self.assertTrue(torch.equal(result.trace[: len(self.trace)], self.trace))
        self.assertTrue(torch.all(result.trace[len(self.trace) :] == 0).item())
        self.assertEqual(int(result.dummy_mask.sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
