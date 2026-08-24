import unittest

import torch

from msawf.augmentation import PrefixOperator, RandomPrefixSampler
from msawf.constants import L_MAX, OMEGA


class PrefixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = torch.tensor([1, -1] * (L_MAX // 2), dtype=torch.float32)
        self.operator = PrefixOperator()

    def test_all_paper_prefix_lengths(self) -> None:
        for prefix_length in OMEGA:
            with self.subTest(prefix_length=prefix_length):
                output = self.operator(self.trace, prefix_length)
                self.assertEqual(tuple(output.shape), (L_MAX,))
                self.assertTrue(torch.equal(output[:prefix_length], self.trace[:prefix_length]))
                if prefix_length < L_MAX:
                    self.assertTrue(torch.all(output[prefix_length:] == 0).item())

    def test_prefix_preserves_arrival_order(self) -> None:
        output = self.operator(self.trace, 3000)
        self.assertTrue(torch.equal(output[:3000], self.trace[:3000]))

    def test_sampler_is_deterministic_and_only_emits_omega(self) -> None:
        first = RandomPrefixSampler(seed=18)
        second = RandomPrefixSampler(seed=18)
        first_values = [first.sample() for _ in range(20)]
        second_values = [second.sample() for _ in range(20)]
        self.assertEqual(first_values, second_values)
        self.assertTrue(set(first_values).issubset(set(OMEGA)))


if __name__ == "__main__":
    unittest.main()
