import random
import unittest

import torch

from msawf.utils import make_generator, seed_everything


class SeedUtilityTests(unittest.TestCase):
    def test_global_seed_is_repeatable(self) -> None:
        seed_everything(18)
        first_python = random.random()
        first_torch = torch.rand(3)
        seed_everything(18)
        self.assertEqual(first_python, random.random())
        torch.testing.assert_close(first_torch, torch.rand(3))

    def test_independent_generator_is_repeatable(self) -> None:
        first = torch.rand(5, generator=make_generator(18))
        second = torch.rand(5, generator=make_generator(18))
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
