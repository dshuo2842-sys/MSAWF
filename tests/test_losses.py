import math
import unittest

import torch

from msawf.losses import (
    bridge_alpha,
    bridge_loss,
    classification_loss,
    consistency_loss,
    stage_three_objective,
)


class LossTests(unittest.TestCase):
    def test_classification_loss_zero_logits_equals_log_two(self) -> None:
        logits = torch.zeros((2, 3))
        targets = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        loss = classification_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)

    def test_consistency_is_zero_for_identical_features(self) -> None:
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(consistency_loss(features, features).item(), 0.0)

    def test_consistency_is_positive_for_different_features(self) -> None:
        clean = torch.zeros((2, 3))
        perturbed = torch.ones((2, 3))
        self.assertGreater(consistency_loss(clean, perturbed).item(), 0.0)
        self.assertEqual(consistency_loss(clean, perturbed).item(), 3.0)

    def test_bridge_alpha_endpoints_and_loss(self) -> None:
        self.assertEqual(bridge_alpha(1, 20), 1.0)
        self.assertEqual(bridge_alpha(20, 20), 0.0)
        source = torch.tensor(2.0)
        target = torch.tensor(4.0)
        self.assertEqual(bridge_loss(source, target, 1.0).item(), 2.0)
        self.assertEqual(bridge_loss(source, target, 0.0).item(), 4.0)

    def test_stage_three_objective_decomposition(self) -> None:
        clean_logits = torch.zeros((1, 2))
        perturbed_logits = torch.zeros((1, 2))
        targets = torch.tensor([[1.0, 0.0]])
        clean_features = torch.zeros((1, 3))
        perturbed_features = torch.ones((1, 3))
        terms = stage_three_objective(
            clean_logits=clean_logits,
            perturbed_logits=perturbed_logits,
            targets=targets,
            clean_features=clean_features,
            perturbed_features=perturbed_features,
            lambda_align=0.1,
            lambda_rob=0.5,
        )
        expected_robust = terms.perturbed_classification + 0.1 * terms.alignment
        expected_total = terms.clean_classification + 0.5 * expected_robust
        torch.testing.assert_close(terms.robust, expected_robust)
        torch.testing.assert_close(terms.total, expected_total)


if __name__ == "__main__":
    unittest.main()
