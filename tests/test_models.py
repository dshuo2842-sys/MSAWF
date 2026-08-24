import unittest

import torch

from msawf.constants import L_MAX, OPEN_WORLD_CLASSES
from msawf.models import Classifier, Encoder, MSAWFModel, ModelOutput


class ModelInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_deterministic_encoder_dimension(self) -> None:
        encoder = Encoder()
        self.assertEqual(Encoder.calculate_output_length(L_MAX), 57)
        self.assertEqual(encoder.output_length, 57)
        self.assertEqual(encoder.feature_dim, 14_592)

    def test_composed_model_returns_features_and_logits(self) -> None:
        model = MSAWFModel(num_classes=100).cpu().eval()
        inputs = torch.zeros((1, 1, L_MAX), dtype=torch.float32)
        with torch.no_grad():
            output = model(inputs)
        self.assertIsInstance(output, ModelOutput)
        self.assertEqual(tuple(output.features.shape), (1, 14_592))
        self.assertEqual(tuple(output.logits.shape), (1, 100))
        self.assertEqual(tuple(output.probabilities.shape), (1, 100))
        self.assertTrue(torch.all(output.probabilities >= 0).item())
        self.assertTrue(torch.all(output.probabilities <= 1).item())

    def test_classifier_supports_open_world_101_outputs(self) -> None:
        classifier = Classifier(14_592, OPEN_WORLD_CLASSES).cpu().eval()
        features = torch.zeros((1, 14_592), dtype=torch.float32)
        with torch.no_grad():
            logits = classifier(features)
        self.assertEqual(tuple(logits.shape), (1, 101))


if __name__ == "__main__":
    unittest.main()
