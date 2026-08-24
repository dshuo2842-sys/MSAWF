import unittest
from pathlib import Path

from msawf.utils import UnresolvedConfigError, load_config


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "configs" / "canonical.json"

    def test_approved_values_load_and_validate(self) -> None:
        config = load_config(self.path)
        self.assertEqual(config.model.feature_dim, 14_592)
        self.assertEqual(config.optimization.learning_rate, 1e-3)
        self.assertEqual(config.optimization.weight_decay, 1e-4)
        self.assertEqual(config.augmentation.stage1_probability, 0.2)
        self.assertEqual(config.model.open_world_classes, 101)

    def test_required_unresolved_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnresolvedConfigError, "source_dataset"):
            load_config(self.path, required_fields=("paths.source_dataset",))

    def test_resolved_required_field_is_accepted(self) -> None:
        config = load_config(
            self.path, required_fields=("optimization.learning_rate",)
        )
        self.assertEqual(config.optimization.scheduler, None)


if __name__ == "__main__":
    unittest.main()
