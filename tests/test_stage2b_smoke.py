import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from msawf.checkpoints import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from msawf.constants import L_MAX
from msawf.methods import get_method_plan, list_method_plans
from msawf.runtime import DeterministicBatchLoader, DeterministicBatchSampler, iter_cycled_pairs
from msawf.runtime.run import initialize_run
from msawf.trainers import Stage1Pretrainer, Stage2BridgeTrainer, Stage3Finetuner, TrainingBatch
from msawf.trainers.common import create_canonical_adamw


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_length = L_MAX
        self.feature_dim = 4
        self.projection = nn.Linear(1, self.feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs.mean(dim=2))


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 3
        self.linear = nn.Linear(4, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_batch(*, role: str, prefix: str, split_id: str | None = None) -> TrainingBatch:
    traces = torch.zeros((2, L_MAX), dtype=torch.float32)
    traces[0, :8] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1])
    traces[1, :8] = torch.tensor([-1, 1, -1, 1, -1, 1, -1, 1])
    labels = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    return TrainingBatch(
        traces=traces,
        labels=labels,
        trace_ids=(f"{prefix}-0", f"{prefix}-1"),
        root_provenance_ids=(f"{prefix}-root-0", f"{prefix}-root-1"),
        roles=(role, role),
        dataset_fingerprint=f"{prefix}-dataset",
        manifest_digest=f"{prefix}-manifest",
        split_id=split_id,
    )


class Stage2BSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_approved_batching_and_cycle_smoke(self) -> None:
        for size in (64, 65, 66, 127, 128, 129):
            first = list(
                DeterministicBatchSampler(
                    size, 64, base_seed=18, domain="smoke", epoch=1
                )
            )
            second = list(
                DeterministicBatchSampler(
                    size, 64, base_seed=18, domain="smoke", epoch=1
                )
            )
            flattened = [index for batch in first for index in batch]
            self.assertEqual(first, second)
            self.assertEqual(sorted(flattened), list(range(size)))
            self.assertTrue(all(len(batch) > 1 for batch in first))
        self.assertEqual(
            DeterministicBatchSampler(
                129, 64, base_seed=18, domain="smoke", epoch=1
            ).batch_sizes,
            (64, 63, 2),
        )

        source = DeterministicBatchLoader(
            list(range(6)),
            batch_size=2,
            base_seed=18,
            domain="source",
            collate_fn=tuple,
        )
        target = DeterministicBatchLoader(
            list(range(2)),
            batch_size=2,
            base_seed=18,
            domain="target",
            collate_fn=tuple,
        )
        pairs = list(iter_cycled_pairs(source, target, epoch=1))
        self.assertEqual(len(pairs), 3)
        self.assertEqual([pair.target_cycle for pair in pairs], [0, 1, 2])

    def test_three_stage_step_and_checkpoint_roundtrip_smoke(self) -> None:
        torch.manual_seed(18)
        encoder = TinyEncoder()
        classifier = TinyClassifier()
        stage1 = Stage1Pretrainer(
            encoder=encoder,
            classifier=classifier,
            optimizer=create_canonical_adamw(encoder, classifier),
            initialization_seed=123,
        )
        source = make_batch(role="source", prefix="source")
        stage1_result = stage1.train_step(source, epoch=1)
        expected_stage1 = (
            stage1_result.clean_classification
            + stage1_result.perturbed_classification
            + 0.1 * stage1_result.consistency
        )
        torch.testing.assert_close(stage1_result.total, expected_stage1)
        self.assertIsNotNone(encoder.projection.weight.grad)
        self.assertIsNotNone(classifier.linear.weight.grad)

        metadata = {
            "config_snapshot": {"schema": "stage2b-smoke", "seed": 18},
            "dataset_fingerprints": {"source": "source-dataset"},
            "manifest_digests": {"source": "source-manifest"},
        }
        stage1_checkpoint = stage1.final_checkpoint(epoch=50, **metadata)

        stage2_encoder = TinyEncoder()
        stage2_classifier = TinyClassifier()
        stage2 = Stage2BridgeTrainer.from_stage1_checkpoint(
            stage1_checkpoint, encoder=stage2_encoder, classifier=stage2_classifier
        )
        self.assertEqual(stage2.optimizer.state, {})
        target = make_batch(role="support", prefix="target", split_id="split-0")
        stage2_result = stage2.train_step(source, target, epoch=1)
        self.assertEqual(stage2_result.alpha, 1.0)
        torch.testing.assert_close(stage2_result.total, stage2_result.source_classification)
        stage2_checkpoint = stage2.final_checkpoint(
            epoch=20,
            config_snapshot=metadata["config_snapshot"],
            dataset_fingerprints={"source": "source-dataset", "target": "target-dataset"},
            manifest_digests={"source": "source-manifest", "target": "target-manifest"},
        )

        stage3_encoder = TinyEncoder()
        stage3_classifier = TinyClassifier()
        stage3 = Stage3Finetuner.from_stage2_checkpoint(
            stage2_checkpoint, encoder=stage3_encoder, classifier=stage3_classifier
        )
        self.assertEqual(stage3.optimizer.state, {})
        stage3_result = stage3.train_step(target, epoch=1)
        expected_robust = stage3_result.perturbed_classification + 0.1 * stage3_result.alignment
        expected_final = stage3_result.clean_classification + 0.5 * expected_robust
        torch.testing.assert_close(stage3_result.robust, expected_robust)
        torch.testing.assert_close(stage3_result.total, expected_final)
        stage3_checkpoint = stage3.final_checkpoint(
            epoch=50,
            config_snapshot=metadata["config_snapshot"],
            dataset_fingerprints={"target": "target-dataset"},
            manifest_digests={"target": "target-manifest"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(stage3_checkpoint, directory)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.content_digest, stage3_checkpoint.content_digest)
        self.assertEqual(loaded.source_checkpoint_digest, stage2_checkpoint.content_digest)

        with self.assertRaises(CheckpointCompatibilityError):
            Stage3Finetuner.from_stage2_checkpoint(
                stage1_checkpoint,
                encoder=TinyEncoder(),
                classifier=TinyClassifier(),
            )

    def test_query_role_is_rejected(self) -> None:
        encoder = TinyEncoder()
        classifier = TinyClassifier()
        trainer = Stage1Pretrainer(
            encoder=encoder,
            classifier=classifier,
            optimizer=create_canonical_adamw(encoder, classifier),
        )
        with self.assertRaises(PermissionError):
            trainer.train_step(make_batch(role="query", prefix="query"), epoch=1)

    def test_method_registry_smoke(self) -> None:
        self.assertEqual(list_method_plans(), ("msawf",))
        self.assertEqual([stage.name for stage in get_method_plan("msawf").stages], ["stage1", "stage2", "stage3"])

    def test_cli_configuration_and_initialization_smoke(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "canonical.json"
        initialized = initialize_run(
            [
                "--config",
                str(config),
                "--class-schema",
                "closed-world",
                "--split-id",
                "split-0",
                "--initialize-only",
            ]
        )
        self.assertEqual(initialized.encoder.feature_dim, 14_592)
        self.assertEqual(initialized.classifier.num_classes, 100)
        self.assertGreater(initialized.parameter_count, 0)
        self.assertEqual(initialized.optimizer.state, {})


if __name__ == "__main__":
    unittest.main()
