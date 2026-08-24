import platform
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from msawf.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    compute_checkpoint_digest,
    save_checkpoint,
)
from msawf.constants import L_MAX, OMEGA
from msawf.evaluation import (
    ClosedWorldEvaluator,
    EarlyDecisionEvaluator,
    EvaluationBatch,
    FixedPrefixEvaluator,
    OpenWorldEvaluator,
    RobustnessEvaluator,
    a_at_k,
    average_decision_length,
    average_observation_ratio,
    degradation_rate,
    early_decision_rate,
    encode_open_world_target,
    multilabel_f1,
    multilabel_precision,
    multilabel_recall,
)
from msawf.evaluation.cli import COMMANDS, initialize_evaluation
from msawf.evaluation.core import PredictionEngine
from msawf.models import Classifier, Encoder
from msawf.utils import load_config, sha256_hex


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_length = L_MAX
        self.feature_dim = 4
        self.projection = nn.Linear(1, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs.mean(dim=2))


class TinyClassifier(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.num_classes = output_dim
        self.linear = nn.Linear(4, output_dim)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(-5.0)
            self.linear.bias[:5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_checkpoint(
    encoder: nn.Module,
    classifier: nn.Module,
    *,
    config_snapshot: dict,
    dataset_fingerprint: str,
    manifest_digest: str,
) -> Checkpoint:
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    document = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_role": "stage3-final",
        "method_variant": "msawf",
        "stage": "stage3",
        "epoch": 50,
        "global_step": 1,
        "encoder_state_dict": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
        "classifier_state_dict": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config_snapshot": config_snapshot,
        "config_digest": sha256_hex(config_snapshot),
        "dataset_fingerprints": {"target": dataset_fingerprint},
        "manifest_digests": {"target": manifest_digest},
        "source_checkpoint_digest": "0" * 64,
        "base_seed": 18,
        "rng_state": {
            "python": random.getstate(),
            "numpy": None,
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "derived_seeds": {},
        },
        "framework_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        },
        "model_dimensions": {
            "input_length": int(getattr(encoder, "input_length")),
            "feature_dim": int(getattr(encoder, "feature_dim")),
            "classifier_output_dim": int(getattr(classifier, "num_classes")),
        },
        "data_contract": {"representation": "packet_direction", "max_length": L_MAX},
        "method_architecture_version": "msawf-cnn-v1",
        "artifact_status": "synthetic_smoke",
        "initialization_seed": 18,
        "content_digest": "",
    }
    document["content_digest"] = compute_checkpoint_digest(document)
    return Checkpoint.from_dict(document)


def make_batch(output_dim: int, *, open_world: bool = False) -> EvaluationBatch:
    traces = torch.zeros((2, L_MAX), dtype=torch.float32)
    traces[0, :8] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1])
    traces[1, :8] = torch.tensor([-1, 1, -1, 1, -1, 1, -1, 1])
    labels = torch.zeros((2, output_dim), dtype=torch.float32)
    if open_world:
        labels[0] = encode_open_world_target((), contains_unmonitored=True)
        labels[1] = encode_open_world_target((0, 3), contains_unmonitored=True)
    else:
        labels[0, :5] = 1
        labels[1, 1:6] = 1
    return EvaluationBatch(
        traces=traces,
        labels=labels,
        trace_ids=("query-0", "query-1"),
        root_provenance_ids=("root-0", "root-1"),
        roles=("query", "query"),
        split_id="split-0",
        dataset_fingerprint="target-dataset",
        manifest_digest="target-manifest",
    )


def make_evaluator(output_dim: int, *, open_world: bool = False):
    source_encoder = TinyEncoder()
    source_classifier = TinyClassifier(output_dim)
    snapshot = {"schema": "evaluation-smoke", "output_dim": output_dim}
    checkpoint = make_checkpoint(
        source_encoder,
        source_classifier,
        config_snapshot=snapshot,
        dataset_fingerprint="target-dataset",
        manifest_digest="target-manifest",
    )
    encoder = TinyEncoder()
    classifier = TinyClassifier(output_dim)
    engine = PredictionEngine(
        checkpoint=checkpoint,
        encoder=encoder,
        classifier=classifier,
        config_digest=sha256_hex(snapshot),
        manifest_digest="target-manifest",
        expected_output_dim=output_dim,
    )
    evaluator = OpenWorldEvaluator(engine) if open_world else ClosedWorldEvaluator(engine)
    return evaluator, encoder, classifier


class EvaluationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_core_metric_toy_example(self) -> None:
        probabilities = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]
        )
        targets = torch.tensor(
            [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.float32
        )
        self.assertAlmostEqual(multilabel_precision(probabilities, targets), 0.2)
        self.assertAlmostEqual(multilabel_recall(probabilities, targets), 0.5)
        self.assertAlmostEqual(multilabel_f1(probabilities, targets), 2 / 7)
        self.assertAlmostEqual(a_at_k(probabilities, targets, k=5), 0.5)
        self.assertEqual(degradation_rate(1.0, 0.75), 25.0)
        self.assertEqual(average_decision_length((5000, 15000)), 10000.0)
        self.assertAlmostEqual(average_observation_ratio((5000, 15000)), 200 / 3)
        self.assertEqual(early_decision_rate((5000, 15000)), 50.0)

    def test_closed_robustness_prefix_and_early_inference_smoke(self) -> None:
        evaluator, encoder, classifier = make_evaluator(100)
        batch = make_batch(100)
        before = [parameter.detach().clone() for parameter in list(encoder.parameters()) + list(classifier.parameters())]
        clean = evaluator.evaluate(batch)
        self.assertEqual(len(clean.predictions), 2)
        self.assertEqual(len(clean.predictions[0].probabilities), 100)
        robustness = RobustnessEvaluator(evaluator).evaluate(batch)
        self.assertEqual(tuple(probability for probability, _ in robustness.by_probability), (0.0, 0.1, 0.2, 0.3))
        prefixes = FixedPrefixEvaluator(evaluator).evaluate(batch)
        self.assertEqual(tuple(prefix for prefix, _ in prefixes.by_prefix), OMEGA)
        early = EarlyDecisionEvaluator(evaluator).evaluate(batch)
        self.assertEqual(tuple(record.decision_length for record in early.predictions), (5000, 5000))
        self.assertEqual(early.aggregate.metrics["edr"], 100.0)
        after = list(encoder.parameters()) + list(classifier.parameters())
        for original, parameter in zip(before, after):
            torch.testing.assert_close(original, parameter)
            self.assertIsNone(parameter.grad)
        self.assertFalse(encoder.training)
        self.assertFalse(classifier.training)

    def test_open_world_101_schema_and_inference_smoke(self) -> None:
        pure = encode_open_world_target((), contains_unmonitored=True)
        mixed = encode_open_world_target((1, 3), contains_unmonitored=True)
        self.assertEqual(tuple(torch.nonzero(pure).flatten().tolist()), (100,))
        self.assertEqual(tuple(torch.nonzero(mixed).flatten().tolist()), (1, 3, 100))
        evaluator, _, _ = make_evaluator(101, open_world=True)
        result = evaluator.evaluate(make_batch(101, open_world=True))
        self.assertEqual(len(result.predictions[0].probabilities), 101)
        self.assertEqual(len(result.predictions[0].top_k_set), 5)

    def test_cli_initialize_only_smoke(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "canonical.json"
        config = load_config(config_path)
        snapshot = asdict(config)
        encoder = Encoder(input_length=config.model.input_length)
        classifier = Classifier(encoder.feature_dim, config.model.closed_world_classes)
        checkpoint = make_checkpoint(
            encoder,
            classifier,
            config_snapshot=snapshot,
            dataset_fingerprint="target-dataset",
            manifest_digest="target-manifest",
        )
        self.assertEqual(
            COMMANDS,
            ("eval_closed_world", "eval_open_world", "eval_robustness", "eval_prefix", "eval_early"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_checkpoint(checkpoint, directory)
            initialized = initialize_evaluation(
                [
                    "eval_closed_world",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--manifest-digest",
                    "target-manifest",
                    "--initialize-only",
                ]
            )
        self.assertEqual(initialized.command, "eval_closed_world")
        self.assertFalse(initialized.training_started)


if __name__ == "__main__":
    unittest.main()
