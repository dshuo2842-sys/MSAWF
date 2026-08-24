"""Strict immutable-checkpoint inference shared by every evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msawf.checkpoints import Checkpoint, CheckpointCompatibilityError

from .metrics import probabilities_from_logits
from .prediction import EvaluationBatch


def _validate_state(module: nn.Module, state: dict[str, Tensor] | object, name: str) -> None:
    if not isinstance(state, dict):
        state = dict(state)  # type: ignore[arg-type]
    expected = module.state_dict()
    if set(expected) != set(state):
        raise CheckpointCompatibilityError(f"{name} checkpoint keys are incompatible")
    for key, tensor in expected.items():
        observed = state[key]
        if not isinstance(observed, Tensor) or observed.shape != tensor.shape or observed.dtype != tensor.dtype:
            raise CheckpointCompatibilityError(f"{name}.{key} checkpoint tensor is incompatible")


@dataclass
class PredictionEngine:
    checkpoint: Checkpoint
    encoder: nn.Module
    classifier: nn.Module
    config_digest: str
    manifest_digest: str
    expected_output_dim: int
    expected_stage: str = "stage3"
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type != "cpu":
            raise ValueError("current evaluation implementation is CPU-only")
        if self.checkpoint.stage != self.expected_stage:
            raise CheckpointCompatibilityError(
                f"expected {self.expected_stage} checkpoint; got {self.checkpoint.stage}"
            )
        if self.checkpoint.artifact_role != f"{self.expected_stage}-final":
            raise CheckpointCompatibilityError("evaluation requires an immutable final checkpoint")
        if self.checkpoint.config_digest != self.config_digest:
            raise CheckpointCompatibilityError("evaluation config digest is incompatible")
        if self.manifest_digest not in self.checkpoint.manifest_digests.values():
            raise CheckpointCompatibilityError("query manifest is not in checkpoint provenance")
        if self.checkpoint.model_dimensions.get("classifier_output_dim") != self.expected_output_dim:
            raise CheckpointCompatibilityError("classifier output dimension is incompatible")
        if self.checkpoint.data_contract.get("representation") != "packet_direction" or self.checkpoint.data_contract.get("max_length") != 15_000:
            raise CheckpointCompatibilityError("evaluation data contract is incompatible")
        _validate_state(self.encoder, self.checkpoint.encoder_state_dict, "encoder")
        _validate_state(self.classifier, self.checkpoint.classifier_state_dict, "classifier")
        self.encoder.load_state_dict(self.checkpoint.encoder_state_dict, strict=True)
        self.classifier.load_state_dict(self.checkpoint.classifier_state_dict, strict=True)
        self.encoder.to(self.device).eval()
        self.classifier.to(self.device).eval()

    def validate_batch(self, batch: EvaluationBatch) -> None:
        if batch.manifest_digest != self.manifest_digest:
            raise PermissionError("evaluation batch does not belong to the immutable query manifest")
        if batch.dataset_fingerprint not in self.checkpoint.dataset_fingerprints.values():
            raise CheckpointCompatibilityError("evaluation dataset fingerprint is incompatible")
        if batch.labels.shape[1] != self.expected_output_dim:
            raise ValueError("evaluation label dimension is incompatible")

    def infer(self, traces: Tensor) -> Tensor:
        self.encoder.eval()
        self.classifier.eval()
        with torch.inference_mode():
            features = self.encoder(traces.to(self.device).unsqueeze(1))
            logits = self.classifier(features)
            probabilities = probabilities_from_logits(logits)
        return probabilities.cpu()
