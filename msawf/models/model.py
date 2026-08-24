"""Composed MSAWF model with explicit feature and logit outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msawf.constants import CLOSED_WORLD_CLASSES, L_MAX

from .classifier import Classifier
from .encoder import Encoder


@dataclass(frozen=True)
class ModelOutput:
    features: Tensor
    logits: Tensor

    @property
    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits)


class MSAWFModel(nn.Module):
    """Compose independent ``E_theta`` and ``C_phi`` modules."""

    def __init__(
        self,
        *,
        num_classes: int = CLOSED_WORLD_CLASSES,
        input_length: int = L_MAX,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_length=input_length)
        self.classifier = Classifier(
            feature_dim=self.encoder.feature_dim, num_classes=num_classes
        )

    def forward(self, inputs: Tensor) -> ModelOutput:
        features = self.encoder(inputs)
        logits = self.classifier(features)
        return ModelOutput(features=features, logits=logits)
