"""Canonical multi-label classifier assembled as named registration steps."""

from __future__ import annotations

from torch import Tensor, nn


class Classifier(nn.Module):
    """Map the approved encoder feature ``z`` to multi-label logits."""

    HIDDEN_WIDTH = 512

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        classifier_layers = nn.Sequential()
        classifier_layers.add_module(
            "0",
            nn.Linear(
                in_features=feature_dim,
                out_features=self.HIDDEN_WIDTH,
                bias=False,
            ),
        )
        classifier_layers.add_module(
            "1", nn.BatchNorm1d(num_features=self.HIDDEN_WIDTH)
        )
        classifier_layers.add_module("2", nn.ReLU(True))
        classifier_layers.add_module("3", nn.Dropout(0.7))
        classifier_layers.add_module(
            "4",
            nn.Linear(
                in_features=self.HIDDEN_WIDTH,
                out_features=self.HIDDEN_WIDTH,
                bias=False,
            ),
        )
        classifier_layers.add_module(
            "5", nn.BatchNorm1d(num_features=self.HIDDEN_WIDTH)
        )
        classifier_layers.add_module("6", nn.ReLU(True))
        classifier_layers.add_module("7", nn.Dropout(0.3))
        classifier_layers.add_module(
            "8",
            nn.Linear(
                in_features=self.HIDDEN_WIDTH,
                out_features=num_classes,
            ),
        )
        self.layers = classifier_layers

    def forward(self, feature_vector: Tensor) -> Tensor:
        if feature_vector.ndim != 2:
            raise ValueError(
                "classifier features must have shape [B,F]; "
                f"got {tuple(feature_vector.shape)}"
            )
        if feature_vector.shape[1] != self.feature_dim:
            raise ValueError(
                f"classifier expects feature_dim={self.feature_dim}; "
                f"got {feature_vector.shape[1]}"
            )
        return self.layers(feature_vector)
