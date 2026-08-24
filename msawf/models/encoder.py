"""Canonical direction-trace encoder with an explicitly assembled feature stack."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from msawf.constants import L_MAX


def _conv_output_length(
    length: int, *, kernel_size: int, stride: int, padding: int, dilation: int = 1
) -> int:
    return (
        (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
    ) + 1


def _pool_output_length(
    length: int, *, kernel_size: int, stride: int, padding: int = 0
) -> int:
    return ((length + 2 * padding - (kernel_size - 1) - 1) // stride) + 1


def _make_nonlinearity(kind: str) -> nn.Module:
    if kind == "elu":
        return nn.ELU(True)
    if kind == "relu":
        return nn.ReLU(True)
    raise ValueError(f"unsupported activation: {kind}")


class DirectionalFeatureStage(nn.Module):
    """Apply the approved two-convolution feature-extraction stage."""

    def __init__(
        self, input_channels: int, output_channels: int, nonlinearity: str
    ) -> None:
        super().__init__()

        operations = nn.Sequential()
        operations.add_module(
            "0",
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=8,
                stride=1,
                padding=4,
                bias=False,
            ),
        )
        operations.add_module(
            "1", nn.BatchNorm1d(num_features=output_channels)
        )
        operations.add_module("2", _make_nonlinearity(nonlinearity))
        operations.add_module(
            "3",
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=8,
                stride=1,
                padding=4,
                bias=False,
            ),
        )
        operations.add_module(
            "4", nn.BatchNorm1d(num_features=output_channels)
        )
        operations.add_module("5", _make_nonlinearity(nonlinearity))
        operations.add_module(
            "6", nn.MaxPool1d(kernel_size=8, stride=4, padding=0)
        )
        operations.add_module("7", nn.Dropout(0.1))
        self.block = operations

    def forward(self, direction_features: Tensor) -> Tensor:
        return self.block(direction_features)


# Preserve the original public symbol without retaining the historical class expression.
ConvBlock = DirectionalFeatureStage


class Encoder(nn.Module):
    """Four-stage CNN whose flattened output is the canonical feature ``z``."""

    CHANNELS = (32, 64, 128, 256)
    ACTIVATIONS = ("elu", "relu", "relu", "relu")

    def __init__(self, input_length: int = L_MAX) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        self.input_length = input_length
        self.output_length = self.calculate_output_length(input_length)
        self.feature_dim = self.CHANNELS[-1] * self.output_length

        feature_stages = nn.Sequential()
        previous_width = 1
        for stage_index, (next_width, nonlinearity) in enumerate(
            zip(self.CHANNELS, self.ACTIVATIONS)
        ):
            feature_stages.add_module(
                str(stage_index),
                DirectionalFeatureStage(
                    input_channels=previous_width,
                    output_channels=next_width,
                    nonlinearity=nonlinearity,
                ),
            )
            previous_width = next_width
        self.blocks = feature_stages
        self.flatten = nn.Flatten(start_dim=1)

    @staticmethod
    def calculate_output_length(input_length: int) -> int:
        """Calculate the temporal length using the exact approved operators."""

        if input_length <= 0:
            raise ValueError("input_length must be positive")
        length = input_length
        for _ in range(4):
            length = _conv_output_length(
                length, kernel_size=8, stride=1, padding=4
            )
            length = _conv_output_length(
                length, kernel_size=8, stride=1, padding=4
            )
            length = _pool_output_length(length, kernel_size=8, stride=4)
            if length <= 0:
                raise ValueError("input_length is too short for the encoder topology")
        return length

    def forward(self, direction_trace: Tensor) -> Tensor:
        if direction_trace.ndim != 3:
            raise ValueError(
                "encoder input must have shape [B,1,L]; "
                f"got {tuple(direction_trace.shape)}"
            )
        if direction_trace.shape[1] != 1:
            raise ValueError(
                f"encoder expects one channel; got {direction_trace.shape[1]}"
            )
        if direction_trace.shape[2] != self.input_length:
            raise ValueError(
                f"encoder expects length {self.input_length}; "
                f"got {direction_trace.shape[2]}"
            )
        encoded_trace = self.blocks(direction_trace)
        feature_vector = self.flatten(encoded_trace)
        if feature_vector.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"calculated feature_dim={self.feature_dim}, "
                f"observed {feature_vector.shape[1]}"
            )
        return feature_vector
