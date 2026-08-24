"""Validation for the paper-defined packet-direction and multi-hot schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from msawf.constants import L_MAX


class SchemaError(ValueError):
    """Raised when data violates the canonical MSAWF contract."""


def _to_tensor(value: Tensor | Sequence[float], *, name: str) -> Tensor:
    try:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be a numeric one-dimensional sequence") from exc
    if tensor.ndim != 1:
        raise SchemaError(f"{name} must be one-dimensional; got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise SchemaError(f"{name} must not be empty")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise SchemaError(f"{name} contains a non-finite value")
    return tensor


def validate_trace(
    trace: Tensor | Sequence[float],
    *,
    max_length: int = L_MAX,
    require_fixed_length: bool = True,
) -> Tensor:
    """Validate direction values and require padding to be one zero suffix.

    Observed packets must be ``-1`` or ``+1``. Once the first zero appears,
    every later value must also be zero. Signed packet magnitudes are rejected.
    """

    tensor = _to_tensor(trace, name="trace")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if require_fixed_length and tensor.numel() != max_length:
        raise SchemaError(
            f"trace length must equal L_max={max_length}; got {tensor.numel()}"
        )
    if not require_fixed_length and tensor.numel() > max_length:
        raise SchemaError(
            f"trace length must not exceed L_max={max_length}; got {tensor.numel()}"
        )

    allowed = (tensor == -1) | (tensor == 0) | (tensor == 1)
    if not allowed.all().item():
        invalid = torch.unique(tensor[~allowed]).detach().cpu().tolist()
        raise SchemaError(
            "trace values must be packet directions {-1,+1} with zero padding; "
            f"found {invalid}"
        )

    zero_positions = torch.nonzero(tensor == 0, as_tuple=False)
    if zero_positions.numel():
        first_zero = int(zero_positions[0].item())
        if torch.any(tensor[first_zero:] != 0).item():
            raise SchemaError("zero padding must form one contiguous suffix")
    return tensor


def validate_labels(
    labels: Tensor | Sequence[float], *, num_classes: int | None = None
) -> Tensor:
    """Validate a one-dimensional binary multi-hot label vector."""

    tensor = _to_tensor(labels, name="labels")
    if num_classes is not None and tensor.numel() != num_classes:
        raise SchemaError(
            f"labels must contain {num_classes} classes; got {tensor.numel()}"
        )
    binary = (tensor == 0) | (tensor == 1)
    if not binary.all().item():
        invalid = torch.unique(tensor[~binary]).detach().cpu().tolist()
        raise SchemaError(f"labels must be binary multi-hot values; found {invalid}")
    return tensor


def observed_length(trace: Tensor | Sequence[float]) -> int:
    """Return the number of observed directions before the zero-padding suffix."""

    tensor = validate_trace(trace, max_length=len(trace), require_fixed_length=False)
    zero_positions = torch.nonzero(tensor == 0, as_tuple=False)
    return int(zero_positions[0].item()) if zero_positions.numel() else tensor.numel()


@dataclass(frozen=True)
class TraceRecord:
    """A validated fixed-length trace and its provenance-bearing label record."""

    trace: Tensor
    labels: Tensor
    trace_id: str
    domain: str
    split_id: str | None = None
    root_provenance_id: str | None = None
    source_provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise SchemaError("trace_id must not be empty")
        if not self.domain.strip():
            raise SchemaError("domain must not be empty")
        root_provenance_id = self.root_provenance_id or self.trace_id
        source_provenance = self.source_provenance or self.domain
        if not root_provenance_id.strip():
            raise SchemaError("root_provenance_id must not be empty")
        if not source_provenance.strip():
            raise SchemaError("source_provenance must not be empty")
        trace = validate_trace(self.trace, max_length=L_MAX, require_fixed_length=True)
        labels = validate_labels(self.labels)
        object.__setattr__(self, "trace", trace.to(dtype=torch.float32).clone())
        object.__setattr__(self, "labels", labels.to(dtype=torch.float32).clone())
        object.__setattr__(self, "root_provenance_id", root_provenance_id)
        object.__setattr__(self, "source_provenance", source_provenance)

    @property
    def observed_length(self) -> int:
        return observed_length(self.trace)

    @property
    def active_label_indices(self) -> tuple[int, ...]:
        """Return sorted active class indices from the multi-hot label."""

        return tuple(
            int(index)
            for index in torch.nonzero(self.labels == 1, as_tuple=False).flatten().tolist()
        )
