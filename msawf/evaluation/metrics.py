"""Paper-defined multi-label and streaming evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


def _validate_probabilities_targets(probabilities: Tensor, targets: Tensor) -> None:
    if probabilities.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same [N,C] shape")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("evaluation tensors must not be empty")
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("probabilities must be finite")
    if not ((probabilities >= 0) & (probabilities <= 1)).all().item():
        raise ValueError("probabilities must be within [0,1]")
    if not (((targets == 0) | (targets == 1)).all().item()):
        raise ValueError("targets must be binary multi-hot values")


def probabilities_from_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or not torch.isfinite(logits).all().item():
        raise ValueError("logits must be a finite [N,C] tensor")
    return torch.sigmoid(logits)


def _sample_counts(
    probabilities: Tensor, targets: Tensor, threshold: float
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_probabilities_targets(probabilities, targets)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0,1]")
    predicted = probabilities >= threshold
    truth = targets == 1
    tp = (predicted & truth).sum(dim=1).to(dtype=torch.float64)
    fp = (predicted & ~truth).sum(dim=1).to(dtype=torch.float64)
    fn = (~predicted & truth).sum(dim=1).to(dtype=torch.float64)
    return tp, fp, fn


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def multilabel_precision(
    probabilities: Tensor, targets: Tensor, *, threshold: float = 0.5
) -> float:
    tp, fp, _ = _sample_counts(probabilities, targets, threshold)
    return float(_safe_ratio(tp, tp + fp).mean().item())


def multilabel_recall(
    probabilities: Tensor, targets: Tensor, *, threshold: float = 0.5
) -> float:
    tp, _, fn = _sample_counts(probabilities, targets, threshold)
    return float(_safe_ratio(tp, tp + fn).mean().item())


def multilabel_f1(
    probabilities: Tensor, targets: Tensor, *, threshold: float = 0.5
) -> float:
    tp, fp, fn = _sample_counts(probabilities, targets, threshold)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return float(_safe_ratio(2 * precision * recall, precision + recall).mean().item())


def top_k_sets(probabilities: Tensor, *, k: int = 5) -> tuple[tuple[int, ...], ...]:
    if probabilities.ndim != 2 or not torch.isfinite(probabilities).all().item():
        raise ValueError("probabilities must be a finite [N,C] tensor")
    if k <= 0 or k > probabilities.shape[1]:
        raise ValueError("k must be between 1 and the class count")
    indices = torch.topk(probabilities, k=k, dim=1).indices.detach().cpu().tolist()
    return tuple(tuple(int(index) for index in row) for row in indices)


def a_at_k(probabilities: Tensor, targets: Tensor, *, k: int = 5) -> float:
    _validate_probabilities_targets(probabilities, targets)
    top_sets = top_k_sets(probabilities, k=k)
    scores: list[float] = []
    for row, top_set in zip(targets, top_sets):
        true_set = set(torch.nonzero(row == 1, as_tuple=False).flatten().tolist())
        scores.append(len(true_set.intersection(top_set)) / len(true_set) if true_set else 0.0)
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class MetricValues:
    precision: float
    recall: float
    f1: float
    a_at_k: float

    def to_dict(self, *, k: int = 5) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            f"a_at_{k}": self.a_at_k,
        }


def evaluate_multilabel_metrics(
    probabilities: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    k: int = 5,
) -> MetricValues:
    return MetricValues(
        precision=multilabel_precision(probabilities, targets, threshold=threshold),
        recall=multilabel_recall(probabilities, targets, threshold=threshold),
        f1=multilabel_f1(probabilities, targets, threshold=threshold),
        a_at_k=a_at_k(probabilities, targets, k=k),
    )


def degradation_rate(clean_f1: float, strong_noise_f1: float) -> float:
    if not all(math.isfinite(value) and value >= 0 for value in (clean_f1, strong_noise_f1)):
        raise ValueError("F1 values must be finite and non-negative")
    return ((clean_f1 - strong_noise_f1) / clean_f1 * 100.0) if clean_f1 > 0 else 0.0


def average_decision_length(decision_lengths: Sequence[int]) -> float:
    if not decision_lengths or any(length <= 0 for length in decision_lengths):
        raise ValueError("decision lengths must be a non-empty positive sequence")
    return sum(decision_lengths) / len(decision_lengths)


def average_observation_ratio(
    decision_lengths: Sequence[int], *, max_length: int = 15_000
) -> float:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return average_decision_length(decision_lengths) / max_length * 100.0


def early_decision_rate(
    decision_lengths: Sequence[int], *, max_length: int = 15_000
) -> float:
    if not decision_lengths or any(length <= 0 or length > max_length for length in decision_lengths):
        raise ValueError("decision lengths must lie within the observation window")
    return sum(length < max_length for length in decision_lengths) / len(decision_lengths) * 100.0


def split_mean_std(values: Sequence[float], *, correction: int) -> tuple[float, float]:
    """Aggregate splits while forcing the caller to state the std convention."""

    if not values or correction not in (0, 1) or len(values) <= correction:
        raise ValueError("values and correction do not define a valid standard deviation")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return float(tensor.mean().item()), float(tensor.std(correction=correction).item())
