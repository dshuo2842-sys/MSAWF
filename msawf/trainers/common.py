"""Shared trainer contracts without paper-specific objective composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor, nn

from msawf.augmentation import InsertionTransform, PrefixOperator
from msawf.checkpoints import Checkpoint, build_checkpoint
from msawf.constants import L_MAX, OMEGA
from msawf.data import TraceRecord, validate_labels, validate_trace
from msawf.utils import derive_seed, make_generator
from msawf.utils.hashing import StableRNG

ModulePairT = TypeVar("ModulePairT", bound=tuple[nn.Module, nn.Module])


@dataclass(frozen=True)
class TrainingBatch:
    """Canonical trainer input with immutable identity and role metadata."""

    traces: Tensor
    labels: Tensor
    trace_ids: tuple[str, ...]
    root_provenance_ids: tuple[str, ...]
    roles: tuple[str, ...]
    dataset_fingerprint: str
    manifest_digest: str
    split_id: str | None = None

    def __post_init__(self) -> None:
        if self.traces.ndim != 2 or self.traces.shape[1] != L_MAX:
            raise ValueError(f"traces must have shape [B,{L_MAX}]")
        if self.labels.ndim != 2 or self.labels.shape[0] != self.traces.shape[0]:
            raise ValueError("labels must have shape [B,C] and match traces")
        batch_size = self.traces.shape[0]
        if batch_size < 2:
            raise ValueError("training batches must contain at least two records")
        metadata_lengths = {
            len(self.trace_ids),
            len(self.root_provenance_ids),
            len(self.roles),
        }
        if metadata_lengths != {batch_size}:
            raise ValueError("identity and role metadata must match batch size")
        if len(set(self.trace_ids)) != batch_size:
            raise ValueError("training batch contains duplicate trace IDs")
        if any(not value.strip() for value in self.trace_ids + self.root_provenance_ids):
            raise ValueError("trace and root provenance IDs must not be empty")
        if any(not role.strip() for role in self.roles):
            raise ValueError("training roles must not be empty")
        if not self.dataset_fingerprint.strip() or not self.manifest_digest.strip():
            raise ValueError("dataset fingerprint and manifest digest are required")
        for trace in self.traces:
            validate_trace(trace, max_length=L_MAX, require_fixed_length=True)
        for labels in self.labels:
            validate_labels(labels, num_classes=self.labels.shape[1])

    @classmethod
    def from_records(
        cls,
        records: Sequence[TraceRecord],
        *,
        role: str,
        dataset_fingerprint: str,
        manifest_digest: str,
        split_id: str | None = None,
    ) -> "TrainingBatch":
        if len(records) < 2:
            raise ValueError("training batches must contain at least two records")
        return cls(
            traces=torch.stack([record.trace for record in records]),
            labels=torch.stack([record.labels for record in records]),
            trace_ids=tuple(record.trace_id for record in records),
            root_provenance_ids=tuple(
                str(record.root_provenance_id) for record in records
            ),
            roles=tuple(role for _ in records),
            dataset_fingerprint=dataset_fingerprint,
            manifest_digest=manifest_digest,
            split_id=split_id,
        )

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            traces=self.traces.to(device=device),
            labels=self.labels.to(device=device),
            trace_ids=self.trace_ids,
            root_provenance_ids=self.root_provenance_ids,
            roles=self.roles,
            dataset_fingerprint=self.dataset_fingerprint,
            manifest_digest=self.manifest_digest,
            split_id=self.split_id,
        )

    def reject_query(self) -> None:
        if "query" in self.roles:
            raise PermissionError("training batches must never contain query records")

    def require_support(self) -> None:
        self.reject_query()
        if set(self.roles) != {"support"}:
            raise PermissionError("target training batches must contain support records only")


@dataclass(frozen=True)
class Stage1TrainerConfig:
    epochs: int = 50
    lambda_con: float = 0.1
    insertion_probability: float = 0.2
    prefix_lengths: tuple[int, ...] = OMEGA

    def __post_init__(self) -> None:
        if (self.epochs, self.lambda_con, self.insertion_probability) != (50, 0.1, 0.2):
            raise ValueError("canonical Stage I config must be epochs=50, lambda_con=0.1, p=0.2")
        if self.prefix_lengths != OMEGA:
            raise ValueError("canonical Stage I prefixes must equal Omega")


@dataclass(frozen=True)
class Stage2TrainerConfig:
    epochs: int = 20
    prefix_lengths: tuple[int, ...] = OMEGA

    def __post_init__(self) -> None:
        if self.epochs != 20 or self.prefix_lengths != OMEGA:
            raise ValueError("canonical Stage II config must use 20 epochs and Omega")


@dataclass(frozen=True)
class Stage3TrainerConfig:
    epochs: int = 50
    lambda_align: float = 0.1
    lambda_rob: float = 0.5
    insertion_probability: float = 0.2
    prefix_lengths: tuple[int, ...] = OMEGA

    def __post_init__(self) -> None:
        values = (self.epochs, self.lambda_align, self.lambda_rob, self.insertion_probability)
        if values != (50, 0.1, 0.5, 0.2):
            raise ValueError("canonical Stage III config must be 50/0.1/0.5/0.2")
        if self.prefix_lengths != OMEGA:
            raise ValueError("canonical Stage III prefixes must equal Omega")


def create_canonical_adamw(encoder: nn.Module, classifier: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )


def initialize_modules(
    factory: Callable[[], ModulePairT],
    *,
    base_seed: int,
    method_variant: str,
    class_schema: str,
    split_id: str,
) -> tuple[ModulePairT, int]:
    """Use PyTorch defaults inside the approved dedicated initialization domain."""

    initialization_seed = derive_seed(
        "trainer-model-initialization-v1",
        base_seed,
        method_variant,
        class_schema,
        split_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        modules = factory()
    return modules, initialization_seed


def sample_prefix_length(
    prefix_lengths: tuple[int, ...],
    *,
    base_seed: int,
    stage: str,
    epoch: int,
    global_step: int,
    split_id: str | None,
) -> int:
    seed = derive_seed(
        "trainer-prefix-v1", base_seed, stage, split_id or "source", epoch, global_step
    )
    return prefix_lengths[StableRNG(seed).randbelow(len(prefix_lengths))]


def construct_paired_views(
    batch: TrainingBatch,
    *,
    prefix_operator: PrefixOperator,
    insertion_transform: InsertionTransform,
    prefix_length: int,
    base_seed: int,
    stage: str,
    epoch: int,
    global_step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    clean = prefix_operator(batch.traces, prefix_length)
    perturbed_rows: list[Tensor] = []
    origins: list[Tensor] = []
    for index, root_id in enumerate(batch.root_provenance_ids):
        seed = derive_seed(
            "trainer-insertion-v1",
            base_seed,
            stage,
            "perturbed",
            batch.split_id or "source",
            epoch,
            global_step,
            root_id,
            prefix_length,
        )
        result = insertion_transform(
            clean[index], batch.labels[index], generator=make_generator(seed)
        )
        if not torch.equal(result.labels, batch.labels[index]):
            raise RuntimeError("insertion transform changed a training label")
        perturbed_rows.append(result.trace)
        origins.append(result.origin_indices)
    return clean, torch.stack(perturbed_rows), torch.stack(origins)


def model_dimensions(encoder: nn.Module, classifier: nn.Module) -> dict[str, int]:
    try:
        return {
            "input_length": int(getattr(encoder, "input_length")),
            "feature_dim": int(getattr(encoder, "feature_dim")),
            "classifier_output_dim": int(getattr(classifier, "num_classes")),
        }
    except (TypeError, AttributeError) as exc:
        raise ValueError("encoder/classifier must expose canonical model dimensions") from exc


def make_final_checkpoint(
    *,
    stage: str,
    final_epoch: int,
    epoch: int,
    global_step: int,
    encoder: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    config_snapshot: Mapping[str, object],
    dataset_fingerprints: Mapping[str, str],
    manifest_digests: Mapping[str, str],
    source_checkpoint_digest: str | None,
    base_seed: int,
    initialization_seed: int | None,
    method_variant: str,
    artifact_status: str,
) -> Checkpoint:
    if epoch != final_epoch:
        raise ValueError("canonical checkpoint output is restricted to the final epoch")
    return build_checkpoint(
        stage=stage,
        artifact_role=f"{stage}-final",
        epoch=epoch,
        global_step=global_step,
        encoder=encoder,
        classifier=classifier,
        optimizer=optimizer,
        config_snapshot=config_snapshot,
        dataset_fingerprints=dataset_fingerprints,
        manifest_digests=manifest_digests,
        source_checkpoint_digest=source_checkpoint_digest,
        base_seed=base_seed,
        model_dimensions=model_dimensions(encoder, classifier),
        method_variant=method_variant,
        artifact_status=artifact_status,
        initialization_seed=initialization_seed,
    )
