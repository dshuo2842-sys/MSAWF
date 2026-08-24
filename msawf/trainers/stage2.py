"""Paper Stage II synthetic-to-real bridge training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from msawf.augmentation import PrefixOperator
from msawf.checkpoints import Checkpoint, CheckpointCompatibility, load_for_stage_transition
from msawf.losses import bridge_alpha, bridge_loss, classification_loss
from msawf.runtime import (
    DeterministicBatchLoader,
    ScalarLogger,
    TrainingEngine,
    iter_cycled_pairs,
)

from .common import (
    Stage2TrainerConfig,
    TrainingBatch,
    make_final_checkpoint,
    model_dimensions,
    sample_prefix_length,
)


@dataclass(frozen=True)
class Stage2StepResult:
    total: Tensor
    source_classification: Tensor
    target_classification: Tensor
    alpha: float
    prefix_length: int
    source_trace_ids: tuple[str, ...]
    target_trace_ids: tuple[str, ...]


class Stage2BridgeTrainer:
    def __init__(
        self,
        *,
        encoder: nn.Module,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        source_checkpoint_digest: str,
        base_seed: int = 18,
        config: Stage2TrainerConfig = Stage2TrainerConfig(),
        prefix_operator: PrefixOperator | None = None,
        device: torch.device | str = "cpu",
        logger: ScalarLogger | None = None,
        method_variant: str = "msawf",
    ) -> None:
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.optimizer = optimizer
        self.engine = TrainingEngine(optimizer)
        self.source_checkpoint_digest = source_checkpoint_digest
        self.base_seed = base_seed
        self.config = config
        self.prefix_operator = prefix_operator or PrefixOperator()
        self.device = torch.device(device)
        self.logger = logger or ScalarLogger()
        self.method_variant = method_variant

    @classmethod
    def from_stage1_checkpoint(
        cls,
        checkpoint: Checkpoint,
        *,
        encoder: nn.Module,
        classifier: nn.Module,
        base_seed: int = 18,
        config: Stage2TrainerConfig = Stage2TrainerConfig(),
        device: torch.device | str = "cpu",
        method_variant: str = "msawf",
        config_digest: str | None = None,
    ) -> "Stage2BridgeTrainer":
        dimensions = model_dimensions(encoder, classifier)
        optimizer = load_for_stage_transition(
            checkpoint,
            encoder=encoder,
            classifier=classifier,
            compatibility=CheckpointCompatibility(
                expected_stage="stage1",
                feature_dim=dimensions["feature_dim"],
                classifier_output_dim=dimensions["classifier_output_dim"],
                method_variant=method_variant,
                config_digest=config_digest,
            ),
        )
        return cls(
            encoder=encoder,
            classifier=classifier,
            optimizer=optimizer,
            source_checkpoint_digest=checkpoint.content_digest,
            base_seed=base_seed,
            config=config,
            device=device,
            method_variant=method_variant,
        )

    def train_step(
        self, source_batch: TrainingBatch, target_batch: TrainingBatch, *, epoch: int
    ) -> Stage2StepResult:
        if not 1 <= epoch <= self.config.epochs:
            raise ValueError("Stage II epoch is outside the canonical range")
        source_batch.reject_query()
        target_batch.require_support()
        source_batch = source_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        self.encoder.train()
        self.classifier.train()
        prefix_length = sample_prefix_length(
            self.config.prefix_lengths,
            base_seed=self.base_seed,
            stage="stage2",
            epoch=epoch,
            global_step=self.engine.global_step,
            split_id=target_batch.split_id,
        )
        source_prefix = self.prefix_operator(source_batch.traces, prefix_length)
        target_prefix = self.prefix_operator(target_batch.traces, prefix_length)
        source_logits = self.classifier(self.encoder(source_prefix.unsqueeze(1)))
        target_logits = self.classifier(self.encoder(target_prefix.unsqueeze(1)))
        source_loss = classification_loss(source_logits, source_batch.labels)
        target_loss = classification_loss(target_logits, target_batch.labels)
        alpha = bridge_alpha(epoch, self.config.epochs)
        total = bridge_loss(source_loss, target_loss, alpha)
        self.engine.step(total)
        self.logger.log(
            stage="stage2",
            epoch=epoch,
            global_step=self.engine.global_step,
            values={"L_s": source_loss, "L_t": target_loss, "alpha": alpha, "L_bridge": total},
        )
        return Stage2StepResult(
            total=total.detach(),
            source_classification=source_loss.detach(),
            target_classification=target_loss.detach(),
            alpha=alpha,
            prefix_length=prefix_length,
            source_trace_ids=source_batch.trace_ids,
            target_trace_ids=target_batch.trace_ids,
        )

    def train_epoch(
        self,
        source_loader: DeterministicBatchLoader[object, TrainingBatch],
        target_loader: DeterministicBatchLoader[object, TrainingBatch],
        *,
        epoch: int,
    ) -> list[Stage2StepResult]:
        return [
            self.train_step(pair.source, pair.target, epoch=epoch)
            for pair in iter_cycled_pairs(source_loader, target_loader, epoch=epoch)
        ]

    def final_checkpoint(
        self,
        *,
        epoch: int,
        config_snapshot: Mapping[str, object],
        dataset_fingerprints: Mapping[str, str],
        manifest_digests: Mapping[str, str],
        artifact_status: str = "synthetic_smoke",
    ) -> Checkpoint:
        return make_final_checkpoint(
            stage="stage2",
            final_epoch=self.config.epochs,
            epoch=epoch,
            global_step=self.engine.global_step,
            encoder=self.encoder,
            classifier=self.classifier,
            optimizer=self.optimizer,
            config_snapshot=config_snapshot,
            dataset_fingerprints=dataset_fingerprints,
            manifest_digests=manifest_digests,
            source_checkpoint_digest=self.source_checkpoint_digest,
            base_seed=self.base_seed,
            initialization_seed=None,
            method_variant=self.method_variant,
            artifact_status=artifact_status,
        )
