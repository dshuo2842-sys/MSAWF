"""Paper Stage III few-shot target-domain robust adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from msawf.augmentation import InsertionTransform, PrefixOperator
from msawf.checkpoints import Checkpoint, CheckpointCompatibility, load_for_stage_transition
from msawf.losses import stage_three_objective
from msawf.runtime import ScalarLogger, TrainingEngine

from .common import (
    Stage3TrainerConfig,
    TrainingBatch,
    construct_paired_views,
    make_final_checkpoint,
    model_dimensions,
    sample_prefix_length,
)


@dataclass(frozen=True)
class Stage3StepResult:
    total: Tensor
    clean_classification: Tensor
    perturbed_classification: Tensor
    alignment: Tensor
    robust: Tensor
    prefix_length: int
    trace_ids: tuple[str, ...]
    clean_traces: Tensor
    perturbed_traces: Tensor
    clean_features: Tensor
    perturbed_features: Tensor


class Stage3Finetuner:
    def __init__(
        self,
        *,
        encoder: nn.Module,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        source_checkpoint_digest: str,
        base_seed: int = 18,
        config: Stage3TrainerConfig = Stage3TrainerConfig(),
        prefix_operator: PrefixOperator | None = None,
        insertion_transform: InsertionTransform | None = None,
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
        self.insertion_transform = insertion_transform or InsertionTransform(0.2)
        if self.insertion_transform.probability != config.insertion_probability:
            raise ValueError("Stage III insertion probability must equal canonical config")
        self.device = torch.device(device)
        self.logger = logger or ScalarLogger()
        self.method_variant = method_variant

    @classmethod
    def from_stage2_checkpoint(
        cls,
        checkpoint: Checkpoint,
        *,
        encoder: nn.Module,
        classifier: nn.Module,
        base_seed: int = 18,
        config: Stage3TrainerConfig = Stage3TrainerConfig(),
        device: torch.device | str = "cpu",
        method_variant: str = "msawf",
        config_digest: str | None = None,
    ) -> "Stage3Finetuner":
        dimensions = model_dimensions(encoder, classifier)
        optimizer = load_for_stage_transition(
            checkpoint,
            encoder=encoder,
            classifier=classifier,
            compatibility=CheckpointCompatibility(
                expected_stage="stage2",
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

    def train_step(self, batch: TrainingBatch, *, epoch: int) -> Stage3StepResult:
        if not 1 <= epoch <= self.config.epochs:
            raise ValueError("Stage III epoch is outside the canonical range")
        batch.require_support()
        batch = batch.to(self.device)
        self.encoder.train()
        self.classifier.train()
        coordinate = self.engine.global_step
        prefix_length = sample_prefix_length(
            self.config.prefix_lengths,
            base_seed=self.base_seed,
            stage="stage3",
            epoch=epoch,
            global_step=coordinate,
            split_id=batch.split_id,
        )
        clean, perturbed, _ = construct_paired_views(
            batch,
            prefix_operator=self.prefix_operator,
            insertion_transform=self.insertion_transform,
            prefix_length=prefix_length,
            base_seed=self.base_seed,
            stage="stage3",
            epoch=epoch,
            global_step=coordinate,
        )
        clean_features = self.encoder(clean.unsqueeze(1))
        perturbed_features = self.encoder(perturbed.unsqueeze(1))
        clean_logits = self.classifier(clean_features)
        perturbed_logits = self.classifier(perturbed_features)
        losses = stage_three_objective(
            clean_logits=clean_logits,
            perturbed_logits=perturbed_logits,
            targets=batch.labels,
            clean_features=clean_features,
            perturbed_features=perturbed_features,
            lambda_align=self.config.lambda_align,
            lambda_rob=self.config.lambda_rob,
        )
        self.engine.step(losses.total)
        self.logger.log(
            stage="stage3",
            epoch=epoch,
            global_step=self.engine.global_step,
            values={
                "L_clean": losses.clean_classification,
                "L_perturbed": losses.perturbed_classification,
                "L_align": losses.alignment,
                "L_rob": losses.robust,
                "L_final": losses.total,
            },
        )
        return Stage3StepResult(
            total=losses.total.detach(),
            clean_classification=losses.clean_classification.detach(),
            perturbed_classification=losses.perturbed_classification.detach(),
            alignment=losses.alignment.detach(),
            robust=losses.robust.detach(),
            prefix_length=prefix_length,
            trace_ids=batch.trace_ids,
            clean_traces=clean.detach(),
            perturbed_traces=perturbed.detach(),
            clean_features=clean_features.detach(),
            perturbed_features=perturbed_features.detach(),
        )

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
            stage="stage3",
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
