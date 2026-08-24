"""Paper Stage I synthetic multi-tab dual-view pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from msawf.augmentation import InsertionTransform, PrefixOperator
from msawf.checkpoints import Checkpoint
from msawf.losses import classification_loss, consistency_loss
from msawf.runtime import ScalarLogger, TrainingEngine

from .common import (
    Stage1TrainerConfig,
    TrainingBatch,
    construct_paired_views,
    make_final_checkpoint,
    sample_prefix_length,
)


@dataclass(frozen=True)
class Stage1StepResult:
    total: Tensor
    clean_classification: Tensor
    perturbed_classification: Tensor
    consistency: Tensor
    prefix_length: int
    trace_ids: tuple[str, ...]
    clean_traces: Tensor
    perturbed_traces: Tensor
    origin_indices: Tensor
    clean_features: Tensor
    perturbed_features: Tensor


class Stage1Pretrainer:
    def __init__(
        self,
        *,
        encoder: nn.Module,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        base_seed: int = 18,
        config: Stage1TrainerConfig = Stage1TrainerConfig(),
        prefix_operator: PrefixOperator | None = None,
        insertion_transform: InsertionTransform | None = None,
        device: torch.device | str = "cpu",
        logger: ScalarLogger | None = None,
        initialization_seed: int | None = None,
        method_variant: str = "msawf",
    ) -> None:
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.optimizer = optimizer
        self.engine = TrainingEngine(optimizer)
        self.base_seed = base_seed
        self.config = config
        self.prefix_operator = prefix_operator or PrefixOperator()
        self.insertion_transform = insertion_transform or InsertionTransform(0.2)
        if self.insertion_transform.probability != config.insertion_probability:
            raise ValueError("Stage I insertion probability must equal the canonical config")
        self.device = torch.device(device)
        self.logger = logger or ScalarLogger()
        self.initialization_seed = initialization_seed
        self.method_variant = method_variant

    def train_step(self, batch: TrainingBatch, *, epoch: int) -> Stage1StepResult:
        if not 1 <= epoch <= self.config.epochs:
            raise ValueError("Stage I epoch is outside the canonical range")
        batch.reject_query()
        batch = batch.to(self.device)
        self.encoder.train()
        self.classifier.train()
        coordinate = self.engine.global_step
        prefix_length = sample_prefix_length(
            self.config.prefix_lengths,
            base_seed=self.base_seed,
            stage="stage1",
            epoch=epoch,
            global_step=coordinate,
            split_id=batch.split_id,
        )
        clean, perturbed, origins = construct_paired_views(
            batch,
            prefix_operator=self.prefix_operator,
            insertion_transform=self.insertion_transform,
            prefix_length=prefix_length,
            base_seed=self.base_seed,
            stage="stage1",
            epoch=epoch,
            global_step=coordinate,
        )
        clean_features = self.encoder(clean.unsqueeze(1))
        perturbed_features = self.encoder(perturbed.unsqueeze(1))
        clean_logits = self.classifier(clean_features)
        perturbed_logits = self.classifier(perturbed_features)
        clean_loss = classification_loss(clean_logits, batch.labels)
        perturbed_loss = classification_loss(perturbed_logits, batch.labels)
        con = consistency_loss(clean_features, perturbed_features)
        total = clean_loss + perturbed_loss + self.config.lambda_con * con
        self.engine.step(total)
        self.logger.log(
            stage="stage1",
            epoch=epoch,
            global_step=self.engine.global_step,
            values={"L_cls_clean": clean_loss, "L_cls_perturbed": perturbed_loss, "L_con": con, "L_pre": total},
        )
        return Stage1StepResult(
            total=total.detach(),
            clean_classification=clean_loss.detach(),
            perturbed_classification=perturbed_loss.detach(),
            consistency=con.detach(),
            prefix_length=prefix_length,
            trace_ids=batch.trace_ids,
            clean_traces=clean.detach(),
            perturbed_traces=perturbed.detach(),
            origin_indices=origins.detach(),
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
            stage="stage1",
            final_epoch=self.config.epochs,
            epoch=epoch,
            global_step=self.engine.global_step,
            encoder=self.encoder,
            classifier=self.classifier,
            optimizer=self.optimizer,
            config_snapshot=config_snapshot,
            dataset_fingerprints=dataset_fingerprints,
            manifest_digests=manifest_digests,
            source_checkpoint_digest=None,
            base_seed=self.base_seed,
            initialization_seed=self.initialization_seed,
            method_variant=self.method_variant,
            artifact_status=artifact_status,
        )
