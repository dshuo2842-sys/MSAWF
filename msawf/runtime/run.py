"""Thin CLI composition for configuration validation and model initialization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from msawf.methods import MethodPlan, get_method_plan
from msawf.models import Classifier, Encoder
from msawf.trainers.common import create_canonical_adamw, initialize_modules
from msawf.utils import ExperimentConfig, load_config


@dataclass(frozen=True)
class InitializedRun:
    config: ExperimentConfig
    encoder: Encoder
    classifier: Classifier
    optimizer: torch.optim.AdamW
    method_plan: MethodPlan
    initialization_seed: int
    class_schema: str
    split_id: str
    stage: str

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.encoder, self.classifier)
            for parameter in module.parameters()
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msawf-train",
        description="Validate an MSAWF config and initialize a declared stage without training.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--class-schema", choices=("closed-world", "open-world"), default="closed-world"
    )
    parser.add_argument("--split-id", default="split-0")
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3"),
        default="stage1",
        help="Declare which stage contract is being initialized.",
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Validate and initialize only; this CLI does not start dataset training.",
    )
    return parser


def initialize_run(argv: Sequence[str] | None = None) -> InitializedRun:
    args = build_parser().parse_args(argv)
    if not args.initialize_only:
        raise ValueError("the public CLI currently requires --initialize-only")
    config = load_config(args.config)
    num_classes = (
        config.model.closed_world_classes
        if args.class_schema == "closed-world"
        else config.model.open_world_classes
    )

    def factory() -> tuple[Encoder, Classifier]:
        encoder = Encoder(input_length=config.model.input_length)
        classifier = Classifier(encoder.feature_dim, num_classes)
        return encoder, classifier

    (encoder, classifier), initialization_seed = initialize_modules(
        factory,
        base_seed=config.training.seed,
        method_variant="msawf",
        class_schema=args.class_schema,
        split_id=args.split_id,
    )
    optimizer = create_canonical_adamw(encoder, classifier)
    return InitializedRun(
        config=config,
        encoder=encoder,
        classifier=classifier,
        optimizer=optimizer,
        method_plan=get_method_plan("msawf"),
        initialization_seed=initialization_seed,
        class_schema=args.class_schema,
        split_id=args.split_id,
        stage=args.stage,
    )


def main(argv: Sequence[str] | None = None) -> int:
    initialized = initialize_run(argv)
    print(
        json.dumps(
            {
                "status": "initialized",
                "method": initialized.method_plan.method_id,
                "stage": initialized.stage,
                "class_schema": initialized.class_schema,
                "split_id": initialized.split_id,
                "initialization_seed": initialized.initialization_seed,
                "parameter_count": initialized.parameter_count,
                "training_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
