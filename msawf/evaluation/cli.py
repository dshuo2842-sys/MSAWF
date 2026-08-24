"""Initialize non-defense evaluators from authenticated checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from msawf.checkpoints import load_checkpoint
from msawf.models import Classifier, Encoder
from msawf.utils import load_config, sha256_hex

from .closed_world import ClosedWorldEvaluator
from .core import PredictionEngine
from .early_decision import EarlyDecisionEvaluator
from .open_world import OpenWorldEvaluator
from .prefix import FixedPrefixEvaluator
from .robustness import RobustnessEvaluator

COMMANDS = (
    "eval_closed_world",
    "eval_open_world",
    "eval_robustness",
    "eval_prefix",
    "eval_early",
)


@dataclass(frozen=True)
class InitializedEvaluation:
    command: str
    evaluator: object
    checkpoint_digest: str
    config_digest: str
    manifest_digest: str
    training_started: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msawf-eval",
        description="Validate and initialize an immutable MSAWF evaluation protocol.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-digest", required=True)
    parser.add_argument("--initialize-only", action="store_true")
    return parser


def initialize_evaluation(argv: Sequence[str] | None = None) -> InitializedEvaluation:
    args = build_parser().parse_args(argv)
    if not args.initialize_only:
        raise ValueError("public data evaluation is not started without --initialize-only")
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    config_digest = sha256_hex(asdict(config))
    open_world = args.command == "eval_open_world"
    output_dim = config.model.open_world_classes if open_world else config.model.closed_world_classes
    encoder = Encoder(input_length=config.model.input_length)
    classifier = Classifier(encoder.feature_dim, output_dim)
    engine = PredictionEngine(
        checkpoint=checkpoint,
        encoder=encoder,
        classifier=classifier,
        config_digest=config_digest,
        manifest_digest=args.manifest_digest,
        expected_output_dim=output_dim,
        expected_stage="stage3",
        device="cpu",
    )
    base = OpenWorldEvaluator(engine) if open_world else ClosedWorldEvaluator(engine)
    if args.command == "eval_robustness":
        evaluator: object = RobustnessEvaluator(base)
    elif args.command == "eval_prefix":
        evaluator = FixedPrefixEvaluator(base)
    elif args.command == "eval_early":
        evaluator = EarlyDecisionEvaluator(base)
    else:
        evaluator = base
    return InitializedEvaluation(
        command=args.command,
        evaluator=evaluator,
        checkpoint_digest=checkpoint.content_digest,
        config_digest=config_digest,
        manifest_digest=args.manifest_digest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    initialized = initialize_evaluation(argv)
    print(
        json.dumps(
            {
                "status": "initialized",
                "command": initialized.command,
                "checkpoint_digest": initialized.checkpoint_digest,
                "config_digest": initialized.config_digest,
                "manifest_digest": initialized.manifest_digest,
                "training_started": initialized.training_started,
                "evaluation_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
