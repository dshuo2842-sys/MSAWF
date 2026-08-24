"""Dependency-free configuration schema for approved Stage 1A decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from msawf.constants import (
    CANONICAL_SEED,
    CLOSED_WORLD_CLASSES,
    L_MAX,
    NOISE_EVALUATION_PROBABILITIES,
    OMEGA,
    OPEN_WORLD_CLASSES,
    STAGE_I_INSERTION_PROBABILITY,
    STAGE_III_INSERTION_PROBABILITY,
)

UNRESOLVED = "UNRESOLVED"


class ConfigError(ValueError):
    """Raised when configuration violates the canonical schema."""


class UnresolvedConfigError(ConfigError):
    """Raised when a run requires a field that remains unresolved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"unknown {context} fields: {sorted(unknown)}")


@dataclass(frozen=True)
class DataConfig:
    representation: str = "packet_direction"
    padding_value: int = 0
    max_length: int = L_MAX
    prefix_lengths: tuple[int, ...] = OMEGA

    def validate(self) -> None:
        _require(self.representation == "packet_direction", "invalid representation")
        _require(self.padding_value == 0, "canonical padding_value must be 0")
        _require(self.max_length == L_MAX, f"canonical max_length must be {L_MAX}")
        _require(self.prefix_lengths == OMEGA, f"canonical prefix_lengths must be {OMEGA}")


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "msawf_cnn_v1"
    input_length: int = L_MAX
    feature_tap: str = "flattened_final_encoder_output"
    feature_dim: int = 14_592
    feature_normalization: bool = False
    closed_world_classes: int = CLOSED_WORLD_CLASSES
    open_world_classes: int = OPEN_WORLD_CLASSES

    def validate(self) -> None:
        _require(self.backbone == "msawf_cnn_v1", "invalid canonical backbone")
        _require(self.input_length == L_MAX, f"model input_length must be {L_MAX}")
        _require(
            self.feature_tap == "flattened_final_encoder_output",
            "invalid canonical feature_tap",
        )
        _require(self.feature_dim == 14_592, "canonical feature_dim must be 14592")
        _require(not self.feature_normalization, "canonical z is not normalized")
        _require(self.closed_world_classes == 100, "closed-world classes must be 100")
        _require(self.open_world_classes == 101, "open-world classes must be 101")


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str | None = None

    def validate(self) -> None:
        _require(self.optimizer == "AdamW", "canonical optimizer must be AdamW")
        _require(self.learning_rate == 1e-3, "canonical learning_rate must be 1e-3")
        _require(self.weight_decay == 1e-4, "canonical weight_decay must be 1e-4")
        _require(self.scheduler is None, "canonical configuration uses no scheduler")


@dataclass(frozen=True)
class AugmentationConfig:
    stage1_probability: float = STAGE_I_INSERTION_PROBABILITY
    stage3_probability: float = STAGE_III_INSERTION_PROBABILITY
    probability_policy: str = "fixed"
    noise_evaluation_probabilities: tuple[float, ...] = (
        NOISE_EVALUATION_PROBABILITIES
    )

    def validate(self) -> None:
        _require(self.stage1_probability == 0.2, "Stage I p must be 0.2")
        _require(self.stage3_probability == 0.2, "Stage III p must be 0.2")
        _require(self.probability_policy == "fixed", "training p must be fixed")
        _require(
            self.noise_evaluation_probabilities == (0.0, 0.1, 0.2, 0.3),
            "noise evaluation probabilities must be (0,0.1,0.2,0.3)",
        )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = CANONICAL_SEED
    batch_size: int = 64
    stage1_epochs: int = 50
    stage2_epochs: int = 20
    stage3_epochs: int = 50
    checkpoint_selection: str = "last_epoch"
    early_stopping: bool = False

    def validate(self) -> None:
        _require(self.seed == 18, "canonical seed must be 18")
        _require(self.batch_size == 64, "canonical batch_size must be 64")
        _require(
            (self.stage1_epochs, self.stage2_epochs, self.stage3_epochs)
            == (50, 20, 50),
            "canonical stage epochs must be 50/20/50",
        )
        _require(
            self.checkpoint_selection == "last_epoch",
            "canonical checkpoint selection must be last_epoch",
        )
        _require(not self.early_stopping, "canonical protocol has no early stopping")


@dataclass(frozen=True)
class LossConfig:
    lambda_con: float = 0.1
    lambda_align: float = 0.1
    lambda_rob: float = 0.5

    def validate(self) -> None:
        _require(
            (self.lambda_con, self.lambda_align, self.lambda_rob) == (0.1, 0.1, 0.5),
            "canonical loss weights must be 0.1/0.1/0.5",
        )


@dataclass(frozen=True)
class EvaluationConfig:
    probability_transform: str = "sigmoid"
    label_threshold: float = 0.5
    multilabel_reduction: str = "samples"
    zero_division: int = 0
    top_k: int = 5
    confidence_threshold: float = 0.70
    stability_threshold: float = 0.70
    observation_points: tuple[int, ...] = OMEGA
    first_point_can_decide: bool = False
    final_point_fallback: bool = True

    def validate(self) -> None:
        _require(self.probability_transform == "sigmoid", "probabilities use sigmoid")
        _require(self.label_threshold == 0.5, "label threshold must be 0.5")
        _require(self.multilabel_reduction == "samples", "reduction must be samples")
        _require(self.zero_division == 0, "zero_division must be 0")
        _require(self.top_k == 5, "five-tab canonical top_k must be 5")
        _require(self.confidence_threshold == 0.70, "tau must be 0.70")
        _require(self.stability_threshold == 0.70, "delta must be 0.70")
        _require(self.observation_points == OMEGA, "observation points must equal Omega")
        _require(not self.first_point_can_decide, "first prefix cannot decide early")
        _require(self.final_point_fallback, "final observation must produce output")


@dataclass(frozen=True)
class PathsConfig:
    source_dataset: str = UNRESOLVED
    target_dataset: str = UNRESOLVED
    split_manifest: str = UNRESOLVED
    defense_dataset: str = UNRESOLVED
    checkpoint_input: str = UNRESOLVED
    checkpoint_output_dir: str = UNRESOLVED

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            _require(isinstance(value, str) and bool(value.strip()), f"{field.name} is empty")


@dataclass(frozen=True)
class FewShotConfig:
    way: int | str = UNRESOLVED
    shot: int | str = UNRESOLVED
    support_size: int | str = UNRESOLVED
    query_size: int | str = UNRESOLVED

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            _require(
                value == UNRESOLVED or (isinstance(value, int) and value > 0),
                f"{field.name} must be a positive integer or UNRESOLVED",
            )


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()
    few_shot: FewShotConfig = FewShotConfig()

    def validate(self, *, required_fields: Iterable[str] = ()) -> None:
        for section in (
            self.data,
            self.model,
            self.optimization,
            self.augmentation,
            self.training,
            self.loss,
            self.evaluation,
            self.paths,
            self.few_shot,
        ):
            section.validate()
        for dotted_name in required_fields:
            value = _get_dotted(self, dotted_name)
            if _contains_unresolved(value):
                raise UnresolvedConfigError(
                    f"required configuration field '{dotted_name}' is UNRESOLVED"
                )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExperimentConfig":
        top_level = {
            "data",
            "model",
            "optimization",
            "augmentation",
            "training",
            "loss",
            "evaluation",
            "paths",
            "few_shot",
        }
        _reject_unknown(mapping, top_level, "top-level")

        def section(name: str, config_type: type[Any]) -> Any:
            raw = mapping.get(name, {})
            if not isinstance(raw, Mapping):
                raise ConfigError(f"section '{name}' must be an object")
            allowed = {field.name for field in fields(config_type)}
            _reject_unknown(raw, allowed, name)
            values = dict(raw)
            for tuple_field in (
                "prefix_lengths",
                "noise_evaluation_probabilities",
                "observation_points",
            ):
                if tuple_field in values:
                    values[tuple_field] = tuple(values[tuple_field])
            return config_type(**values)

        config = cls(
            data=section("data", DataConfig),
            model=section("model", ModelConfig),
            optimization=section("optimization", OptimizationConfig),
            augmentation=section("augmentation", AugmentationConfig),
            training=section("training", TrainingConfig),
            loss=section("loss", LossConfig),
            evaluation=section("evaluation", EvaluationConfig),
            paths=section("paths", PathsConfig),
            few_shot=section("few_shot", FewShotConfig),
        )
        config.validate()
        return config


def _get_dotted(config: Any, dotted_name: str) -> Any:
    value = config
    for part in dotted_name.split("."):
        if not is_dataclass(value) or not hasattr(value, part):
            raise ConfigError(f"unknown required configuration field '{dotted_name}'")
        value = getattr(value, part)
    return value


def _contains_unresolved(value: Any) -> bool:
    if value == UNRESOLVED:
        return True
    if is_dataclass(value):
        return any(_contains_unresolved(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, Mapping):
        return any(_contains_unresolved(item) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return any(_contains_unresolved(item) for item in value)
    return False


def load_config(
    path: str | Path, *, required_fields: Iterable[str] = ()
) -> ExperimentConfig:
    """Load a JSON configuration without adding a YAML dependency."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be an object")
    config = ExperimentConfig.from_mapping(raw)
    config.validate(required_fields=required_fields)
    return config
