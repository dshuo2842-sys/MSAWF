"""Named loss primitives for the three MSAWF stages."""

from .adaptation import (
    StageThreeLosses,
    feature_alignment_loss,
    stage_three_objective,
)
from .bridge import bridge_alpha, bridge_loss
from .classification import classification_loss
from .consistency import consistency_loss

__all__ = [
    "StageThreeLosses",
    "bridge_alpha",
    "bridge_loss",
    "classification_loss",
    "consistency_loss",
    "feature_alignment_loss",
    "stage_three_objective",
]
