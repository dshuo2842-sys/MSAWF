"""In-memory scalar logging that cannot influence checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class ScalarLogRecord:
    stage: str
    epoch: int
    global_step: int
    values: Mapping[str, float]


@dataclass
class ScalarLogger:
    records: list[ScalarLogRecord] = field(default_factory=list)

    def log(
        self,
        *,
        stage: str,
        epoch: int,
        global_step: int,
        values: Mapping[str, Tensor | float],
    ) -> ScalarLogRecord:
        detached = {
            name: float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
            for name, value in values.items()
        }
        record = ScalarLogRecord(stage, epoch, global_step, detached)
        self.records.append(record)
        return record
