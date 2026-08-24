"""Stage-plan registry for the public MSAWF implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagePlan:
    name: str
    domain: str
    view: str
    objective: str
    predecessor: str | None


@dataclass(frozen=True)
class MethodPlan:
    method_id: str
    stages: tuple[StagePlan, ...]


_PLANS = {
    "msawf": MethodPlan(
        method_id="msawf",
        stages=(
            StagePlan("stage1", "source", "paired", "L_pre", None),
            StagePlan("stage2", "source+target-support", "clean", "L_bridge", "stage1"),
            StagePlan("stage3", "target-support", "paired", "L_final", "stage2"),
        ),
    ),
}


def get_method_plan(method_id: str) -> MethodPlan:
    try:
        return _PLANS[method_id]
    except KeyError as exc:
        raise ValueError(f"unknown method plan: {method_id}") from exc


def list_method_plans() -> tuple[str, ...]:
    return tuple(sorted(_PLANS))
