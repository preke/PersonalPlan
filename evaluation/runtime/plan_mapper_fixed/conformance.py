from __future__ import annotations

from typing import List

from .models import ConformanceReport, PlanPayload


def _immutable_snapshot(plan: PlanPayload) -> dict:
    return {
        "agents": [a.model_dump() for a in plan.output.agents],
        "subtasks": [s.model_dump() for s in plan.output.subtasks],
        "execution_order": [
            item if isinstance(item, str) else item.model_dump()
            for item in plan.output.execution_order
        ],
    }


def check_immutability(before: PlanPayload, after: PlanPayload) -> ConformanceReport:
    left = _immutable_snapshot(before)
    right = _immutable_snapshot(after)

    diffs: List[str] = []
    if left["agents"] != right["agents"]:
        diffs.append("Immutable mismatch in output.agents")
    if left["subtasks"] != right["subtasks"]:
        diffs.append("Immutable mismatch in output.subtasks")
    if left["execution_order"] != right["execution_order"]:
        diffs.append("Immutable mismatch in output.execution_order")

    return ConformanceReport(passed=(len(diffs) == 0), immutable_diffs=diffs)
