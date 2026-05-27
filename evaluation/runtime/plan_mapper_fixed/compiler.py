from __future__ import annotations

from typing import Dict, List, Set

from .models import CompileReport, PlanPayload


def _step_index(plan: PlanPayload) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for subtask in plan.output.subtasks:
        for step in subtask.steps:
            mapping[step.id] = step.agent
    return mapping


def compile_plan(plan: PlanPayload) -> CompileReport:
    step_to_agent = _step_index(plan)
    dependency_errors: List[str] = []
    tool_binding_errors: List[str] = []

    known_steps: Set[str] = set(step_to_agent.keys())
    agent_tools = {a.agent_role: set(a.tools) for a in plan.output.agents}

    for subtask in plan.output.subtasks:
        for step in subtask.steps:
            for dep in step.depends_on:
                if dep not in known_steps:
                    dependency_errors.append(
                        f"Step {step.id} depends on missing step {dep}."
                    )

            if step.tool:
                allowed = agent_tools.get(step.agent, set())
                if step.tool not in allowed:
                    tool_binding_errors.append(
                        f"Step {step.id} requires tool {step.tool} not in agent {step.agent} tools."
                    )

    loop_count = 0
    for item in plan.output.execution_order:
        if isinstance(item, str):
            if item not in known_steps:
                dependency_errors.append(
                    f"Execution order references missing step {item}."
                )
        else:
            loop_count += 1
            for sid in item.loop.steps:
                if sid not in known_steps:
                    dependency_errors.append(
                        f"Loop references missing step {sid}."
                    )

    step_count = len(known_steps)
    agent_count = len(plan.output.agents)

    return CompileReport(
        step_count=step_count,
        agent_count=agent_count,
        loop_count=loop_count,
        dependency_errors=dependency_errors,
        tool_binding_errors=tool_binding_errors,
    )
