"""
Validate a plan dict against the inference prompt §9 schema.
Raises PlanSchemaError on any violation.
"""
VALID_TOOLS = {
    "FirecrawlSearchTool", "RagTool", "CodeInterpreterTool",
    "DirectoryReadTool", "FileReadTool", "FileWriterTool",
    "CodeDocsSearchTool", "ArxivPaperTool",
}


class PlanSchemaError(ValueError):
    pass


def _flatten(execution_order):
    """Collect all step ids appearing in execution_order, including loop bodies."""
    out = set()
    for item in execution_order:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and "loop" in item:
            loop = item["loop"]
            if "step" in loop:
                out.add(loop["step"])
            elif "steps" in loop:
                out.update(loop["steps"])
        else:
            raise PlanSchemaError(f"bad execution_order entry: {item}")
    return out


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise PlanSchemaError("plan must be dict")
    if "input" not in plan or "output" not in plan:
        raise PlanSchemaError("missing input/output top-level")

    inp = plan["input"]
    if "query" not in inp or "learner" not in inp:
        raise PlanSchemaError("missing input.query or input.learner")
    lr = inp["learner"]
    if "about_me" not in lr or "top_tags" not in lr:
        raise PlanSchemaError("learner must have about_me + top_tags")

    out = plan["output"]
    for k in ("agents", "subtasks", "execution_order"):
        if k not in out:
            raise PlanSchemaError(f"missing output.{k}")

    agents = out["agents"]
    if not isinstance(agents, list) or not agents:
        raise PlanSchemaError("agents must be non-empty list")
    agent_names = set()
    agent_tools = {}
    for ag in agents:
        for k in ("agent_role", "goal", "backstory", "tools"):
            if k not in ag:
                raise PlanSchemaError(f"agent missing field {k}")
        for t in ag["tools"]:
            if t not in VALID_TOOLS:
                raise PlanSchemaError(f"unknown tool {t}")
        agent_names.add(ag["agent_role"])
        agent_tools[ag["agent_role"]] = set(ag["tools"])

    all_step_ids = set()
    for st in out["subtasks"]:
        for k in ("id", "name", "subtask_objective", "steps"):
            if k not in st:
                raise PlanSchemaError(f"subtask missing field {k}")
        for step in st["steps"]:
            for k in ("id", "agent", "objective", "instruction", "tool",
                      "requires_human_input", "expected_output", "depends_on"):
                if k not in step:
                    raise PlanSchemaError(f"step {step.get('id')} missing {k}")
            if step["agent"] not in agent_names:
                raise PlanSchemaError(
                    f"step uses undeclared agent {step['agent']}")
            if step["tool"] is not None:
                if step["tool"] not in VALID_TOOLS:
                    raise PlanSchemaError(f"bad tool {step['tool']}")
                if step["tool"] not in agent_tools[step["agent"]]:
                    raise PlanSchemaError(
                        f"step's tool not declared on agent")
            all_step_ids.add(step["id"])

    flat = _flatten(out["execution_order"])
    if flat != all_step_ids:
        missing = all_step_ids - flat
        extra = flat - all_step_ids
        raise PlanSchemaError(
            f"execution_order mismatch. missing={missing} extra={extra}")
