"""
L1-L3 (bare LLMs) + F1 (AutoGen) share an identical task description package.

Per user decision B1 (2026-05-16, re-confirmed 2026-05-16 evening):
the system_message is STRICTLY the four prompt sections — PREAMBLE +
§5 (tools) + §9 (schema) + §12 (closing) — and NOTHING else. No
wrapper, no extra task statement, no output-rules block.

The user message ("QUERY: <q>\\nLEARNER: <j>") supplies the per-sample
input. The PREAMBLE itself ("You generate personalized multi-agent
plans...") is the task description.

Use `build_t4_system_message()` for AutoGen (F1) and
`build_t1_system_message` (alias) for L1-L3 bare-LLM baselines. F2 AutoAgents
builds its own `idea` prompt in `baselines/autoagents/upstream_runner.py`
and does NOT use these helpers.
"""
from baselines.common.prompt_sections import compose_t4


def build_t4_system_message() -> str:
    """Return only compose_t4() — i.e. PREAMBLE + §5 + §9 + §12.

    Per user B1 decision (2026-05-16): no additional task wrapper.
    """
    return compose_t4()


# v1 2026-05-16: L1-L3 use the same system message as F1.
build_t1_system_message = build_t4_system_message
