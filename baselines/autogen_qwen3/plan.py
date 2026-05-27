"""
AutoGen baseline (F1 in v2 lineup, 2026-05-19).

F1 single-AssistantAgent wrapper around DashScope qwen3-32b.

Why one agent? AutoGen does not auto-decompose; for a single-deliverable task
(a plan JSON), the framework's minimal-yet-faithful usage is one
`AssistantAgent` with the compose_t4() system message. Multi-agent extensions
would introduce structural choices that go beyond AutoGen's "framework as
coordination wrapper" baseline contract.

Prompt policy: system_message = compose_t4() + T4_TASK_DESCRIPTION (via
`build_t4_system_message()`). §0/§5/§9/§12 only — pedagogy and example
sections excluded (i.e. §1-§4, §6-§8, §10, §11 are intentionally dropped
for F1/F2 framework baselines).

Folder is named `autogen_qwen3/` for historical reasons; the actual backbone
is now `qwen3-32b` (DashScope, OpenAI-compatible).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from pathlib import Path

from dotenv import load_dotenv

from baselines.common.json_repair import fix_json_format
from baselines.common.task_description import build_t4_system_message

# Project root = baselines/autogen_qwen3/plan.py → autogen_qwen3 → baselines → ROOT
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Detect whether AutoGen's AssistantAgent is importable in this venv.
try:  # pragma: no cover - import side effect
    from autogen_agentchat.agents import AssistantAgent  # type: ignore
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore

    _AUTOGEN_AVAILABLE = True
except Exception as _autogen_import_err:  # pragma: no cover
    _AUTOGEN_AVAILABLE = False
    print(
        "[autogen_qwen3] WARNING: autogen-agentchat import failed "
        f"({type(_autogen_import_err).__name__}: {_autogen_import_err}); "
        "falling back to direct LLMClient (1-agent equivalent). "
        "This loses the 'AutoGen baseline' semantics."
    )

# Set DASHSCOPE_API_KEY via environment variable or a project-root .env file.
# No fallback is bundled in the public release.
_FALLBACK_DASHSCOPE_KEY = ""

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_MODEL_NAME = "qwen3-32b"


# ---------------------------------------------------------------------------
# JSON extraction (tolerant)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    """Extract the first top-level JSON object from raw text.

    Strategy:
      1. Strip ```json ... ``` fences if present.
      2. Try direct ``json.loads`` on the largest braces block.
      3. Fall through to ``fix_json_format`` with up to 2 repair levels.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if fenced:
        candidate = fenced.group(1)
    else:
        m = re.search(r"\{[\s\S]*\}", raw)
        candidate = m.group(0) if m else raw

    try:
        return json.loads(candidate)
    except Exception:
        pass
    for attempt in (1, 2):
        try:
            return json.loads(fix_json_format(candidate, repair_attempt=attempt))
        except Exception:
            continue
    return None


def _ensure_input_block(plan: dict, query: str, learner: dict) -> dict:
    """Auto-inject input.{query,learner} if the model dropped them."""
    if not isinstance(plan, dict):
        return plan
    plan.setdefault("input", {})
    if not isinstance(plan["input"], dict):
        plan["input"] = {}
    plan["input"].setdefault("query", query)
    plan["input"].setdefault("learner", learner)
    return plan


# ---------------------------------------------------------------------------
# AutoGen path: 1 AssistantAgent + qwen3-32b
# ---------------------------------------------------------------------------

def _make_autogen_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", _FALLBACK_DASHSCOPE_KEY)
    return OpenAIChatCompletionClient(
        model=_MODEL_NAME,
        api_key=api_key,
        base_url=_DASHSCOPE_BASE_URL,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "qwen",
            "structured_output": True,
            "enable_thinking": False,
        },
        extra_body={"enable_thinking": False},
    )


def _make_autogen_plan_fn():
    """Return a closure that builds a fresh AssistantAgent per call.

    Memory-isolation rationale: `AssistantAgent` is stateful — its
    `model_context` accumulates every (user, assistant) turn that flows
    through `agent.run(...)`. Keeping one agent across 3043 plans would
    bloat the prompt unboundedly and cause behavior drift on later items.
    We therefore rebuild the agent inside the closure (Option A). The
    `OpenAIChatCompletionClient` itself is stateless, so we cache it at
    closure scope; building it once also avoids redundant config setup.

    Construction overhead is ~0.02 ms per call, negligible vs ~48 s
    per-plan DashScope latency.
    """
    system_message = build_t4_system_message()
    client = _make_autogen_client()  # stateless; safe to share

    def _user_message(query: str, learner: dict) -> str:
        return (
            "QUERY: " + query + "\n"
            "LEARNER: " + json.dumps(learner, ensure_ascii=False)
        )

    async def _run_async(agent: "AssistantAgent", task: str) -> str:
        result = await agent.run(task=task)
        # Take the last assistant message.
        for msg in reversed(result.messages):
            if getattr(msg, "source", None) == "planner":
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    return content
        # Fallback: last message of any kind.
        if result.messages:
            return getattr(result.messages[-1], "content", "") or ""
        return ""

    def plan_fn(query: str, learner: dict) -> dict:
        # Fresh AssistantAgent per call -> empty model_context -> no
        # cross-sample memory contamination.
        agent = AssistantAgent(
            name="planner",
            system_message=system_message,
            model_client=client,
        )
        task = _user_message(query, learner)
        try:
            raw = asyncio.run(_run_async(agent, task))
        except Exception as e:
            print(f"[autogen_qwen3] AutoGen run failed: {type(e).__name__}: {e}")
            return {}
        parsed = _extract_json(raw)
        if parsed is None:
            return {}
        return _ensure_input_block(parsed, query, learner)

    return plan_fn


# ---------------------------------------------------------------------------
# Direct LLMClient fallback (architecturally identical: 1 agent = 1 chat call)
# ---------------------------------------------------------------------------

def _make_direct_plan_fn():
    from baselines.common.llm_client import LLMClient

    client = LLMClient(backend="qwen3-32b")
    system_message = build_t4_system_message()

    def plan_fn(query: str, learner: dict) -> dict:
        user_message = (
            "QUERY: " + query + "\n"
            "LEARNER: " + json.dumps(learner, ensure_ascii=False)
        )
        try:
            raw = client.chat(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=16000,
            )
        except Exception as e:
            print(
                f"[autogen_qwen3] LLM call failed: {type(e).__name__}: {e}"
            )
            return {}
        parsed = _extract_json(raw or "")
        if parsed is None:
            return {}
        return _ensure_input_block(parsed, query, learner)

    return plan_fn


# ---------------------------------------------------------------------------
# Public factory + module-level plan_fn
# ---------------------------------------------------------------------------

def make_plan_fn(backend: str = "qwen3-32b"):
    """Build a ``plan_fn(query, learner) -> dict`` callable.

    The optional ``backend`` argument is accepted for backwards-compat with
    ``run.py``; v1 design fixes the AutoGen backbone to ``qwen3-32b``. Anything
    else raises so we surface mistakes early.
    """
    if backend not in ("qwen3-32b",):
        raise ValueError(
            f"autogen_qwen3 v1 baseline is locked to qwen3-32b; got {backend!r}"
        )
    if _AUTOGEN_AVAILABLE:
        return _make_autogen_plan_fn()
    return _make_direct_plan_fn()


# Default module-level plan_fn (lazy-built on first call to keep import cheap).
_plan_fn: Any = None


def plan_fn(query: str, learner: dict) -> dict:
    global _plan_fn
    if _plan_fn is None:
        _plan_fn = make_plan_fn()
    return _plan_fn(query, learner)
