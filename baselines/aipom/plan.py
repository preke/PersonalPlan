"""AIPOM baseline — single-shot agent-aware DAG planner, §9-native output.

Paper: Kim et al., AIPOM: Agent-aware Interactive Planning for
Multi-Agent Systems, EMNLP 2025 System Demonstrations.
Upstream repo: https://github.com/megagonlabs/aipom

We use AIPOM's single-shot planner surface (the conversational +
graph-editing refinement loop requires a human and is out of scope
for batch evaluation).  The four planning principles (explicit agent
assignment, typed I/O dependencies, inspectable DAG structure, no
hidden coordination) are encoded in the meta-prompt; the model emits
a §9 plan directly so there is no post-hoc translator.

Per v1 baseline design (P5/P6):
  - Input package = PREAMBLE + §5 + §9 + §12 (via compose_t4()), same
    as L1-L3 / F1-F2 / M3 AOP. No §1-§4 / §6-§8 / §11 pedagogy prepended.
  - Backbone is qwen3-32b (T5 default).

Native logging (方案 B sidecar): we record the raw assistant message
and the parsed §9 plan, mirroring the AOP / GenMentor pattern.
"""
from __future__ import annotations

import json
import re

from baselines.aipom.meta_prompt import NEW_META_PROMPT
from baselines.common.json_repair import fix_json_format
from baselines.common.llm_client import LLMClient
from baselines.common.native_logger import log_native


BACKBONE = "qwen3-32b"
_llm = LLMClient(backend=BACKBONE)


def _extract_json_any(text: str):
    """Best-effort JSON extraction. Handles ```json fences and prefers a
    top-level object (§9 plan); falls back to a top-level list.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", text)
    if fenced:
        text = fenced.group(1)
    obj_match = re.search(r"\{[\s\S]*\}", text)
    list_match = re.search(r"\[[\s\S]*\]", text)
    candidates = []
    if obj_match:
        candidates.append(obj_match.group(0))
    if list_match:
        candidates.append(list_match.group(0))
    for blob in candidates:
        try:
            return json.loads(blob)
        except Exception:
            pass
        for level in (1, 2):
            try:
                return json.loads(fix_json_format(blob, repair_attempt=level))
            except Exception:
                continue
    return None


def _ensure_input_block(plan: dict, query: str, learner: dict) -> dict:
    if not isinstance(plan, dict):
        return plan
    plan.setdefault("input", {})
    plan["input"]["query"] = query
    plan["input"]["learner"] = learner
    return plan


def plan_fn(query: str, learner: dict) -> dict:
    """Run the AIPOM single-shot planner, return a §9 plan."""
    user_msg = (
        f"User query: {query}\n"
        f"Learner profile: {json.dumps(learner, ensure_ascii=False)}\n\n"
        "Output the JSON plan now:"
    )

    try:
        raw = _llm.chat([
            {"role": "system", "content": NEW_META_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except Exception as err:
        print(f"[aipom] planner call failed: {type(err).__name__}: {err}")
        log_native(
            {"raw_assistant": None, "plan_v9": None},
            extra={"failure": f"{type(err).__name__}: {err}"},
        )
        return {}

    plan_v9 = _extract_json_any(raw)

    log_native(
        {"raw_assistant": raw, "plan_v9": plan_v9 if isinstance(plan_v9, dict) else None},
        extra={"backbone": BACKBONE},
    )

    if not isinstance(plan_v9, dict):
        return {}

    plan_v9 = _ensure_input_block(plan_v9, query, learner)
    return plan_v9
