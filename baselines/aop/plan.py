"""
AOP (Li et al., ICLR 2025) baseline — paper method, §9-native output.

Method preserved:
  - Meta-agent with three principles (solvability / completeness /
    non-redundancy).
  - 8 teaching workers, one per §5 tool (Q2 design A): ConceptTutor /
    CodeValidator / DocsRetriever / WebResearcher / FileWriter /
    PaperSearcher / RagRetriever / DirectoryReader. Replaces upstream
    code/math/search/commonsense pool.
  - Reward signal: LLM-as-judge in [0, 1]. Substitutes the upstream
    pretrained SimilarityMLP because the public AOP repo never shipped
    MLP_high.pt (option C decision). The (sub-task, agent) match-score
    role and the replanning loop semantics are preserved.
  - Replanning loop: identify steps with reward < REWARD_THRESHOLD,
    ask the meta-agent to revise ONLY those steps, up to
    MAX_REPLAN_ROUNDS rounds. Each round re-emits the FULL §9 plan.

方案 A (2026-05-17): the meta-agent emits a §9 plan directly — no
post-hoc translator. The native logger records the §9 plan plus replan
trace for the 方案 B appendix sidecar.

Per v1 baseline design (P5/P6):
  - Input package = PREAMBLE + §5 + §9 + §12 (via compose_t4()), same
    as L1-L3 / F1-F2 / M1-M2. No §1-§4 / §6-§8 / §11 pedagogy is prepended.
  - Backbone for both meta-agent and judge is qwen3-32b (T5 default).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from baselines.aop.meta_prompt import NEW_META_PROMPT, build_replan_prompt
from baselines.aop.teaching_agents_descs import (
    code_validator_descriptions,
    concept_tutor_descriptions,
    directory_reader_descriptions,
    docs_retriever_descriptions,
    file_writer_descriptions,
    paper_searcher_descriptions,
    rag_retriever_descriptions,
    web_researcher_descriptions,
)
from baselines.common.json_repair import fix_json_format
from baselines.common.llm_client import LLMClient
from baselines.common.native_logger import log_native


BACKBONE = "qwen3-32b"
_llm = LLMClient(backend=BACKBONE)

REWARD_THRESHOLD = 0.5
MAX_REPLAN_ROUNDS = 2

# 8-worker pool keyed by the names the meta-agent must use for step.agent.
DESCRIPTIONS = {
    "ConceptTutor": concept_tutor_descriptions[0],
    "CodeValidator": code_validator_descriptions[0],
    "DocsRetriever": docs_retriever_descriptions[0],
    "WebResearcher": web_researcher_descriptions[0],
    "FileWriter": file_writer_descriptions[0],
    "PaperSearcher": paper_searcher_descriptions[0],
    "RagRetriever": rag_retriever_descriptions[0],
    "DirectoryReader": directory_reader_descriptions[0],
}


_JUDGE_PROMPT = """You score how well an AOP teaching-agent can solve a sub-task step.

AGENT NAME: {agent_name}
AGENT CAPABILITY: {agent_desc}

STEP INSTRUCTION:
{step_text}

Score in [0.0, 1.0]:
- 1.0 = the step fits this agent's capability cleanly; one or zero
       cross-agent dependencies; success is verifiable by this agent alone.
- 0.7 = good fit but with small mismatches (e.g. requires output the agent
       cannot directly produce but a downstream agent can).
- 0.4 = partial fit; the agent can do part of the step but at least
       one core operation is outside its capability.
- 0.0 = wrong agent for this step.

Output STRICT JSON only, no fences, no commentary:
{{"score": <float in [0,1]>}}
"""


# --------------------------------------------------------------------
# JSON extraction / scoring
# --------------------------------------------------------------------

def _extract_json_any(text: str):
    """Best-effort JSON extraction. Handles fenced blocks and prefers a
    top-level object (§9 plan); falls back to a top-level list."""
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


def _score_step(step_text: str, agent_name: str) -> Optional[float]:
    """LLM-as-judge score in [0,1]. Returns None for unknown agent (out of
    the 8-worker pool), missing instruction, or parse failure. Unscoreable
    steps are treated as not-weak — we don't replan steps we can't score.
    """
    desc = DESCRIPTIONS.get(agent_name)
    if desc is None or not step_text:
        return None
    prompt = _JUDGE_PROMPT.format(
        agent_name=agent_name, agent_desc=desc, step_text=step_text,
    )
    try:
        raw = _llm.chat([{"role": "user", "content": prompt}])
    except Exception as err:
        print(f"[aop] judge call failed: {type(err).__name__}: {err}")
        return None
    parsed = _extract_json_any(raw)
    if not isinstance(parsed, dict):
        return None
    score = parsed.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _identify_weak_subtasks(plan_v9) -> list:
    """Walk a §9 plan's subtasks/steps and return ids of steps that scored
    below REWARD_THRESHOLD.

    Step id format: "<subtask_index>.<step_index>" (0-indexed), so the
    meta-agent can locate each weak step in its prior plan during replan.
    Steps whose agent is not in DESCRIPTIONS get score=None and are
    treated as not-weak (we do not replan unscoreable steps).
    """
    weak = []
    if not isinstance(plan_v9, dict):
        return weak
    output = plan_v9.get("output")
    if not isinstance(output, dict):
        return weak
    subtasks = output.get("subtasks")
    if not isinstance(subtasks, list):
        return weak
    for i, st in enumerate(subtasks):
        if not isinstance(st, dict):
            continue
        steps = st.get("steps")
        if not isinstance(steps, list):
            continue
        for j, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            agent = step.get("agent")
            instr = step.get("instruction") or ""
            score = _score_step(instr, agent)
            if score is not None and score < REWARD_THRESHOLD:
                weak.append(f"{i}.{j}")
    return weak


def _ensure_input_block(plan: dict, query: str, learner: dict) -> dict:
    if not isinstance(plan, dict):
        return plan
    plan.setdefault("input", {})
    plan["input"]["query"] = query
    plan["input"]["learner"] = learner
    return plan


# --------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------

def plan_fn(query: str, learner: dict) -> dict:
    """Run AOP meta-agent + LLM-judge + replan loop, return a §9 plan.

    Per v1 P5: PREAMBLE+§5+§9+§12 (compose_t4()) is embedded in
    NEW_META_PROMPT; the user message carries (query, learner).
    """
    user_msg = (
        f"User query: {query}\n"
        f"Learner profile: {json.dumps(learner, ensure_ascii=False)}\n\n"
        f"Output the §2 JSON plan:"
    )

    # ---- Meta-agent: initial §9 plan
    try:
        raw = _llm.chat([
            {"role": "system", "content": NEW_META_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except Exception as err:
        print(f"[aop] meta-agent call failed: "
              f"{type(err).__name__}: {err}")
        return {}

    plan_v9 = _extract_json_any(raw)
    if not isinstance(plan_v9, dict):
        log_native(
            {"initial_plan_v9": None, "replan_rounds": [], "final_plan_v9": None},
            extra={"fired_replan": False, "max_replan_rounds": MAX_REPLAN_ROUNDS,
                   "failure": "meta-agent output was not a JSON object"},
        )
        return {}

    initial_plan = plan_v9
    rounds_data: list = []
    fired_replan = False

    # ---- Replanning loop on §9 plan
    for round_idx in range(MAX_REPLAN_ROUNDS):
        try:
            weak_ids = _identify_weak_subtasks(plan_v9)
        except Exception as err:
            print(f"[aop] scoring failed: {type(err).__name__}: {err}")
            break
        weak_ids = sorted(set(weak_ids))
        if not weak_ids:
            break

        fired_replan = True
        prev_plan_json = json.dumps(plan_v9, ensure_ascii=False)
        replan_user = build_replan_prompt(query, prev_plan_json, weak_ids)
        try:
            raw = _llm.chat([
                {"role": "system", "content": NEW_META_PROMPT},
                {"role": "user", "content": replan_user},
            ])
        except Exception as err:
            print(f"[aop] replanning call failed: "
                  f"{type(err).__name__}: {err}")
            rounds_data.append({
                "round": round_idx, "weak_ids": weak_ids,
                "revised_plan_v9": None,
                "error": f"{type(err).__name__}: {err}",
            })
            break
        new_plan_v9 = _extract_json_any(raw)
        rounds_data.append({
            "round": round_idx,
            "weak_ids": weak_ids,
            "revised_plan_v9": new_plan_v9 if isinstance(new_plan_v9, dict) else None,
        })
        if not isinstance(new_plan_v9, dict):
            break
        plan_v9 = new_plan_v9

    if fired_replan:
        print("[aop] LLM-as-judge replanning loop fired for this plan.")

    plan_v9 = _ensure_input_block(plan_v9, query, learner)

    log_native(
        {
            "initial_plan_v9": initial_plan,
            "replan_rounds": rounds_data,
            "final_plan_v9": plan_v9,
        },
        extra={"fired_replan": fired_replan, "max_replan_rounds": MAX_REPLAN_ROUNDS},
    )

    return plan_v9
