"""
EduPlanner (ported) baseline — v1 design + 方案 A+B refactor (2026-05-17).

T5 (educational MAS). The adversarial loop structure is unchanged:
    Question Analyst  (once)
      -> for N rounds:
           Optimizer     emits §2 plan directly (now)
           CIDPP Evaluator scores §2 plan
           early-exit on final_score >= EVAL_PASS_THRESHOLD

方案 A: Optimizer emits §9 directly (compose_t4() prefix); Evaluator
scores §2 plans. No translator / domain remap.

方案 B: Each adversarial round is recorded; the full trace (mistakes,
rounds[], final_score) is saved via native_logger.

Backbone: qwen3-32b.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from baselines.common.json_repair import fix_json_format
from baselines.common.llm_client import LLMClient
from baselines.common.native_logger import log_native
from baselines.common.prompt_sections import compose_t4
from baselines.eduplanner.util import get_students_ability, map_top_tags_to_levels

ABILITY_TREE = Path(__file__).parent / "ability_tree.json"
N_OPTIMIZATION_ROUNDS = 3      # paper uses 20; we use 3 for cost
EVAL_PASS_THRESHOLD = 80       # paper CIDPP pass threshold

_BACKBONE = "qwen3-32b"
_llm = LLMClient(backend=_BACKBONE)
_T4_HEADER = compose_t4()


# --------------------------------------------------------------------
# Paper prompts (reconstructed from external/Edu_Planner/pre_prompt.py
# and external/Edu_Planner/CIDPP.py). Math context -> programming.
# --------------------------------------------------------------------

# Question Analyst (paper: pre_prompt.ana_task + common_mistakes_db).
COMMON_MISTAKES_DB = """1. Off-by-one / boundary errors
2. Mishandling of None / null / empty collections
3. Mutable default argument or shared-state aliasing
4. Misuse of language idioms (list comprehension vs generator, async/await)
5. Misunderstanding of scope / closures / shadowing
6. Misuse of library APIs (calling with wrong signature or deprecated form)
7. Ignoring exceptions or catching too broadly
8. Confusing pass-by-reference vs pass-by-value semantics
9. Incorrect type assumptions / implicit type coercion
10. Forgetting to handle the empty / single-element edge case
"""

ANA_TASK = """You need to calculate the three mistakes that students will make in the above example based on their knowledge background and learning ability, and insert them at the end of the example in order of probability from largest to smallest.
- Combined with the question given above
- Incorporate students' background knowledge but don't reveal it in your response
- Do not output irrelevant content, such as: note and Here are...
- Responses include only Common Mistakes

Reference common-mistakes catalogue (use as inspiration; do NOT restrict
yourself to only these):
{mistakes_db}

Output STRICT JSON ONLY (no markdown fences, no prose):
{{
  "mistakes": [
    {{"mistake": "<short label>", "probability_pct": <0-100>, "remediation_hint": "<one sentence>"}},
    {{"mistake": "...", "probability_pct": ..., "remediation_hint": "..."}},
    {{"mistake": "...", "probability_pct": ..., "remediation_hint": "..."}}
  ]
}}
The three mistakes must be ordered by probability_pct DESCENDING.
"""


# Optimizer — 方案 A: emits §2 plan directly. Prompt is prefixed with
# compose_t4() (PREAMBLE + §5 tool pool + §9 schema + §12 closing) so the
# model sees the schema + tool pool. EduPlanner's Optimizer role + the
# Question Analyst mistakes / persona / prev_plan / suggestions are kept
# as contextual inputs.
OPTI_TASK = """# Role
You are the Plan Optimizer in an adversarial teaching-plan optimization loop
(EduPlanner: Question Analyst -> Optimizer -> CIDPP Evaluator).

In this loop, the artifact you optimize is a §2 plan (see schema above).
You revise it across rounds to maximize the CIDPP Evaluator's final_score
(Clarity / Integrity / Depth / Practicality / Pertinence).

# Inputs
Student ability profile (Skill-Tree derived):
{persona}

Programming question to teach:
{query}

Learner profile (raw):
{learner_json}

Common mistakes this learner is likely to make (from Question Analyst):
{mistakes_block}

Previous §2 plan (if any):
{prev_plan}

Evaluator suggestions to incorporate (if any):
{suggestions}

# Task
Produce a §2 plan that teaches the programming question to this learner.
Follow these EduPlanner Optimizer principles while doing so:
- The teaching topic cannot be changed.
- Cover BOTH knowledge explanation AND exercise walk-through across the
  plan's subtasks/steps (Integrity).
- Insert exercises with new difficulty gradients across the steps and
  explain them in step.expected_output (Depth + Practicality).
- For at least one exercise step, surface the relevant common mistakes
  from the Question Analyst list above inside the step's expected_output
  (with probability and a short remediation hint, descending order).
- The plan must concretely cite the learner's profile (e.g. "since the
  learner is comfortable with X, we skip basics and focus on Y") at
  least twice across agent.backstory / step.instruction / step.expected_output.

# Output
Your output MUST strictly conform to the §2 schema shown above:
{{
  "input":  {{"query": "...", "learner": {{...}}}},
  "output": {{"agents": [...], "subtasks": [...], "execution_order": [...]}}
}}

Use only the tools listed in §1 (AVAILABLE TOOLS). Output STRICT JSON
ONLY — no markdown fences, no prose, start with {{ and end with }}.
"""


# Evaluator — 方案 A: scores §2 plans instead of NATIVE lesson plans.
# CIDPP rubric meaning is reused; the wording is adapted to "§2 plan".
EVAL_TASK = """# Role
You are an impartial CIDPP evaluator, experienced in educational content
analysis and instructional design evaluation.

## Attention
You are responsible for assessing the quality of a given §2 multi-agent
teaching plan based on five specific evaluation criteria. Your evaluation
should be objective and based solely on the Evaluation Standard provided
below.

## Student profile (for Pertinence and Learner-Simulation):
{persona}

## Programming question:
{query}

## §2 Plan (JSON):
{plan_json}

## Evaluation Standard (applied to the §2 plan above):
- [C]  Clarity:     The §2 plan's directness and simplicity — agent roles,
                    subtask objectives, and step instructions avoid
                    unnecessary complexity and redundancy.
- [I]  Integrity:   Whether the §2 plan is complete and systematic across
                    its agents/subtasks/steps, covering both knowledge
                    explanation and exercise walk-through.
- [DD] Depth:       The §2 plan's ability to inspire deep thinking and to
                    surface the underlying connections between knowledge
                    points across its steps.
- [P]  Practicality:The practical application value of the exercise steps,
                    ensuring the learner can use the knowledge to solve
                    real-life problems.
- [PT] Pertinence:  The §2 plan's adaptability to the student's knowledge
                    level and learning needs (judged from agent.backstory,
                    step.instruction, step.expected_output references to
                    the learner profile).

## Constraints
- Avoid any bias in evaluation based on the §2 plan's length or appearance.
- Be as objective as possible in assessing each aspect individually.

## Work flow
1. Score each criterion 0-100.
2. Estimate "learner_simulation_pct": probability (0-100) the student
   described above would solve the exercise(s) correctly after executing
   this §2 plan.
3. final_score = round(mean([C, I, DD, P, PT]))
4. Provide concrete suggestions for the next Optimizer round (what to
   change in the §2 plan: agents / subtasks / steps / tools / etc.).

# Output STRICT JSON ONLY (no markdown, no prose):
{{
  "ciddp": {{"C": 0, "I": 0, "DD": 0, "P": 0, "PT": 0}},
  "learner_simulation_pct": 0,
  "final_score": 0,
  "suggestions": "<one paragraph of concrete actionable improvements>"
}}
"""


# --------------------------------------------------------------------
# JSON parsing helper
# --------------------------------------------------------------------

def _extract_json(text: str):
    """Best-effort: strip code fences, take largest {...}, repair if needed."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
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


# --------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------

def plan_fn(query: str, learner: dict) -> dict:
    levels = map_top_tags_to_levels(learner.get("top_tags", []),
                                    learner.get("about_me", ""))
    persona = get_students_ability(str(ABILITY_TREE), levels)
    learner_json = json.dumps(learner, ensure_ascii=False)

    # ----------------------------------------------------------------
    # Round 0: Question Analyst (paper-faithful, no §1-§12 prefix).
    # ----------------------------------------------------------------
    qa_prompt = (
        f"Programming question:\n{query}\n\n"
        f"Learner profile:\n{learner_json}\n\n"
        f"{persona}\n\n"
        + ANA_TASK.format(mistakes_db=COMMON_MISTAKES_DB)
    )
    try:
        qa_resp = _llm.chat([{"role": "user", "content": qa_prompt}])
        qa_obj = _extract_json(qa_resp) or {}
        mistakes = qa_obj.get("mistakes", []) if isinstance(qa_obj, dict) else []
    except Exception as err:
        print(f"[eduplanner] question-analyst failed: {type(err).__name__}: {err}")
        mistakes = []

    # ----------------------------------------------------------------
    # N rounds of Optimizer <-> Evaluator adversarial loop on §2 plans.
    # ----------------------------------------------------------------
    plan_v9: dict | None = None
    suggestions = ""
    last_score = 0.0
    rounds_trace: list[dict] = []
    round_i = -1

    for round_i in range(N_OPTIMIZATION_ROUNDS):
        opt_body = OPTI_TASK.format(
            persona=persona,
            query=query,
            learner_json=learner_json,
            mistakes_block=json.dumps(mistakes, ensure_ascii=False, indent=2),
            prev_plan=(json.dumps(plan_v9, ensure_ascii=False)
                       if plan_v9 else "(none — this is round 1)"),
            suggestions=suggestions or "(none — this is round 1)",
        )
        opt_prompt = _T4_HEADER + "\n\n" + opt_body
        try:
            opt_resp = _llm.chat([{"role": "user", "content": opt_prompt}])
        except Exception as err:
            print(f"[eduplanner] optimizer round {round_i} failed: "
                  f"{type(err).__name__}: {err}")
            break
        new_plan = _extract_json(opt_resp)
        if new_plan is None:
            print(f"[eduplanner] optimizer round {round_i}: could not parse JSON")
            rounds_trace.append({
                "round": round_i,
                "plan_v9": None,
                "score": None,
                "suggestions": None,
            })
            continue
        new_plan = _ensure_input_block(new_plan, query, learner)
        plan_v9 = new_plan

        try:
            eval_prompt = EVAL_TASK.format(
                persona=persona,
                query=query,
                plan_json=json.dumps(plan_v9, ensure_ascii=False),
            )
            eval_resp = _llm.chat([{"role": "user", "content": eval_prompt}])
        except Exception as err:
            print(f"[eduplanner] evaluator round {round_i} failed: "
                  f"{type(err).__name__}: {err}")
            rounds_trace.append({
                "round": round_i,
                "plan_v9": plan_v9,
                "score": None,
                "suggestions": None,
            })
            break
        eval_result = _extract_json(eval_resp) or {}
        score = eval_result.get("final_score", 0) or 0
        suggestions = eval_result.get("suggestions", "") or ""

        try:
            last_score = float(score)
        except (TypeError, ValueError):
            last_score = 0.0

        rounds_trace.append({
            "round": round_i,
            "plan_v9": plan_v9,
            "score": last_score,
            "suggestions": suggestions,
        })

        print(f"[eduplanner] round {round_i}: final_score={last_score}")
        if last_score >= EVAL_PASS_THRESHOLD:
            break

    # ----------------------------------------------------------------
    # 方案 B: save adversarial loop trace via native_logger.
    # ----------------------------------------------------------------
    log_native(
        {
            "mistakes": mistakes,
            "rounds": rounds_trace,
            "final_score": last_score,
        },
        extra={"rounds_run": round_i + 1},
    )

    if plan_v9 is None:
        return {}
    return _ensure_input_block(plan_v9, query, learner)
