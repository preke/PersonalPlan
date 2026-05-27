"""GRPO prompt-row builder helper.

THIS FILE NO LONGER HAS A CLI. Its old `main()` did row-level random
shuffle to materialize train/valid jsonl — that leaked question_ids
across folds. It has been superseded by the repo-root
``build_splits.py``, which does question_id-stratified train/dev/test
splitting and materializes the GRPO rows via ``build_row`` below.

The verbatim old version is preserved at
  history/build_grpo_prompts_v1_row_shuffle.py
for reference. Do not resurrect that logic here.

THIS FILE STILL EXPORTS the row-builder helpers
(``SYSTEM_PROMPT``, ``USER_TEMPLATE``, ``extract_learner``, ``build_row``)
which are imported by:
  - build_splits.py                  (current split-pipeline entry point)
  - GRPO/build_counterfactual_cache.py

To (re)generate the train/dev/test splits, run from the repo root:
  python build_splits.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


SYSTEM_PROMPT = """You are a multi-agent planner that produces personalized teaching plans for Stack Overflow programming questions.

Given a query and a learner profile, produce ONE complete teaching plan as a single JSON object with three top-level keys:

  agents       : the specialized agents needed for this learner. Each agent has agent_role, goal, backstory, tools.
  subtasks     : the ordered pedagogical milestones. Each subtask has id (S1, S2, ...), name, subtask_objective, steps[].
                  Each step has id (S1-1, ...), agent (must match an agent_role above), objective, instruction, tool
                  (or null), requires_human_input (bool), expected_output, depends_on (list of step ids).
  execution_order : the flat ordered list of step ids. Where iterative attempts with feedback are needed insert a loop
                  block:  {"loop": {"steps": [...], "condition": "<step_id>.<outcome>==<value>", "max_iterations": <int>}}
                  or the single-step variant with "step" instead of "steps".

Personalize aggressively — agent roles, goals, step instructions must reference the learner's background. Output strict
JSON. No markdown fences, no commentary."""


USER_TEMPLATE = """QUERY:
{query}

LEARNER PROFILE:
self_description: {self_description}
skills: {skills}

Produce the complete plan."""


def extract_learner(lrn: dict) -> tuple[str, list[str]]:
    desc = lrn.get("self_description") or lrn.get("about_me") or ""
    skills = lrn.get("skills") or lrn.get("top_tags") or []
    return desc.strip(), list(skills)


def build_row(record: dict) -> dict:
    inp = record["plan"]["input"]
    query = inp["query"]
    desc, skills = extract_learner(inp["learner"])
    user_text = USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "question_id": str(record.get("question_id", "")),
        "profile_index": int(record.get("profile_index", 0)),
        # store gold plan as JSON string — HF datasets can be picky about nested dicts.
        "gold_plan": json.dumps(record["plan"]["output"], ensure_ascii=False),
        # learner profile is also JSON-serialized; reward_personalization_lite
        # parses it back. We keep both naming schemas under one key.
        "learner_profile": json.dumps(
            {"self_description": desc, "skills": skills}, ensure_ascii=False
        ),
    }


if __name__ == "__main__":
    import sys
    print(
        "ERROR: GRPO/build_grpo_prompts.py is no longer runnable as a CLI.\n"
        "Its old `main()` did row-level random shuffle which leaked\n"
        "question_ids across folds. Use the current pipeline instead:\n\n"
        "    python build_splits.py            # from repo root\n\n"
        "The verbatim deprecated CLI is preserved at\n"
        "    history/build_grpo_prompts_v1_row_shuffle.py\n",
        file=sys.stderr,
    )
    sys.exit(2)
