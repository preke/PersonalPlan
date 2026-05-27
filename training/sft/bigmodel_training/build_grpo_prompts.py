"""Build a prompt-only dataset from MAPLE for GRPO.

Each row contains the *prompt* (chat-template-rendered) plus side-channel
fields that the reward function needs:

  prompt        — rendered prompt string (chat-template applied) OR a
                  list of {role, content} messages (the trainer can render
                  if you set `processing_class`)
  question_id   — MAPLE id; used to look up cf_cache + precedence pairs
  profile_index — sub-id for same-question multi-profile rows
  gold_plan     — serialized ground-truth plan JSON (string)
                  parsed back to dict inside the reward function

The output is split into train/valid jsonl files compatible with the
HuggingFace `datasets` library.

Usage:

  python build_grpo_prompts.py \
      --input ../multi_agent_dataset_filtered_qap_latest.jsonl \
      --out data/grpo \
      --val-frac 0.05
"""
from __future__ import annotations

import argparse
import json
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=str(HERE / "data" / "grpo"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(in_path) if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * args.val_frac)))
    valid, train = records[:n_val], records[n_val:]

    print(f"Source        : {in_path}")
    print(f"Total records : {len(records)}")
    print(f"Train / Valid : {len(train)} / {len(valid)} (seed={args.seed})")

    for split, rows in [("train", train), ("valid", valid)]:
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(build_row(r), ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows):>4} rows → {out_path}")


if __name__ == "__main__":
    main()
