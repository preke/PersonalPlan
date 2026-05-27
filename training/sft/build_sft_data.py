"""SFT message-builder helpers for the three-stage hierarchical training
described in §4 of the paper.

THIS FILE NO LONGER HAS A CLI. Its old `main()` did a row-level random
shuffle to materialize PAD/SDP train/valid jsonl — that approach
leaked question_ids across splits (same query / different profile
straddled folds). It has been superseded by the repo-root
``build_splits.py``, which does question_id-stratified train/dev/test
splitting (see EXPERIMENT_PLAN_2026-05-15.html §2.2).

The verbatim old version is preserved at
  history/build_sft_data_v1_row_shuffle.py
for reference. Do not resurrect that logic here.

THIS FILE STILL EXPORTS the message-builder helpers
(``build_pad_messages``, ``build_sdp_messages``, ``pad_target``,
``sdp_target``, ``extract_learner``, ``agent_record``, ``load_jsonl``,
``write_jsonl``) which are imported by:
  - build_splits.py            (current split-pipeline entry point)
  - SFT/joint_alignment_data.py (Stage 3 data generation)

To (re)generate the train/dev/test splits, run from the repo root:
  python build_splits.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prompts import PAD_SYSTEM, PAD_USER_TEMPLATE, SDP_SYSTEM, SDP_USER_TEMPLATE

HERE = Path(__file__).resolve().parent


# ---------- field extractors -------------------------------------------------

def extract_learner(lrn: dict[str, Any]) -> tuple[str, list[str]]:
    """Two schema variants exist; unify."""
    desc = lrn.get("self_description") or lrn.get("about_me") or ""
    skills = lrn.get("skills") or lrn.get("top_tags") or []
    return desc.strip(), list(skills)


def agent_record(a: dict[str, Any]) -> dict[str, Any]:
    """Normalize the agent record. backstory/description are aliases."""
    out = {
        "agent_role": a.get("agent_role", ""),
        "goal": a.get("goal", ""),
        "backstory": a.get("backstory") or a.get("description") or "",
        "tools": list(a.get("tools") or []),
    }
    return out


def pad_target(plan_output: dict[str, Any]) -> dict[str, Any]:
    """High-level slice (T, A): agents + subtask metadata, NO steps."""
    return {
        "agents": [agent_record(a) for a in plan_output["agents"]],
        "subtasks": [
            {
                "id": s["id"],
                "name": s["name"],
                "subtask_objective": s["subtask_objective"],
            }
            for s in plan_output["subtasks"]
        ],
    }


def sdp_target(plan_output: dict[str, Any]) -> dict[str, Any]:
    """Lower-level slice (S, O): steps grouped by subtask + execution_order."""
    return {
        "subtasks": [
            {"id": s["id"], "steps": list(s.get("steps", []))}
            for s in plan_output["subtasks"]
        ],
        "execution_order": list(plan_output.get("execution_order", [])),
    }


# ---------- message builders -------------------------------------------------

def build_pad_messages(record: dict[str, Any]) -> dict[str, Any]:
    inp = record["plan"]["input"]
    query = inp["query"]
    desc, skills = extract_learner(inp["learner"])
    target = pad_target(record["plan"]["output"])

    user_text = PAD_USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
    )
    assistant_text = json.dumps(target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": PAD_SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "question_id": record["question_id"],
        "profile_index": record["profile_index"],
    }


def build_sdp_messages(record: dict[str, Any]) -> dict[str, Any]:
    inp = record["plan"]["input"]
    query = inp["query"]
    desc, skills = extract_learner(inp["learner"])
    scaffold = pad_target(record["plan"]["output"])  # gold (T,A) used as input
    target = sdp_target(record["plan"]["output"])

    user_text = SDP_USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
        agents=json.dumps(scaffold["agents"], ensure_ascii=False, indent=2),
        subtasks=json.dumps(scaffold["subtasks"], ensure_ascii=False, indent=2),
    )
    assistant_text = json.dumps(target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": SDP_SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "question_id": record["question_id"],
        "profile_index": record["profile_index"],
    }


# ---------- driver -----------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys
    print(
        "ERROR: SFT/build_sft_data.py is no longer runnable as a CLI.\n"
        "Its old `main()` did row-level random shuffle which leaked\n"
        "question_ids across folds. Use the current pipeline instead:\n\n"
        "    python build_splits.py            # from repo root\n\n"
        "The verbatim deprecated CLI is preserved at\n"
        "    history/build_sft_data_v1_row_shuffle.py\n",
        file=sys.stderr,
    )
    sys.exit(2)
