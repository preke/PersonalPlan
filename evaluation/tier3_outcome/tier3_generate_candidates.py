#!/usr/bin/env python
"""Generate GPT-5 candidate plans from the filtered Tier 3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You create multi-agent educational execution plans for programming learners.

Return only valid JSON. Do not include markdown, comments, or extra prose."""


USER_TEMPLATE = """Create a personalized educational plan for the learner and programming query below.

The plan should help the learner understand and solve the problem through staged tutoring, tool use when appropriate, validation, feedback, and consolidation. It should not simply reveal the final accepted answer at the beginning.

Query:
```text
{query}
```

Learner profile:
```json
{profile_json}
```

Required output schema:
```json
{{
  "agents": [
    {{
      "agent_role": "string",
      "goal": "string",
      "description": "string",
      "tools": ["string"]
    }}
  ],
  "subtasks": [
    {{
      "id": "S1",
      "name": "string",
      "subtask_objective": "string",
      "steps": [
        {{
          "id": "S1-1",
          "agent": "agent_role",
          "objective": "string",
          "instruction": "string",
          "tool": null,
          "requires_human_input": true,
          "expected_output": "string",
          "depends_on": []
        }}
      ]
    }}
  ],
  "execution_order": ["S1-1"]
}}
```

Quality requirements:

1. Match the learner's declared skills, background, and likely misconceptions.
2. Use a clear instructional progression: diagnose, explain/demonstrate, apply, validate, feedback, consolidate.
3. Include executable code/tool steps only when they genuinely help learning.
4. Preserve dependencies between steps.
5. Avoid direct answer leakage before the learner has attempted the core reasoning.
6. Make the plan specific to this query and this learner, not a generic template."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def get_by_path(row: dict[str, Any], key_path: str) -> Any:
    current: Any = row
    for part in key_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    query = str(get_by_path(row, "input.query") or row.get("query") or "")
    learner = get_by_path(row, "input.learner") or row.get("learner") or row.get("profile") or {}
    user_prompt = USER_TEMPLATE.format(
        query=query,
        profile_json=json.dumps(learner, ensure_ascii=False, indent=2, sort_keys=True),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_json(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, flags=re.S)
        if match:
            return json.loads(match.group(1))
        raise


def call_openai(messages: list[dict[str, str]], model: str, base_url: str | None, api_key: str | None) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package before running candidate generation.") from exc

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=Path("multi_agent_dataset_filtered_qap.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("gpt5_candidate_plans.jsonl"))
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source_file.exists():
        raise FileNotFoundError(f"Missing source JSONL file: {args.source_file}")

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set. Use --api-key-env or --api-key.")

    if args.overwrite and args.out_file.exists():
        args.out_file.unlink()
    elif args.out_file.exists():
        raise FileExistsError(f"{args.out_file} already exists. Use --overwrite to replace it.")

    rows = read_jsonl(args.source_file)
    if args.limit_items is not None:
        rows = rows[: args.limit_items]

    for index, row in enumerate(rows, 1):
        raw_text = call_openai(build_messages(row), args.model, args.base_url, api_key)
        candidate_output = extract_json(raw_text)
        append_jsonl(
            args.out_file,
            {
                "input": row.get("input", {}),
                "output": candidate_output,
                "generation_model": args.model,
            },
        )
        print(f"Generated {index}/{len(rows)} candidate plans", file=sys.stderr)


if __name__ == "__main__":
    main()
