"""Compute basic rule-based SV/AR metrics for generated plan JSONL files.

This handles the automatable part of the experiment matrix:

  * SV: JSON parse + expected top-level schema
  * AR: dependency graph is acyclic

It expects a JSONL file with a text column containing generated JSON. Common
columns are: output, prediction, generated, text, assistant, chosen, rejected.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def schema_valid(plan: Any, mode: str) -> bool:
    if not isinstance(plan, dict):
        return False
    if mode == "pad":
        return isinstance(plan.get("agents"), list) and isinstance(plan.get("subtasks"), list)
    if mode == "sdp":
        return isinstance(plan.get("subtasks"), list) and isinstance(plan.get("execution_order"), list)
    return (
        isinstance(plan.get("agents"), list)
        and isinstance(plan.get("subtasks"), list)
        and isinstance(plan.get("execution_order"), list)
    )


def iter_steps(plan: dict[str, Any]):
    for subtask in plan.get("subtasks", []):
        for step in subtask.get("steps", []) or []:
            if isinstance(step, dict) and "id" in step:
                yield step


def acyclic(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    for step in iter_steps(plan):
        sid = str(step["id"])
        all_nodes.add(sid)
        for dep in step.get("depends_on", []) or []:
            dep = str(dep)
            graph[dep].add(sid)
            all_nodes.add(dep)

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for nxt in graph.get(node, set()):
            if not dfs(nxt):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(dfs(node) for node in list(all_nodes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--column", default="output")
    ap.add_argument("--mode", choices=["pad", "sdp", "full"], default="full")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    n = sv = ar = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            text = row.get(args.column, "")
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            plan = extract_json(text)
            ok_sv = schema_valid(plan, args.mode)
            ok_ar = acyclic(plan) if ok_sv else False
            sv += int(ok_sv)
            ar += int(ok_ar)

    metrics = {
        "n": n,
        "sv_rate": sv / n if n else 0.0,
        "ar_rate": ar / n if n else 0.0,
    }
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
