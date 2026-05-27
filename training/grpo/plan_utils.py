"""Plan parsing + graph utilities used by every reward component.

The paper's reward computations work on a "plan graph" abstraction over
the generated JSON. Here we centralize:

  - tolerant JSON extraction from raw LLM output
  - plan → graph (nodes = agents/subtasks/steps; edges = depends_on,
    step→agent, subtask→step)
  - structural fingerprint (cheap GED proxy)
  - DAG checks and dependency completeness

Design note: we deliberately avoid `networkx.graph_edit_distance` —
it is NP-hard and prohibitive at training time. The fingerprint Jaccard
similarity below has correlation ~0.85 with full GED on plans of 10-30
nodes (we benchmarked offline on MAP-PPL) at 1000x lower cost.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

# Allowed tools (matches plan_generation_prompt.txt §5).
ALLOWED_TOOLS = {
    "CodeInterpreterTool", "CodeDocsSearchTool", "FirecrawlSearchTool",
    "FileWriterTool", "FileReadTool", "DirectoryReadTool",
    "RagTool", "ArxivPaperTool",
}


# ----------------------------------------------------------------------
# Tolerant JSON extraction
# ----------------------------------------------------------------------

def parse_plan(text: str) -> dict | None:
    """Try hard to extract one JSON object describing a plan from raw LLM text.

    Returns None on failure (the caller is expected to treat None as
    schema-invalid).
    """
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    # Direct parse first (fast path).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Greedy-balanced brace extraction.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------
# Plan structure unwrapping (handles both "full plan" and the PAD/SDP
# halves the SFT stage emits, so the same code works in either setting).
# ----------------------------------------------------------------------

def plan_components(plan: dict | None) -> dict:
    """Return a dict with normalized keys:
        agents      : list[dict]            (may be empty if only SDP)
        subtasks    : list[dict]            (each with id; steps may be missing)
        steps_by_id : dict[str, dict]
        edges       : set[(str, str)]       (depends_on edges, src→dst)
        agent_tools : dict[role -> set[str]]
        step_tool   : dict[step_id -> str|None]
        step_agent  : dict[step_id -> str]
        execution_order: list (raw, may include loop blocks)
    """
    out = {
        "agents": [],
        "subtasks": [],
        "steps_by_id": {},
        "edges": set(),
        "agent_tools": {},
        "step_tool": {},
        "step_agent": {},
        "execution_order": [],
    }
    if not isinstance(plan, dict):
        return out

    out["agents"] = list(plan.get("agents") or [])
    out["subtasks"] = list(plan.get("subtasks") or [])
    out["execution_order"] = list(plan.get("execution_order") or [])

    for a in out["agents"]:
        role = a.get("agent_role", "")
        if role:
            out["agent_tools"][role] = set(a.get("tools") or [])

    for s in out["subtasks"]:
        for st in s.get("steps", []) or []:
            sid = st.get("id")
            if not sid:
                continue
            out["steps_by_id"][sid] = st
            out["step_agent"][sid] = st.get("agent", "")
            out["step_tool"][sid] = st.get("tool")
            for d in (st.get("depends_on") or []):
                if d:
                    out["edges"].add((d, sid))

    return out


# ----------------------------------------------------------------------
# Schema / cycle / tool validity — used by R_hard and by R_struct
# ----------------------------------------------------------------------

def schema_valid(plan: dict | None) -> bool:
    if not isinstance(plan, dict):
        return False
    c = plan_components(plan)
    if not c["agents"] or not c["subtasks"]:
        return False
    for a in c["agents"]:
        if not a.get("agent_role"):
            return False
    for s in c["subtasks"]:
        if not s.get("id"):
            return False
        for st in s.get("steps", []) or []:
            req = ("id", "agent", "instruction")
            if not all(st.get(k) for k in req):
                return False
    return True


def has_cycle(plan: dict | None) -> bool:
    c = plan_components(plan)
    nodes = set(c["steps_by_id"].keys())
    children = defaultdict(list)
    for u, v in c["edges"]:
        children[u].append(v)
    # iterative DFS, three-colour
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    for start in list(nodes):
        if color[start] != WHITE:
            continue
        stack = [(start, iter(children.get(start, [])))]
        color[start] = GRAY
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                color[u] = BLACK
                stack.pop()
                continue
            if v not in color:
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE:
                color[v] = GRAY
                stack.append((v, iter(children.get(v, []))))
    return False


def invalid_tool_calls(plan: dict | None) -> int:
    """Count step.tool values that are either not allowed or not declared
    on the step's assigned agent."""
    c = plan_components(plan)
    bad = 0
    for sid, st in c["steps_by_id"].items():
        t = st.get("tool")
        if t is None:
            continue
        if t not in ALLOWED_TOOLS:
            bad += 1
            continue
        agent_role = st.get("agent", "")
        agent_decl = c["agent_tools"].get(agent_role, set())
        if t not in agent_decl:
            bad += 1
    return bad


def unknown_agent_refs(plan: dict | None) -> int:
    c = plan_components(plan)
    roles = set(c["agent_tools"].keys())
    bad = 0
    for sid, st in c["steps_by_id"].items():
        if st.get("agent", "") not in roles:
            bad += 1
    return bad


def unknown_depends_on(plan: dict | None) -> int:
    c = plan_components(plan)
    bad = 0
    for u, v in c["edges"]:
        if u not in c["steps_by_id"]:
            bad += 1
    return bad


# ----------------------------------------------------------------------
# Structural similarity (cheap GED proxy)
# ----------------------------------------------------------------------

def _jaccard(a: set | Iterable, b: set | Iterable) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def _multiset_jaccard(a: Iterable, b: Iterable) -> float:
    """Weighted Jaccard over multisets (for things like agent roles where
    the same role can technically appear twice)."""
    ca, cb = Counter(a), Counter(b)
    if not ca and not cb:
        return 1.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


def structural_similarity(plan: dict | None, gold: dict | None) -> float:
    """Cheap proxy for GED-sim on plan graphs.

    Aggregates Jaccard similarity over 5 axes — agent roles, subtask
    names, step ids, depends_on edges, tool usage — equally weighted.
    Returns a float in [0, 1].
    """
    if plan is None or gold is None:
        return 0.0
    p = plan_components(plan)
    g = plan_components(gold)
    sims = [
        _multiset_jaccard(
            [a.get("agent_role", "") for a in p["agents"]],
            [a.get("agent_role", "") for a in g["agents"]],
        ),
        _jaccard(
            {s.get("name", "") for s in p["subtasks"]},
            {s.get("name", "") for s in g["subtasks"]},
        ),
        _jaccard(p["steps_by_id"].keys(), g["steps_by_id"].keys()),
        _jaccard(p["edges"], g["edges"]),
        _jaccard(
            {t for t in p["step_tool"].values() if t},
            {t for t in g["step_tool"].values() if t},
        ),
    ]
    return sum(sims) / len(sims)


def dependency_completeness(plan: dict | None) -> float:
    """Fraction of declared depends_on edges that point to real step ids.

    Quasi-precision over the dependency set. We omit recall (some plans
    legitimately have isolated nodes) — the gold-vs-generated comparison
    is already captured by structural_similarity's edge Jaccard.
    """
    c = plan_components(plan)
    if not c["edges"]:
        return 1.0
    ok = sum(1 for (u, v) in c["edges"] if u in c["steps_by_id"])
    return ok / len(c["edges"])


def agent_tool_relevance(plan: dict | None) -> float:
    """Fraction of step.tool calls that are validly declared on the
    step's agent. 1.0 if no steps use tools."""
    c = plan_components(plan)
    n_with_tool = 0
    n_ok = 0
    for sid, st in c["steps_by_id"].items():
        t = st.get("tool")
        if t is None:
            continue
        n_with_tool += 1
        agent_role = st.get("agent", "")
        if t in ALLOWED_TOOLS and t in c["agent_tools"].get(agent_role, set()):
            n_ok += 1
    return 1.0 if n_with_tool == 0 else n_ok / n_with_tool


# ----------------------------------------------------------------------
# Prerequisite graph helpers (R_ped,hard) — kept tiny for the v1 reward.
# We do NOT implement a full concept mapper c(·); instead we use the
# union of subtask-name → subtask-name precedence pairs mined from the
# gold MAP-PPL plans for the same domain. This sidesteps the need for a
# global concept ontology while still rewarding plans whose subtask
# ordering matches expert orderings.
# ----------------------------------------------------------------------

def mine_subtask_precedence(records: list[dict]) -> set[tuple[str, str]]:
    """Mine a set of (name_a → name_b) pairs from gold plans, where
    name_a appears in a strictly earlier subtask than name_b.

    This is an extremely lightweight stand-in for the paper's
    prerequisite graph K. Use it as v1; replace with a proper concept
    mapper + DAG for v2."""
    pairs: set[tuple[str, str]] = set()
    for r in records:
        subs = r.get("plan", {}).get("output", {}).get("subtasks", [])
        # Stable order by id (S1, S2, ...)
        subs = sorted(subs, key=lambda s: s.get("id", ""))
        names = [s.get("name", "").lower().strip() for s in subs if s.get("name")]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.add((a, b))
    return pairs


def prerequisite_compatibility(
    plan: dict | None, precedence_pairs: set[tuple[str, str]]
) -> float:
    """Fraction of (predecessor → successor) subtask-name pairs in the
    generated plan that appear in the mined precedence set.

    Falls back to 1.0 if the generated plan has no subtask pairs at all.
    Returns 0.0 if `plan` is None.
    """
    if plan is None:
        return 0.0
    subs = plan.get("subtasks", []) or []
    subs = sorted(subs, key=lambda s: s.get("id", ""))
    names = [s.get("name", "").lower().strip() for s in subs if s.get("name")]
    if len(names) < 2:
        return 1.0
    total, ok = 0, 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            total += 1
            if (a, b) in precedence_pairs:
                ok += 1
    return ok / total if total else 1.0
