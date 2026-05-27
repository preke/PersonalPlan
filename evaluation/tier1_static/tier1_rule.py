#!/usr/bin/env python
"""
Tier 1 §11.1 — 5 rule-based metrics + shared schema / I/O.

Implements:
    SchemaValid_Pass    composite schema validity gate (validate_plan equiv.)
    ToolAgent_Perm      step.tool ∈ step.agent's tool list
    Topo_Consist        dependency edges respect execution_order ordering
    ATR                 agent-tool-subtask semantic alignment
                        (3-indicator mean over tool-bearing steps; embedding
                        mode default with MiniLM cos>threshold)
    GED-sim             graph edit distance similarity to gold

The 3 new internal-consistency metrics (SchemaValid_Pass / ToolAgent_Perm /
Topo_Consist) replace the 2026-05-22 SV/AR/DC/TBV gate set: on the 305-plan
v15 test split × 4 baselines the original 4 metrics saturated at 1.000 (no
discrimination, p ≈ 1.0), while the 3 replacements all show statistically
significant baseline separation (F ≥ 5.25, p ≤ 0.0054).

Also exports the shared Pydantic schema, JSONL helpers, and outer-wrapper
unwrap() used by tier1_judge.py and tier1_counterfactual.py.

Standalone CLI:
    .venvs/tier1_eval/Scripts/python.exe tier1_rule.py \\
        --input  evaluation_results/baselines/aop/plans.jsonl \\
        --gold   multi_agent_dataset_filtered_qap_v15_goodplus.jsonl \\
        --output evaluation_results/baselines/aop/tier1_rule.csv \\
        --limit 3
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm


# ============================================================================
# Shared Pydantic schema  (PlanPayload matches real project layout, see
# baselines/common/schema_validator.py)
# ============================================================================

class StepSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    agent: str
    objective: str = ""
    instruction: str = ""
    tool: Optional[str] = None
    requires_human_input: bool = False
    expected_output: str = ""
    depends_on: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubtaskSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    subtask_objective: str = ""
    steps: list[StepSpec] = Field(default_factory=list)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    agent_role: str
    goal: str = ""
    backstory: str = ""
    tools: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    agents: list[AgentSpec]
    subtasks: list[SubtaskSpec]
    execution_order: list[Union[str, dict]]


class PlanInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str
    learner: dict[str, Any] = Field(default_factory=dict)


class PlanPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: PlanInput
    output: PlanOutput


# ============================================================================
# Shared accessors  (also used by tier1_judge / tier1_counterfactual)
# ============================================================================

def unwrap(row: dict) -> dict:
    """Strip the outer JSONL row wrapper so callers see {input, output}.

    Handles both common patterns:
      * baseline output: {question_id, profile_index, generated_plan: {...}}
      * gold dataset:    {question_id, profile_index, plan: {...}}
    """
    for key in ("generated_plan", "plan"):
        v = row.get(key) if isinstance(row, dict) else None
        if isinstance(v, dict):
            return v
    return row


def get_output(plan: dict) -> dict:
    return plan["output"] if isinstance(plan.get("output"), dict) else plan


def get_input(plan: dict) -> dict:
    return plan["input"] if isinstance(plan.get("input"), dict) else {}


def iter_steps(plan: dict) -> list[dict]:
    """Flatten all steps from subtasks (project schema nests steps)."""
    out: list[dict] = []
    for st in get_output(plan).get("subtasks", []) or []:
        out.extend(st.get("steps", []) or [])
    return out


def subtask_of_step(plan: dict, step_id: str) -> dict:
    for st in get_output(plan).get("subtasks", []) or []:
        for s in st.get("steps", []) or []:
            if s.get("id") == step_id:
                return st
    return {}


def flatten_eo(eo: Any) -> list[str]:
    """Flatten execution_order to a flat ordered list of step ids.

    execution_order entries may be either a step id string, or a loop dict of
    the form {"loop": {"step": "<id>", ...}} or {"loop": {"steps": [...], ...}}.
    Loop bodies are inlined in declaration order (no expansion of iterations).
    """
    out: list[str] = []
    if not eo:
        return out
    for item in eo:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for v in item.values():
                if isinstance(v, dict):
                    if isinstance(v.get("step"), str):
                        out.append(v["step"])
                    if isinstance(v.get("steps"), list):
                        out.extend(flatten_eo(v["steps"]))
                elif isinstance(v, list):
                    out.extend(flatten_eo(v))
    return out


# ============================================================================
# Graph builder  (shared with tier1_counterfactual.ged_sim sanity prints)
# ============================================================================

def build_step_graph(plan: dict) -> nx.DiGraph:
    """Build DAG node=step.id, edge=dep -> step."""
    g = nx.DiGraph()
    for step in iter_steps(plan):
        sid = step.get("id")
        if not sid:
            continue
        g.add_node(sid, agent=step.get("agent"))
        for dep in step.get("depends_on") or []:
            g.add_edge(dep, sid)
    return g


# ============================================================================
# §11.1 Rule-based metrics (5 internal-consistency replacements)
# ============================================================================

VALID_TOOLS: set[str] = {
    "FirecrawlSearchTool", "RagTool", "CodeInterpreterTool",
    "DirectoryReadTool", "FileReadTool", "FileWriterTool",
    "CodeDocsSearchTool", "ArxivPaperTool",
}


def schema_valid_pass(plan: dict) -> int:
    """SchemaValid_Pass ∈ {0,1}: composite gate mirroring the schema check used
    by baselines/common/schema_validator.validate_plan.

    A plan scores 1 only if it satisfies ALL of:
      - top-level input/output blocks present
      - input has query + learner.{about_me, top_tags}
      - output has non-empty agents, subtasks, execution_order
      - every agent has agent_role/goal/backstory/tools, and each declared
        tool is in the 8-tool whitelist
      - every step has id/agent/objective/instruction/tool/requires_human_input/
        expected_output/depends_on
      - step.agent is one of the declared agent roles
      - step.tool is null OR (tool in whitelist AND tool in agent's tools)
      - flatten(execution_order) is exactly the set of all step ids
        (no orphan defined steps, no phantom EO entries)
    """
    try:
        if not isinstance(plan, dict):
            return 0
        if "input" not in plan or "output" not in plan:
            return 0

        inp = plan["input"]
        if not isinstance(inp, dict):
            return 0
        if "query" not in inp or "learner" not in inp:
            return 0
        lr = inp["learner"]
        if not isinstance(lr, dict):
            return 0
        if "about_me" not in lr or "top_tags" not in lr:
            return 0

        out = plan["output"]
        if not isinstance(out, dict):
            return 0
        for k in ("agents", "subtasks", "execution_order"):
            if k not in out:
                return 0

        agents = out["agents"]
        if not isinstance(agents, list) or not agents:
            return 0
        agent_names: set = set()
        agent_tools: dict[str, set] = {}
        for ag in agents:
            if not isinstance(ag, dict):
                return 0
            for k in ("agent_role", "goal", "backstory", "tools"):
                if k not in ag:
                    return 0
            if not isinstance(ag["tools"], list):
                return 0
            for t in ag["tools"]:
                if t not in VALID_TOOLS:
                    return 0
            agent_names.add(ag["agent_role"])
            agent_tools[ag["agent_role"]] = set(ag["tools"])

        if not isinstance(out["subtasks"], list):
            return 0
        all_step_ids: set = set()
        for st in out["subtasks"]:
            if not isinstance(st, dict):
                return 0
            for k in ("id", "name", "subtask_objective", "steps"):
                if k not in st:
                    return 0
            if not isinstance(st["steps"], list):
                return 0
            for step in st["steps"]:
                if not isinstance(step, dict):
                    return 0
                for k in ("id", "agent", "objective", "instruction", "tool",
                          "requires_human_input", "expected_output",
                          "depends_on"):
                    if k not in step:
                        return 0
                if step["agent"] not in agent_names:
                    return 0
                if step["tool"] is not None:
                    if step["tool"] not in VALID_TOOLS:
                        return 0
                    if step["tool"] not in agent_tools[step["agent"]]:
                        return 0
                all_step_ids.add(step["id"])

        flat = set(flatten_eo(out["execution_order"]))
        if flat != all_step_ids:
            return 0
        return 1
    except Exception:
        return 0


def tool_agent_perm(plan: dict) -> float:
    """ToolAgent_Perm ∈ [0,1]: fraction of tool-bearing steps whose declared
    tool is in their agent's allowed tool list.

    Returns NaN when the plan has no tool-bearing step (consistent with ATR's
    no-tool branch — distinguishes "no tool to check" from "all checks passed").
    """
    out = get_output(plan)
    agents = {a.get("agent_role"): set(a.get("tools") or [])
              for a in out.get("agents", []) or []}
    total = valid = 0
    for step in iter_steps(plan):
        t = step.get("tool")
        if not t:
            continue
        total += 1
        if t in agents.get(step.get("agent", ""), set()):
            valid += 1
    return float("nan") if total == 0 else valid / total


def topo_consist(plan: dict) -> float:
    """Topo_Consist ∈ [0,1]: fraction of depends_on edges where the dependency
    appears EARLIER than the dependent step in execution_order.

    Edges whose endpoints aren't in execution_order are dropped from both
    numerator and denominator.  Returns 1.0 when no scorable edges exist."""
    out = get_output(plan)
    flat = flatten_eo(out.get("execution_order") or [])
    pos = {sid: i for i, sid in enumerate(flat)}
    total = valid = 0
    for step in iter_steps(plan):
        sid = step.get("id")
        if sid not in pos:
            continue
        for d in step.get("depends_on") or []:
            if d not in pos:
                continue
            total += 1
            if pos[d] < pos[sid]:
                valid += 1
    return 1.0 if total == 0 else valid / total


# ============================================================================
# ATR (kept) — agent-tool-subtask semantic alignment
# ============================================================================

# Role -> {subtask_kw: [...], recommended: [...]}.
# Keyword/tool lists keep §11.1 doc-style coverage (CodeDemonstrator /
# ConceptInstructor / SecurityAdvisor + 7 extras for real baselines).
# Unknown roles map to empty -> indicators degrade to 0 for that step.
ROLE_TOOL_MAPPING: dict[str, dict[str, list[str]]] = {
    "CodeDemonstrator": {
        "subtask_kw": ["implement", "demo", "example", "code", "build",
                       "develop", "prototype"],
        "recommended": ["CodeInterpreterTool", "FileWriterTool"],
    },
    "ConceptInstructor": {
        "subtask_kw": ["concept", "introduce", "intro", "explain", "theory",
                       "foundation", "overview", "principle"],
        "recommended": ["RagTool", "CodeDocsSearchTool", "ArxivPaperTool"],
    },
    "SecurityAdvisor": {
        "subtask_kw": ["vulnerability", "security", "exploit", "attack",
                       "defense", "scan", "audit"],
        "recommended": ["CodeInterpreterTool", "FirecrawlSearchTool"],
    },
    "DocSpecialist": {
        "subtask_kw": ["doc", "documentation", "reference", "manual", "api",
                       "spec"],
        "recommended": ["CodeDocsSearchTool", "RagTool", "DirectoryReadTool",
                        "FileReadTool"],
    },
    "TestEngineer": {
        "subtask_kw": ["test", "verify", "validate", "assert", "qa", "check"],
        "recommended": ["CodeInterpreterTool", "FileWriterTool"],
    },
    "ResearchAssistant": {
        "subtask_kw": ["research", "literature", "survey", "paper", "review",
                       "study"],
        "recommended": ["ArxivPaperTool", "FirecrawlSearchTool", "RagTool"],
    },
    "DataAnalyst": {
        "subtask_kw": ["data", "analyze", "analysis", "statistic", "visualiz",
                       "metric", "plot"],
        "recommended": ["CodeInterpreterTool", "FileReadTool"],
    },
    "DebugAssistant": {
        "subtask_kw": ["debug", "fix", "error", "trace", "diagnose",
                       "troubleshoot"],
        "recommended": ["CodeInterpreterTool", "FileReadTool"],
    },
    "APIDesigner": {
        "subtask_kw": ["design", "interface", "api", "schema", "contract",
                       "endpoint"],
        "recommended": ["CodeDocsSearchTool", "FileWriterTool"],
    },
    "BackendEngineer": {
        "subtask_kw": ["backend", "server", "database", "service", "deploy"],
        "recommended": ["CodeInterpreterTool", "FileWriterTool",
                        "DirectoryReadTool"],
    },
}


def _role_map_for(role: str) -> dict[str, list[str]]:
    if role in ROLE_TOOL_MAPPING:
        return ROLE_TOOL_MAPPING[role]
    rl = role.lower()
    for k, v in ROLE_TOOL_MAPPING.items():
        if k.lower() in rl or rl in k.lower():
            return v
    return {}


def atr(plan: dict) -> float:
    """ATR ∈ [0,1] keyword mode: mean of (I_AS + I_TA + I_TS) / 3 over
    tool-bearing steps only. Returns NaN when no tool-bearing step.
    Kept for the doc ✅/❌ example regression tests; embedding mode is the
    default for real datasets."""
    out = get_output(plan)
    agents = {a["agent_role"]: a for a in out.get("agents", []) or []}
    tool_steps = [s for s in iter_steps(plan) if s.get("tool")]
    if not tool_steps:
        return float("nan")
    total = 0.0
    for step in tool_steps:
        role = step.get("agent", "") or ""
        ag = agents.get(role, {}) or {}
        rm = _role_map_for(role)
        st = subtask_of_step(plan, step.get("id", ""))
        subtask_text = ((st.get("name") or "") + " " +
                        (st.get("subtask_objective") or "")).lower()
        i_as = int(any(kw in subtask_text for kw in rm.get("subtask_kw", [])))
        tool = step["tool"]
        i_ta = int(tool in (ag.get("tools") or []))
        i_ts = int(tool in rm.get("recommended", []))
        total += (i_as + i_ta + i_ts) / 3.0
    return total / len(tool_steps)


# ----------------------------------------------------------------------------
# ATR embedding mode (sentence-transformers; default)
# ----------------------------------------------------------------------------
TOOL_DESCRIPTIONS: dict[str, str] = {
    "FirecrawlSearchTool":  "web search and crawl for online resources",
    "RagTool":              "retrieval-augmented query of an indexed knowledge base",
    "CodeInterpreterTool":  "sandboxed code execution to run and test code",
    "DirectoryReadTool":    "list files in a directory",
    "FileReadTool":         "read contents of a file",
    "FileWriterTool":       "write contents to a file",
    "CodeDocsSearchTool":   "search code documentation and API references",
    "ArxivPaperTool":       "search and retrieve arxiv academic papers",
}

_EMB_MODEL = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer(model_name)
    return _EMB_MODEL


@lru_cache(maxsize=16384)
def _embed_text(text: str, model_name: str = "all-MiniLM-L6-v2"
                ) -> tuple[float, ...]:
    m = _get_embedding_model(model_name)
    v = m.encode([text], normalize_embeddings=True)[0]
    return tuple(float(x) for x in v)


def _cos_normalized(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize_name(s: str) -> str:
    """snake_case → 'snake case'; CamelCase → 'camel case'; lower."""
    import re as _re
    s = _re.sub(r"[_\-]+", " ", s)
    s = _re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.lower().strip()


def atr_embedding(plan: dict, threshold: float = 0.10,
                  model_name: str = "all-MiniLM-L6-v2") -> float:
    """ATR over tool-bearing steps with 3 SEMANTIC indicators (full doc §11.1
    intent: "agent / tool / subtask 三者的语义一致性").

      I_AS = cos(role_descriptor,  subtask_text   ) > threshold
      I_TA = cos(tool_description, role_descriptor) > threshold
      I_TS = cos(tool_description, step_instruction) > threshold

    Returns NaN when the plan has no tool-bearing step."""
    out = get_output(plan)
    agents = {a["agent_role"]: a for a in out.get("agents", []) or []}
    tool_steps = [s for s in iter_steps(plan) if s.get("tool")]
    if not tool_steps:
        return float("nan")
    total = 0.0
    for step in tool_steps:
        role = step.get("agent", "") or ""
        ag = agents.get(role, {}) or {}
        st = subtask_of_step(plan, step.get("id", ""))

        role_descriptor = _normalize_name(role)
        if ag.get("goal"):
            role_descriptor = (role_descriptor + ". " + ag["goal"]).strip()
        subtask_text = (((st.get("name") or "") + ". ")
                        + (st.get("subtask_objective") or "")).strip()
        step_text = (step.get("instruction") or "").strip()
        if not step_text:
            step_text = (step.get("objective") or "").strip()
        if not step_text:
            step_text = subtask_text
        tool = step["tool"]
        tool_desc = TOOL_DESCRIPTIONS.get(tool, _normalize_name(tool))

        if role_descriptor and subtask_text:
            i_as = int(_cos_normalized(_embed_text(role_descriptor),
                                        _embed_text(subtask_text)) > threshold)
        else:
            i_as = 0
        if tool_desc and role_descriptor:
            i_ta = int(_cos_normalized(_embed_text(tool_desc),
                                        _embed_text(role_descriptor)) > threshold)
        else:
            i_ta = 0
        if tool_desc and step_text:
            i_ts = int(_cos_normalized(_embed_text(tool_desc),
                                        _embed_text(step_text)) > threshold)
        else:
            i_ts = 0
        total += (i_as + i_ta + i_ts) / 3.0
    return total / len(tool_steps)


# ============================================================================
# GED-sim (kept) — graph edit distance similarity to gold
# ============================================================================

def ged_sim(plan: dict, gold: dict, timeout: float = 2.0) -> float:
    """GED-sim ∈ [0,1]: 1 - GED / max(|S|+|O|, |S*|+|O*|).
    Falls back to nx.optimize_graph_edit_distance on timeout.

    Returns NaN when both graphs are empty (data issue, not "perfect similarity").
    """
    g1 = build_step_graph(plan)
    g2 = build_step_graph(gold)
    ged: Optional[float] = None
    try:
        ged = nx.graph_edit_distance(g1, g2, timeout=timeout)
    except Exception:
        ged = None
    if ged is None:
        try:
            ged = next(nx.optimize_graph_edit_distance(g1, g2))
        except (StopIteration, Exception):
            return float("nan")
    denom = max(g1.number_of_nodes() + g1.number_of_edges(),
                g2.number_of_nodes() + g2.number_of_edges())
    if denom == 0:
        return float("nan")
    return max(0.0, 1.0 - ged / denom)


# ============================================================================
# Per-plan driver
# ============================================================================

RULE_COLS = [
    "SchemaValid_Pass", "ToolAgent_Perm", "Topo_Consist", "ATR", "GED_sim",
]


def eval_rule(plan: dict, gold: Optional[dict] = None,
              atr_mode: str = "embedding",
              atr_threshold: float = 0.10) -> dict[str, float]:
    """Per-plan rule metrics (5 columns).

    The first 3 (SchemaValid_Pass / ToolAgent_Perm / Topo_Consist) are
    internal-consistency replacements for the original SV/AR/DC/TBV gate set,
    which saturated at 1.000 across all 4 baselines on the 305 v15 test split
    (p ≈ 1.0).  On the same data the 3 replacements all show statistically
    significant baseline separation (F ≥ 5.25, p ≤ 0.0054).

    ATR default is "embedding" mode (sentence-transformers MiniLM, cos>0.10).
    Use "keyword" only for the doc ✅/❌ example regression tests.
    """
    atr_val = (atr(plan) if atr_mode == "keyword"
               else atr_embedding(plan, threshold=atr_threshold))
    return {
        "SchemaValid_Pass": schema_valid_pass(plan),
        "ToolAgent_Perm":   tool_agent_perm(plan),
        "Topo_Consist":     topo_consist(plan),
        "ATR":              atr_val,
        "GED_sim":          ged_sim(plan, gold) if gold is not None else float("nan"),
    }


# ============================================================================
# Shared JSONL I/O + lookup builders (used by all 3 modules)
# ============================================================================

def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def outer_qid(row: dict) -> Optional[str]:
    return (row.get("question_id")
            or row.get("qid")
            or get_input(unwrap(row)).get("question_id"))


def outer_pidx(row: dict) -> Optional[int]:
    for k in ("profile_index", "profile_idx"):
        if k in row:
            v = row[k]
            break
    else:
        inp = get_input(unwrap(row))
        v = inp.get("profile_index", inp.get("profile_idx"))
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_gold_map(gold_path: str | Path
                   ) -> dict[tuple[Optional[str], Optional[int]], dict]:
    out: dict[tuple[Optional[str], Optional[int]], dict] = {}
    for row in load_jsonl(gold_path):
        qid = row.get("question_id") or row.get("qid")
        try:
            pidx = int(row["profile_index"]) if "profile_index" in row else None
        except (TypeError, ValueError):
            pidx = None
        out[(qid, pidx)] = unwrap(row)
    return out


def build_qap_lookup(qap_path: str | Path
                     ) -> dict[tuple[str, int], str]:
    """{(question_id, profile_index): accepted_answer} from filtered_qap.
    Tolerant: skips lines that fail JSON parsing."""
    out: dict[tuple[str, int], str] = {}
    n_bad = 0
    with open(qap_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            qid = row.get("question_id")
            for idx, pa in enumerate(row.get("profiles_answers") or []):
                ans = (pa or {}).get("answer", "")
                if qid is not None and ans:
                    out[(qid, idx)] = ans
    if n_bad:
        import sys as _sys
        print(f"[build_qap_lookup] WARNING: skipped {n_bad} malformed row(s)",
              file=_sys.stderr)
    return out


# ============================================================================
# CLI
# ============================================================================

def _summarize(df: pd.DataFrame) -> None:
    cols = [c for c in RULE_COLS if c in df.columns]
    if not cols:
        return
    print("\n=== Mean per-metric (Tier 1 §11.1 rule-based) ===")
    print(df[cols].apply(pd.to_numeric, errors="coerce").mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tier 1 §11.1: 5 rule-based metrics (no LLM).")
    p.add_argument("--input", required=True, help="Plans JSONL.")
    p.add_argument("--gold", default=None,
                   help="Gold JSONL for GED-sim (matched by qid+profile_index).")
    p.add_argument("--output", required=True, help="Output CSV.")
    p.add_argument("--limit", type=int, default=None, help="First N rows.")
    p.add_argument("--atr-mode", choices=("keyword", "embedding"),
                   default="embedding",
                   help="ATR I_AS/I_TS implementation. Default: embedding.")
    p.add_argument("--atr-threshold", type=float, default=0.10,
                   help="Cosine threshold for --atr-mode embedding. "
                        "Default 0.10 (calibrated on v15 GT).")
    args = p.parse_args()

    rows_in = load_jsonl(args.input)
    if args.limit:
        rows_in = rows_in[: args.limit]
    gold_map = build_gold_map(args.gold) if args.gold else {}

    out_rows: list[dict] = []
    for row in tqdm(rows_in, desc="rule"):
        qid, pidx = outer_qid(row), outer_pidx(row)
        plan = unwrap(row)
        gold = gold_map.get((qid, pidx))
        try:
            metrics = eval_rule(plan, gold,
                                atr_mode=args.atr_mode,
                                atr_threshold=args.atr_threshold)
            err = ""
        except Exception as e:
            metrics = {k: float("nan") for k in RULE_COLS}
            err = f"{type(e).__name__}: {e}"
        out_rows.append({"question_id": qid, "profile_index": pidx,
                         **metrics, "_error": err})

    df = pd.DataFrame(out_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out_rows)} rows -> {args.output}")
    _summarize(df)


if __name__ == "__main__":
    main()
