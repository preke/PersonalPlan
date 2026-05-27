"""Audit MAPLE SFT data for design problems that algorithmic SFT loss
cannot catch.

Run this BEFORE relying on SFT-trained adapters. SFT directly fits gold
tokens, so the failure mode shifts from "is my reward measuring real
quality?" (the RL question) to "is my gold data itself representative
of real quality?" (the SFT question).

This audit checks 6 dimensions:

  A. Mechanical gold validity            (schema, cycle, tool calls)
  B. PAD target self-consistency         (unique agents, valid ids)
  C. SDP target self-consistency         (step.agent ∈ scaffold,
                                          execution_order refs real ids,
                                          loop blocks well-formed)
  D. PAD ↔ SDP coupling                  (does scaffold suffice to
                                          predict steps?)
  E. Profile sensitivity                 (does gold actually change
                                          when profile changes for the
                                          same query?  THE key check
                                          for LLM-generated gold)
  F. Data split leakage                  (same question_id in train+valid)
  G. Gold diversity                      (mode collapse risk)
  H. Joint-alignment-specific risks      (Stage 3 conceptual)

Outputs:
  - prints summary to stdout
  - writes audit_results.json with all metrics for the HTML report
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Import structural_similarity and plan_components from the GRPO plan_utils
sys.path.insert(0, str(HERE.parent / "GRPO" / "Dynamic_weighting"))
from plan_utils import (                                                # noqa: E402
    ALLOWED_TOOLS,
    has_cycle,
    invalid_tool_calls,
    plan_components,
    schema_valid,
    structural_similarity,
)


def get_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_pad_target(plan_output):
    """Same as build_sft_data.pad_target — high-level scaffold only."""
    return {
        "agents": [
            {
                "agent_role": a.get("agent_role", ""),
                "goal": a.get("goal", ""),
                "backstory": a.get("backstory") or a.get("description") or "",
                "tools": list(a.get("tools") or []),
            }
            for a in plan_output["agents"]
        ],
        "subtasks": [
            {
                "id": s["id"],
                "name": s.get("name", ""),
                "subtask_objective": s.get("subtask_objective", ""),
            }
            for s in plan_output["subtasks"]
        ],
    }


# ----------------------------------------------------------------------
# Audit functions
# ----------------------------------------------------------------------

def audit_mechanical(records: list[dict]) -> dict:
    """A. Mechanical gold validity."""
    n = len(records)
    schema = [schema_valid(r["plan"]["output"]) for r in records]
    cyc = [has_cycle(r["plan"]["output"]) for r in records]
    invalid_tool = [invalid_tool_calls(r["plan"]["output"]) for r in records]
    return {
        "n": n,
        "schema_valid_pct": sum(schema) / n,
        "no_cycle_pct": 1 - sum(cyc) / n,
        "no_invalid_tool_pct": sum(1 for x in invalid_tool if x == 0) / n,
        "any_invalid_tool_count": sum(1 for x in invalid_tool if x > 0),
    }


def audit_pad_target(records: list[dict]) -> dict:
    """B. PAD target self-consistency."""
    n_agents = []
    n_subtasks = []
    duplicate_agent_role = 0
    duplicate_subtask_id = 0
    empty_role = 0
    empty_subtask_name = 0
    for r in records:
        out = r["plan"]["output"]
        agents = out.get("agents", [])
        subs = out.get("subtasks", [])
        n_agents.append(len(agents))
        n_subtasks.append(len(subs))
        roles = [a.get("agent_role", "") for a in agents]
        if len(set(roles)) < len(roles):
            duplicate_agent_role += 1
        if any(not r for r in roles):
            empty_role += 1
        ids = [s.get("id", "") for s in subs]
        if len(set(ids)) < len(ids):
            duplicate_subtask_id += 1
        if any(not s.get("name") for s in subs):
            empty_subtask_name += 1
    return {
        "n_agents": {
            "mean": statistics.mean(n_agents),
            "median": statistics.median(n_agents),
            "min": min(n_agents), "max": max(n_agents),
        },
        "n_subtasks": {
            "mean": statistics.mean(n_subtasks),
            "median": statistics.median(n_subtasks),
            "min": min(n_subtasks), "max": max(n_subtasks),
        },
        "duplicate_agent_role_pct": duplicate_agent_role / len(records),
        "duplicate_subtask_id_pct": duplicate_subtask_id / len(records),
        "empty_agent_role_pct": empty_role / len(records),
        "empty_subtask_name_pct": empty_subtask_name / len(records),
    }


def audit_sdp_target(records: list[dict]) -> dict:
    """C. SDP target self-consistency."""
    n = len(records)
    step_agent_in_scaffold_violations = 0
    step_tool_violations = 0
    exec_order_violations = 0
    bad_loop_blocks = 0
    step_id_format_violations = 0
    total_steps = []

    for r in records:
        out = r["plan"]["output"]
        p = plan_components(out)
        scaffold_roles = {a.get("agent_role", "") for a in out["agents"]}
        agent_tools = {a.get("agent_role", ""): set(a.get("tools") or []) for a in out["agents"]}

        step_count = 0
        for sub in out["subtasks"]:
            sid_pat = sub["id"]
            for st in sub.get("steps", []):
                step_count += 1
                # step.agent must be in scaffold
                if st.get("agent", "") not in scaffold_roles:
                    step_agent_in_scaffold_violations += 1
                # step.tool, if set, must be in agent's tools AND in ALLOWED_TOOLS
                t = st.get("tool")
                if t is not None:
                    if t not in ALLOWED_TOOLS or t not in agent_tools.get(st.get("agent", ""), set()):
                        step_tool_violations += 1
                # step ID format: <subtask_id>-<n>
                if not str(st.get("id", "")).startswith(sid_pat + "-"):
                    step_id_format_violations += 1
        total_steps.append(step_count)

        # execution_order check
        all_step_ids = set(p["steps_by_id"].keys())
        for entry in out.get("execution_order", []):
            if isinstance(entry, str):
                if entry not in all_step_ids:
                    exec_order_violations += 1
            elif isinstance(entry, dict) and "loop" in entry:
                loop = entry["loop"]
                step_or_steps = loop.get("steps") or ([loop["step"]] if "step" in loop else [])
                cond = loop.get("condition", "")
                mx = loop.get("max_iterations")
                if not step_or_steps or not cond or not isinstance(mx, int):
                    bad_loop_blocks += 1
                for s in step_or_steps:
                    if s not in all_step_ids:
                        exec_order_violations += 1
    return {
        "step_agent_in_scaffold_violation_pct": step_agent_in_scaffold_violations / max(sum(total_steps), 1),
        "step_tool_violation_pct": step_tool_violations / max(sum(total_steps), 1),
        "step_id_format_violation_pct": step_id_format_violations / max(sum(total_steps), 1),
        "exec_order_violation_pct": exec_order_violations / n,
        "bad_loop_blocks": bad_loop_blocks,
        "total_steps": {
            "mean": statistics.mean(total_steps),
            "median": statistics.median(total_steps),
            "min": min(total_steps), "max": max(total_steps),
        },
    }


def audit_pad_sdp_coupling(records: list[dict]) -> dict:
    """D. Does PAD scaffold suffice to predict SDP?
    Approximation: check whether step.agent / step.tool fields are
    fully determined by the scaffold (they should be, by construction).
    Also: check whether subtask_objective text is reused in step instructions
    (information leakage that helps SDP learn the task)."""
    objective_word_overlap = []
    for r in records:
        out = r["plan"]["output"]
        for sub in out["subtasks"]:
            obj = (sub.get("subtask_objective", "") or "").lower().split()
            obj_set = set(obj)
            if not obj_set:
                continue
            for st in sub.get("steps", []):
                instr = (st.get("instruction", "") or "").lower().split()
                instr_set = set(instr)
                if not instr_set:
                    continue
                jacc = len(obj_set & instr_set) / len(obj_set | instr_set)
                objective_word_overlap.append(jacc)
    return {
        "n_step_objective_pairs": len(objective_word_overlap),
        "objective_word_overlap_mean": statistics.mean(objective_word_overlap) if objective_word_overlap else 0,
        "objective_word_overlap_median": statistics.median(objective_word_overlap) if objective_word_overlap else 0,
    }


def audit_profile_sensitivity(records: list[dict], n_pairs_sample: int = 2000) -> dict:
    """E. THE KEY CHECK: does gold actually change when profile changes
    for the same query?
    For each question_id with ≥2 profiles, compute pairwise StructSim
    of full plans. Then compare to the cross-question baseline (random
    pairs of plans from different questions).
    """
    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["question_id"]].append(r)

    same_q_sims = []  # different profiles, same query
    same_q_groups = 0
    for qid, group in by_qid.items():
        if len(group) < 2:
            continue
        same_q_groups += 1
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = structural_similarity(
                    group[i]["plan"]["output"], group[j]["plan"]["output"]
                )
                same_q_sims.append(sim)

    # Cross-question baseline
    rng = random.Random(42)
    all_recs = records
    cross_sims = []
    while len(cross_sims) < min(n_pairs_sample, len(all_recs) * (len(all_recs) - 1) // 2):
        a, b = rng.sample(all_recs, 2)
        if a["question_id"] == b["question_id"]:
            continue
        cross_sims.append(structural_similarity(
            a["plan"]["output"], b["plan"]["output"]
        ))

    return {
        "n_questions_total": len(by_qid),
        "n_questions_with_multi_profile": same_q_groups,
        "n_same_q_pairs": len(same_q_sims),
        "same_q_sim_mean": statistics.mean(same_q_sims) if same_q_sims else None,
        "same_q_sim_median": statistics.median(same_q_sims) if same_q_sims else None,
        "n_cross_q_pairs": len(cross_sims),
        "cross_q_sim_mean": statistics.mean(cross_sims),
        "cross_q_sim_median": statistics.median(cross_sims),
        "profile_sensitivity_gap": (
            statistics.mean(same_q_sims) - statistics.mean(cross_sims)
            if same_q_sims else None
        ),
    }


def audit_data_split_leakage(records: list[dict], val_frac: float = 0.10, seed: int = 42) -> dict:
    """F. Replicate build_sft_data's split and check for question_id leakage."""
    shuf = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuf)
    n_val = max(1, int(round(len(shuf) * val_frac)))
    valid = shuf[:n_val]
    train = shuf[n_val:]
    train_qids = set(r["question_id"] for r in train)
    valid_qids = set(r["question_id"] for r in valid)
    leak_qids = train_qids & valid_qids
    return {
        "seed": seed,
        "val_frac": val_frac,
        "n_train_records": len(train),
        "n_valid_records": len(valid),
        "n_train_unique_qids": len(train_qids),
        "n_valid_unique_qids": len(valid_qids),
        "n_leaking_qids": len(leak_qids),
        "leaking_qids_pct_of_valid": len(leak_qids) / max(len(valid_qids), 1),
    }


def audit_diversity(records: list[dict], top_k: int = 10) -> dict:
    """G. Gold diversity — mode collapse risk."""
    all_agent_roles = Counter()
    all_subtask_names = Counter()
    for r in records:
        out = r["plan"]["output"]
        for a in out.get("agents", []):
            all_agent_roles[a.get("agent_role", "")] += 1
        for s in out.get("subtasks", []):
            all_subtask_names[s.get("name", "")] += 1
    return {
        "n_unique_agent_roles": len(all_agent_roles),
        "n_unique_subtask_names": len(all_subtask_names),
        "total_agent_instances": sum(all_agent_roles.values()),
        "total_subtask_instances": sum(all_subtask_names.values()),
        "top_agent_roles": all_agent_roles.most_common(top_k),
        "top_subtask_names": all_subtask_names.most_common(top_k),
        "agent_role_concentration_top10_pct": (
            sum(v for _, v in all_agent_roles.most_common(10)) / sum(all_agent_roles.values())
        ),
        "subtask_name_concentration_top10_pct": (
            sum(v for _, v in all_subtask_names.most_common(10)) / sum(all_subtask_names.values())
        ),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="../multi_agent_dataset_filtered_qap.jsonl",
        help="Full MAPLE dataset (used for SFT in practice)",
    )
    ap.add_argument("--out-json", default="audit_results.json")
    args = ap.parse_args()

    src_path = (HERE / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    print(f"Loading {src_path} ...")
    records = get_records(src_path)
    print(f"Auditing {len(records)} SFT gold records\n")

    print("=" * 60)
    print("A. MECHANICAL GOLD VALIDITY")
    print("=" * 60)
    a = audit_mechanical(records)
    print(f"  schema_valid       : {a['schema_valid_pct']*100:.2f}%")
    print(f"  no_cycle           : {a['no_cycle_pct']*100:.2f}%")
    print(f"  no_invalid_tool    : {a['no_invalid_tool_pct']*100:.2f}%")
    print(f"  records with bad tool: {a['any_invalid_tool_count']}")

    print("\n" + "=" * 60)
    print("B. PAD TARGET (agents + subtasks_metadata)")
    print("=" * 60)
    b = audit_pad_target(records)
    print(f"  n_agents   : mean={b['n_agents']['mean']:.2f}, "
          f"median={b['n_agents']['median']}, range=[{b['n_agents']['min']},{b['n_agents']['max']}]")
    print(f"  n_subtasks : mean={b['n_subtasks']['mean']:.2f}, "
          f"median={b['n_subtasks']['median']}, range=[{b['n_subtasks']['min']},{b['n_subtasks']['max']}]")
    print(f"  duplicate_agent_role : {b['duplicate_agent_role_pct']*100:.2f}%")
    print(f"  duplicate_subtask_id : {b['duplicate_subtask_id_pct']*100:.2f}%")
    print(f"  empty_agent_role     : {b['empty_agent_role_pct']*100:.2f}%")
    print(f"  empty_subtask_name   : {b['empty_subtask_name_pct']*100:.2f}%")

    print("\n" + "=" * 60)
    print("C. SDP TARGET (steps + execution_order)")
    print("=" * 60)
    c = audit_sdp_target(records)
    print(f"  total_steps : mean={c['total_steps']['mean']:.2f}, "
          f"median={c['total_steps']['median']}, range=[{c['total_steps']['min']},{c['total_steps']['max']}]")
    print(f"  step.agent NOT in scaffold : {c['step_agent_in_scaffold_violation_pct']*100:.4f}% of steps")
    print(f"  step.tool violation       : {c['step_tool_violation_pct']*100:.4f}% of steps")
    print(f"  step.id format violation  : {c['step_id_format_violation_pct']*100:.4f}% of steps")
    print(f"  execution_order refs broken : {c['exec_order_violation_pct']*100:.4f}% per plan")
    print(f"  malformed loop blocks       : {c['bad_loop_blocks']}")

    print("\n" + "=" * 60)
    print("D. PAD ↔ SDP COUPLING")
    print("=" * 60)
    d = audit_pad_sdp_coupling(records)
    print(f"  step.instruction vs subtask_objective word-Jaccard:")
    print(f"    mean   = {d['objective_word_overlap_mean']:.3f}")
    print(f"    median = {d['objective_word_overlap_median']:.3f}")
    print(f"  → 高 = SDP target 可从 PAD 一定程度推导，SDP 学习更容易但模型对 PAD 依赖也更强")

    print("\n" + "=" * 60)
    print("E. PROFILE SENSITIVITY (key check for LLM-generated gold)")
    print("=" * 60)
    e = audit_profile_sensitivity(records)
    print(f"  n_questions total                 : {e['n_questions_total']}")
    print(f"  n_questions with ≥2 profiles      : {e['n_questions_with_multi_profile']}")
    print(f"  same-query pairs (n={e['n_same_q_pairs']}):")
    if e['same_q_sim_mean'] is not None:
        print(f"    StructSim mean   = {e['same_q_sim_mean']:.3f}")
        print(f"    StructSim median = {e['same_q_sim_median']:.3f}")
    else:
        print(f"    NO same-query pairs exist (each question has only 1 profile)")
    print(f"  cross-query baseline (n={e['n_cross_q_pairs']}):")
    print(f"    StructSim mean   = {e['cross_q_sim_mean']:.3f}")
    print(f"    StructSim median = {e['cross_q_sim_median']:.3f}")
    if e['profile_sensitivity_gap'] is not None:
        gap = e['profile_sensitivity_gap']
        print(f"  Gap (same - cross): {gap:+.3f}")
        if gap > 0.3:
            print(f"  → 同 query 不同 profile 的 plan 高度相似，profile 几乎不改变 gold ⚠️")
        elif gap > 0.1:
            print(f"  → 同 query 不同 profile 略有差异（轻度个性化）")
        else:
            print(f"  → profile 显著改变 gold（真正的个性化）")
    else:
        print(f"  → 无法测：每个 question_id 只有 1 个 profile，profile sensitivity 不可验证")

    print("\n" + "=" * 60)
    print("F. DATA SPLIT LEAKAGE (build_sft_data.py default seed=42, val_frac=0.10)")
    print("=" * 60)
    f = audit_data_split_leakage(records)
    print(f"  train records: {f['n_train_records']}  ({f['n_train_unique_qids']} unique qids)")
    print(f"  valid records: {f['n_valid_records']}  ({f['n_valid_unique_qids']} unique qids)")
    print(f"  leaking qids : {f['n_leaking_qids']}  "
          f"({f['leaking_qids_pct_of_valid']*100:.1f}% of valid qids appear in train too)")
    if f['n_leaking_qids'] > 0:
        print(f"  → WARNING 数据泄漏：valid 集里有 query 在 train 见过（不同 profile），"
              f"SFT 实际上偷看到了答案的题干一半")

    print("\n" + "=" * 60)
    print("G. GOLD DIVERSITY (mode collapse risk)")
    print("=" * 60)
    g = audit_diversity(records)
    print(f"  unique agent_roles globally : {g['n_unique_agent_roles']}  "
          f"({g['total_agent_instances']} total instances; "
          f"top-10 cover {g['agent_role_concentration_top10_pct']*100:.1f}%)")
    print(f"  unique subtask_names globally: {g['n_unique_subtask_names']}  "
          f"({g['total_subtask_instances']} total; "
          f"top-10 cover {g['subtask_name_concentration_top10_pct']*100:.1f}%)")
    print(f"\n  Top-10 agent_roles:")
    for role, cnt in g['top_agent_roles'][:10]:
        print(f"    {cnt:>4}× {role!r}")
    print(f"\n  Top-10 subtask_names:")
    for name, cnt in g['top_subtask_names'][:10]:
        print(f"    {cnt:>4}× {name!r}")

    # Save full results
    results = {"A_mechanical": a, "B_pad_target": b, "C_sdp_target": c,
               "D_pad_sdp_coupling": d, "E_profile_sensitivity": e,
               "F_split_leakage": f, "G_diversity": g}
    out_path = (HERE / args.out_json) if not Path(args.out_json).is_absolute() else Path(args.out_json)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
