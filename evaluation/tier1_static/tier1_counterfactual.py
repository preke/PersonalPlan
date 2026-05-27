#!/usr/bin/env python
"""
Tier 1 §11.3 — counterfactual probes.

Implements:
    PVS — Profile-Variance Score (rule, GED-sim based).
          For each qid with >=2 profiles, PVS(q) = 1 - mean_pairwise GED-sim.
          PVS = mean over qids.

    PNG — Counterfactual Personalization Gap (LLM judge).
          PNG = mean_i [ Pers(gen(q_i, I_p^i), I_p^i)
                       - Pers(gen(q_i, ~I_p^i), I_p^i) ].
          Also reports the paired t-test p-value per doc §11.3.

Standalone CLI:
    # PVS on the v15 Good+ gold dataset (200 rows, fast)
    .venvs/tier1_eval/Scripts/python.exe tier1_counterfactual.py \\
        --input multi_agent_dataset_filtered_qap_v15_goodplus.jsonl \\
        --pvs --limit 200

    # PNG from a JSONL of triplets (real_plan, shuffled_plan, ref_plan)
    .venvs/tier1_eval/Scripts/python.exe tier1_counterfactual.py \\
        --png-pairs png_triplets.jsonl --judge gemini-3.1-pro-preview

    # Offline structural check (no API, no real data needed)
    .venvs/tier1_eval/Scripts/python.exe tier1_counterfactual.py --probe
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

from tier1_rule import (
    ged_sim,
    get_input,
    load_jsonl,
    outer_pidx,
    outer_qid,
    unwrap,
)


# ============================================================================
# PVS — Profile-Variance Score
# ============================================================================

def pvs(plans_by_qid: dict[Any, list[dict]]) -> tuple[float, dict[Any, float]]:
    """Return (PVS aggregate, per-qid PVS dict).

    plans_by_qid: {qid: [plan_for_profile_1, plan_for_profile_2, ...]}.
    Each plan should be the inner {input, output} payload.
    Only qids with >=2 plans contribute to the aggregate.
    """
    per_q: dict[Any, float] = {}
    for qid, plans in plans_by_qid.items():
        if len(plans) < 2:
            continue
        sims: list[float] = []
        for i, j in combinations(range(len(plans)), 2):
            v = ged_sim(plans[i], plans[j])
            if not math.isnan(v):
                sims.append(v)
        if sims:
            per_q[qid] = 1.0 - sum(sims) / len(sims)
    agg = (sum(per_q.values()) / len(per_q)) if per_q else float("nan")
    return agg, per_q


# ============================================================================
# PNG — Counterfactual Personalization Gap (+ paired t-test)
# ============================================================================

def _paired_t_pvalue(diffs: list[float]) -> tuple[float, float]:
    """Two-sided paired t-test against H0: mean(diffs) = 0.
    Returns (t_stat, p_value).  Uses scipy.stats.ttest_1samp.
    """
    from scipy import stats  # local import — scipy may not be needed otherwise
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    res = stats.ttest_1samp(diffs, popmean=0.0)
    return (float(res.statistic), float(res.pvalue))


def png(real_pairs: list[tuple[dict, dict]],
        judge: Optional[str] = None) -> dict[str, float]:
    """PNG over a list of (real_plan, shuffled_plan) pairs.

    For each pair:
      - real_plan: plan generated for query Q with the REAL learner profile
        (the profile lives inside real_plan.input.learner already).
      - shuffled_plan: plan generated for the SAME query Q but with a
        DIFFERENT learner profile (i.e., not designed for the learner
        we are about to judge against).
    Both plans are then scored with Pers., but using the real_plan's
    profile for BOTH evaluations.  PNG = mean( Pers(real) - Pers(shuf) ).

    Both plans being for the same query Q is essential; only the profile
    they were designed for differs.  This is the doc §11.3 PNG definition
    simplified to a pair (the original tri-tuple was over-engineered —
    ref_plan just duplicated real_plan's profile).

    Data source: multi-profile plans (e.g., v15 Good+ has many qids with
    ≥2 profile variants). For qid Q with plans P_A, P_B, P_C:
      - (P_A, P_B) is a valid PNG pair (real = P_A, shuf = P_B)
      - (P_A, P_C) is a valid PNG pair (real = P_A, shuf = P_C)
    No separate shuffled-profile generation pipeline is needed.

    Imported lazily so this module is usable without an API key for
    PVS-only runs.
    """
    from tier1_judge import DEFAULT_JUDGE, pers  # lazy
    j = judge or DEFAULT_JUDGE
    diffs: list[float] = []
    for real_plan, shuf_plan in real_pairs:
        real_profile = get_input(real_plan).get("learner", {})
        # deep-copy so we don't mutate caller's dicts; overwrite the
        # shuffled plan's profile so Pers. judges both against real_profile.
        rp = json.loads(json.dumps(real_plan))
        sp = json.loads(json.dumps(shuf_plan))
        rp.setdefault("input", {})["learner"] = real_profile
        sp.setdefault("input", {})["learner"] = real_profile
        diffs.append(pers(rp, j)["Pers"] - pers(sp, j)["Pers"])
    if not diffs:
        return {"PNG": float("nan"), "PNG_t": float("nan"),
                "PNG_p": float("nan"), "PNG_n": 0}
    mean = sum(diffs) / len(diffs)
    t, p = _paired_t_pvalue(diffs)
    return {"PNG": mean, "PNG_t": t, "PNG_p": p, "PNG_n": len(diffs)}


def build_png_pairs_from_multi_profile(
        plans_by_qid: dict[Any, list[dict]],
        max_pairs_per_qid: int = 1,
) -> list[tuple[dict, dict]]:
    """Build PNG pairs directly from multi-profile data without a separate
    generation pipeline.  For each qid with k_q >= 2 profile variants:
      - take the first profile's plan as `real`
      - pair it with each other profile's plan as `shuf`
      - cap at max_pairs_per_qid pairs per qid to keep N manageable

    Returns flat list of (real_plan, shuf_plan) pairs.
    """
    pairs: list[tuple[dict, dict]] = []
    for qid, plans in plans_by_qid.items():
        if len(plans) < 2:
            continue
        real = plans[0]
        for shuf in plans[1:1 + max_pairs_per_qid]:
            pairs.append((real, shuf))
    return pairs


# ============================================================================
# I/O helpers
# ============================================================================

def bucket_by_qid(rows: list[dict]) -> dict[Any, list[dict]]:
    """Group inner plans by qid (with all profile variants)."""
    out: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        qid = outer_qid(row)
        if qid is not None:
            out[qid].append(unwrap(row))
    return out


def load_png_pairs(path: str | Path) -> list[tuple[dict, dict]]:
    """Load a JSONL where each row has keys real_plan / shuffled_plan.
    Each value is either the inner {input, output} payload or a wrapper that
    unwrap() will strip."""
    pairs: list[tuple[dict, dict]] = []
    for row in load_jsonl(path):
        rp = unwrap(row["real_plan"])
        sp = unwrap(row["shuffled_plan"])
        pairs.append((rp, sp))
    return pairs


# ============================================================================
# --probe mode (offline structural check)
# ============================================================================

def _probe() -> int:
    print("=== tier1_counterfactual --probe "
          "(offline structural check) ===\n")
    failures: list[str] = []

    # 1) PVS: 3 plans, 2 of them identical -> mean GED-sim ~1 for that pair,
    #    pairs with 3rd plan should be < 1 -> overall PVS(q) in (0, 1).
    from tier1_rule import build_step_graph  # for sanity print
    p_same = {
        "input": {"query": "q1", "learner": {}},
        "output": {
            "agents": [{"agent_role": "X", "goal": "", "backstory": "",
                        "tools": []}],
            "subtasks": [{"id": "S1", "name": "x", "subtask_objective": "",
                          "steps": [
                              {"id": "S1-1", "agent": "X", "objective": "",
                               "instruction": "", "tool": None,
                               "requires_human_input": False,
                               "expected_output": "", "depends_on": []}]}],
            "execution_order": ["S1-1"]}}
    p_diff = {
        "input": {"query": "q1", "learner": {}},
        "output": {
            "agents": [{"agent_role": "X", "goal": "", "backstory": "",
                        "tools": []}],
            "subtasks": [{"id": "S1", "name": "x", "subtask_objective": "",
                          "steps": [
                              {"id": "S1-1", "agent": "X", "objective": "",
                               "instruction": "", "tool": None,
                               "requires_human_input": False,
                               "expected_output": "", "depends_on": []},
                              {"id": "S1-2", "agent": "X", "objective": "",
                               "instruction": "", "tool": None,
                               "requires_human_input": False,
                               "expected_output": "", "depends_on": ["S1-1"]}
                          ]}],
            "execution_order": ["S1-1", "S1-2"]}}
    bucket = {"qA": [p_same, json.loads(json.dumps(p_same)), p_diff]}
    pvs_agg, per_q = pvs(bucket)
    if not (0.0 < pvs_agg < 1.0):
        failures.append(f"PVS expected in (0, 1), got {pvs_agg:.4f}")
    if "qA" not in per_q:
        failures.append("PVS did not record per-qid result")
    print(f"  [1/3] PVS(synthetic qA, 2 identical + 1 different) = "
          f"{pvs_agg:.4f}  (expected in (0, 1))")

    # 2) PVS skips qids with <2 plans
    pvs_skip, per_q_skip = pvs({"single": [p_same]})
    if not math.isnan(pvs_skip) or per_q_skip:
        failures.append("PVS did not skip single-profile qid")
    print(f"  [2/3] PVS({{single-profile qid}}) = NaN  "
          f"(got {pvs_skip}, per_q empty: {not per_q_skip})")

    # 3) paired t-test plumbing
    t, p = _paired_t_pvalue([0.4, 0.3, 0.5, 0.2, 0.45, 0.35])
    if not (t > 0 and 0 < p < 1):
        failures.append(f"paired t-test plumbing broken: t={t}, p={p}")
    print(f"  [3/3] paired t-test on [+0.4, +0.3, +0.5, +0.2, +0.45, +0.35]: "
          f"t={t:.3f}, p={p:.4e}  (expect t>0, 0<p<1)")

    if failures:
        print("\n--- FAILURES ---")
        for f in failures:
            print("  ", f)
        return 1
    print("\nAll offline checks passed.")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Tier 1 §11.3: PVS + PNG counterfactual probes.")
    p.add_argument("--probe", action="store_true",
                   help="Offline structural check (no API, no real data).")
    p.add_argument("--input", default=None,
                   help="Plans JSONL — multiple profiles per qid for PVS.")
    p.add_argument("--pvs", action="store_true",
                   help="Compute PVS over qids with >=2 profile_index in --input.")
    p.add_argument("--png-pairs", default=None,
                   help="JSONL of {real_plan, shuffled_plan, ref_plan} for PNG.")
    p.add_argument("--judge", default=None,
                   help="LLM judge model id for PNG (default: env TIER1_JUDGE_MODEL or gemini-3.1-pro-preview).")
    p.add_argument("--limit", type=int, default=None,
                   help="Use first N rows of --input.")
    p.add_argument("--output", default=None,
                   help="Optional per-qid PVS CSV.")
    args = p.parse_args()

    if args.probe:
        sys.exit(_probe())

    if not (args.pvs or args.png_pairs):
        p.error("Need --probe, --pvs, or --png-pairs.")

    # ---- PVS ---------------------------------------------------------------
    if args.pvs:
        if not args.input:
            p.error("--pvs requires --input.")
        rows = load_jsonl(args.input)
        if args.limit:
            rows = rows[: args.limit]
        bucket = bucket_by_qid(rows)
        multi = {k: v for k, v in bucket.items() if len(v) >= 2}
        print(f"PVS over {len(multi)} multi-profile qids "
              f"(out of {len(bucket)} total qids in {len(rows)} rows)...")
        pvs_agg, per_q = pvs(multi)
        print(f"\nPVS (aggregate) = {pvs_agg:.4f}")
        if per_q:
            vals = list(per_q.values())
            print(f"  per-qid: min={min(vals):.4f}  "
                  f"median={statistics.median(vals):.4f}  "
                  f"max={max(vals):.4f}  n={len(vals)}")
        if args.output and per_q:
            import pandas as pd
            df = pd.DataFrame(
                [{"question_id": k, "PVS_q": v, "n_profiles": len(multi[k])}
                 for k, v in per_q.items()])
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"Per-qid PVS -> {args.output}")

    # ---- PNG ---------------------------------------------------------------
    if args.png_pairs:
        pairs = load_png_pairs(args.png_pairs)
        print(f"\nPNG over {len(pairs)} (real_plan, shuf_plan) pairs via "
              f"judge {args.judge or '(default)'}...")
        result = png(pairs, judge=args.judge)
        print(f"  PNG (mean Pers gap) = {result['PNG']:+.4f}")
        print(f"  paired t (N={result['PNG_n']}): "
              f"t={result['PNG_t']:.3f}  p={result['PNG_p']:.4e}")


if __name__ == "__main__":
    main()
