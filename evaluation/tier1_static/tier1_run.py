#!/usr/bin/env python
"""
Tier 1 Plan Evaluation -- single-file wrapper.

Provide one JSONL of plans; this script runs ALL Tier 1 metrics in one
pass (6 rule + Pers + Ped LLM judges, plus optional PVS / PNG aggregates)
and writes one combined CSV.

Required input file format (one row per plan, outer-wrapped):
    {"question_id": "<qid>", "profile_index": <int>, "generated_plan": {...}}

Minimum usage:
    .venvs/tier1_eval/Scripts/python.exe tier1_run.py \\
        --input  plans.jsonl \\
        --output plans_tier1.csv

Auto-detected defaults (resolved relative to this script's directory):
    --gold  multi_agent_dataset_filtered_qap_v15_goodplus.jsonl   (for GED-sim, PVS)
    --qap   filtered_qap.jsonl                                    (for NDAR)
    --judge $TIER1_JUDGE_MODEL or gpt-5-2025-08-07

Output:
    A single CSV with columns:
        question_id, profile_index,
        SV, AR, DC, ATR, TBV, GED_sim,                            (rule)
        Pers, Pers_SkillMatch, Pers_GoalOrient, Pers_BgAdapt,     (Pers sub-dim)
        Ped,  Ped_PRR, Ped_NDAR, Ped_SPR, Ped_IAR,                (Ped sub-dim)
        error                                                     (per-plan exception, "" if OK)

    Plus a per-metric mean summary printed to stdout.
    PVS / PNG aggregates printed when --pvs / --png passed.
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from tier1_rule import (
    eval_rule, load_jsonl, unwrap, outer_qid, outer_pidx,
    build_gold_map, build_qap_lookup, RULE_COLS,
)
from tier1_judge import (
    eval_judge, inject_accepted_answer, DEFAULT_JUDGE, JUDGE_COLS,
)
from tier1_counterfactual import (
    pvs, png, bucket_by_qid, build_png_pairs_from_multi_profile,
)
DEFAULT_GOLD = PROJECT_ROOT / "multi_agent_dataset_filtered_qap_v15_goodplus.jsonl"
DEFAULT_QAP = PROJECT_ROOT / "filtered_qap.jsonl"


def evaluate_one(row, gold_map, qap_lookup, rule_only,
                 judge, atr_mode, atr_threshold):
    """Run all per-plan metrics on a single outer-wrapped row.

    On exception: returns NaN for every metric and stuffs the error string
    into the `error` column so one bad plan can't kill the whole run.
    """
    qid = outer_qid(row)
    pidx = outer_pidx(row)
    try:
        plan = unwrap(row)
        gold = gold_map.get((qid, pidx)) if qid is not None else None

        rule_metrics = eval_rule(
            plan, gold,
            atr_mode=atr_mode, atr_threshold=atr_threshold,
        )

        judge_metrics = {}
        if not rule_only:
            if qap_lookup:
                inject_accepted_answer(plan, qid, pidx, qap_lookup)
            judge_metrics = eval_judge(plan, gold, judge=judge)

        return {
            "question_id": qid,
            "profile_index": pidx,
            **rule_metrics,
            **judge_metrics,
            "error": "",
        }
    except Exception as exc:
        return {
            "question_id": qid,
            "profile_index": pidx,
            **{c: float("nan") for c in RULE_COLS},
            **({c: float("nan") for c in JUDGE_COLS} if not rule_only else {}),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tier 1 plan evaluation -- all metrics in one CSV.",
    )
    p.add_argument("--input", required=True,
                   help="JSONL of plans (outer-wrapped rows)")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--gold", default=str(DEFAULT_GOLD),
                   help=f"Gold dataset for GED-sim (default {DEFAULT_GOLD.name})")
    p.add_argument("--qap", default=str(DEFAULT_QAP),
                   help=f"QAP lookup for NDAR (default {DEFAULT_QAP.name})")
    p.add_argument("--rule-only", action="store_true",
                   help="Skip LLM judges (no API key needed)")
    p.add_argument("--judge", default=DEFAULT_JUDGE,
                   help=f"LLM judge model (default {DEFAULT_JUDGE})")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel LLM workers (default 4)")
    p.add_argument("--pvs", action="store_true",
                   help="Also compute PVS aggregate (rule, multi-profile data)")
    p.add_argument("--png", action="store_true",
                   help="Also compute PNG aggregate (LLM, multi-profile data)")
    p.add_argument("--atr-mode", choices=("embedding", "keyword"),
                   default="embedding",
                   help="ATR computation mode (default embedding)")
    p.add_argument("--atr-threshold", type=float, default=0.10,
                   help="ATR embedding cosine threshold (default 0.10)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate first N plans only")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[tier1_run] loaded {len(rows)} plans from {args.input}")

    gold_map = {}
    if Path(args.gold).exists():
        gold_map = build_gold_map(args.gold)
        print(f"[tier1_run] gold_map: {len(gold_map)} entries from {args.gold}")
    else:
        print(f"[tier1_run] WARN: gold file {args.gold} not found "
              f"-> GED-sim / PVS will be NaN")

    qap_lookup = {}
    if not args.rule_only:
        if Path(args.qap).exists():
            qap_lookup = build_qap_lookup(args.qap)
            print(f"[tier1_run] qap_lookup: {len(qap_lookup)} entries from {args.qap}")
        else:
            print(f"[tier1_run] WARN: qap file {args.qap} not found "
                  f"-> Ped.NDAR will be NaN")
        print(f"[tier1_run] judge: {args.judge} | workers: {args.workers}")
    else:
        print(f"[tier1_run] rule-only mode (LLM judges skipped)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list = []
    if args.rule_only:
        for row in tqdm(rows, desc="rule"):
            results.append(
                evaluate_one(row, gold_map, qap_lookup,
                             True, args.judge,
                             args.atr_mode, args.atr_threshold))
    else:
        indexed_rows = list(enumerate(rows))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_idx = {
                ex.submit(evaluate_one, row, gold_map, qap_lookup,
                          False, args.judge,
                          args.atr_mode, args.atr_threshold): i
                for i, row in indexed_rows
            }
            with tqdm(total=len(rows), desc="rule+judge") as pbar:
                for fut in as_completed(future_to_idx):
                    results.append((future_to_idx[fut], fut.result()))
                    pbar.update(1)
        results.sort(key=lambda t: t[0])
        results = [r for _, r in results]

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"[tier1_run] wrote {len(df)} rows x {len(df.columns)} cols -> {out_path}")

    print(f"\n=== Per-plan metric summary (mean over {len(df)} plans) ===")
    metric_cols = RULE_COLS + (JUDGE_COLS if not args.rule_only else [])
    for col in metric_cols:
        if col not in df.columns:
            continue
        non_nan = df[col].dropna()
        if len(non_nan) > 0:
            print(f"  {col:20s}  mean={non_nan.mean():.4f}  n={len(non_nan)}")
        else:
            print(f"  {col:20s}  (all NaN)")

    if "error" in df.columns:
        err_count = (df["error"].fillna("") != "").sum()
        if err_count > 0:
            print(f"\n  WARN: {err_count} plans had errors (see 'error' column)")

    if args.pvs:
        plans_by_qid = bucket_by_qid(rows)
        pvs_mean, per_q = pvs(plans_by_qid)
        n_qids = len([v for v in per_q.values() if not math.isnan(v)])
        print(f"\n=== PVS aggregate ===")
        print(f"  PVS = {pvs_mean:.4f}  ({n_qids} qids with >=2 profiles)")

    if args.png:
        if args.rule_only:
            print(f"\n[PNG] skipped: requires LLM judge (remove --rule-only)")
        else:
            plans_by_qid = bucket_by_qid(rows)
            pairs = build_png_pairs_from_multi_profile(plans_by_qid)
            if not pairs:
                print(f"\n[PNG] skipped: no multi-profile pairs in input "
                      f"(need >=2 profiles per qid)")
            else:
                print(f"\n[PNG] computing on {len(pairs)} pairs with "
                      f"{args.judge}...")
                res = png(pairs, judge=args.judge)
                print(f"=== PNG aggregate ===")
                print(f"  PNG = {res['PNG']:.4f}  "
                      f"paired-t p={res['PNG_p']:.4f}  n={res['PNG_n']}")


if __name__ == "__main__":
    main()
