#!/usr/bin/env python3
"""
Execute 30 grpo-v4 plans + evaluate with tier2 metrics.

Steps:
  1. Sample 30 fmt_ok rows from grpo_v4_eval_after.jsonl
  2. Reconstruct full plan (input from main dataset + output from grpo-v4 completion)
  3. Execute via run_single_plan.py (Teacher=gpt-4o, Student=gpt-4o-mini)
  4. Evaluate completed runs with tier2_evaluator_v2.evaluate_run_dir()
  5. Save results to tier2_eval_package/results_grpov4_30.json

Usage:
    cd stage3_execution
    python3 run_grpov4_30.py
    python3 run_grpov4_30.py --n 10 --seed 99
    python3 run_grpov4_30.py --exec-only   # skip tier2 eval
    python3 run_grpov4_30.py --eval-only   # skip execution (use existing runs)
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / "github" / ".env")
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

BASE = Path(__file__).parent
TIER2_PKG = BASE.parent / "Evaluation" / "tier2_eval_package"
EVAL_AFTER = Path.home() / "Desktop" / "grpo_v4_eval_after.jsonl"
MAIN_DATASET = BASE.parent / "multi_agent_dataset_filtered_qap.jsonl"
RUNS_DIR = TIER2_PKG / "runs_grpov4_30"
SUMMARY_PATH = TIER2_PKG / "runs_grpov4_30_summary.json"
RESULTS_PATH = TIER2_PKG / "results_grpov4_30.json"

WORKER = str(BASE / "run_single_plan.py")
PLAN_TIMEOUT = 480  # seconds per plan

sys.path.insert(0, str(TIER2_PKG))


def load_input_index():
    idx = {}
    with open(MAIN_DATASET) as f:
        for line in f:
            r = json.loads(line)
            key = (str(r["question_id"]), int(r["profile_index"]))
            idx[key] = r["plan"]["input"]
    return idx


def run_one(entry, model, student_model):
    qid = entry.get("question_id", "?")
    pidx = entry.get("profile_index", 0)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, WORKER,
             "--entry", json.dumps(entry),
             "--runs-dir", str(RUNS_DIR),
             "--model", model,
             "--student-model", student_model,
             "--out", str(out_path)],
            timeout=PLAN_TIMEOUT,
            capture_output=True,
        )
        if out_path.stat().st_size:
            return json.loads(out_path.read_text())
        stderr = proc.stderr.decode(errors="replace")[:300]
        return {"question_id": qid, "profile_index": pidx,
                "status": "worker_error", "error": stderr or f"exit {proc.returncode}"}
    except subprocess.TimeoutExpired:
        return {"question_id": qid, "profile_index": pidx,
                "status": "timeout", "error": f"exceeded {PLAN_TIMEOUT}s"}
    finally:
        out_path.unlink(missing_ok=True)


def do_execution(sample, input_idx, model, student_model):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing summary for resume
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else []
    done_keys = {(str(e["question_id"]), int(e["profile_index"])) for e in summary}

    for i, row in enumerate(sample):
        qid = str(row["question_id"])
        pidx = int(row["profile_index"])
        key = (qid, pidx)
        if key in done_keys:
            print(f"[{i+1}/{len(sample)}] SKIP (already done): qid={qid} p={pidx}")
            continue

        plan_input = input_idx[key]
        plan_output = json.loads(row["completion"])
        entry = {
            "question_id": qid,
            "profile_index": pidx,
            "plan": {"input": plan_input, "output": plan_output},
        }

        print(f"[{i+1}/{len(sample)}] Executing qid={qid} p={pidx} ...", flush=True)
        result = run_one(entry, model, student_model)
        status = result.get("status", "?")
        run_id = result.get("run_id", "")
        print(f"  -> {status} {run_id}", flush=True)
        if result.get("error"):
            print(f"     error: {result['error'][:120]}")

        summary.append(result)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    ok = sum(1 for e in summary if e.get("status") == "ok")
    print(f"\nExecution complete: {ok}/{len(summary)} ok  ({SUMMARY_PATH})")
    return summary


def do_evaluation(summary, judge_model):
    from tier2_evaluator_v2 import evaluate_run_dir, aggregate_results, load_accepted_answers

    aa = load_accepted_answers()

    # Rebuild grpo-v4 plan lookup from eval_after + input_idx
    eval_after_map = {}
    for line in open(EVAL_AFTER):
        r = json.loads(line)
        key = (str(r["question_id"]), int(r["profile_index"]))
        eval_after_map[key] = r

    input_idx = load_input_index()

    def get_grpov4_plan(qid, pidx):
        key = (str(qid), int(pidx))
        row = eval_after_map.get(key)
        if row is None or not row.get("fmt_ok"):
            return None
        plan_input = input_idx.get(key)
        if plan_input is None:
            return None
        return {"input": plan_input, "output": json.loads(row["completion"])}

    results = []
    issues = []
    ok_runs = [e for e in summary if e.get("status") == "ok"]
    print(f"\nEvaluating {len(ok_runs)} completed runs with tier2 ...")

    for i, entry in enumerate(ok_runs):
        qid = str(entry["question_id"])
        pidx = int(entry["profile_index"])
        run_id = entry["run_id"]
        run_dir = RUNS_DIR / run_id
        accepted = aa.get((qid, pidx), "")

        # Use grpo-v4 plan (not original SFT plan from dataset)
        grpov4_plan = get_grpov4_plan(qid, pidx)

        print(f"[{i+1}/{len(ok_runs)}] Evaluating {run_id} ...", flush=True)
        try:
            res = evaluate_run_dir(
                run_dir,
                plan=grpov4_plan,   # pass grpo-v4 plan explicitly
                accepted_answer=accepted,
                judge_model=judge_model,
                reexec_codesteps=False,
                verbose=False,
            )
            res.update({"run_id": run_id, "question_id": qid, "profile_index": pidx,
                        "accepted_answer_len": len(accepted)})
            results.append(res)
            evr = res["evr"]["evr_pass"]
            pas = res["pas"]["pas"]
            pqs = res["pqs"]["pqs"]
            rsol = res["r_sol"]["r_sol"]
            print(f"  -> EVR={evr} PAS={pas:.3f} PQS={pqs:.3f} r_sol={rsol}")
        except Exception as e:
            issues.append({"run_id": run_id, "issue": f"{type(e).__name__}: {e}"})
            print(f"  -> ERROR: {e}")

        # Incremental save
        RESULTS_PATH.write_text(json.dumps(
            {"n_done": len(results), "n_planned": len(ok_runs),
             "results": results, "issues": issues},
            indent=2, ensure_ascii=False))

    agg = aggregate_results(results)
    final = {
        "judge_model": judge_model,
        "n_runs": len(results),
        "aggregate": agg,
        "issues": issues,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False))

    print(f"\n=== Tier2 Results (grpo-v4, {len(results)} runs) ===")
    print(f"  EVR   = {agg['evr']:.3f}  ({agg['evr_pass_count']}/{agg['n']})")
    print(f"    sub:  " + "  ".join(
        f"{k}={agg['evr_subcheck_pass_rate'].get(k,0):.2f}"
        for k in ("cov","loop","flow","exec")))
    print(f"  PAS   = {agg['pas']:.3f}")
    print(f"  PQS   = {agg['pqs']:.3f}  "
          f"(NDAR={agg['ndar']:.2f} SPR={agg['spr']:.2f} IAR={agg['iar']:.2f})")
    print(f"  r_sol = {agg['r_sol']:.3f}  "
          f"({agg['r_sol_pass_count']}/{agg['r_sol_valid']} valid)")
    print(f"\nResults -> {RESULTS_PATH}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--student-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--exec-only", action="store_true", help="Only run execution, skip evaluation")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing runs")
    args = ap.parse_args()

    # Sample
    if not args.eval_only:
        print("Loading data ...")
        eval_after = [json.loads(l) for l in open(EVAL_AFTER)]
        input_idx = load_input_index()

        valid = [r for r in eval_after
                 if r.get("fmt_ok") and
                 (str(r["question_id"]), int(r["profile_index"])) in input_idx]
        print(f"Valid rows: {len(valid)} (fmt_ok + in dataset)")

        rng = random.Random(args.seed)
        sample = rng.sample(valid, min(args.n, len(valid)))
        print(f"Sampled {len(sample)} rows (seed={args.seed})\n")

        summary = do_execution(sample, input_idx, args.model, args.student_model)
    else:
        if not SUMMARY_PATH.exists():
            print(f"ERROR: No existing summary at {SUMMARY_PATH}")
            sys.exit(1)
        summary = json.loads(SUMMARY_PATH.read_text())
        print(f"Loaded existing summary: {len(summary)} runs")

    if args.exec_only:
        return

    do_evaluation(summary, args.judge_model)


if __name__ == "__main__":
    main()
