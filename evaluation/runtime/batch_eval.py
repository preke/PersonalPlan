#!/usr/bin/env python3
"""
Batch execution: randomly sample 100 plans from the dataset and run in live mode.
Uses bianxie.ai (OpenAI-compatible) API via OPENAI_BASE_URL + OPENAI_API_KEY.

Per project spec (完整流程说明.pdf §6.1):
  - Teaching agents use a strong model (e.g. gpt-4o)
  - Student agent uses a medium/weak model (e.g. gpt-4o-mini)

Usage:
    python3 batch_eval.py
    python3 batch_eval.py --n 10 --model openai/gpt-4o --student-model openai/gpt-4o-mini
    python3 batch_eval.py --seed 42 --runs-dir runs_batch_v2
    python3 batch_eval.py --runs-dir runs_batch_100   # resume from existing summary
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / "github" / ".env")

socket.setdefaulttimeout(30)

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

DATASET_PATH = (
    Path(__file__).parent
    / "multi_agent_datasets"
    / "multi_agent_dataset_filtered_qap_latest.jsonl"
)

PLAN_TIMEOUT = 480  # 8 minutes per plan

WORKER_SCRIPT = str(Path(__file__).parent / "run_single_plan.py")


def load_samples(n: int, seed: int) -> list:
    with open(DATASET_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(seed)
    sampled = rng.sample(lines, min(n, len(lines)))
    return [json.loads(line) for line in sampled]


def run_one_with_timeout(entry: dict, runs_dir: Path, model: str,
                         student_model: str, timeout: int) -> dict:
    question_id = entry.get("question_id", "unknown")
    profile_idx = entry.get("profile_index", 0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, WORKER_SCRIPT,
             "--entry", json.dumps(entry),
             "--runs-dir", str(runs_dir),
             "--model", model,
             "--student-model", student_model,
             "--out", str(out_path)],
            timeout=timeout,
            capture_output=True,
        )
        if out_path.stat().st_size:
            return json.loads(out_path.read_text(encoding="utf-8"))
        stderr = proc.stderr.decode(errors="replace")[:200]
        return {"question_id": question_id, "profile_index": profile_idx,
                "status": "worker_error", "error": stderr or f"exit code {proc.returncode}"}
    except subprocess.TimeoutExpired:
        return {"question_id": question_id, "profile_index": profile_idx,
                "status": "timeout", "error": f"Exceeded {timeout}s"}
    except Exception as e:
        return {"question_id": question_id, "profile_index": profile_idx,
                "status": "runtime_error", "error": str(e)}
    finally:
        out_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="Teaching agent model")
    parser.add_argument("--student-model", default=None,
                        help="Student agent model (defaults to --model)")
    parser.add_argument("--runs-dir", default="runs_batch")
    parser.add_argument("--timeout", type=int, default=PLAN_TIMEOUT,
                        help="Per-plan timeout in seconds (default 600)")
    args = parser.parse_args()

    student_model = args.student_model or args.model

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or "your" in api_key.lower():
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    print(f"API base:       {base_url}")
    print(f"Teaching model: {args.model}")
    print(f"Student model:  {student_model}")
    print(f"Per-plan timeout: {args.timeout}s")

    runs_dir = Path(__file__).parent / args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results
    summary_path = runs_dir / "batch_summary.json"
    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8"))
        done_ids = {r["question_id"] for r in results}
        print(f"Resume: {len(done_ids)} already done, skipping them")
    else:
        results = []
        done_ids = set()

    samples = load_samples(args.n, args.seed)
    pending = [e for e in samples if e["question_id"] not in done_ids]
    total = len(samples)
    print(f"Sampling: {total} plans (seed={args.seed}), {len(pending)} remaining\n")

    success_count = sum(1 for r in results if r["status"] == "ok" and r.get("succeeded"))
    error_count = len(results) - success_count

    for entry in pending:
        i = len(results) + 1
        qid = entry.get("question_id", "?")
        print(f"[{i:3d}/{total}] question_id={qid}", flush=True)

        result = run_one_with_timeout(entry, runs_dir, args.model,
                                      student_model, args.timeout)
        results.append(result)

        if result["status"] == "ok" and result.get("succeeded"):
            success_count += 1
            print(f"         OK  steps={result['completed_steps']}")
        elif result["status"] == "timeout":
            error_count += 1
            print(f"         TIMEOUT  {result['error'][:80]}")
        else:
            error_count += 1
            err = result.get("error") or result.get("errors") or result["status"]
            print(f"         ERR {str(err)[:80]}")

        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Done. {success_count} succeeded, {error_count} failed/timeout.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
