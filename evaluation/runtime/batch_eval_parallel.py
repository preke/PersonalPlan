#!/usr/bin/env python3
"""Parallel wrapper around run_single_plan.py.

Same behavior as batch_eval.py but dispatches N workers via ThreadPoolExecutor.
Resumes by checking summary file. Use modest concurrency (4-6) to avoid bianxie.ai rate limits.

Usage:
    python3 batch_eval_parallel.py --n 50 --workers 5 --runs-dir runs_newprompt_50 --seed 0
"""
from __future__ import annotations
import argparse, json, os, random, socket, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
load_dotenv(Path.home() / "github" / ".env")
socket.setdefaulttimeout(30)
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

DATASET_PATH = (
    Path(__file__).parent / "multi_agent_datasets"
    / "multi_agent_dataset_filtered_qap_latest.jsonl"
)
PLAN_TIMEOUT = 480
WORKER_SCRIPT = str(Path(__file__).parent / "run_single_plan.py")


def load_samples(n, seed):
    with open(DATASET_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in random.Random(seed).sample(lines, min(n, len(lines)))]


def run_one(entry, runs_dir, model, student_model, timeout):
    qid = entry.get("question_id", "?"); pidx = entry.get("profile_index", 0)
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
            timeout=timeout, capture_output=True,
        )
        if out_path.stat().st_size:
            return json.loads(out_path.read_text(encoding="utf-8"))
        return {"question_id": qid, "profile_index": pidx, "status": "worker_error",
                "error": (proc.stderr.decode(errors="replace")[:200] or f"exit {proc.returncode}")}
    except subprocess.TimeoutExpired:
        return {"question_id": qid, "profile_index": pidx, "status": "timeout",
                "error": f"Exceeded {timeout}s"}
    except Exception as e:
        return {"question_id": qid, "profile_index": pidx, "status": "runtime_error",
                "error": str(e)}
    finally:
        out_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--student-model", default=None)
    ap.add_argument("--runs-dir", default="runs_newprompt_50")
    ap.add_argument("--timeout", type=int, default=PLAN_TIMEOUT)
    args = ap.parse_args()
    student_model = args.student_model or args.model

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or "your" in api_key.lower():
        print("ERROR: OPENAI_API_KEY not set."); sys.exit(1)

    print(f"API base:       {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"Teaching model: {args.model}")
    print(f"Student model:  {student_model}")
    print(f"Workers:        {args.workers}  (concurrent)")
    print(f"Per-plan timeout: {args.timeout}s")

    runs_dir = Path(__file__).parent / args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "batch_summary.json"

    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8"))
        done = {(r["question_id"], r.get("profile_index", 0)) for r in results}
        print(f"Resume: {len(done)} done")
    else:
        results, done = [], set()

    samples = load_samples(args.n, args.seed)
    pending = [e for e in samples
               if (e["question_id"], e.get("profile_index", 0)) not in done]
    total = len(samples)
    print(f"Sampling {total} plans (seed={args.seed}); {len(pending)} pending\n", flush=True)

    lock = Lock()
    completed = len(results)

    def submit_one(entry):
        return run_one(entry, runs_dir, args.model, student_model, args.timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(submit_one, e): e for e in pending}
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results.append(res)
                completed += 1
                qid = res.get("question_id"); pidx = res.get("profile_index", 0)
                if res["status"] == "ok" and res.get("succeeded"):
                    tag = f"OK steps={res.get('completed_steps')}"
                elif res["status"] == "timeout":
                    tag = "TIMEOUT"
                else:
                    tag = f"ERR {res['status']}"
                print(f"[{completed:3d}/{total}] qid={qid} p{pidx}  {tag}", flush=True)
                summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    n_ok = sum(1 for r in results if r["status"] == "ok" and r.get("succeeded"))
    print(f"\n{'='*50}\nDone. {n_ok}/{len(results)} succeeded.\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
