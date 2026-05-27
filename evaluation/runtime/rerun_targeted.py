#!/usr/bin/env python3
"""Re-run specific question_ids with fixed code. Pass qids as space-separated args."""
import json, os, random, subprocess, sys, tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / "github" / ".env")
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

DATASET = Path(__file__).parent / "multi_agent_datasets" / "multi_agent_dataset_filtered_qap_latest.jsonl"
WORKER  = str(Path(__file__).parent / "run_single_plan.py")
TIMEOUT = 480
MODEL   = "openai/gpt-4o"
STUDENT = "openai/gpt-4o-mini"
RUNS_DIR = Path(__file__).parent / "runs_batch_rerun"

target_qids = set(sys.argv[1:])
if not target_qids:
    print("Usage: python3 rerun_targeted.py QID1 QID2 ...")
    sys.exit(1)

# Load matching entries from dataset (keep same profile_index as original batch)
orig_summary = json.loads((Path(__file__).parent / "runs_batch_100" / "batch_summary.json").read_text())
orig_profile = {str(r["question_id"]): r.get("profile_index", 0) for r in orig_summary}

entries = []
seen = set()
with open(DATASET, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        qid = str(d.get("question_id", ""))
        pidx = int(d.get("profile_index", 0))
        if qid in target_qids and qid not in seen:
            if pidx == orig_profile.get(qid, 0):
                entries.append(d)
                seen.add(qid)

print(f"Found {len(entries)}/{len(target_qids)} matching entries")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
summary_path = RUNS_DIR / "batch_summary.json"
results = json.loads(summary_path.read_text()) if summary_path.exists() else []
done_ids = {str(r["question_id"]) for r in results}

success = sum(1 for r in results if r.get("succeeded"))
errors  = len(results) - success

for entry in entries:
    qid = str(entry.get("question_id", "?"))
    if qid in done_ids:
        print(f"  SKIP {qid} (already done)")
        continue
    print(f"  [{len(results)+1}/{len(entries)}] qid={qid}", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [sys.executable, WORKER,
             "--entry", json.dumps(entry),
             "--runs-dir", str(RUNS_DIR),
             "--model", MODEL,
             "--student-model", STUDENT,
             "--out", str(out_path)],
            timeout=TIMEOUT, capture_output=True,
        )
        if out_path.stat().st_size:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            stderr = proc.stderr.decode(errors="replace")[:200]
            result = {"question_id": qid, "status": "worker_error", "error": stderr}
    except subprocess.TimeoutExpired:
        result = {"question_id": qid, "status": "timeout", "error": f"Exceeded {TIMEOUT}s"}
    except Exception as e:
        result = {"question_id": qid, "status": "runtime_error", "error": str(e)}
    finally:
        out_path.unlink(missing_ok=True)

    results.append(result)
    if result.get("status") == "ok" and result.get("succeeded"):
        success += 1
        print(f"    OK steps={result.get('completed_steps')}")
    else:
        errors += 1
        print(f"    ERR {str(result.get('error',''))[:80]}")

    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nDone. {success} succeeded, {errors} failed. Summary: {summary_path}")
