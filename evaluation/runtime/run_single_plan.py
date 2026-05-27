#!/usr/bin/env python3
"""
Worker: run a single plan and write result to --out file.
Called by batch_eval.py as a subprocess with a hard timeout.
"""
import argparse
import json
import os
import socket
import sys
import uuid
from pathlib import Path

socket.setdefaulttimeout(30)

from dotenv import load_dotenv
load_dotenv(Path.home() / "github" / ".env")
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

sys.path.insert(0, str(Path(__file__).parent))
from plan_mapper_fixed.compiler import compile_plan
from plan_mapper_fixed.models import PlanPayload, RuntimeConfig
from plan_mapper_fixed.runtime import PlanRuntime


def normalize_plan(plan):
    import copy
    plan = copy.deepcopy(plan)
    learner = plan.get("input", {}).get("learner", {})
    if "about_me" in learner and "self_description" not in learner:
        learner["self_description"] = learner.pop("about_me")
    if "top_tags" in learner and "skills" not in learner:
        learner["skills"] = learner.pop("top_tags")
    for agent in plan.get("output", {}).get("agents", []):
        if "backstory" in agent and "description" not in agent:
            agent["description"] = agent.pop("backstory")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True, help="JSON string of dataset entry")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    entry = json.loads(args.entry)
    question_id = entry.get("question_id", "unknown")
    profile_idx = entry.get("profile_index", 0)
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)

    plan_data = normalize_plan(entry["plan"])

    try:
        plan = PlanPayload.model_validate(plan_data)
    except Exception as e:
        out_path.write_text(json.dumps({"question_id": question_id,
                                        "status": "parse_error", "error": str(e)}))
        return

    compile_report = compile_plan(plan)
    errors = compile_report.dependency_errors + compile_report.tool_binding_errors
    if errors:
        out_path.write_text(json.dumps({"question_id": question_id,
                                        "status": "compile_error", "errors": errors}))
        return

    run_id = f"run-{question_id}-p{profile_idx}-{uuid.uuid4().hex[:6]}"
    run_dir = runs_dir / run_id
    config = RuntimeConfig(mode="live", model=args.model,
                           student_model=args.student_model, run_id=run_id)
    runtime = PlanRuntime(plan=plan, config=config, run_dir=run_dir)

    try:
        report = runtime.run()
        result = {
            "question_id": question_id,
            "profile_index": profile_idx,
            "run_id": run_id,
            "status": "ok",
            "succeeded": report.succeeded,
            "completed_steps": len(report.completed_steps),
            "failed_steps": len(report.failed_steps),
            "artifacts": str(run_dir),
        }
    except Exception as e:
        result = {"question_id": question_id, "profile_index": profile_idx,
                  "run_id": run_id, "status": "runtime_error", "error": str(e)}

    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
