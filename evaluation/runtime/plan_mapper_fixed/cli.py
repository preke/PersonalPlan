from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .compiler import compile_plan
from .conformance import check_immutability
from .evaluator import evaluate_execution
from .flow_codegen import generate_flow_code, write_generated_flow
from .models import PlanPayload, RuntimeConfig
from .runtime import PlanRuntime


def _load_plan(path: Path) -> PlanPayload:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PlanPayload.model_validate(data)


def _plan_requires_human_input(plan: PlanPayload) -> bool:
    for subtask in plan.output.subtasks:
        for step in subtask.steps:
            if getattr(step, "requires_human_input", False):
                return True
    return False


def _find_latest_run_dir(runs_dir: Path) -> Path | None:
    """Find the most recently created run-* directory."""
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_evaluation(plan_path: Path, run_dir: Path, evaluator_model: str,
                    accepted_answer: str | None = None) -> None:
    """Run the 3-check evaluator on execution_log.json in run_dir."""
    log_path = run_dir / "execution_log.json"
    if not log_path.exists():
        print(f"[WARN] No execution_log.json found in {run_dir}, skipping evaluation")
        return

    report_path = run_dir / "evaluation_report.txt"
    print(f"\n{'='*60}")
    print("Running execution verification (3-check evaluation)...")
    print(f"{'='*60}\n")

    report = evaluate_execution(
        plan_path=plan_path,
        execution_log_path=log_path,
        accepted_answer=accepted_answer,
        evaluator_model=evaluator_model,
        output_path=report_path,
    )
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan -> CrewAI runtime mapper")
    parser.add_argument("--plan", required=True, help="Path to plan JSON")
    parser.add_argument(
        "--engine",
        choices=["runtime", "flow"],
        default="flow",
        help="Execution engine: flow (default) or runtime",
    )
    parser.add_argument("--mode", choices=["smoke", "live"], default="smoke")
    parser.add_argument("--model", default=None, help="Optional model override for CrewAI Agent")
    parser.add_argument("--runs-dir", default="runs", help="Run output directory")
    parser.add_argument("--emit-flow", default=None, help="Write generated CrewAI flow .py file")
    parser.add_argument("--student-model", default=None, help="Optional model for student agent in generated flow")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max interaction rounds for interactive steps")
    parser.add_argument(
        "--interactive-mode",
        choices=["auto", "simulated_student", "teacher_only"],
        default="auto",
        help="How requires_human_input steps are handled in Flow mode",
    )
    parser.add_argument("--run-generated", action="store_true", help="Execute generated flow immediately")
    parser.add_argument("--evaluate", default=None,
                        help="Path to execution_log.json to evaluate only (skip execution)")
    parser.add_argument("--auto-evaluate", action="store_true",
                        help="Automatically run 3-check evaluation after execution")
    parser.add_argument("--evaluator-model", default="gpt-4o-mini", help="Model for execution evaluator")
    parser.add_argument("--accepted-answer", default=None, help="Optional accepted answer for evaluation reference")
    args = parser.parse_args()

    load_dotenv()
    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

    plan_path = Path(args.plan)

    # Evaluation-only mode: run 3-check evaluator on existing execution log
    if args.evaluate:
        log_path = Path(args.evaluate)
        report_path = log_path.parent / "evaluation_report.txt"
        report = evaluate_execution(
            plan_path=plan_path,
            execution_log_path=log_path,
            accepted_answer=args.accepted_answer,
            evaluator_model=args.evaluator_model,
            output_path=report_path,
        )
        print(report)
        return

    plan = _load_plan(plan_path)

    compile_report = compile_plan(plan)
    if compile_report.dependency_errors or compile_report.tool_binding_errors:
        run_id = f"failed-compile-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        run_dir = Path(args.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "compile_report.json").write_text(
            json.dumps(compile_report.model_dump(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        raise SystemExit("Compile validation failed. See compile_report.json")

    if args.engine == "flow":
        model_name = args.model or "openai/gpt-4o-mini"
        resolved_interactive_mode = args.interactive_mode
        if resolved_interactive_mode == "auto":
            resolved_interactive_mode = (
                "simulated_student" if _plan_requires_human_input(plan) else "teacher_only"
            )

        if args.emit_flow:
            output_path = Path(args.emit_flow)
            generated_code = write_generated_flow(
                plan=plan,
                output_path=output_path,
                llm_model=model_name,
                student_llm_model=args.student_model,
                max_rounds=args.max_rounds,
                interactive_mode=resolved_interactive_mode,
                flow_runs_dir=args.runs_dir,
            )
            print(f"Generated flow: {output_path}")
            print(f"Interactive mode: {resolved_interactive_mode}")
            if args.run_generated:
                exec(generated_code, {"__name__": "__main__"})
        else:
            generated_code = generate_flow_code(
                plan=plan,
                llm_model=model_name,
                student_llm_model=args.student_model,
                max_interaction_rounds=args.max_rounds,
                interactive_mode=resolved_interactive_mode,
                flow_runs_dir=args.runs_dir,
            )
            print(f"Interactive mode: {resolved_interactive_mode}")
            exec(generated_code, {"__name__": "__main__"})

        # Auto-evaluate after flow execution
        if args.auto_evaluate:
            runs_dir = Path(args.runs_dir)
            run_dir = _find_latest_run_dir(runs_dir)
            if run_dir:
                _run_evaluation(plan_path, run_dir, args.evaluator_model, args.accepted_answer)
        return

    # Runtime engine path
    conformance = check_immutability(plan, plan)
    run_id = f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "compile_report.json").write_text(
        json.dumps(compile_report.model_dump(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (run_dir / "conformance_report.json").write_text(
        json.dumps(conformance.model_dump(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    config = RuntimeConfig(mode=args.mode, model=args.model, student_model=args.student_model, run_id=run_id)
    runtime = PlanRuntime(plan=plan, config=config, run_dir=run_dir)
    report = runtime.run()

    print(f"Run ID: {run_id}")
    print(f"Mode: {report.mode}")
    print(f"Succeeded: {report.succeeded}")
    print(f"Completed steps: {len(report.completed_steps)}")
    print(f"Failed steps: {len(report.failed_steps)}")
    print(f"Artifacts: {run_dir}")

    # Auto-evaluate after runtime execution
    if args.auto_evaluate:
        _run_evaluation(plan_path, run_dir, args.evaluator_model, args.accepted_answer)


if __name__ == "__main__":
    main()
