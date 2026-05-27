"""Smoke-test / batch entry for the AutoGen baseline (v1: qwen3-32b backbone)."""
import argparse
from pathlib import Path

from baselines.autogen_qwen3.plan import make_plan_fn
from baselines.common.runner import run_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit the number of (query, profile) pairs to run.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("evaluation_results/baselines/autogen_qwen3"),
                    help="Output directory for plans / failures / native_outputs.")
    args = ap.parse_args()

    plan_fn = make_plan_fn()
    run_baseline(
        baseline_name="autogen_qwen3",
        plan_fn=plan_fn,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
