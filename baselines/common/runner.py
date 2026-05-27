"""
Generic batch runner. Calls plan_fn(query, learner) for each input,
validates the result, writes plans.jsonl / failures.jsonl,
tracks progress.
"""
import json, traceback
from pathlib import Path
from tqdm import tqdm
from baselines.common.data_loader import load_main_dataset
from baselines.common.schema_validator import validate_plan, PlanSchemaError
from baselines.common.json_repair import fix_json_format
from baselines.common.progress import ProgressTracker
from baselines.common import native_logger


def _force_input_block(plan: dict, query: str, learner: dict) -> dict:
    """Ensure plan.input matches what we fed in (some models drop it)."""
    plan.setdefault("input", {})
    plan["input"]["query"] = query
    plan["input"]["learner"] = learner
    return plan


def _try_parse_and_repair(raw: str) -> dict | None:
    """Try direct json.loads; on failure try fix_json_format up to 2 levels."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        pass
    for attempt in (1, 2):
        try:
            return json.loads(fix_json_format(raw, repair_attempt=attempt))
        except Exception:
            continue
    return None


def run_baseline(baseline_name: str, plan_fn, output_dir: Path,
                 dataset_path: Path = None, limit: int = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_path = output_dir / "plans.jsonl"
    failures_path = output_dir / "failures.jsonl"
    progress = ProgressTracker(output_dir / "progress.json")

    plans_f = open(plans_path, "a", encoding="utf-8")
    fail_f = open(failures_path, "a", encoding="utf-8")

    # 方案 B sidecar (no-op for baselines that never call log_native).
    native_logger.configure(output_dir / "native_outputs.jsonl")

    try:
        for item in tqdm(load_main_dataset(
                dataset_path or load_main_dataset.__defaults__[0],
                limit=limit), desc=baseline_name):
            if progress.is_done(item["key"]):
                continue
            native_logger.set_item(item["question_id"], item["profile_idx"])
            try:
                raw = plan_fn(item["query"], item["learner"])
                plan = _try_parse_and_repair(raw) if not isinstance(raw, dict) else raw
                if plan is None:
                    raise PlanSchemaError("could not parse JSON")
                plan = _force_input_block(plan, item["query"], item["learner"])
                validate_plan(plan)
                plans_f.write(json.dumps({
                    "question_id": item["question_id"],
                    "profile_index": item["profile_idx"],
                    "generated_plan": plan,
                }, ensure_ascii=False) + "\n")
                plans_f.flush()
                progress.mark_done(item["key"])
                progress.save()
            except Exception as e:
                fail_f.write(json.dumps({
                    "key": item["key"],
                    "error": str(e)[:500],
                    "traceback": traceback.format_exc()[:2000],
                }, ensure_ascii=False) + "\n")
                fail_f.flush()
    finally:
        plans_f.close()
        fail_f.close()
        progress.save()
        native_logger.close()
