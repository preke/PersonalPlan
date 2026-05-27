"""
Native-output sidecar logger for 方案 B (appendix native evaluation).

Each baseline plan_fn that runs a paper-method pipeline may produce a
"native" output (lesson plan / bundle / AOP-native list / etc.) in
addition to the §9 plan it returns. To support the 方案 A+B mixed
design (see BASELINE_DESIGN/03_FORMAT_MISMATCH.md), we save those
native outputs to a sidecar file alongside plans.jsonl, keyed by the
same (question_id, profile_index) tuple.

Usage:
  - The runner calls `configure(output_path)` once at startup.
  - For each item, the runner calls `set_item(question_id, profile_idx)`
    before invoking plan_fn.
  - Inside plan_fn, the baseline calls `log_native(native_dict)` once
    after producing its native output.

The module is a no-op until `configure(...)` is called, so plan_fn
modules that always emit a native record are safe in unit tests.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_state = threading.local()


def configure(output_path: str | Path) -> None:
    """Open the sidecar JSONL for appending. Called once by the runner."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _state.output_path = p
    _state.file = open(p, "a", encoding="utf-8")
    _state.current_key = (None, None)


def set_item(question_id: Any, profile_index: Any) -> None:
    """Pin the next log_native(...) call to this item's identity."""
    _state.current_key = (question_id, profile_index)


def log_native(native: Any, extra: dict | None = None) -> None:
    """Append a native record. No-op when configure(...) wasn't called.

    `native` is typically the dict / list produced by the paper method
    before any §9 enforcement. `extra` lets baselines attach metadata
    (e.g., {"replan_rounds": 2, "schema_failure_reason": "..."}).
    """
    f = getattr(_state, "file", None)
    if f is None:
        return
    qid, pidx = getattr(_state, "current_key", (None, None))
    record: dict[str, Any] = {
        "question_id": qid,
        "profile_index": pidx,
        "native": native,
    }
    if extra:
        record["extra"] = extra
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def close() -> None:
    f = getattr(_state, "file", None)
    if f is not None:
        f.close()
        _state.file = None
