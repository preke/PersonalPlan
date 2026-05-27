"""
AutoAgents baseline — real upstream path, scheme A + B.

For each (query, learner) pair, we shell out to the upstream AutoAgents
pipeline running inside the .venvs/autoagents_310 conda environment
(Python 3.10 with all 245 pinned packages from
external/AutoAgents/requirements.txt). The subprocess runs the real
Manager + ObserverAgents + ObserverPlans loop against DashScope's
OpenAI-compatible qwen3-32b endpoint (v1 baseline F-tier backbone) and
writes the Manager's final §9 JSON output (per scheme A: the Manager
is instructed via compose_t4() task description to emit §9 JSON
directly, no markdown headers).

Scheme A (BASELINE_DESIGN/03_FORMAT_MISMATCH.md):
  - No LLM-based markdown -> §9 remap. Output format is part of the
    task definition; non-conforming output is a baseline performance
    failure, not something we paper over downstream.
  - Schema validation is delegated to baselines.common.runner.

Scheme B (sidecar native logging):
  - We still call native_logger.log_native(plan_text=...) so the raw
    Manager output (whatever it is — valid JSON, malformed JSON, or
    leftover markdown) is preserved for appendix analysis.

Method preservation (P1 strategy, target ~95%):
  - The Manager (CreateRoles -> CheckRoles -> CreatePlan -> CheckPlan
    cycle with internal consensus loop), ObserverAgents, and
    ObserverPlans all run as the original upstream code.
  - LLM backbone: DashScope qwen3-32b via the upstream litellm path
    (with `enable_thinking=False` injected for Qwen3 streaming quirk).
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import os
from pathlib import Path

from baselines.common.json_repair import fix_json_format
from baselines.common.native_logger import log_native


def _resolve_venv_python(venv_name: str, env_var: str) -> Path:
    """Locate the Python interpreter inside .venvs/<venv_name>.

    Precedence:
      1. Env var override (absolute path to python interpreter)
      2. Auto-detect conda-style:    <root>/.venvs/<name>/python.exe
      3. Auto-detect Windows venv:   <root>/.venvs/<name>/Scripts/python.exe
      4. Auto-detect Linux/Mac venv: <root>/.venvs/<name>/bin/python
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]  # baselines/autoagents/plan.py → ROOT
    base = root / ".venvs" / venv_name
    for candidate in (base / "python.exe",
                      base / "Scripts" / "python.exe",
                      base / "bin" / "python"):
        if candidate.exists():
            return candidate
    return base / "python.exe"  # fallback; subprocess will surface a clear FileNotFoundError


UPSTREAM_VENV_PY = _resolve_venv_python("autoagents_310", "AUTOAGENTS_VENV_PYTHON")
RUNNER_SCRIPT = Path(__file__).parent / "upstream_runner.py"
SUBPROCESS_TIMEOUT = 900  # seconds per (query, profile)
UPSTREAM_RETRIES = 2      # how many extra attempts on upstream crash


def _extract_json(text: str):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        pass
    for level in (1, 2):
        try:
            return json.loads(fix_json_format(blob, repair_attempt=level))
        except Exception:
            continue
    return None


def _ensure_input_block(plan: dict, query: str, learner: dict) -> dict:
    if not isinstance(plan, dict):
        return plan
    plan.setdefault("input", {})
    plan["input"]["query"] = query
    plan["input"]["learner"] = learner
    return plan


def _run_upstream_once(query: str, learner: dict) -> tuple[bool, str, str]:
    """Single attempt. Returns (ok, plan_text, err_msg)."""
    with tempfile.TemporaryDirectory(prefix="autoagents_") as td:
        in_path = Path(td) / "in.json"
        out_path = Path(td) / "out.json"
        in_path.write_text(
            json.dumps({"query": query, "learner": learner}, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [str(UPSTREAM_VENV_PY), str(RUNNER_SCRIPT),
                 str(in_path), str(out_path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=SUBPROCESS_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "", f"timeout after {SUBPROCESS_TIMEOUT}s"
        if not out_path.exists():
            return False, "", "no out.json produced"
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as err:
            return False, "", f"out.json parse error: {err}"
        if not payload.get("ok"):
            return False, "", payload.get("error", "?")
        return True, payload.get("plan_text", "") or "", ""


def _run_upstream(query: str, learner: dict) -> str:
    """Spawn the upstream pipeline with retries on stochastic crashes.
    Returns "" if every attempt fails."""
    for attempt in range(1 + UPSTREAM_RETRIES):
        ok, plan_text, err = _run_upstream_once(query, learner)
        if ok:
            return plan_text
        print(f"[autoagents] upstream attempt {attempt + 1}/"
              f"{1 + UPSTREAM_RETRIES} failed: {err}")
    return ""


def plan_fn(query: str, learner: dict) -> dict:
    plan_text = _run_upstream(query, learner)
    # Scheme B: persist the raw Manager output before any parsing /
    # validation, so the appendix can audit what the framework
    # actually emitted (even when it fails §9).
    log_native(plan_text, extra={"plan_chars": len(plan_text)})

    plan = _extract_json(plan_text)
    if not isinstance(plan, dict):
        return {}
    return _ensure_input_block(plan, query, learner)
