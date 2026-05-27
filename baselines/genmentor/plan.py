"""
GenMentor baseline -- real upstream path (scheme A+B, 2026-05-17).

For each (query, learner) pair, we shell out to the upstream GenMentor
backend running inside .venvs/genmentor_310 (Python 3.10 with the slim
subset of external/gen-mentor/backend/requirements.txt needed for the
4 module agent classes -- see upstream_runner.py docstring).

The subprocess runs Modules 1-3 natively (Skill Gap Identifier ->
Adaptive Learner Modeler -> Learning Path Scheduler), producing a bundle
{skill_gaps, skill_requirements, learner_profile, learning_path,
session_outlines}. We treat that bundle as the paper method's
intermediate state; Module 4 (Tailored Content) is then re-cast as
"generate the final §9 plan using the bundle as context" — a single
LLMClient(qwen3-32b) call using compose_t4() (PREAMBLE+§5+§9+§12) as
system prompt. This call IS the method's final step, not a post-hoc
schema translator (cf. BASELINE_DESIGN scheme A+B).

Method preservation:
  - Upstream M1-M3 (and M4's session-outline branch) run as native
    GenMentor code with original prompts + pydantic validation, on
    DashScope qwen3-32b.
  - The final §9 assembly call uses the same qwen3-32b backbone so the
    backbone is consistent end-to-end.
  - The raw bundle is also persisted via native_logger.log_native for
    scheme B appendix evaluation.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import os

from baselines.common.json_repair import fix_json_format
from baselines.common.llm_client import LLMClient
from baselines.common.native_logger import log_native
from baselines.common.prompt_sections import compose_t4


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
    root = Path(__file__).resolve().parents[2]  # baselines/genmentor/plan.py → ROOT
    base = root / ".venvs" / venv_name
    for candidate in (base / "python.exe",
                      base / "Scripts" / "python.exe",
                      base / "bin" / "python"):
        if candidate.exists():
            return candidate
    return base / "Scripts" / "python.exe"  # fallback; subprocess will surface a clear FileNotFoundError


UPSTREAM_VENV_PY = _resolve_venv_python("genmentor_310", "GENMENTOR_VENV_PYTHON")
RUNNER_SCRIPT = Path(__file__).parent / "upstream_runner.py"
SUBPROCESS_TIMEOUT = 900  # seconds per (query, profile); 4 modules run sequentially
UPSTREAM_RETRIES = 2      # how many extra attempts on upstream crash

# T5 backbone per v1 baseline design: qwen3-32b via DashScope.
_BACKBONE = "qwen3-32b"
_llm = LLMClient(backend=_BACKBONE)
_T4_SYSTEM = compose_t4()


_ASSEMBLY_INSTRUCTION = """\
GenMentor's Modules 1-3 (Skill Gap Identifier, Adaptive Learner Modeler,
Learning Path Scheduler) just produced the following bundle for this
learner. This bundle is your intermediate state -- skill_gaps,
learner_profile, and learning_path describe what to teach and to whom.

Using that bundle as context, generate the final multi-agent teaching
plan in the strict §2 JSON schema described in your system prompt. This
JSON IS your final method output (GenMentor's Module 4 final step) --
it is NOT a translation of someone else's plan. Design the agents,
subtasks, steps, and execution_order to close the skill gaps the bundle
identified, personalized for the learner_profile it built, following the
learning_path it scheduled.

QUERY:
{query}

LEARNER:
{learner_json}

BUNDLE (M1-M3 output):
{bundle_json}

Output STRICT JSON only -- start with {{ and end with }}. No fences, no
commentary, no preface.
"""


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


def _slim_bundle(bundle: dict) -> dict:
    """Keep only the 4 module outputs relevant to §9 assembly."""
    keep = {"skill_gaps", "skill_requirements", "learner_profile",
            "learning_path", "session_outlines"}
    return {k: v for k, v in bundle.items() if k in keep}


def _run_upstream_once(query: str, learner: dict) -> tuple[bool, dict, str]:
    """Single subprocess attempt. Returns (ok, bundle, err_msg)."""
    with tempfile.TemporaryDirectory(prefix="genmentor_") as td:
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
            return False, {}, f"timeout after {SUBPROCESS_TIMEOUT}s"
        if not out_path.exists():
            return False, {}, "no out.json produced"
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as err:
            return False, {}, f"out.json parse error: {err}"
        if not payload.get("ok"):
            return False, {}, payload.get("error", "?")
        # Strip ok/error fields; the bundle is everything else
        bundle = {k: v for k, v in payload.items() if k not in {"ok", "error"}}
        return True, bundle, ""


def _run_upstream(query: str, learner: dict) -> dict:
    """Spawn the upstream pipeline with retries on stochastic crashes.
    Returns {} if every attempt fails."""
    for attempt in range(1 + UPSTREAM_RETRIES):
        ok, bundle, err = _run_upstream_once(query, learner)
        if ok:
            return bundle
        print(f"[genmentor] upstream attempt {attempt + 1}/"
              f"{1 + UPSTREAM_RETRIES} failed: {err}")
    return {}


def _assemble_section9(bundle: dict, query: str, learner: dict) -> dict:
    """Single LLM call: bundle -> §9 plan, framed as GenMentor's M4 final step.

    System prompt is compose_t4() (PREAMBLE+§5+§9+§12). No schema-feedback
    retry; a single shot. Schema failure = baseline failure (scheme A).
    """
    slim = _slim_bundle(bundle)
    bundle_json = json.dumps(slim, ensure_ascii=False)
    if len(bundle_json) > 28000:
        bundle_json = bundle_json[:28000] + "  /* TRUNCATED */"
    user_msg = _ASSEMBLY_INSTRUCTION.format(
        query=query,
        learner_json=json.dumps(learner, ensure_ascii=False),
        bundle_json=bundle_json,
    )
    raw = _llm.chat([
        {"role": "system", "content": _T4_SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    plan = _extract_json(raw)
    if plan is None:
        return {}
    return _ensure_input_block(plan, query, learner)


def plan_fn(query: str, learner: dict) -> dict:
    bundle = _run_upstream(query, learner)
    if not bundle:
        return {}
    log_native(bundle, extra={"upstream_modules": [
        "M1_SkillRequirement", "M2_SkillGap", "M3_LearnerModeler",
        "M3_PathScheduler", "M4_ContentOutline_if_present",
    ]})
    return _assemble_section9(bundle, query, learner)
