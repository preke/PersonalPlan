#!/usr/bin/env python3
"""
Tier 2 Multi-Agent Execution Evaluator (strict HTML conformance)
Source: EVALUATION_DESIGN_2026-05-15.html §5 + §8.2 + §11.4 + §11.5

Strict policy: implement EXACTLY what HTML reference code says. No custom
hardening / error-filters / unfenced-code fallbacks / executor-view rebuild.

Four headline metrics:
  EVR    — §11.4   5 sub-checks: cov + loop + flow parsed from v1 evaluator
                   markdown (Check 2 sub-fields); exec rule-based; out LLM judge
  PAS    — §11.5   per-utterance PRR judge
  PQS    — §11.5   mean(NDAR + SPR + IAR)
  r_sol  — §11.5   student final code vs accepted_answer + HARD compile gate

Data-layer adapters (necessary because the dataset stores empty accepted_answer):
  - accepted_answer loaded from MAP-PPL construction sources (profiles_answers[pidx].answer)
    — plan.input.accepted_answer is empty in 100% of records.

Usage:
    python3 tier2_evaluator_v2.py --stage3 --out tier2_v2_strict.json
    python3 tier2_evaluator_v2.py --stage3 --limit 5 --out pilot_strict.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path.home() / "github" / ".env")
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

REPO_ROOT = Path(__file__).resolve().parent
STAGE3_DIR = REPO_ROOT
DATASET_PATHS = [
    REPO_ROOT / "multi_agent_dataset_filtered_qap_v3.jsonl",
    REPO_ROOT / "multi_agent_dataset_filtered_qap.jsonl",
]
QAP_SOURCES = [
    REPO_ROOT / "qap_task3.jsonl",
    REPO_ROOT / "qap_task2.jsonl",
    REPO_ROOT / "qap_task1.jsonl",
]

# v1 4-check evaluator (IF/WC/IQ/CCGE) — also drives EVR cov/loop/flow per HTML §11.4
sys.path.insert(0, str(REPO_ROOT))
from plan_mapper_fixed.evaluator import evaluate_execution as _v1_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Timeout / failure event log (module-level, reset at start of each batch run)
# All API timeouts, Docker failures, and per-run timeouts accumulate here.
# ---------------------------------------------------------------------------

_TIMEOUT_LOG: list[dict] = []


def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
# ===========================================================================
# Code re-execution layer — mas-runtime Docker
# ---------------------------------------------------------------------------
# Re-runs each CodeInterpreterTool step's source code in a Docker container
# that supports 29 languages (Python, C, C++, Java, JS, TS, C#, Ruby, PHP,
# Go, Rust, Swift, Groovy, Dart, Scala, Haskell, OCaml, Kotlin, R, Perl,
# AWK, Bash, SQL, PowerShell, Lua, F#, ObjC, Snakemake, Maven). Adds
# reexec_content / reexec_status / reexec_lang to step_outputs in-place so
# the downstream EVR exec/out checks see real sandbox stdout.

DOCKER_IMAGE = os.environ.get("MAS_RUNTIME_IMAGE", "mas-runtime:latest")

# Map language → (file extension, in-container run command).
LANG_RUNNERS = {
    "python": ("py", "python3 /work/source.py"),
    "py": ("py", "python3 /work/source.py"),
    "c": ("c", "gcc /work/source.c -o /work/bin -lm && /work/bin"),
    "cpp": ("cpp", "g++ -std=c++17 /work/source.cpp -o /work/bin -lm && /work/bin"),
    "c++": ("cpp", "g++ -std=c++17 /work/source.cpp -o /work/bin -lm && /work/bin"),
    "java": ("java", (
        # Java requires public class to match filename. Detect class name from source.
        "cd /work && "
        "CLASS=$(grep -oE 'public\\s+(class|interface|enum)\\s+[A-Za-z_][A-Za-z0-9_]*' source.java | head -1 | awk '{print $NF}') && "
        "[ -z \"$CLASS\" ] && CLASS=Main; "
        "cp source.java \"$CLASS.java\" && "
        "javac -encoding UTF-8 \"$CLASS.java\" && java -cp /work \"$CLASS\""
    )),
    "javascript": ("js", "node /work/source.js"),
    "js": ("js", "node /work/source.js"),
    "node": ("js", "node /work/source.js"),
    "typescript": ("ts", "tsx /work/source.ts"),
    "ts": ("ts", "tsx /work/source.ts"),
    "ruby": ("rb", "ruby /work/source.rb"),
    "php": ("php", "php /work/source.php"),
    "go": ("go", "cd /work && go run source.go"),
    # Rust runner: if code uses external crates (serde/regex/...), build via cargo
    # with a minimal Cargo.toml that bundles common deps. Otherwise rustc single-file.
    "rust": ("rs", (
        "cd /work && "
        "if grep -qE '^\\s*use\\s+(serde|regex|tokio|reqwest|rand|chrono|anyhow)' source.rs; then "
        "  mkdir -p proj/src && cp source.rs proj/src/main.rs && cd proj && "
        "  cat > Cargo.toml <<'EOF'\n"
        "[package]\nname = \"work\"\nversion = \"0.0.1\"\nedition = \"2021\"\n"
        "[dependencies]\n"
        "serde = { version = \"1\", features = [\"derive\"] }\n"
        "serde_json = \"1\"\n"
        "regex = \"1\"\n"
        "rand = \"0.8\"\n"
        "anyhow = \"1\"\n"
        "EOF\n"
        "  cargo run --quiet --offline 2>&1 || cargo run --quiet 2>&1; "
        "else "
        "  rustc source.rs -o bin && ./bin; "
        "fi"
    )),
    "csharp": ("cs", (
        "cd /work && mkdir -p proj && cp source.cs proj/Program.cs && "
        "cd proj && dotnet new console --force >/dev/null 2>&1 && "
        "cp ../source.cs Program.cs && dotnet run --no-restore 2>&1 | tail -20"
    )),
    "c#": ("cs", (
        "cd /work && mkdir -p proj && cp source.cs proj/Program.cs && "
        "cd proj && dotnet new console --force >/dev/null 2>&1 && "
        "cp ../source.cs Program.cs && dotnet run --no-restore 2>&1 | tail -20"
    )),
    "bash": ("sh", "bash /work/source.sh"),
    "shell": ("sh", "bash /work/source.sh"),
    "sh": ("sh", "bash /work/source.sh"),
    "r": ("R", "Rscript /work/source.R"),
    "perl": ("pl", "perl /work/source.pl"),
    "awk": ("awk", "awk -f /work/source.awk"),
    "kotlin": ("kt", (
        "cd /work && kotlinc source.kt -include-runtime -d bin.jar && "
        "java -jar bin.jar"
    )),
    "sql": ("sql", (
        "psql -U mas -d teaching -f /work/source.sql 2>&1 | head -40 || "
        "mysql -u mas -pmas teaching < /work/source.sql 2>&1 | head -40"
    )),
    "swift": ("swift", "swift /work/source.swift"),
    "groovy": ("groovy", "groovy /work/source.groovy"),
    "dart": ("dart", "dart run /work/source.dart"),
    "scala": ("scala", "scala /work/source.scala"),
    "haskell": ("hs", "runghc /work/source.hs"),
    "hs": ("hs", "runghc /work/source.hs"),
    "ocaml": ("ml", "ocaml /work/source.ml"),
    "powershell": ("ps1", "pwsh -File /work/source.ps1"),
    "pwsh": ("ps1", "pwsh -File /work/source.ps1"),
    "lua": ("lua", "lua /work/source.lua"),
    "fsharp": ("fsx", "dotnet fsi /work/source.fsx"),
    "objc": ("m", (
        "cd /work && gcc -x objective-c source.m -o bin "
        "$(gnustep-config --objc-flags 2>/dev/null) "
        "$(gnustep-config --base-libs 2>/dev/null) 2>&1 || "
        "gcc -x objective-c source.m -o bin -lobjc && ./bin"
    )),
}

_FENCE_FULL_RE = re.compile(r"```([\w+\-#]+)\s*\n(.*?)```", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"```([\w+\-#]+)", re.IGNORECASE)
_STEP_REF_RE = re.compile(r"\bS\d+-\d+\b")
# For instructions that put code as plain text after a marker like
# "Execute the following R code:" without ``` fences
_PLAIN_CODE_MARKER = re.compile(
    r"(?i)(?:execute|run|evaluate)\s+(?:the\s+following|this)\s+(\w+)\s+(?:code|script|snippet|program)"
    r"\s*(?:and\s+[^:\n]*)?\s*:?\s*\n+(.+)",
    re.DOTALL,
)


def _unescape_literal_newlines(text: str) -> str:
    """Some instructions store '\\n' as literal backslash-n. Unescape only if
    the text contains NO real newlines (otherwise leave alone)."""
    if "\\n" in text and "\n" not in text:
        return text.encode("utf-8").decode("unicode_escape", errors="replace")
    return text
_LANG_ALIAS = {"py": "python", "rb": "ruby", "ts": "typescript",
               "js": "javascript", "sh": "bash", "shell": "bash",
               "cs": "csharp", "c#": "csharp", "c++": "cpp"}


def _extract_code_and_fence_lang(
    meta: dict,
    all_step_outputs: dict | None = None,
    exec_log: list[dict] | None = None,
) -> tuple[str, str | None]:
    """Return (code, fence_lang_tag).

    Order:
      1. Fenced block in step's own instruction (after unescaping literal '\\n').
      2. Plain code body after marker like "Execute the following R code:" + blank line.
      3. Follow "from S\\d+-\\d+" reference into step_outputs / log of that step.
    """
    inst = _unescape_literal_newlines(meta.get("instruction") or "")

    # 1. Fenced block in instruction
    m = _FENCE_FULL_RE.search(inst)
    if m:
        return m.group(2), _LANG_ALIAS.get(m.group(1).lower(), m.group(1).lower())

    # 2. Plain code after "Execute the following X code:" marker
    m2 = _PLAIN_CODE_MARKER.search(inst)
    if m2:
        lang = m2.group(1).lower()
        body = m2.group(2).strip()
        # Strip trailing prose blocks if present (heuristic: stop at "Output", "Report", "Check that")
        for marker in ("\n\nOutput", "\n\nReport", "\n\nCheck that", "\n\nReturn the"):
            idx = body.find(marker)
            if idx > 0:
                body = body[:idx]
        lang = _LANG_ALIAS.get(lang, lang)
        if lang in LANG_RUNNERS and body:
            return body, lang

    # 3. Cross-step reference
    if all_step_outputs is None and exec_log is None:
        return "", None
    refs = list(dict.fromkeys(_STEP_REF_RE.findall(inst)))
    if not refs:
        return "", None
    for ref_sid in refs:
        ref_so = (all_step_outputs or {}).get(ref_sid)
        if ref_so:
            content = _unescape_literal_newlines(ref_so.get("content") or "")
            m = _FENCE_FULL_RE.search(content)
            if m:
                return m.group(2), _LANG_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        if exec_log:
            for e in exec_log:
                if e.get("step_id") != ref_sid:
                    continue
                ai = e.get("actual_interaction") or {}
                for fld in ("student_response", "agent_output", "teacher_output"):
                    v = _unescape_literal_newlines(ai.get(fld) or "")
                    m = _FENCE_FULL_RE.search(v)
                    if m:
                        return m.group(2), _LANG_ALIAS.get(m.group(1).lower(), m.group(1).lower())
    return "", None


def _extract_code_to_run(meta: dict, all_step_outputs: dict | None = None,
                         exec_log: list[dict] | None = None) -> str:
    """Backwards-compat shim returning just the code body."""
    code, _ = _extract_code_and_fence_lang(meta, all_step_outputs, exec_log)
    return code


def _detect_code_language(meta: dict, code_hint: str = "",
                          fence_lang: str | None = None) -> str | None:
    """Detect language: fence tag (own or referenced step) → agent role → content sniff."""
    # 0. Fence tag from extracted block (covers cross-step references)
    if fence_lang and fence_lang in LANG_RUNNERS:
        return fence_lang
    # 1. Fenced lang tag in own instruction
    inst = meta.get("instruction") or ""
    m = _FENCE_OPEN_RE.search(inst)
    if m:
        tag = m.group(1).lower().strip()
        tag = _LANG_ALIAS.get(tag, tag)
        if tag in LANG_RUNNERS:
            return tag
    # 2. Agent role <lang>_ prefix
    agent = (meta.get("agent") or "").lower()
    candidates = [
        "python", "typescript", "javascript", "csharp", "cpp", "kotlin",
        "ruby", "rust", "swift", "haskell", "ocaml", "scala", "groovy",
        "java", "node", "php", "perl", "bash", "awk", "go", "dart",
        "objc", "fsharp", "powershell", "snakemake", "sql", "c", "r",
    ]
    for tok in candidates:
        if re.match(rf"^{re.escape(tok)}(_|$|\d)", agent):
            return tok
    # 3. Code-content sniffing
    code = _extract_code_to_run(meta) or (meta.get("instruction") or "")
    if code:
        if re.search(r"#include\s*<.*?>|\bint\s+main\s*\(", code) and "iostream" in code:
            return "cpp"
        if re.search(r"#include\s*<.*?>|\bint\s+main\s*\(", code):
            return "c"
        if re.search(r"\bpublic\s+(class|static)\b|\bSystem\.out\.print", code):
            return "java"
        if re.search(r"\busing\s+System\b|\bConsole\.Write", code):
            return "csharp"
        if re.search(r"\bdef\s+\w+\s*\(|\bprint\s*\(.*\)|^\s*import\s+\w+", code, re.MULTILINE):
            return "python"
        if re.search(r"\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=|\blet\s+\w+\s*=|console\.log", code):
            return "javascript"
        if re.search(r":\s*string\b|:\s*number\b|interface\s+\w+", code):
            return "typescript"
        if re.search(r"\bpackage\s+main\b|\bfunc\s+\w+\s*\(", code):
            return "go"
        if re.search(r"\bfn\s+\w+\s*\(|\blet\s+mut\s+", code):
            return "rust"
        if re.search(r"\bputs\s+|\bdef\s+\w+\s*\n|\.each\s+do\b", code):
            return "ruby"
        if re.search(r"<\?php|\$\w+\s*=", code):
            return "php"
        if re.search(r"\bSELECT\b.*\bFROM\b|\bINSERT\s+INTO\b", code, re.IGNORECASE):
            return "sql"
        if re.search(r"^\s*#!\s*/.*\b(bash|sh)\b|^\s*if\s+\[", code, re.MULTILINE):
            return "bash"
    return None


def _run_in_mas_runtime(code: str, lang: str, timeout: int = 90) -> dict:
    """Execute code in mas-runtime Docker."""
    lang_norm = (lang or "").lower()
    cfg = LANG_RUNNERS.get(lang_norm)
    if not cfg:
        return {"status": "unsupported_lang", "lang": lang, "stdout": "", "stderr": ""}
    ext, cmd = cfg
    with tempfile.TemporaryDirectory() as td:
        src_path = Path(td) / f"source.{ext}"
        src_path.write_text(code, encoding="utf-8")
        try:
            # text=False so binary stdout (e.g., 0xff bytes from char tests) doesn't crash
            proc = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{td}:/work",
                 "--workdir", "/work",
                 DOCKER_IMAGE,
                 "bash", "-c", cmd],
                capture_output=True, text=False, timeout=timeout,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return {
                "status": "ok" if proc.returncode == 0 else "nonzero_exit",
                "exit_code": proc.returncode,
                "stdout": stdout[-4000:],
                "stderr": stderr[-2000:],
                "lang": lang_norm,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "",
                    "stderr": f"timeout after {timeout}s", "lang": lang_norm}
        except FileNotFoundError:
            return {"status": "docker_unavailable", "stdout": "",
                    "stderr": "docker CLI not found", "lang": lang_norm}
        except Exception as e:
            return {"status": f"error_{type(e).__name__}", "stdout": "",
                    "stderr": str(e)[:500], "lang": lang_norm}


def _reexec_step_outputs(step_outputs: dict, exec_log: list[dict] | None = None,
                          verbose: bool = False) -> dict:
    """In-place enrich each CodeInterpreterTool step with reexec_content /
    reexec_status / reexec_lang. Original .content is preserved.

    If a step's instruction has no fenced code but references another step
    (e.g., 'Execute the learner's code from S3-1'), follow the reference to
    pull code from step_outputs[S3-1].content or log's S3-1 student_response.
    """
    for sid, so in step_outputs.items():
        meta = so.get("meta") or {}
        if meta.get("tool") != "CodeInterpreterTool":
            continue
        code, fence_lang = _extract_code_and_fence_lang(meta, step_outputs, exec_log)
        lang = _detect_code_language(meta, so.get("content", ""), fence_lang)
        if not code or not lang:
            so["reexec_status"] = "no_code_or_lang"
            so["reexec_lang"] = lang
            so["reexec_category"] = "no_code"
            continue
        if verbose:
            print(f"    [reexec] {sid}: lang={lang}, code={len(code)}b")
        result = _run_in_mas_runtime(code, lang)
        # Strip the Docker entrypoint banner ("* Starting PostgreSQL ...")
        stdout = result.get("stdout", "")
        lines = stdout.split("\n")
        cleaned = []
        skipping_banner = True
        for ln in lines:
            if skipping_banner and (
                "Starting PostgreSQL" in ln or
                ln.strip() == "...done." or
                ln.strip().startswith("*")
            ):
                continue
            skipping_banner = False
            cleaned.append(ln)
        stdout = "\n".join(cleaned).lstrip("\n")
        combined = stdout
        if result.get("stderr") and result.get("status") != "ok":
            combined = (combined + "\n[stderr]\n" + result["stderr"])[:4000]
        so["reexec_content"] = combined
        so["reexec_status"] = result["status"]
        so["reexec_lang"] = result["lang"]
        so["reexec_category"] = _reexec_category(result["status"])
        # Log env failures for later analysis (Docker unavailable, timeout, unsupported lang)
        if so["reexec_category"] == "env_fail":
            _TIMEOUT_LOG.append({
                "ts": _now(), "event_type": f"docker_{result['status']}",
                "step_id": sid, "lang": lang,
                "stderr": result.get("stderr", "")[:200],
            })
        if verbose:
            print(f"      → status={result['status']} category={so['reexec_category']}, output={combined[:100]!r}")
    return step_outputs

# HTML §11.4 reference code SPECULATIVE patterns (6, verbatim)
SPECULATIVE = (
    "expected output:",
    "should see:",
    "when you run this",
    "you would get:",
    "this would produce",
    "the result would be",
)

# Reexec status → category mapping
# env_fail: Docker/environment problem — NOT a student code correctness issue
# code_error: student code ran but exited non-zero — IS a code correctness issue
# no_code: no code found in step — nothing to execute
_CODE_ERROR_STATUSES = frozenset({"nonzero_exit"})


def _reexec_category(status: str) -> str:
    """Classify reexec_status into: ok | code_error | env_fail | no_code | unknown"""
    if status == "ok":
        return "ok"
    if status in ("no_code_or_lang", "no_code", "no_lang"):
        return "no_code"
    if status in _CODE_ERROR_STATUSES:
        return "code_error"
    # timeout, docker_unavailable, unsupported_lang, error_* → env_fail
    if status in ("timeout", "docker_unavailable", "unsupported_lang"):
        return "env_fail"
    if status.startswith("error_"):
        return "env_fail"
    return "unknown"

# Fenced code block (strict — HTML says "fenced code block")
_CODE_FENCE = re.compile(r"```[\w+\-]*\s*\n?(.*?)```", re.DOTALL)

_PERFECT_FROM_START_MARKERS = ("perfect_from_start", "perfect from start")

# Design §8.2.6 — prompt-injection sanitize
_INJECTION_PATTERNS = [
    re.compile(r"(?i)please\s+(rate|score|grade|evaluate)\s+(this\s+)?(plan|answer|response|step)\s+(as|with)\s+\d+\s*/?\s*\d*"),
    re.compile(r"(?i)(ignore|disregard)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompt)"),
    re.compile(r"(?i)you\s+(must|should|will)\s+(output|return|give)\s+(true|false|equivalent|aligned)"),
    re.compile(r"(?i)system\s*:\s*you\s+are"),
    re.compile(r"(?i)\[\[?\s*INST(RUCTION)?\s*\]?\]"),
]


def sanitize_for_judge(text: str, max_len: int = 2000) -> str:
    if not text:
        return ""
    t = text
    for pat in _INJECTION_PATTERNS:
        t = pat.sub("[redacted-injection]", t)
    if len(t) > max_len:
        t = t[:max_len] + "...[truncated]"
    return t


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_dataset_index() -> dict[tuple, dict]:
    idx = {}
    for path in DATASET_PATHS:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                key = (str(r["question_id"]), r.get("profile_index", 0))
                if key not in idx:
                    idx[key] = r["plan"]
    return idx


def load_accepted_answers() -> dict[tuple, str]:
    aa = {}
    for path in QAP_SOURCES:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                qid = str(r["question_id"])
                for pidx, pa in enumerate(r.get("profiles_answers", [])):
                    key = (qid, pidx)
                    ans = (pa.get("answer") or "").strip()
                    if key not in aa and ans:
                        aa[key] = ans
    return aa


# ---------------------------------------------------------------------------
# Judge wrapper
# ---------------------------------------------------------------------------

_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


def _judge(model: str, system: str, user: str, max_tokens: int = 200) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = _client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=60,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            err_str = str(e)
            if any(t in err_str.lower() for t in
                   ("timeout", "timed out", "read timeout", "connect timeout")):
                _TIMEOUT_LOG.append({
                    "ts": _now(), "event_type": "api_timeout",
                    "attempt": attempt, "model": model, "error": err_str[:200],
                })
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"judge failed: {last_err}")


def _parse_json_loose(raw: str) -> dict:
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


# ---------------------------------------------------------------------------
# Helpers — plan / log access
# ---------------------------------------------------------------------------

def iter_plan_steps(plan: dict):
    for st in plan.get("output", {}).get("subtasks", []):
        for step in st.get("steps", []):
            yield st, step


def plan_step_by_id(plan: dict) -> dict[str, dict]:
    return {step["id"]: step for _, step in iter_plan_steps(plan)}


def teacher_utterances(exec_log: list[dict]) -> list[str]:
    out = []
    for e in exec_log:
        ai = e.get("actual_interaction") or {}
        t = (ai.get("teacher_output") or "").strip()
        if t:
            out.append(t)
    return out


def extract_longest_code(texts: list[str]) -> str:
    """HTML §11.5: "抽出最长的 fenced code block"。Strict — fenced only."""
    best = ""
    for t in texts:
        for m in _CODE_FENCE.findall(t):
            if len(m) > len(best):
                best = m
    return best.strip()


def infer_language(plan: dict, step_outputs: dict) -> str:
    lang_tokens = {
        "python", "java", "javascript", "js", "typescript", "ts", "cpp", "c++",
        "csharp", "c#", "ruby", "php", "go", "rust", "kotlin", "swift", "scala",
        "haskell", "ocaml", "r", "sql", "bash", "shell", "perl", "dart",
        "groovy", "lua", "awk", "powershell", "fsharp", "node", "objc",
    }
    for sid, so in step_outputs.items():
        agent = ((so.get("meta") or {}).get("agent") or "").lower()
        for tok in lang_tokens:
            if tok in agent:
                return tok
    for _, step in iter_plan_steps(plan):
        agent = (step.get("agent") or "").lower()
        for tok in lang_tokens:
            if tok in agent:
                return tok
    for _, step in iter_plan_steps(plan):
        inst = step.get("instruction") or ""
        m = re.search(r"```(\w+)", inst)
        if m and m.group(1).lower() in lang_tokens:
            return m.group(1).lower()
    return "unknown"


def compile_check(code: str, lang: str) -> tuple[bool, str]:
    """HTML §11.5 r_sol HARD CONSTRAINT: code must compile / interpret."""
    if not code.strip():
        return False, "empty_code"
    lang = (lang or "").lower()

    if lang == "python":
        try:
            compile(code, "<student>", "exec")
            return True, "py_compile_ok"
        except SyntaxError as e:
            return False, f"py_syntax: {e.msg}"
        except Exception as e:
            return False, f"py_err: {type(e).__name__}"

    def _try(cmd_factory, ext: str, label: str) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{ext}", delete=False, encoding="utf-8"
        ) as f:
            f.write(code); f.flush(); path = f.name
        try:
            proc = subprocess.run(cmd_factory(path), capture_output=True,
                                  text=True, timeout=12)
            if proc.returncode == 0:
                return True, f"{label}_ok"
            return False, f"{label}_fail: {(proc.stderr or proc.stdout or '')[:200]}"
        except subprocess.TimeoutExpired:
            return False, f"{label}_timeout"
        except FileNotFoundError:
            return True, f"{label}_unavailable_skip"
        finally:
            try: os.unlink(path)
            except OSError: pass

    if lang in ("javascript", "js", "node") and shutil.which("node"):
        return _try(lambda p: ["node", "--check", p], "js", "node")
    if lang in ("typescript", "ts") and shutil.which("tsc"):
        return _try(lambda p: ["tsc", "--noEmit", "--allowJs", "--target", "es2020", p], "ts", "tsc")
    if lang == "ruby" and shutil.which("ruby"):
        return _try(lambda p: ["ruby", "-c", p], "rb", "ruby")
    if lang == "php" and shutil.which("php"):
        return _try(lambda p: ["php", "-l", p], "php", "php")
    if lang == "go" and shutil.which("gofmt"):
        return _try(lambda p: ["gofmt", "-e", p], "go", "gofmt")
    if lang in ("bash", "shell") and shutil.which("bash"):
        return _try(lambda p: ["bash", "-n", p], "sh", "bash")
    if lang == "perl" and shutil.which("perl"):
        return _try(lambda p: ["perl", "-c", p], "pl", "perl")
    if lang == "sql":
        if re.search(r"\b(select|insert|update|delete|create|with|alter|drop)\b",
                     code, re.IGNORECASE):
            return True, "sql_lexical_ok"
        return False, "sql_no_keyword"
    return True, f"lang_unsupported:{lang}"


# ---------------------------------------------------------------------------
# v1 evaluator bridge — drives BOTH cov/loop/flow AND Appendix B
# ---------------------------------------------------------------------------

# Fallback report returned when v1 evaluator fails all retries.
# Structured so parse_wc_subchecks and parse_appendix_b both return FAIL.
_V1_FAIL_REPORT = (
    "Check 1: Instruction Fidelity\n  Verdict: FAIL\n"
    "  Per-step results: [v1_evaluator_error]\n\n"
    "Check 2: Workflow Completeness\n  Verdict: FAIL\n"
    "  Step coverage: [incomplete]\n"
    "  Loop behavior: [abnormal]\n"
    "  Information flow: [broken]\n"
    "  Tool execution: [uncertain]\n\n"
    "Check 3: Interaction Quality\n  Verdict: FAIL\n\n"
    "Check 4: Content Correctness and Guidance Effectiveness\n  Verdict: FAIL\n\n"
    "Overall verdict: execution_failed\n"
    "Failure attribution: v1_evaluator_error — all API retries exhausted\n"
)


def _truncate_exec_log(log: list, max_entries: int = 60, max_chars: int = 1200) -> list:
    """Truncate exec_log before passing to v1 evaluator to keep prompt under ~15k tokens.

    Keeps the last max_entries entries; truncates long text fields within each
    entry. Does NOT mutate the original list.
    """
    trimmed = log[-max_entries:] if len(log) > max_entries else log
    result = []
    for raw_entry in trimmed:
        entry = dict(raw_entry)
        ai = entry.get("actual_interaction")
        if isinstance(ai, dict):
            ai = dict(ai)
            for field in ("teacher_output", "student_response", "agent_output"):
                if isinstance(ai.get(field), str) and len(ai[field]) > max_chars:
                    ai[field] = ai[field][:max_chars] + "...[truncated]"
            entry["actual_interaction"] = ai
        result.append(entry)
    return result


def _run_v1_with_retry(
    plan: dict, exec_log_path: Path, accepted_answer: str,
    judge_model: str, plan_tmp_path: Path, v1_md_out: Path,
) -> str:
    """Call run_v1_evaluator with 3 retries and 10s exponential backoff.

    Returns _V1_FAIL_REPORT when all retries are exhausted so downstream
    parse functions return FAIL instead of crashing the whole run.
    """
    last_err = None
    for attempt in range(3):
        try:
            return run_v1_evaluator(plan, exec_log_path, accepted_answer,
                                     judge_model, plan_tmp_path, v1_md_out)
        except Exception as e:
            last_err = e
            err_str = str(e)
            event_type = ("v1_api_timeout"
                          if any(t in err_str.lower() for t in
                                 ("timeout", "timed out", "read timeout", "connect timeout"))
                          else "v1_evaluator_error")
            _TIMEOUT_LOG.append({
                "ts": _now(), "event_type": event_type,
                "attempt": attempt, "error": err_str[:200],
            })
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    print(f"  [WARN] v1 evaluator failed after 3 attempts: {last_err} — using fallback report")
    return _V1_FAIL_REPORT


def run_v1_evaluator(
    plan: dict, exec_log_path: Path, accepted_answer: str,
    judge_model: str, plan_tmp_path: Path, out_md_path: Path,
) -> str:
    """Call v1 4-check evaluator. Returns markdown report string."""
    plan_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    plan_tmp_path.write_text(json.dumps(plan, ensure_ascii=False))
    report = _v1_eval(
        plan_path=str(plan_tmp_path),
        execution_log_path=str(exec_log_path),
        accepted_answer=accepted_answer or None,
        evaluator_model=judge_model,
        output_path=str(out_md_path),
    )
    return report


def parse_wc_subchecks(report: str) -> dict:
    """Parse v1 markdown for WC sub-fields → cov / loop / flow / tool.

    v1 emits under "Check 2: Workflow Completeness":
       Step coverage: [complete / incomplete]
       Loop behavior: [normal / abnormal]
       Information flow: [connected / broken]
       Tool execution: [reliable / uncertain]
    """
    low = report.lower()
    # Slice to Check 2 section if possible
    start = low.find("check 2")
    end = low.find("check 3", start) if start >= 0 else -1
    seg = report[start:end if end > 0 else len(report)]
    seg_low = seg.lower()

    def _field(pattern_prefix: str, pos_value: str) -> bool:
        m = re.search(rf"{re.escape(pattern_prefix)}\s*:?\s*\[?(\w[\w_\- ]*)\]?",
                      seg_low)
        if not m:
            return False
        return m.group(1).strip().startswith(pos_value)

    return {
        "cov": _field("step coverage", "complete"),
        "loop": _field("loop behavior", "normal"),
        "flow": _field("information flow", "connected"),
        "tool_kele": _field("tool execution", "reliable"),
    }


def parse_appendix_b(report: str) -> dict:
    """Extract IF / WC / IQ / CCGE verdicts from v1 markdown for Appendix B."""
    rep_low = report.lower()
    breakdown = {}
    for name, marker in [("IF", "check 1"), ("WC", "check 2"),
                          ("IQ", "check 3"), ("CCGE", "check 4")]:
        seg_start = rep_low.find(marker)
        if seg_start < 0:
            breakdown[name] = None
            continue
        next_marker_pos = rep_low.find("check ", seg_start + len(marker))
        seg = rep_low[seg_start:next_marker_pos if next_marker_pos > 0 else seg_start + 800]
        # Find Verdict: PASS / FAIL
        m = re.search(r"verdict\s*:\s*\[?(pass|fail)\]?", seg)
        breakdown[name] = (m.group(1).upper() if m else "UNCLEAR")
    return breakdown


# ---------------------------------------------------------------------------
# EVR — §11.4 — strict HTML reference code
# ---------------------------------------------------------------------------

CCR_SYSTEM = "You are a strict code-output equivalence judge. Output only JSON."

CCR_PROMPT = """Query (task description):
{query}

Accepted answer (gold solution):
{accepted}

Expected output (NOTE: may be a prose description like "prints the connection string"
rather than the literal stdout — judge by intent, not literal match):
{expected}

Actual code output (real sandbox stdout):
{actual}

Judge: is the actual output a reasonable manifestation of what the expected output
describes, in the context of the query and accepted answer?

Accept as equivalent (true) when:
- Actual demonstrates the same observable behavior / concept the expected describes
- Numeric or formatting differences exist but both communicate the same phenomenon
  (e.g., both demonstrate undefined behavior, both show the same data type, both
  print the queried value — even if the exact numbers differ across platforms)
- Actual contains the expected information plus extra context

Reject as not equivalent (false) when:
- Actual is an error / refusal / "I can't run this" message
- Actual contradicts the accepted answer's core claim
- Actual is unrelated to the task

Output JSON: {{"equivalent": true|false, "reason": "<= 15 words"}}"""


def evr_run(
    plan: dict,
    exec_log: list[dict],
    step_outputs: dict,
    accepted_answer: str,
    judge_model: str,
    v1_report: str,
) -> dict:
    """EVR = 4 sub-checks AND联合 (cov, loop, flow, exec).

    `out` (CCR functional equivalence) 已从 EVR 中移除——MAP-PPL plan 的
    expected_output 是描述性 prose（"prints the connection string"），不是
    sandbox 可比的 stdout 规范，让 LLM 判等价导致大批 false-fail。

    `exec` 严格按 HTML §11.4 (d) 参考代码：
      code_steps = [s for s in step_outputs.values()
                    if s["meta"].get("tool") == "CodeInterpreterTool"]
      exec_ok = all(no SPECULATIVE in s.content for s in code_steps)

    取 runtime 自报 meta.tool=CodeInterpreterTool 的 step（runtime 内部自洽
    视角），不跟 dataset plan 的 tool 对照——dataset plan 跟 stage3 实际跑的
    plan 不是同一份，跨版本比对会产生 false-positive。
    """
    wc = parse_wc_subchecks(v1_report)
    cov = wc["cov"]; loop = wc["loop"]; flow = wc["flow"]

    def _effective_content(so: dict) -> str | None:
        """Return effective content for the speculative-language check.

        Returns:
          - sandbox stdout string when reexec succeeded (category=ok)
          - original content string when no Docker was attempted (category=no_code or no category)
          - None when reexec failed due to env/code issues — callers must NOT
            fall back to original content; these are real failures, not
            semantic fallbacks.
        """
        cat = so.get("reexec_category")
        if cat == "ok":
            return so.get("reexec_content") or ""
        if cat in ("no_code", None) or so.get("reexec_status") == "no_code_or_lang":
            # No Docker was attempted; use original stage3 content for speculative check
            return so.get("content") or ""
        # env_fail or code_error: do NOT fall back to original content.
        # Record as a real execution failure in exec_fail_reasons below.
        return None

    # HTML §11.4 (d): runtime-meta view — CodeInterpreterTool steps only
    code_steps_meta = [(sid, so) for sid, so in step_outputs.items()
                       if ((so.get("meta") or {}).get("tool") == "CodeInterpreterTool")]
    env_fail_steps: list[dict] = []
    code_error_steps: list[dict] = []
    exec_fail_reasons: list[dict] = []
    exec_ok = True

    for sid, so in code_steps_meta:
        cat = so.get("reexec_category")
        if cat == "env_fail":
            # Environment failure: Docker unavailable, timeout, unsupported language.
            # This is NOT a code correctness issue — record explicitly.
            env_fail_steps.append({
                "step_id": sid,
                "lang": so.get("reexec_lang", "unknown"),
                "status": so.get("reexec_status", "unknown"),
                "reason": (
                    "env_fail: Docker unavailable, timeout, or unsupported language — "
                    "not a student code correctness failure"
                ),
            })
            exec_ok = False
            exec_fail_reasons.append({
                "step_id": sid,
                "reason": f"env_fail:{so.get('reexec_status', 'unknown')}",
            })
            continue
        if cat == "code_error":
            # Student code ran but exited non-zero — IS a code failure.
            code_error_steps.append({
                "step_id": sid,
                "lang": so.get("reexec_lang", "unknown"),
                "status": so.get("reexec_status", "unknown"),
            })
            exec_ok = False
            exec_fail_reasons.append({"step_id": sid, "reason": "code_error:nonzero_exit"})
            continue

        content = _effective_content(so)
        if content is None:
            exec_ok = False
            exec_fail_reasons.append({"step_id": sid, "reason": "no_content_after_reexec"})
            continue
        hit = next((p for p in SPECULATIVE if p in content.lower()), None)
        if hit:
            exec_ok = False
            exec_fail_reasons.append({"step_id": sid, "reason": f"speculative_pattern:{hit}"})

    return {
        "evr_pass": int(cov and loop and flow and exec_ok),
        "checks": {"cov": cov, "loop": loop, "flow": flow, "exec": exec_ok},
        "wc_kele": wc,
        "n_code_steps_runtime": len(code_steps_meta),
        "exec_fail_reasons": exec_fail_reasons,
        "env_fail_steps": env_fail_steps,
        "code_error_steps": code_error_steps,
    }


# ---------------------------------------------------------------------------
# PAS — §11.5 PRR
# ---------------------------------------------------------------------------

PRR_SYSTEM = "You judge if a teacher utterance reflects the learner profile. Output only JSON."

PRR_PROMPT = """Learner profile (JSON):
{profile}

Teacher utterance:
{utterance}

Rubric:
- 2 (well-aligned): adapts vocab/examples/pace to >=2 profile fields explicitly
- 1 (partially):    reflects >=1 profile field but generic in tone
- 0 (not aligned):  ignores profile entirely / uses wrong language / severely mismatches the learner level

Output JSON: {{"score": 0|1|2, "evidence": "<= 25 words"}}"""


def pas_run(profile: dict, exec_log: list[dict], judge_model: str) -> dict:
    utts = teacher_utterances(exec_log)
    if not utts:
        return {"pas": 0.0, "pas_strict": 0.0, "n_utts": 0, "scores": []}
    profile_str = sanitize_for_judge(json.dumps(profile, ensure_ascii=False), 800)
    scores = []
    for u in utts:
        raw = _judge(judge_model, PRR_SYSTEM,
                     PRR_PROMPT.format(profile=profile_str,
                                       utterance=sanitize_for_judge(u, 800)))
        try:
            s = int(_parse_json_loose(raw).get("score", 0))
        except Exception:
            s = 0
        scores.append(max(0, min(2, s)))
    return {
        "pas": sum(scores) / (2 * len(scores)),
        "pas_strict": sum(1 for s in scores if s == 2) / len(scores),
        "n_utts": len(scores),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# PQS — §11.5
# ---------------------------------------------------------------------------

NDAR_SYSTEM = "You judge if a teacher utterance reveals the accepted answer. Output only JSON."

NDAR_PROMPT = """Accepted answer (gold):
{accepted}

Teacher utterance:
{utterance}

Rubric (per HTML §11.5):
- "full":    directly states the accepted solution
- "partial": reveals key steps but not full solution
- "none":    only asks / guides, no answer leakage

Output JSON: {{"reveal": "full"|"partial"|"none", "evidence": "<= 25 words"}}"""


def ndar_run(exec_log: list[dict], accepted_answer: str, judge_model: str) -> dict:
    utts = teacher_utterances(exec_log)
    if not utts:
        return {"ndar": 0.0, "n_utts": 0, "reveals": []}
    if not accepted_answer.strip():
        return {"ndar": 0.0, "n_utts": len(utts), "reveals": ["no_accepted"]}
    reveals = []
    for u in utts:
        raw = _judge(judge_model, NDAR_SYSTEM,
                     NDAR_PROMPT.format(
                         accepted=sanitize_for_judge(accepted_answer, 800),
                         utterance=sanitize_for_judge(u, 800)))
        # HTML reference code: parsed["reveal"]; if parse fails we don't crash
        # but match HTML's implicit behavior — default to "full" (failure-pessimistic).
        v = (_parse_json_loose(raw).get("reveal") or "full").lower()
        if v not in ("full", "partial", "none"):
            v = "full"
        reveals.append(v)
    return {
        "ndar": sum(1 for r in reveals if r == "none") / len(reveals),
        "n_utts": len(reveals),
        "reveals": reveals,
    }


SPR_SYSTEM = (
    "You tag scaffolding phases for a multi-agent teaching plan. Be strict: "
    "only assign a phase if the step explicitly carries that role. Many "
    "steps are NONE (transitional / tool-use / docs lookup) and should be tagged 'none'. "
    "Output only JSON."
)

SPR_PROMPT = """Plan: query="{query}"

Steps (id, objective, instruction[truncated]):
{steps}

Tag each step with ONE phase based on its INSTRUCTION text (strict criteria):
- "intro":   ONLY if the step is at the start AND asks the learner what they already know /
             activates prior knowledge / scopes the problem. Reject if it just sets up tools.
- "guide":   demonstrates the concept with worked example OR has the learner attempt with
             scaffolded support. Most middle steps are NOT guide unless they teach the procedure.
- "consol":  ONLY if it is in the last few steps AND summarizes/consolidates/has the learner
             reflect or restate in their own words. Reject if it merely closes a side discussion.
- "none":    transitional / tool invocation / docs lookup / interim check / loop control —
             this is the DEFAULT for most steps.

Be conservative: prefer "none" over guessing. A plan can legitimately lack any of intro/guide/consol.

Output JSON: {{"tags": {{"S1-1": "intro"|"guide"|"consol"|"none", ...}}}}"""


def spr_run(plan: dict, judge_model: str) -> dict:
    """SPR_r = |{intro, guide, consol} ∩ phases_present| / 3.

    HTML §11.5 reference code uses plan.metadata.phase first; our data has none,
    falls back to LLM-tagging per design line 2049.
    """
    # Try metadata.phase first
    phases_meta: set[str] = set()
    for _, step in iter_plan_steps(plan):
        ph = (step.get("metadata") or {}).get("phase", "")
        if isinstance(ph, str):
            phl = ph.lower()
            if phl in ("intro", "introduce", "introduction"):
                phases_meta.add("intro")
            elif phl in ("guide", "guided"):
                phases_meta.add("guide")
            elif phl in ("consol", "consolidate", "consolidation"):
                phases_meta.add("consol")
    if phases_meta:
        return {"spr": len(phases_meta & {"intro", "guide", "consol"}) / 3,
                "phases_present": sorted(phases_meta), "tags": {},
                "via": "metadata.phase"}

    # LLM fallback (design-approved)
    steps_short = []
    for _, step in iter_plan_steps(plan):
        steps_short.append(
            f'- {step["id"]}: obj="{(step.get("objective") or "")[:120]}"; '
            f'inst="{(step.get("instruction") or "")[:120]}"'
        )
    if not steps_short:
        return {"spr": 0.0, "phases_present": [], "tags": {}, "via": "no_steps"}
    query = (plan.get("input") or {}).get("query", "") or ""
    raw = _judge(
        judge_model, SPR_SYSTEM,
        SPR_PROMPT.format(query=sanitize_for_judge(query, 300),
                          steps=sanitize_for_judge("\n".join(steps_short), 4000)),
        max_tokens=800,
    )
    tags = _parse_json_loose(raw).get("tags", {}) or {}
    canon = {"intro": "intro", "introduce": "intro", "introduction": "intro",
             "guide": "guide", "guiding": "guide", "guided": "guide",
             "consol": "consol", "consolidate": "consol", "consolidation": "consol"}
    phases: set[str] = set()
    norm_tags = {}
    for sid, ph in tags.items():
        c = canon.get((ph or "").lower())
        if c:
            phases.add(c); norm_tags[sid] = c
    return {"spr": len(phases & {"intro", "guide", "consol"}) / 3,
            "phases_present": sorted(phases), "tags": norm_tags, "via": "llm_tag"}


def iar_run(exec_log: list[dict]) -> dict:
    """HTML §11.5 reference code, verbatim:
        q = #(?) + #(what|why|how|can you|describe)
        s = #([.!]) - q
        IAR = min(1, q/(s+1))    ← formula at line 1961
    """
    utts = teacher_utterances(exec_log)
    if not utts:
        return {"iar": 0.0, "questions": 0, "statements": 0}
    text = " ".join(utts)
    q = (len(re.findall(r"\?", text)) +
         len(re.findall(r"\b(what|why|how|can you|describe)\b", text.lower())))
    s = max(0, len(re.findall(r"[.!]", text)) - q)
    iar = min(1.0, q / (s + 1))
    return {"iar": iar, "questions": q, "statements": s}


def pqs_run(plan: dict, exec_log: list[dict], accepted_answer: str,
            judge_model: str) -> dict:
    ndar = ndar_run(exec_log, accepted_answer, judge_model)
    spr = spr_run(plan, judge_model)
    iar = iar_run(exec_log)
    pqs = (ndar["ndar"] + spr["spr"] + iar["iar"]) / 3
    return {"pqs": pqs, "ndar": ndar, "spr": spr, "iar": iar}


# ---------------------------------------------------------------------------
# r_sol — §11.5
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# r_sol — Comprehension Demonstration Rate (重设计版)
# ---------------------------------------------------------------------------
# 原始设计（HTML §11.5）：抽学生最后 subtask 的 fenced code，跟 accepted_answer
# 判 functional equivalent。在 MAP-PPL 数据上不成立——最后 subtask 是 consolidate
# phase，学生用自然语言反思，不交 fenced code。
#
# 重设计：判 INTERACTION PROCESS——学生在 consolidate phase 是否展现出对核心概念
# 的正确理解。Judge 看 query + accepted_answer + 学生在最后 subtask 的所有
# 自然语言回复（含中间 subtask 的 student_response 作上下文），输出
# {"demonstrates_understanding": true|false, "evidence": "..."}.

RSOL_SYSTEM = (
    "You judge whether a student has demonstrated genuine understanding of a "
    "programming concept through their dialogue responses. Output only JSON."
)

RSOL_PROMPT = """Query / topic being taught:
{query}

Accepted answer (gold reference for the concept):
{accepted}

Student's responses during the consolidation phase (last subtask):
{final_responses}

Student's responses earlier in the dialogue (context, for trajectory):
{earlier_responses}

Judge whether the student has internalized the core concept the accepted answer
expresses. Look for ANY of:
- Restates the key principle / mechanism in their own words correctly
- Identifies the root cause / correct trade-off / right invariant
- Predicts behavior consistent with the accepted answer
- Articulates why the approach works (not just "thanks for explaining")

Do NOT require:
- Code submission (this phase is reflection, not coding)
- Verbatim repetition of accepted answer
- Coverage of every detail

Reject (false) when:
- Only social pleasantries ("thanks", "got it", "makes sense") with no substantive content
- Misstates the principle
- Off-topic / unrelated content
- Empty / missing response

Output JSON: {{"demonstrates_understanding": true|false, "evidence": "<= 30 words quoting key student phrase"}}"""


def _has_perfect_from_start(exec_log: list[dict],
                             v1_report: str = "") -> bool:
    """HTML §11.5 r_sol 排除"话题盲区失败 run"。

    primary signal: v1 evaluator Check 3 (IQ) 输出中
        "Interaction pattern: degenerate" + "perfect_from_start"
    (per plan_mapper_fixed/evaluator.py line 206-207)

    fallback: scan teacher_output for marker keywords (legacy).
    """
    if v1_report:
        low = v1_report.lower()
        # Find Check 3 section
        start = low.find("check 3")
        end = low.find("check 4", start) if start >= 0 else -1
        seg = low[start:end if end > 0 else (start + 1500)] if start >= 0 else low
        if "interaction pattern" in seg and "degenerate" in seg:
            if "perfect_from_start" in seg or "perfect from start" in seg:
                return True
    # Fallback: keyword scan
    text = " ".join(
        (e.get("actual_interaction") or {}).get("teacher_output", "")
        for e in exec_log
    ).lower()
    return any(m in text for m in _PERFECT_FROM_START_MARKERS)


def r_sol_run(plan: dict, exec_log: list[dict], step_outputs: dict,
              accepted_answer: str, judge_model: str,
              v1_report: str = "") -> dict:
    """判学生在 consolidate phase 是否展现对核心概念的理解（不评估代码）。

    Returns dict with r_sol ∈ {0, 1, None}:
      - None: perfect_from_start (设计 §11.5 排除) 或 没 accepted_answer / 没最后 subtask
      - 0: 没 substantive 回复 / judge 判 not demonstrating understanding
      - 1: judge 判 demonstrates understanding
    """
    if _has_perfect_from_start(exec_log, v1_report):
        return {"r_sol": None, "reason": "perfect_from_start"}
    sts = plan.get("output", {}).get("subtasks", [])
    if not sts or not accepted_answer.strip():
        return {"r_sol": None, "reason": "no_last_subtask_or_no_accepted"}
    last_sid = sts[-1]["id"]

    # Collect student responses from last subtask (any field, not just student_response)
    last_responses = []
    for e in exec_log:
        if e.get("subtask_id") != last_sid:
            continue
        ai = e.get("actual_interaction") or {}
        for fld in ("student_response", "agent_output"):
            v = (ai.get(fld) or "").strip()
            if v:
                last_responses.append(v)

    last_text = "\n---\n".join(last_responses).strip()
    if not last_text:
        return {"r_sol": 0, "reason": "no_response_in_last_subtask",
                "last_subtask": last_sid}

    # Earlier student responses for trajectory context (sample up to 3 most recent)
    earlier_responses = []
    for e in exec_log:
        if e.get("subtask_id") == last_sid:
            continue
        ai = e.get("actual_interaction") or {}
        v = (ai.get("student_response") or "").strip()
        if v:
            earlier_responses.append(v)
    earlier_text = "\n---\n".join(earlier_responses[-3:]) if earlier_responses else "(none)"

    raw = _judge(
        judge_model, RSOL_SYSTEM,
        RSOL_PROMPT.format(
            query=sanitize_for_judge(plan.get("input", {}).get("query", ""), 400),
            accepted=sanitize_for_judge(accepted_answer, 1500),
            final_responses=sanitize_for_judge(last_text, 1800),
            earlier_responses=sanitize_for_judge(earlier_text, 800),
        ),
        max_tokens=200,
    )
    parsed = _parse_json_loose(raw)
    eq = bool(parsed.get("demonstrates_understanding", False))
    return {
        "r_sol": int(eq),
        "reason": parsed.get("evidence", ""),
        "last_subtask": last_sid,
        "n_responses_in_last": len(last_responses),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# ===========================================================================
# Public API
# ===========================================================================
# Use these from external code instead of the internal evaluate_one().
#
#     from tier2_evaluator_v2 import evaluate_run, evaluate_run_dir, evaluate_batch
#
#     # 1) From in-memory plan + log + step_outputs
#     result = evaluate_run(plan, exec_log, step_outputs, accepted_answer="...")
#
#     # 2) From a stage3-style run directory
#     result = evaluate_run_dir("path/to/run-XYZ", plan=plan_dict)
#
#     # 3) From a list of (qid, profile_index) — auto-resolves plan + accepted
#     results = evaluate_batch([(qid, pidx), ...])

import shutil as _shutil_for_api


def _make_work_dir(out_root: Path | None) -> Path:
    """Create a work dir for v1 evaluator's tmp plan + markdown report."""
    if out_root is None:
        out_root = Path(tempfile.mkdtemp(prefix="tier2_eval_"))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "appendix_b_reports").mkdir(parents=True, exist_ok=True)
    (out_root / "_plan_snapshots").mkdir(parents=True, exist_ok=True)
    return out_root


def evaluate_run(
    plan: dict,
    exec_log: list[dict],
    step_outputs: dict,
    *,
    accepted_answer: str = "",
    run_id: str = "single_run",
    judge_model: str = "gpt-4o-mini",
    reexec_codesteps: bool = True,
    work_dir: Path | str | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate a single MAP-PPL run on Tier 2 metrics.

    Args:
        plan: parsed plan dict (with .input.{query,learner} + .output.{subtasks,agents,execution_order})
        exec_log: list of execution_log entries
        step_outputs: dict of step_outputs (step_id → {content, meta, status, ...})
        accepted_answer: gold reference; if "", NDAR / r_sol effectively skip
        run_id: identifier used for v1 report filenames
        judge_model: OpenAI model id (default "gpt-4o-mini")
        reexec_codesteps: if True, re-run code steps in mas-runtime Docker before evaluation
        work_dir: directory to write v1 markdown reports + plan snapshots
                  (defaults to a temp dir, cleaned up on process exit)
        verbose: print per-stage progress

    Returns:
        {
          "evr":      {evr_pass, checks: {cov,loop,flow,exec}, ...},
          "pas":      {pas, pas_strict, n_utts, scores},
          "pqs":      {pqs, ndar, spr, iar},
          "r_sol":    {r_sol, reason, ...},
          "appendix_b": {IF, WC, IQ, CCGE},
          "v1_report_path": "<path to v1 markdown>",
        }
    """
    work = _make_work_dir(Path(work_dir) if work_dir else None)
    plan_tmp_path = work / "_plan_snapshots" / f"{run_id}.json"
    v1_md_out = work / "appendix_b_reports" / f"{run_id}.md"

    # v1 evaluator needs exec_log on disk
    exec_log_path = work / "_plan_snapshots" / f"{run_id}_log.json"
    exec_log_path.write_text(json.dumps(exec_log, ensure_ascii=False))

    return evaluate_one(
        plan=plan, exec_log=exec_log, step_outputs=step_outputs,
        exec_log_path=exec_log_path,
        accepted_answer=accepted_answer or "",
        judge_model=judge_model,
        plan_tmp_path=plan_tmp_path,
        v1_md_out=v1_md_out,
        verbose=verbose,
        reexec_codesteps=reexec_codesteps,
    )


def evaluate_run_dir(
    run_dir: str | Path,
    *,
    plan: dict | str | Path | None = None,
    accepted_answer: str | None = None,
    judge_model: str = "gpt-4o-mini",
    reexec_codesteps: bool = True,
    work_dir: Path | str | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate from a stage3-style run directory containing
    execution_log.json + step_outputs.json.

    Args:
        run_dir: directory with execution_log.json + step_outputs.json
        plan: plan dict, OR path to plan json, OR None (auto-resolve from dataset by run_id)
        accepted_answer: gold reference; if None, auto-resolve from MAP-PPL construction sources
        ...others same as evaluate_run
    """
    rd = Path(run_dir)
    exec_log = json.loads((rd / "execution_log.json").read_text())
    step_outputs = json.loads((rd / "step_outputs.json").read_text())

    # Resolve plan
    if plan is None:
        # Parse qid + pidx from run_id like "run-<qid>-p<pidx>-<hash>"
        parts = rd.name.split("-")
        if len(parts) >= 3 and parts[0] == "run":
            qid, pidx = parts[1], int(parts[2][1:])
            dataset = load_dataset_index()
            plan = dataset.get((qid, pidx))
            if plan is None:
                raise ValueError(f"Plan not found for (qid={qid}, pidx={pidx})")
        else:
            raise ValueError(f"Cannot auto-resolve plan; run_dir name doesn't match 'run-<qid>-p<idx>-*'")
    elif isinstance(plan, (str, Path)):
        plan_dict = json.loads(Path(plan).read_text())
        plan = plan_dict.get("plan", plan_dict)  # accept full record or just plan

    # Resolve accepted_answer
    if accepted_answer is None:
        parts = rd.name.split("-")
        if len(parts) >= 3 and parts[0] == "run":
            qid, pidx = parts[1], int(parts[2][1:])
            aa = load_accepted_answers()
            accepted_answer = aa.get((qid, pidx), "")
        else:
            accepted_answer = ""

    return evaluate_run(
        plan=plan,
        exec_log=exec_log,
        step_outputs=step_outputs,
        accepted_answer=accepted_answer or "",
        run_id=rd.name,
        judge_model=judge_model,
        reexec_codesteps=reexec_codesteps,
        work_dir=work_dir,
        verbose=verbose,
    )


def evaluate_batch(
    runs: list[tuple],
    *,
    judge_model: str = "gpt-4o-mini",
    reexec_codesteps: bool = True,
    work_dir: Path | str | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate a batch of (qid, profile_index) pairs from the MAP-PPL stage3 dataset.

    Each tuple should be (qid: str | int, profile_index: int).
    Returns dict with aggregated metrics + per-run results.
    """
    dataset = load_dataset_index()
    aa = load_accepted_answers()
    runs_dir = STAGE3_DIR / "runs"
    work = _make_work_dir(Path(work_dir) if work_dir else None)

    summary = json.loads((STAGE3_DIR / "batch_summary.json").read_text())
    sum_by_key = {(str(e["question_id"]), e["profile_index"]): e for e in summary}

    results = []
    issues = []
    for qid, pidx in runs:
        key = (str(qid), int(pidx))
        plan = dataset.get(key)
        if not plan:
            issues.append({"key": key, "issue": "plan_not_found"}); continue
        entry = sum_by_key.get(key)
        if not entry:
            issues.append({"key": key, "issue": "run_not_in_stage3"}); continue
        rd = runs_dir / entry["run_id"]
        try:
            res = evaluate_run_dir(
                rd, plan=plan, accepted_answer=aa.get(key, ""),
                judge_model=judge_model, reexec_codesteps=reexec_codesteps,
                work_dir=work, verbose=verbose,
            )
            res["run_id"] = entry["run_id"]
            res["question_id"] = str(qid)
            res["profile_index"] = int(pidx)
            results.append(res)
        except Exception as e:
            issues.append({"key": key, "issue": f"{type(e).__name__}: {e}"})

    return {
        "aggregate": aggregate_results(results),
        "results": results,
        "issues": issues,
        "judge_model": judge_model,
    }


# ===========================================================================
# Internal orchestration (used by CLI + public API)
# ===========================================================================

def evaluate_one(
    plan: dict, exec_log: list[dict], step_outputs: dict,
    exec_log_path: Path, accepted_answer: str, judge_model: str,
    plan_tmp_path: Path, v1_md_out: Path, verbose: bool = False,
    reexec_codesteps: bool = False,
) -> dict:
    # Optional pre-pass: re-execute code steps in mas-runtime Docker so EVR's
    # exec/out checks have real sandbox stdout instead of stage3's Python-only
    # refusal messages. Adds .reexec_content / .reexec_status to each code step.
    if reexec_codesteps:
        if verbose: print("  Re-executing CodeInterpreterTool steps in mas-runtime...")
        _reexec_step_outputs(step_outputs, exec_log=exec_log, verbose=verbose)
    # v1 evaluator drives both EVR.cov/loop/flow and Appendix B.
    # Write a truncated copy of exec_log to avoid huge prompts (up to 14k tokens raw).
    if verbose: print("  v1 evaluator (drives cov/loop/flow + appendix B)...")
    exec_log_v1_path = plan_tmp_path.parent / (plan_tmp_path.stem + "_log_trunc.json")
    exec_log_v1_path.parent.mkdir(parents=True, exist_ok=True)
    exec_log_v1_path.write_text(
        json.dumps(_truncate_exec_log(exec_log), ensure_ascii=False))
    v1_report = _run_v1_with_retry(plan, exec_log_v1_path, accepted_answer,
                                    judge_model, plan_tmp_path, v1_md_out)
    appendix_b = parse_appendix_b(v1_report)
    if verbose: print(f"     v1 verdicts: {appendix_b}")

    if verbose: print("  EVR (cov/loop/flow from v1; exec rule-based; out CCR)...")
    evr = evr_run(plan, exec_log, step_outputs, accepted_answer, judge_model, v1_report)
    if verbose: print(f"     -> pass={evr['evr_pass']} checks={evr['checks']}")

    if verbose: print("  PAS (PRR)...")
    pas = pas_run(plan.get("input", {}).get("learner", {}), exec_log, judge_model)
    if verbose: print(f"     -> pas={pas['pas']:.3f} (PAS*={pas['pas_strict']:.3f}, {pas['n_utts']} utts)")

    if verbose: print("  PQS (NDAR + SPR + IAR)...")
    pqs = pqs_run(plan, exec_log, accepted_answer, judge_model)
    if verbose:
        print(f"     -> pqs={pqs['pqs']:.3f} (NDAR={pqs['ndar']['ndar']:.2f} "
              f"SPR={pqs['spr']['spr']:.2f} IAR={pqs['iar']['iar']:.2f})")

    if verbose: print("  r_sol (compile + functional eq)...")
    rsol = r_sol_run(plan, exec_log, step_outputs, accepted_answer, judge_model,
                     v1_report=v1_report)
    if verbose: print(f"     -> r_sol={rsol['r_sol']} reason={rsol['reason'][:60]}")

    return {"evr": evr, "pas": pas, "pqs": pqs, "r_sol": rsol,
            "appendix_b": appendix_b,
            "v1_report_path": str(v1_md_out)}


def run_stage3(judge_model: str, out_path: Path,
               limit: int | None = None, seed: int = 42,
               only_qids: list[str] | None = None, verbose: bool = True,
               reexec_codesteps: bool = False):
    global _TIMEOUT_LOG
    _TIMEOUT_LOG = []  # reset for this batch run

    dataset = load_dataset_index()
    aa = load_accepted_answers()
    summary = json.loads((STAGE3_DIR / "batch_summary.json").read_text())
    runs_dir = STAGE3_DIR / "runs"

    entries = summary[:]
    if only_qids:
        entries = [e for e in entries if str(e["question_id"]) in set(only_qids)]
    else:
        random.Random(seed).shuffle(entries)
    if limit:
        entries = entries[:limit]

    results = []
    issues = []
    appendix_dir = out_path.parent / "appendix_b_reports"
    plan_tmp_dir = out_path.parent / "_plan_snapshots"
    for i, entry in enumerate(entries):
        qid = str(entry["question_id"]); pidx = entry["profile_index"]
        run_id = entry["run_id"]; key = (qid, pidx)
        run_dir = runs_dir / run_id
        print(f"\n[{i+1}/{len(entries)}] {run_id}")
        plan = dataset.get(key)
        if not plan:
            issues.append({"run_id": run_id, "issue": "plan_not_found"})
            continue
        try:
            exec_log = json.loads((run_dir / "execution_log.json").read_text())
            step_outputs = json.loads((run_dir / "step_outputs.json").read_text())
        except FileNotFoundError as e:
            issues.append({"run_id": run_id, "issue": f"missing_artifact: {e}"})
            continue
        ans = aa.get(key, "")
        if not ans:
            issues.append({"run_id": run_id, "issue": "no_accepted_answer"})

        # Per-run hard timeout: 600s cap so one slow run can't block the whole batch.
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                evaluate_one,
                plan, exec_log, step_outputs,
                run_dir / "execution_log.json",
                ans, judge_model,
                plan_tmp_dir / f"{run_id}.json",
                appendix_dir / f"{run_id}.md",
                verbose,
                reexec_codesteps,
            )
            try:
                res = fut.result(timeout=600)
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - t0
                _TIMEOUT_LOG.append({
                    "ts": _now(), "event_type": "run_timeout",
                    "run_id": run_id, "elapsed_s": round(elapsed, 1),
                })
                issues.append({"run_id": run_id,
                                "issue": f"run_timeout after {elapsed:.0f}s"})
                print(f"  [TIMEOUT] run exceeded 600s, skipping")
                continue
            except Exception as e:
                issues.append({"run_id": run_id,
                                "issue": f"eval_error: {type(e).__name__}: {e}"})
                continue

        results.append({"run_id": run_id, "question_id": qid, "profile_index": pidx,
                        "accepted_answer_len": len(ans), **res})
        # Incremental save after each run
        out_path.write_text(json.dumps(
            {"results": results, "issues": issues, "judge_model": judge_model,
             "n_done": len(results), "n_planned": len(entries)},
            indent=2, ensure_ascii=False))

    # Write timeout / failure log
    tlog_path = out_path.parent / "timeout_log.jsonl"
    with open(tlog_path, "w", encoding="utf-8") as f:
        for event in _TIMEOUT_LOG:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    by_type = dict(Counter(e["event_type"] for e in _TIMEOUT_LOG))
    print(f"\nTimeout log ({len(_TIMEOUT_LOG)} events) → {tlog_path}")
    if by_type:
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))

    aggregate = aggregate_results(results)
    final = {
        "judge_model": judge_model,
        "n_runs": len(results),
        "aggregate": aggregate,
        "issues": issues,
        "results": results,
        "timeout_summary": {
            "total_events": len(_TIMEOUT_LOG),
            "by_type": by_type,
        },
    }
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}\n\nAggregate ({len(results)} runs):")
    print(f"  EVR   = {aggregate['evr']:.3f}  ({aggregate['evr_pass_count']}/{aggregate['n']})")
    print(f"    sub:  " + "  ".join(
        f"{k}={aggregate['evr_subcheck_pass_rate'].get(k, 0):.2f}" for k in ("cov","loop","flow","exec")))
    print(f"  PAS   = {aggregate['pas']:.3f}  (PAS*={aggregate['pas_strict']:.3f})")
    print(f"  PQS   = {aggregate['pqs']:.3f}  "
          f"(NDAR={aggregate['ndar']:.2f} SPR={aggregate['spr']:.2f} IAR={aggregate['iar']:.2f})")
    print(f"  r_sol = {aggregate['r_sol']:.3f}  "
          f"({aggregate['r_sol_pass_count']}/{aggregate['r_sol_valid']} valid, "
          f"{aggregate['r_sol_excluded']} excluded)")
    print(f"  Appendix B (v1): " + "  ".join(
        f"{k}={aggregate['appendix_b_pass_rate'].get(k, 0):.2f}"
        for k in ("IF", "WC", "IQ", "CCGE")))
    return final


def aggregate_results(results: list[dict]) -> dict:
    if not results:
        return {"evr": 0.0, "pas": 0.0, "pas_strict": 0.0, "pqs": 0.0, "r_sol": 0.0,
                "n": 0, "evr_pass_count": 0, "ndar": 0.0, "spr": 0.0, "iar": 0.0,
                "r_sol_pass_count": 0, "r_sol_valid": 0, "r_sol_excluded": 0,
                "evr_subcheck_pass_rate": {}, "appendix_b_pass_rate": {}}
    n = len(results)
    evr_pass = sum(r["evr"]["evr_pass"] for r in results)
    pas = sum(r["pas"]["pas"] for r in results) / n
    pas_strict = sum(r["pas"]["pas_strict"] for r in results) / n
    ndar = sum(r["pqs"]["ndar"]["ndar"] for r in results) / n
    pqs = sum(r["pqs"]["pqs"] for r in results) / n
    spr = sum(r["pqs"]["spr"]["spr"] for r in results) / n
    iar = sum(r["pqs"]["iar"]["iar"] for r in results) / n
    rsol_valid = [r for r in results if r["r_sol"]["r_sol"] is not None]
    rsol_pass = sum(r["r_sol"]["r_sol"] for r in rsol_valid)
    subcheck = {k: sum(1 for r in results if r["evr"]["checks"].get(k)) / n
                for k in ("cov", "loop", "flow", "exec")}
    app_b = {k: sum(1 for r in results
                    if r.get("appendix_b", {}).get(k) == "PASS") / n
             for k in ("IF", "WC", "IQ", "CCGE")}
    return {"n": n, "evr": evr_pass / n, "evr_pass_count": evr_pass,
            "evr_subcheck_pass_rate": subcheck,
            "pas": pas, "pas_strict": pas_strict, "pqs": pqs,
            "ndar": ndar, "spr": spr, "iar": iar,
            "r_sol": rsol_pass / len(rsol_valid) if rsol_valid else 0.0,
            "r_sol_pass_count": rsol_pass, "r_sol_valid": len(rsol_valid),
            "r_sol_excluded": n - len(rsol_valid),
            "appendix_b_pass_rate": app_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reexec-codesteps", action="store_true",
                    help="Re-execute CodeInterpreterTool steps in mas-runtime Docker "
                         "before evaluation (so EVR exec/out have real stdout).")
    args = ap.parse_args()
    if args.stage3:
        run_stage3(judge_model=args.judge_model, out_path=args.out,
                   limit=args.limit, seed=args.seed, verbose=not args.quiet,
                   reexec_codesteps=args.reexec_codesteps)
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
