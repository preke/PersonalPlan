# Tier 2 Evaluator — No Timeout + Env-Fail Logging Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this.

**Goal:** Fix all timeout causes in `tier2_evaluator_v2.py` + record every timeout/env-fail event to a log; env failures on code execution must be recorded explicitly — NOT downgraded to semantic fallback evaluation.

**Architecture:** Modify 2 files. `plan_mapper_fixed/evaluator.py` gets a single `timeout=` kwarg. `tier2_evaluator_v2.py` gets (1) per-call timeout on every LLM call, (2) retry+truncation wrapper around v1 evaluator, (3) strict env-fail classification in reexec, (4) no fallback to original content when Docker fails, (5) per-run timeout guard, (6) timeout_log.jsonl output.

**Tech Stack:** Python 3.10+, openai SDK v1, concurrent.futures, pathlib

---

## Root Causes (from analysis)

| Cause | Location | Fix |
|---|---|---|
| v1 evaluator: no timeout, no retry, huge prompt (14k+ tokens) | `plan_mapper_fixed/evaluator.py:252` | Add `timeout=120`; truncate log before call; 3-retry wrapper |
| `_judge`: no timeout parameter | `tier2_evaluator_v2.py:~480` | Add `timeout=60` to every API call |
| Env fails silently → fallback to original content | `_effective_content`, `_reexec_step_outputs` | Classify env fails; return `None` content; log all events |
| No per-run timeout guard | `run_stage3` / `evaluate_one` | Wrap each run in `concurrent.futures` with 600s timeout |
| No timeout/failure log output | entire script | Collect `_timeout_log` list; write `timeout_log.jsonl` |

## User's additional requirement
- **所有timeout都记录到log**: every timeout event (Docker, API, run-level) written to `<out_dir>/timeout_log.jsonl`

---

## File Structure

| File | Change |
|---|---|
| `Evaluation/tier2_evaluator_v2.py` | All changes below (main file) |
| `stage3_execution/plan_mapper_fixed/evaluator.py` | Add `timeout=120` to line 252 |

---

## Task 1: Add `timeout=120` to v1 evaluator API call

**File:** `stage3_execution/plan_mapper_fixed/evaluator.py`

- [ ] Read lines 248–265
- [ ] Edit `client.chat.completions.create(...)` to add `timeout=120`
- [ ] Verify the edit is the only change in this file

---

## Task 2: Add `timeout=60` to `_judge` in tier2_evaluator_v2.py

**File:** `Evaluation/tier2_evaluator_v2.py`

- [ ] Find `_judge` function (search `def _judge`)
- [ ] Add `timeout=60` to `_client().chat.completions.create(...)` call
- [ ] Increase retry sleep: `time.sleep(5 * (attempt + 1))` (was 1.5×)
- [ ] In the `except` block, if `"timeout"` in str(last_err).lower(), add to `_timeout_log`

---

## Task 3: Truncate + retry wrapper for `run_v1_evaluator`

**File:** `Evaluation/tier2_evaluator_v2.py`

The v1 evaluator's prompt includes the full execution_log JSON-dumped. Truncate before writing to disk.

- [ ] In `run_v1_evaluator` (or its call site in `evaluate_one`), before writing `exec_log` to disk, call `_truncate_exec_log(exec_log, max_entries=60, max_chars_per_entry=1200)`:
  ```python
  def _truncate_exec_log(log: list, max_entries=60, max_chars=1200) -> list:
      """Keep last max_entries; truncate each entry's text fields to max_chars."""
      truncated = log[-max_entries:] if len(log) > max_entries else log
      result = []
      for entry in truncated:
          e = dict(entry)
          for field in ("teacher_output", "student_response", "actual_interaction"):
              if isinstance(e.get(field), str) and len(e[field]) > max_chars:
                  e[field] = e[field][:max_chars] + "...[truncated]"
              elif isinstance(e.get(field), dict):
                  for k, v in e[field].items():
                      if isinstance(v, str) and len(v) > max_chars:
                          e[field][k] = v[:max_chars] + "...[truncated]"
          result.append(e)
      return result
  ```
- [ ] Wrap `run_v1_evaluator` call in a retry loop (3 attempts, 10s backoff):
  ```python
  def _run_v1_with_retry(plan, exec_log_path, accepted_answer, judge_model,
                          plan_tmp_path, v1_md_out, verbose, _timeout_log):
      last_err = None
      for attempt in range(3):
          try:
              return run_v1_evaluator(plan, exec_log_path, accepted_answer,
                                       judge_model, plan_tmp_path, v1_md_out)
          except Exception as e:
              last_err = e
              _timeout_log.append({
                  "ts": _now(), "event_type": "v1_evaluator_error",
                  "attempt": attempt, "error": str(e)[:200]
              })
              if attempt < 2:
                  time.sleep(10 * (attempt + 1))
      # All retries failed — return a minimal fail report so parse functions get FAIL
      return _V1_FAIL_REPORT
  ```
- [ ] Define `_V1_FAIL_REPORT` constant (minimal string that `parse_appendix_b` returns all FAIL from):
  ```python
  _V1_FAIL_REPORT = (
      "Check 1: Instruction Fidelity\n  Verdict: FAIL\n"
      "Check 2: Workflow Completeness\n  Verdict: FAIL\n"
      "Check 3: Interaction Quality\n  Verdict: FAIL\n"
      "Check 4: Content Correctness and Guidance Effectiveness\n  Verdict: FAIL\n"
      "Overall verdict: execution_failed\n"
      "v1_evaluator_error: all retries failed\n"
  )
  ```
- [ ] Update `evaluate_one` to pass `_timeout_log` to `_run_v1_with_retry`

---

## Task 4: Env-fail classification in `_reexec_step_outputs`

**File:** `Evaluation/tier2_evaluator_v2.py`

The current code stores `reexec_status` but `_effective_content` falls back to original content on failure.

- [ ] Add env-fail classification function:
  ```python
  _ENV_FAIL_STATUSES = frozenset({
      "timeout", "docker_unavailable", "unsupported_lang",
      "error_build", "error_runtime", "error_unknown",
  })
  _CODE_ERROR_STATUSES = frozenset({"nonzero_exit"})

  def _reexec_category(status: str) -> str:
      """Classify reexec status into: ok | env_fail | code_error | no_code | unknown"""
      if status == "ok": return "ok"
      if status in _ENV_FAIL_STATUSES: return "env_fail"
      if status in _CODE_ERROR_STATUSES: return "code_error"
      if status in ("no_code_or_lang", "no_code", "no_lang"): return "no_code"
      return "unknown"
  ```
- [ ] In `_reexec_step_outputs`, after `so["reexec_status"] = status`, add:
  ```python
  so["reexec_category"] = _reexec_category(status)
  ```
- [ ] Collect env-fail events and return them (or pass to a mutable list):
  The function should accept a `_timeout_log: list` param and append:
  ```python
  if so["reexec_category"] == "env_fail":
      _timeout_log.append({
          "ts": _now(),
          "event_type": f"docker_{status}",
          "step_id": step_id,
          "lang": lang,
          "error": so.get("reexec_stderr", "")[:200],
      })
  ```

---

## Task 5: Fix `_effective_content` — no fallback on env_fail

**File:** `Evaluation/tier2_evaluator_v2.py`

Currently: `_effective_content` returns original content when reexec is not "ok".

- [ ] Change `_effective_content`:
  ```python
  def _effective_content(so: dict) -> str | None:
      """Return sandbox stdout if real execution succeeded, else None.

      Returns None (not original content) for env_fail/timeout/docker_unavailable.
      Returning None signals to callers that we cannot evaluate this step's execution.
      Original content fallback is intentionally removed — env failures must be
      recorded as env_fail, not downgraded to semantic evaluation.
      """
      cat = so.get("reexec_category")
      if cat == "ok":
          return so.get("reexec_content") or ""
      # no_code: step had no code to execute, not an env failure
      if cat == "no_code" or so.get("reexec_status") == "no_code_or_lang":
          return so.get("content") or ""  # original is fine for non-code steps
      # env_fail or code_error: do NOT fall back to original content
      return None
  ```

---

## Task 6: Update EVR exec/out checks to handle `None` from `_effective_content`

**File:** `Evaluation/tier2_evaluator_v2.py`

The EVR `exec` check (CER) and `out` check (CCR) need to handle `None` content.

- [ ] In `evr_run`, when collecting code steps:
  ```python
  env_fail_steps = []
  code_error_steps = []
  exec_ok_steps = []

  for step_id, so in step_outputs.items():
      if so.get("meta", {}).get("tool") != "CodeInterpreterTool":
          continue
      cat = so.get("reexec_category", "unknown")
      content = _effective_content(so)

      if cat == "env_fail":
          env_fail_steps.append({
              "step_id": step_id,
              "lang": so.get("lang", "unknown"),
              "status": so.get("reexec_status"),
              "reason": "env_fail: Docker unavailable, timeout, or unsupported language",
          })
      elif cat == "code_error":
          code_error_steps.append({"step_id": step_id, "status": so.get("reexec_status")})
      elif cat == "ok":
          exec_ok_steps.append((step_id, content, so))

  # exec check (CER): all code steps must be ok (env_fail → FAIL)
  has_code_steps = (len(env_fail_steps) + len(code_error_steps) + len(exec_ok_steps)) > 0
  exec_pass = has_code_steps and (len(env_fail_steps) == 0 and len(code_error_steps) == 0)
  ```
- [ ] Add `env_fail_steps` and `code_error_steps` to the EVR result dict:
  ```python
  return {
      "evr_pass": ...,
      "checks": {...},
      "env_fail_steps": env_fail_steps,
      "code_error_steps": code_error_steps,
      ...
  }
  ```

---

## Task 7: Per-run timeout guard in `run_stage3`

**File:** `Evaluation/tier2_evaluator_v2.py`

- [ ] Import `concurrent.futures` and `datetime`
- [ ] In `run_stage3` loop, wrap `evaluate_one` call:
  ```python
  import concurrent.futures, datetime

  def _now() -> str:
      return datetime.datetime.utcnow().isoformat()

  # Inside run_stage3 loop:
  run_timeout_log = []
  t0 = time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
      fut = ex.submit(evaluate_one, plan, exec_log, step_outputs, ...)
      try:
          res = fut.result(timeout=600)   # 10 min hard cap per run
      except concurrent.futures.TimeoutError:
          elapsed = time.monotonic() - t0
          event = {
              "ts": _now(), "event_type": "run_timeout",
              "run_id": run_id, "elapsed_s": round(elapsed, 1),
          }
          timeout_log.append(event)
          issues.append({"run_id": run_id, "issue": f"run_timeout after {elapsed:.0f}s"})
          continue
      except Exception as e:
          issues.append({"run_id": run_id, "issue": f"eval_error: {type(e).__name__}: {e}"})
          continue
  ```

---

## Task 8: Collect + write `timeout_log.jsonl`

**File:** `Evaluation/tier2_evaluator_v2.py`

- [ ] Initialize `timeout_log: list[dict] = []` at start of `run_stage3`
- [ ] Pass `timeout_log` (mutable list) as a parameter through `evaluate_one → _run_v1_with_retry`, `_judge`, `_reexec_step_outputs` so all timeout events accumulate
- [ ] After the loop, write `timeout_log.jsonl`:
  ```python
  tlog_path = out_path.parent / "timeout_log.jsonl"
  with open(tlog_path, "w") as f:
      for event in timeout_log:
          f.write(json.dumps(event, ensure_ascii=False) + "\n")
  print(f"Timeout log ({len(timeout_log)} events) → {tlog_path}")
  ```
- [ ] Also include `timeout_log` summary in the final JSON output:
  ```python
  final = {
      ...,
      "timeout_summary": {
          "total_events": len(timeout_log),
          "by_type": Counter(e["event_type"] for e in timeout_log),
      }
  }
  ```

---

## Verification

- [ ] Run on 5 runs: `python3 tier2_evaluator_v2.py --stage3 --limit 5 --out /tmp/pilot_no_timeout.json`
- [ ] Confirm no hanging (completes within 10 min for 5 runs)
- [ ] Confirm `timeout_log.jsonl` is written (even if empty = 0 events)
- [ ] Confirm any env-fail steps appear in `env_fail_steps` field of EVR, NOT as semantic fallback
- [ ] Check a result with known code steps: verify `reexec_category` field is set correctly
