"""
AutoAgents upstream runner. Runs INSIDE .venvs/autoagents_310 (Python
3.10 + 245 pinned packages from external/AutoAgents/requirements.txt).
Spawned as a subprocess by baselines/autoagents/plan.py.

Reads {query, learner} JSON from a file, runs the upstream Manager +
ObserverAgents + ObserverPlans pipeline against DashScope's
OpenAI-compatible qwen3-32b endpoint (v1 baseline F-tier backbone), and
writes {plan_text, ...} JSON to a file. Under scheme A (BASELINE_DESIGN
03_FORMAT_MISMATCH.md) the Manager is now instructed to emit our §9
JSON directly, so plan_text is the raw final JSON object text (no
downstream LLM remap step).

DashScope quirks handled here:
  - Qwen3 requires `enable_thinking=False` for non-streaming requests.
    We monkey-patch litellm.completion / litellm.acompletion to inject
    this flag automatically.
  - Model name needs `openai/` prefix to force litellm's openai provider.

Path note: this script runs in the AutoAgents sub-venv, but we still
need to import baselines.common.prompt_sections from the project root.
We insert the project root into sys.path BEFORE the os.chdir() call in
_run_upstream so the import resolves to the project root.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Project root = baselines/autoagents/upstream_runner.py → autoagents → baselines → ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOAGENTS_REPO = Path(os.environ.get("AUTOAGENTS_REPO", PROJECT_ROOT / "external" / "AutoAgents"))
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "openai/qwen3-32b"

# Make the project root importable in this sub-venv so we can pull in
# baselines.common.prompt_sections.compose_t4(). Must happen BEFORE any
# os.chdir() into AUTOAGENTS_REPO that _run_upstream() performs.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_dashscope_key() -> str:
    env_file = PROJECT_ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"DASHSCOPE_API_KEY not in {env_file}")


def _configure_litellm() -> None:
    """Point litellm at DashScope + monkey-patch to inject
    enable_thinking=False on every call (Qwen3 quirk)."""
    api_key = _read_dashscope_key()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = DASHSCOPE_BASE
    os.environ["OPENAI_API_MODEL"] = MODEL_NAME

    import litellm  # noqa: E402  (after env vars set)
    litellm.api_base = DASHSCOPE_BASE
    litellm.api_key = api_key

    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_completion(*args, **kwargs)

    async def _patched_acompletion(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return await _orig_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion


def _patch_environment_parser_roles() -> None:
    """Hot-patch upstream Environment._parser_roles to tolerate Qwen3-32B
    JSON brittleness (per user decision 2026-05-17, Q9 option 1).

    The upstream method does raw ``json.loads(agent.strip())`` on each
    role JSON blob extracted from CreateRoles' markdown output. Qwen3-32B
    occasionally emits role JSON with missing commas / unescaped quotes
    inside string values, causing JSONDecodeError → the whole explorer
    crashes. We wrap that single ``json.loads`` call with
    ``fix_json_format`` fallback (the same 2-level repair we use in main-
    venv plan.py).

    Scope: ONLY the role-blob JSON parsing path. We do NOT modify any
    CreateRoles / CheckRoles / CreatePlan / CheckPlan prompts or
    coordination logic. The fallback is a workmanlike JSON-repair
    (deterministic, no LLM), so it does not introduce attribution
    confounding per P4. Documented in
    ``BASELINE_DESIGN/01_DECISIONS_LOG.md [2026-05-17] Q9 = option 1``.

    Must be called AFTER ``os.chdir(str(AUTOAGENTS_REPO))`` — upstream
    ``autoagents.system.const.get_project_root`` searches for repo
    markers relative to CWD, so importing ``autoagents.environment``
    from the wrong CWD raises ``Exception('Project root not found.')``.
    """
    if str(AUTOAGENTS_REPO) not in sys.path:
        sys.path.insert(0, str(AUTOAGENTS_REPO))
    if Path(os.getcwd()).resolve() != AUTOAGENTS_REPO.resolve():
        os.chdir(str(AUTOAGENTS_REPO))

    import re as _re
    import json as _json
    from autoagents.environment import Environment
    from baselines.common.json_repair import fix_json_format

    # Field-name aliases bridging §9 schema (which our compose_t4() injects
    # into the Manager's idea) and upstream's expected role schema. Qwen3-32B
    # inconsistently picks one or the other across roles inside the same
    # CreateRoles output. Without these aliases, upstream group.py:31 and
    # other upstream sites crash with KeyError.
    #
    # Source (§9 schema) → upstream expected key
    _FIELD_ALIASES = {
        "agent_role": "name",          # §9 §9 plan.output.agents[i].agent_role
        "role_name":  "name",          # paraphrase Qwen3 sometimes emits
        "agent_name": "name",          # paraphrase Qwen3 sometimes emits
        "goal":       "description",   # §9 agent.goal ≈ upstream description
        "backstory":  "description",   # §9 agent.backstory as fallback desc
    }

    # Heuristic: which keys mark a dict as a role definition (vs e.g. a
    # learner profile JSON we accidentally injected into the idea). At
    # least one of these must be present, otherwise the parsed dict is
    # not a role and should be ignored.
    _ROLE_NAME_KEYS = {
        "name", "agent_role", "role_name", "agent_name",
    }

    def _looks_like_role(role: dict) -> bool:
        """True iff the dict has at least one role-name key."""
        if not isinstance(role, dict):
            return False
        return any(k in role for k in _ROLE_NAME_KEYS)

    def _normalize_role_keys(role: dict) -> dict:
        """Add upstream-expected aliases without removing the originals."""
        for src, dst in _FIELD_ALIASES.items():
            if src in role and dst not in role:
                role[dst] = role[src]
        return role

    def _patched_parser_roles(self, text):
        agents = _re.findall(r"{[\s\S]*?}", text)
        agents_args = []
        for agent in agents:
            blob = agent.strip()
            parsed = None
            try:
                parsed = _json.loads(blob)
            except Exception:
                for attempt in (1, 2):
                    try:
                        parsed = _json.loads(
                            fix_json_format(blob, repair_attempt=attempt)
                        )
                        break
                    except Exception:
                        continue
            if parsed is None:
                # Skip malformed role blob silently — the framework will
                # still proceed with whatever roles it successfully
                # parsed. Better than crashing the entire pipeline.
                print(
                    f"[upstream_runner patch] skipped malformed role JSON "
                    f"({len(blob)} chars)",
                    file=sys.stderr,
                )
                continue
            if isinstance(parsed, dict) and len(parsed.keys()) > 0:
                parsed = _normalize_role_keys(parsed)
                if not _looks_like_role(parsed):
                    # Probably a non-role JSON blob the regex caught
                    # (e.g. learner profile JSON echoed back by Manager
                    # from our idea). Skip silently to avoid KeyError
                    # downstream in Group.__init__.
                    print(
                        f"[upstream_runner patch] skipped non-role JSON "
                        f"(keys={list(parsed.keys())[:5]})",
                        file=sys.stderr,
                    )
                    continue
                agents_args.append(parsed)

        print("---------------Agents---------------")
        for i, agent in enumerate(agents_args):
            print("Role", i, agent)
        return agents_args

    Environment._parser_roles = _patched_parser_roles


def _build_idea(query: str, learner: dict) -> str:
    """The single `idea` string fed to Manager.

    Scheme A (BASELINE_DESIGN/03_FORMAT_MISMATCH.md): output format is
    part of the task definition. The compose_t4() task description package
    (PREAMBLE + §5 + §9 + §12 via compose_t4()) is prepended verbatim
    so the Manager sees the §9 schema, then we add a hard instruction
    that the FINAL output must be a §9 JSON object with `input` +
    `output` keys. The internal CreateRoles -> CheckRoles -> CreatePlan
    -> CheckPlan loop is preserved unchanged; only the final emission
    format changes.
    """
    from baselines.common.prompt_sections import compose_t4, PROMPT_FILE

    assert PROMPT_FILE.name == "prompt_for_inference.txt" and PROMPT_FILE.exists(), (
        f"prompt_sections resolved to wrong PROMPT_FILE: {PROMPT_FILE}"
    )

    task_package = compose_t4()
    learner_json = json.dumps(learner, ensure_ascii=False)
    return (
        f"{task_package}\n\n"
        "================================================================\n"
        "AUTOAGENTS MANAGER INSTRUCTIONS\n"
        "================================================================\n"
        "You are the AutoAgents Manager. Run your full upstream workflow: "
        "CreateRoles -> CheckRoles -> CreatePlan -> CheckPlan, with "
        "ObserverAgents and ObserverPlans auditing each stage. Tailor "
        "the roles, their tools, and the execution plan to the learner's "
        "stated background.\n\n"
        "INTERMEDIATE WORKFLOW OUTPUT FORMAT (required for the framework "
        "to operate): keep your CreateRoles / CheckRoles / CreatePlan / "
        "CheckPlan output in your framework's native markdown layout "
        "(the `## Question or Task:`, `## Selected Roles List:`, "
        "`## Created Roles List:`, `## Execution Plan:`, `## RoleFeedback`, "
        "`## PlanFeedback` headers from your FORMAT_EXAMPLE). The "
        "downstream Observer / Check actions parse those headers — they "
        "must be present in every intermediate Manager response.\n\n"
        "FINAL OUTPUT REQUIREMENT (this is what we evaluate): in addition "
        "to the markdown workflow above, your FINAL Manager message "
        "(the one emitted AFTER CheckPlan approves the plan) MUST end "
        "with a single fenced JSON code block that contains the complete "
        "§2 plan corresponding to the markdown plan you produced. "
        "Format it as:\n\n"
        "```json\n"
        "{ \"input\": {\"query\": \"...\", \"learner\": {...}}, "
        "\"output\": {\"agents\": [...], \"subtasks\": [...], "
        "\"execution_order\": [...]} }\n"
        "```\n\n"
        "The agents / subtasks / steps / tools in the JSON block MUST "
        "match the markdown plan above. Tools must come from the §1 "
        "tool pool (8 names). step.tool must be null OR a tool declared "
        "on that step's agent. execution_order must list every step.id.\n\n"
        f"## Question or Task:\n{query}\n\n"
        f"LEARNER PROFILE: {learner_json}\n"
    )


async def _run_upstream(idea: str):
    """Run the real AutoAgents pipeline. Returns (history_str, last_message)
    where history_str is the concatenated `f\"{role}: {content}\"` log of
    every message published (a STRING, per upstream's
    environment.history += f\"\\n{message}\" pattern), and last_message
    is the most recent Message object held in the Manager's _rc.memory."""
    if str(AUTOAGENTS_REPO) not in sys.path:
        sys.path.insert(0, str(AUTOAGENTS_REPO))

    # cfg.py lives at repo root and is read as a flat module — chdir so
    # any relative path resolution it does works the same as upstream.
    os.chdir(str(AUTOAGENTS_REPO))

    from autoagents.roles import Manager, ObserverAgents, ObserverPlans
    from autoagents.explorer import Explorer

    explorer = Explorer()
    manager = Manager()
    explorer.hire([manager, ObserverAgents(), ObserverPlans()])
    explorer.invest(10.0)
    await explorer.start_project(idea=idea, llm_api_key=os.environ["OPENAI_API_KEY"])
    # Bumped 3 -> 5 (PHASE2_REVIEW.md §autoagents): AutoAgents paper
    # expects 4-6 rounds for ObserverAgents/ObserverPlans to fully audit
    # the CreateRoles + CreatePlan output before convergence.
    await explorer.run(n_round=5)

    history_str = str(explorer.environment.history)

    last_content = ""
    try:
        mem = getattr(manager._rc, "memory", None)
        if mem is not None:
            msgs = getattr(mem, "storage", None) or getattr(mem, "_storage", None)
            if msgs:
                last_content = str(getattr(msgs[-1], "content", ""))
                if not last_content:
                    instruct = getattr(msgs[-1], "instruct_content", None)
                    last_content = str(instruct) if instruct is not None else ""
    except Exception:
        last_content = ""
    return history_str, last_content


def _find_last_json_object(text: str) -> str | None:
    """Scan `text` and return the last balanced top-level `{...}` block.

    A simple brace counter that respects double-quoted strings and
    backslash escapes within them. Returns the substring (including the
    outer braces) or None if no balanced object is found.
    """
    if not text:
        return None
    last_obj = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_obj = text[i:j + 1]
                        i = j + 1
                        break
                j += 1
            else:
                # unbalanced — stop scanning this candidate
                break
            continue
        i += 1
    return last_obj


def _extract_json_text(history_str: str, last_message: str) -> str:
    """Locate the Manager's final §9 JSON object.

    Strategy (scheme A): the Manager is now instructed to emit a single
    top-level JSON object as its final message. We search for the LAST
    balanced `{...}` block in history_str first, then fall back to
    last_message, then strip ```json fences as a last resort. If no
    JSON-looking text is found, return history_str so the downstream
    failure log records what came out.
    """
    for src in (history_str, last_message):
        if not src:
            continue
        # ```json ...``` fenced block (take the last fenced match)
        fenced = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", src))
        if fenced:
            return fenced[-1].group(1)
        obj = _find_last_json_object(src)
        if obj is not None:
            return obj
    return history_str or last_message or ""


def main() -> None:
    """
    Usage: python upstream_runner.py <input.json> <output.json>
    AutoAgents prints to stdout during execution; we therefore exchange
    payloads via files, not stdin/stdout pipes.
    """
    if len(sys.argv) != 3:
        sys.stderr.write("usage: upstream_runner.py <input.json> <output.json>\n")
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, encoding="utf-8") as f:
        inp = json.load(f)
    query = inp["query"]
    learner = inp["learner"]

    _configure_litellm()
    idea = _build_idea(query, learner)
    # Patch upstream Environment._parser_roles AFTER _build_idea (no chdir
    # required there) but BEFORE _run_upstream imports Manager. The patch
    # itself performs the chdir into AUTOAGENTS_REPO that upstream
    # const.get_project_root() requires.
    _patch_environment_parser_roles()

    try:
        history_str, last_message = asyncio.run(_run_upstream(idea))
        plan_text = _extract_json_text(history_str, last_message)
        out = {
            "ok": True,
            "plan_text": plan_text,
            "history_chars": len(history_str),
            "last_message_chars": len(last_message),
        }
    except Exception as err:
        import traceback
        out = {
            "ok": False,
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(),
            "plan_text": "",
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
