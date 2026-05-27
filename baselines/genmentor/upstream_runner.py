"""
GenMentor upstream runner. Runs INSIDE .venvs/genmentor_310 (Python 3.10
with the slim subset of external/gen-mentor/backend/requirements.txt
needed for Modules 1-4 core paths -- FastAPI/uvicorn/chroma/embeddings
are NOT installed because we do not call those code paths).

Spawned as a subprocess by baselines/genmentor/plan.py.

Reads {query, learner} JSON from a file passed as argv[1]; calls the
real upstream functions Module 1 -> Module 2 -> Module 3 -> Module 4
(outline only, no RAG / no quiz), and writes
  {ok, skill_gaps, skill_requirements, learner_profile, learning_path,
   session_outlines, ...}
to the file passed as argv[2]. The host process then remaps that
bundle into our §9 JSON via a separate LLM call.

Method preservation: every step uses the upstream pydantic-validated
agent (BaseAgent subclasses) and the upstream system + task prompts.
The only piece NOT exercised is Module 4's full markdown/quiz pipeline
(`create_learning_content_with_llm` for `method_name="genmentor"`)
because its output is per-knowledge-point markdown courseware, which
is downstream of plan creation and would be discarded by our §9 remap.
We do call Module 4's outline preparation (`prepare_content_outline_with_llm`)
once per learning_path session so the remap step has per-session
structure to project into subtasks + steps.

DashScope quirks handled:
  - Qwen3 requires `enable_thinking=False`. We pass it via
    `extra_body={"enable_thinking": False}` to LLMFactory.create; that
    flag is forwarded through langchain's ChatOpenAI to every request.
  - We point langchain at DashScope's OpenAI-compatible endpoint via
    `base_url` + `api_key` kwargs (provider="openai").
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Project root = baselines/genmentor/upstream_runner.py → genmentor → baselines → ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENMENTOR_BACKEND = Path(os.environ.get(
    "GENMENTOR_BACKEND", PROJECT_ROOT / "external" / "gen-mentor" / "backend"
))
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# T5 backbone per v1 baseline design: qwen3-32b on DashScope.
MODEL_NAME = "qwen3-32b"


def _read_dashscope_key() -> str:
    env_file = PROJECT_ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"DASHSCOPE_API_KEY not in {env_file}")


def _build_llm():
    """Construct a langchain ChatOpenAI bound to DashScope qwen3-32b.

    `extra_body` is the only DashScope-specific knob -- it survives all
    the way to the OpenAI HTTP body so the qwen3 reasoning engine
    sees `enable_thinking=false`."""
    if str(GENMENTOR_BACKEND) not in sys.path:
        sys.path.insert(0, str(GENMENTOR_BACKEND))
    from base.llm_factory import LLMFactory  # noqa: E402

    api_key = _read_dashscope_key()
    return LLMFactory.create(
        model=MODEL_NAME,
        model_provider="openai",
        base_url=DASHSCOPE_BASE,
        api_key=api_key,
        temperature=0,
        extra_body={"enable_thinking": False},
    )


def _learner_to_text(learner: dict) -> str:
    """Render our learner profile (about_me + top_tags + ...) as the
    free-form `learner_information` string that GenMentor expects."""
    parts = []
    about = learner.get("about_me") or ""
    if about:
        parts.append(f"About me: {about}")
    tags = learner.get("top_tags") or []
    if tags:
        parts.append(f"Top tags / strong areas: {', '.join(map(str, tags))}")
    for k in ("background", "experience_years", "preferred_style",
              "languages", "tools", "weaknesses"):
        v = learner.get(k)
        if v:
            parts.append(f"{k}: {v}")
    # Any other fields
    extra = {k: v for k, v in learner.items()
             if k not in {"about_me", "top_tags", "background",
                          "experience_years", "preferred_style",
                          "languages", "tools", "weaknesses"}}
    if extra:
        parts.append("Other profile fields: " + json.dumps(extra, ensure_ascii=False))
    return "\n".join(parts) if parts else "No additional learner information."


TARGET_SESSION_COUNT = 3
NO_RAG_NOTE = (
    "No external resources are available for this run. "
    "Use only the learner profile, learning path, and learning session above, "
    "together with your own general knowledge, to produce the outline. "
    "Do NOT fabricate citations or references to external documents."
)


def _schedule_learning_path_fixed_sessions(llm, learner_profile, session_count: int) -> dict:
    """Bug 1 fix: upstream Task A task_prompt references only {learner_profile}
    and never {session_count}, so the LLM is told "between 1 and 10" by the
    system prompt but never sees our explicit constraint. We build the
    LearningPathScheduler ourselves and pass a Task A task_prompt that
    explicitly states the exact number of sessions to produce, while keeping
    the upstream system_prompt + LearningPath pydantic validation intact.
    """
    from modules.personalized_resource_delivery.agents.learning_path_scheduler import (
        LearningPathScheduler,
        SessionSchedulePayload,
    )
    from modules.personalized_resource_delivery.prompts.learning_path_scheduling import (
        learning_path_scheduler_task_prompt_session,
    )
    from modules.personalized_resource_delivery.schemas import LearningPath

    # Append an explicit session_count constraint to the Task A task_prompt.
    # Original ends with "* **Learner Profile**: {learner_profile}\n".
    task_prompt_with_count = (
        learning_path_scheduler_task_prompt_session
        + f"\n\n**HARD CONSTRAINT (overrides the 1-10 range above)**: "
        f"You MUST produce exactly {session_count} sessions in the "
        f'`learning_path` array -- no more, no fewer. The "Quality over '
        f'Quantity" directive still applies *within* this fixed budget.\n'
    )

    scheduler = LearningPathScheduler(llm)
    payload = SessionSchedulePayload(
        learner_profile=learner_profile, session_count=session_count
    ).model_dump()
    raw_output = scheduler.invoke(payload, task_prompt=task_prompt_with_count)
    return LearningPath.model_validate(raw_output).model_dump()


def _prepare_content_outline_no_rag(llm, learner_profile, learning_path, session) -> dict:
    """Bug 2 fix: upstream system_prompt says 'You MUST use the
    external_resources to ensure content is accurate and up-to-date (RAG)',
    but we run with RAG disabled (search_rag_manager=None). Telling the LLM
    to use something we never provide leads to fabricated references and
    confused outputs. We rebuild the LearningContentCreator with a system
    prompt that explicitly drops the RAG requirement, and we pass a clear
    'no external resources' note as the {external_resources} value so the
    task_prompt placeholder is never empty.
    """
    from modules.personalized_resource_delivery.agents.learning_content_creator import (
        LearningContentCreator,
    )
    from modules.personalized_resource_delivery.prompts.learning_content_creator import (
        learning_content_creator_system_prompt,
        learning_content_creator_task_prompt_outline,
    )
    from modules.personalized_resource_delivery.schemas import ContentOutline

    # Strip the mandatory-RAG sentence and replace with an explicit no-RAG
    # directive. The rest of the system prompt (Task A/B/C definitions,
    # output formats) is preserved verbatim.
    rag_sentence = (
        "You MUST use the `external_resources` to ensure content is accurate "
        "and up-to-date (RAG)."
    )
    no_rag_sentence = (
        "External resources / RAG are NOT available for this run. "
        "Rely solely on the learner profile, learning path, learning session, "
        "and your own general knowledge. Do not fabricate citations."
    )
    if rag_sentence in learning_content_creator_system_prompt:
        patched_system_prompt = learning_content_creator_system_prompt.replace(
            rag_sentence, no_rag_sentence
        )
    else:
        # Fallback: prepend a no-RAG note if the exact sentence drifted upstream.
        patched_system_prompt = (
            no_rag_sentence + "\n\n" + learning_content_creator_system_prompt
        )

    creator = LearningContentCreator(llm, search_rag_manager=None)
    creator.set_prompts(system_prompt=patched_system_prompt)

    payload = {
        "learner_profile": learner_profile,
        "learning_path": learning_path,
        "learning_session": session,
        "external_resources": NO_RAG_NOTE,
    }
    raw_output = creator.invoke(
        payload, task_prompt=learning_content_creator_task_prompt_outline
    )
    return ContentOutline.model_validate(raw_output).model_dump()


def _run_modules(query: str, learner: dict) -> dict:
    """Execute Modules 1-4 (core paths) and bundle outputs."""
    if str(GENMENTOR_BACKEND) not in sys.path:
        sys.path.insert(0, str(GENMENTOR_BACKEND))

    from modules.skill_gap_identification.agents.skill_gap_identifier import (
        identify_skill_gap_with_llm,
    )
    from modules.adaptive_learner_modeling.agents.adaptive_learning_profiler import (
        initialize_learner_profile_with_llm,
    )

    llm = _build_llm()
    learner_info = _learner_to_text(learner)

    timings = {}

    t = time.time()
    skill_gaps, skill_requirements = identify_skill_gap_with_llm(
        llm,
        learning_goal=query,
        learner_information=learner_info,
    )
    timings["m1"] = round(time.time() - t, 2)

    t = time.time()
    learner_profile = initialize_learner_profile_with_llm(
        llm,
        learning_goal=query,
        learner_information=learner_info,
        skill_gaps=skill_gaps,
    )
    timings["m2"] = round(time.time() - t, 2)

    t = time.time()
    learning_path = _schedule_learning_path_fixed_sessions(
        llm, learner_profile, session_count=TARGET_SESSION_COUNT,
    )
    timings["m3"] = round(time.time() - t, 2)

    t = time.time()
    session_outlines: list[dict] = []
    for session in learning_path.get("learning_path", []):
        try:
            outline = _prepare_content_outline_no_rag(
                llm, learner_profile, learning_path, session,
            )
        except Exception as e:
            outline = {"_error": f"{type(e).__name__}: {e}",
                       "title": session.get("title", ""), "sections": []}
        session_outlines.append({"session_id": session.get("id"),
                                 "session_title": session.get("title"),
                                 "outline": outline})
    timings["m4"] = round(time.time() - t, 2)

    return {
        "skill_gaps": skill_gaps,
        "skill_requirements": skill_requirements,
        "learner_profile": learner_profile,
        "learning_path": learning_path,
        "session_outlines": session_outlines,
        "timings": timings,
    }


def main() -> None:
    """
    Usage: upstream_runner.py <input.json> <output.json>
    """
    if len(sys.argv) != 3:
        sys.stderr.write("usage: upstream_runner.py <input.json> <output.json>\n")
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, encoding="utf-8") as f:
        inp = json.load(f)
    query = inp["query"]
    learner = inp["learner"]

    try:
        bundle = _run_modules(query, learner)
        out: dict[str, Any] = {"ok": True, **bundle}
    except Exception as err:
        out = {
            "ok": False,
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
