"""AOP meta-agent prompt — paper method preserved, output switched to §9.

The AOP paper's meta-agent (Li et al., ICLR 2025) is given:
  - A free-form user query.
  - Three decomposition principles (solvability / completeness / non-redundancy).
  - A worker pool with capability descriptions.
  - A native output format: a list of (task, id, name, reason, dep) tuples.

Q2 design A + 方案 A (2026-05-17):
  - Worker pool expanded 4 → 8, each worker mapped 1:1 to a §5 tool
    (ConceptTutor / CodeValidator / DocsRetriever / WebResearcher /
    FileWriter / PaperSearcher / RagRetriever / DirectoryReader).
  - The three principles are KEPT (internal planning discipline).
  - The native list-of-objects output format is REPLACED with the §9
    schema directly — the meta-agent is told to output a §9 plan, not
    a flat AOP list. This avoids the post-hoc translator (attribution
    clarity P4).
  - PREAMBLE + §5 + §9 + §12 (via compose_t4()) is embedded at the top
    of the prompt so the model has the tool pool and JSON schema spec
    in its context (per v1 baseline design P5 — same input as T1/T4).

Reward / replanning loop semantics live in plan.py (LLM-as-judge per
option C; the public AOP repo never shipped MLP_high.pt).
"""

from baselines.aop.teaching_agents_descs import (
    code_validator_descriptions,
    concept_tutor_descriptions,
    directory_reader_descriptions,
    docs_retriever_descriptions,
    file_writer_descriptions,
    paper_searcher_descriptions,
    rag_retriever_descriptions,
    web_researcher_descriptions,
)
from baselines.common.prompt_sections import compose_t4


_AGENT_DESCS = (
    f"ConceptTutor: {concept_tutor_descriptions[0]} "
    f"CodeValidator: {code_validator_descriptions[0]} "
    f"DocsRetriever: {docs_retriever_descriptions[0]} "
    f"WebResearcher: {web_researcher_descriptions[0]} "
    f"FileWriter: {file_writer_descriptions[0]} "
    f"PaperSearcher: {paper_searcher_descriptions[0]} "
    f"RagRetriever: {rag_retriever_descriptions[0]} "
    f"DirectoryReader: {directory_reader_descriptions[0]}"
)


_TASK_PACKAGE = compose_t4()


# Meta-agent prompt: paper three principles + 8-worker pool, output
# switched to §9. The task package (PREAMBLE + §5 + §9 + §12) is
# included verbatim so the model sees the schema and tool list once.
NEW_META_PROMPT = f'''{_TASK_PACKAGE}

---
You are a planning agent. Decompose the given query into sub-tasks and choose the most suitable worker for each sub-task. Use these three AOP principles for INTERNAL planning discipline (they are not the output format):
  (i)  solvability — every sub-task must be solvable by one of the available workers below;
  (ii) completeness — the sub-tasks together must fully address the query;
  (iii) non-redundancy — no two sub-tasks may overlap in scope.

The available workers (all of these names are valid values for step.agent) and their capabilities:
[{_AGENT_DESCS}]

YOUR FINAL OUTPUT FORMAT:
  - It is NOT the AOP-native list of {{"task", "id", "name", "reason", "dep"}} objects.
  - It IS a single JSON object conforming to the §2 schema shown above (the OUTPUT FORMAT section). That means a top-level object with "input", "output", and "execution_order" keys; "output" contains "subtasks", each subtask contains "steps", each step has fields including "agent" and "tool".
  - For each step, set "agent" to one of the 8 worker names above. Set "tool" to the §1 tool listed in the worker's capability description (or null for ConceptTutor).
  - Follow the §1 Tool rules and the §2 schema strictly. Use the §3 closing instructions for what to output.

Given the user query and learner profile, output the §2 JSON plan directly. Do not output the AOP-native list. Do not output any commentary outside the JSON.
'''


# Replan prompt: meta-agent is shown the prior §9 plan and the ids of
# weak steps, asked to revise ONLY those steps and re-emit the FULL
# §9 plan. Worker pool + principles are re-stated for clarity.
REPLAN_PROMPT_TEMPLATE = '''You previously produced a §2 plan for the following user query. The plan is repeated below. Some steps received low reward scores from the matching judge and need revision.

Three principles still apply: solvability, completeness, non-redundancy.

The available workers (use these names for step.agent): [{agent_descs}]

User query: {query}

Previous §2 plan:
{prev_plan}

The weak step ids (format "subtask_index.step_index", as listed in the plan) that scored below threshold and must be revised: {weak_ids}

Revise ONLY those steps — keep all other subtasks and steps unchanged. Re-pick the agent / tool / instruction for each weak step so it matches a worker's capability cleanly. Then output the FULL revised plan as a single §2 JSON object (same schema as before). Do not output the AOP-native list. Do not output any commentary outside the JSON.
'''


def build_replan_prompt(query: str, prev_plan_json: str, weak_ids: list) -> str:
    return REPLAN_PROMPT_TEMPLATE.format(
        agent_descs=_AGENT_DESCS,
        query=query,
        prev_plan=prev_plan_json,
        weak_ids=weak_ids,
    )
