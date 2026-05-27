from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import PlanPayload


TOOL_REGISTRY: Dict[str, str] = {
    "FirecrawlSearchTool": "crewai_tools.FirecrawlSearchTool",
    "RagTool": "crewai_tools.RagTool",
    "CodeInterpreterTool": "crewai_tools.CodeInterpreterTool",
    "DirectoryReadTool": "crewai_tools.DirectoryReadTool",
    "FileReadTool": "crewai_tools.FileReadTool",
    "FileWriterTool": "crewai_tools.FileWriterTool",
    "GithubSearchTool": "crewai_tools.GithubSearchTool",
    "CodeDocsSearchTool": "crewai_tools.CodeDocsSearchTool",
    "ArxivPaperTool": "crewai_tools.ArxivPaperTool",
    "SerperDevTool": "crewai_tools.SerperDevTool",
    "ScrapeWebsiteTool": "crewai_tools.ScrapeWebsiteTool",
}


@dataclass
class ExecutionNode:
    step_id: str


@dataclass
class LoopNode:
    steps: List[str]
    condition: str
    max_iterations: int


def parse_execution_order(raw_order: List[Any]) -> List[ExecutionNode | LoopNode]:
    nodes: List[ExecutionNode | LoopNode] = []
    for item in raw_order:
        if isinstance(item, str):
            nodes.append(ExecutionNode(step_id=item))
            continue

        if hasattr(item, "loop"):
            loop = item.loop
            steps = list(loop.steps) if loop.steps else ([loop.step] if getattr(loop, "step", None) else [])
            nodes.append(
                LoopNode(
                    steps=steps,
                    condition=loop.condition,
                    max_iterations=loop.max_iterations,
                )
            )
            continue

        if isinstance(item, dict) and "loop" in item:
            loop_dict = item["loop"]
            steps = list(loop_dict.get("steps", []))
            if not steps and "step" in loop_dict:
                steps = [loop_dict["step"]]
            nodes.append(
                LoopNode(
                    steps=steps,
                    condition=loop_dict["condition"],
                    max_iterations=loop_dict.get("max_iterations", 2),
                )
            )
            continue

        raise ValueError(f"Unsupported execution_order item: {item}")
    return nodes


def _extract_topic_blindness(plan: PlanPayload) -> str:
    """Extract key concepts from plan subtask objectives for topic blindness."""
    concepts = []
    for subtask in plan.output.subtasks:
        obj = subtask.subtask_objective
        if obj:
            concepts.append(obj)
    if not concepts:
        return ""
    lines = ["TOPIC BLINDNESS — You have NEVER encountered these concepts:"]
    for i, c in enumerate(concepts, 1):
        lines.append(f"   {i}. {c}")
    lines.append(
        "You MUST NOT state, use, or reference ANY of the above ideas\n"
        "   until the instructor explicitly teaches them to you in THIS conversation.\n"
        "   If asked to predict, you should guess INCORRECTLY or say you don't know."
    )
    return "\n".join(lines)


def _build_student_backstory(plan: PlanPayload) -> str:
    learner = plan.input.learner
    skills = ", ".join(learner.skills) if learner.skills else "no listed skills"
    topic_blindness = _extract_topic_blindness(plan)
    return textwrap.dedent(
        f"""\
You are a learner interacting with a teaching system.
You are here because you encountered a programming problem
you cannot solve on your own.

Your background:
- Description: {learner.self_description}
- Skills: {skills}

The problem you need help with:
{plan.input.query}

You found this question because you could not solve it.
Respond accordingly.

Behavioral rules:

1. CORE CONSTRAINT
   You do NOT know the answer to the problem being taught.
   This is non-negotiable. You are someone who needs to learn
   this topic through this conversation.

   {topic_blindness}

2. REASONING FROM BACKGROUND
   You may use your declared skills as reference frames.
   For example, if you know Python and the topic is Java,
   you may say "In Python I would do X, is Java similar?"
   But your reasoning about the unfamiliar target topic
   should be tentative - use phrases like "I think maybe",
   "my guess would be", "I'm not sure but".

3. LEARNING PROGRESSION
   - Before any instruction: uncertain, may guess WRONG
   - After initial explanation: partial understanding, may still have gaps
   - After targeted feedback: clearer grasp, can apply to the specific case
   - After practice with correction: confident and correct

4. AUTHENTICITY
   - If confused, say so.
   - If an explanation helped, say what specifically helped and why.
   - If asked to write code, your first attempt SHOULD have bugs.
   - Do NOT produce perfect answers before receiving sufficient instruction.
   - Do NOT reference information not yet presented in this conversation."""
    ).strip()


def _build_student_goal(plan: PlanPayload) -> str:
    return (
        f"Learn and understand the solution to this question: {plan.input.query[:150]}. "
        "Engage authentically with the instructor and revise when corrected."
    )


def generate_flow_code(
    plan: PlanPayload,
    llm_model: str = "openai/gpt-4o-mini",
    student_llm_model: Optional[str] = None,
    max_interaction_rounds: int = 3,
    interactive_mode: str = "simulated_student",
    flow_runs_dir: str = "runs_generated",
) -> str:
    if student_llm_model is None:
        student_llm_model = "openai/gpt-3.5-turbo"

    exec_nodes = parse_execution_order(plan.output.execution_order)

    step_specs = {}
    step_to_subtask = {}
    for subtask in plan.output.subtasks:
        for step in subtask.steps:
            step_specs[step.id] = step
            step_to_subtask[step.id] = subtask

    condition_fields: Dict[str, str] = {}
    import re

    cond_pattern = re.compile(r"^([A-Za-z0-9\-]+)\.([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$")
    for node in exec_nodes:
        if isinstance(node, LoopNode):
            m = cond_pattern.match(node.condition.strip())
            if m:
                condition_fields[m.group(1)] = m.group(2)

    phases: List[tuple[str, Any]] = []
    current_linear: List[str] = []
    for node in exec_nodes:
        if isinstance(node, ExecutionNode):
            current_linear.append(node.step_id)
        else:
            if current_linear:
                phases.append(("linear", current_linear))
                current_linear = []
            phases.append(("loop", node))
    if current_linear:
        phases.append(("linear", current_linear))

    used_tools = set()
    for subtask in plan.output.subtasks:
        for step in subtask.steps:
            if step.tool:
                used_tools.add(step.tool)
    for agent in plan.output.agents:
        for tool in agent.tools:
            used_tools.add(tool)

    student_backstory = _build_student_backstory(plan)
    student_goal = _build_student_goal(plan)

    lines: List[str] = []
    lines.append('"""')
    lines.append("Auto-generated CrewAI flow from Plan JSON")
    lines.append(f"Query: {plan.input.query[:100]}")
    lines.append('"""')
    lines.append("")
    lines.append("from crewai import Agent, Crew, Process, Task")
    lines.append("from crewai.flow.flow import Flow, listen, router, start")
    lines.append("from pydantic import BaseModel, Field")
    lines.append("from typing import Any, Dict, List")
    lines.append("import importlib")
    lines.append("import os")
    lines.append("from types import MethodType")
    lines.append("import json")
    lines.append("import re")
    lines.append("from pathlib import Path")
    lines.append("from datetime import UTC, datetime")
    lines.append("")
    lines.append(f"TOOL_REGISTRY = {TOOL_REGISTRY!r}")
    lines.append("if os.getenv('OPENAI_API_BASE') and not os.getenv('OPENAI_BASE_URL'):")
    lines.append("    os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_API_BASE', '')")
    lines.append("")

    lines.append(f"MAX_INTERACTION_ROUNDS = {max_interaction_rounds}")
    lines.append("STUDENT_AGENT_ROLE = 'Student Learner'")
    lines.append(f"INTERACTIVE_MODE = {interactive_mode!r}")
    lines.append(f"FLOW_RUNS_DIR = {flow_runs_dir!r}")
    lines.append(f"CONDITION_FIELDS = {condition_fields!r}")
    lines.append("TOOL_ARG_HINTS = {")
    lines.append("    'CodeDocsSearchTool': \"Use EXACT keys: {'search_query': <string>, 'docs_url': <string>}. Do not send 'description' key.\",")
    lines.append("    'CodeInterpreterTool': \"Use EXACT keys: {'code': <string>, 'libraries_used': <string or list>}.\",")
    lines.append("    'GithubSearchTool': \"Use EXACT keys: {'search_query': <string>, 'github_repo': <string>, 'content_types': <list>}.\",")
    lines.append("}")
    lines.append("")

    lines.append("class TeachingFlowState(BaseModel):")
    lines.append("    task_results: dict = {}")
    lines.append("    conversation_history: list = []")
    lines.append("    execution_log: list = []")
    lines.append("")

    lines.append("def _build_agents() -> Dict[str, Agent]:")
    lines.append("    agents = {}")
    lines.append("")
    for agent in plan.output.agents:
        tool_list: List[str] = []
        for tool in agent.tools:
            if tool in TOOL_REGISTRY:
                tool_list.append(f"_init_tool('{tool}')")
        tools_str = f"[t for t in [{', '.join(tool_list)}] if t is not None]" if tool_list else "[]"
        lines.append(f"    agents[{agent.agent_role!r}] = Agent(")
        lines.append(f"        role={agent.agent_role!r},")
        lines.append(f"        goal={agent.goal!r},")
        lines.append(f"        backstory={agent.description!r},")
        lines.append(f"        tools={tools_str},")
        lines.append("        allow_delegation=False,")
        lines.append("        verbose=True,")
        lines.append(f"        llm={llm_model!r},")
        lines.append("    )")
        lines.append("")

    lines.append("    if INTERACTIVE_MODE == 'simulated_student':")
    lines.append("        agents[STUDENT_AGENT_ROLE] = Agent(")
    lines.append("            role='Student Learner',")
    lines.append(f"            goal={student_goal!r},")
    lines.append(f"            backstory={student_backstory!r},")
    lines.append("            tools=[],")
    lines.append("            allow_delegation=False,")
    lines.append("            verbose=True,")
    lines.append(f"            llm={student_llm_model!r},")
    lines.append("        )")
    lines.append("    return agents")
    lines.append("")

    lines.append("STEP_CONFIGS = {")
    for sid, spec in step_specs.items():
        tool_code = "None"
        if spec.tool and spec.tool in TOOL_REGISTRY:
            tool_code = f"[{spec.tool!r}]"
        desc = f"[Step {spec.id}] {spec.objective}\\n\\nInstruction:\\n{spec.instruction}"
        desc = desc.replace('"', '\\"')
        subtask = step_to_subtask[sid]
        lines.append(f"    {sid!r}: {{")
        lines.append(f'        "objective": {spec.objective!r},')
        lines.append(f'        "instruction": {spec.instruction!r},')
        lines.append(f'        "description": "{desc}",')
        lines.append(f'        "expected_output": {spec.expected_output!r},')
        lines.append(f'        "agent_role": {spec.agent!r},')
        lines.append(f'        "requires_human_input": {spec.requires_human_input},')
        lines.append(f'        "tools": {tool_code},')
        lines.append(f'        "depends_on": {spec.depends_on!r},')
        lines.append(f'        "subtask_id": {subtask.id!r},')
        lines.append("    },")
    lines.append("}")
    lines.append(f"AGENT_CONFIGS = {[a.model_dump() for a in plan.output.agents]!r}")
    lines.append(f"QUERY = {plan.input.query!r}")
    lines.append("")

    helpers = r'''
HISTORY_CHAR_BUDGET = 12000
MAX_EMPTY_RETRIES = 2


def _format_history(conversation_history: list, max_chars: int = None,
                    pinned_step_ids: list = None, loop_step_ids: list = None) -> str:
    if not conversation_history:
        return ''
    if max_chars is None:
        max_chars = HISTORY_CHAR_BUDGET
    pinned = set(pinned_step_ids or [])
    loop_sids = set(loop_step_ids or [])

    # Separate pinned vs unpinned, and for loop steps keep only the latest iteration
    seen_loop_steps = {}
    entries_with_index = []
    for i, entry in enumerate(conversation_history):
        sid = entry.get('step_id', '')
        is_pinned = sid in pinned
        entries_with_index.append((i, entry, is_pinned))
        if loop_sids and sid in loop_sids:
            seen_loop_steps[sid] = i  # track last occurrence index

    # Build formatted parts with tiered truncation
    total = len(entries_with_index)
    parts = []
    for idx, (i, entry, is_pinned) in enumerate(entries_with_index):
        sid = entry.get('step_id', '')
        content = entry.get('content', '')
        role_tag = '[Teacher]' if entry['role'] != STUDENT_AGENT_ROLE else '[Student]'

        # For loop steps, aggressively summarize older iterations (keep only latest full)
        if loop_sids and sid in loop_sids and i != seen_loop_steps.get(sid):
            content = content[:150] + ('... [prior iteration]' if len(content) > 150 else '')
        elif is_pinned:
            pass  # keep full content for pinned entries
        else:
            recency = total - idx
            if recency <= 5:
                pass  # recent: full content
            elif recency <= 15:
                if len(content) > 800:
                    content = content[:800] + '... [truncated]'
            else:
                if len(content) > 200:
                    content = content[:200] + '... [truncated]'

        parts.append((i, f"{role_tag} (Step {sid}): {content}", is_pinned))

    # Apply character budget: drop oldest unpinned entries first
    total_len = sum(len(p[1]) for p in parts)
    while total_len > max_chars and parts:
        # Find oldest unpinned entry
        drop_idx = None
        for j, (i, text, is_p) in enumerate(parts):
            if not is_p:
                drop_idx = j
                break
        if drop_idx is None:
            break  # only pinned left, can't drop more
        total_len -= len(parts[drop_idx][1])
        parts.pop(drop_idx)

    if not parts:
        return ''
    formatted = '\n'.join(p[1] for p in parts)
    return '\n\nConversation history:\n' + formatted


_TOOL_CACHE = {}

def _init_tool(tool_name: str):
    if tool_name in _TOOL_CACHE:
        return _TOOL_CACHE[tool_name]
    spec = TOOL_REGISTRY.get(tool_name)
    if not spec:
        print(f"[WARN] Tool {tool_name} not found in TOOL_REGISTRY")
        return None

    try:
        module_name, class_name = spec.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:
        print(f"[WARN] Tool import failed for {tool_name}: {exc}")
        return None

    try:
        if tool_name == 'CodeDocsSearchTool':
            class _CodeDocsSearchCompatInput(BaseModel):
                search_query: str = Field(default='')
                docs_url: str
                description: str | None = None

            tool = cls()
            tool.args_schema = _CodeDocsSearchCompatInput
            original_run = tool._run

            def _compat_run(self, search_query: str = '', docs_url: str = '', description: str | None = None, **kwargs):
                q = search_query or description or kwargs.get('query', '')
                def _fallback_docs_url(query: str):
                    qq = (query or '').lower()
                    if 'iterator' in qq or 'scala' in qq:
                        return 'https://www.scala-lang.org/api/current/scala/collection/Iterator.html'
                    return 'https://spark.apache.org/docs/latest/api/scala/org/apache/spark/rdd/RDD.html'

                try:
                    return original_run(search_query=q, docs_url=docs_url)
                except Exception as exc:
                    msg = str(exc)
                    if '404' in msg or 'Unable to fetch documentation' in msg:
                        return original_run(search_query=q, docs_url=_fallback_docs_url(q))
                    raise

            tool._run = MethodType(_compat_run, tool)
            _TOOL_CACHE[tool_name] = tool
            return tool

        if tool_name == 'GithubSearchTool':
            token = os.getenv('GITHUB_TOKEN')
            if not token:
                print('[WARN] GithubSearchTool skipped: missing GITHUB_TOKEN')
                return None
            tool = cls(gh_token=token)
            _TOOL_CACHE[tool_name] = tool
            return tool
        tool = cls()
        _TOOL_CACHE[tool_name] = tool
        return tool
    except Exception as exc:
        print(f"[WARN] Tool init failed for {tool_name}: {exc}")
        return None


def _extract_json_dict(text: str):
    text = (text or '').strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(text)
    brace = re.search(r"\{[^{}]*\}", text)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _extract_field_value(text: str, field: str):
    obj = _extract_json_dict(text)
    if isinstance(obj, dict) and field in obj:
        return obj[field]
    pattern = re.compile(rf"{re.escape(field)}\s*[:=]\s*(true|false)", re.IGNORECASE)
    m = pattern.search(text or '')
    if m:
        return m.group(1).lower() == 'true'
    return None


def _parse_rhs(raw: str):
    raw = raw.strip()
    if raw.lower() == 'true':
        return True
    if raw.lower() == 'false':
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        if '.' in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def _evaluate_loop_condition(condition: str, task_results: dict):
    m = re.match(r"^([A-Za-z0-9\-]+)\.([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$", (condition or '').strip())
    if not m:
        return False
    step_id, field, rhs_raw = m.group(1), m.group(2), m.group(3)
    rhs = _parse_rhs(rhs_raw)
    val = _extract_field_value(str(task_results.get(step_id, '')), field)
    return val == rhs


def _build_log_entry(step_id, loop_context=None):
    """Build a structured execution log entry skeleton per Section 3.2."""
    cfg = STEP_CONFIGS[step_id]
    return {
        'step_id': step_id,
        'subtask_id': cfg.get('subtask_id', ''),
        'agent_role': cfg['agent_role'],
        'requires_human_input': cfg['requires_human_input'],
        'plan_instruction': cfg['instruction'],
        'plan_expected_output': cfg['expected_output'],
        'actual_interaction': {},
        'loop_context': loop_context or {
            'in_loop': False,
            'iteration': None,
            'exit_reason': None,
        },
    }


def _run_interactive_step(step_id, agents, conversation_history,
                          dep_context=None, loop_step_ids=None,
                          loop_context=None):
    cfg = STEP_CONFIGS[step_id]
    teacher_agent = agents[cfg['agent_role']]
    student_agent = agents[STUDENT_AGENT_ROLE]
    expected = cfg['expected_output']
    teacher_expected = f"Instructional message for step: {cfg['objective']}"
    pinned = list((dep_context or {}).keys())

    teacher_output = ''
    student_output = ''

    for round_num in range(MAX_INTERACTION_ROUNDS):
        history_str = _format_history(conversation_history,
                                      pinned_step_ids=pinned,
                                      loop_step_ids=loop_step_ids)
        teacher_desc = cfg['description'] if round_num == 0 else (
            cfg['description'] + '\n\nStudent previous response:\n' + (conversation_history[-1]['content'] if conversation_history else '')
        )
        # Inject dependency context on first round
        if round_num == 0 and dep_context:
            dep_section = '\n\n--- Required context from prior steps ---\n'
            for dep_id, dep_output in dep_context.items():
                dep_cfg = STEP_CONFIGS.get(dep_id, {})
                dep_section += f"\n[Step {dep_id} - {dep_cfg.get('objective', '')}]:\n{dep_output}\n"
            dep_section += '--- End prior step context ---\n'
            teacher_desc = dep_section + teacher_desc
        if cfg['tools']:
            tool_hints = [TOOL_ARG_HINTS.get(name, '') for name in cfg['tools']]
            tool_hints = [h for h in tool_hints if h]
            if tool_hints:
                teacher_desc += '\n\nTool Input Contract:\n- ' + '\n- '.join(tool_hints)
            tool_names = ', '.join(cfg['tools'])
            teacher_desc += f"\n\nIMPORTANT: You MUST use the {tool_names} tool to complete this step. Do NOT skip the tool call or attempt to answer without using it. Execute the code and report the actual results."
        if history_str:
            teacher_desc += history_str

        teacher_task = Task(description=teacher_desc, expected_output=teacher_expected, agent=teacher_agent, human_input=False)
        if cfg['tools']:
            teacher_task.tools = [t for t in [_init_tool(name) for name in cfg['tools']] if t is not None]

        # Student only sees teacher's current output, NOT the full history/context
        # This prevents the student from seeing prior step answers and "cheating"
        student_task = Task(
            description='The instructor has sent you the following message. Respond as a learner:\n',
            expected_output=expected,
            agent=student_agent,
            context=[teacher_task],
            human_input=False,
        )
        eval_task = Task(
            description='Evaluate learner response. Return SATISFACTORY or NEEDS_IMPROVEMENT.',
            expected_output='SATISFACTORY or NEEDS_IMPROVEMENT with reason',
            agent=teacher_agent,
            context=[teacher_task, student_task],
            human_input=False,
        )

        Crew(agents=[teacher_agent, student_agent], tasks=[teacher_task, student_task, eval_task], process=Process.sequential, verbose=True).kickoff()

        teacher_output = str(teacher_task.output) if teacher_task.output else ''
        student_output = str(student_task.output) if student_task.output else ''
        eval_output = str(eval_task.output) if eval_task.output else ''

        conversation_history.append({'step_id': step_id, 'role': cfg['agent_role'], 'content': teacher_output})
        conversation_history.append({'step_id': step_id, 'role': STUDENT_AGENT_ROLE, 'content': student_output})

        if 'satisfactory' in eval_output.lower() and 'needs_improvement' not in eval_output.lower():
            break

    # Build structured log entry
    log_entry = _build_log_entry(step_id, loop_context=loop_context)
    log_entry['actual_interaction'] = {
        'teacher_output': teacher_output,
        'student_response': student_output,
    }
    return conversation_history, log_entry


def _run_single_step(step_id, agents, conversation_history,
                     dep_context=None, loop_step_ids=None,
                     loop_context=None):
    cfg = STEP_CONFIGS[step_id]
    desc = cfg['description']
    pinned = list((dep_context or {}).keys())
    history = _format_history(conversation_history,
                              pinned_step_ids=pinned,
                              loop_step_ids=loop_step_ids)
    # Inject dependency context
    if dep_context:
        dep_section = '\n\n--- Required context from prior steps ---\n'
        for dep_id, dep_output in dep_context.items():
            dep_cfg = STEP_CONFIGS.get(dep_id, {})
            dep_section += f"\n[Step {dep_id} - {dep_cfg.get('objective', '')}]:\n{dep_output}\n"
        dep_section += '--- End prior step context ---\n'
        desc = dep_section + desc
    if cfg['tools']:
        tool_hints = [TOOL_ARG_HINTS.get(name, '') for name in cfg['tools']]
        tool_hints = [h for h in tool_hints if h]
        if tool_hints:
            desc += '\n\nTool Input Contract:\n- ' + '\n- '.join(tool_hints)
        tool_names = ', '.join(cfg['tools'])
        desc += f"\n\nIMPORTANT: You MUST use the {tool_names} tool to complete this step. Do NOT skip the tool call or attempt to answer without using it. Execute the code and report the actual results."
    if cfg['tools'] and history:
        desc += "\n\nNote: Find the learner's code in the conversation history above. Extract it carefully before proceeding with validation."
    if history:
        desc += history
    hint = CONDITION_FIELDS.get(step_id)
    if hint:
        desc += f"\n\nOutput JSON with boolean field '{hint}'. Example: {{\"{hint}\": true}}"
        task_expected = cfg['expected_output'] + f". Include JSON object with '{hint}' boolean field."
    else:
        task_expected = cfg['expected_output']

    # Execute with retry on empty output
    output_text = ''
    run_desc = desc
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        task = Task(description=run_desc, expected_output=task_expected, agent=agents[cfg['agent_role']], human_input=False)
        if cfg['tools']:
            task.tools = [t for t in [_init_tool(name) for name in cfg['tools']] if t is not None]
        result = Crew(agents=[agents[cfg['agent_role']]], tasks=[task], process=Process.sequential, verbose=True).kickoff()
        output_text = str(task.output) if task.output else str(result)
        if output_text.strip():
            break
        if attempt < MAX_EMPTY_RETRIES:
            print(f"[WARN] Step {step_id} produced empty output, retrying (attempt {attempt + 2}/{MAX_EMPTY_RETRIES + 1})")
            run_desc = f"IMPORTANT: Your previous attempt produced no output. You MUST provide a substantive response.\n\n{desc}"

    if not output_text.strip():
        output_text = f"[EMPTY OUTPUT - Step {step_id} failed to produce output after {MAX_EMPTY_RETRIES + 1} attempts]"
        print(f"[ERROR] Step {step_id} produced no output after all retries")

    conversation_history.append({'step_id': step_id, 'role': cfg['agent_role'], 'content': output_text})

    # Build structured log entry
    log_entry = _build_log_entry(step_id, loop_context=loop_context)
    log_entry['actual_interaction'] = {
        'agent_output': output_text,
    }
    return conversation_history, log_entry


def _run_steps(step_ids, agents, conversation_history=None, loop_step_ids=None,
               loop_context=None):
    if conversation_history is None:
        conversation_history = []
    last_result = ''
    step_outputs = {}
    step_log_entries = []
    completed_step_ids = {e['step_id'] for e in conversation_history} if conversation_history else set()
    for sid in step_ids:
        cfg = STEP_CONFIGS[sid]
        depends = cfg.get('depends_on', [])
        # Build dependency context: collect actual outputs of dependent steps
        dep_context = {}
        for dep in depends:
            if dep in step_outputs:
                dep_context[dep] = step_outputs[dep]
            else:
                # Search conversation_history for the dep's latest non-student output
                for entry in reversed(conversation_history):
                    if entry.get('step_id') == dep and entry.get('role') != STUDENT_AGENT_ROLE:
                        dep_context[dep] = entry.get('content', '')
                        break
            if dep not in dep_context:
                print(f"[WARN] Step {sid} depends on {dep} which has not been executed yet")
        # Warn about empty dependency outputs
        empty_deps = [d for d, c in dep_context.items() if not c.strip() or '[EMPTY OUTPUT' in c]
        if empty_deps:
            print(f"[WARN] Step {sid} has empty dependencies: {empty_deps}")

        if cfg['requires_human_input'] and INTERACTIVE_MODE == 'simulated_student':
            conversation_history, log_entry = _run_interactive_step(
                sid, agents, conversation_history,
                dep_context=dep_context or None,
                loop_step_ids=loop_step_ids,
                loop_context=loop_context)
            last_result = _latest_teacher_output(conversation_history, sid)
            step_outputs[sid] = last_result
        else:
            conversation_history, log_entry = _run_single_step(
                sid, agents, conversation_history,
                dep_context=dep_context or None,
                loop_step_ids=loop_step_ids,
                loop_context=loop_context)
            if conversation_history:
                last_result = conversation_history[-1]['content']
                step_outputs[sid] = last_result
        step_log_entries.append(log_entry)
    return last_result, conversation_history, step_outputs, step_log_entries


def _latest_teacher_output(conversation_history, step_id):
    for entry in reversed(conversation_history):
        if entry.get('step_id') == step_id and entry.get('role') != STUDENT_AGENT_ROLE:
            return entry.get('content', '')
    return ''


def _collect_interactions(conversation_history):
    rows = []
    for entry in conversation_history:
        sid = entry.get('step_id')
        cfg = STEP_CONFIGS.get(sid, {})
        if not cfg.get('requires_human_input'):
            continue
        role = 'STUDENT' if entry.get('role') == STUDENT_AGENT_ROLE else 'TEACHER'
        rows.append({'step_id': sid, 'role': role, 'content': entry.get('content', '')})
    return rows


def _write_outputs(flow):
    run_id = f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run_dir = Path(FLOW_RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    interactions = _collect_interactions(flow.state.conversation_history)
    interaction_md = ['# Interaction Transcript', '', f'- Run ID: {run_id}', '']
    for row in interactions:
        interaction_md.append(f"- [{row['role']}] {row['step_id']}: {row['content']}")
    (run_dir / 'interaction_log.md').write_text('\n'.join(interaction_md), encoding='utf-8')

    # Write structured execution log (Section 3.2 format)
    (run_dir / 'execution_log.json').write_text(
        json.dumps(flow.state.execution_log, indent=2, ensure_ascii=False), encoding='utf-8')

    final = {
        'run_id': run_id,
        'query': QUERY,
        'agents': AGENT_CONFIGS,
        'steps': [],
    }
    for sid, cfg in STEP_CONFIGS.items():
        final['steps'].append({
            'step_id': sid,
            'objective': cfg.get('objective', ''),
            'instruction': cfg.get('instruction', ''),
            'expected_output': cfg.get('expected_output', ''),
            'requires_human_input': cfg.get('requires_human_input', False),
            'agent_role': cfg.get('agent_role', ''),
            'output': _latest_teacher_output(flow.state.conversation_history, sid),
        })
    (run_dir / 'final_result.json').write_text(json.dumps(final, indent=2, ensure_ascii=True), encoding='utf-8')

    final_md = ['# Final Result', '', f'- Run ID: {run_id}', f'- Query: {QUERY}', '']
    for step in final['steps']:
        final_md.append(f"## {step['step_id']} - {step['objective']}")
        final_md.append(f"- Agent: {step['agent_role']}")
        final_md.append(f"- requires_human_input: {step['requires_human_input']}")
        final_md.append('- Output:')
        final_md.append('```text')
        final_md.append(step['output'] or '')
        final_md.append('```')
        final_md.append('')
    (run_dir / 'final_result.md').write_text('\n'.join(final_md), encoding='utf-8')

    print(f'Artifacts: {run_dir}')
'''
    lines.append(helpers)

    lines.append("class TeachingFlow(Flow[TeachingFlowState]):")
    lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    lines.append("        self.agents = _build_agents()")
    lines.append("")

    prev_method = None
    for pi, (ptype, pdata) in enumerate(phases):
        method_name = f"phase_{pi}"
        if ptype == "linear":
            step_ids = pdata
            lines.append("    @start()" if pi == 0 else f"    @listen({prev_method})")
            lines.append(f"    def {method_name}(self):")
            lines.append(f"        result, history, step_outputs, log_entries = _run_steps({step_ids!r}, self.agents, self.state.conversation_history)")
            lines.append("        self.state.conversation_history = history")
            lines.append("        self.state.execution_log.extend(log_entries)")
            lines.append(f"        for sid in {step_ids!r}:")
            lines.append("            self.state.task_results[sid] = step_outputs.get(sid, result)")
            lines.append("        return result")
            lines.append("")
            prev_method = method_name
            continue

        loop_node: LoopNode = pdata
        lines.append("    @start()" if pi == 0 else f"    @listen({prev_method})")
        lines.append(f"    def {method_name}(self):")
        lines.append(f"        iterations = 0")
        lines.append(f"        exit_reason = None")
        lines.append(f"        all_loop_log_entries = []")
        lines.append(f"        while iterations < {loop_node.max_iterations}:")
        lines.append(f"            print(f'[LOOP] phase_{pi} iteration {{iterations + 1}}/{loop_node.max_iterations}')")
        lines.append(f"            loop_ctx = {{'in_loop': True, 'iteration': iterations + 1, 'exit_reason': None}}")
        lines.append(
            f"            result, history, step_outputs, log_entries = _run_steps({loop_node.steps!r}, self.agents, self.state.conversation_history, loop_step_ids={loop_node.steps!r}, loop_context=loop_ctx)"
        )
        lines.append("            self.state.conversation_history = history")
        lines.append("            all_loop_log_entries.extend(log_entries)")
        lines.append(f"            for sid in {loop_node.steps!r}:")
        lines.append("                self.state.task_results[sid] = step_outputs.get(sid, result)")
        lines.append(f"            iterations += 1")
        lines.append(
            f"            condition_true = _evaluate_loop_condition({loop_node.condition!r}, self.state.task_results)"
        )
        cond_escaped = loop_node.condition.replace("'", "\\'")
        lines.append(f'            print(f"[LOOP] phase_{pi} condition=\'{cond_escaped}\' result={{condition_true}}")')
        lines.append(f"            if not condition_true:")
        lines.append(f"                exit_reason = 'condition_met'")
        lines.append(f"                break")
        lines.append(f"        if exit_reason is None:")
        lines.append(f"            exit_reason = 'max_iterations'")
        # Set exit_reason only on the last iteration's log entries (per spec: only last round fills exit_reason)
        lines.append(f"        last_iter = iterations")
        lines.append(f"        for entry in all_loop_log_entries:")
        lines.append(f"            if entry['loop_context']['iteration'] == last_iter:")
        lines.append(f"                entry['loop_context']['exit_reason'] = exit_reason")
        lines.append(f"        self.state.execution_log.extend(all_loop_log_entries)")
        lines.append(f"        print(f'[LOOP] phase_{pi} exited after {{iterations}} iteration(s), reason={{exit_reason}}')")
        lines.append("        return result")
        lines.append("")
        prev_method = method_name

    lines.append("if __name__ == '__main__':")
    lines.append("    flow = TeachingFlow()")
    lines.append("    flow.kickoff()")
    lines.append("    _write_outputs(flow)")

    return "\n".join(lines)


def write_generated_flow(
    plan: PlanPayload,
    output_path: Path,
    llm_model: str,
    student_llm_model: Optional[str],
    max_rounds: int,
    interactive_mode: str,
    flow_runs_dir: str,
) -> str:
    code = generate_flow_code(
        plan,
        llm_model=llm_model,
        student_llm_model=student_llm_model,
        max_interaction_rounds=max_rounds,
        interactive_mode=interactive_mode,
        flow_runs_dir=flow_runs_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code, encoding="utf-8")
    return code
