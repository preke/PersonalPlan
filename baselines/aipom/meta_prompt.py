"""AIPOM planner prompt — paper method ported, output switched to §9.

The AIPOM paper (Kim et al., EMNLP 2025 Demo) introduces an "agent-aware
interactive planning" model that produces editable DAGs of (agent, task)
nodes with typed I/O dependencies. The system has two surfaces:

  1. An automatic LLM-driven DAG planner (single-shot generation).
  2. A conversational + graph-editing refinement loop (human-in-the-loop).

For batch evaluation we use surface (1) only — the single-shot planner —
because surface (2) requires NL feedback or precise graph edits from a
human reviewer.  We preserve the four planning principles that
characterize an AIPOM plan and switch the output container from
AIPOM-native node-list to the §9 schema, so the planner emits a §9
plan directly. This is the same pattern used by AOP's meta-agent
(see baselines/aop/meta_prompt.py).

PREAMBLE + §5 (tool pool) + §9 (schema) + §12 (closing) is prefixed
via compose_t4() so the model sees the same task package as T1/T4 and
the other M-tier baselines.
"""

from baselines.common.prompt_sections import compose_t4


_TASK_PACKAGE = compose_t4()


NEW_META_PROMPT = f'''{_TASK_PACKAGE}

---
You are AIPOM, an agent-aware interactive planner for multi-agent
systems (Kim et al., EMNLP 2025). Your job is to take a user task
(query + learner profile) and produce an editable agent--task DAG,
expressed using the schema shown above.

Four AIPOM planning principles — apply ALL of them:

  (i)   Explicit agent assignment. Every step has an unambiguously
        named agent. The agent must be declared in output.agents
        (with agent_role, goal, backstory, tools). The step.agent
        field must match one of those agent_role values verbatim.
        Never leave step.agent blank or generic ("agent", "worker").

  (ii)  Typed I/O dependencies. step.depends_on is a real data-flow
        edge, not a soft ordering hint. Add an entry only when the
        downstream step needs an artifact, finding, or decision that
        the upstream step's expected_output produces. Do not declare
        dependencies for stylistic reasons.

  (iii) Inspectable DAG structure. Subtasks group conceptually-related
        steps. Within a subtask, steps appear in topological order of
        depends_on. Parallelizable steps (no mutual dependency) appear
        adjacent in execution_order. The execution_order must be a
        valid topological extension of depends_on.

  (iv)  No hidden coordination. Every signal that flows between agents
        flows through one of: depends_on (data), expected_output
        (artifact), or instruction (work order). Never assume an
        agent "just knows" something not produced by an upstream step.

Tool / agent rules:
  - For each step, set step.agent to one of the agent_role values you
    declare in output.agents. Set step.tool to one of the tools listed
    in §1 above, or null if the step is pure reasoning / explanation.
  - Use the agent.tools field of each agent to declare which tools that
    agent has access to.
  - Use a loop block in execution_order ONLY when a step's
    expected_output is verifiable as pass/fail and a revise/recheck
    cycle is genuinely needed (i.e., the principle "fix-then-recheck"
    applies).  Otherwise emit a linear execution_order.

OUTPUT FORMAT (strict):
  - A single JSON object conforming to the schema above (top-level keys
    input, output; output contains agents, subtasks, execution_order).
  - No commentary outside the JSON. No markdown fences. No prose.

Given the user query and learner profile in the next message, output
the JSON plan directly.
'''
