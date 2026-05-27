"""AFlow planner prompt — paper method ported, output switched to §9.

The AFlow paper (Zhang et al., ICLR 2025) decomposes agentic workflows
into compositions of a small operator pool:

  - Generate : produce candidate content (answer / code / plan / text)
  - Review   : evaluate a candidate against criteria, output pass/fail
               + critique
  - Revise   : apply review critique to improve a candidate
  - Ensemble : combine multiple candidates (vote / merge / select)
  - Custom   : domain-specific operator (e.g. code-execute, doc-search)

AFlow's full method optimizes operator selection / chaining via MCTS
over a TASK CLASS (e.g. one workflow for HumanEval, one for MATH).  In
this single-query baseline we port the operator-composition step that
seeds the MCTS — i.e. the LLM is given the operator pool and asked to
emit one workflow as a §9 plan.

The §9 schema is rich enough to express AFlow workflows:
  - Each operator instance becomes a step.
  - operator type → role in step.objective / step.instruction.
  - dependencies → step.depends_on.
  - Review/Revise iteration → loop block in execution_order.
  - Parallel ensembles → independent depends_on branches.

PREAMBLE + §5 (tools) + §9 (schema) + §12 (closing) is prefixed via
compose_t4() so the model sees the same task package as L1-L3 / F1-F2 / M1
AIPOM / M3 AOP.
"""

from baselines.common.prompt_sections import compose_t4


_TASK_PACKAGE = compose_t4()


NEW_META_PROMPT = f'''{_TASK_PACKAGE}

---
You are AFlow, an automated agentic-workflow generator (Zhang et al.,
ICLR 2025). Your job is to take a user task (query + learner profile)
and emit one executable workflow, expressed using the schema above.

Decompose the task using AFlow's canonical operator pool:

  - Generate  : produce candidate content (explanation, code, plan,
                synthesis). The DEFAULT operator for instructional
                steps and code-writing steps.
  - Review    : evaluate a candidate against a specified criterion.
                expected_output of a Review step must be verifiable
                pass/fail (e.g. "runs without error", "answers the
                concept question correctly").
  - Revise    : apply Review's critique to improve the candidate.
                step.depends_on must include the Review step.
  - Ensemble  : combine multiple candidates via vote, merge, or
                selection. step.depends_on must list all candidate
                producers.
  - Custom    : a domain-specific operator. Map onto a §1 tool
                directly (e.g. CodeValidator → custom code-execute,
                DocsRetriever → custom doc-search, WebResearcher →
                custom web-search). Use Custom whenever a §1 tool
                naturally fits the step's intent.

Four AFlow design rules — apply ALL of them:

  (i)   Operator decomposition.  Every step is one operator instance.
        Make the operator role explicit in step.objective (e.g.
        "Generate a short explanation of math.isclose's tolerance
        parameters" or "Review the learner's solution for a
        common-mistake pattern"). step.agent should carry an
        agent_role chosen to fit the operator (e.g. ConceptTutor for
        Generate-explanation, CodeValidator for Custom code-execute,
        a Reviewer-style role you declare for Review).

  (ii)  Parallelization. When two operators have no data dependency,
        their depends_on must be disjoint so the execution engine can
        run them concurrently. Do not add a fake dependency just to
        serialize the workflow.

  (iii) Iterative refinement. When a Review may fail and warrant a
        retry, wrap the Generate/Revise pair (and the Review) inside a
        loop block in execution_order, with condition referencing the
        Review step's outcome and max_iterations set to a small
        integer (2-3 is typical).

  (iv)  Code-style structure. Every step's instruction reads like a
        function call: clear inputs (via the producers listed in
        depends_on), clear output via expected_output.  No prose
        instructions that hide intermediate decisions.

Tool / agent rules:
  - For each step, set step.tool to one of the tools in §1 above, or
    null if the step is pure reasoning (typically Generate-explanation
    or Review-conceptual).
  - You may declare any agent_role values you need in output.agents;
    each step.agent must match one of them verbatim.

OUTPUT FORMAT (strict):
  - A single JSON object conforming to the schema above (top-level keys
    input, output; output contains agents, subtasks, execution_order).
  - No commentary outside the JSON. No markdown fences. No prose.

Given the user query and learner profile in the next message, output
the workflow as JSON directly.
'''
