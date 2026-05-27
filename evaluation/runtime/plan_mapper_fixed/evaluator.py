"""
Execution Verification Evaluator (Spec Section 4-7).

Takes execution_log.json + plan JSON + optional accepted_answer,
performs FOUR checks via an evaluator LLM, and outputs a structured report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _build_evaluator_prompt(
    plan_json: Dict[str, Any],
    execution_log: List[Dict[str, Any]],
    accepted_answer: Optional[str] = None,
) -> str:
    """Build the evaluator system + user prompt."""

    # Build plan summary for context
    plan_steps_info = []
    for subtask in plan_json["output"]["subtasks"]:
        for step in subtask["steps"]:
            plan_steps_info.append({
                "step_id": step["id"],
                "subtask_id": subtask["id"],
                "subtask_objective": subtask.get("subtask_objective", ""),
                "agent": step["agent"],
                "objective": step["objective"],
                "instruction": step["instruction"],
                "expected_output": step["expected_output"],
                "requires_human_input": step.get("requires_human_input", False),
                "depends_on": step.get("depends_on", []),
                "tool": step.get("tool"),
            })

    system_prompt = """\
You are an execution verification evaluator for a teaching plan system.
You will receive a teaching plan, the execution log from running that plan, and an accepted answer.
Your job is to perform FOUR checks and output a structured evaluation report.

IMPORTANT PRINCIPLES:
- Be strict and objective. Evaluate based on evidence in the logs, not assumptions.
- The student agent is an LLM, not a real learner. Do NOT try to infer whether it "truly learned."
  Instead, treat the student's response sequence as an observable behavioral trace.
- If the student's outputs shift from incorrect toward correct after specific teaching steps,
  that is evidence that the plan's instructional flow is working.
- Output the report in English only.
- Use the EXACT output format specified at the end."""

    user_prompt = f"""\
## Plan Information

Query: {plan_json['input']['query']}

Learner Profile: {json.dumps(plan_json['input']['learner'], ensure_ascii=False)}

Accepted Answer (correctness reference): {accepted_answer if accepted_answer else "Not provided."}

## Plan Steps (design intent)

{json.dumps(plan_steps_info, indent=2, ensure_ascii=False)}

## Execution Order

{json.dumps(plan_json['output']['execution_order'], indent=2, ensure_ascii=False)}

## Execution Log (actual runtime results)

{json.dumps(execution_log, indent=2, ensure_ascii=False)}

---

Please perform the following FOUR checks and output the evaluation report.

### Check 1: Instruction Fidelity

For each learner-facing step (requires_human_input=true), compare plan_instruction against teacher_output.
For loop steps, only check the LAST iteration.

Judge two aspects per step:

(a) Action Alignment: Read plan_instruction, extract the core action intent (e.g., "elicit a prediction", "explain X using Y as analogy", "ask student to write code"). Then check: does teacher_output carry out that intent?
- "aligned": the agent does what the instruction asks. Wording differences are fine.
- "partial_deviation": the agent does the requested action BUT adds unrequested content (e.g., reveals the answer early).
- "severe_deviation": the agent does something fundamentally different (e.g., instruction says "elicit prediction", agent gives a lecture).

(b) Boundary Compliance: Check whether teacher_output teaches content that belongs to LATER steps. Look at subsequent steps' instructions and subtask_objectives.
- "within_bounds": only covers current step's content.
- "minor_overstep": briefly mentions a later topic without teaching it.
- "severe_overstep": teaches a later step's core content, making that step redundant.

Verdict: PASS if no step has severe_deviation or severe_overstep. FAIL otherwise.

### Check 2: Workflow Completeness

(a) Step Coverage: Go through execution_order. For each step ID, verify the log has at least one entry with non-empty actual_interaction (not just error messages like "No result variable found" or "An error occurred").

(b) Loop Behavior: For each loop in execution_order, check:
1. Does the log have entries for it?
2. Does the final iteration have an exit_reason ("condition_met" or "max_iterations")?
3. If "condition_met", does the teacher's output contain a judgment consistent with the condition being satisfied?
4. If the loop exited after only 1 iteration via "condition_met": verify the judgment is substantiated, not just a default pass-through.

(c) Information Flow: For each step with depends_on, verify upstream step produced relevant output. Only flag obvious breaks — upstream output is empty, missing, or clearly irrelevant.

(d) Tool Execution Reliability: For steps that declare a tool (especially CodeInterpreterTool), check whether the actual output contains real execution results vs. speculative language ("Expected output:", "should see:", "When you run this"). If the agent describes expected behavior instead of showing actual execution output, flag as "tool_execution_uncertain".

Verdict: PASS if coverage complete, loops normal, flow connected, tools reliable. FAIL if any fails.

### Check 3: Interaction Quality

(a) Participation — rate each student_response:
- "substantive": directly addresses the teacher's question with specific content — a prediction, explanation, code attempt, reasoning, or concrete question.
- "shallow": acknowledges the teacher but says nothing specific. Could be pasted into any conversation unchanged.
- "non_participatory": ignores the teacher's question, goes off-topic, or echoes the question back.

(b) Interaction Pattern — holistic assessment across ALL learner-facing interactions:
- "reasonable": teacher guides → student attempts → teacher progresses. Student responses vary. A student who gives incorrect/incomplete answers early but improves after teaching is a POSITIVE signal.
- "degenerate": Three failure modes:
  (1) Persistently helpless: every response is "I don't know" with no improvement.
  (2) Perfect from the start: student gives fully correct and complete answers from the VERY FIRST learner-facing step, BEFORE any teaching has occurred. This means the student agent's constraint is not working.
  (3) Repetitive: responses are substantively identical across steps.

IMPORTANT DISTINCTION:
  "Student gets better over time" = Reasonable (teaching is working)
  "Student is perfect from the beginning" = Degenerate (constraint failed)
  The dividing line is WHETHER the student showed any gap BEFORE the teaching that led to the improvement.

Verdict: PASS if >70% substantive AND pattern is reasonable. FAIL if >30% shallow/non-participatory OR pattern is degenerate.

### Check 4: Content Correctness and Guidance Effectiveness

(a) Core Solution Coverage:
Step 1: From the accepted answer, extract the core solution in one sentence.
Step 2: Scan ALL teacher_output and agent_output fields. Was this core solution explicitly taught?
- "covered": substantively taught in at least one step.
- "mentioned_only": briefly referenced but not the teaching focus.
- "not_covered": never appears anywhere.

(b) Guidance Trace — examine student_response fields in execution order (including every loop iteration):

EARLY (first subtask): Does the student show uncertainty, error, or incompleteness about the target topic?
- If yes: gap confirmed.
- If the student gives fully correct answers from the first step: mark as "no observable gap."

MID (middle subtasks, loop iterations): Does the student's response improve AFTER specific teaching? The key test is causal traceability:
- Pick a student response that improved compared to earlier.
- Can you point to a specific preceding teacher_output that introduced the knowledge the student is now using?

LATE (last subtask): Does the student's final code or explanation align with the accepted answer's approach?

Rate:
- "clear_guidance": early gap → teaching → observable shift toward correct direction. At least one improvement traceable to a specific teaching step.
- "weak_guidance": some directional shift but cannot be clearly traced to specific teaching steps.
- "no_guidance_signal": student responses do not change; OR student was correct from the start; OR changes are unrelated to teaching content.

(c) Final Output Correctness: Compare the last subtask's outputs against the accepted answer.
- "correct": final output presents a correct solution aligned with accepted answer.
- "partially_correct": right direction but meaningful errors or omissions.
- "incorrect": wrong solution, contradicts accepted answer, or no solution reached.

Verdict:
- PASS requires ALL of: coverage=covered, guidance=clear or weak, final_output=correct or partially_correct.
- FAIL if ANY of: coverage=not_covered, guidance=no_guidance_signal, final_output=incorrect.
- Borderline: coverage "mentioned_only" + final_output "correct" + guidance "clear" → PASS.

---

## Output Format

Use this EXACT structure. Do NOT add extra sections or change the field names.

```
Check 1: Instruction Fidelity
  Verdict: [PASS / FAIL]
  Per-step results (learner-facing steps only, last iteration for loop steps):
    - [step_id]:
      Action alignment: [aligned / partial_deviation / severe_deviation]
      Boundary compliance: [within_bounds / minor_overstep / severe_overstep]
      Notes: <if any deviation or overstep, briefly describe>
  Summary: <if FAIL, state whether individual or systematic>

Check 2: Workflow Completeness
  Verdict: [PASS / FAIL]
  Step coverage: [complete / incomplete]
    <if incomplete, list missing/empty step IDs>
  Loop behavior: [normal / abnormal]
    - Loop [step_ids]: exit_reason=<reason>, iterations=<N>, consistent_with_content=[yes/no]
      <if 1-iteration exit with max>=2, state whether judgment is substantiated>
  Information flow: [connected / broken]
    <if broken, list each break>
  Tool execution: [reliable / uncertain]
    <if uncertain, list step IDs where tool output appears speculative rather than actual>

Check 3: Interaction Quality
  Verdict: [PASS / FAIL]
  Engagement statistics:
    Substantive: X / N learner-facing steps
    Shallow: X / N
    Non-participatory: X / N
  Interaction pattern: [reasonable / degenerate]
    <if degenerate, state which failure mode: persistently_helpless / perfect_from_start / repetitive>
    <describe evidence: cite the first learner-facing step's student response — was it already correct?>

Check 4: Content Correctness and Guidance Effectiveness
  Verdict: [PASS / FAIL]
  Core solution: <one sentence from accepted answer>
  Coverage: [covered / mentioned_only / not_covered]
    Evidence: <which steps taught it>
  Guidance trace: [clear_guidance / weak_guidance / no_guidance_signal]
    Evidence: <describe student trajectory, cite specific steps and iterations,
    state which improvements trace to which teaching actions.
    If student was correct from the start, state "no observable gap — cannot establish guidance trace.">
  Final output: [correct / partially_correct / incorrect]
    Evidence: <cite final steps, relate to accepted answer>

Overall verdict: [execution_verified / execution_failed]
Failure attribution (only if failed): <which check failed, root cause, recommended modification target>
```"""

    return system_prompt, user_prompt


def evaluate_execution(
    plan_path: str | Path,
    execution_log_path: str | Path,
    accepted_answer: Optional[str] = None,
    evaluator_model: str = "gpt-4o-mini",
    output_path: Optional[str | Path] = None,
) -> str:
    """Run the four-check evaluation on an execution log.

    Returns the evaluation report text.
    """
    plan_json = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    execution_log = json.loads(Path(execution_log_path).read_text(encoding="utf-8"))

    # Try to get accepted_answer from plan if not provided
    if accepted_answer is None:
        accepted_answer = plan_json.get("input", {}).get("accepted_answer")

    system_prompt, user_prompt = _build_evaluator_prompt(
        plan_json, execution_log, accepted_answer
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=evaluator_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=8192,
        timeout=120,
    )

    report = response.choices[0].message.content

    # Write report if output path specified
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Evaluation report written to: {out}")

    return report
