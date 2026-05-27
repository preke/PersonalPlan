#!/usr/bin/env python3
"""Render before/after teacher_prompt on a real plan with CodeInterpreterTool step.

Pure test: no LLM calls. Just constructs the prompt string the same way runtime.py
does and prints both versions side by side so we can eyeball what changed.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_mapper_fixed.runtime import CODE_TOOL_NUDGE, TOOL_ARG_HINTS


def render_old(step_objective, step_instruction, step_expected, context_text, step_tool):
    return (
        f"Objective: {step_objective}\n"
        f"Instruction: {step_instruction}\n"
        f"Expected Output: {step_expected}\n"
        f"Context:\n{context_text}\n"
        + (f"Required Tool: {step_tool}. Use this tool when needed.\n" if step_tool else "")
        + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step_tool, '')}\n" if step_tool else "")
        + "Provide your instructional message to the student."
    )


def render_new(step_objective, step_instruction, step_expected, context_text, step_tool):
    return (
        f"Objective: {step_objective}\n"
        f"Instruction: {step_instruction}\n"
        f"Expected Output: {step_expected}\n"
        f"Context:\n{context_text}\n"
        + ((f"Required Tool: {step_tool}. You MUST invoke this tool to compile/run the code in the sandbox; do not describe expected output.\n"
            if step_tool == "CodeInterpreterTool"
            else f"Required Tool: {step_tool}. Use this tool when needed.\n")
           if step_tool else "")
        + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step_tool, '')}\n" if step_tool else "")
        + (CODE_TOOL_NUDGE if step_tool == "CodeInterpreterTool" else "")
        + "Provide your instructional message to the student."
    )


def find_sample_step():
    """Find a real plan with a CodeInterpreterTool step (prefer compiled lang)."""
    repo = Path(__file__).resolve().parents[2]
    dataset = repo / "multi_agent_dataset_filtered_qap.jsonl"
    samples = {"compiled": None, "python": None, "other_tool": None, "no_tool": None}
    with dataset.open() as f:
        for line in f:
            r = json.loads(line)
            plan = r.get("plan") or {}
            query = (plan.get("input") or {}).get("query") or ""
            for st in (plan.get("output") or {}).get("subtasks", []):
                for step in st.get("steps", []):
                    tool = step.get("tool")
                    rec = {"query": query, "step": step, "subtask_id": st.get("id")}
                    if tool == "CodeInterpreterTool":
                        # crude lang detect from query
                        ql = query.lower()
                        if any(k in ql for k in ("java", "c++", "kotlin", "swift", "rust", " c ")):
                            if samples["compiled"] is None:
                                samples["compiled"] = rec
                        elif samples["python"] is None:
                            samples["python"] = rec
                    elif tool and samples["other_tool"] is None:
                        samples["other_tool"] = rec
                    elif tool is None and samples["no_tool"] is None:
                        samples["no_tool"] = rec
            if all(samples.values()):
                break
    return samples


def diff_show(label, sample):
    if sample is None:
        print(f"\n=== {label}: no sample found ===\n")
        return
    step = sample["step"]
    print(f"\n{'=' * 78}")
    print(f"=== {label}")
    print(f"=== query: {sample['query'][:90]}...")
    print(f"=== step:  {step['id']} (subtask {sample['subtask_id']}, tool={step.get('tool')})")
    print('=' * 78)

    obj = step.get("objective", "")
    instr = step.get("instruction", "")[:200]
    expected = step.get("expected_output", "")[:120]
    context = "[Concept Tutor] (Step S1-1): The learner predicted that strings compare with =="

    old = render_old(obj, instr, expected, context, step.get("tool"))
    new = render_new(obj, instr, expected, context, step.get("tool"))

    print("\n--- OLD prompt (before change) ---")
    print(old)
    print("\n--- NEW prompt (after change) ---")
    print(new)
    print(f"\n[Δ length]  OLD: {len(old)} chars  →  NEW: {len(new)} chars  (+{len(new)-len(old)})")


def main():
    samples = find_sample_step()
    diff_show("CASE A · CodeInterpreterTool + compiled-lang query", samples["compiled"])
    diff_show("CASE B · CodeInterpreterTool + python query",        samples["python"])
    diff_show("CASE C · other tool (not affected)",                 samples["other_tool"])
    diff_show("CASE D · no tool (not affected)",                    samples["no_tool"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
