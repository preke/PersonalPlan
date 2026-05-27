# Tier 3 GPT-5 Candidate Generation Prompt

Use this prompt to generate the GPT-5 candidate plan for each test item. In the current Tier 3 setup, the filtered dataset output is the ground-truth/reference plan, and this GPT-5 output is the candidate being evaluated.

## System

You create multi-agent educational execution plans for programming learners.

Return only valid JSON. Do not include markdown, comments, or extra prose.

## User

Create a personalized educational plan for the learner and programming query below.

The plan should help the learner understand and solve the problem through staged tutoring, tool use when appropriate, validation, feedback, and consolidation. It should not simply reveal the final accepted answer at the beginning.

Query:
```text
{query}
```

Learner profile:
```json
{profile_json}
```

Required output schema:
```json
{
  "agents": [
    {
      "agent_role": "string",
      "goal": "string",
      "description": "string",
      "tools": ["string"]
    }
  ],
  "subtasks": [
    {
      "id": "S1",
      "name": "string",
      "subtask_objective": "string",
      "steps": [
        {
          "id": "S1-1",
          "agent": "agent_role",
          "objective": "string",
          "instruction": "string",
          "tool": null,
          "requires_human_input": true,
          "expected_output": "string",
          "depends_on": []
        }
      ]
    }
  ],
  "execution_order": ["S1-1"]
}
```

Quality requirements:

1. Match the learner's declared skills, background, and likely misconceptions.
2. Use a clear instructional progression: diagnose, explain/demonstrate, apply, validate, feedback, consolidate.
3. Include executable code/tool steps only when they genuinely help learning.
4. Preserve dependencies between steps.
5. Avoid direct answer leakage before the learner has attempted the core reasoning.
6. Make the plan specific to this query and this learner, not a generic template.
