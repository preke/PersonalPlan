"""AIPOM baseline (M1 in v2 lineup, 2026-05-19).

Paper: Kim et al., AIPOM: Agent-aware Interactive Planning for
Multi-Agent Systems, EMNLP 2025 System Demonstrations.
Upstream repo: https://github.com/megagonlabs/aipom

Method preserved (single-shot, batch-friendly subset):
  - Agent-aware planning: explicit agent-to-task assignment
  - Typed I/O dependencies: depends_on reflects real data flow
  - Inspectable DAG: subtasks group conceptually-related steps,
    parallelizable steps adjacent in execution_order
  - No hidden coordination: all coordination via depends_on /
    expected_output / instruction

Method NOT preserved (out of scope for batch evaluation):
  - Conversational refinement loop (requires human NL feedback)
  - Visual DAG editing (requires UI)

Output: §9 plan directly (same schema as all other baselines).
Backbone: qwen3-32b (T5 default per v1 baseline design).
"""
