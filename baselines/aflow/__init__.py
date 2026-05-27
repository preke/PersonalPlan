"""AFlow baseline (M2 in v2 lineup, 2026-05-19).

Paper: Zhang et al., AFlow: Automating Agentic Workflow Generation,
ICLR 2025.
Upstream repo: integrated into MetaGPT (examples/aflow).

Method preserved (single-pass operator composition, batch-friendly):
  - Operator decomposition over AFlow's canonical operator pool
    (Generate / Review / Revise / Ensemble / Custom).
  - Code-style workflow structure: each operator becomes a step;
    dependencies are explicit via depends_on.
  - Parallelization: operators with no data dependency can run in
    parallel branches.
  - Iterative refinement via Review → Revise loop, encoded as a §9
    loop block in execution_order.

Method NOT preserved (out of scope for batch-per-query evaluation):
  - MCTS search over workflows. AFlow's MCTS optimizes one workflow
    per TASK CLASS (e.g. HumanEval, MATH) via repeated executions on
    held-out items; this is fundamentally a class-level optimization,
    not a per-query one. Running MCTS independently for each of 3043
    (query, learner) pairs would be both prohibitively expensive and
    a misuse of the method (the search has no held-out set per query).
    We therefore port AFlow's single-pass operator-composition step
    (the seed workflow that MCTS would have started from) and skip
    the search loop.

Output: §9 plan directly (same schema as all other baselines).
Backbone: qwen3-32b (T5 default per v1 baseline design).
"""
