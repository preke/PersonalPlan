# Reproducing MAP-PPL paper results

> **Status:** placeholders — fill the exact commands once each component is
> smoke-tested end-to-end from a fresh clone.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt   # TODO: generate one — currently missing
bash scripts/download_data.sh
```

Required API keys (export or place in `.env` at the repo root):

| variable | used by |
|---|---|
| `OPENAI_API_KEY` | execution layer (Teacher/Student via CrewAI), some judges |
| `ANTHROPIC_API_KEY` | dataset construction (Claude Sonnet 4.6) |
| `DASHSCOPE_API_KEY` | Qwen-based baselines (`baselines/autogen_qwen3/`, `autoagents/`, `genmentor/`) |
| `A8_API_KEY` | Tier 3 cross-method judges (GPT-5 / Claude Opus 4.6 / Gemini 3 Pro) |
| `POE_API_KEY` | optional, used by some construction-time annotators |

## 1. Dataset construction (Section 3)

Recreates the 3,043-plan v15 release from Stack Overflow dumps.

```bash
# TODO: end-to-end command. See construction/pipeline/{task_1,task_2,task_3}/
#       and construction/analysis/ for the per-stage scripts.
```

## 2. Hierarchical SFT (Section 4)

PAD / SDP / Joint Alignment, small and big model variants.

```bash
# Small model (e.g. Qwen3-8B):
bash training/sft/train_pad.sh
bash training/sft/train_sdp.sh
bash training/sft/train_joint.sh

# Big model: see training/sft/bigmodel_training/
```

## 3. GRPO with 4 reward families (Section 4.2)

```bash
python training/grpo/build_grpo_prompts.py            # build prompt set
python training/grpo/build_counterfactual_cache.py    # build counterfactual cache
bash training/grpo/train_grpo.sh                      # GRPO training
```

Reward weights and dynamic-weighting how-to:
[`training/grpo/dynamic_reward_weighting_howto.html`](../training/grpo/dynamic_reward_weighting_howto.html).

## 4. Plan execution runtime

Turns a plan JSON into a CrewAI Teacher/Student run.

```bash
python evaluation/runtime/run_single_plan.py \
  --plan data/examples/plan_01_errors_short_noloop.json \
  --out runs/example
```

## 5. Tier 1 — static evaluation

```bash
python evaluation/tier1_static/tier1_run.py \
  --input data/examples/sample.jsonl \
  --out runs/tier1_demo
```

## 6. Tier 2 — execution evaluation

```bash
python evaluation/tier2_execution/batch_eval.py \
  --input data/examples/sample.jsonl \
  --out runs/tier2_demo
```

## 7. Tier 3 — cross-method outcome evaluation

```bash
# Step 1: generate candidate plans (e.g. GPT-5)
python evaluation/tier3_outcome/tier3_generate_candidates.py ...

# Step 2: pairwise judge with M-judge product
python evaluation/tier3_outcome/tier3_pairwise_eval.py ...
```

Full Tier 3 commands and rubric:
[`evaluation/tier3_outcome/TIER3_EXECUTION.md`](../evaluation/tier3_outcome/TIER3_EXECUTION.md).

## 8. Baselines

```bash
# AutoGen + Qwen3-32B
python baselines/autogen_qwen3/plan.py ...

# AFlow / AutoAgents / EduPlanner / GenMentor / AOP / AIPoM
# Each baseline ships a README in its own folder.
```

## 9. Table-to-command map (paper §5)

| Paper artifact | Command(s) |
|---|---|
| Table 1: dataset statistics | TODO |
| Table 2: Tier 1 results | TODO |
| Table 3: Tier 2 results | TODO |
| Table 4: Tier 3 pairwise | TODO |
| Figure 1: workflow | already in `docs/workflow_of_dataset_construction.png` |
| Figure 2: ablation | TODO |
