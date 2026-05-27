#!/usr/bin/env bash
# Stage 4 — Reinforcement Optimization with Verifiable + Counterfactual Rewards.
#
# Prereqs:
#   pip install "trl>=0.13" "transformers>=4.45" peft accelerate datasets vllm pyyaml
#
# Workflow:
#   0. SFT stages already run (PAD + SDP + optional joint alignment).
#   1. Build the prompt-only dataset:
#        python build_grpo_prompts.py \
#            --input ../multi_agent_dataset_filtered_qap_latest.jsonl \
#            --out data/grpo
#   2. (Optional, for R_pers) Build counterfactual cache:
#        python build_counterfactual_cache.py \
#            --src ../multi_agent_dataset_filtered_qap_latest.jsonl \
#            --base-model Qwen/Qwen2.5-7B-Instruct \
#            --sft-adapter ../SFT/maple-pad-dpo-iter1 \
#            --out data/grpo/cf_cache.jsonl
#      then set `counterfactual_cache: data/grpo/cf_cache.jsonl`
#      and `reward.enable_pers: true` in configs/grpo.yaml.
#   3. Launch:
#        ./train_grpo.sh
#

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONFIG="${CONFIG:-configs/grpo.yaml}"
python grpo_train.py --config "$CONFIG"
