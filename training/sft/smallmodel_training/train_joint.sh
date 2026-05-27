#!/usr/bin/env bash
# Stage 3 — Joint Alignment of PAD ↔ SDP.
#
# Alternates between two AutoTrain runs (Phase A and Phase B) for K outer
# iterations. The paper sets K ∈ {1, 2}.
#
# Phase A: on-policy SDP SFT
#   - generate data via joint_alignment_data.py phase-a
#   - run AutoTrain SFT continuing from the SDP adapter
#
# Phase B: PAD DPO with scaffold-level structural preferences
#   - generate data via joint_alignment_data.py phase-b (rule-based
#     scaffold-vs-gold Jaccard; no SDP forward pass needed)
#   - run AutoTrain DPO continuing from the PAD adapter
#
# Naming note: CLI labels (phase-a, phase-b) are inverted vs paper §3.2,
# where Phase A modifies PAD and Phase B modifies SDP. Order of execution
# here (a then b) means SDP is updated first, then PAD — reversed from
# the paper's stated order. Both orders are valid for K=1.
#
# After each phase, the relevant adapter is replaced by the updated checkpoint
# before the next phase begins.
#
# Required env variables (no defaults — be explicit):
#   BASE_MODEL    e.g.  Qwen/Qwen2.5-7B-Instruct
#   PAD_ADAPTER   e.g.  ./maple-pad-sft
#   SDP_ADAPTER   e.g.  ./maple-sdp-sft
#   SRC           e.g.  multi_agent_dataset_filtered_qap_v3.jsonl
#
# Optional:
#   K             outer iterations (default 1)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${BASE_MODEL:?set BASE_MODEL}"
: "${PAD_ADAPTER:?set PAD_ADAPTER}"
: "${SDP_ADAPTER:?set SDP_ADAPTER}"
: "${SRC:?set SRC (path to MAPLE JSONL)}"
K="${K:-1}"

PAD_CURRENT="$PAD_ADAPTER"
SDP_CURRENT="$SDP_ADAPTER"

for ((iter = 1; iter <= K; iter++)); do
  echo "===================="
  echo "Joint alignment outer iteration $iter / $K"
  echo "  PAD = $PAD_CURRENT"
  echo "  SDP = $SDP_CURRENT"
  echo "===================="

  # ---- Phase A: on-policy SDP ----------------------------------------------
  PHASE_A_DIR="data/joint/iter${iter}/phase_a"
  python joint_alignment_data.py phase-a \
      --src "$SRC" \
      --base-model "$BASE_MODEL" \
      --pad-adapter "$PAD_CURRENT" \
      --samples-per-query 1 \
      --out "$PHASE_A_DIR"

  # Edit the config in-place to use the current SDP adapter / phase-A data
  # (sed is portable; if you prefer, regenerate the YAML from a template).
  cp configs/joint_phase_a_autotrain.yml configs/_phase_a_iter${iter}.yml
  sed -i.bak \
      -e "s|^  peft_model:.*|  peft_model: $SDP_CURRENT|" \
      -e "s|^  path:.*|  path: $PHASE_A_DIR|" \
      -e "s|^project_name:.*|project_name: maple-sdp-onpolicy-iter${iter}|" \
      configs/_phase_a_iter${iter}.yml
  rm -f configs/_phase_a_iter${iter}.yml.bak

  autotrain --config configs/_phase_a_iter${iter}.yml
  SDP_CURRENT="./maple-sdp-onpolicy-iter${iter}"

  # ---- Phase B: PAD DPO ----------------------------------------------------
  PHASE_B_DIR="data/joint/iter${iter}/phase_b"
  python joint_alignment_data.py phase-b \
      --src "$SRC" \
      --base-model "$BASE_MODEL" \
      --pad-adapter "$PAD_CURRENT" \
      --pairs-per-query 1 \
      --out "$PHASE_B_DIR"

  cp configs/joint_phase_b_autotrain.yml configs/_phase_b_iter${iter}.yml
  sed -i.bak \
      -e "s|^  peft_model:.*|  peft_model: $PAD_CURRENT|" \
      -e "s|^  path:.*|  path: $PHASE_B_DIR|" \
      -e "s|^project_name:.*|project_name: maple-pad-dpo-iter${iter}|" \
      configs/_phase_b_iter${iter}.yml
  rm -f configs/_phase_b_iter${iter}.yml.bak

  autotrain --config configs/_phase_b_iter${iter}.yml
  PAD_CURRENT="./maple-pad-dpo-iter${iter}"
done

echo ""
echo "Joint alignment complete."
echo "  Final PAD adapter: $PAD_CURRENT"
echo "  Final SDP adapter: $SDP_CURRENT"
