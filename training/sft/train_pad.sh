#!/usr/bin/env bash
# Stage 1 — train the PAD LoRA adapter via AutoTrain.
#
# Prereqs:
#   pip install autotrain-advanced
#   python build_sft_data.py        # builds data/pad/{train,valid}.jsonl
#
# Output:
#   ./maple-pad-sft/                project directory with the LoRA adapter
#                                    (rename via PROJECT env var below)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Allow overriding base model from the env so the same script can target
# different backbones without editing the YAML.
if [[ -n "${BASE_MODEL:-}" ]]; then
  export AUTOTRAIN_BASE_MODEL="$BASE_MODEL"
fi
if [[ -n "${PROJECT:-}" ]]; then
  export AUTOTRAIN_PROJECT_NAME="$PROJECT"
fi

autotrain --config configs/pad_autotrain.yml
