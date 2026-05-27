#!/usr/bin/env bash
# Stage 2 — train the SDP LoRA adapter via AutoTrain.
# (Independent of PAD — can run in parallel.)
#
# Output:
#   ./maple-sdp-sft/                project directory with the LoRA adapter

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ -n "${BASE_MODEL:-}" ]]; then
  export AUTOTRAIN_BASE_MODEL="$BASE_MODEL"
fi
if [[ -n "${PROJECT:-}" ]]; then
  export AUTOTRAIN_PROJECT_NAME="$PROJECT"
fi

autotrain --config configs/sdp_autotrain.yml
