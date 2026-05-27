#!/usr/bin/env bash
# Stage 1b: train the SDP LoRA adapter with TRL SFTTrainer.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONFIG="${CONFIG:-configs/sdp_autotrain.yml}"

python train_sft_lora.py --config "$CONFIG"
