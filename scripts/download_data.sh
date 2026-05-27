#!/usr/bin/env bash
# Download the MAP-PPL dataset from Hugging Face into ./data/
# Usage:
#   bash scripts/download_data.sh             # full dataset
#   bash scripts/download_data.sh --sample    # 100-row sample only

set -euo pipefail

REPO_ID="${MAPLE_HF_REPO:-wenzhy7/MAP-PPL}"   # set MAPLE_HF_REPO to override
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

# Check huggingface-cli availability
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ "${1:-}" == "--sample" ]]; then
  echo "Downloading 100-row sample from ${REPO_ID}..."
  huggingface-cli download "${REPO_ID}" \
    --repo-type dataset \
    --include "sample.jsonl" \
    --local-dir "${DATA_DIR}" \
    --local-dir-use-symlinks False
  echo "Sample saved to ${DATA_DIR}/sample.jsonl"
else
  echo "Downloading full MAP-PPL dataset from ${REPO_ID}..."
  huggingface-cli download "${REPO_ID}" \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --local-dir-use-symlinks False
  echo "Dataset saved to ${DATA_DIR}/"
fi

echo "Done."
ls -lh "${DATA_DIR}"
