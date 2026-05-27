#!/usr/bin/env bash
# Upload the MAP-PPL dataset to Hugging Face Hub.
#
# Prerequisites:
#   1. pip install -U huggingface_hub
#   2. huggingface-cli login   (paste a HF write token)
#   3. Create the dataset repo on the website first, OR allow this script
#      to create it via the API.
#
# Usage:
#   HF_REPO_ID=your-username/maple bash scripts/upload_hf_dataset.sh
#   HF_REPO_ID=your-username/maple HF_STAGING=/path/to/staging bash scripts/upload_hf_dataset.sh

set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-}"
HF_STAGING="${HF_STAGING:-$(cd "$(dirname "$0")/../.." && pwd)/maple-hf-staging}"

if [[ -z "${HF_REPO_ID}" ]]; then
  echo "ERROR: set HF_REPO_ID, e.g. HF_REPO_ID=your-username/maple bash $0" >&2
  exit 1
fi

if [[ ! -d "${HF_STAGING}" ]]; then
  echo "ERROR: staging dir not found at ${HF_STAGING}" >&2
  echo "Run scripts/build_canonical_splits.py first to materialize it." >&2
  exit 1
fi

# Sanity check expected files
for f in README.md train.jsonl dev.jsonl test.jsonl sample.jsonl; do
  if [[ ! -f "${HF_STAGING}/${f}" ]]; then
    echo "ERROR: missing ${f} in ${HF_STAGING}" >&2
    exit 1
  fi
done

echo "Uploading staging dir to HF dataset repo: ${HF_REPO_ID}"
echo "  source: ${HF_STAGING}"

# huggingface-cli upload <repo_id> <local_path> <path_in_repo> --repo-type=dataset
# Create the repo first if it doesn't exist.
huggingface-cli repo create "${HF_REPO_ID}" --type dataset --yes || true

huggingface-cli upload "${HF_REPO_ID}" "${HF_STAGING}" . \
  --repo-type=dataset \
  --commit-message "Initial MAP-PPL v15 release: 3,043 plans, stratified 80/10/10 split"

echo ""
echo "Done. View the dataset at: https://huggingface.co/datasets/${HF_REPO_ID}"
