#!/usr/bin/env bash
# Create the GitHub repository and push the initial commit.
#
# Prerequisites:
#   - gh CLI installed: https://cli.github.com/
#   - Authenticated:   gh auth login
#
# Usage:
#   GH_REPO_NAME=maple bash scripts/push_to_github.sh             # public, under your user
#   GH_REPO_NAME=org/maple GH_VISIBILITY=private bash scripts/push_to_github.sh

set -euo pipefail

GH_REPO_NAME="${GH_REPO_NAME:-maple}"
GH_VISIBILITY="${GH_VISIBILITY:-public}"
GH_DESCRIPTION="${GH_DESCRIPTION:-MAP-PPL: Multi-Agent Personalized Learning Plans — dataset and training/evaluation pipeline}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

if [[ ! -d .git ]]; then
  echo "ERROR: ${REPO_DIR} is not a git repository. Run 'git init' first." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not found. Install with: brew install gh" >&2
  echo "Or manually:" >&2
  echo "  1. Create the repo on github.com" >&2
  echo "  2. git remote add origin git@github.com:<user>/<repo>.git" >&2
  echo "  3. git push -u origin main" >&2
  exit 1
fi

echo "Creating GitHub repo: ${GH_REPO_NAME} (${GH_VISIBILITY})"
gh repo create "${GH_REPO_NAME}" \
  --"${GH_VISIBILITY}" \
  --description "${GH_DESCRIPTION}" \
  --source=. \
  --remote=origin \
  --push

echo ""
echo "Done. View the repo at:"
gh repo view "${GH_REPO_NAME}" --json url --jq .url
