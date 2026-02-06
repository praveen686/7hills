#!/bin/bash
# Safe cargo check wrapper - runs G0-OPS gate first
#
# Usage: ./scripts/run_check.sh [cargo check args...]
# Example: ./scripts/run_check.sh -p quantlaxmi-core
#          ./scripts/run_check.sh --workspace --all-targets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Run gate first
if ! ./scripts/ops/gate_no_hidden_workers.sh; then
    echo ""
    echo "Aborting cargo check - gate failed"
    exit 1
fi

echo ""
echo "Running: cargo check $*"
echo ""

cargo check "$@"
