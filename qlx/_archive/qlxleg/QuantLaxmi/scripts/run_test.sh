#!/bin/bash
# Safe cargo test wrapper - runs G0-OPS gate first
#
# Usage: ./scripts/run_test.sh [cargo test args...]
# Example: ./scripts/run_test.sh -p quantlaxmi-core
#          ./scripts/run_test.sh --workspace

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Run gate first
if ! ./scripts/ops/gate_no_hidden_workers.sh; then
    echo ""
    echo "Aborting cargo test - gate failed"
    exit 1
fi

echo ""
echo "Running: cargo test $*"
echo ""

cargo test "$@"
