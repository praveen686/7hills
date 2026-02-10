#!/bin/bash
# Safe cargo build wrapper - runs G0-OPS gate first
# NOTE: Prefer run_check.sh for iteration; use this only for final builds
#
# Usage: ./scripts/run_build.sh [cargo build args...]
# Example: ./scripts/run_build.sh --release -p quantlaxmi-crypto

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Run gate first (require 8 GiB for full builds)
if ! ./scripts/ops/gate_no_hidden_workers.sh 8; then
    echo ""
    echo "Aborting cargo build - gate failed"
    exit 1
fi

echo ""
echo "Running: cargo build $*"
echo ""

cargo build "$@"
