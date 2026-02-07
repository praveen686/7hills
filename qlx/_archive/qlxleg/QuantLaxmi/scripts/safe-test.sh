#!/usr/bin/env bash
# safe-test.sh — Run cargo tests with memory limits to prevent OOM/system reboot
# Usage:
#   ./scripts/safe-test.sh                    # Run all workspace tests
#   ./scripts/safe-test.sh -p quantlaxmi-core # Run single crate tests
#   ./scripts/safe-test.sh -p quantlaxmi-runner-crypto -- --test-threads=1

set -euo pipefail

# Memory limit: 8GB virtual memory cap (prevents OOM from killing VS Code)
# Adjust if you add more RAM to the system
MEMLIMIT_KB=$((8 * 1024 * 1024))  # 8 GB

ulimit -v "$MEMLIMIT_KB" 2>/dev/null || echo "Warning: could not set memory limit (ulimit -v)"

export CARGO_INCREMENTAL=0
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-1}"

echo "=== Safe Test Runner ==="
echo "  Memory limit: 8 GB"
echo "  Test threads: $RUST_TEST_THREADS"
echo "  Incremental:  disabled"
echo "  Args:         ${*:-(workspace)}"
echo "========================"

exec cargo test --workspace "$@"
