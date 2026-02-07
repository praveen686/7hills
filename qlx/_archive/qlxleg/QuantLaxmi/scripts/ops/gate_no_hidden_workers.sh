#!/bin/bash
# G0-OPS: "No hidden workers" gate
# Must pass before any heavy Rust action (edits/builds/migrations)
#
# Fails if:
#   - rust-analyzer is running
#   - cargo build/check/test is already running
#   - Available memory is below threshold (default 6 GiB)
#
# Usage: ./scripts/ops/gate_no_hidden_workers.sh [min_mem_gib]

set -euo pipefail

MIN_MEM_GIB="${1:-6}"
MIN_MEM_KB=$((MIN_MEM_GIB * 1024 * 1024))

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "G0-OPS: No Hidden Workers Gate"
echo "========================================"
echo ""

FAILED=0

# Check 1: rust-analyzer
echo -n "Checking rust-analyzer... "
RA_PIDS=$(pgrep -f "rust-analyzer" 2>/dev/null || true)
if [ -n "$RA_PIDS" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  rust-analyzer is running (PIDs: $RA_PIDS)"
    echo "  Run: pkill rust-analyzer"
    FAILED=1
else
    echo -e "${GREEN}OK${NC} (not running)"
fi

# Check 2: cargo processes
echo -n "Checking cargo processes... "
CARGO_PIDS=$(pgrep -af "cargo (build|check|test|clippy)" 2>/dev/null || true)
if [ -n "$CARGO_PIDS" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  cargo is already running:"
    echo "$CARGO_PIDS" | sed 's/^/    /'
    FAILED=1
else
    echo -e "${GREEN}OK${NC} (no cargo build/check/test running)"
fi

# Check 3: Memory headroom
echo -n "Checking memory (need ${MIN_MEM_GIB} GiB available)... "
MEM_AVAILABLE_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
MEM_AVAILABLE_GIB=$(echo "scale=2; $MEM_AVAILABLE_KB / 1024 / 1024" | bc)

if [ "$MEM_AVAILABLE_KB" -lt "$MIN_MEM_KB" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  Available: ${MEM_AVAILABLE_GIB} GiB (need ${MIN_MEM_GIB} GiB)"
    echo "  Top memory consumers:"
    ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "    %s %s %s%%\n", $11, $2, $4}'
    FAILED=1
else
    echo -e "${GREEN}OK${NC} (${MEM_AVAILABLE_GIB} GiB available)"
fi

# Audit: top memory processes
echo ""
echo "Memory audit (top 10 by RSS):"
ps aux --sort=-%mem | head -11 | tail -10 | awk '{printf "  %5.1f%% %6.0fMB  %s\n", $4, $6/1024, $11}'

echo ""
echo "========================================"
if [ "$FAILED" -eq 1 ]; then
    echo -e "${RED}GATE FAILED${NC} - Do not proceed with heavy Rust work"
    exit 1
else
    echo -e "${GREEN}GATE PASSED${NC} - Safe to proceed"
    exit 0
fi
