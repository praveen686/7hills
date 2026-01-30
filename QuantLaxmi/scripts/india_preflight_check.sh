#!/usr/bin/env bash
# india_preflight_check.sh - Pre-merge sanity checks for India session
#
# Usage: ./scripts/india_preflight_check.sh <session_dir>
#
# Checks:
# 1. No writers (capture process not holding files)
# 2. Tick counts per instrument
# 3. First/last line JSON validity (no truncation)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <session_dir>"
    exit 1
fi

SESSION_DIR="$1"

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Error: Session directory not found: $SESSION_DIR"
    exit 1
fi

echo "=== Phase 27-IND Preflight Check ==="
echo "Session: $SESSION_DIR"
echo ""

# 1. Check for active writers
echo "=== 1. Checking for active writers ==="
WRITERS=$(lsof +D "$SESSION_DIR" 2>/dev/null | grep -v "^COMMAND" || true)
if [[ -n "$WRITERS" ]]; then
    echo "WARNING: Active writers detected!"
    echo "$WRITERS"
    echo ""
    echo "Stop capture before merging:"
    echo "  pgrep -f 'quantlaxmi-india.*capture' | xargs -r kill -INT"
    exit 1
else
    echo "OK: No active writers"
fi
echo ""

# 2. Tick counts
echo "=== 2. Tick counts per instrument ==="
TICK_FILES=$(find "$SESSION_DIR" -name "ticks.jsonl" -type f 2>/dev/null | sort)
TOTAL=0
ATM_CE=0
ATM_PE=0

for f in $TICK_FILES; do
    COUNT=$(wc -l < "$f")
    INSTRUMENT=$(basename "$(dirname "$f")")
    TOTAL=$((TOTAL + COUNT))
    printf "%7d  %s\n" "$COUNT" "$INSTRUMENT"

    # Track ATM for gate check
    if [[ "$INSTRUMENT" == *"59700CE"* ]]; then
        ATM_CE=$COUNT
    elif [[ "$INSTRUMENT" == *"59700PE"* ]]; then
        ATM_PE=$COUNT
    fi
done
echo ""
echo "Total ticks: $TOTAL"
echo "ATM CE ticks: $ATM_CE"
echo "ATM PE ticks: $ATM_PE"

# Gate check
if [[ $ATM_CE -lt 7500 ]] || [[ $ATM_PE -lt 7500 ]]; then
    echo "WARNING: ATM ticks below 7500 gate minimum"
fi
echo ""

# 3. JSON validity check (first/last line)
echo "=== 3. JSON validity check ==="
ERRORS=0
for f in $TICK_FILES; do
    INSTRUMENT=$(basename "$(dirname "$f")")

    # Check first line
    if ! head -1 "$f" | jq -e . >/dev/null 2>&1; then
        echo "ERROR: Invalid first line in $INSTRUMENT"
        ERRORS=$((ERRORS + 1))
    fi

    # Check last line
    if ! tail -1 "$f" | jq -e . >/dev/null 2>&1; then
        echo "ERROR: Invalid last line in $INSTRUMENT (truncated?)"
        ERRORS=$((ERRORS + 1))
    fi
done

if [[ $ERRORS -eq 0 ]]; then
    echo "OK: All first/last lines valid JSON"
else
    echo "FAILED: $ERRORS JSON errors found"
    exit 1
fi
echo ""

# 4. Check for CLOSE_MARKER
echo "=== 4. Close marker ==="
if [[ -f "$SESSION_DIR/CLOSE_MARKER.txt" ]]; then
    echo "OK: Close marker exists: $(cat "$SESSION_DIR/CLOSE_MARKER.txt")"
else
    echo "INFO: No close marker yet (create after stopping capture)"
fi
echo ""

echo "=== Preflight PASSED ==="
echo "Ready for: ./scripts/india_make_quotes.sh $SESSION_DIR --merge"
