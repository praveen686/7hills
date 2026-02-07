#!/usr/bin/env bash
# india_make_quotes.sh - Convert India tick capture to KiteSim QuoteEvent format
#
# Usage:
#   ./scripts/india_make_quotes.sh <session_dir> [--merge]
#
# Input:  Session directory containing per-instrument subdirs with ticks.jsonl
# Output: Per-instrument quotes.jsonl files
#         If --merge: also creates quotes_all.jsonl (sorted by timestamp)
#
# Schema transformation:
#   ticks.jsonl:  {ts, tradingsymbol, instrument_token, bid_price, ask_price, bid_qty, ask_qty, ltp, ltq, volume, price_exponent, integrity_tier}
#   quotes.jsonl: {ts, tradingsymbol, bid, ask, bid_qty, ask_qty, price_exponent}
#
# Phase 27-IND deterministic replay requires this canonical format.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <session_dir> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --merge              Create quotes_all.jsonl (all instruments merged, sorted by ts)"
    echo "  --symbols <regex>    Filter merged output to matching tradingsymbols"
    echo ""
    echo "Examples:"
    echo "  $0 data/sessions/foo --merge"
    echo "  $0 data/sessions/foo --merge --symbols 'BANKNIFTY26FEB59700CE|BANKNIFTY26FEB59700PE'"
    exit 1
fi

SESSION_DIR="$1"
MERGE=false

# Parse arguments
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --merge)
            MERGE=true
            ;;
        --symbols)
            # Filter to specific symbols (regex pattern)
            # Example: --symbols 'BANKNIFTY26FEB59700CE|BANKNIFTY26FEB59700PE'
            MERGE_SYMBOLS="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Default: no symbol filter (merge all)
MERGE_SYMBOLS="${MERGE_SYMBOLS:-}"

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "Error: Session directory not found: $SESSION_DIR"
    exit 1
fi

echo "Converting ticks → quotes in: $SESSION_DIR"

# Find all ticks.jsonl files (null-delimited for space-safe paths)
TICK_COUNT_CHECK=$(find "$SESSION_DIR" -name "ticks.jsonl" -type f 2>/dev/null | wc -l)

if [[ "$TICK_COUNT_CHECK" -eq 0 ]]; then
    echo "Error: No ticks.jsonl files found in $SESSION_DIR"
    exit 1
fi

CONVERTED=0
TOTAL_TICKS=0

# Process files in deterministic sorted order, space-safe
while IFS= read -r -d '' TICK_FILE; do
    INSTRUMENT_DIR=$(dirname "$TICK_FILE")
    INSTRUMENT=$(basename "$INSTRUMENT_DIR")
    QUOTE_FILE="$INSTRUMENT_DIR/quotes.jsonl"

    # Convert using jq: rename bid_price→bid, ask_price→ask, drop extra fields
    # -e: exit non-zero on invalid JSON (fail-fast for gates)
    jq -e -c '{ts, tradingsymbol, bid: .bid_price, ask: .ask_price, bid_qty, ask_qty, price_exponent}' \
        "$TICK_FILE" > "$QUOTE_FILE"

    TICK_COUNT=$(wc -l < "$QUOTE_FILE")
    TOTAL_TICKS=$((TOTAL_TICKS + TICK_COUNT))
    CONVERTED=$((CONVERTED + 1))

    echo "  [$CONVERTED] $INSTRUMENT: $TICK_COUNT quotes"
done < <(find "$SESSION_DIR" -name "ticks.jsonl" -type f -print0 | sort -z)

echo ""
echo "Converted $CONVERTED instruments, $TOTAL_TICKS total quotes"

# Optionally merge all quotes into single sorted file
if [[ "$MERGE" == true ]]; then
    MERGED_FILE="$SESSION_DIR/quotes_all.jsonl"
    echo ""
    echo "Merging all quotes into: $MERGED_FILE"

    # Streaming deterministic merge (no RAM slurp):
    # 1. Concatenate files in deterministic filename order
    # 2. Normalize to compact JSONL (one object per line)
    # 3. Extract sortable key per line (ts, tradingsymbol, bid, ask)
    #    - Strict: fail if bid/ask are non-numeric (gate-grade validation)
    # 4. External sort (streaming-friendly, LC_ALL=C for bytewise stability)
    # 5. Strip key, keep JSON
    #
    # This avoids jq -s which loads everything into RAM.
    # Ordering: ts (string), tradingsymbol (string), bid (numeric), ask (numeric)
    if [[ -n "$MERGE_SYMBOLS" ]]; then
        echo "Symbol filter: $MERGE_SYMBOLS"
    fi

    export LC_ALL=C
    find "$SESSION_DIR" -name "quotes.jsonl" -type f -print0 \
        | sort -z \
        | xargs -0 cat \
        | jq -c '.' \
        | jq -e -rc --arg re "$MERGE_SYMBOLS" '
            if ($re != "" and (.tradingsymbol | test($re) | not)) then
              empty
            elif (.ts|type) != "string" or (.tradingsymbol|type) != "string" then
              error("bad ts/tradingsymbol: " + (.|@json))
            elif (.bid|type) != "number" or (.ask|type) != "number" then
              error("non-numeric bid/ask: " + (.|@json))
            else
              [.ts, .tradingsymbol, .bid, .ask, (.|@json)] | @tsv
            end
          ' \
        | sort --stable -t $'\t' -k1,1 -k2,2 -k3,3n -k4,4n \
        | cut -f5- \
        > "$MERGED_FILE"

    MERGED_COUNT=$(wc -l < "$MERGED_FILE")
    echo "Merged file: $MERGED_COUNT quotes (sorted by ts)"

    # Compute SHA256 for determinism verification
    MERGED_SHA256=$(sha256sum "$MERGED_FILE" | cut -d' ' -f1)
    echo "SHA256: $MERGED_SHA256"
fi

echo ""
echo "Done. Quotes ready for KiteSim backtest."
