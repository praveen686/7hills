#!/bin/bash
# Phase Alpha-1.3: Batch Scoring Grid for Calendar Carry Strategy
#
# Runs score_calendar_carry across multiple sessions with parameter grid:
# - holding: 300s, 600s
# - staleness: 500, 1000, 2000, 0 (disabled)
# - friction: 1 bps
# - fill mode: bid/ask (realistic)
#
# Usage:
#   ./scripts/batch_score_calendar_carry.sh <session_dir> [signal_run_id]
#   ./scripts/batch_score_calendar_carry.sh --all-sessions <sessions_root>
#
# Output:
#   Creates summary CSV at <session_dir>/runs/score_calendar_carry/batch_summary.csv

set -euo pipefail

# Configuration
HOLDINGS=(300 600)
STALENESS_CAPS=(500 1000 2000 0)
FRICTION_BPS=1
LATENCY_MS=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Find signal run ID (latest if not specified)
find_signal_run_id() {
    local session_dir="$1"
    local signal_run_dir="${session_dir}/runs/run_calendar_carry"

    if [[ ! -d "$signal_run_dir" ]]; then
        log_error "No run_calendar_carry runs found in $session_dir"
        return 1
    fi

    # Get most recent run (by modification time)
    local latest=$(ls -1t "$signal_run_dir" 2>/dev/null | head -1)
    if [[ -z "$latest" ]]; then
        log_error "No signal runs found in $signal_run_dir"
        return 1
    fi

    echo "$latest"
}

# Score a single session with the parameter grid
score_session() {
    local session_dir="$1"
    local signal_run_id="${2:-}"

    log_info "Scoring session: $session_dir"

    # Auto-detect signal run ID if not provided
    if [[ -z "$signal_run_id" ]]; then
        signal_run_id=$(find_signal_run_id "$session_dir") || return 1
        log_info "Auto-detected signal run: $signal_run_id"
    fi

    local signal_run_dir="${session_dir}/runs/run_calendar_carry/${signal_run_id}"
    if [[ ! -f "${signal_run_dir}/signals.jsonl" ]]; then
        log_error "signals.jsonl not found in $signal_run_dir"
        return 1
    fi

    # Create summary output directory
    local score_dir="${session_dir}/runs/score_calendar_carry"
    mkdir -p "$score_dir"

    local summary_file="${score_dir}/batch_summary.csv"

    # CSV header
    echo "session,signal_run_id,holding_secs,max_quote_age_ms,num_signals,num_trades,num_dropped,gross_pnl,net_pnl,hit_rate,max_drawdown,avg_pnl,avg_quote_age_ms,max_quote_age_ms,symbol_not_found,missing_entry,missing_exit,stale_entry,stale_exit,bad_quote,max_stale_age_ms" > "$summary_file"

    local session_name=$(basename "$session_dir")

    # Run the grid
    for holding in "${HOLDINGS[@]}"; do
        for staleness in "${STALENESS_CAPS[@]}"; do
            log_info "  holding=${holding}s staleness=${staleness}ms"

            # Run scorer
            local output
            output=$(cargo run --release -p quantlaxmi-runner-india --bin score_calendar_carry -- \
                --session-dir "$session_dir" \
                --signal-run-id "$signal_run_id" \
                --holding-secs "$holding" \
                --max-quote-age-ms "$staleness" \
                --friction-bps "$FRICTION_BPS" \
                --latency-ms "$LATENCY_MS" \
                2>&1) || {
                log_warn "    Failed: $output"
                continue
            }

            # Find the run directory (from config hash)
            local latest_run=$(ls -1t "$score_dir" 2>/dev/null | grep -v batch_summary | head -1)
            if [[ -z "$latest_run" ]]; then
                log_warn "    No output run found"
                continue
            fi

            local metrics_file="${score_dir}/${latest_run}/metrics.json"
            if [[ ! -f "$metrics_file" ]]; then
                log_warn "    metrics.json not found"
                continue
            fi

            # Extract metrics using jq
            local metrics
            metrics=$(cat "$metrics_file")

            local num_signals=$(echo "$metrics" | jq -r '.num_signals')
            local num_trades=$(echo "$metrics" | jq -r '.num_trades')
            local num_dropped=$(echo "$metrics" | jq -r '.num_dropped')
            local gross_pnl=$(echo "$metrics" | jq -r '.gross_pnl')
            local net_pnl=$(echo "$metrics" | jq -r '.net_pnl')
            local hit_rate=$(echo "$metrics" | jq -r '.hit_rate')
            local max_drawdown=$(echo "$metrics" | jq -r '.max_drawdown')
            local avg_pnl=$(echo "$metrics" | jq -r '.avg_pnl')
            local avg_quote_age=$(echo "$metrics" | jq -r '.avg_quote_age_ms')
            local max_quote_age=$(echo "$metrics" | jq -r '.max_quote_age_ms')

            # Drop reasons
            local sym_not_found=$(echo "$metrics" | jq -r '.drop_reasons.symbol_not_found')
            local missing_entry=$(echo "$metrics" | jq -r '.drop_reasons.missing_entry_quote')
            local missing_exit=$(echo "$metrics" | jq -r '.drop_reasons.missing_exit_quote')
            local stale_entry=$(echo "$metrics" | jq -r '.drop_reasons.stale_entry_quote')
            local stale_exit=$(echo "$metrics" | jq -r '.drop_reasons.stale_exit_quote')
            local bad_quote=$(echo "$metrics" | jq -r '.drop_reasons.bad_quote')
            local max_stale_age=$(echo "$metrics" | jq -r '.drop_reasons.max_stale_age_ms')

            # Append to CSV
            echo "${session_name},${signal_run_id},${holding},${staleness},${num_signals},${num_trades},${num_dropped},${gross_pnl},${net_pnl},${hit_rate},${max_drawdown},${avg_pnl},${avg_quote_age},${max_quote_age},${sym_not_found},${missing_entry},${missing_exit},${stale_entry},${stale_exit},${bad_quote},${max_stale_age}" >> "$summary_file"

            log_info "    trades=${num_trades} net_pnl=${net_pnl} dropped=${num_dropped}"
        done
    done

    log_info "Summary written to: $summary_file"
}

# Score all sessions in a directory
score_all_sessions() {
    local sessions_root="$1"

    log_info "Scanning for sessions in: $sessions_root"

    # Find all session directories (those with session_manifest.json or symbol subdirs)
    local sessions=()
    for dir in "$sessions_root"/*; do
        if [[ -d "$dir" ]]; then
            if [[ -f "${dir}/session_manifest.json" ]] || [[ -d "${dir}/runs/run_calendar_carry" ]]; then
                sessions+=("$dir")
            fi
        fi
    done

    if [[ ${#sessions[@]} -eq 0 ]]; then
        log_error "No valid sessions found in $sessions_root"
        exit 1
    fi

    log_info "Found ${#sessions[@]} sessions"

    # Aggregate summary
    local aggregate_file="${sessions_root}/batch_aggregate.csv"
    echo "session,signal_run_id,holding_secs,max_quote_age_ms,num_signals,num_trades,num_dropped,gross_pnl,net_pnl,hit_rate,max_drawdown,avg_pnl,avg_quote_age_ms,max_quote_age_ms,symbol_not_found,missing_entry,missing_exit,stale_entry,stale_exit,bad_quote,max_stale_age_ms" > "$aggregate_file"

    for session in "${sessions[@]}"; do
        score_session "$session" "" || {
            log_warn "Failed to score session: $session"
            continue
        }

        # Append session summary to aggregate (skip header)
        local session_summary="${session}/runs/score_calendar_carry/batch_summary.csv"
        if [[ -f "$session_summary" ]]; then
            tail -n +2 "$session_summary" >> "$aggregate_file"
        fi
    done

    log_info "Aggregate summary: $aggregate_file"

    # Print summary statistics
    if command -v column &> /dev/null; then
        log_info "Aggregate Results:"
        column -t -s',' "$aggregate_file" | head -20
    fi
}

# Main entry point
main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage:"
        echo "  $0 <session_dir> [signal_run_id]"
        echo "  $0 --all-sessions <sessions_root>"
        echo ""
        echo "Examples:"
        echo "  $0 /data/sessions/india_2026-01-20"
        echo "  $0 --all-sessions /data/sessions"
        exit 1
    fi

    # Build the scorer in release mode first
    log_info "Building score_calendar_carry (release)..."
    cargo build --release -p quantlaxmi-runner-india --bin score_calendar_carry

    if [[ "$1" == "--all-sessions" ]]; then
        if [[ $# -lt 2 ]]; then
            log_error "Missing sessions root directory"
            exit 1
        fi
        score_all_sessions "$2"
    else
        score_session "$1" "${2:-}"
    fi
}

main "$@"
