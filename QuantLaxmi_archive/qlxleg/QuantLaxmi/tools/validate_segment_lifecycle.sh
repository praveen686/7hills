#!/usr/bin/env bash
# Phase 2B.1 Operational Re-validation Script
# Validates segment manifest lifecycle under graceful and ungraceful stops
#
# Usage:
#   ./tools/validate_segment_lifecycle.sh graceful   # Test graceful stop (Ctrl+C)
#   ./tools/validate_segment_lifecycle.sh ungraceful # Test ungraceful stop (SIGHUP)
#   ./tools/validate_segment_lifecycle.sh both       # Run both tests
#
# Prerequisites:
#   - cargo build --release
#   - Network access to Binance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_OUTPUT_DIR="${REPO_ROOT}/data/phase2b_validation"
BINARY="${REPO_ROOT}/target/release/quantlaxmi-crypto"
DURATION_SECS=90

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_check() { echo -e "  ${GREEN}✓${NC} $1"; }
log_fail() { echo -e "  ${RED}✗${NC} $1"; }

# Build release binary
build_binary() {
    log_info "Building release binary..."
    cd "$REPO_ROOT"
    cargo build --release --package quantlaxmi-runner-crypto 2>/dev/null
    if [[ ! -x "$BINARY" ]]; then
        log_error "Binary not found at $BINARY"
        exit 1
    fi
    log_check "Binary built successfully"
}

# Validate manifest state
validate_manifest() {
    local segment_dir="$1"
    local expected_state="$2"
    local manifest_path="${segment_dir}/segment_manifest.json"

    if [[ ! -f "$manifest_path" ]]; then
        log_fail "Manifest not found: $manifest_path"
        return 1
    fi

    local actual_state
    actual_state=$(jq -r '.state' "$manifest_path")

    if [[ "$actual_state" != "$expected_state" ]]; then
        log_fail "State mismatch: expected=$expected_state, actual=$actual_state"
        return 1
    fi
    log_check "State correct: $actual_state"

    # Check schema version
    local schema_version
    schema_version=$(jq -r '.schema_version' "$manifest_path")
    if [[ "$schema_version" != "3" ]]; then
        log_fail "Schema version mismatch: expected=3, actual=$schema_version"
        return 1
    fi
    log_check "Schema version: $schema_version"

    # Check quote_schema
    local quote_schema
    quote_schema=$(jq -r '.quote_schema' "$manifest_path")
    if [[ "$quote_schema" != "canonical_v1" ]]; then
        log_fail "Quote schema mismatch: expected=canonical_v1, actual=$quote_schema"
        return 1
    fi
    log_check "Quote schema: $quote_schema"

    # Check binary_hash exists
    local binary_hash
    binary_hash=$(jq -r '.binary_hash' "$manifest_path")
    if [[ -z "$binary_hash" || "$binary_hash" == "null" ]]; then
        log_fail "Binary hash missing"
        return 1
    fi
    log_check "Binary hash: ${binary_hash:0:16}..."

    # Check config exists
    local config_exists
    config_exists=$(jq -r '.config | type' "$manifest_path")
    if [[ "$config_exists" != "object" ]]; then
        log_fail "Config missing or invalid"
        return 1
    fi
    log_check "Config snapshot present"

    return 0
}

# Validate digests present
validate_digests() {
    local segment_dir="$1"
    local manifest_path="${segment_dir}/segment_manifest.json"

    local digests_type
    digests_type=$(jq -r '.digests | type' "$manifest_path")

    if [[ "$digests_type" == "null" ]]; then
        log_fail "Digests missing (null)"
        return 1
    fi

    # Check perp digest (should always exist for perp session)
    local perp_sha256
    perp_sha256=$(jq -r '.digests.perp.sha256 // "null"' "$manifest_path")
    if [[ "$perp_sha256" != "null" && -n "$perp_sha256" ]]; then
        log_check "Perp digest: ${perp_sha256:0:16}..."
    else
        log_warn "Perp digest missing (may be expected if no perp events)"
    fi

    # Check funding digest
    local funding_sha256
    funding_sha256=$(jq -r '.digests.funding.sha256 // "null"' "$manifest_path")
    if [[ "$funding_sha256" != "null" && -n "$funding_sha256" ]]; then
        log_check "Funding digest: ${funding_sha256:0:16}..."
    else
        log_warn "Funding digest missing (may be expected if no funding events)"
    fi

    return 0
}

# Test graceful stop
test_graceful_stop() {
    log_info "=== TEST: Graceful Stop (Ctrl+C / SIGINT) ==="

    local test_dir="${TEST_OUTPUT_DIR}/graceful_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$test_dir"

    log_info "Starting capture to: $test_dir"
    log_info "Duration: ${DURATION_SECS}s (will send SIGINT after)"

    # Start capture in background
    "$BINARY" capture-perp-session \
        --symbols BTCUSDT \
        --out-dir "$test_dir" \
        --duration-secs 3600 \
        --include-spot true \
        --include-depth false &
    local pid=$!

    log_info "Capture started (PID: $pid)"

    # Wait for bootstrap manifest to appear
    sleep 5
    local segment_dir
    segment_dir=$(find "$test_dir" -maxdepth 1 -type d -name "perp_*" | head -1)

    if [[ -z "$segment_dir" ]]; then
        log_error "No segment directory created"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    log_info "Segment directory: $segment_dir"

    # Validate bootstrap manifest exists immediately
    log_info "Checking bootstrap manifest (should exist immediately)..."
    if ! validate_manifest "$segment_dir" "BOOTSTRAP"; then
        log_error "Bootstrap manifest validation failed"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    # Wait for capture duration
    log_info "Waiting ${DURATION_SECS}s for data capture..."
    sleep "$DURATION_SECS"

    # Send SIGINT (graceful stop)
    log_info "Sending SIGINT (Ctrl+C equivalent)..."
    kill -INT $pid 2>/dev/null || true

    # Wait for clean shutdown
    sleep 5

    # Validate finalized manifest
    log_info "Checking finalized manifest..."
    if ! validate_manifest "$segment_dir" "FINALIZED"; then
        log_error "Finalized manifest validation failed"
        return 1
    fi

    # Validate digests
    log_info "Checking digests..."
    if ! validate_digests "$segment_dir"; then
        log_error "Digests validation failed"
        return 1
    fi

    # Print summary
    log_info "=== Graceful Stop Test PASSED ==="
    echo ""
    echo "Manifest contents:"
    jq '{state, schema_version, quote_schema, segment_id, stop_reason, events, digests: (.digests | keys)}' "$segment_dir/segment_manifest.json"
    echo ""

    return 0
}

# Test ungraceful stop
test_ungraceful_stop() {
    log_info "=== TEST: Ungraceful Stop (SIGHUP) ==="

    local test_dir="${TEST_OUTPUT_DIR}/ungraceful_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$test_dir"

    log_info "Starting capture to: $test_dir"
    log_info "Duration: ${DURATION_SECS}s (will send SIGHUP after)"

    # Start capture in background
    "$BINARY" capture-perp-session \
        --symbols BTCUSDT \
        --out-dir "$test_dir" \
        --duration-secs 3600 \
        --include-spot true \
        --include-depth false &
    local pid=$!

    log_info "Capture started (PID: $pid)"

    # Wait for bootstrap manifest to appear
    sleep 5
    local segment_dir
    segment_dir=$(find "$test_dir" -maxdepth 1 -type d -name "perp_*" | head -1)

    if [[ -z "$segment_dir" ]]; then
        log_error "No segment directory created"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    log_info "Segment directory: $segment_dir"

    # Validate bootstrap manifest exists immediately
    log_info "Checking bootstrap manifest (should exist immediately)..."
    if ! validate_manifest "$segment_dir" "BOOTSTRAP"; then
        log_error "Bootstrap manifest validation failed"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    # Wait for capture duration
    log_info "Waiting ${DURATION_SECS}s for data capture..."
    sleep "$DURATION_SECS"

    # Send SIGHUP (ungraceful stop - simulates terminal disconnect)
    log_info "Sending SIGHUP (ungraceful stop)..."
    kill -HUP $pid 2>/dev/null || true

    # Wait for process to die
    sleep 3

    # Verify process is dead
    if kill -0 $pid 2>/dev/null; then
        log_warn "Process still alive, sending SIGKILL"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi

    # Validate manifest is still BOOTSTRAP (ungraceful stop)
    log_info "Checking manifest after ungraceful stop (should still be BOOTSTRAP)..."
    if ! validate_manifest "$segment_dir" "BOOTSTRAP"; then
        # It might have been finalized if SIGHUP was caught
        local actual_state
        actual_state=$(jq -r '.state' "$segment_dir/segment_manifest.json")
        if [[ "$actual_state" == "FINALIZED" ]]; then
            log_warn "SIGHUP was caught and handled gracefully (state=FINALIZED)"
            log_info "This is acceptable - SIGHUP handler worked"
            return 0
        fi
        log_error "Manifest validation failed"
        return 1
    fi

    # Now run retro-finalize
    log_info "Running finalize-segment command..."
    if ! "$BINARY" finalize-segment --segment-dir "$segment_dir"; then
        log_error "finalize-segment command failed"
        return 1
    fi

    # Validate manifest is now FINALIZED_RETRO
    log_info "Checking manifest after retro-finalize..."
    if ! validate_manifest "$segment_dir" "FINALIZED_RETRO"; then
        log_error "Retro-finalized manifest validation failed"
        return 1
    fi

    # Validate digests
    log_info "Checking digests after retro-finalize..."
    if ! validate_digests "$segment_dir"; then
        log_error "Digests validation failed"
        return 1
    fi

    # Print summary
    log_info "=== Ungraceful Stop Test PASSED ==="
    echo ""
    echo "Manifest contents:"
    jq '{state, schema_version, quote_schema, segment_id, stop_reason, events, digests: (.digests | keys)}' "$segment_dir/segment_manifest.json"
    echo ""

    return 0
}

# Main
main() {
    local mode="${1:-both}"

    echo "=========================================="
    echo "Phase 2B.1 Operational Re-validation"
    echo "=========================================="
    echo ""

    build_binary
    mkdir -p "$TEST_OUTPUT_DIR"

    local graceful_result=0
    local ungraceful_result=0

    case "$mode" in
        graceful)
            test_graceful_stop || graceful_result=1
            ;;
        ungraceful)
            test_ungraceful_stop || ungraceful_result=1
            ;;
        both)
            test_graceful_stop || graceful_result=1
            echo ""
            echo "=========================================="
            echo ""
            test_ungraceful_stop || ungraceful_result=1
            ;;
        *)
            echo "Usage: $0 [graceful|ungraceful|both]"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "SUMMARY"
    echo "=========================================="

    if [[ "$mode" == "both" || "$mode" == "graceful" ]]; then
        if [[ $graceful_result -eq 0 ]]; then
            log_check "Graceful stop test: PASSED"
        else
            log_fail "Graceful stop test: FAILED"
        fi
    fi

    if [[ "$mode" == "both" || "$mode" == "ungraceful" ]]; then
        if [[ $ungraceful_result -eq 0 ]]; then
            log_check "Ungraceful stop test: PASSED"
        else
            log_fail "Ungraceful stop test: FAILED"
        fi
    fi

    if [[ $graceful_result -eq 0 && $ungraceful_result -eq 0 ]]; then
        echo ""
        log_info "Phase 2B.1 Operational Re-validation: ALL TESTS PASSED"
        echo ""
        echo "Promotion gate satisfied. Ready for Phase 2B.2."
        return 0
    else
        echo ""
        log_error "Phase 2B.1 Operational Re-validation: SOME TESTS FAILED"
        return 1
    fi
}

main "$@"
