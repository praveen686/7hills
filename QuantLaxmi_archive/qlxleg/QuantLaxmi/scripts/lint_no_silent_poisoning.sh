#!/usr/bin/env bash
# lint_no_silent_poisoning.sh
#
# CI lint script enforcing the No Silent Poisoning doctrine.
# See: docs/DOCTRINE_NO_SILENT_POISONING.md
#
# Exit codes:
#   0 = All checks pass
#   1 = Violations found
#
# Usage:
#   ./scripts/lint_no_silent_poisoning.sh
#   ./scripts/lint_no_silent_poisoning.sh --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERBOSE=false
if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=true
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

VIOLATIONS=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_verbose() {
    if $VERBOSE; then
        echo -e "       $1"
    fi
}

# =============================================================================
# Allowlist handling
# =============================================================================
ALLOWLIST_FILE="$REPO_ROOT/docs/lint_allowlist_no_silent_poisoning.txt"
ALLOWLIST_ENTRIES=()
SUPPRESSED_COUNT=0

load_allowlist() {
    if [[ ! -f "$ALLOWLIST_FILE" ]]; then
        log_verbose "No allowlist file found at $ALLOWLIST_FILE"
        return
    fi

    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        # Strip inline comments (everything after first unescaped #)
        # Use printf to preserve backslashes
        local entry
        entry=$(printf '%s' "$line" | sed 's/[[:space:]]*#.*//')
        # Trim whitespace
        entry=$(printf '%s' "$entry" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -n "$entry" ]] && ALLOWLIST_ENTRIES+=("$entry")
    done < "$ALLOWLIST_FILE"

    log_verbose "Loaded ${#ALLOWLIST_ENTRIES[@]} allowlist entries"
}

# Check if a warning matches an allowlist entry
# Args: $1 = filepath, $2 = line content
# Returns: 0 if allowlisted, 1 if not
is_allowlisted() {
    local filepath="$1"
    local content="$2"

    for entry in "${ALLOWLIST_ENTRIES[@]}"; do
        # Entry format: path_regex:pattern_regex
        local path_regex pattern_regex
        path_regex=$(printf '%s' "$entry" | cut -d: -f1)
        pattern_regex=$(printf '%s' "$entry" | cut -d: -f2-)

        if printf '%s' "$filepath" | grep -qE "$path_regex" && \
           printf '%s' "$content" | grep -qE "$pattern_regex"; then
            return 0
        fi
    done
    return 1
}

# =============================================================================
# Check 1: No serde(default) in internal crates
# =============================================================================
check_internal_serde_defaults() {
    log_info "Checking for serde(default) in internal crates..."

    # Internal crate paths (not vendor connectors)
    INTERNAL_PATHS=(
        "crates/quantlaxmi-models/src"
        "crates/quantlaxmi-wal/src"
        "crates/quantlaxmi-events/src"
        "crates/quantlaxmi-gates/src"
        "crates/quantlaxmi-runner-common/src"
        "crates/quantlaxmi-runner-crypto/src"
        "crates/quantlaxmi-runner-india/src"
        "crates/quantlaxmi-core/src"
        "crates/quantlaxmi-strategy/src"
    )

    local found=0

    for path in "${INTERNAL_PATHS[@]}"; do
        if [[ -d "$REPO_ROOT/$path" ]]; then
            # Search for serde(default) but exclude Option<T> fields and tests
            # This regex catches: #[serde(default)] and #[serde(default = "...")]
            local matches
            matches=$(grep -rn '#\[serde(default' "$REPO_ROOT/$path" \
                --include="*.rs" \
                2>/dev/null || true)

            if [[ -n "$matches" ]]; then
                # Filter out legitimate cases:
                # - Option<T> fields (where None is a real state)
                # - Test modules
                # - External API response types (Kite*, Zerodha*, Binance*)
                while IFS= read -r line; do
                    local filepath
                    filepath=$(echo "$line" | cut -d: -f1)

                    # Skip if in test module
                    if echo "$line" | grep -q "#\[cfg(test)\]"; then
                        log_verbose "Skipping test: $line"
                        continue
                    fi

                    # Skip if in vendor capture files (these handle external API responses)
                    if echo "$filepath" | grep -qE "zerodha_capture|binance_capture|kite_"; then
                        log_verbose "Skipping vendor capture file: $line"
                        continue
                    fi

                    # To check if field is Option<T>, we need to look at the next line
                    # For now, use a heuristic: check if the match file+line has Option nearby
                    local linenum
                    linenum=$(echo "$line" | cut -d: -f2)
                    local context
                    context=$(sed -n "${linenum},$((linenum+1))p" "$filepath" 2>/dev/null || echo "")

                    # Skip if field is Option<...> (serde(default) on Option = legitimate)
                    if echo "$context" | grep -q "Option<"; then
                        log_verbose "Skipping Option<T> field: $line"
                        continue
                    fi

                    log_error "D2 violation: serde(default) in internal crate"
                    echo "       $line"
                    found=$((found + 1))
                done <<< "$matches"
            fi
        fi
    done

    if [[ $found -gt 0 ]]; then
        VIOLATIONS=$((VIOLATIONS + found))
        log_error "Found $found serde(default) violations in internal crates"
    else
        log_info "No serde(default) violations in internal crates"
    fi
}

# =============================================================================
# Check 2: No unwrap_or(0) on vendor quantity/volume fields
# =============================================================================
check_vendor_unwrap_or_zero() {
    log_info "Checking for unwrap_or(0) on vendor fields..."

    # Patterns that indicate vendor field poisoning
    # These field names are the ones we changed from u64 to Option<u64>
    # NOTE: We match field.unwrap_or directly to avoid false positives from:
    # - self.volumes.back().unwrap_or() (collection access, not vendor field)
    # - self.volume_ma (different field entirely)
    VENDOR_FIELD_PATTERNS=(
        "buy_quantity\.unwrap_or.*0"
        "sell_quantity\.unwrap_or.*0"
        "last_quantity\.unwrap_or.*0"
        "\.volume\.unwrap_or.*0"
    )

    local found=0

    for pattern in "${VENDOR_FIELD_PATTERNS[@]}"; do
        local matches
        matches=$(grep -rn "$pattern" "$REPO_ROOT/crates" \
            --include="*.rs" \
            2>/dev/null || true)

        if [[ -n "$matches" ]]; then
            while IFS= read -r line; do
                # Skip if explicitly marked as cosmetic-only
                if echo "$line" | grep -qi "cosmetic"; then
                    log_verbose "Skipping cosmetic-only: $line"
                    continue
                fi

                # Skip if in test file
                if echo "$line" | grep -q "_test\.rs\|tests/\|#\[test\]"; then
                    log_verbose "Skipping test: $line"
                    continue
                fi

                log_error "D1 violation: unwrap_or(0) on vendor field"
                echo "       $line"
                found=$((found + 1))
            done <<< "$matches"
        fi
    done

    if [[ $found -gt 0 ]]; then
        VIOLATIONS=$((VIOLATIONS + found))
        log_error "Found $found unwrap_or(0) violations on vendor fields"
    else
        log_info "No unwrap_or(0) violations on vendor fields"
    fi
}

# =============================================================================
# Check 3: Generic Option<...>.unwrap_or(0) without cosmetic label
# =============================================================================
check_generic_option_unwrap_zero() {
    log_info "Checking for Option<...>.unwrap_or(0) patterns..."

    # This is a broader check - any Option unwrapped to 0 in signal paths
    # We check connectors and core signal code
    SIGNAL_PATHS=(
        "crates/quantlaxmi-connectors-zerodha/src"
        "crates/quantlaxmi-connectors-binance/src"
        "crates/quantlaxmi-core/src"
        "crates/quantlaxmi-strategy/src"
    )

    local found=0
    local warned=0

    for path in "${SIGNAL_PATHS[@]}"; do
        if [[ -d "$REPO_ROOT/$path" ]]; then
            # Look for Option<u64> or Option<i64> followed by unwrap_or(0)
            # This is a heuristic - may have false positives
            local matches
            matches=$(grep -rn "\.unwrap_or(0)" "$REPO_ROOT/$path" \
                --include="*.rs" \
                2>/dev/null || true)

            if [[ -n "$matches" ]]; then
                while IFS= read -r line; do
                    # Skip if explicitly marked as cosmetic-only or display-only
                    if echo "$line" | grep -qiE "cosmetic|display|debug|log|trace"; then
                        log_verbose "Skipping labeled: $line"
                        continue
                    fi

                    # Skip timestamp conversions (timestamp_nanos_opt().unwrap_or(0))
                    if echo "$line" | grep -q "timestamp.*unwrap_or(0)"; then
                        continue
                    fi

                    # Skip instrument token lookups (these are instrument IDs, not quantities)
                    if echo "$line" | grep -q "token.*unwrap_or(0)\|lookup.*unwrap_or(0)"; then
                        continue
                    fi

                    # Skip parse errors (these are input parsing, not vendor fields)
                    if echo "$line" | grep -q "parse().*unwrap_or(0)"; then
                        continue
                    fi

                    # Skip file operations
                    if echo "$line" | grep -q "count_file\|lines\|size"; then
                        continue
                    fi

                    # Check allowlist before warning
                    local filepath content
                    filepath=$(echo "$line" | cut -d: -f1)
                    content=$(echo "$line" | cut -d: -f3-)
                    if is_allowlisted "$filepath" "$content"; then
                        log_verbose "Suppressed via allowlist: $line"
                        SUPPRESSED_COUNT=$((SUPPRESSED_COUNT + 1))
                        continue
                    fi

                    # Warn but don't fail for ambiguous cases
                    log_warn "Review needed: unwrap_or(0) in signal path"
                    echo "       $line"
                    warned=$((warned + 1))
                done <<< "$matches"
            fi
        fi
    done

    if [[ $warned -gt 0 ]]; then
        log_warn "Found $warned cases to review (not failing build)"
    elif [[ $SUPPRESSED_COUNT -gt 0 ]]; then
        log_info "All unwrap_or(0) patterns allowlisted ($SUPPRESSED_COUNT suppressed)"
    else
        log_info "No suspicious unwrap_or(0) patterns in signal paths"
    fi
}

# =============================================================================
# Check 4: Verify vendor_fields.rs exists and exports required helpers
# =============================================================================
check_vendor_fields_module() {
    log_info "Checking vendor_fields module exists..."

    local vendor_fields="$REPO_ROOT/crates/quantlaxmi-connectors-zerodha/src/vendor_fields.rs"

    if [[ ! -f "$vendor_fields" ]]; then
        log_error "D1 violation: vendor_fields.rs not found"
        VIOLATIONS=$((VIOLATIONS + 1))
        return
    fi

    # Check for required exports
    local required_exports=(
        "MissingVendorField"
        "require_u64"
        "book_imbalance_fixed"
        "IMBALANCE_EXP"
    )

    for export in "${required_exports[@]}"; do
        if ! grep -q "pub.*$export\|pub const $export" "$vendor_fields"; then
            log_error "vendor_fields.rs missing required export: $export"
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done

    log_info "vendor_fields module verified"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=============================================="
    echo "No Silent Poisoning Doctrine Lint"
    echo "=============================================="
    echo ""

    cd "$REPO_ROOT"

    # Load allowlist for suppressing known-safe warnings
    load_allowlist

    check_internal_serde_defaults
    echo ""

    check_vendor_unwrap_or_zero
    echo ""

    check_generic_option_unwrap_zero
    echo ""

    check_vendor_fields_module
    echo ""

    echo "=============================================="
    if [[ $VIOLATIONS -gt 0 ]]; then
        log_error "FAILED: $VIOLATIONS doctrine violations found"
        echo ""
        echo "See: docs/DOCTRINE_NO_SILENT_POISONING.md"
        exit 1
    else
        if [[ $SUPPRESSED_COUNT -gt 0 ]]; then
            log_info "PASSED: No doctrine violations ($SUPPRESSED_COUNT warnings suppressed via allowlist)"
        else
            log_info "PASSED: No doctrine violations"
        fi
        exit 0
    fi
}

main "$@"
