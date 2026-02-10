# QuantLaxmi Redundancy Analysis

**Version**: 1.0.0
**Date**: 2026-01-30
**Analysis Scope**: 183 Rust source files, 125,252 lines of code

---

## Executive Summary

This document presents findings from a comprehensive redundancy analysis of the QuantLaxmi codebase. Issues are categorized by severity and actionability.

**Overall Health**: The codebase is well-structured with most "dead code" being intentional scaffolding. However, there are critical duplications that should be addressed.

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Dead Code | 0 | 0 | 2 | 14 |
| Duplications | 1 | 1 | 3 | 1 |
| Obsolete Patterns | 0 | 1 | 3 | 3 |
| **Total** | **1** | **2** | **8** | **18** |

---

## 1. CRITICAL: Duplicate `parse_to_mantissa_pure()` Implementation

**Severity**: CRITICAL
**Impact**: Maintenance burden, potential inconsistency bugs

The canonical mantissa parsing function is implemented **4 times**:

| Location | Status |
|----------|--------|
| `quantlaxmi-models/src/events.rs:711` | **CANONICAL** - Keep this |
| `quantlaxmi-runner-crypto/src/binance_perp_capture.rs:134` | DUPLICATE - Remove |
| `quantlaxmi-runner-crypto/src/binance_funding_capture.rs:134` | DUPLICATE - Remove |
| `quantlaxmi-runner-crypto/src/binance_sbe_depth_capture.rs:68` | DUPLICATE - Remove |

**Recommended Fix**:
```rust
// In all runner-crypto files, replace local function with:
use quantlaxmi_models::parse_to_mantissa_pure;
```

---

## 2. HIGH: Duplicate SBE Decoder

**Severity**: HIGH
**Impact**: 585+ lines of duplicated binary parsing logic

Two nearly identical SBE decoders exist:

| File | Lines | Purpose |
|------|-------|---------|
| `quantlaxmi-sbe/src/lib.rs` | 303 | **CANONICAL** |
| `quantlaxmi-connectors-binance/src/sbe.rs` | 282 | DUPLICATE |

**Recommended Fix**:
```toml
# In quantlaxmi-connectors-binance/Cargo.toml, add:
quantlaxmi-sbe = { path = "../quantlaxmi-sbe" }

# Then remove sbe.rs and import from canonical crate
```

---

## 3. HIGH: Rust Edition Inconsistency

**Severity**: HIGH
**Impact**: Tooling complications, inconsistent language features

6 crates use **edition 2021** while 11 use **edition 2024**:

**Edition 2021 (outdated)**:
- `quantlaxmi-sbe/Cargo.toml`
- `quantlaxmi-options/Cargo.toml`
- `quantlaxmi-executor/Cargo.toml`
- `quantlaxmi-data/Cargo.toml`
- `quantlaxmi-risk/Cargo.toml`

**Recommended Fix**: Migrate all to edition 2024.

---

## 4. MEDIUM: Scattered Encoding Utilities

**Severity**: MEDIUM
**Impact**: Code duplication, maintenance overhead

Binary encoding utilities are defined in two places:

| File | Functions | Scope |
|------|-----------|-------|
| `quantlaxmi-strategy/src/canonical.rs:41-88` | `encode_i8`, `encode_i64`, `encode_i32`, `encode_string`, etc. | Public |
| `quantlaxmi-events/src/trace.rs:488-631` | `encode_i8`, `encode_i64`, `encode_string`, `encode_datetime`, etc. | Private |

**Overlap**: `encode_i8()`, `encode_i64()`, `encode_string()` are duplicated.

**Recommended Fix**: Create shared `quantlaxmi-encoding` module or consolidate into `quantlaxmi-models`.

---

## 5. MEDIUM: Duplicate `fetch_depth_snapshot()`

**Severity**: MEDIUM
**Impact**: API inconsistency, maintenance burden

| File | Function | Returns |
|------|----------|---------|
| `quantlaxmi-connectors-binance/src/binance.rs:574` | `fetch_depth_snapshot()` | `DepthResponse` |
| `quantlaxmi-runner-crypto/src/binance_sbe_depth_capture.rs:130` | `fetch_depth_snapshot()` | `DepthSnapshot` |

**Recommended Fix**: Consolidate into connector crate with unified return type.

---

## 6. MEDIUM: Unimplemented Scaffold Gates

**Severity**: MEDIUM
**Impact**: Incomplete validation pipeline

### G2 BacktestCorrectness (`g2_backtest_correctness.rs`)
Returns placeholder "SCAFFOLD" for:
- `no_lookahead` (line 90-93)
- `fill_realism` (line 95-98)
- `transaction_costs` (line 100-103)
- `market_impact` (line 105-108)
- `data_quality` (line 110-113)

### G3 Robustness (`g3_robustness.rs`)
Returns placeholder "SCAFFOLD" for:
- `connection_loss` (line 85-88)
- `data_gaps` (line 90-93)
- `extreme_prices` (line 95-98)
- `high_latency` (line 100-103)
- `partial_fills` (line 105-108)
- `memory_pressure` (line 110-113)

**Recommended Fix**: Implement these checks or mark gates as "Phase 3" in docs.

---

## 7. LOW: Unused Imports

**Severity**: LOW
**Impact**: Minor code cleanliness

| File | Import |
|------|--------|
| `binance_funding_capture.rs:25` | `use futures_util::StreamExt;` |
| `binance_perp_capture.rs:20` | `use futures_util::StreamExt;` |

**Recommended Fix**: Remove unused imports.

---

## 8. LOW: Unused Private Methods

**Severity**: LOW
**Impact**: Minor dead code

| File | Method |
|------|--------|
| `funding_bias.rs:266` | `fn is_flat(&self) -> bool` |

**Note**: Companion to `is_long()` and `is_short()` - may be retained for API completeness.

---

## 9. LOW: Dead Code Suppressions

**Severity**: LOW
**Impact**: Technical debt marker

62 `#[allow(dead_code)]` annotations across codebase. Most are intentional:
- Test utilities
- Debug helpers
- Future expansion points
- API completeness (paired methods)

**Notable Patterns**:
- `sanos_*.rs` binaries: 24+ suppressions for debug tooling
- Gate structs: Config fields retained for future phases

---

## 10. INFORMATIONAL: Legacy Schema Support

The codebase maintains backward compatibility with:

- **V1/V2 MarketSnapshot schemas** in `quantlaxmi-models`
- **Legacy float fallback** in backtest engine
- **canonical_v1 schema** requirement in session validation

**Status**: Intentional design, not redundancy. Document sunset timeline.

---

## Action Plan

### Immediate (Before Live Trading)

1. ~~**Remove duplicate `parse_to_mantissa_pure()`**~~ - ✅ DONE (2026-01-30)
   - `binance_funding_capture.rs`: Replaced 60-line duplicate with 1-line delegation
   - `binance_sbe_depth_capture.rs`: Replaced 60-line duplicate with 1-line delegation
   - `binance_perp_capture.rs`: Already using delegation pattern

2. ~~**Remove duplicate SBE decoder**~~ - ✅ DONE (2026-01-30)
   - Deleted `quantlaxmi-connectors-binance/src/sbe.rs` (282 lines)
   - Updated lib.rs to re-export from `quantlaxmi-sbe` crate

3. ~~**Remove unused imports**~~ - ✅ DONE (2026-01-30)
   - `binance_perp_capture.rs`: Removed `futures_util::StreamExt`
   - `binance_funding_capture.rs`: Removed `futures_util::StreamExt`

### Short-term (This Sprint)

4. ~~**Upgrade edition 2021 → 2024**~~ - ✅ DONE (2026-01-30)
   - Updated: quantlaxmi-data, quantlaxmi-executor, quantlaxmi-options, quantlaxmi-risk, quantlaxmi-sbe

5. ~~**Consolidate encoding utilities**~~ - ✅ DONE (2026-01-30)
   - Created `quantlaxmi-models/src/encoding.rs` with shared primitives
   - Updated `quantlaxmi-strategy/src/canonical.rs` to use shared encoders
   - Updated `quantlaxmi-events/src/trace.rs` to use shared encoders

### Medium-term (This Quarter)

6. ~~**Implement G2/G3 scaffold methods**~~ - ✅ DONE (2026-01-30)
   - G2 BacktestCorrectness: Implemented `check_lookahead`, `check_fill_realism`, `check_transaction_costs`, `check_data_quality`, `check_market_impact`
   - G3 Robustness: Implemented `check_connection_loss`, `check_data_gaps`, `check_extreme_prices`, `check_high_latency`, `check_partial_fills`, `check_memory_pressure`
   - Added `CheckResult::warn()` method for warning-level checks
   - Added `SystemConfig` struct for robustness configuration validation
7. **Consolidate depth snapshot functions** - 2 files - PENDING
8. **Clean up `#[allow(dead_code)]`** - Review 62 occurrences - PENDING

---

## Verification Commands

```bash
# Check for unused imports
cargo clippy --workspace 2>&1 | grep "unused import"

# Check for dead code
cargo clippy --workspace 2>&1 | grep "never used"

# Find duplicate function names
rg "fn parse_to_mantissa" crates/

# Verify edition consistency
grep -r "edition = " crates/*/Cargo.toml
```

---

## Appendix: Files Modified in This Analysis

None - analysis only. See Action Plan for recommended changes.
