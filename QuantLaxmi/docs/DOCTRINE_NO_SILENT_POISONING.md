# Doctrine: No Silent Poisoning

**Status:** ENFORCED
**Effective:** 2026-01-27
**Scope:** All QuantLaxmi crates

## Purpose

This doctrine eliminates the most dangerous class of bugs in quantitative systems:
**plausible but false data**.

When vendor data is missing, incomplete, or uncertain, the system must either:
1. Propagate that uncertainty explicitly (`Option<T>`, `Result<T, E>`)
2. Refuse to compute (return `None` or `Err`)
3. Fail loudly at deserialization

The system must **never** silently substitute a default value that could change a trading decision.

---

## Rules

### D1: No Defaulting Vendor Omissions into Numeric Values

**Rationale:** `0` is a real market value (no depth, no volume). `None` means "unknown."
Conflating them poisons liquidity signals and can cause catastrophic position sizing errors.

**Banned patterns:**
```rust
// BANNED: serde(default) on non-Option numeric vendor fields
#[serde(default)]
pub buy_quantity: u64,  // If vendor omits, becomes 0 = "no buyers" (LIE)

// BANNED: unwrap_or(0) on vendor Option<Number>
let qty = quote.buy_quantity.unwrap_or(0);  // Silent poisoning

// BANNED: unwrap_or_default() on vendor Option<Number>
let vol = quote.volume.unwrap_or_default();  // Same problem
```

**Required patterns:**
```rust
// CORRECT: Option<T> for vendor fields that may be omitted
pub buy_quantity: Option<u64>,

// CORRECT: Propagate Option
let imbalance = book_imbalance_fixed(quote.buy_quantity, quote.sell_quantity)?;

// CORRECT: Use require_*() helpers
let qty = require_u64(quote.buy_quantity, "buy_quantity")?;

// CORRECT: Explicit "cosmetic-only" label if truly display-only
let display_vol = quote.volume.unwrap_or(0);  // OK only in UI, never in signals
```

**Exception:** `unwrap_or(0)` is permitted ONLY for:
- Pure display/logging (never feeds computation)
- Explicitly labeled with `// cosmetic-only` comment
- Never in signal/feature/risk computation paths

---

### D2: Internals Are Strict

**Rationale:** Internal artifacts (WAL, manifests, configs, events) are produced by our code.
A missing field means a producer bug, not vendor uncertainty. Fail-fast exposes bugs early.

**Banned patterns:**
```rust
// BANNED in internal types (WAL, manifests, configs, events):
#[serde(default)]
pub depth: u64,

#[serde(default = "default_value")]
pub schema_version: u32,
```

**Required patterns:**
```rust
// CORRECT: Required fields fail deserialization if missing
pub depth: u64,  // No serde(default)
pub schema_version: u32,  // No serde(default)

// CORRECT: Option<T> only when None is a real lifecycle state
pub trace_binding: Option<TraceBinding>,  // None = not yet computed (valid state)
pub end_ts: Option<DateTime<Utc>>,  // None = segment still running (valid state)
```

**Rule:** `Option<T>` in internal types must represent a **real lifecycle state**, not missing data.

---

### D3: Deterministic Rounding Is Explicit

**Rationale:** Fixed-point arithmetic requires frozen rounding rules. Implicit rounding
creates non-deterministic replay and subtle cross-system divergence.

**Required:**
1. Any fixed-point division must state rounding semantics in doc comment
2. At least one test must verify the exact rounding behavior
3. Use named constants for scale factors (e.g., `IMBALANCE_EXP`, `IMBALANCE_MAX`)

**Example:**
```rust
/// ## Fixed-Point Arithmetic
/// - Rounding: **truncation toward zero** (Rust integer division semantics)
///   - Example: -3 / 2 = -1 (not -2)
///   - Why acceptable: canonical rounding rule; max error 1 mantissa unit
pub fn book_imbalance_fixed(...) -> Option<(i64, i8)> { ... }

#[test]
fn test_truncation_toward_zero() {
    // Verify: -10000 / 3 truncates toward zero = -3333
    assert_eq!(-10000_i128 / 3, -3333);
}
```

---

### D4: Signal Safety Boundary

**Rationale:** Any computation that can change a trading decision must have explicit
data requirements. Uncertainty must be surfaced, not hidden.

**Required:**
1. Signal/feature computation functions must accept `Option<T>` or `Result<T, E>` inputs
2. If required data is missing, return `None` or `Err` (never fabricate a value)
3. The "refuse to compute" path must be observable (logs, metrics, or decision trace)

**Example:**
```rust
// CORRECT: Returns None if vendor omitted required fields
pub fn book_imbalance_fixed(
    buy_quantity: Option<u64>,
    sell_quantity: Option<u64>,
) -> Option<(i64, i8)> {
    let buy = buy_quantity?;  // Propagate None
    let sell = sell_quantity?;  // Propagate None
    // ... compute only if both present
}

// CORRECT: Caller handles uncertainty explicitly
match book_imbalance_fixed(quote.buy_quantity, quote.sell_quantity) {
    Some((mantissa, exp)) => use_signal(mantissa, exp),
    None => {
        tracing::debug!("skipped imbalance signal: missing vendor field");
        // Don't trade on fabricated data
    }
}
```

---

## Enforcement

### CI Lint (Automated)

`scripts/lint_no_silent_poisoning.sh` runs on every PR and fails if:

1. `#[serde(default)]` appears in internal crate paths (WAL, manifests, configs, events)
2. `unwrap_or(0)` pattern on vendor quantity/volume fields
3. `Option<...>.*unwrap_or(0)` without `// cosmetic-only` comment

### Code Review (Manual)

Reviewers must verify:
1. New vendor fields are `Option<T>` if omission is possible
2. New signal computations propagate `Option`/`Result` correctly
3. Fixed-point divisions have rounding doc + test

### Crate-Level Lints (Optional)

Internal crates MAY enable:
```rust
#![deny(clippy::unwrap_used)]  // Forces explicit error handling
#![deny(clippy::expect_used)]
```

---

## Rationale Summary

| Violation | Risk | Outcome |
|-----------|------|---------|
| Default 0 for missing quantity | Position sized on phantom liquidity | Catastrophic loss |
| Default in internal manifest | Corrupted artifact accepted silently | Replay divergence |
| Implicit rounding | Cross-system numeric drift | Audit failure |
| Signal on uncertain data | Trading decision on fabricated input | Compliance violation |

---

## MarketSnapshot: Versioned Encoding with Explicit Presence

**Status:** IMPLEMENTED (Phase 19)

The `MarketSnapshot` type uses a versioned enum to explicitly track field presence:

```rust
#[serde(tag = "schema", rename_all = "snake_case")]
pub enum MarketSnapshot {
    V1(MarketSnapshotV1),  // Legacy: no presence tracking
    V2(MarketSnapshotV2),  // Current: explicit l1_state_bits
}
```

### FieldState Enum

Each L1 field (bid_price, ask_price, bid_qty, ask_qty) has an explicit state:

| State | Value | Meaning |
|-------|-------|---------|
| `Absent` | 0 | Vendor did not send this field |
| `Null` | 1 | Vendor explicitly sent null |
| `Value` | 2 | Vendor sent a valid value (mantissa is meaningful) |
| `Malformed` | 3 | Vendor sent unparseable data |

### l1_state_bits Encoding

Packed into a `u16` (2 bits per field):
- Bits 0-1: bid_price
- Bits 2-3: ask_price
- Bits 4-5: bid_qty
- Bits 6-7: ask_qty

### Canonical Bytes Encoding

**ENCODING_VERSION = 0x03**

| Field | Offset | Size | Notes |
|-------|--------|------|-------|
| schema discriminant | 0 | 1 | 0x01=V1, 0x02=V2 |
| bid_price_mantissa | 1 | 8 | i64 LE |
| ask_price_mantissa | 9 | 8 | i64 LE |
| bid_qty_mantissa | 17 | 8 | i64 LE |
| ask_qty_mantissa | 25 | 8 | i64 LE |
| price_exponent | 33 | 1 | i8 |
| qty_exponent | 34 | 1 | i8 |
| spread_bps_mantissa | 35 | 8 | i64 LE |
| book_ts_ns | 43 | 8 | i64 LE |
| l1_state_bits (V2) | 51 | 2 | u16 LE |

**Key invariants:**
- V1 canonical bytes: 51 bytes (discriminant 0x01)
- V2 canonical bytes: 53 bytes (discriminant 0x02)
- Replay parity is version-scoped: cross-version comparison is a category error

### Serialization vs Canonical Encoding

| Layer | Format | Purpose |
|-------|--------|---------|
| Serde (JSON/WAL) | `{"schema": "v1", ...}` | Human-readable, backward-compatible |
| Canonical bytes | `0x01` / `0x02` prefix | Cryptographic identity, digest-stable |

This separation is intentional: human-readable evolution is handled by serde;
cryptographic identity is handled by canonical bytes.

---

## References

- `crates/quantlaxmi-models/src/events.rs` — FieldState, MarketSnapshot V1/V2
- `crates/quantlaxmi-events/src/trace.rs` — Canonical encoding, ENCODING_VERSION
- `crates/quantlaxmi-connectors-zerodha/src/vendor_fields.rs` — Reference implementation
- `scripts/lint_no_silent_poisoning.sh` — CI enforcement script

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-28 | Added MarketSnapshot V2 with l1_state_bits, ENCODING_VERSION 0x03 |
| 2026-01-27 | Initial doctrine established |
