//! Vendor field validation helpers.
//!
//! Handles the distinction between "vendor omitted field" (None) and
//! "vendor asserted zero" (Some(0)). This prevents silent poisoning
//! of liquidity/volume signals.
//!
//! ## Rule
//! If a None would silently change a trading decision, refuse to compute.
//! Return Option<T> or Result<T, MissingVendorField> and force caller to decide.
//!
//! ## Policy for Call Sites
//!
//! **Execution-critical features** (signals that affect order routing/sizing):
//! - Return `Result<_, MissingVendorField>`
//! - Surface error to control/observability layer
//! - Example: book imbalance used in position sizing
//!
//! **Analytics-only features** (monitoring, dashboards, non-trading):
//! - Return `Option<_>`
//! - Log "skipped due to missing vendor field" at debug level
//! - Example: historical volume stats for UI
//!
//! **Never**: `unwrap_or(0)` or any silent default on vendor-omitted fields.

use std::fmt;

// =============================================================================
// Constants
// =============================================================================

/// Exponent for book imbalance fixed-point representation.
///
/// ## Scaling Convention
/// - Mantissa range: -10000 to +10000
/// - Exponent: -4 (so mantissa / 10^4 = decimal value)
///
/// ## Examples
/// | Mantissa | Decimal | Meaning                    |
/// |----------|---------|----------------------------|
/// | +10000   | +1.0000 | Full buy dominance         |
/// | +5000    | +0.5000 | 75% buy, 25% sell          |
/// | 0        | 0.0000  | Balanced (or both zero)    |
/// | -5000    | -0.5000 | 25% buy, 75% sell          |
/// | -10000   | -1.0000 | Full sell dominance        |
pub const IMBALANCE_EXP: i8 = -4;

/// Maximum imbalance mantissa (full buy dominance).
pub const IMBALANCE_MAX: i64 = 10000;

/// Minimum imbalance mantissa (full sell dominance).
pub const IMBALANCE_MIN: i64 = -10000;

// =============================================================================
// Error Types
// =============================================================================

/// Error when a required vendor field is missing.
///
/// Use this for fields that are REQUIRED for a specific computation,
/// not for fields that are always optional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingVendorField {
    /// Field name (for logging/metrics).
    pub field: &'static str,
}

impl fmt::Display for MissingVendorField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vendor omitted required field: {}", self.field)
    }
}

impl std::error::Error for MissingVendorField {}

/// Require a u64 field that the vendor may have omitted.
///
/// Use this when the field is REQUIRED for the current computation.
/// Returns Err if None, allowing caller to skip/log/propagate.
///
/// # Example
/// ```ignore
/// let buy = require_u64(quote.buy_quantity, "buy_quantity")?;
/// let sell = require_u64(quote.sell_quantity, "sell_quantity")?;
/// // Both present, safe to compute imbalance
/// ```
#[inline]
pub fn require_u64(field: Option<u64>, name: &'static str) -> Result<u64, MissingVendorField> {
    field.ok_or(MissingVendorField { field: name })
}

/// Require an i64 field that the vendor may have omitted.
#[inline]
pub fn require_i64(field: Option<i64>, name: &'static str) -> Result<i64, MissingVendorField> {
    field.ok_or(MissingVendorField { field: name })
}

/// Pass through optional field unchanged.
///
/// Trivial helper for symmetry with require_* functions.
/// Use when field is truly optional for the computation.
#[inline]
pub fn optional_u64(field: Option<u64>) -> Option<u64> {
    field
}

/// Compute book imbalance from buy/sell quantities (fixed-point).
///
/// ## Return Value
/// Returns `(imbalance_mantissa, IMBALANCE_EXP)` where:
/// - Mantissa range: [`IMBALANCE_MIN`, `IMBALANCE_MAX`] (-10000 to +10000)
/// - Exponent: [`IMBALANCE_EXP`] (-4)
/// - Formula: `(buy - sell) / (buy + sell)`
///
/// ## Semantics
/// - `None` = "cannot compute, vendor omitted field" (propagate uncertainty)
/// - `Some((0, IMBALANCE_EXP))` = "computed, market is balanced or empty" (real state)
/// - `Some((10000, IMBALANCE_EXP))` = "computed, full buy dominance" (real state)
///
/// ## Edge Cases
/// - Both quantities zero: returns `Some((0, IMBALANCE_EXP))` (balanced, not unknown)
/// - One quantity missing: returns `None` (cannot compute)
/// - Overflow: impossible (uses i128 for both numerator and denominator)
///
/// ## Fixed-Point Arithmetic
/// - Scale factor: 10^4 (so 10000 = 1.0000)
/// - Rounding: **truncation toward zero** (Rust integer division semantics)
///   - This is deterministic but differs from Python/pandas floor division
///   - Example: -3 / 2 = -1 (not -2)
///   - Why acceptable: we're mapping to a fixed-point indicator where truncation
///     is the canonical rounding rule; the error is at most 1 mantissa unit (0.0001)
/// - Clamp: defensive only, mathematically redundant for valid inputs
///
/// ## Overflow Analysis
/// - `buy`, `sell`: u64 (max ~1.8e19 each)
/// - `total_i128 = buy + sell`: max ~3.6e19, fits in i128
/// - `num = (buy - sell) * 10000`: max magnitude ~1.8e23, fits in i128
/// - Result: always in [-10000, 10000] when total > 0
pub fn book_imbalance_fixed(
    buy_quantity: Option<u64>,
    sell_quantity: Option<u64>,
) -> Option<(i64, i8)> {
    let buy = buy_quantity?;
    let sell = sell_quantity?;

    // Use i128 for ALL arithmetic to avoid any overflow/saturation issues.
    // This makes the function mathematically correct for all representable inputs.
    let buy_i128 = buy as i128;
    let sell_i128 = sell as i128;
    let total_i128 = buy_i128 + sell_i128; // Safe: max is 2 * u64::MAX ≈ 3.6e19

    if total_i128 == 0 {
        // Both zero = balanced (real market state, not unknown)
        return Some((0, IMBALANCE_EXP));
    }

    // (buy - sell) * IMBALANCE_MAX / total
    // Numerator max magnitude: u64::MAX * 10000 ≈ 1.8e23, fits in i128
    let mantissa = ((buy_i128 - sell_i128) * (IMBALANCE_MAX as i128)) / total_i128;

    // Clamp: mathematically redundant (result is always in [-10000, 10000]),
    // but defensive against future edits or subtle bugs.
    let mantissa = mantissa.clamp(IMBALANCE_MIN as i128, IMBALANCE_MAX as i128) as i64;

    Some((mantissa, IMBALANCE_EXP))
}

// =============================================================================
// Signal Requirements Declaration (Phase 18 Integration)
// =============================================================================

use quantlaxmi_models::{SignalRequirements, VendorField};

/// Signal ID for book imbalance.
pub const BOOK_IMBALANCE_SIGNAL_ID: &str = "book_imbalance";

/// Returns the admission requirements for the book imbalance signal.
///
/// Required vendor fields:
/// - BuyQuantity
/// - SellQuantity
///
/// No internal fields required.
/// No optional fields.
pub fn book_imbalance_requirements() -> SignalRequirements {
    SignalRequirements::new(
        BOOK_IMBALANCE_SIGNAL_ID,
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    )
}

/// Compute book imbalance with invariant check.
///
/// This function should only be called after admission confirms the required
/// fields are present. If admission said Admit but computation returns None,
/// that's a bug invariant violation.
///
/// # Panics
/// Panics if `buy_quantity` or `sell_quantity` is None (invariant violation).
/// This should never happen if admission was properly checked first.
pub fn book_imbalance_after_admission(
    buy_quantity: Option<u64>,
    sell_quantity: Option<u64>,
) -> (i64, i8) {
    book_imbalance_fixed(buy_quantity, sell_quantity)
        .expect("INVARIANT VIOLATION: admission said Admit but required field is None")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_u64_present() {
        let result = require_u64(Some(100), "test_field");
        assert_eq!(result, Ok(100));
    }

    #[test]
    fn test_require_u64_missing() {
        let result = require_u64(None, "buy_quantity");
        assert_eq!(
            result,
            Err(MissingVendorField {
                field: "buy_quantity"
            })
        );
    }

    #[test]
    fn test_missing_vendor_field_display() {
        let err = MissingVendorField {
            field: "sell_quantity",
        };
        assert_eq!(
            err.to_string(),
            "vendor omitted required field: sell_quantity"
        );
    }

    #[test]
    fn test_book_imbalance_both_present() {
        // buy=60, sell=40, total=100
        // imbalance = (60-40)/100 = 0.2 = 2000 with IMBALANCE_EXP
        let result = book_imbalance_fixed(Some(60), Some(40));
        assert_eq!(result, Some((2000, IMBALANCE_EXP)));
    }

    #[test]
    fn test_book_imbalance_balanced() {
        // buy=50, sell=50 -> imbalance = 0
        let result = book_imbalance_fixed(Some(50), Some(50));
        assert_eq!(result, Some((0, IMBALANCE_EXP)));
    }

    #[test]
    fn test_book_imbalance_all_buy() {
        // buy=100, sell=0 -> imbalance = 1.0 = IMBALANCE_MAX
        let result = book_imbalance_fixed(Some(100), Some(0));
        assert_eq!(result, Some((IMBALANCE_MAX, IMBALANCE_EXP)));
    }

    #[test]
    fn test_book_imbalance_all_sell() {
        // buy=0, sell=100 -> imbalance = -1.0 = IMBALANCE_MIN
        let result = book_imbalance_fixed(Some(0), Some(100));
        assert_eq!(result, Some((IMBALANCE_MIN, IMBALANCE_EXP)));
    }

    #[test]
    fn test_book_imbalance_both_zero() {
        // Real market state: no depth on either side
        let result = book_imbalance_fixed(Some(0), Some(0));
        assert_eq!(result, Some((0, IMBALANCE_EXP)));
    }

    #[test]
    fn test_book_imbalance_buy_missing() {
        // Vendor omitted buy_quantity -> cannot compute
        let result = book_imbalance_fixed(None, Some(100));
        assert_eq!(result, None);
    }

    #[test]
    fn test_book_imbalance_sell_missing() {
        // Vendor omitted sell_quantity -> cannot compute
        let result = book_imbalance_fixed(Some(100), None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_book_imbalance_both_missing() {
        // Vendor omitted both -> cannot compute
        let result = book_imbalance_fixed(None, None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_book_imbalance_large_values() {
        // Test with large values that could overflow u64 addition
        let buy = u64::MAX / 2;
        let sell = u64::MAX / 2 - 1000;
        let result = book_imbalance_fixed(Some(buy), Some(sell));
        // Should not panic, and result should be valid
        assert!(result.is_some());
        let (mantissa, exp) = result.unwrap();
        assert_eq!(exp, IMBALANCE_EXP);
        // With such large totals, the 1000 difference is insignificant
        // and rounds to 0 - this is mathematically correct
        assert!(mantissa >= 0);
        assert!(mantissa <= IMBALANCE_MAX);
    }

    #[test]
    fn test_book_imbalance_large_but_significant() {
        // Test with large values where the difference IS significant
        let buy = 1_000_000_000u64;
        let sell = 900_000_000u64;
        let result = book_imbalance_fixed(Some(buy), Some(sell));
        assert!(result.is_some());
        let (mantissa, exp) = result.unwrap();
        assert_eq!(exp, IMBALANCE_EXP);
        // (1B - 0.9B) / 1.9B = 0.1B / 1.9B ≈ 0.0526 = 526 with IMBALANCE_EXP
        assert!(
            mantissa > 500 && mantissa < 550,
            "mantissa was {}",
            mantissa
        );
    }

    #[test]
    fn test_constants_consistency() {
        // Verify constants are consistent with each other
        assert_eq!(IMBALANCE_EXP, -4);
        assert_eq!(IMBALANCE_MAX, 10000);
        assert_eq!(IMBALANCE_MIN, -10000);
        assert_eq!(IMBALANCE_MAX, -IMBALANCE_MIN);
    }

    #[test]
    fn test_clamp_never_triggers_extreme_values() {
        // Verify that clamp is mathematically redundant for all valid inputs.
        // For any non-zero total, result must be in [-10000, 10000].

        let extreme_cases: Vec<(u64, u64, &str)> = vec![
            (u64::MAX, u64::MAX, "both max"),
            (u64::MAX, 0, "buy=max, sell=0"),
            (0, u64::MAX, "buy=0, sell=max"),
            (u64::MAX, 1, "buy=max, sell=1"),
            (1, u64::MAX, "buy=1, sell=max"),
            (u64::MAX, u64::MAX - 1, "nearly equal at max"),
            (u64::MAX / 2, u64::MAX / 2, "half max each"),
            (u64::MAX / 2 + 1, u64::MAX / 2, "half max, slight imbalance"),
        ];

        for (buy, sell, desc) in extreme_cases {
            let result = book_imbalance_fixed(Some(buy), Some(sell));
            assert!(result.is_some(), "case '{}' returned None", desc);

            let (mantissa, exp) = result.unwrap();
            assert_eq!(exp, IMBALANCE_EXP, "case '{}' wrong exponent", desc);

            // Verify result is within valid range (clamp should never have activated)
            assert!(
                (IMBALANCE_MIN..=IMBALANCE_MAX).contains(&mantissa),
                "case '{}': mantissa {} out of range [{}, {}]",
                desc,
                mantissa,
                IMBALANCE_MIN,
                IMBALANCE_MAX
            );

            // Verify mathematical correctness for extreme cases
            // Note: when difference is insignificant compared to total,
            // truncation toward zero can produce mantissa=0 even if buy != sell
            if buy == sell {
                assert_eq!(mantissa, 0, "case '{}' should be balanced", desc);
            } else if buy > sell {
                assert!(mantissa >= 0, "case '{}' should be non-negative", desc);
            } else {
                assert!(mantissa <= 0, "case '{}' should be non-positive", desc);
            }
        }
    }

    #[test]
    fn test_truncation_toward_zero() {
        // Verify truncation toward zero (not floor) for negative results
        // Example: buy=1, sell=3, total=4
        // imbalance = (1-3) * 10000 / 4 = -20000 / 4 = -5000
        let result = book_imbalance_fixed(Some(1), Some(3));
        assert_eq!(result, Some((-5000, IMBALANCE_EXP)));

        // Example: buy=1, sell=2, total=3
        // imbalance = (1-2) * 10000 / 3 = -10000 / 3 = -3333 (truncated, not -3334)
        let result = book_imbalance_fixed(Some(1), Some(2));
        assert_eq!(result, Some((-3333, IMBALANCE_EXP)));

        // Verify: -10000 / 3 truncates toward zero = -3333
        assert_eq!(-10000_i128 / 3, -3333);
    }

    #[test]
    fn test_exact_extremes_small_denominator() {
        // Test (0, 1) and (1, 0) with small denominators to catch scaling mistakes.
        // These should produce exact -10000 and +10000.

        // buy=1, sell=0, total=1
        // imbalance = (1-0) * 10000 / 1 = 10000
        let result = book_imbalance_fixed(Some(1), Some(0));
        assert_eq!(result, Some((IMBALANCE_MAX, IMBALANCE_EXP)));

        // buy=0, sell=1, total=1
        // imbalance = (0-1) * 10000 / 1 = -10000
        let result = book_imbalance_fixed(Some(0), Some(1));
        assert_eq!(result, Some((IMBALANCE_MIN, IMBALANCE_EXP)));

        // Also test (2, 0) and (0, 2) to verify scaling with total > 1
        // buy=2, sell=0, total=2
        // imbalance = (2-0) * 10000 / 2 = 20000 / 2 = 10000
        let result = book_imbalance_fixed(Some(2), Some(0));
        assert_eq!(result, Some((IMBALANCE_MAX, IMBALANCE_EXP)));

        // buy=0, sell=2, total=2
        // imbalance = (0-2) * 10000 / 2 = -20000 / 2 = -10000
        let result = book_imbalance_fixed(Some(0), Some(2));
        assert_eq!(result, Some((IMBALANCE_MIN, IMBALANCE_EXP)));
    }

    #[test]
    fn test_scale_invariant() {
        // Property-style test: for all inputs, verify:
        // 1. abs(mantissa) <= IMBALANCE_MAX
        // 2. mantissa == 0 when buy == sell

        // Representative sample covering different magnitudes and ratios
        let test_cases: Vec<(u64, u64)> = vec![
            // Equal values (should all produce 0)
            (0, 0),
            (1, 1),
            (100, 100),
            (1_000_000, 1_000_000),
            (u64::MAX, u64::MAX),
            // Unequal values (should produce non-zero in valid range)
            (1, 0),
            (0, 1),
            (100, 50),
            (50, 100),
            (1_000_000, 1),
            (1, 1_000_000),
            (u64::MAX, 0),
            (0, u64::MAX),
            (u64::MAX / 3, u64::MAX / 2),
            // Edge ratios
            (3, 1),  // 75% buy
            (1, 3),  // 75% sell
            (99, 1), // 99% buy
            (1, 99), // 99% sell
        ];

        for (buy, sell) in test_cases {
            let result = book_imbalance_fixed(Some(buy), Some(sell));
            assert!(result.is_some(), "({}, {}) returned None", buy, sell);

            let (mantissa, exp) = result.unwrap();
            assert_eq!(exp, IMBALANCE_EXP);

            // Invariant 1: abs(mantissa) <= IMBALANCE_MAX
            assert!(
                mantissa.abs() <= IMBALANCE_MAX,
                "({}, {}): |{}| > {}",
                buy,
                sell,
                mantissa,
                IMBALANCE_MAX
            );

            // Invariant 2: mantissa == 0 when buy == sell
            if buy == sell {
                assert_eq!(
                    mantissa, 0,
                    "({}, {}): expected 0, got {}",
                    buy, sell, mantissa
                );
            }
        }
    }

    // =========================================================================
    // DOCTRINE COMPLIANCE TEST
    // =========================================================================
    //
    // This test encodes the No Silent Poisoning doctrine boundary:
    //   - None MUST propagate as None (never silently become Some(0))
    //   - The helper layer MUST refuse to fabricate values
    //
    // If this test fails, the doctrine has been violated.
    //
    // See: docs/DOCTRINE_NO_SILENT_POISONING.md

    #[test]
    fn test_doctrine_none_propagation_is_mandatory() {
        // D1: Missing vendor field MUST return None, never Some(0)
        //
        // These assertions are logically redundant with earlier tests,
        // but their explicit naming encodes the doctrine contract.

        // Case 1: buy_quantity missing → cannot compute imbalance
        assert_eq!(
            book_imbalance_fixed(None, Some(100)),
            None,
            "DOCTRINE VIOLATION: None buy_quantity must return None, not Some(0)"
        );

        // Case 2: sell_quantity missing → cannot compute imbalance
        assert_eq!(
            book_imbalance_fixed(Some(100), None),
            None,
            "DOCTRINE VIOLATION: None sell_quantity must return None, not Some(0)"
        );

        // Case 3: both missing → cannot compute imbalance
        assert_eq!(
            book_imbalance_fixed(None, None),
            None,
            "DOCTRINE VIOLATION: None inputs must return None, not Some(0)"
        );

        // Case 4: require_u64 must fail on None, not return Ok(0)
        assert!(
            require_u64(None, "test").is_err(),
            "DOCTRINE VIOLATION: require_u64(None) must return Err, not Ok(0)"
        );

        // Case 5: require_i64 must fail on None, not return Ok(0)
        assert!(
            require_i64(None, "test").is_err(),
            "DOCTRINE VIOLATION: require_i64(None) must return Err, not Ok(0)"
        );

        // Sanity: Some(0) is a VALID input (vendor asserted zero depth)
        // This MUST produce Some((0, exp)), NOT None
        assert_eq!(
            book_imbalance_fixed(Some(0), Some(0)),
            Some((0, IMBALANCE_EXP)),
            "DOCTRINE VIOLATION: Some(0) is valid vendor data, must compute (not return None)"
        );
    }

    // =========================================================================
    // PHASE 18 ADMISSION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_book_imbalance_requirements() {
        let req = book_imbalance_requirements();

        assert_eq!(req.signal_id, BOOK_IMBALANCE_SIGNAL_ID);
        assert_eq!(req.required_vendor_fields.len(), 2);
        assert!(
            req.required_vendor_fields
                .contains(&VendorField::BuyQuantity)
        );
        assert!(
            req.required_vendor_fields
                .contains(&VendorField::SellQuantity)
        );
        assert!(req.required_internal_fields.is_empty());
        assert!(req.optional_vendor_fields.is_empty());
    }

    #[test]
    fn test_book_imbalance_after_admission_success() {
        // After admission confirms fields present, compute should succeed
        let (mantissa, exp) = book_imbalance_after_admission(Some(60), Some(40));
        assert_eq!(mantissa, 2000);
        assert_eq!(exp, IMBALANCE_EXP);
    }

    #[test]
    #[should_panic(expected = "INVARIANT VIOLATION")]
    fn test_book_imbalance_after_admission_invariant_violation() {
        // This should never happen in correct code: admission would have refused
        // But if it does, we panic loudly rather than silently fabricating
        let _ = book_imbalance_after_admission(None, Some(40));
    }
}
