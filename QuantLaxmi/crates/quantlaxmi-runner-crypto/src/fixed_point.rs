//! Deterministic fixed-point parsing utilities.
//!
//! Binance streams provide decimal strings. We must convert to integer mantissas
//! without floating point intermediates to preserve cross-platform determinism.

use anyhow::{Context, Result, bail};

/// Pure string-to-mantissa parser (NO float conversion).
///
/// Parses decimal strings like "90000.12" directly to mantissa without
/// intermediate f64 conversion, avoiding cross-platform float drift.
///
/// # Examples
/// - "90000.12" with exponent -2 → 9000012
/// - "1.50000000" with exponent -8 → 150000000
pub fn parse_to_mantissa_pure(s: &str, exponent: i8) -> Result<i64> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty string");
    }

    // Handle negative numbers
    let (is_negative, s) = if let Some(stripped) = s.strip_prefix('-') {
        (true, stripped)
    } else {
        (false, s)
    };

    // Split on decimal point
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() > 2 {
        bail!("invalid decimal format: {}", s);
    }

    let int_part = parts[0];
    let frac_part = if parts.len() == 2 { parts[1] } else { "" };

    // Target decimal places = -exponent (e.g., exponent=-2 means 2 decimals)
    let target_decimals = (-exponent) as usize;

    // Build mantissa string: integer part + fractional part padded/truncated
    let mut mantissa_str = String::with_capacity(int_part.len() + target_decimals);
    mantissa_str.push_str(int_part);

    if frac_part.len() >= target_decimals {
        mantissa_str.push_str(&frac_part[..target_decimals]);

        // Round using next digit if any
        if frac_part.len() > target_decimals {
            let next_digit = frac_part.chars().nth(target_decimals).unwrap_or('0');
            if next_digit >= '5' {
                let mut val: i64 = mantissa_str
                    .parse()
                    .with_context(|| format!("parse mantissa: {}", mantissa_str))?;
                val += 1;
                return Ok(if is_negative { -val } else { val });
            }
        }
    } else {
        mantissa_str.push_str(frac_part);
        for _ in 0..(target_decimals - frac_part.len()) {
            mantissa_str.push('0');
        }
    }

    let val: i64 = mantissa_str
        .parse()
        .with_context(|| format!("parse mantissa: {}", mantissa_str))?;
    Ok(if is_negative { -val } else { val })
}
