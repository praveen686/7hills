//! Binance Spot exchangeInfo -> SpecStore utilities.
//!
//! Purpose: populate tick size (price) and step size (quantity) per symbol
//! so KiteSim validation is real (not placeholder).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

#[derive(Debug, Deserialize)]
struct SymbolInfo {
    symbol: String,
    filters: Vec<Filter>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "filterType")]
enum Filter {
    #[serde(rename = "PRICE_FILTER")]
    Price {
        #[serde(rename = "tickSize")]
        tick_size: String,
    },
    #[serde(rename = "LOT_SIZE")]
    Lot {
        #[serde(rename = "stepSize")]
        step_size: String,
    },
    #[serde(other)]
    Other,
}

/// Count decimal places in a Binance step/tick string like "0.00100000".
fn decimals_from_str(s: &str) -> u32 {
    if let Some(dot) = s.find('.') {
        let frac = &s[dot + 1..];
        let trimmed = frac.trim_end_matches('0');
        trimmed.len() as u32
    } else {
        0
    }
}

/// Compute price exponent from tick_size.
/// E.g., tick_size 0.00001 → exponent -5
pub fn exponent_from_tick_size(tick_size: f64) -> i8 {
    if tick_size >= 1.0 {
        0
    } else {
        let decimals = (-tick_size.log10()).ceil() as i8;
        -decimals
    }
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("parse f64: {}", s))
}

/// Compute qty_scale from LOT_SIZE stepSize.
/// For stepSize 0.001 -> scale 1000 (internal units are 1/1000 base units).
fn qty_scale_from_step(step: &str) -> u32 {
    let d = decimals_from_str(step);
    10u32.saturating_pow(d)
}

/// Fetch Binance Spot exchangeInfo for the provided symbols and return:
/// symbol -> (tick_size, qty_scale)
pub fn fetch_spot_specs(symbols: &HashSet<String>) -> Result<HashMap<String, (f64, u32)>> {
    if symbols.is_empty() {
        return Ok(HashMap::new());
    }

    // Binance expects symbols=["BTCUSDT","ETHUSDT"] as a query param.
    let sym_list: Vec<String> = symbols.iter().cloned().collect();
    let sym_json = serde_json::to_string(&sym_list)?;
    let sym_encoded = urlencoding::encode(&sym_json);

    let url = format!(
        "https://api.binance.com/api/v3/exchangeInfo?symbols={}",
        sym_encoded
    );

    let resp = reqwest::blocking::get(&url)
        .with_context(|| format!("GET {}", url))?
        .error_for_status()
        .with_context(|| "non-200 from Binance exchangeInfo")?;

    let info: ExchangeInfo = resp.json().with_context(|| "parse exchangeInfo JSON")?;

    let mut out: HashMap<String, (f64, u32)> = HashMap::new();

    for s in info.symbols {
        let mut tick: Option<f64> = None;
        let mut step: Option<String> = None;

        for f in s.filters {
            match f {
                Filter::Price { tick_size } => {
                    tick = Some(parse_f64(&tick_size)?);
                }
                Filter::Lot { step_size } => {
                    step = Some(step_size);
                }
                _ => {}
            }
        }

        let tick_size = tick.unwrap_or(0.01);
        let qty_scale = step.map(|st| qty_scale_from_step(&st)).unwrap_or(1);

        out.insert(s.symbol, (tick_size, qty_scale));
    }

    Ok(out)
}

/// Fetch price exponents for symbols from Binance exchangeInfo.
/// Returns symbol -> price_exponent mapping.
/// This ensures correct precision for each symbol (e.g., SOLBTC needs -7, BTCUSDT needs -2).
pub fn fetch_price_exponents(symbols: &[String]) -> Result<HashMap<String, i8>> {
    let sym_set: HashSet<String> = symbols.iter().cloned().collect();
    let specs = fetch_spot_specs(&sym_set)?;

    let mut exponents: HashMap<String, i8> = HashMap::new();
    for (symbol, (tick_size, _)) in specs {
        let exp = exponent_from_tick_size(tick_size);
        tracing::info!("  {} tick_size={} → price_exp={}", symbol, tick_size, exp);
        exponents.insert(symbol, exp);
    }

    Ok(exponents)
}
