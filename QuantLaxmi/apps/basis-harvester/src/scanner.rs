//! Binance Futures basis scanner.
//!
//! Reuses `GET /fapi/v1/premiumIndex` (public, no API key) and adds
//! `rank_basis_opportunities()` sorted by |basis_bps| for mean-reversion.
//!
//! **Fix 1**: Filters by 24h quote volume (liquidity) and caps |basis| to
//! exclude structurally dislocated illiquid altcoins.

use anyhow::Result;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// Raw response from Binance `GET /fapi/v1/premiumIndex`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PremiumIndexEntry {
    pub symbol: String,
    pub mark_price: String,
    pub index_price: String,
    pub estimated_settle_price: String,
    pub last_funding_rate: String,
    pub interest_rate: String,
    pub next_funding_time: i64,
    pub time: i64,
}

/// Parsed basis opportunity for ranking.
#[derive(Debug, Clone)]
pub struct BasisOpportunity {
    pub symbol: String,
    pub mark_price: f64,
    pub index_price: f64,
    pub basis_bps: f64,
    pub abs_basis_bps: f64,
    pub funding_rate_bps: f64,
    pub volume_usd_24h: f64,
}

// ---------------------------------------------------------------------------
// 24h volume fetch
// ---------------------------------------------------------------------------

/// Raw response from Binance `GET /fapi/v1/ticker/24hr`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ticker24h {
    symbol: String,
    quote_volume: String,
}

/// Fetch 24h quote volume for all futures symbols.
/// Returns symbol → 24h quote volume in USDT.
pub async fn fetch_24h_volumes() -> Result<HashMap<String, f64>> {
    let url = "https://fapi.binance.com/fapi/v1/ticker/24hr";
    let client = reqwest::Client::new();
    let tickers: Vec<Ticker24h> = client.get(url).send().await?.json().await?;
    let map: HashMap<String, f64> = tickers
        .into_iter()
        .filter_map(|t| {
            let vol = t.quote_volume.parse::<f64>().ok()?;
            Some((t.symbol, vol))
        })
        .collect();
    Ok(map)
}

// ---------------------------------------------------------------------------
// Liquidity filter parameters
// ---------------------------------------------------------------------------

/// Minimum 24h quote volume in USDT to consider a symbol liquid.
/// $2M catches volatile mid-caps while filtering out truly illiquid tokens.
const MIN_VOLUME_USD: f64 = 2_000_000.0; // $2M/day

/// Maximum |basis| in bps — exclude extreme outliers (delistings, broken feeds).
const MAX_ABS_BASIS_BPS: f64 = 200.0;

/// Maximum |funding rate| in bps — pairs with extreme funding are structurally
/// dislocated (e.g. SYNUSDT at -68 bps). Their basis doesn't mean-revert.
const MAX_ABS_FUNDING_BPS: f64 = 25.0;

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/// Fetch all premium index entries from Binance Futures (public endpoint).
pub async fn fetch_premium_index() -> Result<Vec<PremiumIndexEntry>> {
    let url = "https://fapi.binance.com/fapi/v1/premiumIndex";
    let client = reqwest::Client::new();
    let entries: Vec<PremiumIndexEntry> = client.get(url).send().await?.json().await?;
    Ok(entries)
}

/// Parse raw entries into basis opportunities, filtered by liquidity.
///
/// 1. Require 24h quote volume >= $2M (liquid enough to fill)
/// 2. Require |basis| <= 200 bps (exclude only extreme outliers)
/// 3. Sort by |basis| descending within the liquid set
pub fn rank_basis_opportunities(
    entries: &[PremiumIndexEntry],
    volumes: &HashMap<String, f64>,
) -> Vec<BasisOpportunity> {
    let mut opps: Vec<BasisOpportunity> = entries
        .iter()
        .filter(|e| e.symbol.ends_with("USDT") && e.symbol.is_ascii())
        .filter_map(|e| {
            let mark = e.mark_price.parse::<f64>().ok()?;
            let index = e.index_price.parse::<f64>().ok()?;
            let rate = e.last_funding_rate.parse::<f64>().ok()?;

            if mark <= 0.0 || index <= 0.0 {
                return None;
            }

            let volume = volumes.get(&e.symbol).copied().unwrap_or(0.0);

            // Liquidity gate: skip low-volume symbols
            if volume < MIN_VOLUME_USD {
                return None;
            }

            let basis = (mark - index) / index * 10_000.0;

            // Structural dislocation gate: skip extreme basis
            if basis.abs() > MAX_ABS_BASIS_BPS {
                return None;
            }

            let funding_bps = rate * 10_000.0;

            // Funding rate gate: extreme funding = structural dislocation
            if funding_bps.abs() > MAX_ABS_FUNDING_BPS {
                return None;
            }

            Some(BasisOpportunity {
                symbol: e.symbol.clone(),
                mark_price: mark,
                index_price: index,
                basis_bps: basis,
                abs_basis_bps: basis.abs(),
                funding_rate_bps: funding_bps,
                volume_usd_24h: volume,
            })
        })
        .collect();

    // Sort by |basis| descending (most volatile basis among liquid symbols)
    opps.sort_by(|a, b| {
        b.abs_basis_bps
            .partial_cmp(&a.abs_basis_bps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    opps
}

/// Format a single opportunity as a table row.
pub fn format_row(i: usize, opp: &BasisOpportunity) -> String {
    let vol_m = opp.volume_usd_24h / 1_000_000.0;
    format!(
        "{:>3}  {:<14} {:>10.4} {:>7.1} {:>7.1} {:>7.2} {:>7.1}M",
        i + 1,
        opp.symbol,
        opp.mark_price,
        opp.basis_bps,
        opp.abs_basis_bps,
        opp.funding_rate_bps,
        vol_m,
    )
}

/// Print the table header.
pub fn format_header() -> String {
    format!(
        "{:>3}  {:<14} {:>10} {:>7} {:>7} {:>7} {:>8}",
        "#", "SYMBOL", "MARK", "BASIS", "|BASIS|", "FUND", "VOL24h"
    )
}

// ---------------------------------------------------------------------------
// Spot availability check
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SpotSymbol>,
}

#[derive(Debug, Deserialize)]
struct SpotSymbol {
    symbol: String,
    status: String,
}

/// Fetch the set of actively trading USDT spot pairs on Binance.
pub async fn fetch_spot_symbols() -> Result<HashSet<String>> {
    let url = "https://api.binance.com/api/v3/exchangeInfo?permissions=SPOT";
    let client = reqwest::Client::new();
    let info: ExchangeInfo = client.get(url).send().await?.json().await?;
    let set: HashSet<String> = info
        .symbols
        .into_iter()
        .filter(|s| s.status == "TRADING" && s.symbol.ends_with("USDT"))
        .map(|s| s.symbol)
        .collect();
    Ok(set)
}

/// Filter opportunities to only those with active Binance spot pairs.
pub fn filter_spot_available(
    opps: &[BasisOpportunity],
    spot_symbols: &HashSet<String>,
) -> Vec<BasisOpportunity> {
    opps.iter()
        .filter(|o| spot_symbols.contains(&o.symbol))
        .cloned()
        .collect()
}
