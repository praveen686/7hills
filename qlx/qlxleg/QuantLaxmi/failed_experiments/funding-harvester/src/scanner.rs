//! Binance Futures funding rate scanner.
//!
//! Fetches `GET /fapi/v1/premiumIndex` (public, no API key) and ranks
//! all USDT perpetual contracts by edge-adjusted score:
//!   edge_score = funding_rate_bps / max(combined_spread_bps, 1.0)

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashSet;

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

/// Parsed and ranked funding opportunity.
#[derive(Debug, Clone)]
pub struct FundingOpportunity {
    pub symbol: String,
    pub mark_price: f64,
    pub index_price: f64,
    pub funding_rate: f64,
    pub funding_rate_bps: f64,
    pub annualized_pct: f64,
    pub basis_bps: f64,
    pub next_funding_time_ms: i64,
    /// Edge-adjusted score: funding_rate_bps / max(|basis_bps|, 1.0).
    /// Higher = better risk/reward. Penalizes wide-basis (illiquid) pairs.
    pub edge_score: f64,
}

/// Fetch all premium index entries from Binance Futures (public endpoint).
pub async fn fetch_premium_index() -> Result<Vec<PremiumIndexEntry>> {
    let url = "https://fapi.binance.com/fapi/v1/premiumIndex";
    let client = reqwest::Client::new();
    let entries: Vec<PremiumIndexEntry> = client.get(url).send().await?.json().await?;
    Ok(entries)
}

/// Parse raw entries into ranked funding opportunities.
///
/// Filters to USDT pairs only, sorts by edge_score descending.
pub fn rank_opportunities(entries: &[PremiumIndexEntry]) -> Vec<FundingOpportunity> {
    let mut opps: Vec<FundingOpportunity> = entries
        .iter()
        .filter(|e| e.symbol.ends_with("USDT") && e.symbol.is_ascii())
        .filter_map(|e| {
            let mark = e.mark_price.parse::<f64>().ok()?;
            let index = e.index_price.parse::<f64>().ok()?;
            let rate = e.last_funding_rate.parse::<f64>().ok()?;

            // Skip zero/invalid prices
            if mark <= 0.0 || index <= 0.0 {
                return None;
            }

            let rate_bps = rate * 10_000.0;
            // 3 settlements per day, 365 days
            let annualized = rate * 3.0 * 365.0 * 100.0;
            let basis = (mark - index) / index * 10_000.0;

            // Edge score: funding_rate / basis penalty
            // Uses |basis| as a proxy for spread (correlated with liquidity)
            let edge_score = if rate_bps > 0.0 {
                rate_bps / basis.abs().max(1.0)
            } else {
                0.0
            };

            Some(FundingOpportunity {
                symbol: e.symbol.clone(),
                mark_price: mark,
                index_price: index,
                funding_rate: rate,
                funding_rate_bps: rate_bps,
                annualized_pct: annualized,
                basis_bps: basis,
                next_funding_time_ms: e.next_funding_time,
                edge_score,
            })
        })
        .collect();

    // Sort by edge_score descending (best risk-adjusted yield first)
    opps.sort_by(|a, b| {
        b.edge_score
            .partial_cmp(&a.edge_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    opps
}

/// Format a single opportunity as a table row.
pub fn format_row(i: usize, opp: &FundingOpportunity) -> String {
    let next_ts = chrono::DateTime::from_timestamp_millis(opp.next_funding_time_ms)
        .map(|dt| dt.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "---".into());

    format!(
        "{:>3}  {:<16} {:>12.4} {:>8.2} {:>8.1}% {:>8.1} {:>6.2} {:>10}",
        i + 1,
        opp.symbol,
        opp.mark_price,
        opp.funding_rate_bps,
        opp.annualized_pct,
        opp.basis_bps,
        opp.edge_score,
        next_ts,
    )
}

/// Print the table header.
pub fn format_header() -> String {
    format!(
        "{:>3}  {:<16} {:>12} {:>8} {:>9} {:>8} {:>6} {:>10}",
        "#", "SYMBOL", "MARK", "RATE(bp)", "ANN.%", "BASIS", "SCORE", "NEXT_FUND"
    )
}

// ---------------------------------------------------------------------------
// Spot availability check
// ---------------------------------------------------------------------------

/// Raw response from Binance spot `GET /api/v3/exchangeInfo`.
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
    opps: &[FundingOpportunity],
    spot_symbols: &HashSet<String>,
) -> Vec<FundingOpportunity> {
    opps.iter()
        .filter(|o| spot_symbols.contains(&o.symbol))
        .cloned()
        .collect()
}
