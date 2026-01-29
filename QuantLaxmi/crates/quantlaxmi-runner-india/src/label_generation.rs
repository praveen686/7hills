//! B1.2 Label Generation: Join routing decisions with fills to create training data.
//!
//! This module implements the deterministic join algorithm that produces labels.jsonl,
//! the supervised learning dataset for execution prediction.
//!
//! ## Schema: quantlaxmi.labels.execution.v1
//! Each row = one intent_id (routing decision → zero or more fills)
//!
//! ## Join Algorithm
//! 1. Index routing decisions by intent_id
//! 2. Aggregate fills by intent_id (VWAP for partials)
//! 3. Compute execution_tax_bps vs mid_at_decision
//! 4. Compute horizon labels (30s, 2m, 5m forward)
//! 5. Emit labels.jsonl

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Schema version for labels
const LABELS_SCHEMA: &str = "quantlaxmi.labels.execution.v1";
const LABELS_SCHEMA_REV: u32 = 1;

// ============================================================================
// Input Schemas (from routing_decisions.jsonl and fills.jsonl)
// ============================================================================

/// Routing decision record (subset of fields we need)
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct RoutingDecision {
    record_type: String,
    ts_utc: String,
    symbol: String,
    quote: QuoteSnapshot,
    features: Features,
    signal: Signal,
    decision: Decision,
    ids: Ids,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct QuoteSnapshot {
    bid: f64,
    ask: f64,
    mid: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Features {
    spread_bps: f64,
    pressure: f64,
    vel_bps_sec: f64,
    vel_abs_bps_sec: f64,
    vel_used: bool,
    dt_ms: i64,
    signal_strength: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct Signal {
    side: String,
    direction: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct Decision {
    order_type: String,
    price: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct Ids {
    intent_id: String,
}

/// Fill record from fills.jsonl
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct FillRecord {
    intent_id: Option<String>,
    order_id: Option<String>,
    tradingsymbol: String,
    side: String,
    order_type: String,
    status: String,
    fill_price: Option<f64>,
    filled_qty: u32,
    requested_qty: u32,
}

/// Quote event for horizon labels (mantissa-based format)
#[derive(Debug, Clone, Deserialize)]
struct QuoteEvent {
    ts: DateTime<Utc>,
    tradingsymbol: String,
    bid: i64, // Mantissa
    ask: i64, // Mantissa
    #[serde(default)]
    price_exponent: i32, // Default -2 for NSE
}

impl QuoteEvent {
    fn bid_f64(&self) -> f64 {
        self.bid as f64 * 10_f64.powi(self.price_exponent)
    }

    fn ask_f64(&self) -> f64 {
        self.ask as f64 * 10_f64.powi(self.price_exponent)
    }

    fn mid_f64(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }
}

// ============================================================================
// Output Schema (labels.jsonl)
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct LabelRecord {
    schema: &'static str,
    schema_rev: u32,

    intent_id: String,
    /// Decision timestamp (for time-safe train/test split)
    ts_utc: String,

    filled: bool,
    partial_fill: bool,
    fill_count: u32,

    side: String,
    order_type: String,

    requested_qty: u32,
    filled_qty: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    fill_price: Option<f64>,
    mid_at_decision: f64,

    /// Fill value in INR = fill_price × filled_qty (for PnL computation)
    #[serde(skip_serializing_if = "Option::is_none")]
    fill_value_inr: Option<f64>,

    /// Negative = cost, Positive = favorable
    execution_tax_bps: f64,

    /// Gross execution edge in INR (positive = favorable fill)
    /// = execution_tax_bps / 10000 × fill_value_inr
    #[serde(skip_serializing_if = "Option::is_none")]
    gross_edge_inr: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    time_to_fill_ms: Option<i64>,

    queue_blocked: bool,

    horizons: HorizonLabels,

    // Include features for convenience (optional, can be joined from routing_decisions)
    features: Features,
}

#[derive(Debug, Clone, Serialize)]
struct HorizonLabels {
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_30s_bps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_2m_bps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_5m_bps: Option<f64>,
}

// ============================================================================
// Aggregated Fill Data
// ============================================================================

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct AggregatedFill {
    total_filled_qty: u32,
    total_requested_qty: u32,
    fill_count: u32,
    // For VWAP: sum(price * qty)
    vwap_numerator: f64,
    first_fill_ts: Option<DateTime<Utc>>,
}

impl AggregatedFill {
    fn add_fill(&mut self, fill: &FillRecord, _fill_ts: Option<DateTime<Utc>>) {
        self.fill_count += 1;
        self.total_filled_qty += fill.filled_qty;
        self.total_requested_qty = fill.requested_qty; // Same for all fills of same intent

        if let Some(px) = fill.fill_price {
            self.vwap_numerator += px * fill.filled_qty as f64;
        }
    }

    fn vwap(&self) -> Option<f64> {
        if self.total_filled_qty > 0 {
            Some(self.vwap_numerator / self.total_filled_qty as f64)
        } else {
            None
        }
    }

    fn filled(&self) -> bool {
        self.total_filled_qty > 0
    }

    fn partial_fill(&self) -> bool {
        self.total_filled_qty > 0 && self.total_filled_qty < self.total_requested_qty
    }
}

// ============================================================================
// Core Join Logic
// ============================================================================

/// Main entry point for label generation
pub async fn run_generate_labels(
    routing_decisions_path: &str,
    fills_path: &str,
    quotes_path: &str,
    out_path: &str,
) -> Result<()> {
    tracing::info!("B1.2 Label Generation starting...");
    tracing::info!("  routing_decisions: {}", routing_decisions_path);
    tracing::info!("  fills: {}", fills_path);
    tracing::info!("  quotes: {}", quotes_path);
    tracing::info!("  out: {}", out_path);

    // Step A: Index routing decisions by intent_id
    tracing::info!("Step A: Indexing routing decisions...");
    let decisions = load_routing_decisions(routing_decisions_path)?;
    tracing::info!("  Loaded {} routing decisions", decisions.len());

    // Step B: Aggregate fills by intent_id
    tracing::info!("Step B: Aggregating fills...");
    let fills = load_and_aggregate_fills(fills_path)?;
    tracing::info!("  Aggregated fills for {} intent_ids", fills.len());

    // Step C: Load quotes for horizon labels
    tracing::info!("Step C: Loading quotes for horizon labels...");
    let quotes = load_quotes_by_symbol(quotes_path)?;
    let total_quotes: usize = quotes.values().map(|v| v.len()).sum();
    tracing::info!(
        "  Loaded {} quotes across {} symbols",
        total_quotes,
        quotes.len()
    );

    // Step D: Generate labels
    tracing::info!("Step D: Generating labels...");
    let labels = generate_labels(&decisions, &fills, &quotes)?;
    tracing::info!("  Generated {} labels", labels.len());

    // Step E: Write labels.jsonl
    tracing::info!("Step E: Writing labels.jsonl...");
    write_labels(out_path, &labels)?;
    tracing::info!("Labels written to: {}", out_path);

    // Summary stats
    let filled_count = labels.iter().filter(|l| l.filled).count();
    let unfilled_count = labels.len() - filled_count;
    let partial_count = labels.iter().filter(|l| l.partial_fill).count();
    let queue_blocked_count = labels.iter().filter(|l| l.queue_blocked).count();

    tracing::info!("\n=== B1.2 Label Generation Complete ===");
    tracing::info!("Total labels: {}", labels.len());
    tracing::info!(
        "Filled: {} ({:.1}%)",
        filled_count,
        filled_count as f64 / labels.len() as f64 * 100.0
    );
    tracing::info!(
        "Unfilled: {} ({:.1}%)",
        unfilled_count,
        unfilled_count as f64 / labels.len() as f64 * 100.0
    );
    tracing::info!("Partial fills: {}", partial_count);
    tracing::info!("Queue blocked (unfilled LIMIT): {}", queue_blocked_count);

    // Execution tax summary for filled
    let filled_labels: Vec<_> = labels.iter().filter(|l| l.filled).collect();
    if !filled_labels.is_empty() {
        let mean_tax: f64 = filled_labels
            .iter()
            .map(|l| l.execution_tax_bps)
            .sum::<f64>()
            / filled_labels.len() as f64;
        tracing::info!("Mean execution_tax_bps (filled): {:.2}", mean_tax);
    }

    Ok(())
}

/// Load routing decisions into a HashMap indexed by intent_id
fn load_routing_decisions(path: &str) -> Result<HashMap<String, RoutingDecision>> {
    let file = File::open(path).with_context(|| format!("open routing_decisions: {}", path))?;
    let reader = BufReader::new(file);
    let mut decisions = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Parse as generic JSON first to check record_type
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let record_type = v.get("record_type").and_then(|r| r.as_str()).unwrap_or("");

        if record_type == "decision" {
            let decision: RoutingDecision = serde_json::from_str(&line)?;
            decisions.insert(decision.ids.intent_id.clone(), decision);
        }
        // Skip run_header and run_footer
    }

    Ok(decisions)
}

/// Load fills and aggregate by intent_id
fn load_and_aggregate_fills(path: &str) -> Result<HashMap<String, AggregatedFill>> {
    let file = File::open(path).with_context(|| format!("open fills: {}", path))?;
    let reader = BufReader::new(file);
    let mut fills: HashMap<String, AggregatedFill> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fill: FillRecord = serde_json::from_str(&line)?;

        if let Some(intent_id) = &fill.intent_id {
            let agg = fills.entry(intent_id.clone()).or_default();
            agg.add_fill(&fill, None); // TODO: parse fill timestamp if available
        }
    }

    Ok(fills)
}

/// Load quotes sorted by timestamp for each symbol
fn load_quotes_by_symbol(path: &str) -> Result<HashMap<String, Vec<QuoteEvent>>> {
    let file = File::open(path).with_context(|| format!("open quotes: {}", path))?;
    let reader = BufReader::new(file);
    let mut quotes: HashMap<String, Vec<QuoteEvent>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as QuoteEvent
        if let Ok(quote) = serde_json::from_str::<QuoteEvent>(&line) {
            quotes
                .entry(quote.tradingsymbol.clone())
                .or_default()
                .push(quote);
        }
    }

    // Sort each symbol's quotes by timestamp
    for quotes_vec in quotes.values_mut() {
        quotes_vec.sort_by_key(|q| q.ts);
    }

    Ok(quotes)
}

/// Generate labels by joining decisions with fills
fn generate_labels(
    decisions: &HashMap<String, RoutingDecision>,
    fills: &HashMap<String, AggregatedFill>,
    quotes: &HashMap<String, Vec<QuoteEvent>>,
) -> Result<Vec<LabelRecord>> {
    let mut labels = Vec::with_capacity(decisions.len());

    for (intent_id, decision) in decisions {
        let agg = fills.get(intent_id);
        let filled = agg.map(|a| a.filled()).unwrap_or(false);
        let partial_fill = agg.map(|a| a.partial_fill()).unwrap_or(false);
        let fill_count = agg.map(|a| a.fill_count).unwrap_or(0);
        let filled_qty = agg.map(|a| a.total_filled_qty).unwrap_or(0);
        let requested_qty = agg.map(|a| a.total_requested_qty).unwrap_or(0);
        let fill_price = agg.and_then(|a| a.vwap());

        // Parse decision timestamp
        let decision_ts = parse_rfc3339(&decision.ts_utc)?;
        let mid_at_decision = decision.quote.mid;

        // Compute execution_tax_bps and fill_value
        let (execution_tax_bps, fill_value_inr, gross_edge_inr) = if let Some(fp) = fill_price {
            let tax = compute_execution_tax_bps(&decision.signal.side, fp, mid_at_decision);
            let fill_value = fp * filled_qty as f64;
            let gross_edge = (tax / 10_000.0) * fill_value;
            (tax, Some(fill_value), Some(gross_edge))
        } else {
            (0.0, None, None) // Unfilled = no execution cost
        };

        // Time to fill (if we had fill timestamps, we'd compute this)
        let time_to_fill_ms: Option<i64> = None; // TODO: need fill timestamps

        // Queue blocked = unfilled LIMIT order
        let queue_blocked = !filled && decision.decision.order_type == "Limit";

        // Compute horizon labels
        let horizons = compute_horizons(
            &decision.signal.side,
            mid_at_decision,
            decision_ts,
            &decision.symbol,
            quotes,
        );

        labels.push(LabelRecord {
            schema: LABELS_SCHEMA,
            schema_rev: LABELS_SCHEMA_REV,
            intent_id: intent_id.clone(),
            ts_utc: decision.ts_utc.clone(),
            filled,
            partial_fill,
            fill_count,
            side: decision.signal.side.clone(),
            order_type: decision.decision.order_type.clone(),
            requested_qty,
            filled_qty,
            fill_price,
            mid_at_decision,
            fill_value_inr,
            execution_tax_bps,
            gross_edge_inr,
            time_to_fill_ms,
            queue_blocked,
            horizons,
            features: decision.features.clone(),
        });
    }

    // Sort by timestamp for time-safe train/test split
    labels.sort_by(|a, b| a.ts_utc.cmp(&b.ts_utc));

    Ok(labels)
}

/// Compute execution tax in bps
/// Negative = cost (filled worse than mid)
/// Positive = favorable (filled better than mid)
fn compute_execution_tax_bps(side: &str, fill_price: f64, mid_at_decision: f64) -> f64 {
    if mid_at_decision <= 0.0 {
        return 0.0;
    }

    match side {
        "Buy" => {
            // BUY: tax = (mid - fill_price) / mid * 10000
            // If fill_price > mid, we paid more = negative tax (cost)
            (mid_at_decision - fill_price) / mid_at_decision * 10_000.0
        }
        "Sell" => {
            // SELL: tax = (fill_price - mid) / mid * 10000
            // If fill_price < mid, we received less = negative tax (cost)
            (fill_price - mid_at_decision) / mid_at_decision * 10_000.0
        }
        _ => 0.0,
    }
}

/// Compute horizon labels (30s, 2m, 5m forward edge)
fn compute_horizons(
    side: &str,
    mid_at_decision: f64,
    decision_ts: DateTime<Utc>,
    symbol: &str,
    quotes: &HashMap<String, Vec<QuoteEvent>>,
) -> HorizonLabels {
    let symbol_quotes = match quotes.get(symbol) {
        Some(q) => q,
        None => {
            return HorizonLabels {
                edge_30s_bps: None,
                edge_2m_bps: None,
                edge_5m_bps: None,
            };
        }
    };

    let edge_30s_bps = compute_horizon_edge(
        side,
        mid_at_decision,
        decision_ts,
        symbol_quotes,
        Duration::seconds(30),
    );
    let edge_2m_bps = compute_horizon_edge(
        side,
        mid_at_decision,
        decision_ts,
        symbol_quotes,
        Duration::minutes(2),
    );
    let edge_5m_bps = compute_horizon_edge(
        side,
        mid_at_decision,
        decision_ts,
        symbol_quotes,
        Duration::minutes(5),
    );

    HorizonLabels {
        edge_30s_bps,
        edge_2m_bps,
        edge_5m_bps,
    }
}

/// Compute edge at a specific horizon
fn compute_horizon_edge(
    side: &str,
    mid_at_decision: f64,
    decision_ts: DateTime<Utc>,
    quotes: &[QuoteEvent],
    horizon: Duration,
) -> Option<f64> {
    let target_ts = decision_ts + horizon;

    // Binary search for first quote >= target_ts
    let idx = quotes.partition_point(|q| q.ts < target_ts);

    if idx >= quotes.len() {
        return None; // No quote at or after horizon
    }

    let horizon_quote = &quotes[idx];
    let mid_at_horizon = horizon_quote.mid_f64();

    if mid_at_decision <= 0.0 || mid_at_horizon <= 0.0 {
        return None;
    }

    // Edge = directional profit in bps
    let edge_bps = match side {
        "Buy" => {
            // BUY: profit if price went up
            (mid_at_horizon - mid_at_decision) / mid_at_decision * 10_000.0
        }
        "Sell" => {
            // SELL: profit if price went down
            (mid_at_decision - mid_at_horizon) / mid_at_decision * 10_000.0
        }
        _ => return None,
    };

    Some(edge_bps)
}

/// Write labels to JSONL file
fn write_labels(path: &str, labels: &[LabelRecord]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create labels file: {}", path))?;
    let mut writer = BufWriter::new(file);

    for label in labels {
        writeln!(writer, "{}", serde_json::to_string(label)?)?;
    }

    writer.flush()?;
    Ok(())
}

/// Parse RFC3339 timestamp
fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .with_context(|| format!("parse timestamp: {}", s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_tax_buy() {
        // BUY at 100, mid was 99 -> paid 1% more = -100 bps
        assert!((compute_execution_tax_bps("Buy", 100.0, 99.0) - (-101.01)).abs() < 0.1);

        // BUY at 99, mid was 100 -> paid 1% less = +100 bps (favorable)
        assert!((compute_execution_tax_bps("Buy", 99.0, 100.0) - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_execution_tax_sell() {
        // SELL at 100, mid was 101 -> received ~1% less = -99 bps
        assert!((compute_execution_tax_bps("Sell", 100.0, 101.0) - (-99.01)).abs() < 0.1);

        // SELL at 101, mid was 100 -> received 1% more = +100 bps (favorable)
        assert!((compute_execution_tax_bps("Sell", 101.0, 100.0) - 100.0).abs() < 0.1);
    }
}
