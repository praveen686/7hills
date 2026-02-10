//! Intent-Level PnL Generation
//!
//! Generates intent_pnl.jsonl by joining labels.jsonl with fee_ledger.jsonl.
//! This enables computation of win rate, profit factor, and Sharpe ratio.
//!
//! ## Schema: quantlaxmi.pnl.intent.v1
//!
//! ## Trade Definition
//! Trade unit = "intent" (one routing decision = one trade)
//! - gross_pnl_inr: execution edge vs mid_at_decision
//! - fees_inr: brokerage + STT + txn + stamp + SEBI + GST
//! - net_pnl_inr: gross - fees

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Schema constants
const INTENT_PNL_SCHEMA: &str = "quantlaxmi.pnl.intent.v1";
const INTENT_PNL_SCHEMA_REV: u32 = 1;

// ============================================================================
// Input Schemas
// ============================================================================

/// Label record (subset of fields we need)
#[derive(Debug, Clone, Deserialize)]
struct LabelInput {
    intent_id: String,
    ts_utc: String,
    filled: bool,
    side: String,
    order_type: String,
    filled_qty: u32,
    #[serde(default)]
    fill_value_inr: Option<f64>,
    #[serde(default)]
    gross_edge_inr: Option<f64>,
    execution_tax_bps: f64,
}

/// Fee ledger record (subset of fields we need)
#[derive(Debug, Clone, Deserialize)]
struct FeeInput {
    intent_id: Option<String>,
    #[serde(rename = "total_inr")]
    fees_total_inr: f64,
}

// ============================================================================
// Output Schema (intent_pnl.jsonl)
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct IntentPnlRecord {
    pub schema: &'static str,
    pub schema_rev: u32,

    pub intent_id: String,
    pub ts_utc: String,
    pub filled: bool,
    pub side: String,
    pub order_type: String,
    pub filled_qty: u32,

    /// Fill value in INR
    pub fill_value_inr: f64,

    /// Gross execution edge in INR (before fees)
    pub gross_pnl_inr: f64,

    /// Total fees in INR
    pub fees_inr: f64,

    /// Net PnL in INR (gross - fees)
    pub net_pnl_inr: f64,

    /// Execution tax in bps
    pub execution_tax_bps: f64,

    /// Is this trade a winner? (net_pnl > 0)
    pub is_winner: bool,
}

// ============================================================================
// Performance Metrics
// ============================================================================

/// Summary metrics computed from intent PnL stream
#[derive(Debug, Clone, Serialize, Default)]
pub struct IntentPnlSummary {
    /// Total intents processed
    pub total_intents: u32,

    /// Filled intents only
    pub filled_intents: u32,

    /// Number of winning trades (net_pnl > 0)
    pub winners: u32,

    /// Number of losing trades (net_pnl <= 0)
    pub losers: u32,

    /// Win rate as percentage
    pub win_rate_pct: f64,

    /// Sum of positive PnL
    pub gross_profit_inr: f64,

    /// Sum of negative PnL (absolute value)
    pub gross_loss_inr: f64,

    /// Profit factor = gross_profit / gross_loss
    pub profit_factor: f64,

    /// Total gross PnL (before fees)
    pub total_gross_pnl_inr: f64,

    /// Total fees
    pub total_fees_inr: f64,

    /// Total net PnL (after fees)
    pub total_net_pnl_inr: f64,

    /// Mean net PnL per trade
    pub mean_net_pnl_inr: f64,

    /// Std dev of net PnL
    pub std_net_pnl_inr: f64,

    /// Intent-level Sharpe (mean / std)
    /// Note: Not annualized, use for relative comparison only
    pub sharpe_intent: f64,

    /// Average winner PnL
    pub avg_winner_inr: f64,

    /// Average loser PnL
    pub avg_loser_inr: f64,

    /// Expectancy = (win_rate × avg_win) + ((1 - win_rate) × avg_loss)
    pub expectancy_inr: f64,
}

impl IntentPnlSummary {
    pub fn compute(records: &[IntentPnlRecord]) -> Self {
        let filled: Vec<_> = records.iter().filter(|r| r.filled).collect();

        if filled.is_empty() {
            return Self::default();
        }

        let total_intents = records.len() as u32;
        let filled_intents = filled.len() as u32;

        let winners: Vec<_> = filled.iter().filter(|r| r.net_pnl_inr > 0.0).collect();
        let losers: Vec<_> = filled.iter().filter(|r| r.net_pnl_inr <= 0.0).collect();

        let n_winners = winners.len() as u32;
        let n_losers = losers.len() as u32;

        let win_rate_pct = if filled_intents > 0 {
            (n_winners as f64 / filled_intents as f64) * 100.0
        } else {
            0.0
        };

        let gross_profit_inr: f64 = winners.iter().map(|r| r.net_pnl_inr).sum();
        let gross_loss_inr: f64 = losers.iter().map(|r| r.net_pnl_inr.abs()).sum();

        let profit_factor = if gross_loss_inr > 0.0 {
            gross_profit_inr / gross_loss_inr
        } else if gross_profit_inr > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let total_gross_pnl_inr: f64 = filled.iter().map(|r| r.gross_pnl_inr).sum();
        let total_fees_inr: f64 = filled.iter().map(|r| r.fees_inr).sum();
        let total_net_pnl_inr: f64 = filled.iter().map(|r| r.net_pnl_inr).sum();

        let mean_net_pnl_inr = total_net_pnl_inr / filled_intents as f64;

        // Compute std dev
        let variance: f64 = filled
            .iter()
            .map(|r| (r.net_pnl_inr - mean_net_pnl_inr).powi(2))
            .sum::<f64>()
            / filled_intents as f64;
        let std_net_pnl_inr = variance.sqrt();

        let sharpe_intent = if std_net_pnl_inr > 0.0 {
            mean_net_pnl_inr / std_net_pnl_inr
        } else {
            0.0
        };

        let avg_winner_inr = if !winners.is_empty() {
            gross_profit_inr / winners.len() as f64
        } else {
            0.0
        };

        let avg_loser_inr = if !losers.is_empty() {
            -gross_loss_inr / losers.len() as f64
        } else {
            0.0
        };

        // Expectancy = P(win) × avg_win + P(loss) × avg_loss
        let win_prob = n_winners as f64 / filled_intents as f64;
        let loss_prob = n_losers as f64 / filled_intents as f64;
        let expectancy_inr = (win_prob * avg_winner_inr) + (loss_prob * avg_loser_inr);

        Self {
            total_intents,
            filled_intents,
            winners: n_winners,
            losers: n_losers,
            win_rate_pct,
            gross_profit_inr,
            gross_loss_inr,
            profit_factor,
            total_gross_pnl_inr,
            total_fees_inr,
            total_net_pnl_inr,
            mean_net_pnl_inr,
            std_net_pnl_inr,
            sharpe_intent,
            avg_winner_inr,
            avg_loser_inr,
            expectancy_inr,
        }
    }
}

// ============================================================================
// Core Logic
// ============================================================================

/// Generate intent_pnl.jsonl by joining labels + fee_ledger
pub fn run_generate_intent_pnl(
    labels_path: &str,
    fee_ledger_path: &str,
    out_path: &str,
) -> Result<IntentPnlSummary> {
    tracing::info!("Intent PnL Generation starting...");
    tracing::info!("  labels: {}", labels_path);
    tracing::info!("  fee_ledger: {}", fee_ledger_path);
    tracing::info!("  out: {}", out_path);

    // Step 1: Load labels
    tracing::info!("Step 1: Loading labels...");
    let labels = load_labels(labels_path)?;
    tracing::info!("  Loaded {} labels", labels.len());

    // Step 2: Aggregate fees by intent_id
    tracing::info!("Step 2: Loading and aggregating fees...");
    let fees = load_and_aggregate_fees(fee_ledger_path)?;
    tracing::info!("  Aggregated fees for {} intents", fees.len());

    // Step 3: Join and generate intent PnL records
    tracing::info!("Step 3: Joining labels with fees...");
    let records = generate_intent_pnl_records(&labels, &fees);
    tracing::info!("  Generated {} intent PnL records", records.len());

    // Step 4: Compute summary metrics
    tracing::info!("Step 4: Computing summary metrics...");
    let summary = IntentPnlSummary::compute(&records);

    // Step 5: Write intent_pnl.jsonl
    tracing::info!("Step 5: Writing intent_pnl.jsonl...");
    write_intent_pnl(out_path, &records)?;
    tracing::info!("Intent PnL written to: {}", out_path);

    // Print summary
    tracing::info!("\n=== Intent PnL Summary ===");
    tracing::info!("Total intents: {}", summary.total_intents);
    tracing::info!("Filled intents: {}", summary.filled_intents);
    tracing::info!(
        "Win rate: {:.1}% ({} winners / {} filled)",
        summary.win_rate_pct,
        summary.winners,
        summary.filled_intents
    );
    tracing::info!("Profit factor: {:.2}", summary.profit_factor);
    tracing::info!("Sharpe (intent): {:.3}", summary.sharpe_intent);
    tracing::info!("Total gross PnL: ₹{:.2}", summary.total_gross_pnl_inr);
    tracing::info!("Total fees: ₹{:.2}", summary.total_fees_inr);
    tracing::info!("Total net PnL: ₹{:.2}", summary.total_net_pnl_inr);
    tracing::info!("Mean net PnL: ₹{:.2}", summary.mean_net_pnl_inr);
    tracing::info!("Avg winner: ₹{:.2}", summary.avg_winner_inr);
    tracing::info!("Avg loser: ₹{:.2}", summary.avg_loser_inr);
    tracing::info!("Expectancy: ₹{:.2}/trade", summary.expectancy_inr);

    Ok(summary)
}

/// Load labels from labels.jsonl
fn load_labels(path: &str) -> Result<Vec<LabelInput>> {
    let file = File::open(path).with_context(|| format!("open labels: {}", path))?;
    let reader = BufReader::new(file);
    let mut labels = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let label: LabelInput = serde_json::from_str(&line)
            .with_context(|| format!("parse label on line {}", i + 1))?;
        labels.push(label);
    }

    Ok(labels)
}

/// Load fee ledger and aggregate by intent_id
fn load_and_aggregate_fees(path: &str) -> Result<HashMap<String, f64>> {
    let file = File::open(path).with_context(|| format!("open fee_ledger: {}", path))?;
    let reader = BufReader::new(file);
    let mut fees: HashMap<String, f64> = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let fee: FeeInput =
            serde_json::from_str(&line).with_context(|| format!("parse fee on line {}", i + 1))?;

        if let Some(intent_id) = fee.intent_id {
            *fees.entry(intent_id).or_default() += fee.fees_total_inr;
        }
    }

    Ok(fees)
}

/// Generate IntentPnlRecord by joining labels with fees
fn generate_intent_pnl_records(
    labels: &[LabelInput],
    fees: &HashMap<String, f64>,
) -> Vec<IntentPnlRecord> {
    let mut records = Vec::with_capacity(labels.len());

    for label in labels {
        let fees_inr = fees.get(&label.intent_id).copied().unwrap_or(0.0);
        let fill_value_inr = label.fill_value_inr.unwrap_or(0.0);
        let gross_pnl_inr = label.gross_edge_inr.unwrap_or(0.0);
        let net_pnl_inr = gross_pnl_inr - fees_inr;

        records.push(IntentPnlRecord {
            schema: INTENT_PNL_SCHEMA,
            schema_rev: INTENT_PNL_SCHEMA_REV,
            intent_id: label.intent_id.clone(),
            ts_utc: label.ts_utc.clone(),
            filled: label.filled,
            side: label.side.clone(),
            order_type: label.order_type.clone(),
            filled_qty: label.filled_qty,
            fill_value_inr,
            gross_pnl_inr,
            fees_inr,
            net_pnl_inr,
            execution_tax_bps: label.execution_tax_bps,
            is_winner: net_pnl_inr > 0.0,
        });
    }

    records
}

/// Write intent PnL records to JSONL
fn write_intent_pnl(path: &str, records: &[IntentPnlRecord]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create intent_pnl: {}", path))?;
    let mut writer = BufWriter::new(file);

    for record in records {
        writeln!(writer, "{}", serde_json::to_string(record)?)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_compute() {
        let records = vec![
            IntentPnlRecord {
                schema: INTENT_PNL_SCHEMA,
                schema_rev: INTENT_PNL_SCHEMA_REV,
                intent_id: "a".to_string(),
                ts_utc: "2026-01-01T00:00:00Z".to_string(),
                filled: true,
                side: "Buy".to_string(),
                order_type: "Market".to_string(),
                filled_qty: 30,
                fill_value_inr: 30000.0,
                gross_pnl_inr: 100.0,
                fees_inr: 50.0,
                net_pnl_inr: 50.0, // Winner
                execution_tax_bps: 33.33,
                is_winner: true,
            },
            IntentPnlRecord {
                schema: INTENT_PNL_SCHEMA,
                schema_rev: INTENT_PNL_SCHEMA_REV,
                intent_id: "b".to_string(),
                ts_utc: "2026-01-01T00:01:00Z".to_string(),
                filled: true,
                side: "Sell".to_string(),
                order_type: "Limit".to_string(),
                filled_qty: 30,
                fill_value_inr: 30000.0,
                gross_pnl_inr: -100.0,
                fees_inr: 50.0,
                net_pnl_inr: -150.0, // Loser
                execution_tax_bps: -33.33,
                is_winner: false,
            },
        ];

        let summary = IntentPnlSummary::compute(&records);

        assert_eq!(summary.filled_intents, 2);
        assert_eq!(summary.winners, 1);
        assert_eq!(summary.losers, 1);
        assert!((summary.win_rate_pct - 50.0).abs() < 0.1);
        assert!((summary.gross_profit_inr - 50.0).abs() < 0.1);
        assert!((summary.gross_loss_inr - 150.0).abs() < 0.1);
        assert!((summary.profit_factor - (50.0 / 150.0)).abs() < 0.01);
    }
}
