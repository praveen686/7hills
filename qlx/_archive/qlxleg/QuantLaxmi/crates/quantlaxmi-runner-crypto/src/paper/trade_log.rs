//! Trade Logger: Writes completed trades to trades.jsonl and summary.json.
//!
//! Provides deterministic post-mortem analysis with full entry/exit details.

use crate::paper::position_manager::CompletedTrade;
use crate::paper::sniper::SniperConfig;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use tracing::{error, info};

/// Run configuration saved to manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub run_id: String,
    pub symbol: String,
    pub mode: String,
    pub start_ts_ms: u64,
    // Canary/sniper config
    pub allowed_regimes: Vec<String>,
    pub confidence_min: f64,
    pub toxicity_max: f64,
    pub imbalance_min_abs: f64,
    pub require_fti_confirm: bool,
    pub cooldown_seconds: u64,
    pub max_entries_per_hour: u32,
    // Exit config
    pub time_stop_seconds: u64,
    pub take_profit_bps: f64,
    pub stop_loss_bps: f64,
    pub fee_rate: f64,
    // Position
    pub qty: f64,
    pub no_flip: bool,
}

impl RunConfig {
    pub fn from_sniper_config(
        run_id: &str,
        symbol: &str,
        mode: &str,
        config: &SniperConfig,
        time_stop_seconds: u64,
        take_profit_bps: f64,
        stop_loss_bps: f64,
    ) -> Self {
        Self {
            run_id: run_id.to_string(),
            symbol: symbol.to_string(),
            mode: mode.to_string(),
            start_ts_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            allowed_regimes: config
                .allowed_regimes
                .iter()
                .map(|s| s.to_string())
                .collect(),
            confidence_min: config.confidence_min,
            toxicity_max: config.toxicity_max,
            imbalance_min_abs: config.imbalance_min_abs,
            require_fti_confirm: config.require_fti_confirm,
            cooldown_seconds: config.cooldown_seconds,
            max_entries_per_hour: config.max_entries_per_hour,
            time_stop_seconds,
            take_profit_bps,
            stop_loss_bps,
            fee_rate: config.fee_rate,
            qty: config.qty,
            no_flip: config.no_flip,
        }
    }
}

/// Summary statistics for the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub end_ts_ms: u64,
    pub duration_secs: u64,
    // Trade counts
    pub accepted_trades: u64,
    pub completed_trades: u64,
    // PnL
    pub gross_pnl_total: f64,
    pub fees_total: f64,
    pub net_pnl_total: f64,
    pub avg_net_per_trade: f64,
    // Per-trade diagnostics (makes "why not profitable" self-evident)
    pub avg_gross_per_trade: f64,
    pub avg_fees_per_trade: f64,
    pub avg_notional: f64,
    /// Break-even in bps: how many bps of edge needed just to cover fees
    pub break_even_bps: f64,
    // Fee analysis
    pub fee_share: f64,
    // Win/loss
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub win_rate: f64,
    // Exit breakdown
    pub exits_tp: u64,
    pub exits_sl: u64,
    pub exits_time: u64,
    pub exits_manual: u64,
    // Verdict
    pub expectancy_positive: bool,
}

enum TradeLogMsg {
    Trade(Box<CompletedTrade>),
    Shutdown,
}

/// Async trade logger with background writer thread.
pub struct TradeLogger {
    log_dir: PathBuf,
    run_id: String,
    tx: mpsc::Sender<TradeLogMsg>,
    handle: Option<thread::JoinHandle<()>>,
    // Summary tracking
    start_ts_ms: u64,
    pub accepted_trades: u64,
    pub completed_trades: u64,
    pub gross_pnl_total: f64,
    pub fees_total: f64,
    pub net_pnl_total: f64,
    pub notional_total: f64, // For break-even calculation
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub exits_tp: u64,
    pub exits_sl: u64,
    pub exits_time: u64,
    pub exits_manual: u64,
}

impl TradeLogger {
    /// Create a new trade logger.
    pub fn new(log_dir: PathBuf, run_id: &str, config: &RunConfig) -> Result<Self, String> {
        fs::create_dir_all(&log_dir).map_err(|e| format!("Failed to create log dir: {}", e))?;

        // Write run manifest
        let manifest_path = log_dir.join(format!("{}_run_config.json", run_id));
        let manifest_json = serde_json::to_string_pretty(config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        fs::write(&manifest_path, manifest_json)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;
        info!(
            "[TRADE_LOG] Run config written to {}",
            manifest_path.display()
        );

        // Create trades file
        let trades_path = log_dir.join(format!("{}_trades.jsonl", run_id));
        let file = File::create(&trades_path)
            .map_err(|e| format!("Failed to create trades file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Create channel
        let (tx, rx) = mpsc::channel::<TradeLogMsg>();

        // Spawn writer thread
        let handle = thread::spawn(move || {
            for msg in rx {
                match msg {
                    TradeLogMsg::Trade(trade) => {
                        if let Ok(json) = serde_json::to_string(&trade) {
                            let _ = writeln!(writer, "{}", json);
                            let _ = writer.flush();
                        }
                    }
                    TradeLogMsg::Shutdown => break,
                }
            }
            let _ = writer.flush();
        });

        let start_ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(Self {
            log_dir,
            run_id: run_id.to_string(),
            tx,
            handle: Some(handle),
            start_ts_ms,
            accepted_trades: 0,
            completed_trades: 0,
            gross_pnl_total: 0.0,
            fees_total: 0.0,
            net_pnl_total: 0.0,
            notional_total: 0.0,
            winning_trades: 0,
            losing_trades: 0,
            exits_tp: 0,
            exits_sl: 0,
            exits_time: 0,
            exits_manual: 0,
        })
    }

    /// Log a completed trade.
    pub fn log_trade(&mut self, trade: CompletedTrade) {
        // Update counters
        self.completed_trades += 1;
        self.gross_pnl_total += trade.gross_pnl;
        self.fees_total += trade.fees_total;
        self.net_pnl_total += trade.net_pnl;
        self.notional_total += trade.entry_qty * trade.entry_price;

        if trade.net_pnl > 0.0 {
            self.winning_trades += 1;
        } else {
            self.losing_trades += 1;
        }

        match trade.exit_reason.as_str() {
            "TP" => self.exits_tp += 1,
            "SL" => self.exits_sl += 1,
            "TIME" => self.exits_time += 1,
            "MANUAL" => self.exits_manual += 1,
            _ => {}
        }

        // Send to writer
        let _ = self.tx.send(TradeLogMsg::Trade(Box::new(trade)));
    }

    /// Record an accepted entry (for counting accepted_trades).
    pub fn record_entry(&mut self) {
        self.accepted_trades += 1;
    }

    /// Average net PnL per trade.
    pub fn avg_net_per_trade(&self) -> f64 {
        if self.completed_trades == 0 {
            0.0
        } else {
            self.net_pnl_total / self.completed_trades as f64
        }
    }

    /// Write summary.json and shutdown.
    pub fn finalize(mut self) -> RunSummary {
        let _ = self.tx.send(TradeLogMsg::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        let end_ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let duration_secs = (end_ts_ms - self.start_ts_ms) / 1000;

        let gross_abs = self.gross_pnl_total.abs().max(1e-12);
        let fee_share = self.fees_total / gross_abs;
        let win_rate = if self.completed_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.completed_trades as f64
        };

        // Per-trade diagnostics
        let n = self.completed_trades.max(1) as f64;
        let avg_gross_per_trade = self.gross_pnl_total / n;
        let avg_fees_per_trade = self.fees_total / n;
        let avg_notional = self.notional_total / n;
        // break_even_bps = how many bps of edge needed just to cover fees
        let break_even_bps = if avg_notional > 1e-12 {
            (avg_fees_per_trade / avg_notional) * 10_000.0
        } else {
            0.0
        };

        let summary = RunSummary {
            run_id: self.run_id.clone(),
            end_ts_ms,
            duration_secs,
            accepted_trades: self.accepted_trades,
            completed_trades: self.completed_trades,
            gross_pnl_total: self.gross_pnl_total,
            fees_total: self.fees_total,
            net_pnl_total: self.net_pnl_total,
            avg_net_per_trade: self.avg_net_per_trade(),
            avg_gross_per_trade,
            avg_fees_per_trade,
            avg_notional,
            break_even_bps,
            fee_share,
            winning_trades: self.winning_trades,
            losing_trades: self.losing_trades,
            win_rate,
            exits_tp: self.exits_tp,
            exits_sl: self.exits_sl,
            exits_time: self.exits_time,
            exits_manual: self.exits_manual,
            expectancy_positive: self.avg_net_per_trade() > 0.0,
        };

        // Write summary.json
        let summary_path = self.log_dir.join(format!("{}_summary.json", self.run_id));
        if let Ok(json) = serde_json::to_string_pretty(&summary) {
            if let Err(e) = fs::write(&summary_path, json) {
                error!("[TRADE_LOG] Failed to write summary: {}", e);
            } else {
                info!("[TRADE_LOG] Summary written to {}", summary_path.display());
            }
        }

        summary
    }
}

impl Drop for TradeLogger {
    fn drop(&mut self) {
        let _ = self.tx.send(TradeLogMsg::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
