//! # VectorBT Export Module
//!
//! Unified export format for VectorBT Pro analysis.
//!
//! ## Output Files
//! - `market.csv` - OHLCV or quote data for price reference
//! - `fills.csv` - Trade fills with entry/exit pairs
//! - `summary.json` - Performance metrics summary
//!
//! ## Compatibility
//! Designed to work with VectorBT Pro portfolio analysis:
//! ```python
//! import vectorbtpro as vbt
//! market = vbt.Data.from_csv("market.csv", parse_dates=['timestamp'])
//! fills = pd.read_csv("fills.csv", parse_dates=['entry_time', 'exit_time'])
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Market data row for VectorBT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRow {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Best bid (for quote data)
    pub bid: Option<f64>,
    /// Best ask (for quote data)
    pub ask: Option<f64>,
}

/// Fill record for VectorBT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRow {
    pub fill_id: String,
    pub symbol: String,
    pub side: String, // "BUY" or "SELL"
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub order_id: Option<String>,
    /// Strategy that generated this fill
    pub strategy: Option<String>,
    /// Expert within strategy (HYDRA)
    pub expert: Option<String>,
    /// Regime at time of fill
    pub regime: Option<String>,
    /// Slippage in basis points
    pub slippage_bps: Option<f64>,
    /// Commission/fees
    pub commission: Option<f64>,
}

/// Trade (entry+exit pair) for VectorBT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRow {
    pub trade_id: String,
    pub symbol: String,
    pub direction: String, // "LONG" or "SHORT"
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub duration_secs: f64,
    /// Strategy that generated this trade
    pub strategy: Option<String>,
    /// Entry regime
    pub entry_regime: Option<String>,
    /// Exit regime
    pub exit_regime: Option<String>,
    /// Entry gate state
    pub entry_gate_state: Option<String>,
    /// Exit reason
    pub exit_reason: Option<String>,
}

/// Portfolio metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    /// Run metadata
    pub run_id: String,
    pub family: String,
    pub watermark: String,

    /// Time range
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_days: f64,

    /// Capital
    pub initial_capital: f64,
    pub final_equity: f64,
    pub total_return: f64,
    pub total_return_pct: f64,

    /// Trade statistics
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub expectancy: f64,

    /// Risk metrics
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: f64,
    pub var_95: f64,
    pub cvar_95: f64,

    /// Exposure
    pub avg_exposure: f64,
    pub max_exposure: f64,
    pub time_in_market_pct: f64,

    /// Strategy-specific
    pub regime_breakdown: Option<serde_json::Value>,
    pub gate_effectiveness: Option<serde_json::Value>,
}

impl Default for PortfolioSummary {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            family: String::new(),
            watermark: String::new(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            duration_days: 0.0,
            initial_capital: 0.0,
            final_equity: 0.0,
            total_return: 0.0,
            total_return_pct: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            expectancy: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_duration_days: 0.0,
            var_95: 0.0,
            cvar_95: 0.0,
            avg_exposure: 0.0,
            max_exposure: 0.0,
            time_in_market_pct: 0.0,
            regime_breakdown: None,
            gate_effectiveness: None,
        }
    }
}

/// VectorBT exporter
pub struct VectorBTExporter {
    market_rows: Vec<MarketRow>,
    fill_rows: Vec<FillRow>,
    trade_rows: Vec<TradeRow>,
    summary: PortfolioSummary,
}

impl VectorBTExporter {
    pub fn new() -> Self {
        Self {
            market_rows: Vec::new(),
            fill_rows: Vec::new(),
            trade_rows: Vec::new(),
            summary: PortfolioSummary::default(),
        }
    }

    /// Add market data row
    pub fn add_market_row(&mut self, row: MarketRow) {
        self.market_rows.push(row);
    }

    /// Add fill row
    pub fn add_fill(&mut self, fill: FillRow) {
        self.fill_rows.push(fill);
    }

    /// Add trade row
    pub fn add_trade(&mut self, trade: TradeRow) {
        self.trade_rows.push(trade);
    }

    /// Set portfolio summary
    pub fn set_summary(&mut self, summary: PortfolioSummary) {
        self.summary = summary;
    }

    /// Get mutable reference to summary
    pub fn summary_mut(&mut self) -> &mut PortfolioSummary {
        &mut self.summary
    }

    /// Export market data to CSV
    pub fn export_market(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .with_context(|| format!("Failed to create market file: {:?}", path))?;

        // Write header
        writeln!(file, "timestamp,symbol,open,high,low,close,volume,bid,ask")?;

        // Write rows
        for row in &self.market_rows {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                row.timestamp.to_rfc3339(),
                row.symbol,
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                row.bid.map(|v| v.to_string()).unwrap_or_default(),
                row.ask.map(|v| v.to_string()).unwrap_or_default(),
            )?;
        }

        Ok(())
    }

    /// Export fills to CSV
    pub fn export_fills(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .with_context(|| format!("Failed to create fills file: {:?}", path))?;

        // Write header
        writeln!(
            file,
            "fill_id,symbol,side,quantity,price,timestamp,order_id,strategy,expert,regime,slippage_bps,commission"
        )?;

        // Write rows
        for row in &self.fill_rows {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                row.fill_id,
                row.symbol,
                row.side,
                row.quantity,
                row.price,
                row.timestamp.to_rfc3339(),
                row.order_id.as_deref().unwrap_or(""),
                row.strategy.as_deref().unwrap_or(""),
                row.expert.as_deref().unwrap_or(""),
                row.regime.as_deref().unwrap_or(""),
                row.slippage_bps.map(|v| v.to_string()).unwrap_or_default(),
                row.commission.map(|v| v.to_string()).unwrap_or_default(),
            )?;
        }

        Ok(())
    }

    /// Export trades to CSV
    pub fn export_trades(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)
            .with_context(|| format!("Failed to create trades file: {:?}", path))?;

        // Write header
        writeln!(
            file,
            "trade_id,symbol,direction,entry_time,exit_time,entry_price,exit_price,quantity,pnl,pnl_pct,duration_secs,strategy,entry_regime,exit_regime,entry_gate_state,exit_reason"
        )?;

        // Write rows
        for row in &self.trade_rows {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                row.trade_id,
                row.symbol,
                row.direction,
                row.entry_time.to_rfc3339(),
                row.exit_time.to_rfc3339(),
                row.entry_price,
                row.exit_price,
                row.quantity,
                row.pnl,
                row.pnl_pct,
                row.duration_secs,
                row.strategy.as_deref().unwrap_or(""),
                row.entry_regime.as_deref().unwrap_or(""),
                row.exit_regime.as_deref().unwrap_or(""),
                row.entry_gate_state.as_deref().unwrap_or(""),
                row.exit_reason.as_deref().unwrap_or(""),
            )?;
        }

        Ok(())
    }

    /// Export summary to JSON
    pub fn export_summary(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.summary)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Export all to a directory
    pub fn export_all(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;

        self.export_market(&dir.join("market.csv"))?;
        self.export_fills(&dir.join("fills.csv"))?;
        self.export_trades(&dir.join("trades.csv"))?;
        self.export_summary(&dir.join("summary.json"))?;

        Ok(())
    }

    /// Compute summary metrics from fills and trades
    pub fn compute_summary(&mut self, initial_capital: f64) {
        let total_trades = self.trade_rows.len() as u64;
        if total_trades == 0 {
            return;
        }

        let wins: Vec<&TradeRow> = self.trade_rows.iter().filter(|t| t.pnl > 0.0).collect();
        let losses: Vec<&TradeRow> = self.trade_rows.iter().filter(|t| t.pnl <= 0.0).collect();

        let winning_trades = wins.len() as u64;
        let losing_trades = losses.len() as u64;

        let total_wins: f64 = wins.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losses.iter().map(|t| t.pnl.abs()).sum();

        let avg_win = if !wins.is_empty() {
            total_wins / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            total_losses / losses.len() as f64
        } else {
            0.0
        };

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let total_pnl: f64 = self.trade_rows.iter().map(|t| t.pnl).sum();
        let win_rate = winning_trades as f64 / total_trades as f64;
        let expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss);

        // Time range
        if let (Some(first), Some(last)) = (
            self.trade_rows.first(),
            self.trade_rows.last()
        ) {
            self.summary.start_time = first.entry_time;
            self.summary.end_time = last.exit_time;
            self.summary.duration_days = (last.exit_time - first.entry_time).num_seconds() as f64 / 86400.0;
        }

        self.summary.initial_capital = initial_capital;
        self.summary.final_equity = initial_capital + total_pnl;
        self.summary.total_return = total_pnl;
        self.summary.total_return_pct = (total_pnl / initial_capital) * 100.0;
        self.summary.total_trades = total_trades;
        self.summary.winning_trades = winning_trades;
        self.summary.losing_trades = losing_trades;
        self.summary.win_rate = win_rate;
        self.summary.avg_win = avg_win;
        self.summary.avg_loss = avg_loss;
        self.summary.profit_factor = profit_factor;
        self.summary.expectancy = expectancy;
    }
}

impl Default for VectorBTExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert QuoteEvent to MarketRow (for compatibility with replay data)
pub fn quote_to_market_row(
    timestamp: DateTime<Utc>,
    symbol: &str,
    bid: f64,
    ask: f64,
) -> MarketRow {
    let mid = (bid + ask) / 2.0;
    MarketRow {
        timestamp,
        symbol: symbol.to_string(),
        open: mid,
        high: ask,
        low: bid,
        close: mid,
        volume: 0.0,
        bid: Some(bid),
        ask: Some(ask),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vectorbt_export() {
        let temp = TempDir::new().unwrap();
        let mut exporter = VectorBTExporter::new();

        // Add some test data
        let now = Utc::now();

        exporter.add_market_row(MarketRow {
            timestamp: now,
            symbol: "BTCUSDT".to_string(),
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 50050.0,
            volume: 1000.0,
            bid: Some(50040.0),
            ask: Some(50060.0),
        });

        exporter.add_fill(FillRow {
            fill_id: "fill-1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            quantity: 1.0,
            price: 50050.0,
            timestamp: now,
            order_id: Some("order-1".to_string()),
            strategy: Some("AEON".to_string()),
            expert: None,
            regime: Some("Trend".to_string()),
            slippage_bps: Some(1.5),
            commission: Some(10.0),
        });

        exporter.add_trade(TradeRow {
            trade_id: "trade-1".to_string(),
            symbol: "BTCUSDT".to_string(),
            direction: "LONG".to_string(),
            entry_time: now,
            exit_time: now + chrono::Duration::hours(1),
            entry_price: 50050.0,
            exit_price: 50150.0,
            quantity: 1.0,
            pnl: 100.0,
            pnl_pct: 0.2,
            duration_secs: 3600.0,
            strategy: Some("AEON".to_string()),
            entry_regime: Some("Trend".to_string()),
            exit_regime: Some("Trend".to_string()),
            entry_gate_state: Some("Open".to_string()),
            exit_reason: Some("Target".to_string()),
        });

        exporter.export_all(temp.path()).unwrap();

        assert!(temp.path().join("market.csv").exists());
        assert!(temp.path().join("fills.csv").exists());
        assert!(temp.path().join("trades.csv").exists());
        assert!(temp.path().join("summary.json").exists());
    }
}
