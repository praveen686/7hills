//! Harvester state for TUI rendering.
//!
//! Broadcast via `watch::Sender<HarvesterState>` from the engine loop.

use crate::scanner::FundingOpportunity;

/// Per-position display state.
#[derive(Debug, Clone, Default)]
pub struct PositionDisplay {
    pub symbol: String,
    pub notional_usd: f64,
    pub funding_rate: f64,
    pub annualized_pct: f64,
    pub basis_bps: f64,
    pub funding_collected: f64,
    pub settlements: u32,
    pub spot_bid: f64,
    pub spot_ask: f64,
    pub perp_bid: f64,
    pub perp_ask: f64,
}

/// Complete state for TUI rendering.
#[derive(Debug, Clone, Default)]
pub struct HarvesterState {
    /// Top funding opportunities from scanner
    pub scanner_top: Vec<FundingOpportunity>,

    /// Active positions
    pub positions: Vec<PositionDisplay>,

    /// Portfolio summary
    pub initial_capital: f64,
    pub allocated_usd: f64,
    pub available_usd: f64,
    pub total_funding_collected: f64,
    pub funding_yield_annualized: f64,
    pub position_count: usize,

    /// Engine state (from PaperState)
    pub equity: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub fees_paid: f64,
    pub fills: u64,
    pub rejections: u64,
    pub last_decision: String,

    /// Liveness
    pub is_finished: bool,
    pub tick_count: u64,
}

impl HarvesterState {
    /// Build from portfolio and scanner data.
    pub fn from_portfolio(
        portfolio: &crate::portfolio::Portfolio,
        scanner_top: Vec<FundingOpportunity>,
        snapshot: &crate::snapshot::FundingArbSnapshot,
    ) -> Self {
        let positions: Vec<PositionDisplay> = portfolio
            .positions
            .iter()
            .map(|(sym, pf)| {
                let state = snapshot.symbols.get(sym);
                PositionDisplay {
                    symbol: sym.clone(),
                    notional_usd: pf.notional_usd,
                    funding_rate: state.map(|s| s.funding_rate).unwrap_or(0.0),
                    annualized_pct: state.map(|s| s.annualized_pct()).unwrap_or(0.0),
                    basis_bps: state.map(|s| s.basis_bps()).unwrap_or(0.0),
                    funding_collected: pf.total_funding_usd,
                    settlements: pf.settlements,
                    spot_bid: state.map(|s| s.spot_bid).unwrap_or(0.0),
                    spot_ask: state.map(|s| s.spot_ask).unwrap_or(0.0),
                    perp_bid: state.map(|s| s.perp_bid).unwrap_or(0.0),
                    perp_ask: state.map(|s| s.perp_ask).unwrap_or(0.0),
                }
            })
            .collect();

        Self {
            scanner_top,
            positions,
            initial_capital: portfolio.initial_capital,
            allocated_usd: portfolio.allocated_usd,
            available_usd: portfolio.available_capital(),
            total_funding_collected: portfolio.total_funding_collected,
            funding_yield_annualized: portfolio.funding_yield_annualized(),
            position_count: portfolio.position_count(),
            ..Default::default()
        }
    }
}
