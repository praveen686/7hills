//! Paper trading engine combining simulator, risk, and fill sink.

use quantlaxmi_models::depth::DepthEvent;

use crate::RiskEnvelope;
use crate::paper::risk::{PaperRisk, RiskDecision};
use crate::paper::sink::FillSink;
use crate::sim::{Fill, Order, SimConfig, Simulator};

/// Paper trading engine that combines:
/// - Simulator (order matching)
/// - Risk envelope (position limits)
/// - Fill sink (persistence/logging)
pub struct PaperEngine<S: FillSink> {
    /// The unified simulator
    pub sim: Simulator,
    /// Risk adapter
    risk: PaperRisk,
    /// Fill sink for persistence
    sink: S,
}

impl<S: FillSink> PaperEngine<S> {
    /// Create a new paper engine.
    pub fn new(sim_cfg: SimConfig, risk: RiskEnvelope, sink: S) -> Self {
        Self {
            sim: Simulator::new(sim_cfg),
            risk: PaperRisk::new(risk),
            sink,
        }
    }

    /// Process a depth event.
    ///
    /// Updates the order book and may trigger fills for pending orders.
    pub fn on_depth(&mut self, event: &DepthEvent) -> anyhow::Result<Vec<Fill>> {
        let symbol = &event.tradingsymbol;
        let fills = self.sim.on_depth(symbol, event);

        for f in &fills {
            self.sink.on_fill(f)?;
        }
        Ok(fills)
    }

    /// Submit an order.
    ///
    /// The order goes through risk checks before being submitted to the simulator.
    pub fn submit(&mut self, ts_ns: u64, order: Order) -> anyhow::Result<Vec<Fill>> {
        // Get current position and price for risk check
        let current_position = self.sim.position(&order.symbol);
        let price = order
            .limit_price
            .or_else(|| self.sim.best_ask(&order.symbol))
            .or_else(|| self.sim.best_bid(&order.symbol))
            .unwrap_or(0.0);

        // Risk check
        match self.risk.decide(&order, current_position, price) {
            RiskDecision::Accept => {
                let fills = self.sim.submit(ts_ns, order);
                for f in &fills {
                    self.sink.on_fill(f)?;
                }
                Ok(fills)
            }
            RiskDecision::Reject { reason } => Err(anyhow::anyhow!(reason)),
        }
    }

    /// Flush the fill sink.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        self.sink.flush()
    }

    /// Get best bid for a symbol.
    pub fn best_bid(&self, symbol: &str) -> Option<f64> {
        self.sim.best_bid(symbol)
    }

    /// Get best ask for a symbol.
    pub fn best_ask(&self, symbol: &str) -> Option<f64> {
        self.sim.best_ask(symbol)
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.sim.position(symbol)
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.sim.realized_pnl()
    }

    /// Get unrealized PnL.
    pub fn unrealized_pnl(&self) -> f64 {
        self.sim.unrealized_pnl()
    }

    /// Get cash balance.
    pub fn cash(&self) -> f64 {
        self.sim.cash()
    }

    /// Get all fills.
    pub fn fills(&self) -> &[Fill] {
        self.sim.fills()
    }

    /// Get pending orders for a symbol.
    pub fn pending_orders(&self, symbol: &str) -> Vec<&Order> {
        self.sim.pending_orders(symbol)
    }
}
