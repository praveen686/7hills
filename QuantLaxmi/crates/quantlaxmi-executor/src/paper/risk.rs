//! Risk adapter for paper trading.

use crate::RiskEnvelope;
use crate::sim::Order;

/// Risk decision outcome.
#[derive(Debug, Clone)]
pub enum RiskDecision {
    Accept,
    Reject { reason: String },
}

/// Adapter boundary so paper engine can use existing RiskEnvelope.
pub struct PaperRisk {
    env: RiskEnvelope,
}

impl PaperRisk {
    pub fn new(env: RiskEnvelope) -> Self {
        Self { env }
    }

    /// Decide whether to accept or reject an order based on risk limits.
    pub fn decide(&self, order: &Order, current_position: f64, price: f64) -> RiskDecision {
        if !self.env.enabled {
            return RiskDecision::Accept;
        }

        let order_notional = order.qty * price;

        // Check max order notional
        if order_notional > self.env.max_order_notional_usd {
            return RiskDecision::Reject {
                reason: format!(
                    "Order notional ${:.0} exceeds limit ${:.0}",
                    order_notional, self.env.max_order_notional_usd
                ),
            };
        }

        // Calculate projected position
        let delta = match order.side {
            crate::sim::Side::Buy => order.qty,
            crate::sim::Side::Sell => -order.qty,
        };
        let new_position = current_position + delta;
        let new_notional = new_position.abs() * price;

        // Check max symbol notional
        if new_notional > self.env.max_symbol_notional_usd {
            return RiskDecision::Reject {
                reason: format!(
                    "Symbol notional ${:.0} would exceed limit ${:.0}",
                    new_notional, self.env.max_symbol_notional_usd
                ),
            };
        }

        RiskDecision::Accept
    }
}
