//! Order Permission Gate — Pre-trade permission chokepoint.
//!
//! Phase 22C: Stateless gate that enforces execution class constraints.
//!
//! ## Rule Order (Frozen v1)
//! 1. Advisory → refuse all orders
//! 2. Passive + Market → refuse
//! 3. allow_short == false + Sell → refuse
//! 4. allow_long == false + Buy → refuse
//! 5. Else: permit
//!
//! ## Design
//! - Stateless (v1): No rate limiting or position tracking
//! - Stateful (v2, Phase 24+): Rate limiting, position limits
//!
//! ## Usage
//! ```ignore
//! use quantlaxmi_gates::{OrderPermissionGate, OrderIntentRef, OrderSide, OrderType};
//!
//! let intent = OrderIntentRef {
//!     side: OrderSide::Buy,
//!     order_type: OrderType::Limit,
//!     quantity: 100.0,
//!     strategy_id: "spread_passive",
//! };
//!
//! match OrderPermissionGate::authorize(&strategy_spec, &intent) {
//!     OrderPermission::Permit => { /* proceed */ }
//!     OrderPermission::Refuse(reason) => {
//!         tracing::warn!("Order refused: {}", reason.description());
//!     }
//! }
//! ```

use crate::strategies_manifest::{ExecutionClass, StrategySpec};
use serde::{Deserialize, Serialize};

// =============================================================================
// OrderSide — Buy or Sell
// =============================================================================

/// Order side (direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "buy"),
            OrderSide::Sell => write!(f, "sell"),
        }
    }
}

// =============================================================================
// OrderType — Limit or Market
// =============================================================================

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Limit,
    Market,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Limit => write!(f, "limit"),
            OrderType::Market => write!(f, "market"),
        }
    }
}

// =============================================================================
// OrderRefuseReason — Why the order was refused
// =============================================================================

/// Reasons an order intent can be refused.
///
/// These are static policy violations, not runtime limits.
/// Runtime limits (rate, position) are Phase 24+.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrderRefuseReason {
    /// Advisory strategies cannot emit orders
    AdvisoryCannotOrder { strategy_id: String },

    /// Passive strategies cannot emit market orders
    PassiveCannotMarketOrder { strategy_id: String },

    /// Strategy does not allow short positions (sell)
    ShortNotAllowed { strategy_id: String },

    /// Strategy does not allow long positions (buy)
    LongNotAllowed { strategy_id: String },
}

impl OrderRefuseReason {
    /// Human-readable description of the refusal reason.
    pub fn description(&self) -> String {
        match self {
            OrderRefuseReason::AdvisoryCannotOrder { strategy_id } => {
                format!(
                    "Advisory strategy '{}' cannot emit orders (execution_class=advisory)",
                    strategy_id
                )
            }
            OrderRefuseReason::PassiveCannotMarketOrder { strategy_id } => {
                format!(
                    "Passive strategy '{}' cannot emit market orders (execution_class=passive)",
                    strategy_id
                )
            }
            OrderRefuseReason::ShortNotAllowed { strategy_id } => {
                format!(
                    "Strategy '{}' does not allow short positions (allow_short=false)",
                    strategy_id
                )
            }
            OrderRefuseReason::LongNotAllowed { strategy_id } => {
                format!(
                    "Strategy '{}' does not allow long positions (allow_long=false)",
                    strategy_id
                )
            }
        }
    }

    /// Get the strategy ID from the reason.
    pub fn strategy_id(&self) -> &str {
        match self {
            OrderRefuseReason::AdvisoryCannotOrder { strategy_id }
            | OrderRefuseReason::PassiveCannotMarketOrder { strategy_id }
            | OrderRefuseReason::ShortNotAllowed { strategy_id }
            | OrderRefuseReason::LongNotAllowed { strategy_id } => strategy_id,
        }
    }
}

// =============================================================================
// OrderPermission — The permission verdict
// =============================================================================

/// Result of order permission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderPermission {
    /// Order is permitted to proceed
    Permit,
    /// Order is refused with reason
    Refuse(OrderRefuseReason),
}

impl OrderPermission {
    /// Check if the permission is granted.
    pub fn is_permitted(&self) -> bool {
        matches!(self, OrderPermission::Permit)
    }

    /// Check if the permission is refused.
    pub fn is_refused(&self) -> bool {
        matches!(self, OrderPermission::Refuse(_))
    }

    /// Get the refusal reason if refused.
    pub fn refuse_reason(&self) -> Option<&OrderRefuseReason> {
        match self {
            OrderPermission::Permit => None,
            OrderPermission::Refuse(reason) => Some(reason),
        }
    }
}

// =============================================================================
// OrderIntentRef — Minimal order intent for permission checking
// =============================================================================

/// Minimal order intent reference for permission checking.
///
/// This is a lightweight view into an order intent, containing only
/// the fields needed for permission checking.
#[derive(Debug, Clone)]
pub struct OrderIntentRef<'a> {
    /// Order side (Buy/Sell)
    pub side: OrderSide,

    /// Order type (Limit/Market)
    pub order_type: OrderType,

    /// Order quantity (v1: pass-through, not validated here)
    pub quantity: f64,

    /// Strategy ID (for logging/diagnostics)
    pub strategy_id: &'a str,
}

// =============================================================================
// OrderPermissionGate — Stateless permission gate (v1)
// =============================================================================

/// Stateless order permission gate.
///
/// Enforces execution class constraints on order intents.
/// Does NOT track rate limits or positions (that's v2/Phase 24+).
///
/// ## Rule Order (Frozen)
/// 1. Advisory → refuse all orders
/// 2. Passive + Market → refuse
/// 3. allow_short == false + Sell → refuse
/// 4. allow_long == false + Buy → refuse
/// 5. Else: permit
pub struct OrderPermissionGate;

impl OrderPermissionGate {
    /// Check if an order intent is permitted by the strategy spec.
    ///
    /// Rule order is frozen:
    /// 1. Advisory → refuse all
    /// 2. Passive + Market → refuse
    /// 3. allow_short=false + Sell → refuse
    /// 4. allow_long=false + Buy → refuse
    /// 5. Else: permit
    pub fn authorize(strategy_spec: &StrategySpec, intent: &OrderIntentRef<'_>) -> OrderPermission {
        // Rule 1: Advisory cannot order
        if strategy_spec.execution_class == ExecutionClass::Advisory {
            return OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder {
                strategy_id: strategy_spec.strategy_id.clone(),
            });
        }

        // Rule 2: Passive cannot market order
        if strategy_spec.execution_class == ExecutionClass::Passive
            && intent.order_type == OrderType::Market
        {
            return OrderPermission::Refuse(OrderRefuseReason::PassiveCannotMarketOrder {
                strategy_id: strategy_spec.strategy_id.clone(),
            });
        }

        // Rule 3: Check allow_short
        if !strategy_spec.allow_short && intent.side == OrderSide::Sell {
            return OrderPermission::Refuse(OrderRefuseReason::ShortNotAllowed {
                strategy_id: strategy_spec.strategy_id.clone(),
            });
        }

        // Rule 4: Check allow_long
        if !strategy_spec.allow_long && intent.side == OrderSide::Buy {
            return OrderPermission::Refuse(OrderRefuseReason::LongNotAllowed {
                strategy_id: strategy_spec.strategy_id.clone(),
            });
        }

        // Rule 5: Permit
        OrderPermission::Permit
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_advisory_spec() -> StrategySpec {
        StrategySpec {
            strategy_id: "test_advisory".to_string(),
            description: "Test advisory".to_string(),
            signals: vec!["spread".to_string()],
            execution_class: ExecutionClass::Advisory,
            max_orders_per_min: 0,
            max_position_abs: 0,
            allow_short: false,
            allow_long: false,
            allow_market_orders: false,
            tags: vec![],
        }
    }

    fn make_passive_spec() -> StrategySpec {
        StrategySpec {
            strategy_id: "test_passive".to_string(),
            description: "Test passive".to_string(),
            signals: vec!["spread".to_string()],
            execution_class: ExecutionClass::Passive,
            max_orders_per_min: 120,
            max_position_abs: 10000,
            allow_short: true,
            allow_long: true,
            allow_market_orders: false,
            tags: vec![],
        }
    }

    fn make_aggressive_spec() -> StrategySpec {
        StrategySpec {
            strategy_id: "test_aggressive".to_string(),
            description: "Test aggressive".to_string(),
            signals: vec!["spread".to_string()],
            execution_class: ExecutionClass::Aggressive,
            max_orders_per_min: 30,
            max_position_abs: 5000,
            allow_short: true,
            allow_long: true,
            allow_market_orders: true,
            tags: vec![],
        }
    }

    fn make_intent<'a>(
        side: OrderSide,
        order_type: OrderType,
        strategy_id: &'a str,
    ) -> OrderIntentRef<'a> {
        OrderIntentRef {
            side,
            order_type,
            quantity: 100.0,
            strategy_id,
        }
    }

    // =========================================================================
    // Advisory tests
    // =========================================================================

    #[test]
    fn test_advisory_refuses_limit_buy() {
        let spec = make_advisory_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Limit, "test_advisory");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        match result {
            OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder { strategy_id }) => {
                assert_eq!(strategy_id, "test_advisory");
            }
            _ => panic!("Expected AdvisoryCannotOrder"),
        }
    }

    #[test]
    fn test_advisory_refuses_limit_sell() {
        let spec = make_advisory_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_advisory");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        matches!(
            result,
            OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder { .. })
        );
    }

    #[test]
    fn test_advisory_refuses_market_buy() {
        let spec = make_advisory_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Market, "test_advisory");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        matches!(
            result,
            OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder { .. })
        );
    }

    #[test]
    fn test_advisory_refuses_market_sell() {
        let spec = make_advisory_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Market, "test_advisory");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        matches!(
            result,
            OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder { .. })
        );
    }

    // =========================================================================
    // Passive tests
    // =========================================================================

    #[test]
    fn test_passive_permits_limit_buy() {
        let spec = make_passive_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_passive_permits_limit_sell() {
        let spec = make_passive_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_passive_refuses_market_buy() {
        let spec = make_passive_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Market, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        match result {
            OrderPermission::Refuse(OrderRefuseReason::PassiveCannotMarketOrder {
                strategy_id,
            }) => {
                assert_eq!(strategy_id, "test_passive");
            }
            _ => panic!("Expected PassiveCannotMarketOrder"),
        }
    }

    #[test]
    fn test_passive_refuses_market_sell() {
        let spec = make_passive_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Market, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        matches!(
            result,
            OrderPermission::Refuse(OrderRefuseReason::PassiveCannotMarketOrder { .. })
        );
    }

    // =========================================================================
    // Aggressive tests
    // =========================================================================

    #[test]
    fn test_aggressive_permits_limit_buy() {
        let spec = make_aggressive_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Limit, "test_aggressive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_aggressive_permits_market_buy() {
        let spec = make_aggressive_spec();
        let intent = make_intent(OrderSide::Buy, OrderType::Market, "test_aggressive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_aggressive_permits_limit_sell() {
        let spec = make_aggressive_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_aggressive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_aggressive_permits_market_sell() {
        let spec = make_aggressive_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Market, "test_aggressive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    // =========================================================================
    // allow_short / allow_long tests
    // =========================================================================

    #[test]
    fn test_allow_short_false_refuses_sell() {
        let mut spec = make_passive_spec();
        spec.allow_short = false;

        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        match result {
            OrderPermission::Refuse(OrderRefuseReason::ShortNotAllowed { strategy_id }) => {
                assert_eq!(strategy_id, "test_passive");
            }
            _ => panic!("Expected ShortNotAllowed"),
        }
    }

    #[test]
    fn test_allow_short_false_permits_buy() {
        let mut spec = make_passive_spec();
        spec.allow_short = false;

        let intent = make_intent(OrderSide::Buy, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    #[test]
    fn test_allow_long_false_refuses_buy() {
        let mut spec = make_passive_spec();
        spec.allow_long = false;

        let intent = make_intent(OrderSide::Buy, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_refused());
        match result {
            OrderPermission::Refuse(OrderRefuseReason::LongNotAllowed { strategy_id }) => {
                assert_eq!(strategy_id, "test_passive");
            }
            _ => panic!("Expected LongNotAllowed"),
        }
    }

    #[test]
    fn test_allow_long_false_permits_sell() {
        let mut spec = make_passive_spec();
        spec.allow_long = false;

        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        assert!(result.is_permitted());
    }

    // =========================================================================
    // Description tests
    // =========================================================================

    #[test]
    fn test_descriptions_non_empty_and_stable() {
        let reasons = vec![
            OrderRefuseReason::AdvisoryCannotOrder {
                strategy_id: "test".to_string(),
            },
            OrderRefuseReason::PassiveCannotMarketOrder {
                strategy_id: "test".to_string(),
            },
            OrderRefuseReason::ShortNotAllowed {
                strategy_id: "test".to_string(),
            },
            OrderRefuseReason::LongNotAllowed {
                strategy_id: "test".to_string(),
            },
        ];

        for reason in reasons {
            let desc = reason.description();
            assert!(!desc.is_empty(), "Description should not be empty");
            assert!(
                desc.contains("test"),
                "Description should contain strategy_id"
            );

            // Verify stable (call twice, same result)
            assert_eq!(desc, reason.description());
        }
    }

    #[test]
    fn test_strategy_id_accessor() {
        let reason = OrderRefuseReason::AdvisoryCannotOrder {
            strategy_id: "my_strategy".to_string(),
        };
        assert_eq!(reason.strategy_id(), "my_strategy");
    }

    // =========================================================================
    // OrderPermission helper tests
    // =========================================================================

    #[test]
    fn test_order_permission_helpers() {
        let permit = OrderPermission::Permit;
        assert!(permit.is_permitted());
        assert!(!permit.is_refused());
        assert!(permit.refuse_reason().is_none());

        let refuse = OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder {
            strategy_id: "test".to_string(),
        });
        assert!(!refuse.is_permitted());
        assert!(refuse.is_refused());
        assert!(refuse.refuse_reason().is_some());
    }

    // =========================================================================
    // Display impl tests
    // =========================================================================

    #[test]
    fn test_order_side_display() {
        assert_eq!(format!("{}", OrderSide::Buy), "buy");
        assert_eq!(format!("{}", OrderSide::Sell), "sell");
    }

    #[test]
    fn test_order_type_display() {
        assert_eq!(format!("{}", OrderType::Limit), "limit");
        assert_eq!(format!("{}", OrderType::Market), "market");
    }

    // =========================================================================
    // Rule priority test (advisory checked before allow_short/allow_long)
    // =========================================================================

    #[test]
    fn test_rule_priority_advisory_first() {
        // Advisory spec with allow_short=false
        // Should fail on Advisory rule first, not ShortNotAllowed
        let spec = make_advisory_spec();
        let intent = make_intent(OrderSide::Sell, OrderType::Limit, "test_advisory");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        match result {
            OrderPermission::Refuse(OrderRefuseReason::AdvisoryCannotOrder { .. }) => {}
            _ => panic!("Expected AdvisoryCannotOrder (rule priority)"),
        }
    }

    #[test]
    fn test_rule_priority_passive_market_before_short() {
        // Passive spec with allow_short=false, market sell
        // Should fail on PassiveCannotMarketOrder first
        let mut spec = make_passive_spec();
        spec.allow_short = false;

        let intent = make_intent(OrderSide::Sell, OrderType::Market, "test_passive");
        let result = OrderPermissionGate::authorize(&spec, &intent);

        match result {
            OrderPermission::Refuse(OrderRefuseReason::PassiveCannotMarketOrder { .. }) => {}
            _ => panic!("Expected PassiveCannotMarketOrder (rule priority)"),
        }
    }
}
