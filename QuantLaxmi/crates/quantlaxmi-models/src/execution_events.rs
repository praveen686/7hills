//! # Execution Events (Phase 14.2)
//!
//! Canonical event types for live execution lifecycle.
//!
//! ## Core Question
//! "How do budgets become executed trades with deterministic audit trail?"
//!
//! ## Event Flow
//! ```text
//! Intent → BudgetCheck → Reserve → Submit → Ack/Reject → Fill/Cancel → Reconcile
//! ```
//!
//! ## Invariants
//! - All IDs are derived deterministically (SHA-256, no UUIDs)
//! - All events are WAL-bound (immutable once written)
//! - All monetary values use fixed-point (no f64)
//! - Every state transition produces an auditable event

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const EXECUTION_EVENTS_SCHEMA_VERSION: &str = "execution_events_v1.0";

// =============================================================================
// Identifier Types
// =============================================================================

/// Unique identifier for an order intent.
/// Derived deterministically from strategy_id + bucket_id + ts_ns + seq.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentId(pub String);

impl IntentId {
    /// Derive intent ID deterministically.
    pub fn derive(strategy_id: &str, bucket_id: &str, ts_ns: i64, seq: u64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"intent:");
        hasher.update(strategy_id.as_bytes());
        hasher.update(b":");
        hasher.update(bucket_id.as_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        hasher.update(b":");
        hasher.update(seq.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for IntentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for an order submitted to exchange.
/// Derived deterministically from intent_id + submit_ts_ns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientOrderId(pub String);

impl ClientOrderId {
    /// Derive client order ID deterministically.
    /// Format: First 32 hex chars to fit exchange limits.
    pub fn derive(intent_id: &IntentId, submit_ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"client_order:");
        hasher.update(intent_id.0.as_bytes());
        hasher.update(b":");
        hasher.update(submit_ts_ns.to_le_bytes());
        let hash = format!("{:x}", hasher.finalize());
        // Truncate to 32 chars for exchange compatibility
        Self(hash[..32].to_string())
    }
}

impl std::fmt::Display for ClientOrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Exchange-assigned order ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExchangeOrderId(pub String);

impl ExchangeOrderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for ExchangeOrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a fill event.
/// Derived from exchange fill ID + client order ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FillId(pub String);

impl FillId {
    /// Derive fill ID deterministically.
    pub fn derive(exchange_fill_id: &str, client_order_id: &ClientOrderId) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"fill:");
        hasher.update(exchange_fill_id.as_bytes());
        hasher.update(b":");
        hasher.update(client_order_id.0.as_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for FillId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Order Side and Type (Re-exported from parent for convenience)
// =============================================================================

/// Order side (direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionSide {
    Buy,
    Sell,
}

impl std::fmt::Display for ExecutionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionSide::Buy => write!(f, "BUY"),
            ExecutionSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionOrderType {
    Market,
    Limit,
}

impl std::fmt::Display for ExecutionOrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionOrderType::Market => write!(f, "MARKET"),
            ExecutionOrderType::Limit => write!(f, "LIMIT"),
        }
    }
}

// =============================================================================
// Live Order State Machine
// =============================================================================

/// State machine for live order lifecycle.
/// All transitions are explicit and auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LiveOrderState {
    /// Intent created, not yet budget-checked.
    IntentCreated,
    /// Budget reserved for this order.
    BudgetReserved,
    /// Submitted to exchange, awaiting ack.
    Submitted,
    /// Acknowledged by exchange.
    Acked,
    /// Partially filled.
    PartFilled,
    /// Fully filled.
    Filled,
    /// Cancelled (by user or exchange).
    Cancelled,
    /// Rejected by exchange.
    Rejected,
}

impl std::fmt::Display for LiveOrderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveOrderState::IntentCreated => write!(f, "INTENT_CREATED"),
            LiveOrderState::BudgetReserved => write!(f, "BUDGET_RESERVED"),
            LiveOrderState::Submitted => write!(f, "SUBMITTED"),
            LiveOrderState::Acked => write!(f, "ACKED"),
            LiveOrderState::PartFilled => write!(f, "PART_FILLED"),
            LiveOrderState::Filled => write!(f, "FILLED"),
            LiveOrderState::Cancelled => write!(f, "CANCELLED"),
            LiveOrderState::Rejected => write!(f, "REJECTED"),
        }
    }
}

// =============================================================================
// Fixed-Point Price/Quantity
// =============================================================================

/// Fixed-point representation for prices and quantities.
/// All values use mantissa + exponent to avoid floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedPointValue {
    /// The mantissa (significant digits).
    pub mantissa: i128,
    /// The exponent (power of 10).
    pub exponent: i8,
}

impl FixedPointValue {
    pub fn new(mantissa: i128, exponent: i8) -> Self {
        Self { mantissa, exponent }
    }

    pub fn zero(exponent: i8) -> Self {
        Self {
            mantissa: 0,
            exponent,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    pub fn is_positive(&self) -> bool {
        self.mantissa > 0
    }

    /// Add another value (must have same exponent).
    pub fn checked_add(&self, other: &FixedPointValue) -> Option<FixedPointValue> {
        if self.exponent != other.exponent {
            return None;
        }
        Some(FixedPointValue {
            mantissa: self.mantissa.checked_add(other.mantissa)?,
            exponent: self.exponent,
        })
    }

    /// Subtract another value (must have same exponent).
    pub fn checked_sub(&self, other: &FixedPointValue) -> Option<FixedPointValue> {
        if self.exponent != other.exponent {
            return None;
        }
        Some(FixedPointValue {
            mantissa: self.mantissa.checked_sub(other.mantissa)?,
            exponent: self.exponent,
        })
    }
}

impl std::fmt::Display for FixedPointValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}e{}", self.mantissa, self.exponent)
    }
}

// =============================================================================
// Canonical Events
// =============================================================================

/// Order intent event - strategy wants to trade.
/// This is the entry point for the execution pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntentEvent {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Event timestamp (nanoseconds since epoch).
    pub ts_ns: i64,
    /// Deterministic intent ID.
    pub intent_id: IntentId,
    /// Strategy requesting the trade.
    pub strategy_id: String,
    /// Bucket providing capital.
    pub bucket_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Order side.
    pub side: ExecutionSide,
    /// Order type.
    pub order_type: ExecutionOrderType,
    /// Desired quantity (mantissa).
    pub quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Limit price (mantissa, None for market orders).
    pub limit_price_mantissa: Option<i128>,
    /// Price exponent.
    pub price_exponent: i8,
    /// Reference price at decision time (for slippage calculation).
    pub reference_price_mantissa: i128,
    /// Correlation: parent decision ID.
    pub parent_decision_id: String,
    /// Current order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderIntentEvent {
    /// Compute deterministic digest.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.intent_id.0.as_bytes());
        hasher.update(self.strategy_id.as_bytes());
        hasher.update(self.bucket_id.as_bytes());
        hasher.update(self.symbol.as_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(format!("{}", self.order_type).as_bytes());
        hasher.update(self.quantity_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        if let Some(price) = self.limit_price_mantissa {
            hasher.update(price.to_le_bytes());
        }
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.reference_price_mantissa.to_le_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Order submit event - order sent to exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSubmitEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Intent being submitted.
    pub intent_id: IntentId,
    /// Client order ID sent to exchange.
    pub client_order_id: ClientOrderId,
    /// Budget delta ID for reservation.
    pub budget_delta_id: String,
    /// Reserved amount (mantissa).
    pub reserved_mantissa: i128,
    /// Reserved exponent.
    pub reserved_exponent: i8,
    /// New order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderSubmitEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.intent_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.budget_delta_id.as_bytes());
        hasher.update(self.reserved_mantissa.to_le_bytes());
        hasher.update([self.reserved_exponent as u8]);
        format!("{:x}", hasher.finalize())
    }
}

/// Order ack event - exchange acknowledged the order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAckEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds) - exchange time.
    pub ts_ns: i64,
    /// Our client order ID.
    pub client_order_id: ClientOrderId,
    /// Exchange-assigned order ID.
    pub exchange_order_id: ExchangeOrderId,
    /// Exchange timestamp (nanoseconds).
    pub exchange_ts_ns: i64,
    /// New order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderAckEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.exchange_order_id.0.as_bytes());
        hasher.update(self.exchange_ts_ns.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Order reject event - exchange rejected the order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRejectEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Our client order ID.
    pub client_order_id: ClientOrderId,
    /// Exchange-assigned order ID (if any).
    pub exchange_order_id: Option<ExchangeOrderId>,
    /// Rejection reason from exchange.
    pub reject_reason: String,
    /// Rejection code from exchange (if available).
    pub reject_code: Option<i32>,
    /// Budget delta ID for rollback.
    pub rollback_delta_id: String,
    /// Released amount (mantissa).
    pub released_mantissa: i128,
    /// Released exponent.
    pub released_exponent: i8,
    /// New order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderRejectEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        if let Some(ref eid) = self.exchange_order_id {
            hasher.update(eid.0.as_bytes());
        }
        hasher.update(self.reject_reason.as_bytes());
        hasher.update(self.rollback_delta_id.as_bytes());
        hasher.update(self.released_mantissa.to_le_bytes());
        hasher.update([self.released_exponent as u8]);
        format!("{:x}", hasher.finalize())
    }
}

/// Order fill event - partial or full fill from exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFillEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds) - exchange time.
    pub ts_ns: i64,
    /// Our client order ID.
    pub client_order_id: ClientOrderId,
    /// Exchange-assigned order ID.
    pub exchange_order_id: ExchangeOrderId,
    /// Deterministic fill ID.
    pub fill_id: FillId,
    /// Exchange fill ID.
    pub exchange_fill_id: String,
    /// Fill price (mantissa).
    pub fill_price_mantissa: i128,
    /// Fill quantity (mantissa).
    pub fill_quantity_mantissa: i128,
    /// Price/quantity exponent.
    pub exponent: i8,
    /// Commission amount (mantissa).
    pub commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Commission asset.
    pub commission_asset: String,
    /// Is this the final fill (order complete)?
    pub is_final: bool,
    /// Cumulative filled quantity (mantissa).
    pub cumulative_filled_mantissa: i128,
    /// Remaining quantity (mantissa).
    pub remaining_quantity_mantissa: i128,
    /// Budget delta ID for commitment.
    pub commitment_delta_id: String,
    /// Committed amount (mantissa).
    pub committed_mantissa: i128,
    /// Committed exponent.
    pub committed_exponent: i8,
    /// New order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderFillEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.exchange_order_id.0.as_bytes());
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.exchange_fill_id.as_bytes());
        hasher.update(self.fill_price_mantissa.to_le_bytes());
        hasher.update(self.fill_quantity_mantissa.to_le_bytes());
        hasher.update([self.exponent as u8]);
        hasher.update(self.commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.commission_asset.as_bytes());
        hasher.update([self.is_final as u8]);
        hasher.update(self.cumulative_filled_mantissa.to_le_bytes());
        hasher.update(self.remaining_quantity_mantissa.to_le_bytes());
        hasher.update(self.commitment_delta_id.as_bytes());
        hasher.update(self.committed_mantissa.to_le_bytes());
        hasher.update([self.committed_exponent as u8]);
        format!("{:x}", hasher.finalize())
    }
}

/// Order cancel event - order cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCancelEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Our client order ID.
    pub client_order_id: ClientOrderId,
    /// Exchange-assigned order ID.
    pub exchange_order_id: ExchangeOrderId,
    /// Cancel source (user, exchange, timeout).
    pub cancel_source: CancelSource,
    /// Cancel reason.
    pub cancel_reason: String,
    /// Budget delta ID for release.
    pub release_delta_id: String,
    /// Released amount (mantissa).
    pub released_mantissa: i128,
    /// Released exponent.
    pub released_exponent: i8,
    /// Filled quantity before cancel (mantissa).
    pub filled_before_cancel_mantissa: i128,
    /// New order state.
    pub state: LiveOrderState,
    /// Deterministic digest.
    pub digest: String,
}

impl OrderCancelEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.exchange_order_id.0.as_bytes());
        hasher.update(format!("{:?}", self.cancel_source).as_bytes());
        hasher.update(self.cancel_reason.as_bytes());
        hasher.update(self.release_delta_id.as_bytes());
        hasher.update(self.released_mantissa.to_le_bytes());
        hasher.update([self.released_exponent as u8]);
        hasher.update(self.filled_before_cancel_mantissa.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Source of order cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelSource {
    /// User/strategy requested cancellation.
    User,
    /// Exchange cancelled (e.g., margin call).
    Exchange,
    /// Timeout (order expired).
    Timeout,
    /// Risk system cancelled.
    RiskSystem,
}

/// Position close event - position fully closed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionCloseEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Strategy that held the position.
    pub strategy_id: String,
    /// Bucket that funded the position.
    pub bucket_id: String,
    /// Symbol.
    pub symbol: String,
    /// Side of the original position.
    pub side: ExecutionSide,
    /// Entry price (mantissa).
    pub entry_price_mantissa: i128,
    /// Exit price (mantissa).
    pub exit_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Position size (mantissa).
    pub size_mantissa: i128,
    /// Size exponent.
    pub size_exponent: i8,
    /// Realized PnL (mantissa).
    pub realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Total commission (mantissa).
    pub total_commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Holding time (nanoseconds).
    pub holding_time_ns: i64,
    /// Number of fills to open.
    pub open_fill_count: u32,
    /// Number of fills to close.
    pub close_fill_count: u32,
    /// Budget delta ID for release.
    pub release_delta_id: String,
    /// Released capital (mantissa).
    pub released_capital_mantissa: i128,
    /// Released capital exponent.
    pub released_capital_exponent: i8,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionCloseEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.strategy_id.as_bytes());
        hasher.update(self.bucket_id.as_bytes());
        hasher.update(self.symbol.as_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(self.entry_price_mantissa.to_le_bytes());
        hasher.update(self.exit_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.size_mantissa.to_le_bytes());
        hasher.update([self.size_exponent as u8]);
        hasher.update(self.realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.total_commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.holding_time_ns.to_le_bytes());
        hasher.update(self.open_fill_count.to_le_bytes());
        hasher.update(self.close_fill_count.to_le_bytes());
        hasher.update(self.release_delta_id.as_bytes());
        hasher.update(self.released_capital_mantissa.to_le_bytes());
        hasher.update([self.released_capital_exponent as u8]);
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Idempotency Key
// =============================================================================

/// Idempotency key for exchange events.
/// Used to ensure reprocessing the same event doesn't double-reserve or double-commit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdempotencyKey(pub String);

impl IdempotencyKey {
    /// Derive from exchange event (fill_id, order_id, etc.).
    pub fn from_fill(exchange_fill_id: &str, exchange_order_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"idem:fill:");
        hasher.update(exchange_fill_id.as_bytes());
        hasher.update(b":");
        hasher.update(exchange_order_id.as_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }

    pub fn from_ack(exchange_order_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"idem:ack:");
        hasher.update(exchange_order_id.as_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }

    pub fn from_reject(client_order_id: &str, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"idem:reject:");
        hasher.update(client_order_id.as_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }

    pub fn from_cancel(exchange_order_id: &str, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"idem:cancel:");
        hasher.update(exchange_order_id.as_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_id_deterministic() {
        let id1 = IntentId::derive("strategy_001", "bucket_001", 1234567890, 1);
        let id2 = IntentId::derive("strategy_001", "bucket_001", 1234567890, 1);
        assert_eq!(id1, id2);

        // Different seq -> different ID
        let id3 = IntentId::derive("strategy_001", "bucket_001", 1234567890, 2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_client_order_id_deterministic() {
        let intent = IntentId::derive("strategy_001", "bucket_001", 1234567890, 1);
        let coid1 = ClientOrderId::derive(&intent, 1234567890);
        let coid2 = ClientOrderId::derive(&intent, 1234567890);
        assert_eq!(coid1, coid2);
        assert_eq!(coid1.0.len(), 32); // Truncated for exchange compatibility
    }

    #[test]
    fn test_fill_id_deterministic() {
        let coid = ClientOrderId("abc123".to_string());
        let fid1 = FillId::derive("exchange_fill_001", &coid);
        let fid2 = FillId::derive("exchange_fill_001", &coid);
        assert_eq!(fid1, fid2);
    }

    #[test]
    fn test_order_intent_digest_deterministic() {
        let event = OrderIntentEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns: 1234567890,
            intent_id: IntentId::derive("strategy_001", "bucket_001", 1234567890, 1),
            strategy_id: "strategy_001".to_string(),
            bucket_id: "bucket_001".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: ExecutionSide::Buy,
            order_type: ExecutionOrderType::Limit,
            quantity_mantissa: 100_000_000,
            quantity_exponent: -8,
            limit_price_mantissa: Some(50000_00000000),
            price_exponent: -8,
            reference_price_mantissa: 50000_00000000,
            parent_decision_id: "decision_001".to_string(),
            state: LiveOrderState::IntentCreated,
            digest: String::new(),
        };

        let digest1 = event.compute_digest();
        let digest2 = event.compute_digest();
        assert_eq!(digest1, digest2);
    }

    #[test]
    fn test_idempotency_key_deterministic() {
        let key1 = IdempotencyKey::from_fill("fill_001", "order_001");
        let key2 = IdempotencyKey::from_fill("fill_001", "order_001");
        assert_eq!(key1, key2);

        let key3 = IdempotencyKey::from_fill("fill_002", "order_001");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_fixed_point_value_arithmetic() {
        let a = FixedPointValue::new(100_00000000, -8);
        let b = FixedPointValue::new(25_00000000, -8);

        let sum = a.checked_add(&b).unwrap();
        assert_eq!(sum.mantissa, 125_00000000);
        assert_eq!(sum.exponent, -8);

        let diff = a.checked_sub(&b).unwrap();
        assert_eq!(diff.mantissa, 75_00000000);
        assert_eq!(diff.exponent, -8);

        // Mismatched exponents fail
        let c = FixedPointValue::new(100, -4);
        assert!(a.checked_add(&c).is_none());
    }

    #[test]
    fn test_live_order_state_display() {
        assert_eq!(format!("{}", LiveOrderState::IntentCreated), "INTENT_CREATED");
        assert_eq!(format!("{}", LiveOrderState::Filled), "FILLED");
        assert_eq!(format!("{}", LiveOrderState::Rejected), "REJECTED");
    }
}
