//! Order Intent types for WAL observability (Phase 23B).
//!
//! Records every order intent BEFORE permission evaluation. Provides complete
//! audit trail of what the strategy wanted to do, regardless of whether
//! the permission gate allowed it.
//!
//! ## Core Question
//! "What order did the strategy intend to place, and what was the permission outcome?"
//!
//! ## Hard Laws (Frozen)
//! - ORDER_INTENT_SCHEMA_VERSION: "1.0.0"
//! - OrderIntentPermission variants + canonical bytes
//! - OrderIntentRecord canonical bytes order
//! - Key for determinism: (session_id, seq)
//! - Fixed-point numbers only (no f64)
//!
//! ## WAL File
//! Written to: `wal/order_intent.jsonl`
//!
//! See: Phase 23B spec for full details.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

/// Schema version for order intent record serialization.
pub const ORDER_INTENT_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// OrderIntentPermission — The binary permission outcome
// =============================================================================

/// Outcome of order permission evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderIntentPermission {
    /// Order was permitted (passed all gates)
    Permit,
    /// Order was refused (blocked by permission gate)
    Refuse,
}

impl OrderIntentPermission {
    /// Canonical byte representation (frozen).
    pub fn canonical_byte(self) -> u8 {
        match self {
            OrderIntentPermission::Permit => 0x01,
            OrderIntentPermission::Refuse => 0x02,
        }
    }
}

impl std::fmt::Display for OrderIntentPermission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderIntentPermission::Permit => write!(f, "Permit"),
            OrderIntentPermission::Refuse => write!(f, "Refuse"),
        }
    }
}

// =============================================================================
// OrderIntentSide — Order direction
// =============================================================================

/// Order side (direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderIntentSide {
    Buy,
    Sell,
}

impl OrderIntentSide {
    /// Canonical byte representation (frozen).
    pub fn canonical_byte(self) -> u8 {
        match self {
            OrderIntentSide::Buy => 0x01,
            OrderIntentSide::Sell => 0x02,
        }
    }
}

// =============================================================================
// OrderIntentType — Order type
// =============================================================================

/// Order type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderIntentType {
    Market,
    Limit,
}

impl OrderIntentType {
    /// Canonical byte representation (frozen).
    pub fn canonical_byte(self) -> u8 {
        match self {
            OrderIntentType::Market => 0x01,
            OrderIntentType::Limit => 0x02,
        }
    }
}

// =============================================================================
// OrderRefuseReason — Why was the order refused?
// =============================================================================

/// Typed reasons for order permission refusal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrderRefuseReason {
    /// Signal not admitted at L1 layer
    SignalNotAdmitted { signal_id: String },

    /// Signal not promoted (no status file)
    SignalNotPromoted { signal_id: String },

    /// Strategy not bound in manifest
    StrategyNotBound { strategy_id: String },

    /// Risk limit exceeded
    RiskLimitExceeded { limit_name: String },

    /// Position limit exceeded
    PositionLimitExceeded { current: i64, max: i64 },

    /// Rate limit exceeded
    RateLimitExceeded { window_ms: u64, count: u64, max: u64 },

    /// Execution disabled for strategy
    ExecutionDisabled { strategy_id: String },

    /// Custom refusal reason
    Custom { reason: String },
}

impl OrderRefuseReason {
    /// Canonical bytes for hashing (frozen field order).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        match self {
            OrderRefuseReason::SignalNotAdmitted { signal_id } => {
                bytes.push(0x01);
                write_string(&mut bytes, signal_id);
            }
            OrderRefuseReason::SignalNotPromoted { signal_id } => {
                bytes.push(0x02);
                write_string(&mut bytes, signal_id);
            }
            OrderRefuseReason::StrategyNotBound { strategy_id } => {
                bytes.push(0x03);
                write_string(&mut bytes, strategy_id);
            }
            OrderRefuseReason::RiskLimitExceeded { limit_name } => {
                bytes.push(0x04);
                write_string(&mut bytes, limit_name);
            }
            OrderRefuseReason::PositionLimitExceeded { current, max } => {
                bytes.push(0x05);
                bytes.extend_from_slice(&current.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            OrderRefuseReason::RateLimitExceeded {
                window_ms,
                count,
                max,
            } => {
                bytes.push(0x06);
                bytes.extend_from_slice(&window_ms.to_le_bytes());
                bytes.extend_from_slice(&count.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
            OrderRefuseReason::ExecutionDisabled { strategy_id } => {
                bytes.push(0x07);
                write_string(&mut bytes, strategy_id);
            }
            OrderRefuseReason::Custom { reason } => {
                bytes.push(0x08);
                write_string(&mut bytes, reason);
            }
        }

        bytes
    }

    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            OrderRefuseReason::SignalNotAdmitted { signal_id } => {
                format!("Signal '{}' was not admitted at L1 layer", signal_id)
            }
            OrderRefuseReason::SignalNotPromoted { signal_id } => {
                format!("Signal '{}' is not promoted", signal_id)
            }
            OrderRefuseReason::StrategyNotBound { strategy_id } => {
                format!("Strategy '{}' not bound in manifest", strategy_id)
            }
            OrderRefuseReason::RiskLimitExceeded { limit_name } => {
                format!("Risk limit '{}' exceeded", limit_name)
            }
            OrderRefuseReason::PositionLimitExceeded { current, max } => {
                format!("Position limit exceeded: {} > {}", current, max)
            }
            OrderRefuseReason::RateLimitExceeded {
                window_ms,
                count,
                max,
            } => {
                format!(
                    "Rate limit exceeded: {} orders in {}ms (max {})",
                    count, window_ms, max
                )
            }
            OrderRefuseReason::ExecutionDisabled { strategy_id } => {
                format!("Execution disabled for strategy '{}'", strategy_id)
            }
            OrderRefuseReason::Custom { reason } => reason.clone(),
        }
    }
}

// =============================================================================
// OrderIntentRecord — The WAL audit artifact
// =============================================================================

/// WAL event for order intent records.
///
/// Written to `wal/order_intent.jsonl` BEFORE permission evaluation.
/// Provides audit trail for every order intent the strategy attempted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntentRecord {
    /// Schema version for forward compatibility
    pub schema_version: String,

    /// Timestamp in nanoseconds since epoch
    pub ts_ns: i64,

    /// Session identifier
    pub session_id: String,

    /// Sequence number within session (monotonically increasing)
    pub seq: u64,

    /// Strategy that generated this intent
    pub strategy_id: String,

    /// Symbol for the order
    pub symbol: String,

    /// Order side (buy/sell)
    pub side: OrderIntentSide,

    /// Order type (market/limit)
    pub order_type: OrderIntentType,

    /// Quantity mantissa (fixed-point)
    pub qty_mantissa: i64,

    /// Quantity exponent (typically negative, e.g., -8 for 8 decimals)
    pub qty_exponent: i8,

    /// Limit price mantissa (None for market orders)
    pub limit_price_mantissa: Option<i64>,

    /// Price exponent (typically negative, e.g., -2 for 2 decimals)
    pub price_exponent: i8,

    /// Permission outcome (Permit or Refuse)
    pub permission: OrderIntentPermission,

    /// Refusal reason (empty if Permit)
    pub refuse_reason: Option<OrderRefuseReason>,

    /// Correlation ID linking to upstream market event
    pub correlation_id: String,

    /// Parent admission digest (links to strategy_admission record)
    pub parent_admission_digest: String,

    /// SHA-256 digest of canonical representation (hex string)
    pub digest: String,
}

impl OrderIntentRecord {
    /// Check if this intent was permitted.
    pub fn is_permitted(&self) -> bool {
        self.permission == OrderIntentPermission::Permit
    }

    /// Check if this intent was refused.
    pub fn is_refused(&self) -> bool {
        self.permission == OrderIntentPermission::Refuse
    }

    /// Compute canonical bytes for hashing (frozen field order).
    ///
    /// Field order:
    /// 1. schema_version (u32 LE len + UTF-8)
    /// 2. ts_ns (i64 LE)
    /// 3. session_id (u32 LE len + UTF-8)
    /// 4. seq (u64 LE)
    /// 5. strategy_id (u32 LE len + UTF-8)
    /// 6. symbol (u32 LE len + UTF-8)
    /// 7. side (u8)
    /// 8. order_type (u8)
    /// 9. qty_mantissa (i64 LE)
    /// 10. qty_exponent (i8)
    /// 11. limit_price_mantissa (0x00 + nothing OR 0x01 + i64 LE)
    /// 12. price_exponent (i8)
    /// 13. permission (u8)
    /// 14. refuse_reason (0x00 OR 0x01 + reason canonical bytes)
    /// 15. correlation_id (u32 LE len + UTF-8)
    /// 16. parent_admission_digest (u32 LE len + UTF-8)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. schema_version
        write_string(&mut bytes, &self.schema_version);

        // 2. ts_ns
        bytes.extend_from_slice(&self.ts_ns.to_le_bytes());

        // 3. session_id
        write_string(&mut bytes, &self.session_id);

        // 4. seq
        bytes.extend_from_slice(&self.seq.to_le_bytes());

        // 5. strategy_id
        write_string(&mut bytes, &self.strategy_id);

        // 6. symbol
        write_string(&mut bytes, &self.symbol);

        // 7. side
        bytes.push(self.side.canonical_byte());

        // 8. order_type
        bytes.push(self.order_type.canonical_byte());

        // 9. qty_mantissa
        bytes.extend_from_slice(&self.qty_mantissa.to_le_bytes());

        // 10. qty_exponent
        bytes.push(self.qty_exponent as u8);

        // 11. limit_price_mantissa (Option encoding)
        match self.limit_price_mantissa {
            None => bytes.push(0x00),
            Some(price) => {
                bytes.push(0x01);
                bytes.extend_from_slice(&price.to_le_bytes());
            }
        }

        // 12. price_exponent
        bytes.push(self.price_exponent as u8);

        // 13. permission
        bytes.push(self.permission.canonical_byte());

        // 14. refuse_reason (Option encoding)
        match &self.refuse_reason {
            None => bytes.push(0x00),
            Some(reason) => {
                bytes.push(0x01);
                bytes.extend_from_slice(&reason.canonical_bytes());
            }
        }

        // 15. correlation_id
        write_string(&mut bytes, &self.correlation_id);

        // 16. parent_admission_digest
        write_string(&mut bytes, &self.parent_admission_digest);

        bytes
    }

    /// Compute SHA-256 digest of canonical bytes.
    pub fn compute_digest(&self) -> String {
        let bytes = self.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Create a builder for constructing an OrderIntentRecord.
    pub fn builder(
        strategy_id: impl Into<String>,
        symbol: impl Into<String>,
    ) -> OrderIntentRecordBuilder {
        OrderIntentRecordBuilder::new(strategy_id, symbol)
    }
}

// =============================================================================
// OrderIntentRecordBuilder — Fluent construction
// =============================================================================

/// Builder for constructing OrderIntentRecord with fluent API.
///
/// # Example
/// ```ignore
/// let record = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
///     .ts_ns(1000000000)
///     .session_id("session_1")
///     .seq(42)
///     .side(OrderIntentSide::Buy)
///     .order_type(OrderIntentType::Limit)
///     .qty(100000000, -8)
///     .limit_price(5000000, -2)
///     .correlation_id("corr_123")
///     .parent_admission_digest("abc123...")
///     .build_permit();
/// ```
#[derive(Debug, Clone)]
pub struct OrderIntentRecordBuilder {
    strategy_id: String,
    symbol: String,
    ts_ns: i64,
    session_id: String,
    seq: u64,
    side: OrderIntentSide,
    order_type: OrderIntentType,
    qty_mantissa: i64,
    qty_exponent: i8,
    limit_price_mantissa: Option<i64>,
    price_exponent: i8,
    correlation_id: String,
    parent_admission_digest: String,
}

impl OrderIntentRecordBuilder {
    /// Create a new builder with required strategy and symbol identifiers.
    pub fn new(strategy_id: impl Into<String>, symbol: impl Into<String>) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            symbol: symbol.into(),
            ts_ns: 0,
            session_id: String::new(),
            seq: 0,
            side: OrderIntentSide::Buy,
            order_type: OrderIntentType::Market,
            qty_mantissa: 0,
            qty_exponent: 0,
            limit_price_mantissa: None,
            price_exponent: 0,
            correlation_id: String::new(),
            parent_admission_digest: String::new(),
        }
    }

    /// Set the timestamp in nanoseconds since epoch.
    pub fn ts_ns(mut self, ts_ns: i64) -> Self {
        self.ts_ns = ts_ns;
        self
    }

    /// Set the session identifier.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// Set the sequence number.
    pub fn seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }

    /// Set the order side.
    pub fn side(mut self, side: OrderIntentSide) -> Self {
        self.side = side;
        self
    }

    /// Set the order type.
    pub fn order_type(mut self, order_type: OrderIntentType) -> Self {
        self.order_type = order_type;
        self
    }

    /// Set the quantity (mantissa and exponent).
    pub fn qty(mut self, mantissa: i64, exponent: i8) -> Self {
        self.qty_mantissa = mantissa;
        self.qty_exponent = exponent;
        self
    }

    /// Set the limit price (mantissa and exponent). Sets order_type to Limit.
    pub fn limit_price(mut self, mantissa: i64, exponent: i8) -> Self {
        self.limit_price_mantissa = Some(mantissa);
        self.price_exponent = exponent;
        self.order_type = OrderIntentType::Limit;
        self
    }

    /// Set the price exponent (for market orders or when price is set separately).
    pub fn price_exponent(mut self, exponent: i8) -> Self {
        self.price_exponent = exponent;
        self
    }

    /// Set the correlation ID for linkage to upstream context.
    pub fn correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = correlation_id.into();
        self
    }

    /// Set the parent admission digest for linkage to strategy_admission record.
    pub fn parent_admission_digest(mut self, digest: impl Into<String>) -> Self {
        self.parent_admission_digest = digest.into();
        self
    }

    /// Build a Permit record (no refuse reason).
    pub fn build_permit(self) -> OrderIntentRecord {
        self.build_with_permission(OrderIntentPermission::Permit, None)
    }

    /// Build a Refuse record with the given reason.
    pub fn build_refuse(self, reason: OrderRefuseReason) -> OrderIntentRecord {
        self.build_with_permission(OrderIntentPermission::Refuse, Some(reason))
    }

    /// Internal: build record with given permission and reason.
    fn build_with_permission(
        self,
        permission: OrderIntentPermission,
        refuse_reason: Option<OrderRefuseReason>,
    ) -> OrderIntentRecord {
        let mut record = OrderIntentRecord {
            schema_version: ORDER_INTENT_SCHEMA_VERSION.to_string(),
            ts_ns: self.ts_ns,
            session_id: self.session_id,
            seq: self.seq,
            strategy_id: self.strategy_id,
            symbol: self.symbol,
            side: self.side,
            order_type: self.order_type,
            qty_mantissa: self.qty_mantissa,
            qty_exponent: self.qty_exponent,
            limit_price_mantissa: self.limit_price_mantissa,
            price_exponent: self.price_exponent,
            permission,
            refuse_reason,
            correlation_id: self.correlation_id,
            parent_admission_digest: self.parent_admission_digest,
            digest: String::new(),
        };
        record.digest = record.compute_digest();
        record
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Write a string with length prefix (u32 LE len + UTF-8 bytes).
fn write_string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_canonical_bytes() {
        assert_eq!(OrderIntentPermission::Permit.canonical_byte(), 0x01);
        assert_eq!(OrderIntentPermission::Refuse.canonical_byte(), 0x02);
    }

    #[test]
    fn test_side_canonical_bytes() {
        assert_eq!(OrderIntentSide::Buy.canonical_byte(), 0x01);
        assert_eq!(OrderIntentSide::Sell.canonical_byte(), 0x02);
    }

    #[test]
    fn test_order_type_canonical_bytes() {
        assert_eq!(OrderIntentType::Market.canonical_byte(), 0x01);
        assert_eq!(OrderIntentType::Limit.canonical_byte(), 0x02);
    }

    #[test]
    fn test_refuse_reason_canonical_bytes() {
        let reason1 = OrderRefuseReason::SignalNotAdmitted {
            signal_id: "spread".to_string(),
        };
        let bytes1 = reason1.canonical_bytes();
        assert_eq!(bytes1[0], 0x01);

        let reason2 = OrderRefuseReason::SignalNotPromoted {
            signal_id: "test".to_string(),
        };
        let bytes2 = reason2.canonical_bytes();
        assert_eq!(bytes2[0], 0x02);

        let reason3 = OrderRefuseReason::StrategyNotBound {
            strategy_id: "strat".to_string(),
        };
        let bytes3 = reason3.canonical_bytes();
        assert_eq!(bytes3[0], 0x03);

        let reason4 = OrderRefuseReason::RiskLimitExceeded {
            limit_name: "max_notional".to_string(),
        };
        let bytes4 = reason4.canonical_bytes();
        assert_eq!(bytes4[0], 0x04);

        let reason5 = OrderRefuseReason::PositionLimitExceeded {
            current: 100,
            max: 50,
        };
        let bytes5 = reason5.canonical_bytes();
        assert_eq!(bytes5[0], 0x05);

        let reason6 = OrderRefuseReason::RateLimitExceeded {
            window_ms: 1000,
            count: 10,
            max: 5,
        };
        let bytes6 = reason6.canonical_bytes();
        assert_eq!(bytes6[0], 0x06);

        let reason7 = OrderRefuseReason::ExecutionDisabled {
            strategy_id: "test".to_string(),
        };
        let bytes7 = reason7.canonical_bytes();
        assert_eq!(bytes7[0], 0x07);

        let reason8 = OrderRefuseReason::Custom {
            reason: "custom".to_string(),
        };
        let bytes8 = reason8.canonical_bytes();
        assert_eq!(bytes8[0], 0x08);
    }

    #[test]
    fn test_record_digest_deterministic() {
        let record1 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit)
            .qty(100000000, -8)
            .limit_price(5000000, -2)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_permit();

        let record2 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit)
            .qty(100000000, -8)
            .limit_price(5000000, -2)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_permit();

        assert_eq!(record1.digest, record2.digest);

        // Run 100 times to prove determinism
        for _ in 0..100 {
            let r = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
                .ts_ns(1000000000)
                .session_id("session_1")
                .seq(42)
                .side(OrderIntentSide::Buy)
                .order_type(OrderIntentType::Limit)
                .qty(100000000, -8)
                .limit_price(5000000, -2)
                .correlation_id("corr_123")
                .parent_admission_digest("abc123")
                .build_permit();
            assert_eq!(r.digest, record1.digest);
        }
    }

    #[test]
    fn test_record_digest_changes_with_content() {
        let record1 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .qty(100000000, -8)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_permit();

        // Change timestamp
        let record2 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000001)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .qty(100000000, -8)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_permit();

        assert_ne!(record1.digest, record2.digest);

        // Change seq
        let record3 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(43)
            .side(OrderIntentSide::Buy)
            .qty(100000000, -8)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_permit();

        assert_ne!(record1.digest, record3.digest);

        // Change permission
        let record4 = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .qty(100000000, -8)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_refuse(OrderRefuseReason::ExecutionDisabled {
                strategy_id: "spread_passive".to_string(),
            });

        assert_ne!(record1.digest, record4.digest);
    }

    #[test]
    fn test_permit_refuse_builders() {
        let permit = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();
        assert!(permit.is_permitted());
        assert!(!permit.is_refused());
        assert!(permit.refuse_reason.is_none());

        let refuse = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session")
            .seq(2)
            .side(OrderIntentSide::Sell)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_refuse(OrderRefuseReason::RiskLimitExceeded {
                limit_name: "max_position".to_string(),
            });
        assert!(!refuse.is_permitted());
        assert!(refuse.is_refused());
        assert!(refuse.refuse_reason.is_some());
    }

    #[test]
    fn test_refuse_reason_descriptions() {
        let r1 = OrderRefuseReason::SignalNotAdmitted {
            signal_id: "spread".to_string(),
        };
        assert!(r1.description().contains("spread"));
        assert!(r1.description().contains("L1"));

        let r2 = OrderRefuseReason::SignalNotPromoted {
            signal_id: "test".to_string(),
        };
        assert!(r2.description().contains("test"));
        assert!(r2.description().contains("promoted"));

        let r3 = OrderRefuseReason::StrategyNotBound {
            strategy_id: "strat".to_string(),
        };
        assert!(r3.description().contains("strat"));
        assert!(r3.description().contains("manifest"));

        let r4 = OrderRefuseReason::RiskLimitExceeded {
            limit_name: "max_notional".to_string(),
        };
        assert!(r4.description().contains("max_notional"));

        let r5 = OrderRefuseReason::PositionLimitExceeded {
            current: 100,
            max: 50,
        };
        assert!(r5.description().contains("100"));
        assert!(r5.description().contains("50"));

        let r6 = OrderRefuseReason::RateLimitExceeded {
            window_ms: 1000,
            count: 10,
            max: 5,
        };
        assert!(r6.description().contains("10"));
        assert!(r6.description().contains("5"));

        let r7 = OrderRefuseReason::ExecutionDisabled {
            strategy_id: "test".to_string(),
        };
        assert!(r7.description().contains("test"));
        assert!(r7.description().contains("disabled"));

        let r8 = OrderRefuseReason::Custom {
            reason: "custom reason".to_string(),
        };
        assert_eq!(r8.description(), "custom reason");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let record = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_1")
            .seq(42)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit)
            .qty(100000000, -8)
            .limit_price(5000000, -2)
            .correlation_id("corr_123")
            .parent_admission_digest("abc123")
            .build_refuse(OrderRefuseReason::PositionLimitExceeded {
                current: 100,
                max: 50,
            });

        let json = serde_json::to_string(&record).unwrap();
        let parsed: OrderIntentRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.schema_version, record.schema_version);
        assert_eq!(parsed.ts_ns, record.ts_ns);
        assert_eq!(parsed.session_id, record.session_id);
        assert_eq!(parsed.seq, record.seq);
        assert_eq!(parsed.strategy_id, record.strategy_id);
        assert_eq!(parsed.symbol, record.symbol);
        assert_eq!(parsed.side, record.side);
        assert_eq!(parsed.order_type, record.order_type);
        assert_eq!(parsed.qty_mantissa, record.qty_mantissa);
        assert_eq!(parsed.qty_exponent, record.qty_exponent);
        assert_eq!(parsed.limit_price_mantissa, record.limit_price_mantissa);
        assert_eq!(parsed.price_exponent, record.price_exponent);
        assert_eq!(parsed.permission, record.permission);
        assert_eq!(parsed.refuse_reason, record.refuse_reason);
        assert_eq!(parsed.correlation_id, record.correlation_id);
        assert_eq!(parsed.parent_admission_digest, record.parent_admission_digest);
        assert_eq!(parsed.digest, record.digest);
    }

    #[test]
    fn test_schema_version() {
        let record = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session")
            .seq(1)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();
        assert_eq!(record.schema_version, "1.0.0");
    }

    #[test]
    fn test_market_vs_limit_order() {
        // Market order (no limit price)
        let market = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Market)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();
        assert_eq!(market.order_type, OrderIntentType::Market);
        assert!(market.limit_price_mantissa.is_none());

        // Limit order (with limit price)
        let limit = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session")
            .seq(2)
            .side(OrderIntentSide::Sell)
            .qty(1000, -3)
            .limit_price(50000, -2) // This sets order_type to Limit
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();
        assert_eq!(limit.order_type, OrderIntentType::Limit);
        assert_eq!(limit.limit_price_mantissa, Some(50000));

        // Digests should differ
        assert_ne!(market.digest, limit.digest);
    }

    #[test]
    fn test_key_uniqueness_session_seq() {
        // Key is (session_id, seq) - same key should produce same digest if all else equal
        let r1 = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_A")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();

        let r2 = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_A")
            .seq(2) // Different seq
            .side(OrderIntentSide::Buy)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();

        let r3 = OrderIntentRecord::builder("strategy", "BTCUSDT")
            .ts_ns(1000000000)
            .session_id("session_B") // Different session
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(1000, -3)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();

        // All three should have different digests
        assert_ne!(r1.digest, r2.digest);
        assert_ne!(r1.digest, r3.digest);
        assert_ne!(r2.digest, r3.digest);
    }
}
