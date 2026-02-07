//! # Position Events (Phase 14.3)
//!
//! Canonical event types for position lifecycle.
//!
//! ## Core Question
//! "How do fills become authoritative position state with deterministic reconciliation?"
//!
//! ## Event Types
//! - `PositionOpenEvent` — first fill in new direction
//! - `PositionIncreaseEvent` — same direction fill
//! - `PositionReduceEvent` — opposite direction fill (partial)
//! - `PositionClosedEvent` — reduced to zero
//! - `PositionFlipEvent` — crossed zero to opposite side
//!
//! ## Invariants
//! - All arithmetic is fixed-point (no f64)
//! - All IDs are derived deterministically (SHA-256)
//! - All events have deterministic digests

use crate::{ClientOrderId, FillId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const POSITION_EVENTS_SCHEMA_VERSION: &str = "position_events_v1.0";

// =============================================================================
// Venue Enum (Canonical - matches capital_eligibility)
// =============================================================================

/// Trading venue for position isolation.
/// Must match the Venue enum in quantlaxmi-gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PositionVenue {
    /// Binance perpetual futures
    BinancePerp,
    /// Binance spot
    BinanceSpot,
    /// NSE India futures
    NseF,
    /// NSE India options
    NseO,
    /// Paper trading (any venue simulation)
    Paper,
}

impl std::fmt::Display for PositionVenue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionVenue::BinancePerp => write!(f, "BinancePerp"),
            PositionVenue::BinanceSpot => write!(f, "BinanceSpot"),
            PositionVenue::NseF => write!(f, "NseF"),
            PositionVenue::NseO => write!(f, "NseO"),
            PositionVenue::Paper => write!(f, "Paper"),
        }
    }
}

// =============================================================================
// Position Key
// =============================================================================

/// Unique identifier for a position.
/// A position is uniquely identified by strategy + bucket + symbol + venue.
/// Amendment B: venue is an enum, not a string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositionKey {
    /// Strategy owning this position.
    pub strategy_id: String,
    /// Bucket providing capital.
    pub bucket_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Trading venue (enum, not string - Amendment B).
    pub venue: PositionVenue,
}

impl PositionKey {
    /// Create a new position key.
    pub fn new(
        strategy_id: impl Into<String>,
        bucket_id: impl Into<String>,
        symbol: impl Into<String>,
        venue: PositionVenue,
    ) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            bucket_id: bucket_id.into(),
            symbol: symbol.into(),
            venue,
        }
    }

    /// Canonical bytes for hashing (deterministic ordering).
    /// Format: len(strategy_id) + strategy_id + len(bucket_id) + bucket_id + len(symbol) + symbol + venue_byte
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Strategy ID: length-prefixed
        bytes.extend_from_slice(&(self.strategy_id.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.strategy_id.as_bytes());

        // Bucket ID: length-prefixed
        bytes.extend_from_slice(&(self.bucket_id.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.bucket_id.as_bytes());

        // Symbol: length-prefixed
        bytes.extend_from_slice(&(self.symbol.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.symbol.as_bytes());

        // Venue: single byte discriminant
        let venue_byte = match self.venue {
            PositionVenue::BinancePerp => 0u8,
            PositionVenue::BinanceSpot => 1u8,
            PositionVenue::NseF => 2u8,
            PositionVenue::NseO => 3u8,
            PositionVenue::Paper => 4u8,
        };
        bytes.push(venue_byte);

        bytes
    }

    /// Derive deterministic position ID (SHA-256).
    pub fn position_id(&self) -> PositionId {
        PositionId::derive(self)
    }
}

impl std::fmt::Display for PositionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}",
            self.strategy_id, self.bucket_id, self.symbol, self.venue
        )
    }
}

// =============================================================================
// Position ID
// =============================================================================

/// Deterministic position identifier derived from PositionKey.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositionId(pub String);

impl PositionId {
    /// Derive from PositionKey (SHA-256 hex).
    pub fn derive(key: &PositionKey) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"position:");
        hasher.update(key.canonical_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for PositionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Position Side
// =============================================================================

/// Direction of position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Long => write!(f, "LONG"),
            PositionSide::Short => write!(f, "SHORT"),
            PositionSide::Flat => write!(f, "FLAT"),
        }
    }
}

// =============================================================================
// Snapshot ID
// =============================================================================

/// Unique identifier for a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(pub String);

impl SnapshotId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from timestamp and sequence.
    pub fn derive(ts_ns: i64, seq: u64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"snapshot:");
        hasher.update(ts_ns.to_le_bytes());
        hasher.update(b":");
        hasher.update(seq.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Position Events
// =============================================================================

/// Position opened (first fill in new direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionOpenEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds since epoch).
    pub ts_ns: i64,
    /// Deterministic position ID.
    pub position_id: PositionId,
    /// Position key.
    pub key: PositionKey,
    /// Position side (Long or Short).
    pub side: PositionSide,
    /// Quantity (mantissa, always positive).
    pub quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Entry price (mantissa).
    pub entry_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Cost basis (quantity * entry_price, mantissa).
    pub cost_basis_mantissa: i128,
    /// Commission (mantissa).
    pub commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Fill that opened this position.
    pub fill_id: FillId,
    /// Client order ID.
    pub client_order_id: ClientOrderId,
    /// Parent decision ID for correlation.
    pub parent_decision_id: String,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionOpenEvent {
    /// Compute deterministic digest.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(self.key.canonical_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(self.quantity_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        hasher.update(self.entry_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.cost_basis_mantissa.to_le_bytes());
        hasher.update(self.commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Position increased (same direction fill).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionIncreaseEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Position ID.
    pub position_id: PositionId,
    /// Position side.
    pub side: PositionSide,
    /// Quantity before fill (mantissa).
    pub quantity_before_mantissa: i128,
    /// Quantity after fill (mantissa).
    pub quantity_after_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Fill quantity (mantissa).
    pub fill_quantity_mantissa: i128,
    /// Fill price (mantissa).
    pub fill_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Average entry price before (mantissa).
    pub avg_entry_price_before_mantissa: i128,
    /// Average entry price after (mantissa).
    pub avg_entry_price_after_mantissa: i128,
    /// Cost basis after (mantissa).
    pub cost_basis_after_mantissa: i128,
    /// Commission (mantissa).
    pub commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Fill that increased this position.
    pub fill_id: FillId,
    /// Client order ID.
    pub client_order_id: ClientOrderId,
    /// Parent decision ID.
    pub parent_decision_id: String,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionIncreaseEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(self.quantity_before_mantissa.to_le_bytes());
        hasher.update(self.quantity_after_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        hasher.update(self.fill_quantity_mantissa.to_le_bytes());
        hasher.update(self.fill_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.avg_entry_price_before_mantissa.to_le_bytes());
        hasher.update(self.avg_entry_price_after_mantissa.to_le_bytes());
        hasher.update(self.cost_basis_after_mantissa.to_le_bytes());
        hasher.update(self.commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Position reduced (opposite direction fill, partial).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionReduceEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Position ID.
    pub position_id: PositionId,
    /// Position side.
    pub side: PositionSide,
    /// Quantity before (mantissa).
    pub quantity_before_mantissa: i128,
    /// Quantity after (mantissa).
    pub quantity_after_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Fill quantity (mantissa).
    pub fill_quantity_mantissa: i128,
    /// Fill price (mantissa).
    pub fill_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Realized PnL from this reduction (mantissa).
    pub realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Commission (mantissa).
    pub commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Fill that reduced this position.
    pub fill_id: FillId,
    /// Client order ID.
    pub client_order_id: ClientOrderId,
    /// Parent decision ID.
    pub parent_decision_id: String,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionReduceEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(self.quantity_before_mantissa.to_le_bytes());
        hasher.update(self.quantity_after_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        hasher.update(self.fill_quantity_mantissa.to_le_bytes());
        hasher.update(self.fill_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Position closed (reduced to zero).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionClosedEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Position ID.
    pub position_id: PositionId,
    /// Position key.
    pub key: PositionKey,
    /// Position side that was closed.
    pub side: PositionSide,
    /// Final realized PnL (mantissa).
    pub final_realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Total commission across position lifetime (mantissa).
    pub total_commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Total fills in this position.
    pub total_fill_count: u32,
    /// Position duration (nanoseconds).
    pub duration_ns: i64,
    /// Entry price (first fill, mantissa).
    pub entry_price_mantissa: i128,
    /// Exit price (last fill, mantissa).
    pub exit_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Maximum quantity reached (mantissa).
    pub max_quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Fill that closed this position.
    pub fill_id: FillId,
    /// Client order ID.
    pub client_order_id: ClientOrderId,
    /// Parent decision ID.
    pub parent_decision_id: String,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionClosedEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(self.key.canonical_bytes());
        hasher.update(format!("{}", self.side).as_bytes());
        hasher.update(self.final_realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.total_commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.total_fill_count.to_le_bytes());
        hasher.update(self.duration_ns.to_le_bytes());
        hasher.update(self.entry_price_mantissa.to_le_bytes());
        hasher.update(self.exit_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.max_quantity_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Position flipped (crossed zero to opposite side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionFlipEvent {
    /// Schema version.
    pub schema_version: String,
    /// Event timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Position ID.
    pub position_id: PositionId,
    /// Position key.
    pub key: PositionKey,
    /// Side before flip.
    pub side_before: PositionSide,
    /// Side after flip.
    pub side_after: PositionSide,
    /// Realized PnL from closing old position (mantissa).
    pub realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// New position quantity after flip (mantissa).
    pub new_quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// New position entry price (mantissa).
    pub new_entry_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Commission for this fill (mantissa).
    pub commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Fill that caused the flip.
    pub fill_id: FillId,
    /// Client order ID.
    pub client_order_id: ClientOrderId,
    /// Parent decision ID.
    pub parent_decision_id: String,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionFlipEvent {
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(self.key.canonical_bytes());
        hasher.update(format!("{}", self.side_before).as_bytes());
        hasher.update(format!("{}", self.side_after).as_bytes());
        hasher.update(self.realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.new_quantity_mantissa.to_le_bytes());
        hasher.update([self.quantity_exponent as u8]);
        hasher.update(self.new_entry_price_mantissa.to_le_bytes());
        hasher.update([self.price_exponent as u8]);
        hasher.update(self.commission_mantissa.to_le_bytes());
        hasher.update([self.commission_exponent as u8]);
        hasher.update(self.fill_id.0.as_bytes());
        hasher.update(self.client_order_id.0.as_bytes());
        hasher.update(self.parent_decision_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Position event variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionEventKind {
    Open(PositionOpenEvent),
    Increase(PositionIncreaseEvent),
    Reduce(PositionReduceEvent),
    Closed(PositionClosedEvent),
    Flip(PositionFlipEvent),
}

impl PositionEventKind {
    /// Get the position ID for this event.
    pub fn position_id(&self) -> &PositionId {
        match self {
            PositionEventKind::Open(e) => &e.position_id,
            PositionEventKind::Increase(e) => &e.position_id,
            PositionEventKind::Reduce(e) => &e.position_id,
            PositionEventKind::Closed(e) => &e.position_id,
            PositionEventKind::Flip(e) => &e.position_id,
        }
    }

    /// Get the fill ID that caused this event.
    pub fn fill_id(&self) -> &FillId {
        match self {
            PositionEventKind::Open(e) => &e.fill_id,
            PositionEventKind::Increase(e) => &e.fill_id,
            PositionEventKind::Reduce(e) => &e.fill_id,
            PositionEventKind::Closed(e) => &e.fill_id,
            PositionEventKind::Flip(e) => &e.fill_id,
        }
    }

    /// Get the timestamp of this event.
    pub fn ts_ns(&self) -> i64 {
        match self {
            PositionEventKind::Open(e) => e.ts_ns,
            PositionEventKind::Increase(e) => e.ts_ns,
            PositionEventKind::Reduce(e) => e.ts_ns,
            PositionEventKind::Closed(e) => e.ts_ns,
            PositionEventKind::Flip(e) => e.ts_ns,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_key_canonical_bytes_deterministic() {
        let key1 = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let key2 = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );

        assert_eq!(key1.canonical_bytes(), key2.canonical_bytes());
    }

    #[test]
    fn test_position_key_different_venue_different_bytes() {
        let key1 = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let key2 = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinanceSpot,
        );

        assert_ne!(key1.canonical_bytes(), key2.canonical_bytes());
    }

    #[test]
    fn test_position_id_deterministic() {
        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let id1 = PositionId::derive(&key);
        let id2 = PositionId::derive(&key);

        assert_eq!(id1, id2);
        assert_eq!(id1.0.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_position_id_different_key_different_id() {
        let key1 = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let key2 = PositionKey::new(
            "strategy_002",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );

        assert_ne!(PositionId::derive(&key1), PositionId::derive(&key2));
    }

    #[test]
    fn test_snapshot_id_deterministic() {
        let id1 = SnapshotId::derive(1234567890, 1);
        let id2 = SnapshotId::derive(1234567890, 1);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_position_open_event_digest_deterministic() {
        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let event = PositionOpenEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns: 1_234_567_890_000_000_000,
            position_id: key.position_id(),
            key: key.clone(),
            side: PositionSide::Long,
            quantity_mantissa: 100_000_000,
            quantity_exponent: -8,
            entry_price_mantissa: 50000_00000000,
            price_exponent: -8,
            cost_basis_mantissa: 5_000_000_000_000_000,
            commission_mantissa: 25_000_000,
            commission_exponent: -8,
            fill_id: FillId("fill_001".to_string()),
            client_order_id: ClientOrderId("order_001".to_string()),
            parent_decision_id: "decision_001".to_string(),
            digest: String::new(),
        };

        let digest1 = event.compute_digest();
        let digest2 = event.compute_digest();
        assert_eq!(digest1, digest2);
    }

    #[test]
    fn test_position_side_display() {
        assert_eq!(format!("{}", PositionSide::Long), "LONG");
        assert_eq!(format!("{}", PositionSide::Short), "SHORT");
        assert_eq!(format!("{}", PositionSide::Flat), "FLAT");
    }

    #[test]
    fn test_position_venue_display() {
        assert_eq!(format!("{}", PositionVenue::BinancePerp), "BinancePerp");
        assert_eq!(format!("{}", PositionVenue::Paper), "Paper");
    }
}
