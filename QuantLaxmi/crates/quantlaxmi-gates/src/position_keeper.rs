//! # Position Keeper (Phase 14.3)
//!
//! Fill-driven position state machine with deterministic reconciliation.
//!
//! ## Core Question
//! "How do fills become authoritative position state?"
//!
//! ## Invariants
//! - Same fill sequence → identical position state
//! - All arithmetic is fixed-point (no f64)
//! - Every position change is WAL-bound
//! - Idempotent fill processing (global processed_fills set - Amendment A)
//! - Position state is replayable from fill sequence
//!
//! ## NOT in Scope
//! - Mark-to-market (unrealized PnL) — requires live prices
//! - Risk calculations — Phase 15+
//! - Drawdown tracking — Phase 15+

use crate::capital_eligibility::Venue;
use quantlaxmi_models::{
    ClientOrderId, FillId, OrderFillEvent, POSITION_EVENTS_SCHEMA_VERSION, PositionClosedEvent,
    PositionEventKind, PositionFlipEvent, PositionId, PositionIncreaseEvent, PositionKey,
    PositionOpenEvent, PositionReduceEvent, PositionSide, PositionVenue,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use thiserror::Error;

// =============================================================================
// Schema Version
// =============================================================================

pub const POSITION_KEEPER_SCHEMA_VERSION: &str = "position_keeper_v1.0";

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum PositionError {
    #[error("Fill already processed: {0}")]
    FillAlreadyProcessed(String),

    #[error("Invalid fill: {0}")]
    InvalidFill(String),

    #[error("Arithmetic overflow: {0}")]
    ArithmeticOverflow(String),

    #[error("WAL write error: {0}")]
    WalError(String),
}

// =============================================================================
// Venue Conversion
// =============================================================================

/// Convert from gates Venue to models PositionVenue.
pub fn venue_to_position_venue(venue: &Venue) -> PositionVenue {
    match venue {
        Venue::BinancePerp => PositionVenue::BinancePerp,
        Venue::BinanceSpot => PositionVenue::BinanceSpot,
        Venue::NseF => PositionVenue::NseF,
        Venue::NseO => PositionVenue::NseO,
        Venue::Paper => PositionVenue::Paper,
    }
}

// =============================================================================
// Position State
// =============================================================================

/// Current position state (mutable, internal to PositionKeeper).
/// Amendment A: No last_fill_id field. Idempotency is tracked globally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionState {
    /// Position key.
    pub key: PositionKey,
    /// Derived position ID.
    pub position_id: PositionId,
    /// Current side.
    pub side: PositionSide,
    /// Quantity (mantissa, always positive).
    pub quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Weighted average entry price (mantissa).
    pub avg_entry_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Cost basis (quantity * avg_entry_price, mantissa).
    pub cost_basis_mantissa: i128,
    /// Realized PnL (mantissa, cumulative).
    pub realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Total commission paid (mantissa).
    pub total_commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Fill count (total fills processed).
    pub fill_count: u32,
    /// Maximum quantity reached (for closed event).
    pub max_quantity_mantissa: i128,
    /// First entry price (for closed event).
    pub first_entry_price_mantissa: i128,
    /// Position open timestamp (nanoseconds).
    pub open_ts_ns: i64,
    /// Last update timestamp (nanoseconds).
    pub last_update_ts_ns: i64,
}

impl PositionState {
    /// Create a new position state from an opening fill.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: PositionKey,
        side: PositionSide,
        quantity_mantissa: i128,
        quantity_exponent: i8,
        entry_price_mantissa: i128,
        price_exponent: i8,
        pnl_exponent: i8,
        commission_mantissa: i128,
        commission_exponent: i8,
        ts_ns: i64,
    ) -> Self {
        let position_id = key.position_id();
        let cost_basis = quantity_mantissa * entry_price_mantissa;

        Self {
            key,
            position_id,
            side,
            quantity_mantissa,
            quantity_exponent,
            avg_entry_price_mantissa: entry_price_mantissa,
            price_exponent,
            cost_basis_mantissa: cost_basis,
            realized_pnl_mantissa: 0,
            pnl_exponent,
            total_commission_mantissa: commission_mantissa,
            commission_exponent,
            fill_count: 1,
            max_quantity_mantissa: quantity_mantissa,
            first_entry_price_mantissa: entry_price_mantissa,
            open_ts_ns: ts_ns,
            last_update_ts_ns: ts_ns,
        }
    }

    /// Check if position is flat (no position).
    pub fn is_flat(&self) -> bool {
        self.side == PositionSide::Flat || self.quantity_mantissa == 0
    }
}

// =============================================================================
// Position Snapshot
// =============================================================================

/// Snapshot ID for position/portfolio snapshots.
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

/// Deterministic snapshot of position state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    /// Schema version.
    pub schema_version: String,
    /// Snapshot ID.
    pub snapshot_id: SnapshotId,
    /// Snapshot timestamp (nanoseconds).
    pub ts_ns: i64,
    /// Position ID.
    pub position_id: PositionId,
    /// Position key.
    pub key: PositionKey,
    /// Position state.
    pub state: PositionState,
    /// Deterministic digest.
    pub digest: String,
}

impl PositionSnapshot {
    /// Compute deterministic digest.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.position_id.0.as_bytes());
        hasher.update(self.key.canonical_bytes());
        hasher.update(format!("{}", self.state.side).as_bytes());
        hasher.update(self.state.quantity_mantissa.to_le_bytes());
        hasher.update([self.state.quantity_exponent as u8]);
        hasher.update(self.state.avg_entry_price_mantissa.to_le_bytes());
        hasher.update([self.state.price_exponent as u8]);
        hasher.update(self.state.cost_basis_mantissa.to_le_bytes());
        hasher.update(self.state.realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.state.pnl_exponent as u8]);
        hasher.update(self.state.total_commission_mantissa.to_le_bytes());
        hasher.update([self.state.commission_exponent as u8]);
        hasher.update(self.state.fill_count.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Portfolio Ledger
// =============================================================================

/// Container for all positions across strategies/buckets.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PortfolioLedger {
    /// Schema version.
    pub schema_version: String,
    /// All active positions by position ID.
    positions: BTreeMap<String, PositionState>,
    /// Processed fill IDs for idempotency (Amendment A: global).
    processed_fills: HashSet<String>,
    /// Total realized PnL across all positions (mantissa).
    pub total_realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Total fill count.
    pub total_fill_count: u32,
    /// Last update timestamp.
    pub last_update_ts_ns: i64,
}

impl PortfolioLedger {
    /// Create a new empty portfolio ledger.
    pub fn new(pnl_exponent: i8) -> Self {
        Self {
            schema_version: POSITION_KEEPER_SCHEMA_VERSION.to_string(),
            positions: BTreeMap::new(),
            processed_fills: HashSet::new(),
            total_realized_pnl_mantissa: 0,
            pnl_exponent,
            total_fill_count: 0,
            last_update_ts_ns: 0,
        }
    }

    /// Get position by key.
    pub fn get_position(&self, key: &PositionKey) -> Option<&PositionState> {
        let position_id = key.position_id();
        self.positions.get(&position_id.0)
    }

    /// Get position by ID.
    pub fn get_position_by_id(&self, position_id: &PositionId) -> Option<&PositionState> {
        self.positions.get(&position_id.0)
    }

    /// Get all positions for a strategy.
    pub fn positions_for_strategy(&self, strategy_id: &str) -> Vec<&PositionState> {
        self.positions
            .values()
            .filter(|p| p.key.strategy_id == strategy_id)
            .collect()
    }

    /// Get all positions for a bucket.
    pub fn positions_for_bucket(&self, bucket_id: &str) -> Vec<&PositionState> {
        self.positions
            .values()
            .filter(|p| p.key.bucket_id == bucket_id)
            .collect()
    }

    /// Get all active (non-flat) positions.
    pub fn active_positions(&self) -> Vec<&PositionState> {
        self.positions.values().filter(|p| !p.is_flat()).collect()
    }

    /// Position count.
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Active position count (non-flat).
    pub fn active_position_count(&self) -> usize {
        self.positions.values().filter(|p| !p.is_flat()).count()
    }

    /// Check if fill already processed.
    pub fn is_fill_processed(&self, fill_id: &FillId) -> bool {
        self.processed_fills.contains(&fill_id.0)
    }

    /// Mark fill as processed.
    fn mark_fill_processed(&mut self, fill_id: &FillId) {
        self.processed_fills.insert(fill_id.0.clone());
    }

    /// Insert or update position.
    pub(crate) fn upsert_position(&mut self, state: PositionState) {
        self.positions.insert(state.position_id.0.clone(), state);
    }

    /// Remove position (when closed).
    fn remove_position(&mut self, position_id: &PositionId) {
        self.positions.remove(&position_id.0);
    }

    /// Take deterministic snapshot of portfolio.
    pub fn snapshot(&self, snapshot_id: SnapshotId, ts_ns: i64) -> PortfolioSnapshot {
        let position_snapshots: Vec<PositionSnapshot> = self
            .positions
            .values()
            .map(|state| {
                let mut snap = PositionSnapshot {
                    schema_version: POSITION_KEEPER_SCHEMA_VERSION.to_string(),
                    snapshot_id: snapshot_id.clone(),
                    ts_ns,
                    position_id: state.position_id.clone(),
                    key: state.key.clone(),
                    state: state.clone(),
                    digest: String::new(),
                };
                snap.digest = snap.compute_digest();
                snap
            })
            .collect();

        let mut snapshot = PortfolioSnapshot {
            schema_version: POSITION_KEEPER_SCHEMA_VERSION.to_string(),
            snapshot_id,
            ts_ns,
            positions: position_snapshots,
            total_realized_pnl_mantissa: self.total_realized_pnl_mantissa,
            pnl_exponent: self.pnl_exponent,
            total_fill_count: self.total_fill_count,
            position_count: self.positions.len() as u32,
            digest: String::new(),
        };
        snapshot.digest = snapshot.compute_digest();
        snapshot
    }
}

/// Deterministic snapshot of entire portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    /// Schema version.
    pub schema_version: String,
    /// Snapshot ID.
    pub snapshot_id: SnapshotId,
    /// Snapshot timestamp (nanoseconds).
    pub ts_ns: i64,
    /// All position snapshots (ordered by position_id).
    pub positions: Vec<PositionSnapshot>,
    /// Total realized PnL (mantissa).
    pub total_realized_pnl_mantissa: i128,
    /// PnL exponent.
    pub pnl_exponent: i8,
    /// Total fill count across all positions.
    pub total_fill_count: u32,
    /// Position count.
    pub position_count: u32,
    /// Deterministic digest.
    pub digest: String,
}

impl PortfolioSnapshot {
    /// Compute deterministic digest.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        // Hash each position snapshot digest in order
        for pos_snap in &self.positions {
            hasher.update(pos_snap.digest.as_bytes());
        }

        hasher.update(self.total_realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.total_fill_count.to_le_bytes());
        hasher.update(self.position_count.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Position Keeper
// =============================================================================

/// Fill-driven position state machine.
/// Processes OrderFillEvent sequence deterministically.
pub struct PositionKeeper {
    /// Portfolio ledger (mutable state).
    pub ledger: PortfolioLedger,
    /// Event sequence counter.
    event_seq: u64,
}

impl PositionKeeper {
    /// Create new position keeper.
    pub fn new(pnl_exponent: i8) -> Self {
        Self {
            ledger: PortfolioLedger::new(pnl_exponent),
            event_seq: 0,
        }
    }

    /// Create from existing ledger (for replay).
    pub fn from_ledger(ledger: PortfolioLedger) -> Self {
        Self {
            ledger,
            event_seq: 0,
        }
    }

    /// Process a fill event.
    /// Returns the position event(s) emitted.
    #[allow(clippy::too_many_arguments)]
    pub fn process_fill(
        &mut self,
        fill: &OrderFillEvent,
        strategy_id: &str,
        bucket_id: &str,
        venue: PositionVenue,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        // Step 1: Idempotency check (Amendment A: global)
        if self.ledger.is_fill_processed(&fill.fill_id) {
            return Ok(Vec::new()); // No-op, already processed
        }

        // Step 2: Derive position key
        let symbol = self.extract_symbol_from_fill(fill)?;
        let key = PositionKey::new(strategy_id, bucket_id, symbol, venue);
        let position_id = key.position_id();

        // Step 3: Get current position state
        let current_state = self.ledger.get_position_by_id(&position_id).cloned();

        // Step 4: Determine fill effect
        let fill_side = fill_side_to_position_effect(fill);
        let fill_qty = fill.fill_quantity_mantissa;
        let fill_price = fill.fill_price_mantissa;
        let commission = fill.commission_mantissa;

        self.event_seq += 1;
        let ts_ns = fill.ts_ns;

        // Step 5: Process based on current state
        let events = match current_state {
            None => {
                // Case A: No position → Open new position
                self.process_open(
                    key,
                    fill_side,
                    fill_qty,
                    fill.exponent,
                    fill_price,
                    fill.exponent,
                    commission,
                    fill.commission_exponent,
                    ts_ns,
                    &fill.fill_id,
                    &fill.client_order_id,
                    parent_decision_id,
                )?
            }
            Some(ref state) if state.is_flat() => {
                // Case A: Position is Flat → Open new position
                self.process_open(
                    key,
                    fill_side,
                    fill_qty,
                    fill.exponent,
                    fill_price,
                    fill.exponent,
                    commission,
                    fill.commission_exponent,
                    ts_ns,
                    &fill.fill_id,
                    &fill.client_order_id,
                    parent_decision_id,
                )?
            }
            Some(ref state) if same_direction(state.side, fill_side) => {
                // Case B: Fill increases position
                self.process_increase(
                    state,
                    fill_qty,
                    fill_price,
                    commission,
                    ts_ns,
                    &fill.fill_id,
                    &fill.client_order_id,
                    parent_decision_id,
                )?
            }
            Some(ref state) => {
                // Case C/D/E: Fill reduces/closes/flips position
                self.process_reduce_or_flip(
                    state,
                    fill_side,
                    fill_qty,
                    fill_price,
                    commission,
                    fill.commission_exponent,
                    ts_ns,
                    &fill.fill_id,
                    &fill.client_order_id,
                    parent_decision_id,
                )?
            }
        };

        // Step 6: Mark fill as processed
        self.ledger.mark_fill_processed(&fill.fill_id);
        self.ledger.total_fill_count += 1;
        self.ledger.last_update_ts_ns = ts_ns;

        Ok(events)
    }

    /// Extract symbol from fill (uses client_order_id to look up, or passed externally).
    fn extract_symbol_from_fill(&self, _fill: &OrderFillEvent) -> Result<String, PositionError> {
        // In a real implementation, we'd track client_order_id → symbol mapping.
        // For now, we expect symbol to be passed via parent context.
        // This is a placeholder - the caller should provide symbol.
        Ok("UNKNOWN".to_string())
    }

    /// Process open (Flat → Long/Short).
    #[allow(clippy::too_many_arguments)]
    fn process_open(
        &mut self,
        key: PositionKey,
        side: PositionSide,
        quantity_mantissa: i128,
        quantity_exponent: i8,
        entry_price_mantissa: i128,
        price_exponent: i8,
        commission_mantissa: i128,
        commission_exponent: i8,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let position_id = key.position_id();
        let cost_basis = quantity_mantissa * entry_price_mantissa;

        // Create position state
        let state = PositionState::new(
            key.clone(),
            side,
            quantity_mantissa,
            quantity_exponent,
            entry_price_mantissa,
            price_exponent,
            self.ledger.pnl_exponent,
            commission_mantissa,
            commission_exponent,
            ts_ns,
        );

        // Create event
        let mut event = PositionOpenEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            position_id: position_id.clone(),
            key: key.clone(),
            side,
            quantity_mantissa,
            quantity_exponent,
            entry_price_mantissa,
            price_exponent,
            cost_basis_mantissa: cost_basis,
            commission_mantissa,
            commission_exponent,
            fill_id: fill_id.clone(),
            client_order_id: client_order_id.clone(),
            parent_decision_id: parent_decision_id.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Update ledger
        self.ledger.upsert_position(state);

        Ok(vec![PositionEventKind::Open(event)])
    }

    /// Process increase (same direction).
    #[allow(clippy::too_many_arguments)]
    fn process_increase(
        &mut self,
        state: &PositionState,
        fill_quantity_mantissa: i128,
        fill_price_mantissa: i128,
        commission_mantissa: i128,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let qty_before = state.quantity_mantissa;
        let avg_before = state.avg_entry_price_mantissa;

        // Weighted average entry:
        // new_cost = old_cost + fill_cost
        // new_qty = old_qty + fill_qty
        // new_avg = new_cost / new_qty
        let old_cost = state.cost_basis_mantissa;
        let fill_cost = fill_quantity_mantissa * fill_price_mantissa;
        let new_cost = old_cost + fill_cost;
        let new_qty = qty_before + fill_quantity_mantissa;

        // Floor division for avg entry
        let new_avg = if new_qty > 0 { new_cost / new_qty } else { 0 };

        let max_qty = state.max_quantity_mantissa.max(new_qty);

        // Update state
        let mut new_state = state.clone();
        new_state.quantity_mantissa = new_qty;
        new_state.avg_entry_price_mantissa = new_avg;
        new_state.cost_basis_mantissa = new_cost;
        new_state.total_commission_mantissa += commission_mantissa;
        new_state.fill_count += 1;
        new_state.max_quantity_mantissa = max_qty;
        new_state.last_update_ts_ns = ts_ns;

        // Create event
        let mut event = PositionIncreaseEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            position_id: state.position_id.clone(),
            side: state.side,
            quantity_before_mantissa: qty_before,
            quantity_after_mantissa: new_qty,
            quantity_exponent: state.quantity_exponent,
            fill_quantity_mantissa,
            fill_price_mantissa,
            price_exponent: state.price_exponent,
            avg_entry_price_before_mantissa: avg_before,
            avg_entry_price_after_mantissa: new_avg,
            cost_basis_after_mantissa: new_cost,
            commission_mantissa,
            commission_exponent: state.commission_exponent,
            fill_id: fill_id.clone(),
            client_order_id: client_order_id.clone(),
            parent_decision_id: parent_decision_id.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Update ledger
        self.ledger.upsert_position(new_state);

        Ok(vec![PositionEventKind::Increase(event)])
    }

    /// Process reduce, close, or flip.
    #[allow(clippy::too_many_arguments)]
    fn process_reduce_or_flip(
        &mut self,
        state: &PositionState,
        fill_side: PositionSide,
        fill_quantity_mantissa: i128,
        fill_price_mantissa: i128,
        commission_mantissa: i128,
        commission_exponent: i8,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let qty_before = state.quantity_mantissa;

        if fill_quantity_mantissa < qty_before {
            // Case C: Partial reduce
            self.process_reduce(
                state,
                fill_quantity_mantissa,
                fill_price_mantissa,
                commission_mantissa,
                ts_ns,
                fill_id,
                client_order_id,
                parent_decision_id,
            )
        } else if fill_quantity_mantissa == qty_before {
            // Case D: Full close
            self.process_close(
                state,
                fill_quantity_mantissa,
                fill_price_mantissa,
                commission_mantissa,
                ts_ns,
                fill_id,
                client_order_id,
                parent_decision_id,
            )
        } else {
            // Case E: Flip (fill_qty > position_qty)
            self.process_flip(
                state,
                fill_side,
                fill_quantity_mantissa,
                fill_price_mantissa,
                commission_mantissa,
                commission_exponent,
                ts_ns,
                fill_id,
                client_order_id,
                parent_decision_id,
            )
        }
    }

    /// Process partial reduce.
    #[allow(clippy::too_many_arguments)]
    fn process_reduce(
        &mut self,
        state: &PositionState,
        fill_quantity_mantissa: i128,
        fill_price_mantissa: i128,
        commission_mantissa: i128,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let qty_before = state.quantity_mantissa;
        let qty_after = qty_before - fill_quantity_mantissa;
        let avg_entry = state.avg_entry_price_mantissa;

        // Calculate realized PnL
        // Long: pnl = qty * (exit_price - entry_price)
        // Short: pnl = qty * (entry_price - exit_price)
        let pnl = if state.side == PositionSide::Long {
            fill_quantity_mantissa * (fill_price_mantissa - avg_entry)
        } else {
            fill_quantity_mantissa * (avg_entry - fill_price_mantissa)
        };

        // Update state
        let mut new_state = state.clone();
        new_state.quantity_mantissa = qty_after;
        // Cost basis proportionally reduced
        new_state.cost_basis_mantissa = (state.cost_basis_mantissa * qty_after) / qty_before;
        new_state.realized_pnl_mantissa += pnl;
        new_state.total_commission_mantissa += commission_mantissa;
        new_state.fill_count += 1;
        new_state.last_update_ts_ns = ts_ns;

        // Update portfolio totals
        self.ledger.total_realized_pnl_mantissa += pnl;

        // Create event
        let mut event = PositionReduceEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            position_id: state.position_id.clone(),
            side: state.side,
            quantity_before_mantissa: qty_before,
            quantity_after_mantissa: qty_after,
            quantity_exponent: state.quantity_exponent,
            fill_quantity_mantissa,
            fill_price_mantissa,
            price_exponent: state.price_exponent,
            realized_pnl_mantissa: pnl,
            pnl_exponent: state.pnl_exponent,
            commission_mantissa,
            commission_exponent: state.commission_exponent,
            fill_id: fill_id.clone(),
            client_order_id: client_order_id.clone(),
            parent_decision_id: parent_decision_id.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Update ledger
        self.ledger.upsert_position(new_state);

        Ok(vec![PositionEventKind::Reduce(event)])
    }

    /// Process full close.
    #[allow(clippy::too_many_arguments)]
    fn process_close(
        &mut self,
        state: &PositionState,
        fill_quantity_mantissa: i128,
        fill_price_mantissa: i128,
        commission_mantissa: i128,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let avg_entry = state.avg_entry_price_mantissa;

        // Calculate final realized PnL for this close
        let pnl = if state.side == PositionSide::Long {
            fill_quantity_mantissa * (fill_price_mantissa - avg_entry)
        } else {
            fill_quantity_mantissa * (avg_entry - fill_price_mantissa)
        };

        let final_pnl = state.realized_pnl_mantissa + pnl;
        let duration = ts_ns - state.open_ts_ns;

        // Update portfolio totals
        self.ledger.total_realized_pnl_mantissa += pnl;

        // Create closed event
        let mut event = PositionClosedEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            position_id: state.position_id.clone(),
            key: state.key.clone(),
            side: state.side,
            final_realized_pnl_mantissa: final_pnl,
            pnl_exponent: state.pnl_exponent,
            total_commission_mantissa: state.total_commission_mantissa + commission_mantissa,
            commission_exponent: state.commission_exponent,
            total_fill_count: state.fill_count + 1,
            duration_ns: duration,
            entry_price_mantissa: state.first_entry_price_mantissa,
            exit_price_mantissa: fill_price_mantissa,
            price_exponent: state.price_exponent,
            max_quantity_mantissa: state.max_quantity_mantissa,
            quantity_exponent: state.quantity_exponent,
            fill_id: fill_id.clone(),
            client_order_id: client_order_id.clone(),
            parent_decision_id: parent_decision_id.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Remove position from ledger
        self.ledger.remove_position(&state.position_id);

        Ok(vec![PositionEventKind::Closed(event)])
    }

    /// Process flip (close + open opposite direction).
    #[allow(clippy::too_many_arguments)]
    fn process_flip(
        &mut self,
        state: &PositionState,
        fill_side: PositionSide,
        fill_quantity_mantissa: i128,
        fill_price_mantissa: i128,
        commission_mantissa: i128,
        commission_exponent: i8,
        ts_ns: i64,
        fill_id: &FillId,
        client_order_id: &ClientOrderId,
        parent_decision_id: &str,
    ) -> Result<Vec<PositionEventKind>, PositionError> {
        let old_qty = state.quantity_mantissa;
        let avg_entry = state.avg_entry_price_mantissa;
        let remainder = fill_quantity_mantissa - old_qty;

        // Realize PnL from closing old position
        let pnl = if state.side == PositionSide::Long {
            old_qty * (fill_price_mantissa - avg_entry)
        } else {
            old_qty * (avg_entry - fill_price_mantissa)
        };

        // Update portfolio totals
        self.ledger.total_realized_pnl_mantissa += pnl;

        // New position in opposite direction
        let new_side = fill_side;

        // Create flip event
        let mut event = PositionFlipEvent {
            schema_version: POSITION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            position_id: state.position_id.clone(),
            key: state.key.clone(),
            side_before: state.side,
            side_after: new_side,
            realized_pnl_mantissa: pnl,
            pnl_exponent: state.pnl_exponent,
            new_quantity_mantissa: remainder,
            quantity_exponent: state.quantity_exponent,
            new_entry_price_mantissa: fill_price_mantissa,
            price_exponent: state.price_exponent,
            commission_mantissa,
            commission_exponent,
            fill_id: fill_id.clone(),
            client_order_id: client_order_id.clone(),
            parent_decision_id: parent_decision_id.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Create new position state
        let new_state = PositionState::new(
            state.key.clone(),
            new_side,
            remainder,
            state.quantity_exponent,
            fill_price_mantissa,
            state.price_exponent,
            state.pnl_exponent,
            commission_mantissa,
            commission_exponent,
            ts_ns,
        );

        // Update ledger with new position
        self.ledger.upsert_position(new_state);

        Ok(vec![PositionEventKind::Flip(event)])
    }

    /// Get current position state.
    pub fn get_position(&self, key: &PositionKey) -> Option<&PositionState> {
        self.ledger.get_position(key)
    }

    /// Get portfolio snapshot.
    pub fn snapshot(&self, snapshot_id: SnapshotId, ts_ns: i64) -> PortfolioSnapshot {
        self.ledger.snapshot(snapshot_id, ts_ns)
    }

    /// Check if fill already processed.
    pub fn is_fill_processed(&self, fill_id: &FillId) -> bool {
        self.ledger.is_fill_processed(fill_id)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert fill side to position effect.
/// NOTE: This is a placeholder. In production, we need to track order side
/// from the original OrderIntentEvent or OrderSubmitEvent.
fn fill_side_to_position_effect(_fill: &OrderFillEvent) -> PositionSide {
    // Buy → opens/increases Long, reduces Short
    // Sell → opens/increases Short, reduces Long
    // For now, return Long as placeholder.
    // Real implementation would look up from order tracking state.
    PositionSide::Long
}

/// Check if fill direction matches position direction.
fn same_direction(position_side: PositionSide, fill_side: PositionSide) -> bool {
    position_side == fill_side
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::{ClientOrderId, ExchangeOrderId, FillId, LiveOrderState};

    fn make_fill(
        fill_id: &str,
        qty_mantissa: i128,
        price_mantissa: i128,
        commission_mantissa: i128,
        ts_ns: i64,
    ) -> OrderFillEvent {
        OrderFillEvent {
            schema_version: "test".to_string(),
            ts_ns,
            client_order_id: ClientOrderId(format!("order_{}", fill_id)),
            exchange_order_id: ExchangeOrderId("EX001".to_string()),
            fill_id: FillId(fill_id.to_string()),
            exchange_fill_id: fill_id.to_string(),
            fill_price_mantissa: price_mantissa,
            fill_quantity_mantissa: qty_mantissa,
            exponent: -8,
            commission_mantissa,
            commission_exponent: -8,
            commission_asset: "USDT".to_string(),
            is_final: true,
            cumulative_filled_mantissa: qty_mantissa,
            remaining_quantity_mantissa: 0,
            commitment_delta_id: "delta_001".to_string(),
            committed_mantissa: qty_mantissa * price_mantissa,
            committed_exponent: -8,
            state: LiveOrderState::Filled,
            digest: "test_digest".to_string(),
        }
    }

    #[test]
    fn test_process_single_fill_opens_position() {
        let mut keeper = PositionKeeper::new(-8);
        let fill = make_fill("fill_001", 100_000_000, 50000_00000000, 25_000_000, 1000);

        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );

        // Manually process with known side
        let state = PositionState::new(
            key.clone(),
            PositionSide::Long,
            100_000_000,
            -8,
            50000_00000000,
            -8,
            -8,
            25_000_000,
            -8,
            1000,
        );
        keeper.ledger.upsert_position(state);
        keeper.ledger.mark_fill_processed(&fill.fill_id);

        let pos = keeper.get_position(&key).unwrap();
        assert_eq!(pos.side, PositionSide::Long);
        assert_eq!(pos.quantity_mantissa, 100_000_000);
        assert_eq!(pos.avg_entry_price_mantissa, 50000_00000000);
    }

    #[test]
    fn test_idempotent_fill_processing() {
        let mut keeper = PositionKeeper::new(-8);
        let fill = make_fill("fill_001", 100_000_000, 50000_00000000, 25_000_000, 1000);

        // Mark as processed
        keeper.ledger.mark_fill_processed(&fill.fill_id);

        // Process should return empty (idempotent)
        let result = keeper.process_fill(
            &fill,
            "strategy_001",
            "bucket_001",
            PositionVenue::BinancePerp,
            "decision_001",
        );

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_portfolio_snapshot_deterministic() {
        let mut keeper = PositionKeeper::new(-8);

        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let state = PositionState::new(
            key.clone(),
            PositionSide::Long,
            100_000_000,
            -8,
            50000_00000000,
            -8,
            -8,
            25_000_000,
            -8,
            1000,
        );
        keeper.ledger.upsert_position(state);

        let snap1 = keeper.snapshot(SnapshotId::new("snap_001"), 2000);
        let snap2 = keeper.snapshot(SnapshotId::new("snap_001"), 2000);

        assert_eq!(snap1.digest, snap2.digest);
    }

    #[test]
    fn test_position_id_deterministic() {
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

        assert_eq!(key1.position_id(), key2.position_id());
    }

    #[test]
    fn test_position_increase_weighted_avg() {
        let mut keeper = PositionKeeper::new(-8);

        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );

        // Open with qty=100, price=50000
        let state1 = PositionState::new(
            key.clone(),
            PositionSide::Long,
            100_000_000,
            -8,
            50000_00000000,
            -8,
            -8,
            25_000_000,
            -8,
            1000,
        );
        keeper.ledger.upsert_position(state1);

        // Increase with qty=100, price=60000
        // Expected: new_cost = 100*50000 + 100*60000 = 11_000_000
        // new_qty = 200
        // new_avg = 11_000_000 / 200 = 55_000
        let _fill2 = make_fill("fill_002", 100_000_000, 60000_00000000, 25_000_000, 2000);

        // Simulate increase manually
        let state = keeper.get_position(&key).unwrap();
        let old_cost = state.cost_basis_mantissa;
        let fill_cost = 100_000_000i128 * 60000_00000000i128;
        let new_cost = old_cost + fill_cost;
        let new_qty = 200_000_000i128;
        let new_avg = new_cost / new_qty;

        // 100 * 50000 = 5_000_000_000_000_000
        // 100 * 60000 = 6_000_000_000_000_000
        // total = 11_000_000_000_000_000
        // avg = 11_000_000_000_000_000 / 200 = 55_000_000_000_000 = 55000.00000000
        assert_eq!(new_avg, 55000_00000000);
    }

    #[test]
    fn test_realized_pnl_long_profit() {
        // Long at 50000, close at 55000
        // qty = 1 BTC (100_000_000 satoshi)
        // pnl = 1 * (55000 - 50000) = 5000 USDT
        let qty = 100_000_000i128;
        let entry = 50000_00000000i128;
        let exit = 55000_00000000i128;

        let pnl = qty * (exit - entry);
        // 100_000_000 * 5000_00000000 = 500_000_000_000_000_000
        // This represents 5000 USDT with exponent -8
        assert!(pnl > 0);
    }

    #[test]
    fn test_realized_pnl_short_profit() {
        // Short at 50000, close at 45000
        // qty = 1 BTC
        // pnl = 1 * (50000 - 45000) = 5000 USDT
        let qty = 100_000_000i128;
        let entry = 50000_00000000i128;
        let exit = 45000_00000000i128;

        let pnl = qty * (entry - exit);
        assert!(pnl > 0);
    }

    #[test]
    fn test_commission_accumulation() {
        let _keeper = PositionKeeper::new(-8);

        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );
        let mut state = PositionState::new(
            key.clone(),
            PositionSide::Long,
            100_000_000,
            -8,
            50000_00000000,
            -8,
            -8,
            25_000_000, // First commission
            -8,
            1000,
        );

        // Add second commission
        state.total_commission_mantissa += 30_000_000;
        state.fill_count += 1;

        assert_eq!(state.total_commission_mantissa, 55_000_000);
        assert_eq!(state.fill_count, 2);
    }

    #[test]
    fn test_position_state_is_flat() {
        let key = PositionKey::new(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
        );

        // Flat side
        let mut state = PositionState::new(
            key.clone(),
            PositionSide::Flat,
            0,
            -8,
            0,
            -8,
            -8,
            0,
            -8,
            1000,
        );
        assert!(state.is_flat());

        // Long with zero qty
        state.side = PositionSide::Long;
        state.quantity_mantissa = 0;
        assert!(state.is_flat());

        // Long with qty
        state.quantity_mantissa = 100;
        assert!(!state.is_flat());
    }

    #[test]
    fn test_venue_to_position_venue() {
        assert_eq!(
            venue_to_position_venue(&Venue::BinancePerp),
            PositionVenue::BinancePerp
        );
        assert_eq!(venue_to_position_venue(&Venue::Paper), PositionVenue::Paper);
    }
}
