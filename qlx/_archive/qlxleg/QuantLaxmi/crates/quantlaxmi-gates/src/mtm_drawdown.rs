//! # Mark-to-Market & Drawdown Layer (Phase 15.2)
//!
//! Deterministic MTM valuation and drawdown tracking using WAL-sourced prices.
//!
//! ## Core Question
//! "Given portfolio positions and WAL-sourced prices, what is the current equity state,
//! and should execution proceed based on drawdown/leverage limits?"
//!
//! ## Hard Laws
//! - L1: Deterministic Replay — Prices MUST come from WAL (MarketSnapshot), never live REST
//! - L2: Fixed-Point Only — All PnL, equity, drawdown use i128 mantissa + i8 exponent (no f64)
//! - L3: Read-Only — MTM layer never mutates positions, budgets, or orders
//! - L4: Monotonic Peak — Peak equity only updates upward; drawdown relative to historical high
//! - L5: Audit-First — Every snapshot has deterministic SHA-256 digest
//!
//! ## Normalization Rule (frozen)
//! - raw_exp > target_exp: divide by 10^(raw - target) using floor division
//! - raw_exp < target_exp: multiply by 10^(target - raw)
//!
//! ## Stale Price Rule (frozen)
//! - Stale positions remain in MTM with last-known mark
//! - Increment stale_price_count, emit StalePositionWarning
//! - HALT if stale_price_count > max_stale_positions

use crate::position_keeper::{PortfolioSnapshot, PositionSnapshot};
use crate::risk_exposure::normalize_notional;
use quantlaxmi_models::PositionSide;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

// =============================================================================
// Schema Versions
// =============================================================================

pub const MTM_SNAPSHOT_SCHEMA_VERSION: &str = "mtm_snapshot_v1.0";
pub const DRAWDOWN_SNAPSHOT_SCHEMA_VERSION: &str = "drawdown_snapshot_v1.0";
pub const EQUITY_POLICY_SCHEMA_VERSION: &str = "equity_policy_v1.0";

// =============================================================================
// Price Source (WAL-bound)
// =============================================================================

/// Single price point from MarketSnapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    /// Mid price mantissa = (bid + ask) / 2 using floor division.
    pub mid_price_mantissa: i128,
    /// Price exponent.
    pub price_exponent: i8,
    /// Timestamp of the price update (nanoseconds).
    pub ts_ns: i64,
}

/// Price source for MTM valuation (deterministic, WAL-bound).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSource {
    /// Symbol -> PricePoint mapping.
    pub prices: BTreeMap<String, PricePoint>,
    /// Source session ID (for audit trail).
    pub session_id: String,
    /// Latest price timestamp across all symbols.
    pub latest_ts_ns: i64,
    /// Deterministic digest of price state.
    pub digest: String,
}

impl PriceSource {
    /// Create empty price source for session.
    pub fn new(session_id: &str) -> Self {
        let mut source = Self {
            prices: BTreeMap::new(),
            session_id: session_id.to_string(),
            latest_ts_ns: 0,
            digest: String::new(),
        };
        source.digest = source.compute_digest();
        source
    }

    /// Build from MarketSnapshot-like data (deterministic).
    /// MUST be called with events from WAL replay, never live data.
    ///
    /// Each snapshot provides: symbol, bid_mantissa, ask_mantissa, price_exponent, ts_ns
    #[allow(clippy::type_complexity)]
    pub fn from_market_data(
        snapshots: &[(String, Option<i128>, Option<i128>, i8, i64)],
        session_id: &str,
    ) -> Self {
        let mut source = Self::new(session_id);
        for (symbol, bid, ask, exp, ts_ns) in snapshots {
            source.update_price(symbol, *bid, *ask, *exp, *ts_ns);
        }
        source.digest = source.compute_digest();
        source
    }

    /// Update with new price data.
    /// Mid price computation (frozen rule):
    /// - Both sides: (bid + ask) / 2 using floor division
    /// - One side only: use that side
    /// - Neither side: no update for that symbol
    pub fn update_price(
        &mut self,
        symbol: &str,
        bid_mantissa: Option<i128>,
        ask_mantissa: Option<i128>,
        price_exponent: i8,
        ts_ns: i64,
    ) {
        let mid = match (bid_mantissa, ask_mantissa) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2), // Floor division
            (Some(bid), None) => Some(bid),
            (None, Some(ask)) => Some(ask),
            (None, None) => None,
        };

        if let Some(mid_price_mantissa) = mid {
            self.prices.insert(
                symbol.to_string(),
                PricePoint {
                    mid_price_mantissa,
                    price_exponent,
                    ts_ns,
                },
            );
            if ts_ns > self.latest_ts_ns {
                self.latest_ts_ns = ts_ns;
            }
        }
    }

    /// Recompute digest after updates.
    pub fn finalize(&mut self) {
        self.digest = self.compute_digest();
    }

    /// Get price for symbol.
    pub fn get_price(&self, symbol: &str) -> Option<&PricePoint> {
        self.prices.get(symbol)
    }

    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.session_id.as_bytes());
        hasher.update(self.latest_ts_ns.to_le_bytes());

        // Prices in sorted order (BTreeMap guarantees this)
        hasher.update((self.prices.len() as u32).to_le_bytes());
        for (symbol, point) in &self.prices {
            hasher.update((symbol.len() as u32).to_le_bytes());
            hasher.update(symbol.as_bytes());
            hasher.update(point.mid_price_mantissa.to_le_bytes());
            hasher.update([point.price_exponent as u8]);
            hasher.update(point.ts_ns.to_le_bytes());
        }

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Equity Policy
// =============================================================================

/// Policy for equity-based risk gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPolicy {
    /// Schema version.
    pub schema_version: String,

    /// Maximum allowed drawdown percentage (mantissa, exponent -4).
    /// e.g., 1000 with exp -4 = 10.00% max drawdown
    /// Breach triggers HALT.
    pub max_drawdown_pct_mantissa: i64,

    /// Warning drawdown percentage (triggers soft alert, no halt).
    pub warning_drawdown_pct_mantissa: i64,

    /// Drawdown percentage exponent (shared).
    pub drawdown_pct_exponent: i8,

    /// Maximum allowed leverage (mantissa, exponent -4).
    /// e.g., 30000 with exp -4 = 3.0x max leverage
    /// Breach triggers REJECT for new positions.
    pub max_leverage_mantissa: i64,

    /// Leverage exponent.
    pub leverage_exponent: i8,

    /// Equity floor (mantissa) — absolute minimum equity.
    /// Breach triggers HALT.
    pub equity_floor_mantissa: i128,

    /// Equity floor exponent.
    pub equity_floor_exponent: i8,

    /// Price staleness threshold (nanoseconds).
    pub staleness_threshold_ns: i64,

    /// Maximum allowed stale positions before HALT.
    pub max_stale_positions: u32,

    /// Policy fingerprint (derived, never user-supplied).
    pub fingerprint: String,
}

impl EquityPolicy {
    /// Compute deterministic fingerprint (SHA-256).
    pub fn compute_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.max_drawdown_pct_mantissa.to_le_bytes());
        hasher.update(self.warning_drawdown_pct_mantissa.to_le_bytes());
        hasher.update([self.drawdown_pct_exponent as u8]);
        hasher.update(self.max_leverage_mantissa.to_le_bytes());
        hasher.update([self.leverage_exponent as u8]);
        hasher.update(self.equity_floor_mantissa.to_le_bytes());
        hasher.update([self.equity_floor_exponent as u8]);
        hasher.update(self.staleness_threshold_ns.to_le_bytes());
        hasher.update(self.max_stale_positions.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Conservative preset: 5% max DD, 2x leverage, 10% equity floor.
    pub fn conservative(starting_capital_mantissa: i128, capital_exponent: i8) -> Self {
        let equity_floor = starting_capital_mantissa * 10 / 100; // 10% of starting
        let mut policy = Self {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 500,     // 5.00%
            warning_drawdown_pct_mantissa: 300, // 3.00%
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 20000, // 2.0x
            leverage_exponent: -4,
            equity_floor_mantissa: equity_floor,
            equity_floor_exponent: capital_exponent,
            staleness_threshold_ns: 60_000_000_000, // 60 seconds
            max_stale_positions: 3,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }

    /// Moderate preset: 10% max DD, 3x leverage, 5% equity floor.
    pub fn moderate(starting_capital_mantissa: i128, capital_exponent: i8) -> Self {
        let equity_floor = starting_capital_mantissa * 5 / 100; // 5% of starting
        let mut policy = Self {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 1000,    // 10.00%
            warning_drawdown_pct_mantissa: 700, // 7.00%
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 30000, // 3.0x
            leverage_exponent: -4,
            equity_floor_mantissa: equity_floor,
            equity_floor_exponent: capital_exponent,
            staleness_threshold_ns: 120_000_000_000, // 120 seconds
            max_stale_positions: 5,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }

    /// Aggressive preset: 20% max DD, 5x leverage, 2% equity floor.
    pub fn aggressive(starting_capital_mantissa: i128, capital_exponent: i8) -> Self {
        let equity_floor = starting_capital_mantissa * 2 / 100; // 2% of starting
        let mut policy = Self {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 2000,     // 20.00%
            warning_drawdown_pct_mantissa: 1500, // 15.00%
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 50000, // 5.0x
            leverage_exponent: -4,
            equity_floor_mantissa: equity_floor,
            equity_floor_exponent: capital_exponent,
            staleness_threshold_ns: 300_000_000_000, // 300 seconds
            max_stale_positions: 10,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }
}

// =============================================================================
// Equity Violation Types
// =============================================================================

/// Equity-based violation types.
/// Ordering: Halts first, then Rejects, then Warnings (lexicographic within category).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum EquityViolationType {
    // === HALT violations (category 0) ===
    /// Drawdown exceeds maximum (HALT).
    DrawdownBreach {
        current_pct_mantissa: i64,
        max_pct_mantissa: i64,
    },

    /// Equity below floor (HALT).
    EquityFloorBreach {
        current_mantissa: i128,
        floor_mantissa: i128,
    },

    /// Too many positions with stale prices (HALT).
    StalePriceBreach { stale_count: u32, max_allowed: u32 },

    // === REJECT violations (category 1) ===
    /// Leverage exceeds maximum (REJECT new positions).
    LeverageBreach {
        current_mantissa: i64,
        max_mantissa: i64,
    },

    // === WARNING violations (category 2) ===
    /// Drawdown exceeds warning threshold (soft).
    DrawdownWarning {
        current_pct_mantissa: i64,
        warning_pct_mantissa: i64,
    },

    /// Individual position has stale price (warning).
    StalePositionWarning {
        position_key: String,
        age_ns: i64,
        threshold_ns: i64,
    },
}

impl EquityViolationType {
    /// Get violation code string.
    pub fn code(&self) -> &'static str {
        match self {
            EquityViolationType::DrawdownBreach { .. } => "DRAWDOWN_BREACH",
            EquityViolationType::EquityFloorBreach { .. } => "EQUITY_FLOOR_BREACH",
            EquityViolationType::StalePriceBreach { .. } => "STALE_PRICE_BREACH",
            EquityViolationType::LeverageBreach { .. } => "LEVERAGE_BREACH",
            EquityViolationType::DrawdownWarning { .. } => "DRAWDOWN_WARNING",
            EquityViolationType::StalePositionWarning { .. } => "STALE_POSITION_WARNING",
        }
    }

    /// Check if this violation triggers HALT.
    pub fn is_halt(&self) -> bool {
        matches!(
            self,
            EquityViolationType::DrawdownBreach { .. }
                | EquityViolationType::EquityFloorBreach { .. }
                | EquityViolationType::StalePriceBreach { .. }
        )
    }

    /// Check if this violation triggers REJECT.
    pub fn is_reject(&self) -> bool {
        matches!(self, EquityViolationType::LeverageBreach { .. })
    }

    /// Check if this violation is a warning only.
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            EquityViolationType::DrawdownWarning { .. }
                | EquityViolationType::StalePositionWarning { .. }
        )
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.code().as_bytes());
        bytes.push(b':');

        match self {
            EquityViolationType::DrawdownBreach {
                current_pct_mantissa,
                max_pct_mantissa,
            } => {
                bytes.extend_from_slice(&current_pct_mantissa.to_le_bytes());
                bytes.extend_from_slice(&max_pct_mantissa.to_le_bytes());
            }
            EquityViolationType::EquityFloorBreach {
                current_mantissa,
                floor_mantissa,
            } => {
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&floor_mantissa.to_le_bytes());
            }
            EquityViolationType::StalePriceBreach {
                stale_count,
                max_allowed,
            } => {
                bytes.extend_from_slice(&stale_count.to_le_bytes());
                bytes.extend_from_slice(&max_allowed.to_le_bytes());
            }
            EquityViolationType::LeverageBreach {
                current_mantissa,
                max_mantissa,
            } => {
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&max_mantissa.to_le_bytes());
            }
            EquityViolationType::DrawdownWarning {
                current_pct_mantissa,
                warning_pct_mantissa,
            } => {
                bytes.extend_from_slice(&current_pct_mantissa.to_le_bytes());
                bytes.extend_from_slice(&warning_pct_mantissa.to_le_bytes());
            }
            EquityViolationType::StalePositionWarning {
                position_key,
                age_ns,
                threshold_ns,
            } => {
                bytes.extend_from_slice(&(position_key.len() as u32).to_le_bytes());
                bytes.extend_from_slice(position_key.as_bytes());
                bytes.extend_from_slice(&age_ns.to_le_bytes());
                bytes.extend_from_slice(&threshold_ns.to_le_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Position Valuation
// =============================================================================

/// Per-position MTM valuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionValuation {
    /// Position key (strategy:symbol).
    pub position_key: String,

    /// Current mark price (mantissa).
    pub mark_price_mantissa: i128,
    pub price_exponent: i8,

    /// Entry cost basis (mantissa) — from position's avg_entry_price * qty.
    pub cost_basis_mantissa: i128,

    /// Current notional value (mantissa) — mark_price * qty.
    pub notional_mantissa: i128,

    /// Unrealized PnL (mantissa).
    /// Long: (mark - entry) * qty
    /// Short: (entry - mark) * qty
    pub unrealized_pnl_mantissa: i128,

    /// Shared exponent for cost_basis, notional, unrealized_pnl.
    pub value_exponent: i8,

    /// Price staleness (ns since last price update).
    pub price_age_ns: i64,

    /// Is this position considered stale?
    pub is_stale: bool,
}

impl PositionValuation {
    /// Compute valuation for a position.
    ///
    /// Normalization rule (frozen):
    /// - raw_exp = qty_exp + price_exp
    /// - normalize to value_exponent via floor division
    pub fn compute(
        position: &PositionSnapshot,
        mark_price: &PricePoint,
        value_exponent: i8,
        current_ts_ns: i64,
        staleness_threshold_ns: i64,
    ) -> Self {
        let position_key = format!("{}:{}", position.key.strategy_id, position.key.symbol);
        let state = &position.state;

        // Check if flat (no position)
        if state.is_flat() {
            return Self {
                position_key,
                mark_price_mantissa: mark_price.mid_price_mantissa,
                price_exponent: mark_price.price_exponent,
                cost_basis_mantissa: 0,
                notional_mantissa: 0,
                unrealized_pnl_mantissa: 0,
                value_exponent,
                price_age_ns: 0,
                is_stale: false,
            };
        }

        // Get position state fields
        let qty_mantissa = state.quantity_mantissa;
        let qty_exponent = state.quantity_exponent;
        let entry_price_mantissa = state.avg_entry_price_mantissa;
        let entry_price_exponent = state.price_exponent;

        // Determine if short for PnL calculation
        let is_short = state.side == PositionSide::Short;

        // Compute cost basis: entry_price * qty
        let raw_cost = entry_price_mantissa.saturating_mul(qty_mantissa.abs());
        let raw_cost_exp = entry_price_exponent as i32 + qty_exponent as i32;
        let cost_basis_mantissa = normalize_notional(raw_cost, raw_cost_exp as i8, value_exponent);

        // Compute notional: mark_price * qty
        let raw_notional = mark_price
            .mid_price_mantissa
            .saturating_mul(qty_mantissa.abs());
        let raw_notional_exp = mark_price.price_exponent as i32 + qty_exponent as i32;
        let notional_mantissa =
            normalize_notional(raw_notional, raw_notional_exp as i8, value_exponent);

        // Compute unrealized PnL
        // Long: (mark - entry) * qty = notional - cost_basis
        // Short: (entry - mark) * qty = cost_basis - notional
        let unrealized_pnl_mantissa = if is_short {
            cost_basis_mantissa - notional_mantissa
        } else {
            notional_mantissa - cost_basis_mantissa
        };

        // Compute price age
        let price_age_ns = (current_ts_ns - mark_price.ts_ns).max(0);
        let is_stale = price_age_ns > staleness_threshold_ns;

        Self {
            position_key,
            mark_price_mantissa: mark_price.mid_price_mantissa,
            price_exponent: mark_price.price_exponent,
            cost_basis_mantissa,
            notional_mantissa,
            unrealized_pnl_mantissa,
            value_exponent,
            price_age_ns,
            is_stale,
        }
    }
}

// =============================================================================
// MTM Metrics
// =============================================================================

/// Aggregated MTM metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtmMetrics {
    /// Total unrealized PnL (mantissa).
    pub total_unrealized_pnl_mantissa: i128,

    /// Total realized PnL (mantissa) — from closed positions in session.
    pub total_realized_pnl_mantissa: i128,

    /// Total PnL = unrealized + realized.
    pub total_pnl_mantissa: i128,

    /// PnL exponent (shared).
    pub pnl_exponent: i8,

    /// Starting capital (mantissa) — from session config.
    pub starting_capital_mantissa: i128,

    /// Current equity = starting_capital + total_pnl.
    pub equity_mantissa: i128,

    /// Equity exponent.
    pub equity_exponent: i8,

    /// Total notional exposure (mantissa).
    pub total_notional_mantissa: i128,
    pub notional_exponent: i8,

    /// Leverage = total_notional / equity (mantissa, exponent -4).
    /// e.g., 25000 with exp -4 = 2.5x leverage
    pub leverage_mantissa: i64,
    pub leverage_exponent: i8,

    /// Count of positions with stale prices.
    pub stale_price_count: u32,

    /// Staleness threshold (ns).
    pub staleness_threshold_ns: i64,
}

// =============================================================================
// MTM Snapshot
// =============================================================================

/// Unique MTM snapshot ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MtmSnapshotId(pub String);

impl MtmSnapshotId {
    /// Derive from portfolio_digest + price_digest + ts_ns.
    pub fn derive(portfolio_digest: &str, price_digest: &str, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"mtm_snapshot:");
        hasher.update(portfolio_digest.as_bytes());
        hasher.update(b":");
        hasher.update(price_digest.as_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for MtmSnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Mark-to-market snapshot of portfolio valuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtmSnapshot {
    /// Schema version.
    pub schema_version: String,

    /// Unique snapshot ID (deterministic).
    pub snapshot_id: MtmSnapshotId,

    /// Timestamp (nanoseconds).
    pub ts_ns: i64,

    /// Source portfolio snapshot digest.
    pub portfolio_snapshot_digest: String,

    /// Source price digest.
    pub price_source_digest: String,

    /// Per-position valuations (sorted by position_key).
    pub position_valuations: BTreeMap<String, PositionValuation>,

    /// Aggregated metrics.
    pub metrics: MtmMetrics,

    /// Deterministic digest.
    pub digest: String,
}

impl MtmSnapshot {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.portfolio_snapshot_digest.as_bytes());
        hasher.update(self.price_source_digest.as_bytes());

        // Position valuations (BTreeMap is sorted)
        hasher.update((self.position_valuations.len() as u32).to_le_bytes());
        for (key, val) in &self.position_valuations {
            hasher.update((key.len() as u32).to_le_bytes());
            hasher.update(key.as_bytes());
            hasher.update(val.mark_price_mantissa.to_le_bytes());
            hasher.update([val.price_exponent as u8]);
            hasher.update(val.cost_basis_mantissa.to_le_bytes());
            hasher.update(val.notional_mantissa.to_le_bytes());
            hasher.update(val.unrealized_pnl_mantissa.to_le_bytes());
            hasher.update([val.value_exponent as u8]);
            hasher.update(val.price_age_ns.to_le_bytes());
            hasher.update([val.is_stale as u8]);
        }

        // Metrics
        hasher.update(self.metrics.total_unrealized_pnl_mantissa.to_le_bytes());
        hasher.update(self.metrics.total_realized_pnl_mantissa.to_le_bytes());
        hasher.update(self.metrics.total_pnl_mantissa.to_le_bytes());
        hasher.update([self.metrics.pnl_exponent as u8]);
        hasher.update(self.metrics.starting_capital_mantissa.to_le_bytes());
        hasher.update(self.metrics.equity_mantissa.to_le_bytes());
        hasher.update([self.metrics.equity_exponent as u8]);
        hasher.update(self.metrics.total_notional_mantissa.to_le_bytes());
        hasher.update([self.metrics.notional_exponent as u8]);
        hasher.update(self.metrics.leverage_mantissa.to_le_bytes());
        hasher.update([self.metrics.leverage_exponent as u8]);
        hasher.update(self.metrics.stale_price_count.to_le_bytes());
        hasher.update(self.metrics.staleness_threshold_ns.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Drawdown Snapshot
// =============================================================================

/// Unique drawdown snapshot ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DrawdownSnapshotId(pub String);

impl DrawdownSnapshotId {
    /// Derive from mtm_digest + peak + ts_ns.
    pub fn derive(mtm_digest: &str, peak_mantissa: i128, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"drawdown_snapshot:");
        hasher.update(mtm_digest.as_bytes());
        hasher.update(b":");
        hasher.update(peak_mantissa.to_le_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for DrawdownSnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Drawdown tracking snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownSnapshot {
    /// Schema version.
    pub schema_version: String,

    /// Unique snapshot ID (deterministic).
    pub snapshot_id: DrawdownSnapshotId,

    /// Timestamp (nanoseconds).
    pub ts_ns: i64,

    /// Source MTM snapshot digest.
    pub mtm_snapshot_digest: String,

    /// Peak equity observed (mantissa) — monotonically increasing.
    pub peak_equity_mantissa: i128,

    /// Timestamp when peak was established.
    pub peak_ts_ns: i64,

    /// Current equity (mantissa).
    pub current_equity_mantissa: i128,

    /// Equity exponent (shared).
    pub equity_exponent: i8,

    /// Current drawdown (mantissa) = peak - current (always >= 0).
    pub drawdown_mantissa: i128,

    /// Current drawdown percentage (mantissa, exponent -4).
    /// = (peak - current) / peak * 10000
    pub drawdown_pct_mantissa: i64,
    pub drawdown_pct_exponent: i8,

    /// Maximum drawdown observed in session (mantissa).
    pub max_drawdown_mantissa: i128,

    /// Maximum drawdown percentage (mantissa, exponent -4).
    pub max_drawdown_pct_mantissa: i64,

    /// Timestamp of max drawdown.
    pub max_drawdown_ts_ns: i64,

    /// Deterministic digest.
    pub digest: String,
}

impl DrawdownSnapshot {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.mtm_snapshot_digest.as_bytes());
        hasher.update(self.peak_equity_mantissa.to_le_bytes());
        hasher.update(self.peak_ts_ns.to_le_bytes());
        hasher.update(self.current_equity_mantissa.to_le_bytes());
        hasher.update([self.equity_exponent as u8]);
        hasher.update(self.drawdown_mantissa.to_le_bytes());
        hasher.update(self.drawdown_pct_mantissa.to_le_bytes());
        hasher.update([self.drawdown_pct_exponent as u8]);
        hasher.update(self.max_drawdown_mantissa.to_le_bytes());
        hasher.update(self.max_drawdown_pct_mantissa.to_le_bytes());
        hasher.update(self.max_drawdown_ts_ns.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// MTM Evaluator
// =============================================================================

/// Mark-to-market evaluator (deterministic, read-only).
pub struct MtmEvaluator {
    /// Equity policy.
    policy: EquityPolicy,

    /// Starting capital for equity calculation.
    starting_capital_mantissa: i128,
    #[allow(dead_code)]
    starting_capital_exponent: i8,

    /// Accumulated realized PnL from closed positions.
    realized_pnl_mantissa: i128,

    /// Value exponent for all calculations.
    value_exponent: i8,

    /// Peak equity tracker (monotonically increasing).
    peak_equity_mantissa: i128,
    peak_ts_ns: i64,

    /// Max drawdown tracker.
    max_drawdown_mantissa: i128,
    max_drawdown_pct_mantissa: i64,
    max_drawdown_ts_ns: i64,

    /// Snapshot sequence counter.
    snapshot_seq: u64,
}

impl MtmEvaluator {
    /// Create new evaluator.
    pub fn new(
        policy: EquityPolicy,
        starting_capital_mantissa: i128,
        starting_capital_exponent: i8,
    ) -> Self {
        Self {
            policy,
            starting_capital_mantissa,
            starting_capital_exponent,
            realized_pnl_mantissa: 0,
            value_exponent: starting_capital_exponent,
            peak_equity_mantissa: starting_capital_mantissa,
            peak_ts_ns: 0,
            max_drawdown_mantissa: 0,
            max_drawdown_pct_mantissa: 0,
            max_drawdown_ts_ns: 0,
            snapshot_seq: 0,
        }
    }

    /// Get current policy.
    pub fn policy(&self) -> &EquityPolicy {
        &self.policy
    }

    /// Record realized PnL from position close event.
    /// Source: PositionClosedEvent / PositionFlipEvent only (frozen rule).
    pub fn record_realized_pnl(&mut self, pnl_mantissa: i128, pnl_exponent: i8) {
        let normalized = normalize_notional(pnl_mantissa, pnl_exponent, self.value_exponent);
        self.realized_pnl_mantissa += normalized;
    }

    /// Compute MTM snapshot from portfolio and prices.
    /// DETERMINISTIC: same portfolio + prices → identical snapshot.
    pub fn compute_mtm_snapshot(
        &mut self,
        portfolio: &PortfolioSnapshot,
        prices: &PriceSource,
        ts_ns: i64,
    ) -> MtmSnapshot {
        self.snapshot_seq += 1;

        let mut position_valuations = BTreeMap::new();
        let mut total_unrealized_pnl: i128 = 0;
        let mut total_notional: i128 = 0;
        let mut stale_count: u32 = 0;

        // Compute valuations for all positions
        for position in &portfolio.positions {
            if position.state.is_flat() {
                continue;
            }

            if let Some(price) = prices.get_price(&position.key.symbol) {
                let valuation = PositionValuation::compute(
                    position,
                    price,
                    self.value_exponent,
                    ts_ns,
                    self.policy.staleness_threshold_ns,
                );

                total_unrealized_pnl += valuation.unrealized_pnl_mantissa;
                total_notional += valuation.notional_mantissa.abs();

                if valuation.is_stale {
                    stale_count += 1;
                }

                let position_key = format!("{}:{}", position.key.strategy_id, position.key.symbol);
                position_valuations.insert(position_key, valuation);
            }
            // If no price, position is not valued (missing price)
        }

        // Compute metrics
        let total_pnl = total_unrealized_pnl + self.realized_pnl_mantissa;
        let equity = self.starting_capital_mantissa + total_pnl;

        // Leverage = notional / equity (with exp -4 for 4 decimals)
        let leverage_mantissa = if equity > 0 {
            ((total_notional * 10000) / equity) as i64
        } else {
            0
        };

        let metrics = MtmMetrics {
            total_unrealized_pnl_mantissa: total_unrealized_pnl,
            total_realized_pnl_mantissa: self.realized_pnl_mantissa,
            total_pnl_mantissa: total_pnl,
            pnl_exponent: self.value_exponent,
            starting_capital_mantissa: self.starting_capital_mantissa,
            equity_mantissa: equity,
            equity_exponent: self.value_exponent,
            total_notional_mantissa: total_notional,
            notional_exponent: self.value_exponent,
            leverage_mantissa,
            leverage_exponent: -4,
            stale_price_count: stale_count,
            staleness_threshold_ns: self.policy.staleness_threshold_ns,
        };

        let snapshot_id = MtmSnapshotId::derive(&portfolio.digest, &prices.digest, ts_ns);

        let mut snapshot = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id,
            ts_ns,
            portfolio_snapshot_digest: portfolio.digest.clone(),
            price_source_digest: prices.digest.clone(),
            position_valuations,
            metrics,
            digest: String::new(),
        };
        snapshot.digest = snapshot.compute_digest();

        snapshot
    }

    /// Compute drawdown snapshot from MTM snapshot.
    /// Updates peak/max drawdown trackers.
    pub fn compute_drawdown_snapshot(&mut self, mtm: &MtmSnapshot, ts_ns: i64) -> DrawdownSnapshot {
        let current_equity = mtm.metrics.equity_mantissa;

        // Update peak (monotonically increasing only)
        if current_equity > self.peak_equity_mantissa {
            self.peak_equity_mantissa = current_equity;
            self.peak_ts_ns = ts_ns;
        }

        // Compute current drawdown
        let drawdown = (self.peak_equity_mantissa - current_equity).max(0);

        // Compute drawdown percentage (with exp -4)
        let drawdown_pct = if self.peak_equity_mantissa > 0 {
            ((drawdown * 10000) / self.peak_equity_mantissa) as i64
        } else {
            0
        };

        // Update max drawdown
        if drawdown > self.max_drawdown_mantissa {
            self.max_drawdown_mantissa = drawdown;
            self.max_drawdown_pct_mantissa = drawdown_pct;
            self.max_drawdown_ts_ns = ts_ns;
        }

        let snapshot_id = DrawdownSnapshotId::derive(&mtm.digest, self.peak_equity_mantissa, ts_ns);

        let mut snapshot = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id,
            ts_ns,
            mtm_snapshot_digest: mtm.digest.clone(),
            peak_equity_mantissa: self.peak_equity_mantissa,
            peak_ts_ns: self.peak_ts_ns,
            current_equity_mantissa: current_equity,
            equity_exponent: mtm.metrics.equity_exponent,
            drawdown_mantissa: drawdown,
            drawdown_pct_mantissa: drawdown_pct,
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: self.max_drawdown_mantissa,
            max_drawdown_pct_mantissa: self.max_drawdown_pct_mantissa,
            max_drawdown_ts_ns: self.max_drawdown_ts_ns,
            digest: String::new(),
        };
        snapshot.digest = snapshot.compute_digest();

        snapshot
    }

    /// Evaluate equity violations from drawdown and MTM snapshots.
    /// Returns violations in stable order: Halts first, then Rejects, then Warnings.
    pub fn evaluate_violations(
        &self,
        drawdown: &DrawdownSnapshot,
        mtm: &MtmSnapshot,
    ) -> Vec<EquityViolationType> {
        let mut violations = Vec::new();

        // Check drawdown breach (HALT)
        if drawdown.drawdown_pct_mantissa > self.policy.max_drawdown_pct_mantissa {
            violations.push(EquityViolationType::DrawdownBreach {
                current_pct_mantissa: drawdown.drawdown_pct_mantissa,
                max_pct_mantissa: self.policy.max_drawdown_pct_mantissa,
            });
        }

        // Check equity floor (HALT)
        let equity_floor_normalized = normalize_notional(
            self.policy.equity_floor_mantissa,
            self.policy.equity_floor_exponent,
            mtm.metrics.equity_exponent,
        );
        if mtm.metrics.equity_mantissa < equity_floor_normalized {
            violations.push(EquityViolationType::EquityFloorBreach {
                current_mantissa: mtm.metrics.equity_mantissa,
                floor_mantissa: equity_floor_normalized,
            });
        }

        // Check stale price breach (HALT)
        if mtm.metrics.stale_price_count > self.policy.max_stale_positions {
            violations.push(EquityViolationType::StalePriceBreach {
                stale_count: mtm.metrics.stale_price_count,
                max_allowed: self.policy.max_stale_positions,
            });
        }

        // Check leverage breach (REJECT)
        if mtm.metrics.leverage_mantissa > self.policy.max_leverage_mantissa {
            violations.push(EquityViolationType::LeverageBreach {
                current_mantissa: mtm.metrics.leverage_mantissa,
                max_mantissa: self.policy.max_leverage_mantissa,
            });
        }

        // Check drawdown warning (WARNING)
        if drawdown.drawdown_pct_mantissa > self.policy.warning_drawdown_pct_mantissa
            && drawdown.drawdown_pct_mantissa <= self.policy.max_drawdown_pct_mantissa
        {
            violations.push(EquityViolationType::DrawdownWarning {
                current_pct_mantissa: drawdown.drawdown_pct_mantissa,
                warning_pct_mantissa: self.policy.warning_drawdown_pct_mantissa,
            });
        }

        // Add stale position warnings
        for (key, val) in &mtm.position_valuations {
            if val.is_stale {
                violations.push(EquityViolationType::StalePositionWarning {
                    position_key: key.clone(),
                    age_ns: val.price_age_ns,
                    threshold_ns: self.policy.staleness_threshold_ns,
                });
            }
        }

        // Sort for deterministic ordering
        violations.sort();

        violations
    }

    /// Check if execution should be halted.
    pub fn should_halt(&self, violations: &[EquityViolationType]) -> bool {
        violations.iter().any(|v| v.is_halt())
    }

    /// Check if new position should be rejected.
    pub fn should_reject_new_position(&self, violations: &[EquityViolationType]) -> bool {
        violations.iter().any(|v| v.is_reject() || v.is_halt())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PositionState;
    use crate::position_keeper::{PositionKeeper, SnapshotId};
    use quantlaxmi_models::{PositionKey, PositionSide, PositionVenue};

    fn create_test_position_state(
        strategy_id: &str,
        symbol: &str,
        venue: PositionVenue,
        side: PositionSide,
        qty_mantissa: i128,
        entry_price_mantissa: i128,
        ts_ns: i64,
    ) -> PositionState {
        let key = PositionKey::new(strategy_id, "bucket_001", symbol, venue);
        PositionState::new(
            key,
            side,
            qty_mantissa,
            -8, // qty_exponent
            entry_price_mantissa,
            -8,         // price_exponent
            -8,         // pnl_exponent
            25_000_000, // commission
            -8,         // commission_exponent
            ts_ns,
        )
    }

    fn create_test_portfolio() -> PortfolioSnapshot {
        let mut keeper = PositionKeeper::new(-8);

        // Add a long position
        let state = create_test_position_state(
            "strategy_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
            PositionSide::Long,
            100_000_000,    // 1 BTC
            50000_00000000, // $50,000 entry
            1000000000000000000,
        );
        keeper.ledger.upsert_position(state);

        keeper.snapshot(SnapshotId::new("snap_001"), 1000000000000000000)
    }

    fn create_test_prices() -> PriceSource {
        let mut source = PriceSource::new("test_session");
        // BTC at $51,000 (2% gain)
        source.update_price(
            "BTCUSDT",
            Some(51000_00000000),
            Some(51000_00000000),
            -8,
            1000000000000000000,
        );
        source.finalize();
        source
    }

    #[test]
    fn test_price_source_from_snapshots() {
        let snapshots = vec![
            (
                "BTCUSDT".to_string(),
                Some(50000_00000000i128),
                Some(50100_00000000i128),
                -8i8,
                1000i64,
            ),
            (
                "ETHUSDT".to_string(),
                Some(3000_00000000i128),
                Some(3010_00000000i128),
                -8i8,
                1001i64,
            ),
        ];
        let source = PriceSource::from_market_data(&snapshots, "test_session");

        assert_eq!(source.prices.len(), 2);
        assert!(source.get_price("BTCUSDT").is_some());
        assert!(source.get_price("ETHUSDT").is_some());
        assert_eq!(source.latest_ts_ns, 1001);
    }

    #[test]
    fn test_price_source_digest_deterministic() {
        let snapshots = vec![(
            "BTCUSDT".to_string(),
            Some(50000_00000000i128),
            Some(50100_00000000i128),
            -8i8,
            1000i64,
        )];
        let source1 = PriceSource::from_market_data(&snapshots, "test_session");
        let source2 = PriceSource::from_market_data(&snapshots, "test_session");

        assert_eq!(source1.digest, source2.digest);
    }

    #[test]
    fn test_price_mid_computation_floor_division() {
        let mut source = PriceSource::new("test");
        // Odd sum: 101 + 102 = 203, floor(203/2) = 101
        source.update_price("TEST", Some(101), Some(102), -8, 1000);
        source.finalize();

        let price = source.get_price("TEST").unwrap();
        assert_eq!(price.mid_price_mantissa, 101); // Floor division
    }

    #[test]
    fn test_position_valuation_long() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let position = portfolio.positions.first().unwrap();
        let price = prices.get_price("BTCUSDT").unwrap();

        let valuation =
            PositionValuation::compute(position, price, -8, 1000000000000000000, 60_000_000_000);

        // Entry at $50,000, mark at $51,000, 1 BTC
        // Unrealized PnL = (51000 - 50000) * 1 = $1,000
        assert!(valuation.unrealized_pnl_mantissa > 0);
        assert!(!valuation.is_stale);
    }

    #[test]
    fn test_position_valuation_short() {
        let mut keeper = PositionKeeper::new(-8);

        // Add a short position
        let state = create_test_position_state(
            "strategy_001",
            "BTCUSDT",
            PositionVenue::BinancePerp,
            PositionSide::Short,
            100_000_000,    // 1 BTC short
            50000_00000000, // $50,000 entry
            1000000000000000000,
        );
        keeper.ledger.upsert_position(state);

        let portfolio = keeper.snapshot(SnapshotId::new("snap_001"), 1000000000000000000);
        let position = portfolio.positions.first().unwrap();

        // Price went up to $51,000 - short loses
        let mut source = PriceSource::new("test");
        source.update_price(
            "BTCUSDT",
            Some(51000_00000000),
            Some(51000_00000000),
            -8,
            1000000000000000000,
        );
        source.finalize();
        let price = source.get_price("BTCUSDT").unwrap();

        let valuation =
            PositionValuation::compute(position, price, -8, 1000000000000000000, 60_000_000_000);

        // Short at $50,000, mark at $51,000 -> loss
        assert!(valuation.unrealized_pnl_mantissa < 0);
    }

    #[test]
    fn test_mtm_snapshot_digest_deterministic() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let mtm1 = evaluator.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);

        // Reset and compute again
        let policy2 = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator2 = MtmEvaluator::new(policy2, 100000_00000000, -8);
        let mtm2 = evaluator2.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);

        assert_eq!(mtm1.digest, mtm2.digest);
    }

    #[test]
    fn test_mtm_metrics_aggregation() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);

        assert!(mtm.metrics.total_unrealized_pnl_mantissa > 0);
        assert_eq!(mtm.metrics.starting_capital_mantissa, 100000_00000000);
        assert!(mtm.metrics.equity_mantissa > mtm.metrics.starting_capital_mantissa);
    }

    #[test]
    fn test_leverage_calculation() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);

        // 1 BTC at $51,000 = $51,000 notional
        // Equity ~= $100,000 + unrealized
        // Leverage should be approximately 0.5x
        assert!(mtm.metrics.leverage_mantissa > 0);
        assert!(mtm.metrics.leverage_mantissa < 10000); // Less than 1x
    }

    #[test]
    fn test_stale_price_detection() {
        let portfolio = create_test_portfolio();

        let mut source = PriceSource::new("test");
        // Old price (ts=0)
        source.update_price("BTCUSDT", Some(51000_00000000), Some(51000_00000000), -8, 0);
        source.finalize();

        // Use moderate policy which has 120 second staleness threshold
        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        // Current time is 200 seconds later (> 120 second threshold)
        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &source, 200_000_000_000);

        assert_eq!(mtm.metrics.stale_price_count, 1);
    }

    #[test]
    fn test_drawdown_from_peak() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);
        let dd = evaluator.compute_drawdown_snapshot(&mtm, 1000000000000000000);

        // No drawdown yet (equity at peak)
        assert_eq!(dd.drawdown_mantissa, 0);
        assert_eq!(dd.drawdown_pct_mantissa, 0);
    }

    #[test]
    fn test_peak_monotonic() {
        let portfolio = create_test_portfolio();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        // First snapshot with high price
        let mut source1 = PriceSource::new("test");
        source1.update_price(
            "BTCUSDT",
            Some(52000_00000000),
            Some(52000_00000000),
            -8,
            1000,
        );
        source1.finalize();
        let mtm1 = evaluator.compute_mtm_snapshot(&portfolio, &source1, 1000);
        let dd1 = evaluator.compute_drawdown_snapshot(&mtm1, 1000);
        let peak1 = dd1.peak_equity_mantissa;

        // Second snapshot with lower price
        let mut source2 = PriceSource::new("test");
        source2.update_price(
            "BTCUSDT",
            Some(50000_00000000),
            Some(50000_00000000),
            -8,
            2000,
        );
        source2.finalize();
        let mtm2 = evaluator.compute_mtm_snapshot(&portfolio, &source2, 2000);
        let dd2 = evaluator.compute_drawdown_snapshot(&mtm2, 2000);

        // Peak should NOT decrease
        assert_eq!(dd2.peak_equity_mantissa, peak1);
        assert!(dd2.drawdown_mantissa > 0);
    }

    #[test]
    fn test_max_drawdown_tracking() {
        let portfolio = create_test_portfolio();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        // Create drawdown
        let mut source = PriceSource::new("test");
        source.update_price(
            "BTCUSDT",
            Some(45000_00000000),
            Some(45000_00000000),
            -8,
            1000,
        );
        source.finalize();
        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &source, 1000);
        let dd = evaluator.compute_drawdown_snapshot(&mtm, 1000);

        assert!(dd.max_drawdown_mantissa > 0);
        assert!(dd.max_drawdown_pct_mantissa > 0);
    }

    #[test]
    fn test_drawdown_snapshot_digest_deterministic() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator1 = MtmEvaluator::new(policy.clone(), 100000_00000000, -8);
        let mtm1 = evaluator1.compute_mtm_snapshot(&portfolio, &prices, 1000);
        let dd1 = evaluator1.compute_drawdown_snapshot(&mtm1, 1000);

        let mut evaluator2 = MtmEvaluator::new(policy, 100000_00000000, -8);
        let mtm2 = evaluator2.compute_mtm_snapshot(&portfolio, &prices, 1000);
        let dd2 = evaluator2.compute_drawdown_snapshot(&mtm2, 1000);

        assert_eq!(dd1.digest, dd2.digest);
    }

    #[test]
    fn test_drawdown_breach_halt() {
        let policy = EquityPolicy {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 500, // 5%
            warning_drawdown_pct_mantissa: 300,
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 30000,
            leverage_exponent: -4,
            equity_floor_mantissa: 5000_00000000,
            equity_floor_exponent: -8,
            staleness_threshold_ns: 60_000_000_000,
            max_stale_positions: 5,
            fingerprint: String::new(),
        };

        let evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        // Create a drawdown snapshot with 10% drawdown
        let dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("test".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "test".to_string(),
            peak_equity_mantissa: 100000_00000000,
            peak_ts_ns: 500,
            current_equity_mantissa: 90000_00000000,
            equity_exponent: -8,
            drawdown_mantissa: 10000_00000000,
            drawdown_pct_mantissa: 1000, // 10%
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 10000_00000000,
            max_drawdown_pct_mantissa: 1000,
            max_drawdown_ts_ns: 1000,
            digest: String::new(),
        };

        let mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("test".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "test".to_string(),
            price_source_digest: "test".to_string(),
            position_valuations: BTreeMap::new(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: -10000_00000000,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: -10000_00000000,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 90000_00000000,
                equity_exponent: -8,
                total_notional_mantissa: 50000_00000000,
                notional_exponent: -8,
                leverage_mantissa: 5555, // ~0.55x
                leverage_exponent: -4,
                stale_price_count: 0,
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };

        let violations = evaluator.evaluate_violations(&dd, &mtm);

        assert!(
            violations
                .iter()
                .any(|v| matches!(v, EquityViolationType::DrawdownBreach { .. }))
        );
        assert!(evaluator.should_halt(&violations));
    }

    #[test]
    fn test_leverage_breach_reject() {
        let policy = EquityPolicy {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 1000,
            warning_drawdown_pct_mantissa: 700,
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 20000, // 2x
            leverage_exponent: -4,
            equity_floor_mantissa: 5000_00000000,
            equity_floor_exponent: -8,
            staleness_threshold_ns: 60_000_000_000,
            max_stale_positions: 5,
            fingerprint: String::new(),
        };

        let evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("test".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "test".to_string(),
            peak_equity_mantissa: 100000_00000000,
            peak_ts_ns: 1000,
            current_equity_mantissa: 100000_00000000,
            equity_exponent: -8,
            drawdown_mantissa: 0,
            drawdown_pct_mantissa: 0,
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 0,
            max_drawdown_pct_mantissa: 0,
            max_drawdown_ts_ns: 0,
            digest: String::new(),
        };

        let mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("test".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "test".to_string(),
            price_source_digest: "test".to_string(),
            position_valuations: BTreeMap::new(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: 0,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: 0,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 100000_00000000,
                equity_exponent: -8,
                total_notional_mantissa: 300000_00000000, // 3x leverage
                notional_exponent: -8,
                leverage_mantissa: 30000, // 3x
                leverage_exponent: -4,
                stale_price_count: 0,
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };

        let violations = evaluator.evaluate_violations(&dd, &mtm);

        assert!(
            violations
                .iter()
                .any(|v| matches!(v, EquityViolationType::LeverageBreach { .. }))
        );
        assert!(evaluator.should_reject_new_position(&violations));
        assert!(!evaluator.should_halt(&violations));
    }

    #[test]
    fn test_equity_floor_breach_halt() {
        let policy = EquityPolicy {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 5000, // 50%
            warning_drawdown_pct_mantissa: 4000,
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 50000,
            leverage_exponent: -4,
            equity_floor_mantissa: 50000_00000000, // $50,000 floor
            equity_floor_exponent: -8,
            staleness_threshold_ns: 60_000_000_000,
            max_stale_positions: 5,
            fingerprint: String::new(),
        };

        let evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("test".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "test".to_string(),
            peak_equity_mantissa: 100000_00000000,
            peak_ts_ns: 500,
            current_equity_mantissa: 40000_00000000, // Below floor
            equity_exponent: -8,
            drawdown_mantissa: 60000_00000000,
            drawdown_pct_mantissa: 6000, // 60%
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 60000_00000000,
            max_drawdown_pct_mantissa: 6000,
            max_drawdown_ts_ns: 1000,
            digest: String::new(),
        };

        let mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("test".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "test".to_string(),
            price_source_digest: "test".to_string(),
            position_valuations: BTreeMap::new(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: -60000_00000000,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: -60000_00000000,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 40000_00000000,
                equity_exponent: -8,
                total_notional_mantissa: 50000_00000000,
                notional_exponent: -8,
                leverage_mantissa: 12500,
                leverage_exponent: -4,
                stale_price_count: 0,
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };

        let violations = evaluator.evaluate_violations(&dd, &mtm);

        assert!(
            violations
                .iter()
                .any(|v| matches!(v, EquityViolationType::EquityFloorBreach { .. }))
        );
        assert!(evaluator.should_halt(&violations));
    }

    #[test]
    fn test_stale_price_breach_halt() {
        let policy = EquityPolicy {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 1000,
            warning_drawdown_pct_mantissa: 700,
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 30000,
            leverage_exponent: -4,
            equity_floor_mantissa: 5000_00000000,
            equity_floor_exponent: -8,
            staleness_threshold_ns: 60_000_000_000,
            max_stale_positions: 2, // Only allow 2 stale
            fingerprint: String::new(),
        };

        let evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("test".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "test".to_string(),
            peak_equity_mantissa: 100000_00000000,
            peak_ts_ns: 1000,
            current_equity_mantissa: 100000_00000000,
            equity_exponent: -8,
            drawdown_mantissa: 0,
            drawdown_pct_mantissa: 0,
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 0,
            max_drawdown_pct_mantissa: 0,
            max_drawdown_ts_ns: 0,
            digest: String::new(),
        };

        let mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("test".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "test".to_string(),
            price_source_digest: "test".to_string(),
            position_valuations: BTreeMap::new(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: 0,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: 0,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 100000_00000000,
                equity_exponent: -8,
                total_notional_mantissa: 50000_00000000,
                notional_exponent: -8,
                leverage_mantissa: 5000,
                leverage_exponent: -4,
                stale_price_count: 5, // More than allowed
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };

        let violations = evaluator.evaluate_violations(&dd, &mtm);

        assert!(
            violations
                .iter()
                .any(|v| matches!(v, EquityViolationType::StalePriceBreach { .. }))
        );
        assert!(evaluator.should_halt(&violations));
    }

    #[test]
    fn test_equity_policy_fingerprint() {
        let policy1 = EquityPolicy::moderate(100000_00000000, -8);
        let policy2 = EquityPolicy::moderate(100000_00000000, -8);
        let policy3 = EquityPolicy::aggressive(100000_00000000, -8);

        assert_eq!(policy1.fingerprint, policy2.fingerprint);
        assert_ne!(policy1.fingerprint, policy3.fingerprint);
    }

    #[test]
    fn test_evaluate_order_with_equity() {
        let portfolio = create_test_portfolio();
        let prices = create_test_prices();

        let policy = EquityPolicy::moderate(100000_00000000, -8);
        let mut evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        let mtm = evaluator.compute_mtm_snapshot(&portfolio, &prices, 1000000000000000000);
        let dd = evaluator.compute_drawdown_snapshot(&mtm, 1000000000000000000);

        let violations = evaluator.evaluate_violations(&dd, &mtm);

        // No violations expected with healthy portfolio
        assert!(!evaluator.should_halt(&violations));
        assert!(!evaluator.should_reject_new_position(&violations));
    }

    #[test]
    fn test_stable_violation_ordering_deterministic() {
        let policy = EquityPolicy {
            schema_version: EQUITY_POLICY_SCHEMA_VERSION.to_string(),
            max_drawdown_pct_mantissa: 500,
            warning_drawdown_pct_mantissa: 300,
            drawdown_pct_exponent: -4,
            max_leverage_mantissa: 20000,
            leverage_exponent: -4,
            equity_floor_mantissa: 90000_00000000,
            equity_floor_exponent: -8,
            staleness_threshold_ns: 60_000_000_000,
            max_stale_positions: 0,
            fingerprint: String::new(),
        };

        let evaluator = MtmEvaluator::new(policy, 100000_00000000, -8);

        // Create snapshot that triggers multiple violations
        let mut valuations = BTreeMap::new();
        valuations.insert(
            "strat:BTC".to_string(),
            PositionValuation {
                position_key: "strat:BTC".to_string(),
                mark_price_mantissa: 50000_00000000,
                price_exponent: -8,
                cost_basis_mantissa: 55000_00000000,
                notional_mantissa: 50000_00000000,
                unrealized_pnl_mantissa: -5000_00000000,
                value_exponent: -8,
                price_age_ns: 120_000_000_000, // Stale
                is_stale: true,
            },
        );

        let dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("test".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "test".to_string(),
            peak_equity_mantissa: 100000_00000000,
            peak_ts_ns: 500,
            current_equity_mantissa: 85000_00000000,
            equity_exponent: -8,
            drawdown_mantissa: 15000_00000000,
            drawdown_pct_mantissa: 1500, // 15% > 5% max
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 15000_00000000,
            max_drawdown_pct_mantissa: 1500,
            max_drawdown_ts_ns: 1000,
            digest: String::new(),
        };

        let mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("test".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "test".to_string(),
            price_source_digest: "test".to_string(),
            position_valuations: valuations.clone(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: -5000_00000000,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: -5000_00000000,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 85000_00000000,
                equity_exponent: -8,
                total_notional_mantissa: 250000_00000000, // 2.9x leverage
                notional_exponent: -8,
                leverage_mantissa: 29411,
                leverage_exponent: -4,
                stale_price_count: 1,
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };

        let violations1 = evaluator.evaluate_violations(&dd, &mtm);
        let violations2 = evaluator.evaluate_violations(&dd, &mtm);

        // Same ordering
        assert_eq!(violations1.len(), violations2.len());
        for (v1, v2) in violations1.iter().zip(violations2.iter()) {
            assert_eq!(v1.code(), v2.code());
        }

        // Halts come before rejects come before warnings
        let codes: Vec<&str> = violations1.iter().map(|v| v.code()).collect();
        let halt_idx = codes.iter().position(|c| *c == "DRAWDOWN_BREACH");
        let reject_idx = codes.iter().position(|c| *c == "LEVERAGE_BREACH");
        let warn_idx = codes.iter().position(|c| *c == "STALE_POSITION_WARNING");

        if let (Some(h), Some(r)) = (halt_idx, reject_idx) {
            assert!(h < r);
        }
        if let (Some(r), Some(w)) = (reject_idx, warn_idx) {
            assert!(r < w);
        }
    }
}
