//! # Trading Event Models Module
//!
//! Defines core event types for the QuantLaxmi trading system.
//!
//! ## Description
//! This module provides the canonical data structures for all events flowing
//! through the system: market data, signals, orders, and portfolio updates.
//! All types implement `Serialize`/`Deserialize` for persistence and IPC.
//!
//! ## Event Hierarchy
//! ```text
//! Events
//! ├── MarketEvent (ticks, bars, L2 updates)
//! ├── SignalEvent (strategy signals)
//! ├── OrderEvent (order lifecycle)
//! └── PortfolioSnapshot (position state)
//! ```
//!
//! ## Deterministic Replay Types
//! The `depth` module provides scaled-integer types for deterministic L2 replay:
//! - `DepthEvent`: Order book update with gap detection
//! - `DepthLevel`: Single price level (mantissa representation)
//!
//! ## References
//! - IEEE Std 1016-2009: Software Design Descriptions
//! - FIX Protocol 5.0 SP2 for field naming conventions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Deterministic depth types for L2 replay (scaled integers, gap detection)
pub mod depth;
pub use depth::{DepthEvent, DepthLevel, IntegrityTier};

// Canonical events with fixed-point representation and correlation IDs
pub mod events;
pub use events::{
    CONFIDENCE_EXPONENT, CorrelationContext, DecisionEvent, MarketSnapshot, ParseMantissaError,
    QuoteEvent as CanonicalQuoteEvent, SPREAD_BPS_EXPONENT, parse_to_mantissa_pure,
};

// Option Greeks and pricing primitives (moved here to break dependency cycles)
pub mod greeks;
pub use greeks::{OptionGreeks, OptionType};

// Tournament types for alpha discovery (Phase 12)
pub mod tournament;
pub use tournament::{
    ArtifactDigest, InputSegment, LEADERBOARD_SCHEMA, LeaderboardRow, LeaderboardV1,
    RUN_MANIFEST_SCHEMA, RunManifestV1, RunRecord, RunResultPaths, TOURNAMENT_MANIFEST_SCHEMA,
    TournamentManifestV1, TournamentPreset, compare_rows, compute_bundle_digest, generate_run_id,
    generate_run_key, generate_tournament_id, is_meaningful_run,
};

// Execution events for live trading lifecycle (Phase 14.2)
pub mod execution_events;
pub use execution_events::{
    CancelSource, ClientOrderId, EXECUTION_EVENTS_SCHEMA_VERSION, ExchangeOrderId,
    ExecutionOrderType, ExecutionSide, FillId, FixedPointValue, IdempotencyKey, IntentId,
    LiveOrderState, OrderAckEvent, OrderCancelEvent, OrderFillEvent, OrderIntentEvent,
    OrderRejectEvent, OrderSubmitEvent, PositionCloseEvent,
};

// Position events for position lifecycle (Phase 14.3)
pub mod position_events;
pub use position_events::{
    POSITION_EVENTS_SCHEMA_VERSION, PositionClosedEvent, PositionEventKind, PositionFlipEvent,
    PositionId, PositionIncreaseEvent, PositionKey, PositionOpenEvent, PositionReduceEvent,
    PositionSide, PositionVenue, SnapshotId as PositionSnapshotId,
};

/// Market data event from exchange or data feed.
///
/// # Description
/// Encapsulates all market data types: ticks, OHLCV bars, and Level 2 updates.
/// Used as the primary input to strategies.
///
/// # Fields
/// * `exchange_time` - Timestamp from exchange (authoritative)
/// * `local_time` - Local receipt timestamp for latency measurement
/// * `symbol` - Trading symbol (e.g., "BTCUSDT")
/// * `payload` - Specific data type ([`MarketPayload`])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    /// Exchange-provided timestamp (authoritative for sequencing).
    pub exchange_time: DateTime<Utc>,
    /// Local receipt time for latency metrics.
    pub local_time: DateTime<Utc>,
    /// Symbol identifier (e.g., "BTCUSDT", "NIFTY 50").
    pub symbol: String,
    /// Event payload variant.
    pub payload: MarketPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketPayload {
    Tick {
        price: f64,
        size: f64,
        side: Side,
    },
    /// Individual trade event with trade ID.
    Trade {
        /// Trade ID from exchange.
        trade_id: i64,
        /// Trade price.
        price: f64,
        /// Trade quantity.
        quantity: f64,
        /// True if buyer was market maker.
        is_buyer_maker: bool,
    },
    Bar {
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        interval_ms: u64,
    },
    L2Update(L2Update),
    L2Snapshot(L2Snapshot),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Update {
    pub bids: Vec<L2Level>,
    pub asks: Vec<L2Level>,
    /// First update ID in this event (U) - used for Binance sequencing
    pub first_update_id: u64,
    /// Last update ID in this event (u) - the primary sequence number
    pub last_update_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Snapshot {
    pub bids: Vec<L2Level>,
    pub asks: Vec<L2Level>,
    pub update_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Level {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Alpha signal generated by a trading strategy.
///
/// # Description
/// Represents a high-level trading intent. Abstracted from specific order types,
/// a signal defines the desired side and quantity at a reference price.
///
/// # Fields
/// * `timestamp` - Generation time of the signal.
/// * `strategy_id` - Originating strategy component.
/// * `symbol` - Target asset identifier.
/// * `side` - Direction of the intent (Buy/Sell).
/// * `price` - Reference price for signal evaluation.
/// * `quantity` - Intended position size change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEvent {
    pub timestamp: DateTime<Utc>,
    pub strategy_id: String,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    /// Unique identifier for the intent that generated this signal.
    pub intent_id: Option<Uuid>,

    // ========== HFT Decision Context (V2) ==========
    /// Best bid price at decision time
    #[serde(default)]
    pub decision_bid: f64,
    /// Best ask price at decision time
    #[serde(default)]
    pub decision_ask: f64,
    /// Mid price at decision time ((bid + ask) / 2)
    #[serde(default)]
    pub decision_mid: f64,
    /// Spread in basis points at decision time
    #[serde(default)]
    pub spread_bps: f64,
    /// Book timestamp (exchange time) in nanoseconds - for causality checks
    #[serde(default)]
    pub book_ts_ns: i64,
    /// Expected edge in basis points (from strategy signal)
    #[serde(default)]
    pub expected_edge_bps: f64,
}

/// Normalized order lifecycle event.
///
/// # Description
/// The primary vessel for order submission and status updates. Supports
/// multiple operation types through its payload.
///
/// # Correlation Chain (Phase 3)
/// Every order event carries `parent_decision_id` for audit-grade traceability:
/// `DecisionEvent → OrderIntent → OrderEvent → FillEvent → TradeAttribution`
///
/// # Fields
/// * `order_id` - RFC 4122 compliant unique identifier.
/// * `parent_decision_id` - Decision that originated this order (for attribution).
/// * `intent_id` - Link to the original strategy intent.
/// * `timestamp` - Event creation time.
/// * `symbol` - Asset symbol.
/// * `side` - Transaction side.
/// * `payload` - Functional intent ([`OrderPayload`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub order_id: Uuid,
    /// Parent decision that originated this order (Phase 3 correlation).
    /// Required for audit-grade traceability and PnL attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_decision_id: Option<Uuid>,
    pub intent_id: Option<Uuid>,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: Side,
    pub payload: OrderPayload,
}

/// Functional variants for order operations.
///
/// # Variants
/// * `New` - Submit a fresh order to the matching engine.
/// * `Update` - Status notification from the venue.
/// * `Modify` - Amend price or quantity of a pending order.
/// * `Cancel` - Request termination of an open order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderPayload {
    New {
        symbol: String,
        side: Side,
        price: Option<f64>,
        quantity: f64,
        order_type: OrderType,
    },
    Update {
        status: OrderStatus,
        filled_quantity: f64,
        avg_price: f64,
        commission: f64,
    },
    /// Modify an existing order's price and/or quantity.
    Modify {
        /// New limit price (None = keep existing).
        new_price: Option<f64>,
        /// New quantity (None = keep existing).
        new_quantity: Option<f64>,
    },
    Cancel,
}

/// Specificity of execution execution logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    /// Immediate execution against available liquidity.
    Market,
    /// Execution only at or better than a specified price.
    Limit,
}

/// Lifecycle stages of a venue order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Initial submission state.
    Pending,
    /// Confirmed by the exchange matching engine.
    Accepted,
    /// Refused by exchange or risk engine.
    Rejected,
    /// Partially filled; remaining quantity stays open.
    PartiallyFilled,
    /// Fully executed.
    Filled,
    /// Successfully withdrawn from the market.
    Cancelled,
}

/// Snapshot of a specific asset position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
}

/// Full state of the trading account at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_equity: f64,
    /// Available cash balance.
    pub cash: f64,
    /// Total margin used.
    pub margin_used: f64,
    pub positions: Vec<PositionUpdate>,
}

// ============================================================================
// RISK & SYSTEM EVENTS (C2 FIX)
// ============================================================================

/// Risk event for monitoring and alerting.
///
/// # Description
/// Published when risk limits are breached or approaching thresholds.
/// Enables real-time risk monitoring and circuit breaker triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    /// Event timestamp.
    pub timestamp: DateTime<Utc>,
    /// Risk event type.
    pub event_type: RiskEventType,
    /// Affected symbol (if applicable).
    pub symbol: Option<String>,
    /// Affected strategy (if applicable).
    pub strategy_id: Option<String>,
    /// Current value that triggered the event.
    pub current_value: f64,
    /// Threshold that was breached.
    pub threshold: f64,
    /// Human-readable message.
    pub message: String,
}

/// Types of risk events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEventType {
    /// Kill switch activated.
    KillSwitchTriggered,
    /// Order rejected by risk engine.
    OrderRejected,
    /// Position limit warning (approaching threshold).
    PositionLimitWarning,
    /// Position limit breached.
    PositionLimitBreached,
    /// Drawdown limit warning.
    DrawdownWarning,
    /// Drawdown limit breached.
    DrawdownBreached,
    /// Strategy circuit breaker tripped.
    StrategyCircuitBreaker,
    /// Daily loss limit reached.
    DailyLossLimit,
    /// Order rate limit exceeded.
    OrderRateLimit,
}

/// System health event for monitoring infrastructure.
///
/// # Description
/// Published for system-level events: connection status, latency spikes,
/// memory pressure, etc. Enables operational monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthEvent {
    /// Event timestamp.
    pub timestamp: DateTime<Utc>,
    /// Health event type.
    pub event_type: SystemHealthType,
    /// Component that generated the event.
    pub component: String,
    /// Health status.
    pub status: HealthStatus,
    /// Additional details.
    pub details: Option<String>,
}

/// Types of system health events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemHealthType {
    /// Connector connection status change.
    ConnectorStatus,
    /// Latency spike detected.
    LatencySpike,
    /// Memory pressure warning.
    MemoryPressure,
    /// Disk space warning.
    DiskSpace,
    /// Message queue backlog.
    QueueBacklog,
    /// Heartbeat timeout.
    HeartbeatTimeout,
}

/// Health status levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System healthy.
    Healthy,
    /// Warning - degraded performance.
    Warning,
    /// Critical - action required.
    Critical,
    /// Component down.
    Down,
}

/// Fill event for trade audit trail.
///
/// # Description
/// Discrete event for each order fill, separate from OrderPayload::Update.
/// Enables detailed trade logging and reconciliation.
///
/// # Correlation Chain (Phase 3)
/// Every fill carries `parent_decision_id` for direct attribution traceability:
/// `DecisionEvent → OrderIntent → OrderEvent → FillEvent → TradeAttribution`
///
/// This enables:
/// - Direct lookup: fill → decision (no need to traverse intent chain)
/// - WAL replay with preserved correlation
/// - Per-decision PnL attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    /// Fill timestamp.
    pub timestamp: DateTime<Utc>,
    /// Original order ID.
    pub order_id: Uuid,
    /// Parent decision that originated this fill (Phase 3 correlation).
    /// Required for audit-grade traceability and PnL attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_decision_id: Option<Uuid>,
    /// Link to the original strategy intent.
    pub intent_id: Option<Uuid>,
    /// Unique fill ID.
    pub fill_id: String,
    /// Symbol traded.
    pub symbol: String,
    /// Fill side.
    pub side: Side,
    /// Fill price.
    pub price: f64,
    /// Fill quantity.
    pub quantity: f64,
    /// Commission paid.
    pub commission: f64,
    /// Commission asset.
    pub commission_asset: String,
    /// Venue that executed the fill.
    pub venue: String,
    /// Whether this fill completed the order.
    pub is_final: bool,
}

// ============================================================================
// TRADE ATTRIBUTION (PHASE 3)
// ============================================================================

/// Trade attribution event for deterministic PnL accounting.
///
/// # Description
/// Records the outcome of a decision: realized PnL, fees, slippage.
/// Uses fixed-point representation for deterministic cross-platform replay.
///
/// # Correlation Chain
/// Each attribution event links back to a specific decision:
/// `Decision → OrderIntent → Order → Fill → TradeAttribution`
///
/// # Fixed-Point Policy
/// ALL monetary values use mantissa + exponent (i128/i64 + i8).
/// NO f64 fields - this ensures deterministic attribution hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAttributionEvent {
    /// Attribution timestamp (when decision was closed/flushed)
    pub ts_ns: i64,
    /// Symbol traded
    pub symbol: String,
    /// Venue/exchange identifier
    pub venue: String,

    /// Parent decision that originated this attribution (correlation)
    pub parent_decision_id: Uuid,
    /// Strategy ID (from manifest StrategyBinding)
    pub strategy_id: String,

    // === Fixed-point attribution (mantissa + exponent) ===
    /// Gross PnL before fees (mantissa)
    pub gross_pnl_mantissa: i128,
    /// Total fees paid (mantissa)
    pub fees_mantissa: i128,
    /// Net PnL after fees (mantissa) = gross_pnl - fees
    pub net_pnl_mantissa: i128,
    /// PnL exponent (typically -8 for crypto)
    pub pnl_exponent: i8,

    // === Trade metrics ===
    /// Holding time in nanoseconds (from first fill to last fill)
    pub holding_time_ns: i64,
    /// Number of fills that contributed to this decision
    pub num_fills: u32,

    // === Slippage proxy (fixed-point) ===
    //
    // SLIPPAGE SEMANTIC DEFINITION (locked - do not change without schema bump):
    //
    // Reference price: DecisionEvent.reference_price_mantissa (mid price at decision time)
    //   = (market_snapshot.bid_price_mantissa + market_snapshot.ask_price_mantissa) / 2
    //
    // Signed quantity convention:
    //   - Buy fills: positive qty contributes positive slippage when fill > mid
    //   - Sell fills: positive qty contributes positive slippage when fill < mid
    //
    // Formula (per fill):
    //   slippage_contribution = (fill_price - decision_mid) * qty / 100
    //   For buys: positive contribution = unfavorable (paid more than mid)
    //   For sells: contribution = -(fill_price - decision_mid) * qty / 100
    //             positive = unfavorable (received less than mid)
    //
    // Total slippage = sum of contributions across all fills
    //   - Positive = unfavorable execution (total cost > expected)
    //   - Negative = favorable execution (total cost < expected)
    //   - Zero = executed exactly at decision mid price
    //
    // Exponent: same as pnl_exponent (typically -8 for crypto)
    //
    /// Slippage = sum of (fill_price - decision_mid_price) * signed_qty across fills.
    ///
    /// - **Reference price**: Decision mid price (bid+ask)/2 at decision time
    /// - **Sign convention**: Positive = unfavorable, Negative = favorable
    /// - **Unit**: Same as PnL (e.g., USD notional with pnl_exponent)
    pub slippage_mantissa: i128,
    /// Slippage exponent (same as pnl_exponent, typically -8 for crypto)
    pub slippage_exponent: i8,
}

impl TradeAttributionEvent {
    /// Get gross PnL as f64 (for display only).
    ///
    /// # Warning
    /// Only use for display/logging. Do NOT use for computations.
    pub fn gross_pnl_f64(&self) -> f64 {
        self.gross_pnl_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }

    /// Get fees as f64 (for display only).
    pub fn fees_f64(&self) -> f64 {
        self.fees_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }

    /// Get net PnL as f64 (for display only).
    pub fn net_pnl_f64(&self) -> f64 {
        self.net_pnl_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }

    /// Get slippage as f64 (for display only).
    pub fn slippage_f64(&self) -> f64 {
        self.slippage_mantissa as f64 * 10f64.powi(self.slippage_exponent as i32)
    }

    /// Get holding time in seconds (for display only).
    pub fn holding_time_secs(&self) -> f64 {
        self.holding_time_ns as f64 / 1_000_000_000.0
    }
}

/// Per-decision attribution accumulator.
///
/// Tracks fills for a single decision until flushed to TradeAttributionEvent.
#[derive(Debug, Clone, Default)]
pub struct DecisionAttributionAccumulator {
    /// Parent decision ID
    pub decision_id: Uuid,
    /// Strategy ID
    pub strategy_id: String,
    /// Symbol
    pub symbol: String,
    /// Venue
    pub venue: String,
    /// First fill timestamp (nanoseconds)
    pub first_fill_ts_ns: Option<i64>,
    /// Last fill timestamp (nanoseconds)
    pub last_fill_ts_ns: Option<i64>,

    // === Accumulation state ===
    /// Total signed quantity filled (positive = long, negative = short)
    pub signed_qty_mantissa: i128,
    /// Total cost basis (entry notional)
    pub cost_basis_mantissa: i128,
    /// Total proceeds (exit notional)
    pub proceeds_mantissa: i128,
    /// Total fees
    pub fees_mantissa: i128,
    /// Number of fills
    pub num_fills: u32,

    // === Slippage tracking ===
    /// Decision mid price (reference for slippage)
    pub decision_mid_mantissa: i64,
    pub price_exponent: i8,
    /// Sum of (fill_price - decision_mid) * qty for slippage calculation
    pub slippage_accum_mantissa: i128,

    /// Quantity exponent
    pub qty_exponent: i8,
}

impl DecisionAttributionAccumulator {
    /// Create a new accumulator for a decision.
    pub fn new(
        decision_id: Uuid,
        strategy_id: String,
        symbol: String,
        venue: String,
        decision_mid_mantissa: i64,
        price_exponent: i8,
        qty_exponent: i8,
    ) -> Self {
        Self {
            decision_id,
            strategy_id,
            symbol,
            venue,
            decision_mid_mantissa,
            price_exponent,
            qty_exponent,
            ..Default::default()
        }
    }

    /// Add a fill to the accumulator.
    ///
    /// # Arguments
    /// * `ts_ns` - Fill timestamp in nanoseconds
    /// * `price_mantissa` - Fill price (mantissa)
    /// * `qty_mantissa` - Fill quantity (mantissa, always positive)
    /// * `fee_mantissa` - Fee amount (mantissa)
    /// * `is_buy` - True for buy, false for sell
    pub fn add_fill(
        &mut self,
        ts_ns: i64,
        price_mantissa: i64,
        qty_mantissa: i64,
        fee_mantissa: i64,
        is_buy: bool,
    ) {
        // Track timestamps
        if self.first_fill_ts_ns.is_none() {
            self.first_fill_ts_ns = Some(ts_ns);
        }
        self.last_fill_ts_ns = Some(ts_ns);

        // Accumulate fees
        self.fees_mantissa += fee_mantissa as i128;
        self.num_fills += 1;

        // Notional = price * qty, adjusted for exponents
        // price (exp -2) * qty (exp -8) = notional (exp -10)
        // Normalize to qty_exponent for consistency
        let notional_mantissa = (price_mantissa as i128) * (qty_mantissa as i128) / 100;

        if is_buy {
            self.signed_qty_mantissa += qty_mantissa as i128;
            self.cost_basis_mantissa += notional_mantissa;
        } else {
            self.signed_qty_mantissa -= qty_mantissa as i128;
            self.proceeds_mantissa += notional_mantissa;
        }

        // Accumulate slippage: (fill_price - decision_mid) * signed_qty
        let price_diff = price_mantissa as i128 - self.decision_mid_mantissa as i128;
        let slippage_contribution = if is_buy {
            price_diff * qty_mantissa as i128 / 100 // Buyer: positive diff = unfavorable
        } else {
            -price_diff * qty_mantissa as i128 / 100 // Seller: negative diff = unfavorable
        };
        self.slippage_accum_mantissa += slippage_contribution;
    }

    /// Check if the decision is closed (position returned to zero).
    pub fn is_closed(&self) -> bool {
        self.signed_qty_mantissa == 0 && self.num_fills > 0
    }

    /// Flush the accumulator to a TradeAttributionEvent.
    ///
    /// Call this at end-of-run or when position returns to zero.
    pub fn flush(&self, flush_ts_ns: i64) -> TradeAttributionEvent {
        // Gross PnL = proceeds - cost_basis
        let gross_pnl_mantissa = self.proceeds_mantissa - self.cost_basis_mantissa;
        // Net PnL = gross - fees
        let net_pnl_mantissa = gross_pnl_mantissa - self.fees_mantissa;

        // Holding time
        let holding_time_ns = match (self.first_fill_ts_ns, self.last_fill_ts_ns) {
            (Some(first), Some(last)) => last - first,
            _ => 0,
        };

        TradeAttributionEvent {
            ts_ns: flush_ts_ns,
            symbol: self.symbol.clone(),
            venue: self.venue.clone(),
            parent_decision_id: self.decision_id,
            strategy_id: self.strategy_id.clone(),
            gross_pnl_mantissa,
            fees_mantissa: self.fees_mantissa,
            net_pnl_mantissa,
            pnl_exponent: self.qty_exponent, // PnL uses qty exponent
            holding_time_ns,
            num_fills: self.num_fills,
            slippage_mantissa: self.slippage_accum_mantissa,
            slippage_exponent: self.qty_exponent,
        }
    }
}

// ============================================================================
// ATTRIBUTION SUMMARY + ALPHA SCORE (PHASE 4)
// ============================================================================

/// Attribution summary aggregated across all decisions for a strategy run.
///
/// # Description
/// Deterministic aggregation of TradeAttributionEvents into comparable metrics.
/// Used for strategy ranking and alpha evaluation.
///
/// # Fixed-Point Policy
/// ALL numeric values use mantissa + exponent (i128/i64 + i8).
/// NO f64 fields - this ensures deterministic summary hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionSummary {
    /// Strategy ID (from manifest StrategyBinding)
    pub strategy_id: String,
    /// Run ID for this backtest/evaluation
    pub run_id: String,
    /// Symbol(s) traded
    pub symbols: Vec<String>,
    /// Summary generation timestamp (nanoseconds)
    pub generated_ts_ns: i64,

    // === Decision/Trade Counts ===
    /// Total number of decisions made
    pub total_decisions: u32,
    /// Total number of fills executed
    pub total_fills: u32,
    /// Number of winning decisions (net_pnl > 0)
    pub winning_decisions: u32,
    /// Number of losing decisions (net_pnl < 0)
    pub losing_decisions: u32,
    /// Number of complete round-trips (entry + exit)
    pub round_trips: u32,

    // === PnL Aggregates (fixed-point) ===
    /// Total gross PnL (mantissa)
    pub total_gross_pnl_mantissa: i128,
    /// Total fees paid (mantissa)
    pub total_fees_mantissa: i128,
    /// Total net PnL (mantissa) = gross - fees
    pub total_net_pnl_mantissa: i128,
    /// PnL exponent (typically -8 for crypto)
    pub pnl_exponent: i8,

    // === Derived Metrics (fixed-point) ===
    /// Win rate = winning_decisions * 10000 / total_decisions (basis points, 10000 = 100%)
    pub win_rate_bps: u32,
    /// Average net PnL per decision (mantissa)
    pub avg_pnl_per_decision_mantissa: i128,
    /// Total slippage (mantissa)
    pub total_slippage_mantissa: i128,
    /// Slippage exponent
    pub slippage_exponent: i8,

    // === Risk Metrics (fixed-point) ===
    /// Max single-decision loss (fixed-point drawdown proxy).
    ///
    /// # Semantics (LOCKED - do not change without version bump):
    /// - **Definition**: Largest single-decision loss, expressed as positive magnitude
    /// - **Exponent**: Uses the SAME `pnl_exponent` as all other PnL fields
    /// - **Computation**: `max(-net_pnl_mantissa)` across all losing decisions
    /// - **NOT**: A statistical risk model (not VaR, CVaR, or equity-curve max drawdown)
    /// - **NOT**: Computed from floating-point equity curves
    ///
    /// This is a simple, deterministic, per-decision metric for alpha scoring.
    /// It represents the worst single trade outcome, not portfolio-level risk.
    pub max_loss_mantissa: i128,
    /// Total holding time across all decisions (nanoseconds)
    pub total_holding_time_ns: i64,
}

impl AttributionSummary {
    /// Get total net PnL as f64 (for display only).
    pub fn total_net_pnl_f64(&self) -> f64 {
        self.total_net_pnl_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }

    /// Get win rate as percentage (for display only).
    pub fn win_rate_pct(&self) -> f64 {
        self.win_rate_bps as f64 / 100.0
    }

    /// Get average PnL per decision as f64 (for display only).
    pub fn avg_pnl_per_decision_f64(&self) -> f64 {
        self.avg_pnl_per_decision_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }

    /// Get max loss as f64 (for display only).
    pub fn max_loss_f64(&self) -> f64 {
        self.max_loss_mantissa as f64 * 10f64.powi(self.pnl_exponent as i32)
    }
}

/// Attribution summary aggregator.
///
/// Collects TradeAttributionEvents and produces an AttributionSummary.
#[derive(Debug, Clone, Default)]
pub struct AttributionSummaryBuilder {
    strategy_id: String,
    run_id: String,
    symbols: Vec<String>,
    pnl_exponent: i8,

    // Aggregation state
    total_decisions: u32,
    total_fills: u32,
    winning_decisions: u32,
    losing_decisions: u32,
    round_trips: u32,
    total_gross_pnl_mantissa: i128,
    total_fees_mantissa: i128,
    total_net_pnl_mantissa: i128,
    total_slippage_mantissa: i128,
    max_loss_mantissa: i128,
    total_holding_time_ns: i64,
}

impl AttributionSummaryBuilder {
    /// Create a new summary builder.
    pub fn new(
        strategy_id: String,
        run_id: String,
        symbols: Vec<String>,
        pnl_exponent: i8,
    ) -> Self {
        Self {
            strategy_id,
            run_id,
            symbols,
            pnl_exponent,
            ..Default::default()
        }
    }

    /// Add a TradeAttributionEvent to the summary.
    pub fn add_event(&mut self, event: &TradeAttributionEvent) {
        self.total_decisions += 1;
        self.total_fills += event.num_fills;
        self.total_gross_pnl_mantissa += event.gross_pnl_mantissa;
        self.total_fees_mantissa += event.fees_mantissa;
        self.total_net_pnl_mantissa += event.net_pnl_mantissa;
        self.total_slippage_mantissa += event.slippage_mantissa;
        self.total_holding_time_ns += event.holding_time_ns;

        if event.net_pnl_mantissa > 0 {
            self.winning_decisions += 1;
        } else if event.net_pnl_mantissa < 0 {
            self.losing_decisions += 1;
            // Track max loss: largest single-decision loss as positive magnitude.
            // Uses SAME exponent space as net_pnl (no conversion needed).
            // This is a per-decision metric, NOT an equity-curve drawdown.
            let loss = -event.net_pnl_mantissa;
            if loss > self.max_loss_mantissa {
                self.max_loss_mantissa = loss;
            }
        }
    }

    /// Finalize and produce the AttributionSummary.
    pub fn build(self, generated_ts_ns: i64) -> AttributionSummary {
        // Win rate in basis points (10000 = 100%)
        let win_rate_bps = if self.total_decisions > 0 {
            ((self.winning_decisions as u64) * 10000 / (self.total_decisions as u64)) as u32
        } else {
            0
        };

        // Average PnL per decision
        let avg_pnl_per_decision_mantissa = if self.total_decisions > 0 {
            self.total_net_pnl_mantissa / (self.total_decisions as i128)
        } else {
            0
        };

        AttributionSummary {
            strategy_id: self.strategy_id,
            run_id: self.run_id,
            symbols: self.symbols,
            generated_ts_ns,
            total_decisions: self.total_decisions,
            total_fills: self.total_fills,
            winning_decisions: self.winning_decisions,
            losing_decisions: self.losing_decisions,
            round_trips: self.round_trips,
            total_gross_pnl_mantissa: self.total_gross_pnl_mantissa,
            total_fees_mantissa: self.total_fees_mantissa,
            total_net_pnl_mantissa: self.total_net_pnl_mantissa,
            pnl_exponent: self.pnl_exponent,
            win_rate_bps,
            avg_pnl_per_decision_mantissa,
            total_slippage_mantissa: self.total_slippage_mantissa,
            slippage_exponent: self.pnl_exponent,
            max_loss_mantissa: self.max_loss_mantissa,
            total_holding_time_ns: self.total_holding_time_ns,
        }
    }

    /// Add pre-computed metrics directly (for tournament runner).
    ///
    /// This allows adding summary metrics from an external backtest result
    /// without needing individual TradeAttributionEvents.
    pub fn add_metrics(&mut self, metrics: BacktestMetrics) {
        self.total_decisions = metrics.decisions;
        self.total_fills = metrics.fills;
        self.winning_decisions = metrics.winning;
        self.losing_decisions = metrics.losing;
        self.total_net_pnl_mantissa = metrics.net_pnl_mantissa;
        self.max_loss_mantissa = metrics.max_loss_mantissa;
        self.round_trips = metrics.round_trips;
    }
}

/// Pre-computed metrics from a backtest result.
#[derive(Debug, Clone, Default)]
pub struct BacktestMetrics {
    pub decisions: u32,
    pub fills: u32,
    pub winning: u32,
    pub losing: u32,
    pub net_pnl_mantissa: i128,
    pub max_loss_mantissa: i128,
    pub round_trips: u32,
}

/// Alpha Score v1: Deterministic strategy quality metric.
///
/// # Formula (locked - do not change without version bump):
///
/// ```text
/// AlphaScoreV1 = (net_pnl * SCALE) / (max_loss + EPSILON)
/// ```
///
/// Where:
/// - `net_pnl`: Total net PnL (mantissa)
/// - `max_loss`: Max single-decision loss (mantissa, positive)
/// - `SCALE`: 10000 (for basis-point-like resolution)
/// - `EPSILON`: 1_000_000 (to avoid division by zero, ~$0.01 at exp -8)
///
/// # Interpretation
/// - Positive score = profitable with controlled drawdown
/// - Higher score = better risk-adjusted returns
/// - Score is comparable across runs with same pnl_exponent
///
/// # Fixed-Point Policy
/// ALL fields use mantissa + exponent. NO f64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaScoreV1 {
    /// Score value (mantissa)
    pub score_mantissa: i128,
    /// Score exponent (typically -4 for basis-point resolution)
    pub score_exponent: i8,
    /// Version identifier for this scoring formula
    pub formula_version: &'static str,

    // === Inputs (for auditability) ===
    /// Net PnL used in calculation (mantissa)
    pub input_net_pnl_mantissa: i128,
    /// Max loss used in calculation (mantissa)
    pub input_max_loss_mantissa: i128,
    /// PnL exponent
    pub input_pnl_exponent: i8,
}

impl AlphaScoreV1 {
    /// Score scaling factor
    pub const SCALE: i128 = 10000;
    /// Epsilon to avoid division by zero (~$0.01 at exp -8)
    pub const EPSILON: i128 = 1_000_000;
    /// Formula version identifier
    pub const VERSION: &'static str = "alpha_score_v1.0";

    /// Compute AlphaScoreV1 from an AttributionSummary.
    ///
    /// Formula: score = (net_pnl * SCALE) / (max_loss + EPSILON)
    pub fn from_summary(summary: &AttributionSummary) -> Self {
        let net_pnl = summary.total_net_pnl_mantissa;
        let max_loss = summary.max_loss_mantissa;

        // score = (net_pnl * SCALE) / (max_loss + EPSILON)
        let denominator = max_loss + Self::EPSILON;
        let score_mantissa = (net_pnl * Self::SCALE) / denominator;

        Self {
            score_mantissa,
            score_exponent: -4, // Basis-point-like resolution
            formula_version: Self::VERSION,
            input_net_pnl_mantissa: net_pnl,
            input_max_loss_mantissa: max_loss,
            input_pnl_exponent: summary.pnl_exponent,
        }
    }

    /// Get score as f64 (for display only).
    pub fn score_f64(&self) -> f64 {
        self.score_mantissa as f64 * 10f64.powi(self.score_exponent as i32)
    }

    /// Check if the score indicates profitability.
    pub fn is_profitable(&self) -> bool {
        self.score_mantissa > 0
    }
}

// ============================================================================
// G1 PROMOTION GATE (PHASE 4D)
// ============================================================================

/// G1 Promotion Gate: Threshold-based strategy promotion decision.
///
/// Determines if a strategy meets the minimum requirements for G1 (Generation 1)
/// promotion status. G1 strategies are considered production-ready after passing
/// these checks.
///
/// # Thresholds (locked - do not change without version bump):
///
/// | Threshold         | Value    | Description                          |
/// |-------------------|----------|--------------------------------------|
/// | min_alpha_score   | 1000     | Minimum alpha score (mantissa)       |
/// | min_win_rate_bps  | 4000     | Minimum 40% win rate                 |
/// | min_decisions     | 10       | Minimum # of decisions for validity  |
/// | max_loss_pct_bps  | 5000     | Max loss ≤ 50% of net PnL            |
///
/// # Decision Logic
/// All thresholds must be met for promotion. The gate returns a detailed
/// result with pass/fail reasons for transparency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G1PromotionGate {
    /// Gate version identifier
    pub version: &'static str,

    // === Threshold Configuration (immutable after construction) ===
    /// Minimum alpha score (mantissa at score_exponent -4)
    pub min_alpha_score_mantissa: i128,
    /// Minimum win rate in basis points (10000 = 100%)
    pub min_win_rate_bps: u32,
    /// Minimum number of decisions for statistical validity
    pub min_decisions: u32,
    /// Maximum loss as percentage of net PnL in basis points (5000 = 50%)
    pub max_loss_pct_bps: u32,
}

impl Default for G1PromotionGate {
    fn default() -> Self {
        Self::new()
    }
}

impl G1PromotionGate {
    /// Gate version identifier
    pub const VERSION: &'static str = "g1_gate_v1.0";

    /// Default minimum alpha score (1000 mantissa at -4 exponent = 0.1)
    pub const DEFAULT_MIN_ALPHA_SCORE: i128 = 1000;
    /// Default minimum win rate (40%)
    pub const DEFAULT_MIN_WIN_RATE_BPS: u32 = 4000;
    /// Default minimum decisions
    pub const DEFAULT_MIN_DECISIONS: u32 = 10;
    /// Default max loss percentage (50% of net PnL)
    pub const DEFAULT_MAX_LOSS_PCT_BPS: u32 = 5000;

    /// Create a new G1 promotion gate with default thresholds.
    pub fn new() -> Self {
        Self {
            version: Self::VERSION,
            min_alpha_score_mantissa: Self::DEFAULT_MIN_ALPHA_SCORE,
            min_win_rate_bps: Self::DEFAULT_MIN_WIN_RATE_BPS,
            min_decisions: Self::DEFAULT_MIN_DECISIONS,
            max_loss_pct_bps: Self::DEFAULT_MAX_LOSS_PCT_BPS,
        }
    }

    /// Create a gate with custom thresholds.
    pub fn with_thresholds(
        min_alpha_score_mantissa: i128,
        min_win_rate_bps: u32,
        min_decisions: u32,
        max_loss_pct_bps: u32,
    ) -> Self {
        Self {
            version: Self::VERSION,
            min_alpha_score_mantissa,
            min_win_rate_bps,
            min_decisions,
            max_loss_pct_bps,
        }
    }

    /// Evaluate a strategy for G1 promotion.
    ///
    /// Returns a detailed result indicating pass/fail with reasons.
    pub fn evaluate(
        &self,
        summary: &AttributionSummary,
        alpha_score: &AlphaScoreV1,
    ) -> G1PromotionResult {
        let mut reasons = Vec::new();

        // Check alpha score
        let alpha_pass = alpha_score.score_mantissa >= self.min_alpha_score_mantissa;
        if !alpha_pass {
            reasons.push(format!(
                "Alpha score {} < minimum {}",
                alpha_score.score_mantissa, self.min_alpha_score_mantissa
            ));
        }

        // Check win rate
        let win_rate_pass = summary.win_rate_bps >= self.min_win_rate_bps;
        if !win_rate_pass {
            reasons.push(format!(
                "Win rate {}bps < minimum {}bps",
                summary.win_rate_bps, self.min_win_rate_bps
            ));
        }

        // Check minimum decisions
        let decisions_pass = summary.total_decisions >= self.min_decisions;
        if !decisions_pass {
            reasons.push(format!(
                "Decisions {} < minimum {}",
                summary.total_decisions, self.min_decisions
            ));
        }

        // Check max loss percentage (only if profitable)
        let loss_pct_pass = if summary.total_net_pnl_mantissa > 0 {
            // loss_pct = (max_loss * 10000) / net_pnl
            let loss_pct_bps =
                (summary.max_loss_mantissa * 10000) / summary.total_net_pnl_mantissa.max(1);
            loss_pct_bps as u32 <= self.max_loss_pct_bps
        } else {
            // If not profitable, fail this check
            reasons.push("Strategy is not profitable".to_string());
            false
        };
        if !loss_pct_pass && summary.total_net_pnl_mantissa > 0 {
            let loss_pct_bps =
                (summary.max_loss_mantissa * 10000) / summary.total_net_pnl_mantissa.max(1);
            reasons.push(format!(
                "Max loss {}bps > maximum {}bps of net PnL",
                loss_pct_bps, self.max_loss_pct_bps
            ));
        }

        let passed = alpha_pass && win_rate_pass && decisions_pass && loss_pct_pass;

        G1PromotionResult {
            passed,
            gate_version: self.version.to_string(),
            strategy_id: summary.strategy_id.clone(),
            alpha_score_mantissa: alpha_score.score_mantissa,
            win_rate_bps: summary.win_rate_bps,
            total_decisions: summary.total_decisions,
            reasons,
        }
    }
}

/// Result of G1 promotion gate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G1PromotionResult {
    /// Whether the strategy passed all thresholds
    pub passed: bool,
    /// Gate version used for evaluation
    pub gate_version: String,
    /// Strategy ID evaluated
    pub strategy_id: String,
    /// Alpha score at evaluation time (mantissa)
    pub alpha_score_mantissa: i128,
    /// Win rate at evaluation time (basis points)
    pub win_rate_bps: u32,
    /// Total decisions at evaluation time
    pub total_decisions: u32,
    /// Reasons for failure (empty if passed)
    pub reasons: Vec<String>,
}

impl G1PromotionResult {
    /// Check if the result indicates promotion.
    pub fn is_promoted(&self) -> bool {
        self.passed
    }

    /// Get a summary string for the result.
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "PASS: {} promoted to G1 (alpha={}, win_rate={}bps, decisions={})",
                self.strategy_id,
                self.alpha_score_mantissa,
                self.win_rate_bps,
                self.total_decisions
            )
        } else {
            format!(
                "FAIL: {} not promoted - {}",
                self.strategy_id,
                self.reasons.join("; ")
            )
        }
    }
}

// ============================================================================
// ROUTER V1: REGIME BUCKETS + STRATEGY ROUTING (PHASE 5)
// ============================================================================

/// Market regime labels for deterministic strategy routing.
///
/// # Regime Classification (LOCKED - do not change without version bump):
///
/// | Regime       | Description                                      |
/// |--------------|--------------------------------------------------|
/// | Normal       | Typical market conditions, default strategy      |
/// | HighVol      | Elevated volatility, may require wider stops     |
/// | LowLiquidity | Thin order books, reduce position sizes          |
/// | Trending     | Strong directional bias, momentum strategies     |
/// | MeanRevert   | Range-bound, mean reversion strategies           |
/// | FundingSkew  | Extreme funding rates (crypto), carry strategies |
/// | Halt         | No trading allowed (circuit breaker, etc.)       |
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RegimeLabel {
    /// Default/typical market conditions
    #[default]
    Normal,
    /// Elevated volatility regime
    HighVol,
    /// Low liquidity / thin order books
    LowLiquidity,
    /// Strong trending market
    Trending,
    /// Range-bound / mean reverting market
    MeanRevert,
    /// Extreme funding rate skew (crypto)
    FundingSkew,
    /// Trading halted (circuit breaker, maintenance)
    Halt,
}

impl RegimeLabel {
    /// Get the regime label as a string for logging.
    pub fn as_str(&self) -> &'static str {
        match self {
            RegimeLabel::Normal => "NORMAL",
            RegimeLabel::HighVol => "HIGH_VOL",
            RegimeLabel::LowLiquidity => "LOW_LIQUIDITY",
            RegimeLabel::Trending => "TRENDING",
            RegimeLabel::MeanRevert => "MEAN_REVERT",
            RegimeLabel::FundingSkew => "FUNDING_SKEW",
            RegimeLabel::Halt => "HALT",
        }
    }
}

/// Fixed-point inputs for regime classification.
///
/// All values are expressed in basis points (bps) or mantissa form
/// for deterministic cross-platform computation.
///
/// # Fields (LOCKED - do not change without version bump):
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegimeInputs {
    /// Timestamp of the regime evaluation (nanoseconds)
    pub ts_ns: i64,
    /// Symbol being evaluated
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub symbol: String,

    // === Spread Tier (fixed-point) ===
    /// Bid-ask spread in basis points (e.g., 10 = 0.10%)
    pub spread_bps: i32,
    /// Spread tier classification (0=tight, 1=normal, 2=wide, 3=extreme)
    pub spread_tier: u8,

    // === Volatility Proxy (fixed-point) ===
    /// Recent price volatility in basis points (e.g., 100 = 1% move)
    pub volatility_bps: i32,
    /// Volatility tier (0=low, 1=normal, 2=high, 3=extreme)
    pub volatility_tier: u8,

    // === Liquidity Tier (fixed-point) ===
    /// Order book depth in quote currency (mantissa, typically exp -2)
    pub depth_mantissa: i64,
    /// Depth exponent
    pub depth_exponent: i8,
    /// Liquidity tier (0=deep, 1=normal, 2=thin, 3=empty)
    pub liquidity_tier: u8,

    // === Funding State (crypto-specific) ===
    /// Funding rate in basis points (positive = longs pay shorts)
    pub funding_rate_bps: i32,
    /// Funding tier (0=neutral, 1=mild, 2=elevated, 3=extreme)
    pub funding_tier: u8,

    // === Trend Indicator (fixed-point) ===
    /// Trend strength indicator (-10000 to +10000, 0=no trend)
    pub trend_strength: i32,
}

impl Default for RegimeInputs {
    fn default() -> Self {
        Self {
            ts_ns: 0,
            symbol: String::new(),
            spread_bps: 0,
            spread_tier: 1,
            volatility_bps: 0,
            volatility_tier: 1,
            depth_mantissa: 0,
            depth_exponent: -2,
            liquidity_tier: 1,
            funding_rate_bps: 0,
            funding_tier: 0,
            trend_strength: 0,
        }
    }
}

impl RegimeInputs {
    /// Create new regime inputs with timestamp and symbol.
    pub fn new(ts_ns: i64, symbol: String) -> Self {
        Self {
            ts_ns,
            symbol,
            ..Default::default()
        }
    }

    /// Canonical bytes for hashing (deterministic serialization).
    ///
    /// Format: All fields in fixed order, big-endian encoding.
    /// Does NOT include symbol (it's context, not classification input).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);

        // ts_ns: 8 bytes
        buf.extend_from_slice(&self.ts_ns.to_be_bytes());
        // spread_bps: 4 bytes
        buf.extend_from_slice(&self.spread_bps.to_be_bytes());
        // spread_tier: 1 byte
        buf.push(self.spread_tier);
        // volatility_bps: 4 bytes
        buf.extend_from_slice(&self.volatility_bps.to_be_bytes());
        // volatility_tier: 1 byte
        buf.push(self.volatility_tier);
        // depth_mantissa: 8 bytes
        buf.extend_from_slice(&self.depth_mantissa.to_be_bytes());
        // depth_exponent: 1 byte
        buf.push(self.depth_exponent as u8);
        // liquidity_tier: 1 byte
        buf.push(self.liquidity_tier);
        // funding_rate_bps: 4 bytes
        buf.extend_from_slice(&self.funding_rate_bps.to_be_bytes());
        // funding_tier: 1 byte
        buf.push(self.funding_tier);
        // trend_strength: 4 bytes
        buf.extend_from_slice(&self.trend_strength.to_be_bytes());

        buf
    }
}

/// Router decision event: captures regime classification + strategy selection.
///
/// This is a first-class event that gets logged and hashed for replay parity.
/// The router produces these events to document WHY a particular strategy
/// was selected for a given market regime.
///
/// # Replay Parity
/// Router decisions are included in the decision trace and contribute to
/// the trace hash. Identical inputs must produce identical decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterDecisionEvent {
    /// Timestamp of the routing decision (nanoseconds)
    pub ts_ns: i64,
    /// Unique ID for this routing decision
    pub decision_id: uuid::Uuid,
    /// Symbol(s) this decision applies to
    pub symbols: Vec<String>,

    // === Regime Classification ===
    /// The inputs used for regime classification
    pub inputs: RegimeInputs,
    /// The classified regime label
    pub regime: RegimeLabel,
    /// Confidence in the classification (0-10000 basis points)
    pub confidence_bps: u32,

    // === Strategy Selection ===
    /// Selected strategy ID (format: "name:version:config_hash")
    pub selected_strategy_id: String,
    /// Alternative strategy IDs that were considered
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
    /// Reason for selection (for debugging/audit)
    pub selection_reason: String,

    // === Router Identity ===
    /// Router config hash (for manifest binding)
    pub router_config_hash: String,
    /// Router version
    pub router_version: String,
}

impl RouterDecisionEvent {
    /// Canonical bytes for hashing (deterministic serialization).
    ///
    /// Used to include router decisions in trace hash computation.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        // ts_ns: 8 bytes
        buf.extend_from_slice(&self.ts_ns.to_be_bytes());
        // decision_id: 16 bytes
        buf.extend_from_slice(self.decision_id.as_bytes());
        // regime inputs canonical bytes
        buf.extend_from_slice(&self.inputs.canonical_bytes());
        // regime label: 1 byte (enum discriminant)
        buf.push(self.regime as u8);
        // confidence: 4 bytes
        buf.extend_from_slice(&self.confidence_bps.to_be_bytes());
        // selected_strategy_id length + bytes
        buf.extend_from_slice(&(self.selected_strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.selected_strategy_id.as_bytes());
        // router_config_hash length + bytes
        buf.extend_from_slice(&(self.router_config_hash.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.router_config_hash.as_bytes());

        buf
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(self.canonical_bytes());
        hex::encode(hash)
    }
}

/// Router configuration: defines the routing rules.
///
/// This is a hashable configuration that determines how the router
/// classifies regimes and selects strategies.
///
/// # Versioning
/// The config hash is bound into the manifest to ensure reproducibility.
/// Any change to routing rules should produce a different hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Router version identifier
    pub version: String,
    /// List of strategy profiles available for routing
    pub strategy_profiles: Vec<StrategyProfile>,
    /// Regime classification thresholds
    pub thresholds: RegimeThresholds,
    /// Default strategy ID when no regime-specific rule matches
    pub default_strategy_id: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            version: "router_v1.0".to_string(),
            strategy_profiles: Vec::new(),
            thresholds: RegimeThresholds::default(),
            default_strategy_id: String::new(),
        }
    }
}

impl RouterConfig {
    /// Router version constant
    pub const VERSION: &'static str = "router_v1.0";

    /// Create a new router config with default settings.
    pub fn new() -> Self {
        Self {
            version: Self::VERSION.to_string(),
            ..Default::default()
        }
    }

    /// Canonical bytes for hashing (deterministic serialization).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        // version length + bytes
        buf.extend_from_slice(&(self.version.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.version.as_bytes());

        // number of strategy profiles
        buf.extend_from_slice(&(self.strategy_profiles.len() as u32).to_be_bytes());
        for profile in &self.strategy_profiles {
            buf.extend_from_slice(&profile.canonical_bytes());
        }

        // thresholds canonical bytes
        buf.extend_from_slice(&self.thresholds.canonical_bytes());

        // default_strategy_id length + bytes
        buf.extend_from_slice(&(self.default_strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.default_strategy_id.as_bytes());

        buf
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(self.canonical_bytes());
        hex::encode(hash)
    }
}

/// Strategy profile: a strategy + the regimes it's suited for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyProfile {
    /// Strategy ID (format: "name:version:config_hash")
    pub strategy_id: String,
    /// Human-readable name
    pub name: String,
    /// Regimes this strategy is suited for
    pub suitable_regimes: Vec<RegimeLabel>,
    /// Priority within suitable regimes (higher = preferred)
    pub priority: u32,
    /// Whether this strategy is currently enabled
    pub enabled: bool,
}

impl StrategyProfile {
    /// Canonical bytes for hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);

        // strategy_id length + bytes
        buf.extend_from_slice(&(self.strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.strategy_id.as_bytes());

        // number of suitable regimes + regime discriminants
        buf.extend_from_slice(&(self.suitable_regimes.len() as u32).to_be_bytes());
        for regime in &self.suitable_regimes {
            buf.push(*regime as u8);
        }

        // priority
        buf.extend_from_slice(&self.priority.to_be_bytes());

        // enabled
        buf.push(self.enabled as u8);

        buf
    }
}

/// Regime classification thresholds (all in basis points or tiers).
///
/// # Thresholds (LOCKED - do not change without version bump):
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegimeThresholds {
    // === Spread Thresholds ===
    /// Spread >= this triggers WIDE tier (bps)
    pub spread_wide_bps: i32,
    /// Spread >= this triggers EXTREME tier (bps)
    pub spread_extreme_bps: i32,

    // === Volatility Thresholds ===
    /// Volatility >= this triggers HIGH tier (bps)
    pub vol_high_bps: i32,
    /// Volatility >= this triggers EXTREME tier (bps)
    pub vol_extreme_bps: i32,

    // === Liquidity Thresholds ===
    /// Depth <= this triggers THIN tier (in quote currency, mantissa at exp -2)
    pub depth_thin_mantissa: i64,
    /// Depth <= this triggers EMPTY tier
    pub depth_empty_mantissa: i64,

    // === Funding Thresholds (crypto) ===
    /// |Funding| >= this triggers ELEVATED tier (bps)
    pub funding_elevated_bps: i32,
    /// |Funding| >= this triggers EXTREME tier (bps)
    pub funding_extreme_bps: i32,

    // === Trend Thresholds ===
    /// |Trend| >= this triggers TRENDING regime
    pub trend_threshold: i32,
}

impl Default for RegimeThresholds {
    fn default() -> Self {
        Self {
            // Spread: tight < 5bps, normal < 15bps, wide < 50bps, extreme >= 50bps
            spread_wide_bps: 15,
            spread_extreme_bps: 50,

            // Volatility: low < 50bps, normal < 150bps, high < 300bps, extreme >= 300bps
            vol_high_bps: 150,
            vol_extreme_bps: 300,

            // Depth: deep > $100k, normal > $10k, thin > $1k, empty <= $1k
            // (mantissa at exp -2, so $10k = 1_000_000)
            depth_thin_mantissa: 1_000_000, // $10k
            depth_empty_mantissa: 100_000,  // $1k

            // Funding: neutral < 5bps, mild < 20bps, elevated < 50bps, extreme >= 50bps
            funding_elevated_bps: 20,
            funding_extreme_bps: 50,

            // Trend: significant trend if |strength| >= 3000 (30%)
            trend_threshold: 3000,
        }
    }
}

impl RegimeThresholds {
    /// Canonical bytes for hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);

        buf.extend_from_slice(&self.spread_wide_bps.to_be_bytes());
        buf.extend_from_slice(&self.spread_extreme_bps.to_be_bytes());
        buf.extend_from_slice(&self.vol_high_bps.to_be_bytes());
        buf.extend_from_slice(&self.vol_extreme_bps.to_be_bytes());
        buf.extend_from_slice(&self.depth_thin_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.depth_empty_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.funding_elevated_bps.to_be_bytes());
        buf.extend_from_slice(&self.funding_extreme_bps.to_be_bytes());
        buf.extend_from_slice(&self.trend_threshold.to_be_bytes());

        buf
    }
}

// ============================================================================
// ROUTER V1: DETERMINISTIC RULE-BASED ROUTER
// ============================================================================

/// Deterministic rule-based router for strategy selection.
///
/// The router classifies market regimes based on fixed-point inputs and
/// selects the most appropriate strategy from configured profiles.
///
/// # Determinism Guarantee
/// Given identical inputs and configuration:
/// - The same regime classification is produced
/// - The same strategy is selected
/// - The same RouterDecisionEvent is emitted
///
/// # Algorithm (LOCKED - do not change without version bump):
/// 1. Compute tiers from raw inputs using thresholds
/// 2. Classify primary regime (priority: Halt > LowLiquidity > HighVol > FundingSkew > Trending > Normal)
/// 3. Find highest-priority enabled strategy suitable for that regime
/// 4. Fall back to default strategy if no match
pub struct Router {
    /// Router configuration
    config: RouterConfig,
    /// Cached config hash
    config_hash: String,
}

impl Router {
    /// Create a new router with the given configuration.
    pub fn new(config: RouterConfig) -> Self {
        let config_hash = config.compute_hash();
        Self {
            config,
            config_hash,
        }
    }

    /// Get the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get the router config hash.
    pub fn config_hash(&self) -> &str {
        &self.config_hash
    }

    /// Classify regime from inputs.
    ///
    /// # Algorithm (priority order):
    /// 1. HALT: liquidity_tier == 3 (empty order books)
    /// 2. LOW_LIQUIDITY: liquidity_tier >= 2 (thin books)
    /// 3. HIGH_VOL: volatility_tier >= 2 (high volatility)
    /// 4. FUNDING_SKEW: funding_tier >= 2 (elevated funding)
    /// 5. TRENDING: |trend_strength| >= threshold
    /// 6. MEAN_REVERT: spread_tier == 0 && volatility_tier == 0 (tight spread, low vol)
    /// 7. NORMAL: default
    pub fn classify_regime(&self, inputs: &RegimeInputs) -> (RegimeLabel, u32) {
        let thresholds = &self.config.thresholds;

        // Priority 1: HALT (empty liquidity)
        if inputs.liquidity_tier >= 3 {
            return (RegimeLabel::Halt, 10000); // 100% confidence
        }

        // Priority 2: LOW_LIQUIDITY (thin books)
        if inputs.liquidity_tier >= 2 {
            return (RegimeLabel::LowLiquidity, 9000);
        }

        // Priority 3: HIGH_VOL (extreme volatility)
        if inputs.volatility_tier >= 2 || inputs.volatility_bps >= thresholds.vol_high_bps {
            let confidence = if inputs.volatility_tier >= 3 {
                9500
            } else {
                8000
            };
            return (RegimeLabel::HighVol, confidence);
        }

        // Priority 4: FUNDING_SKEW (crypto extreme funding)
        if inputs.funding_tier >= 2
            || inputs.funding_rate_bps.abs() >= thresholds.funding_elevated_bps
        {
            let confidence = if inputs.funding_tier >= 3 { 9000 } else { 7500 };
            return (RegimeLabel::FundingSkew, confidence);
        }

        // Priority 5: TRENDING (strong directional move)
        if inputs.trend_strength.abs() >= thresholds.trend_threshold {
            let confidence = inputs.trend_strength.unsigned_abs().min(9000);
            return (RegimeLabel::Trending, confidence);
        }

        // Priority 6: MEAN_REVERT (tight spread, low vol)
        if inputs.spread_tier == 0 && inputs.volatility_tier <= 1 {
            return (RegimeLabel::MeanRevert, 6000);
        }

        // Default: NORMAL
        (RegimeLabel::Normal, 8000)
    }

    /// Select a strategy for the given regime.
    ///
    /// Returns (strategy_id, alternatives, reason).
    pub fn select_strategy(&self, regime: RegimeLabel) -> (String, Vec<String>, String) {
        // Find all enabled strategies suitable for this regime
        let mut candidates: Vec<&StrategyProfile> = self
            .config
            .strategy_profiles
            .iter()
            .filter(|p| p.enabled && p.suitable_regimes.contains(&regime))
            .collect();

        // Sort by priority (descending)
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority));

        if let Some(best) = candidates.first() {
            let alternatives: Vec<String> = candidates
                .iter()
                .skip(1)
                .map(|p| p.strategy_id.clone())
                .collect();

            let reason = format!(
                "Selected {} for {} regime (priority {})",
                best.name,
                regime.as_str(),
                best.priority
            );

            (best.strategy_id.clone(), alternatives, reason)
        } else {
            // Fall back to default
            let reason = format!(
                "No strategy configured for {} regime, using default",
                regime.as_str()
            );
            (self.config.default_strategy_id.clone(), Vec::new(), reason)
        }
    }

    /// Route: classify regime and select strategy.
    ///
    /// This is the main entry point for routing decisions.
    /// Returns a fully-populated RouterDecisionEvent.
    pub fn route(&self, inputs: RegimeInputs, symbols: Vec<String>) -> RouterDecisionEvent {
        let (regime, confidence_bps) = self.classify_regime(&inputs);
        let (selected_strategy_id, alternatives, selection_reason) = self.select_strategy(regime);

        RouterDecisionEvent {
            ts_ns: inputs.ts_ns,
            decision_id: uuid::Uuid::new_v4(),
            symbols,
            inputs,
            regime,
            confidence_bps,
            selected_strategy_id,
            alternatives,
            selection_reason,
            router_config_hash: self.config_hash.clone(),
            router_version: RouterConfig::VERSION.to_string(),
        }
    }

    /// Route with a specific decision ID (for replay determinism).
    ///
    /// Use this when replaying to ensure the same decision_id is used.
    pub fn route_with_id(
        &self,
        inputs: RegimeInputs,
        symbols: Vec<String>,
        decision_id: uuid::Uuid,
    ) -> RouterDecisionEvent {
        let (regime, confidence_bps) = self.classify_regime(&inputs);
        let (selected_strategy_id, alternatives, selection_reason) = self.select_strategy(regime);

        RouterDecisionEvent {
            ts_ns: inputs.ts_ns,
            decision_id,
            symbols,
            inputs,
            regime,
            confidence_bps,
            selected_strategy_id,
            alternatives,
            selection_reason,
            router_config_hash: self.config_hash.clone(),
            router_version: RouterConfig::VERSION.to_string(),
        }
    }
}

/// Builder for RouterConfig with fluent API.
#[derive(Debug, Default)]
pub struct RouterConfigBuilder {
    config: RouterConfig,
}

impl RouterConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: RouterConfig::new(),
        }
    }

    /// Set the default strategy ID.
    pub fn default_strategy(mut self, strategy_id: &str) -> Self {
        self.config.default_strategy_id = strategy_id.to_string();
        self
    }

    /// Add a strategy profile.
    pub fn add_profile(mut self, profile: StrategyProfile) -> Self {
        self.config.strategy_profiles.push(profile);
        self
    }

    /// Set custom thresholds.
    pub fn thresholds(mut self, thresholds: RegimeThresholds) -> Self {
        self.config.thresholds = thresholds;
        self
    }

    /// Build the RouterConfig.
    pub fn build(self) -> RouterConfig {
        self.config
    }
}

// ============================================================================
// STRATEGY SANDBOXING (IPC MODELS)
// ============================================================================

/// Event sent from Runner to isolated Strategy Host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostEvent {
    Start,
    Tick(MarketEvent),
    Bar(MarketEvent),
    Order(OrderEvent),
    Risk(RiskEvent),
    Timer(u64),
    Stop,
}

/// Response/Signal sent from isolated Strategy Host to Runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostResponse {
    Signal(SignalEvent),
    Ready,
}

// ============================================================================
// G2 ROBUSTNESS GATES (PHASE 6A)
// ============================================================================

/// Time-shift degradation test result.
///
/// Tests whether strategy performance degrades appropriately when
/// execution is delayed by k events/ticks. Overfit strategies often
/// show abnormal sensitivity to execution timing.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeShiftResult {
    /// Shift amount in ticks/events
    pub shift_k: u32,
    /// Alpha score with this shift applied
    pub score_mantissa: i128,
    /// Score exponent (matches AlphaScoreV1)
    pub score_exponent: i8,
    /// Win rate with shift (basis points)
    pub win_rate_bps: u32,
    /// Total decisions evaluated
    pub total_decisions: u32,
    /// Net PnL with shift (mantissa)
    pub net_pnl_mantissa: i128,
    /// PnL exponent
    pub pnl_exponent: i8,
    /// Degradation ratio vs base (basis points, 10000 = 100%)
    pub degradation_ratio_bps: u32,
    /// Summary hash for this shifted run
    pub summary_sha256: String,
}

/// Cost sensitivity test result.
///
/// Tests whether strategy survives realistic cost increases.
/// Fragile strategies often collapse at 2x or 5x fees/slippage.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSensitivityResult {
    /// Fee multiplier applied (1, 2, or 5)
    pub fee_multiplier: u32,
    /// Slippage multiplier applied (1, 2, or 5)
    pub slippage_multiplier: u32,
    /// Alpha score with costs applied
    pub score_mantissa: i128,
    /// Score exponent
    pub score_exponent: i8,
    /// Win rate with costs (basis points)
    pub win_rate_bps: u32,
    /// Net PnL with costs (mantissa)
    pub net_pnl_mantissa: i128,
    /// PnL exponent
    pub pnl_exponent: i8,
    /// Retention ratio vs base (basis points, 10000 = 100%)
    pub retention_ratio_bps: u32,
    /// Summary hash for this cost scenario
    pub summary_sha256: String,
}

/// Random baseline comparison result.
///
/// Tests whether strategy outperforms a deterministic random baseline
/// with identical constraints (same number of decisions, holding period).
/// Detects "market drift harvesting" masquerading as alpha.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomBaselineResult {
    /// Seed used for deterministic PRNG (derived from segment/run hash)
    pub seed_hex: String,
    /// Baseline alpha score
    pub baseline_score_mantissa: i128,
    /// Baseline score exponent
    pub baseline_score_exponent: i8,
    /// Baseline win rate (basis points)
    pub baseline_win_rate_bps: u32,
    /// Baseline net PnL (mantissa)
    pub baseline_net_pnl_mantissa: i128,
    /// Strategy alpha score (for comparison)
    pub strategy_score_mantissa: i128,
    /// Edge ratio: strategy_score / baseline_score (basis points, 10000 = 1.0x)
    pub edge_ratio_bps: u32,
    /// Absolute edge: strategy_score - baseline_score
    pub absolute_edge_mantissa: i128,
    /// Baseline summary hash
    pub baseline_summary_sha256: String,
}

/// G2 gate thresholds configuration.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Thresholds {
    /// Maximum allowed score retention at k=1 shift (basis points, e.g., 9000 = 90%)
    pub max_shift_1_retention_bps: u32,
    /// Maximum allowed score retention at k=3 shift (basis points, e.g., 7500 = 75%)
    pub max_shift_3_retention_bps: u32,
    /// Maximum allowed score retention at k=5 shift (basis points, e.g., 6000 = 60%)
    pub max_shift_5_retention_bps: u32,
    /// Minimum score retention at 2x/2x costs (basis points, e.g., 5000 = 50%)
    pub min_cost_2x_retention_bps: u32,
    /// Minimum score retention at 5x/5x costs (basis points, e.g., 2000 = 20%)
    pub min_cost_5x_retention_bps: u32,
    /// Minimum edge ratio vs baseline (basis points, e.g., 12500 = 1.25x)
    pub min_baseline_edge_ratio_bps: u32,
}

impl Default for G2Thresholds {
    fn default() -> Self {
        Self {
            max_shift_1_retention_bps: 9000, // Score should degrade to ≤90% at k=1
            max_shift_3_retention_bps: 7500, // ≤75% at k=3
            max_shift_5_retention_bps: 6000, // ≤60% at k=5
            min_cost_2x_retention_bps: 5000, // ≥50% at 2x costs
            min_cost_5x_retention_bps: 2000, // ≥20% at 5x costs
            min_baseline_edge_ratio_bps: 12500, // ≥1.25x vs baseline
        }
    }
}

/// G2 Robustness Report.
///
/// Comprehensive report of all G2 robustness tests.
/// This artifact is written as g2_report.json and bound to manifest.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Report {
    /// Report version for schema evolution
    pub version: String,
    /// Timestamp when report was generated (nanoseconds)
    pub generated_ts_ns: i64,
    /// Strategy ID being evaluated
    pub strategy_id: String,
    /// Run ID for traceability
    pub run_id: String,

    // === Base Run Reference ===
    /// Base (unmodified) run summary hash
    pub base_summary_sha256: String,
    /// Base alpha score (mantissa)
    pub base_score_mantissa: i128,
    /// Base score exponent
    pub base_score_exponent: i8,

    // === Time-Shift Results ===
    /// Time-shift test results for k=1,3,5
    pub time_shift_results: Vec<TimeShiftResult>,
    /// Whether time-shift tests passed
    pub time_shift_passed: bool,
    /// Reasons for time-shift failure (empty if passed)
    pub time_shift_reasons: Vec<String>,

    // === Cost Sensitivity Results ===
    /// Cost sensitivity results for various multiplier combinations
    pub cost_sensitivity_results: Vec<CostSensitivityResult>,
    /// Whether cost sensitivity tests passed
    pub cost_sensitivity_passed: bool,
    /// Reasons for cost sensitivity failure
    pub cost_sensitivity_reasons: Vec<String>,

    // === Random Baseline Results ===
    /// Random baseline comparison result
    pub baseline_result: RandomBaselineResult,
    /// Whether baseline comparison passed
    pub baseline_passed: bool,
    /// Reasons for baseline failure
    pub baseline_reasons: Vec<String>,

    // === Overall Result ===
    /// Thresholds used for evaluation
    pub thresholds: G2Thresholds,
    /// Overall G2 pass/fail
    pub passed: bool,
    /// Combined reasons for any failures
    pub all_reasons: Vec<String>,
}

impl G2Report {
    /// G2 Report version constant.
    pub const VERSION: &'static str = "g2_report_v1.0";

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2048);

        // Version
        buf.extend_from_slice(&(self.version.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.version.as_bytes());

        // Timestamps and IDs
        buf.extend_from_slice(&self.generated_ts_ns.to_be_bytes());
        buf.extend_from_slice(&(self.strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.strategy_id.as_bytes());
        buf.extend_from_slice(&(self.run_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.run_id.as_bytes());

        // Base reference
        buf.extend_from_slice(&(self.base_summary_sha256.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.base_summary_sha256.as_bytes());
        buf.extend_from_slice(&self.base_score_mantissa.to_be_bytes());
        buf.push(self.base_score_exponent as u8);

        // Time-shift results (sorted by shift_k for determinism)
        buf.extend_from_slice(&(self.time_shift_results.len() as u32).to_be_bytes());
        for result in &self.time_shift_results {
            buf.extend_from_slice(&result.shift_k.to_be_bytes());
            buf.extend_from_slice(&result.score_mantissa.to_be_bytes());
            buf.extend_from_slice(&result.degradation_ratio_bps.to_be_bytes());
        }
        buf.push(self.time_shift_passed as u8);

        // Cost sensitivity results (sorted by fee_mult, then slip_mult)
        buf.extend_from_slice(&(self.cost_sensitivity_results.len() as u32).to_be_bytes());
        for result in &self.cost_sensitivity_results {
            buf.extend_from_slice(&result.fee_multiplier.to_be_bytes());
            buf.extend_from_slice(&result.slippage_multiplier.to_be_bytes());
            buf.extend_from_slice(&result.score_mantissa.to_be_bytes());
            buf.extend_from_slice(&result.retention_ratio_bps.to_be_bytes());
        }
        buf.push(self.cost_sensitivity_passed as u8);

        // Baseline result
        buf.extend_from_slice(&(self.baseline_result.seed_hex.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.baseline_result.seed_hex.as_bytes());
        buf.extend_from_slice(&self.baseline_result.baseline_score_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.baseline_result.strategy_score_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.baseline_result.edge_ratio_bps.to_be_bytes());
        buf.push(self.baseline_passed as u8);

        // Overall
        buf.push(self.passed as u8);

        buf
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(self.canonical_bytes());
        hex::encode(hash)
    }
}

// ============================================================================
// G3 WALK-FORWARD STABILITY (PHASE 6B)
// ============================================================================

/// Walk-forward fold result.
///
/// Represents metrics for a single chronological fold in walk-forward testing.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardFold {
    /// Fold index (0-based)
    pub fold_index: u32,
    /// Fold start timestamp (nanoseconds)
    pub start_ts_ns: i64,
    /// Fold end timestamp (nanoseconds)
    pub end_ts_ns: i64,
    /// Number of decisions in this fold
    pub num_decisions: u32,
    /// Alpha score for this fold (mantissa)
    pub score_mantissa: i128,
    /// Score exponent
    pub score_exponent: i8,
    /// Win rate for this fold (basis points)
    pub win_rate_bps: u32,
    /// Net PnL for this fold (mantissa)
    pub net_pnl_mantissa: i128,
    /// PnL exponent
    pub pnl_exponent: i8,
    /// Max loss in this fold (mantissa)
    pub max_loss_mantissa: i128,
    /// Fold summary hash
    pub summary_sha256: String,
}

/// Walk-forward stability metrics.
///
/// Aggregate statistics across all folds to assess consistency.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Median score across folds (mantissa)
    pub median_score_mantissa: i128,
    /// Minimum score across folds (mantissa)
    pub min_score_mantissa: i128,
    /// Maximum score across folds (mantissa)
    pub max_score_mantissa: i128,
    /// Score dispersion: max - min (mantissa)
    pub score_dispersion_mantissa: i128,
    /// Interquartile range of scores (mantissa)
    pub score_iqr_mantissa: i128,
    /// Median win rate across folds (basis points)
    pub median_win_rate_bps: u32,
    /// Minimum win rate across folds (basis points)
    pub min_win_rate_bps: u32,
    /// Number of profitable folds
    pub profitable_folds: u32,
    /// Total folds
    pub total_folds: u32,
    /// Consistency ratio: profitable_folds / total_folds (basis points)
    pub consistency_ratio_bps: u32,
}

/// G3 gate thresholds configuration.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3Thresholds {
    /// Minimum median score (must meet G1 threshold)
    pub min_median_score_mantissa: i128,
    /// Minimum score in any fold (0 = must not lose, negative = some loss ok)
    pub min_fold_score_mantissa: i128,
    /// Maximum allowed dispersion (max - min) as ratio of median (basis points)
    pub max_dispersion_ratio_bps: u32,
    /// Minimum consistency ratio (basis points, e.g., 6000 = 60% of folds profitable)
    pub min_consistency_ratio_bps: u32,
    /// Number of folds for walk-forward
    pub num_folds: u32,
}

impl Default for G3Thresholds {
    fn default() -> Self {
        Self {
            min_median_score_mantissa: 1000, // Same as G1 default
            min_fold_score_mantissa: 0,      // No fold should be negative
            max_dispersion_ratio_bps: 20000, // Max 200% dispersion relative to median
            min_consistency_ratio_bps: 6000, // At least 60% of folds profitable
            num_folds: 5,                    // Default K=5
        }
    }
}

/// G3 Walk-Forward Report.
///
/// Comprehensive report of walk-forward stability analysis.
/// This artifact is written as g3_walkforward.json and bound to manifest.
///
/// # Fields (LOCKED - do not change without version bump)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3Report {
    /// Report version for schema evolution
    pub version: String,
    /// Timestamp when report was generated (nanoseconds)
    pub generated_ts_ns: i64,
    /// Strategy ID being evaluated
    pub strategy_id: String,
    /// Run ID for traceability
    pub run_id: String,

    // === Segment Reference ===
    /// Full segment start timestamp (nanoseconds)
    pub segment_start_ts_ns: i64,
    /// Full segment end timestamp (nanoseconds)
    pub segment_end_ts_ns: i64,

    // === Fold Results ===
    /// Per-fold results (ordered by fold_index)
    pub folds: Vec<WalkForwardFold>,

    // === Stability Metrics ===
    /// Aggregate stability metrics
    pub stability_metrics: StabilityMetrics,

    // === Gate Decision ===
    /// Thresholds used for evaluation
    pub thresholds: G3Thresholds,
    /// Whether G3 gate passed
    pub passed: bool,
    /// Reasons for failure (empty if passed)
    pub reasons: Vec<String>,
}

impl G3Report {
    /// G3 Report version constant.
    pub const VERSION: &'static str = "g3_report_v1.0";

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1024);

        // Version
        buf.extend_from_slice(&(self.version.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.version.as_bytes());

        // Timestamps and IDs
        buf.extend_from_slice(&self.generated_ts_ns.to_be_bytes());
        buf.extend_from_slice(&(self.strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.strategy_id.as_bytes());
        buf.extend_from_slice(&(self.run_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.run_id.as_bytes());

        // Segment bounds
        buf.extend_from_slice(&self.segment_start_ts_ns.to_be_bytes());
        buf.extend_from_slice(&self.segment_end_ts_ns.to_be_bytes());

        // Folds (already ordered by fold_index)
        buf.extend_from_slice(&(self.folds.len() as u32).to_be_bytes());
        for fold in &self.folds {
            buf.extend_from_slice(&fold.fold_index.to_be_bytes());
            buf.extend_from_slice(&fold.start_ts_ns.to_be_bytes());
            buf.extend_from_slice(&fold.end_ts_ns.to_be_bytes());
            buf.extend_from_slice(&fold.num_decisions.to_be_bytes());
            buf.extend_from_slice(&fold.score_mantissa.to_be_bytes());
            buf.extend_from_slice(&fold.win_rate_bps.to_be_bytes());
            buf.extend_from_slice(&fold.net_pnl_mantissa.to_be_bytes());
        }

        // Stability metrics
        buf.extend_from_slice(&self.stability_metrics.median_score_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.stability_metrics.min_score_mantissa.to_be_bytes());
        buf.extend_from_slice(&self.stability_metrics.max_score_mantissa.to_be_bytes());
        buf.extend_from_slice(
            &self
                .stability_metrics
                .score_dispersion_mantissa
                .to_be_bytes(),
        );
        buf.extend_from_slice(&self.stability_metrics.consistency_ratio_bps.to_be_bytes());

        // Result
        buf.push(self.passed as u8);

        buf
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(self.canonical_bytes());
        hex::encode(hash)
    }
}

// ============================================================================
// G2/G3 HARNESS (PHASE 6)
// ============================================================================

/// G2 Robustness Gate evaluator.
///
/// Evaluates strategy robustness using pre-computed test results.
/// The actual test execution happens in the runner; this evaluates results.
#[derive(Debug, Clone)]
pub struct G2Gate {
    /// Gate version
    pub version: String,
    /// Thresholds for evaluation
    pub thresholds: G2Thresholds,
}

impl G2Gate {
    /// G2 Gate version constant.
    pub const VERSION: &'static str = "g2_gate_v1.0";

    /// Create a new G2 gate with default thresholds.
    pub fn new() -> Self {
        Self {
            version: Self::VERSION.to_string(),
            thresholds: G2Thresholds::default(),
        }
    }

    /// Create a G2 gate with custom thresholds.
    pub fn with_thresholds(thresholds: G2Thresholds) -> Self {
        Self {
            version: Self::VERSION.to_string(),
            thresholds,
        }
    }

    /// Evaluate time-shift test results.
    ///
    /// Returns (passed, reasons).
    pub fn evaluate_time_shift(
        &self,
        base_score: i128,
        results: &[TimeShiftResult],
    ) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        if base_score <= 0 {
            reasons.push("Base score is not positive".to_string());
            return (false, reasons);
        }

        for result in results {
            let retention_bps = ((result.score_mantissa * 10000) / base_score) as u32;
            let max_retention = match result.shift_k {
                1 => self.thresholds.max_shift_1_retention_bps,
                3 => self.thresholds.max_shift_3_retention_bps,
                5 => self.thresholds.max_shift_5_retention_bps,
                _ => continue,
            };

            // For time-shift, we want score to DEGRADE (retention should be BELOW max)
            if retention_bps > max_retention {
                reasons.push(format!(
                    "Time-shift k={}: score retention {}bps > max {}bps (insufficient degradation)",
                    result.shift_k, retention_bps, max_retention
                ));
            }
        }

        (reasons.is_empty(), reasons)
    }

    /// Evaluate cost sensitivity test results.
    ///
    /// Returns (passed, reasons).
    pub fn evaluate_cost_sensitivity(
        &self,
        base_score: i128,
        results: &[CostSensitivityResult],
    ) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        if base_score <= 0 {
            reasons.push("Base score is not positive".to_string());
            return (false, reasons);
        }

        for result in results {
            let retention_bps = if result.score_mantissa <= 0 {
                0
            } else {
                ((result.score_mantissa * 10000) / base_score) as u32
            };

            let min_retention = match (result.fee_multiplier, result.slippage_multiplier) {
                (2, 2) => self.thresholds.min_cost_2x_retention_bps,
                (5, 5) => self.thresholds.min_cost_5x_retention_bps,
                _ => 0, // No threshold for other combinations
            };

            if min_retention > 0 && retention_bps < min_retention {
                reasons.push(format!(
                    "Cost {}x/{}x: retention {}bps < min {}bps",
                    result.fee_multiplier, result.slippage_multiplier, retention_bps, min_retention
                ));
            }
        }

        (reasons.is_empty(), reasons)
    }

    /// Evaluate random baseline comparison.
    ///
    /// Returns (passed, reasons).
    pub fn evaluate_baseline(&self, result: &RandomBaselineResult) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        if result.edge_ratio_bps < self.thresholds.min_baseline_edge_ratio_bps {
            reasons.push(format!(
                "Baseline edge ratio {}bps < min {}bps",
                result.edge_ratio_bps, self.thresholds.min_baseline_edge_ratio_bps
            ));
        }

        (reasons.is_empty(), reasons)
    }
}

impl Default for G2Gate {
    fn default() -> Self {
        Self::new()
    }
}

/// G3 Walk-Forward Gate evaluator.
///
/// Evaluates strategy stability using walk-forward fold results.
#[derive(Debug, Clone)]
pub struct G3Gate {
    /// Gate version
    pub version: String,
    /// Thresholds for evaluation
    pub thresholds: G3Thresholds,
}

impl G3Gate {
    /// G3 Gate version constant.
    pub const VERSION: &'static str = "g3_gate_v1.0";

    /// Create a new G3 gate with default thresholds.
    pub fn new() -> Self {
        Self {
            version: Self::VERSION.to_string(),
            thresholds: G3Thresholds::default(),
        }
    }

    /// Create a G3 gate with custom thresholds.
    pub fn with_thresholds(thresholds: G3Thresholds) -> Self {
        Self {
            version: Self::VERSION.to_string(),
            thresholds,
        }
    }

    /// Compute stability metrics from fold results.
    pub fn compute_stability_metrics(&self, folds: &[WalkForwardFold]) -> StabilityMetrics {
        if folds.is_empty() {
            return StabilityMetrics {
                median_score_mantissa: 0,
                min_score_mantissa: 0,
                max_score_mantissa: 0,
                score_dispersion_mantissa: 0,
                score_iqr_mantissa: 0,
                median_win_rate_bps: 0,
                min_win_rate_bps: 0,
                profitable_folds: 0,
                total_folds: 0,
                consistency_ratio_bps: 0,
            };
        }

        // Collect scores and sort for percentile calculations
        let mut scores: Vec<i128> = folds.iter().map(|f| f.score_mantissa).collect();
        scores.sort();

        let mut win_rates: Vec<u32> = folds.iter().map(|f| f.win_rate_bps).collect();
        win_rates.sort();

        let n = scores.len();
        let median_score = scores[n / 2];
        let min_score = scores[0];
        let max_score = scores[n - 1];
        let dispersion = max_score - min_score;

        // IQR: Q3 - Q1
        let q1 = scores[n / 4];
        let q3 = scores[(3 * n) / 4];
        let iqr = q3 - q1;

        let median_win_rate = win_rates[n / 2];
        let min_win_rate = win_rates[0];

        let profitable_folds = folds.iter().filter(|f| f.score_mantissa > 0).count() as u32;
        let total_folds = folds.len() as u32;
        let consistency_ratio_bps = (profitable_folds * 10000) / total_folds;

        StabilityMetrics {
            median_score_mantissa: median_score,
            min_score_mantissa: min_score,
            max_score_mantissa: max_score,
            score_dispersion_mantissa: dispersion,
            score_iqr_mantissa: iqr,
            median_win_rate_bps: median_win_rate,
            min_win_rate_bps: min_win_rate,
            profitable_folds,
            total_folds,
            consistency_ratio_bps,
        }
    }

    /// Evaluate stability metrics against thresholds.
    ///
    /// Returns (passed, reasons).
    pub fn evaluate(&self, metrics: &StabilityMetrics) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        // Check median score
        if metrics.median_score_mantissa < self.thresholds.min_median_score_mantissa {
            reasons.push(format!(
                "Median score {} < min {}",
                metrics.median_score_mantissa, self.thresholds.min_median_score_mantissa
            ));
        }

        // Check min fold score
        if metrics.min_score_mantissa < self.thresholds.min_fold_score_mantissa {
            reasons.push(format!(
                "Min fold score {} < min {}",
                metrics.min_score_mantissa, self.thresholds.min_fold_score_mantissa
            ));
        }

        // Check dispersion (only if median is positive to avoid division issues)
        if metrics.median_score_mantissa > 0 {
            let dispersion_ratio_bps = ((metrics.score_dispersion_mantissa.abs() * 10000)
                / metrics.median_score_mantissa) as u32;
            if dispersion_ratio_bps > self.thresholds.max_dispersion_ratio_bps {
                reasons.push(format!(
                    "Score dispersion {}bps > max {}bps",
                    dispersion_ratio_bps, self.thresholds.max_dispersion_ratio_bps
                ));
            }
        }

        // Check consistency
        if metrics.consistency_ratio_bps < self.thresholds.min_consistency_ratio_bps {
            reasons.push(format!(
                "Consistency ratio {}bps < min {}bps",
                metrics.consistency_ratio_bps, self.thresholds.min_consistency_ratio_bps
            ));
        }

        (reasons.is_empty(), reasons)
    }
}

impl Default for G3Gate {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined promotion decision record.
///
/// Final artifact recording the full promotion evaluation: G1 + G2 + G3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionDecision {
    /// Decision version
    pub version: String,
    /// Timestamp of decision (nanoseconds)
    pub decision_ts_ns: i64,
    /// Strategy ID
    pub strategy_id: String,
    /// Run ID
    pub run_id: String,

    // === Gate Results ===
    /// G1 gate version used
    pub g1_version: String,
    /// G1 passed
    pub g1_passed: bool,
    /// G1 failure reasons
    pub g1_reasons: Vec<String>,

    /// G2 gate version used
    pub g2_version: String,
    /// G2 passed
    pub g2_passed: bool,
    /// G2 failure reasons
    pub g2_reasons: Vec<String>,
    /// G2 report hash
    pub g2_report_sha256: String,

    /// G3 gate version used
    pub g3_version: String,
    /// G3 passed
    pub g3_passed: bool,
    /// G3 failure reasons
    pub g3_reasons: Vec<String>,
    /// G3 report hash
    pub g3_report_sha256: String,

    // === Overall Decision ===
    /// Final promotion decision
    pub promoted: bool,
    /// All failure reasons combined
    pub all_reasons: Vec<String>,
}

impl PromotionDecision {
    /// Promotion decision version constant.
    pub const VERSION: &'static str = "promotion_v1.0";

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        buf.extend_from_slice(&(self.version.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.version.as_bytes());
        buf.extend_from_slice(&self.decision_ts_ns.to_be_bytes());
        buf.extend_from_slice(&(self.strategy_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.strategy_id.as_bytes());
        buf.extend_from_slice(&(self.run_id.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.run_id.as_bytes());

        // Gate results
        buf.push(self.g1_passed as u8);
        buf.push(self.g2_passed as u8);
        buf.extend_from_slice(&(self.g2_report_sha256.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.g2_report_sha256.as_bytes());
        buf.push(self.g3_passed as u8);
        buf.extend_from_slice(&(self.g3_report_sha256.len() as u32).to_be_bytes());
        buf.extend_from_slice(self.g3_report_sha256.as_bytes());

        buf.push(self.promoted as u8);

        buf
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(self.canonical_bytes());
        hex::encode(hash)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution_accumulator_buy_sell_round_trip() {
        // Test a simple buy/sell round trip
        let decision_id = Uuid::new_v4();
        let mut acc = DecisionAttributionAccumulator::new(
            decision_id,
            "test_strategy:1.0.0:abc123".to_string(),
            "BTCUSDT".to_string(),
            "paper".to_string(),
            10_000_000, // $100,000 mid price (exp -2)
            -2,
            -8,
        );

        // Buy 0.01 BTC at $100,000
        // price_mantissa = 10_000_000 (exp -2)
        // qty_mantissa = 1_000_000 (exp -8 = 0.01 BTC)
        // fee_mantissa = 10_000 (exp -8)
        acc.add_fill(1_000_000_000, 10_000_000, 1_000_000, 10_000, true);

        assert_eq!(acc.num_fills, 1);
        assert_eq!(acc.signed_qty_mantissa, 1_000_000);
        assert!(!acc.is_closed());

        // Sell 0.01 BTC at $101,000
        // price_mantissa = 10_100_000 (exp -2)
        acc.add_fill(2_000_000_000, 10_100_000, 1_000_000, 10_000, false);

        assert_eq!(acc.num_fills, 2);
        assert_eq!(acc.signed_qty_mantissa, 0);
        assert!(acc.is_closed());

        // Flush to attribution event
        let event = acc.flush(2_000_000_000);

        assert_eq!(event.parent_decision_id, decision_id);
        assert_eq!(event.num_fills, 2);
        assert_eq!(event.holding_time_ns, 1_000_000_000);

        // PnL calculation:
        // Cost basis: 10_000_000 * 1_000_000 / 100 = 100_000_000_000 (exp -8)
        // Proceeds: 10_100_000 * 1_000_000 / 100 = 101_000_000_000 (exp -8)
        // Gross: 101_000_000_000 - 100_000_000_000 = 1_000_000_000 (exp -8 = $10)
        // Net: 1_000_000_000 - 20_000 (fees) = 999_980_000 (exp -8)

        assert_eq!(event.gross_pnl_mantissa, 1_000_000_000);
        assert_eq!(event.fees_mantissa, 20_000);
        assert_eq!(event.net_pnl_mantissa, 1_000_000_000 - 20_000);

        // Verify f64 conversions
        let gross_pnl_f64 = event.gross_pnl_f64();
        assert!(
            (gross_pnl_f64 - 10.0).abs() < 0.01,
            "Expected ~$10 gross PnL, got {}",
            gross_pnl_f64
        );
    }

    #[test]
    fn test_attribution_accumulator_slippage() {
        let decision_id = Uuid::new_v4();
        let mut acc = DecisionAttributionAccumulator::new(
            decision_id,
            "test_strategy".to_string(),
            "BTCUSDT".to_string(),
            "paper".to_string(),
            10_000_000, // $100,000 decision mid
            -2,
            -8,
        );

        // Buy at $100,100 (100 basis points above mid)
        // Slippage = (100100 - 100000) * 0.01 = $1 unfavorable
        acc.add_fill(1_000_000_000, 10_010_000, 1_000_000, 10_000, true);

        let event = acc.flush(1_000_000_000);

        // Slippage: (10_010_000 - 10_000_000) * 1_000_000 / 100 = 100_000_000 (exp -8 = $1)
        assert_eq!(event.slippage_mantissa, 100_000_000);
        assert!(
            (event.slippage_f64() - 1.0).abs() < 0.01,
            "Expected ~$1 slippage, got {}",
            event.slippage_f64()
        );
    }

    #[test]
    fn test_attribution_event_serialization() {
        let event = TradeAttributionEvent {
            ts_ns: 1_234_567_890_000_000_000,
            symbol: "BTCUSDT".to_string(),
            venue: "paper".to_string(),
            parent_decision_id: Uuid::new_v4(),
            strategy_id: "funding_bias:2.0.0:abc123".to_string(),
            gross_pnl_mantissa: 1_000_000_000,
            fees_mantissa: 20_000,
            net_pnl_mantissa: 999_980_000,
            pnl_exponent: -8,
            holding_time_ns: 1_000_000_000,
            num_fills: 2,
            slippage_mantissa: 100_000_000,
            slippage_exponent: -8,
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: TradeAttributionEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.gross_pnl_mantissa, event.gross_pnl_mantissa);
        assert_eq!(parsed.fees_mantissa, event.fees_mantissa);
        assert_eq!(parsed.net_pnl_mantissa, event.net_pnl_mantissa);
        assert_eq!(parsed.slippage_mantissa, event.slippage_mantissa);
    }

    #[test]
    fn test_attribution_determinism() {
        // Verify identical fills produce identical attribution
        let decision_id = Uuid::new_v4();

        let make_acc = || {
            let mut acc = DecisionAttributionAccumulator::new(
                decision_id,
                "test".to_string(),
                "BTCUSDT".to_string(),
                "paper".to_string(),
                10_000_000,
                -2,
                -8,
            );
            acc.add_fill(1_000, 10_000_000, 1_000_000, 10_000, true);
            acc.add_fill(2_000, 10_100_000, 1_000_000, 10_000, false);
            acc
        };

        let event1 = make_acc().flush(2_000);
        let event2 = make_acc().flush(2_000);

        assert_eq!(event1.gross_pnl_mantissa, event2.gross_pnl_mantissa);
        assert_eq!(event1.fees_mantissa, event2.fees_mantissa);
        assert_eq!(event1.net_pnl_mantissa, event2.net_pnl_mantissa);
        assert_eq!(event1.slippage_mantissa, event2.slippage_mantissa);
    }

    // =========================================================================
    // Phase 4: Attribution Summary + Alpha Score Tests
    // =========================================================================

    #[test]
    fn test_attribution_summary_builder() {
        let mut builder = AttributionSummaryBuilder::new(
            "test_strategy:1.0:abc".to_string(),
            "run_001".to_string(),
            vec!["BTCUSDT".to_string()],
            -8,
        );

        // Add winning event (+$10)
        let event1 = TradeAttributionEvent {
            ts_ns: 1_000,
            symbol: "BTCUSDT".to_string(),
            venue: "paper".to_string(),
            parent_decision_id: Uuid::new_v4(),
            strategy_id: "test_strategy:1.0:abc".to_string(),
            gross_pnl_mantissa: 1_000_000_000, // $10
            fees_mantissa: 20_000,             // $0.0002
            net_pnl_mantissa: 999_980_000,     // ~$9.9998
            pnl_exponent: -8,
            holding_time_ns: 1_000_000_000,
            num_fills: 2,
            slippage_mantissa: 50_000_000,
            slippage_exponent: -8,
        };
        builder.add_event(&event1);

        // Add losing event (-$5)
        let event2 = TradeAttributionEvent {
            ts_ns: 2_000,
            symbol: "BTCUSDT".to_string(),
            venue: "paper".to_string(),
            parent_decision_id: Uuid::new_v4(),
            strategy_id: "test_strategy:1.0:abc".to_string(),
            gross_pnl_mantissa: -500_000_000, // -$5
            fees_mantissa: 15_000,            // $0.00015
            net_pnl_mantissa: -500_015_000,   // -$5.00015
            pnl_exponent: -8,
            holding_time_ns: 500_000_000,
            num_fills: 1,
            slippage_mantissa: -10_000_000,
            slippage_exponent: -8,
        };
        builder.add_event(&event2);

        let summary = builder.build(3_000);

        // Verify counts
        assert_eq!(summary.total_decisions, 2);
        assert_eq!(summary.total_fills, 3);
        assert_eq!(summary.winning_decisions, 1);
        assert_eq!(summary.losing_decisions, 1);

        // Verify PnL aggregates
        assert_eq!(summary.total_gross_pnl_mantissa, 500_000_000); // $10 - $5 = $5
        assert_eq!(summary.total_fees_mantissa, 35_000);
        assert_eq!(summary.total_net_pnl_mantissa, 999_980_000 - 500_015_000);

        // Verify win rate (50% = 5000 bps)
        assert_eq!(summary.win_rate_bps, 5000);

        // Verify max loss (the losing decision)
        assert_eq!(summary.max_loss_mantissa, 500_015_000);

        // Verify holding time
        assert_eq!(summary.total_holding_time_ns, 1_500_000_000);
    }

    #[test]
    fn test_alpha_score_v1_profitable() {
        let summary = AttributionSummary {
            strategy_id: "test".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_000,
            total_decisions: 10,
            total_fills: 20,
            winning_decisions: 7,
            losing_decisions: 3,
            round_trips: 5,
            total_gross_pnl_mantissa: 10_000_000_000, // $100
            total_fees_mantissa: 100_000,             // $0.001
            total_net_pnl_mantissa: 9_999_900_000,    // ~$99.999
            pnl_exponent: -8,
            win_rate_bps: 7000,
            avg_pnl_per_decision_mantissa: 999_990_000,
            total_slippage_mantissa: 500_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 1_000_000_000, // $10 max loss
            total_holding_time_ns: 10_000_000_000,
        };

        let score = AlphaScoreV1::from_summary(&summary);

        // Verify formula: score = (net_pnl * 10000) / (max_loss + epsilon)
        // = (9_999_900_000 * 10000) / (1_000_000_000 + 1_000_000)
        // = 99_999_000_000_000 / 1_001_000_000
        // = 99_899_100 (approximately)
        assert!(score.is_profitable());
        assert!(score.score_mantissa > 0);
        assert_eq!(score.formula_version, "alpha_score_v1.0");
        assert_eq!(score.input_net_pnl_mantissa, 9_999_900_000);
        assert_eq!(score.input_max_loss_mantissa, 1_000_000_000);
    }

    #[test]
    fn test_alpha_score_v1_unprofitable() {
        let summary = AttributionSummary {
            strategy_id: "test".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_000,
            total_decisions: 5,
            total_fills: 10,
            winning_decisions: 1,
            losing_decisions: 4,
            round_trips: 3,
            total_gross_pnl_mantissa: -5_000_000_000, // -$50
            total_fees_mantissa: 50_000,
            total_net_pnl_mantissa: -5_000_050_000, // -$50.0005
            pnl_exponent: -8,
            win_rate_bps: 2000,
            avg_pnl_per_decision_mantissa: -1_000_010_000,
            total_slippage_mantissa: 200_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 2_000_000_000, // $20 max loss
            total_holding_time_ns: 5_000_000_000,
        };

        let score = AlphaScoreV1::from_summary(&summary);

        assert!(!score.is_profitable());
        assert!(score.score_mantissa < 0);
    }

    #[test]
    fn test_alpha_score_v1_determinism() {
        let make_summary = || AttributionSummary {
            strategy_id: "test".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_000,
            total_decisions: 10,
            total_fills: 20,
            winning_decisions: 7,
            losing_decisions: 3,
            round_trips: 5,
            total_gross_pnl_mantissa: 10_000_000_000,
            total_fees_mantissa: 100_000,
            total_net_pnl_mantissa: 9_999_900_000,
            pnl_exponent: -8,
            win_rate_bps: 7000,
            avg_pnl_per_decision_mantissa: 999_990_000,
            total_slippage_mantissa: 500_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 1_000_000_000,
            total_holding_time_ns: 10_000_000_000,
        };

        let score1 = AlphaScoreV1::from_summary(&make_summary());
        let score2 = AlphaScoreV1::from_summary(&make_summary());

        // Must be identical
        assert_eq!(score1.score_mantissa, score2.score_mantissa);
        assert_eq!(score1.score_exponent, score2.score_exponent);

        // JSON serialization must also match
        let json1 = serde_json::to_string(&score1).unwrap();
        let json2 = serde_json::to_string(&score2).unwrap();
        assert_eq!(json1, json2);
    }

    #[test]
    fn test_attribution_summary_serialization() {
        let summary = AttributionSummary {
            strategy_id: "funding_bias:2.0:abc123".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 10,
            total_fills: 25,
            winning_decisions: 6,
            losing_decisions: 4,
            round_trips: 5,
            total_gross_pnl_mantissa: 5_000_000_000,
            total_fees_mantissa: 50_000,
            total_net_pnl_mantissa: 4_999_950_000,
            pnl_exponent: -8,
            win_rate_bps: 6000,
            avg_pnl_per_decision_mantissa: 499_995_000,
            total_slippage_mantissa: 100_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000,
            total_holding_time_ns: 5_000_000_000,
        };

        let json = serde_json::to_string_pretty(&summary).unwrap();
        let parsed: AttributionSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.strategy_id, summary.strategy_id);
        assert_eq!(parsed.total_decisions, summary.total_decisions);
        assert_eq!(
            parsed.total_net_pnl_mantissa,
            summary.total_net_pnl_mantissa
        );
        assert_eq!(parsed.win_rate_bps, summary.win_rate_bps);
    }

    // =========================================================================
    // G1 Promotion Gate Tests (Phase 4D)
    // =========================================================================

    #[test]
    fn test_g1_gate_pass_all_thresholds() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 100, // >> 10 minimum
            total_fills: 200,
            winning_decisions: 60,
            losing_decisions: 40,
            round_trips: 30,
            total_gross_pnl_mantissa: 10_000_000_000, // $100
            total_fees_mantissa: 100_000,
            total_net_pnl_mantissa: 9_999_900_000, // $99.999
            pnl_exponent: -8,
            win_rate_bps: 6000, // 60% >> 40% minimum
            avg_pnl_per_decision_mantissa: 99_999_000,
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000, // $5 max loss (5% of net PnL)
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha_score);

        assert!(result.passed, "Should pass: {}", result.summary());
        assert!(result.reasons.is_empty());
        assert_eq!(result.gate_version, G1PromotionGate::VERSION);
    }

    #[test]
    fn test_g1_gate_fail_low_alpha() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 100,
            total_fills: 200,
            winning_decisions: 60,
            losing_decisions: 40,
            round_trips: 30,
            total_gross_pnl_mantissa: 100_000, // Very small $0.001 PnL
            total_fees_mantissa: 1_000,
            total_net_pnl_mantissa: 99_000, // Very small net PnL
            pnl_exponent: -8,
            win_rate_bps: 6000,
            avg_pnl_per_decision_mantissa: 990,
            total_slippage_mantissa: 1_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000, // Max loss >> net PnL
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha_score);

        assert!(!result.passed, "Should fail due to low alpha score");
        assert!(result.reasons.iter().any(|r| r.contains("Alpha score")));
    }

    #[test]
    fn test_g1_gate_fail_low_win_rate() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 100,
            total_fills: 200,
            winning_decisions: 30,
            losing_decisions: 70,
            round_trips: 30,
            total_gross_pnl_mantissa: 10_000_000_000,
            total_fees_mantissa: 100_000,
            total_net_pnl_mantissa: 9_999_900_000,
            pnl_exponent: -8,
            win_rate_bps: 3000, // 30% < 40% minimum
            avg_pnl_per_decision_mantissa: 99_999_000,
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000,
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha_score);

        assert!(!result.passed, "Should fail due to low win rate");
        assert!(result.reasons.iter().any(|r| r.contains("Win rate")));
    }

    #[test]
    fn test_g1_gate_fail_few_decisions() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 5, // < 10 minimum
            total_fills: 10,
            winning_decisions: 3,
            losing_decisions: 2,
            round_trips: 2,
            total_gross_pnl_mantissa: 10_000_000_000,
            total_fees_mantissa: 100_000,
            total_net_pnl_mantissa: 9_999_900_000,
            pnl_exponent: -8,
            win_rate_bps: 6000,
            avg_pnl_per_decision_mantissa: 1_999_980_000,
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000,
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha_score);

        assert!(!result.passed, "Should fail due to few decisions");
        assert!(result.reasons.iter().any(|r| r.contains("Decisions")));
    }

    #[test]
    fn test_g1_gate_fail_unprofitable() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 100,
            total_fills: 200,
            winning_decisions: 40,
            losing_decisions: 60,
            round_trips: 30,
            total_gross_pnl_mantissa: -1_000_000_000, // -$10 gross
            total_fees_mantissa: 100_000,
            total_net_pnl_mantissa: -1_000_100_000, // -$10.001 net
            pnl_exponent: -8,
            win_rate_bps: 4000,
            avg_pnl_per_decision_mantissa: -10_001_000,
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000,
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha_score);

        assert!(!result.passed, "Should fail due to being unprofitable");
        assert!(
            result
                .reasons
                .iter()
                .any(|r| r.contains("profitable") || r.contains("Alpha"))
        );
    }

    #[test]
    fn test_g1_gate_custom_thresholds() {
        let summary = AttributionSummary {
            strategy_id: "test:1.0:abc".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_234_567_890,
            total_decisions: 50,
            total_fills: 100,
            winning_decisions: 30,
            losing_decisions: 20,
            round_trips: 15,
            total_gross_pnl_mantissa: 5_000_000_000,
            total_fees_mantissa: 50_000,
            total_net_pnl_mantissa: 4_999_950_000,
            pnl_exponent: -8,
            win_rate_bps: 6000,
            avg_pnl_per_decision_mantissa: 99_999_000,
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000,
            total_holding_time_ns: 1_000_000_000,
        };

        let alpha_score = AlphaScoreV1::from_summary(&summary);

        // Create gate with stricter thresholds
        let strict_gate = G1PromotionGate::with_thresholds(
            50000, // Very high alpha requirement
            7000,  // 70% win rate
            100,   // 100 decisions
            2000,  // Max 20% loss
        );
        let result = strict_gate.evaluate(&summary, &alpha_score);
        assert!(!result.passed, "Should fail with strict thresholds");

        // Create gate with looser thresholds
        let loose_gate = G1PromotionGate::with_thresholds(
            100,   // Low alpha requirement
            3000,  // 30% win rate
            5,     // 5 decisions
            10000, // Max 100% loss
        );
        let result = loose_gate.evaluate(&summary, &alpha_score);
        assert!(
            result.passed,
            "Should pass with loose thresholds: {}",
            result.summary()
        );
    }

    #[test]
    fn test_g1_gate_result_serialization() {
        let result = G1PromotionResult {
            passed: true,
            gate_version: "g1_gate_v1.0".to_string(),
            strategy_id: "funding_bias:2.0:abc123".to_string(),
            alpha_score_mantissa: 5000,
            win_rate_bps: 6000,
            total_decisions: 100,
            reasons: Vec::new(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: G1PromotionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.passed, result.passed);
        assert_eq!(parsed.strategy_id, result.strategy_id);
        assert_eq!(parsed.alpha_score_mantissa, result.alpha_score_mantissa);
    }

    // =========================================================================
    // Router V1 Tests (Phase 5)
    // =========================================================================

    #[test]
    fn test_regime_label_default() {
        let regime = RegimeLabel::default();
        assert_eq!(regime, RegimeLabel::Normal);
        assert_eq!(regime.as_str(), "NORMAL");
    }

    #[test]
    fn test_regime_label_serialization() {
        let regime = RegimeLabel::HighVol;
        let json = serde_json::to_string(&regime).unwrap();
        assert_eq!(json, "\"HIGH_VOL\"");

        let parsed: RegimeLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, regime);
    }

    #[test]
    fn test_regime_inputs_canonical_bytes_determinism() {
        let inputs1 = RegimeInputs {
            ts_ns: 1_234_567_890,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 10,
            spread_tier: 1,
            volatility_bps: 100,
            volatility_tier: 2,
            depth_mantissa: 50_000_000,
            depth_exponent: -2,
            liquidity_tier: 1,
            funding_rate_bps: 25,
            funding_tier: 1,
            trend_strength: 1500,
        };

        let inputs2 = inputs1.clone();

        let bytes1 = inputs1.canonical_bytes();
        let bytes2 = inputs2.canonical_bytes();

        assert_eq!(
            bytes1, bytes2,
            "Identical inputs must produce identical canonical bytes"
        );
    }

    #[test]
    fn test_regime_inputs_serialization() {
        let inputs = RegimeInputs {
            ts_ns: 1_234_567_890,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 10,
            spread_tier: 1,
            volatility_bps: 100,
            volatility_tier: 2,
            depth_mantissa: 50_000_000,
            depth_exponent: -2,
            liquidity_tier: 1,
            funding_rate_bps: 25,
            funding_tier: 1,
            trend_strength: 1500,
        };

        let json = serde_json::to_string(&inputs).unwrap();
        let parsed: RegimeInputs = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.spread_bps, inputs.spread_bps);
        assert_eq!(parsed.volatility_tier, inputs.volatility_tier);
        assert_eq!(parsed.funding_rate_bps, inputs.funding_rate_bps);
    }

    #[test]
    fn test_router_config_hash_determinism() {
        let config1 = RouterConfig {
            version: "router_v1.0".to_string(),
            strategy_profiles: vec![StrategyProfile {
                strategy_id: "funding_bias:2.0:abc".to_string(),
                name: "Funding Bias".to_string(),
                suitable_regimes: vec![RegimeLabel::Normal, RegimeLabel::FundingSkew],
                priority: 100,
                enabled: true,
            }],
            thresholds: RegimeThresholds::default(),
            default_strategy_id: "funding_bias:2.0:abc".to_string(),
        };

        let config2 = config1.clone();

        let hash1 = config1.compute_hash();
        let hash2 = config2.compute_hash();

        assert_eq!(
            hash1, hash2,
            "Identical configs must produce identical hashes"
        );
        assert!(!hash1.is_empty());
        assert_eq!(hash1.len(), 64); // SHA-256 hex is 64 chars
    }

    #[test]
    fn test_router_config_hash_changes_with_content() {
        let config1 = RouterConfig {
            version: "router_v1.0".to_string(),
            strategy_profiles: Vec::new(),
            thresholds: RegimeThresholds::default(),
            default_strategy_id: "strategy_a".to_string(),
        };

        let mut config2 = config1.clone();
        config2.default_strategy_id = "strategy_b".to_string();

        let hash1 = config1.compute_hash();
        let hash2 = config2.compute_hash();

        assert_ne!(
            hash1, hash2,
            "Different configs must produce different hashes"
        );
    }

    #[test]
    fn test_router_decision_event_canonical_bytes() {
        let decision = RouterDecisionEvent {
            ts_ns: 1_234_567_890,
            decision_id: uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            symbols: vec!["BTCUSDT".to_string()],
            inputs: RegimeInputs::default(),
            regime: RegimeLabel::Normal,
            confidence_bps: 9500,
            selected_strategy_id: "funding_bias:2.0:abc".to_string(),
            alternatives: vec!["momentum:1.0:xyz".to_string()],
            selection_reason: "Default for NORMAL regime".to_string(),
            router_config_hash: "abcdef1234567890".to_string(),
            router_version: "router_v1.0".to_string(),
        };

        let bytes1 = decision.canonical_bytes();
        let bytes2 = decision.canonical_bytes();

        assert_eq!(bytes1, bytes2);
        assert!(!bytes1.is_empty());
    }

    #[test]
    fn test_router_decision_event_hash() {
        let decision = RouterDecisionEvent {
            ts_ns: 1_234_567_890,
            decision_id: uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            symbols: vec!["BTCUSDT".to_string()],
            inputs: RegimeInputs::default(),
            regime: RegimeLabel::Normal,
            confidence_bps: 9500,
            selected_strategy_id: "funding_bias:2.0:abc".to_string(),
            alternatives: Vec::new(),
            selection_reason: "Default".to_string(),
            router_config_hash: "abcdef1234567890".to_string(),
            router_version: "router_v1.0".to_string(),
        };

        let hash1 = decision.compute_hash();
        let hash2 = decision.compute_hash();

        assert_eq!(hash1, hash2, "Same decision must produce same hash");
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn test_regime_thresholds_defaults() {
        let thresholds = RegimeThresholds::default();

        // Verify documented defaults
        assert_eq!(thresholds.spread_wide_bps, 15);
        assert_eq!(thresholds.spread_extreme_bps, 50);
        assert_eq!(thresholds.vol_high_bps, 150);
        assert_eq!(thresholds.vol_extreme_bps, 300);
        assert_eq!(thresholds.depth_thin_mantissa, 1_000_000);
        assert_eq!(thresholds.depth_empty_mantissa, 100_000);
        assert_eq!(thresholds.funding_elevated_bps, 20);
        assert_eq!(thresholds.funding_extreme_bps, 50);
        assert_eq!(thresholds.trend_threshold, 3000);
    }

    #[test]
    fn test_router_config_version() {
        assert_eq!(RouterConfig::VERSION, "router_v1.0");

        let config = RouterConfig::new();
        assert_eq!(config.version, "router_v1.0");
    }

    #[test]
    fn test_router_classify_regime_normal() {
        let config = RouterConfigBuilder::new()
            .default_strategy("default:1.0:abc")
            .build();
        let router = Router::new(config);

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 5,
            spread_tier: 1,
            volatility_bps: 50,
            volatility_tier: 1,
            depth_mantissa: 100_000_000,
            depth_exponent: -2,
            liquidity_tier: 0,
            funding_rate_bps: 5,
            funding_tier: 0,
            trend_strength: 500,
        };

        let (regime, _confidence) = router.classify_regime(&inputs);
        assert_eq!(regime, RegimeLabel::Normal);
    }

    #[test]
    fn test_router_classify_regime_halt() {
        let config = RouterConfig::new();
        let router = Router::new(config);

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            liquidity_tier: 3, // Empty order books
            ..Default::default()
        };

        let (regime, confidence) = router.classify_regime(&inputs);
        assert_eq!(regime, RegimeLabel::Halt);
        assert_eq!(confidence, 10000); // 100% confidence
    }

    #[test]
    fn test_router_classify_regime_high_vol() {
        let config = RouterConfig::new();
        let router = Router::new(config);

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            volatility_bps: 200,
            volatility_tier: 2,
            ..Default::default()
        };

        let (regime, _confidence) = router.classify_regime(&inputs);
        assert_eq!(regime, RegimeLabel::HighVol);
    }

    #[test]
    fn test_router_classify_regime_funding_skew() {
        let config = RouterConfig::new();
        let router = Router::new(config);

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            funding_rate_bps: 30, // Above 20bps threshold
            funding_tier: 2,
            ..Default::default()
        };

        let (regime, _confidence) = router.classify_regime(&inputs);
        assert_eq!(regime, RegimeLabel::FundingSkew);
    }

    #[test]
    fn test_router_classify_regime_trending() {
        let config = RouterConfig::new();
        let router = Router::new(config);

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            trend_strength: 5000, // Strong uptrend
            ..Default::default()
        };

        let (regime, _confidence) = router.classify_regime(&inputs);
        assert_eq!(regime, RegimeLabel::Trending);
    }

    #[test]
    fn test_router_select_strategy_with_profiles() {
        let config = RouterConfigBuilder::new()
            .default_strategy("default:1.0:abc")
            .add_profile(StrategyProfile {
                strategy_id: "funding_bias:2.0:def".to_string(),
                name: "Funding Bias".to_string(),
                suitable_regimes: vec![RegimeLabel::Normal, RegimeLabel::FundingSkew],
                priority: 100,
                enabled: true,
            })
            .add_profile(StrategyProfile {
                strategy_id: "momentum:1.0:ghi".to_string(),
                name: "Momentum".to_string(),
                suitable_regimes: vec![RegimeLabel::Trending],
                priority: 90,
                enabled: true,
            })
            .build();

        let router = Router::new(config);

        // Normal regime should select funding_bias
        let (strategy_id, _, _) = router.select_strategy(RegimeLabel::Normal);
        assert_eq!(strategy_id, "funding_bias:2.0:def");

        // Trending regime should select momentum
        let (strategy_id, _, _) = router.select_strategy(RegimeLabel::Trending);
        assert_eq!(strategy_id, "momentum:1.0:ghi");

        // HighVol has no configured strategy, should use default
        let (strategy_id, _, _) = router.select_strategy(RegimeLabel::HighVol);
        assert_eq!(strategy_id, "default:1.0:abc");
    }

    #[test]
    fn test_router_route_determinism() {
        let config = RouterConfigBuilder::new()
            .default_strategy("default:1.0:abc")
            .add_profile(StrategyProfile {
                strategy_id: "funding_bias:2.0:def".to_string(),
                name: "Funding Bias".to_string(),
                suitable_regimes: vec![RegimeLabel::Normal],
                priority: 100,
                enabled: true,
            })
            .build();

        let router = Router::new(config);
        let decision_id = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 5,
            spread_tier: 1,
            volatility_bps: 50,
            volatility_tier: 1,
            ..Default::default()
        };

        // Route twice with same inputs and ID
        let decision1 =
            router.route_with_id(inputs.clone(), vec!["BTCUSDT".to_string()], decision_id);
        let decision2 = router.route_with_id(inputs, vec!["BTCUSDT".to_string()], decision_id);

        // Must be identical
        assert_eq!(decision1.regime, decision2.regime);
        assert_eq!(
            decision1.selected_strategy_id,
            decision2.selected_strategy_id
        );
        assert_eq!(decision1.confidence_bps, decision2.confidence_bps);
        assert_eq!(decision1.router_config_hash, decision2.router_config_hash);

        // Hash must be identical
        let hash1 = decision1.compute_hash();
        let hash2 = decision2.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_router_priority_order() {
        // Test that regime priority is correct: Halt > LowLiquidity > HighVol > FundingSkew > Trending
        let config = RouterConfig::new();
        let router = Router::new(config);

        // Multiple conditions: liquidity_tier=3 (HALT) AND volatility_tier=3 (HIGH_VOL)
        // HALT should win due to priority
        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            liquidity_tier: 3,
            volatility_tier: 3,
            funding_tier: 3,
            trend_strength: 5000,
            ..Default::default()
        };

        let (regime, _) = router.classify_regime(&inputs);
        assert_eq!(
            regime,
            RegimeLabel::Halt,
            "HALT should have highest priority"
        );
    }

    #[test]
    fn test_router_builder() {
        let config = RouterConfigBuilder::new()
            .default_strategy("default:1.0:abc")
            .add_profile(StrategyProfile {
                strategy_id: "strategy_a:1.0:123".to_string(),
                name: "Strategy A".to_string(),
                suitable_regimes: vec![RegimeLabel::Normal],
                priority: 100,
                enabled: true,
            })
            .add_profile(StrategyProfile {
                strategy_id: "strategy_b:1.0:456".to_string(),
                name: "Strategy B".to_string(),
                suitable_regimes: vec![RegimeLabel::HighVol],
                priority: 80,
                enabled: true,
            })
            .build();

        assert_eq!(config.default_strategy_id, "default:1.0:abc");
        assert_eq!(config.strategy_profiles.len(), 2);
        assert_eq!(
            config.strategy_profiles[0].strategy_id,
            "strategy_a:1.0:123"
        );
        assert_eq!(
            config.strategy_profiles[1].strategy_id,
            "strategy_b:1.0:456"
        );
    }
}
