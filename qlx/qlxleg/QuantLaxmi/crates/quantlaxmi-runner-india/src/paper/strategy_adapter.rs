//! Strategy Adapter for India F&O Paper Trading
//!
//! Adapts the existing `CalendarCarryStrategy` to work with:
//! - `OptionsSnapshot` as market data input
//! - `TradeIntent` as output (venue-agnostic, explicit)
//!
//! ## Key Design Principles
//!
//! 1. **Explicit Accept/Refuse**: No "empty intents means refuse"
//! 2. **Structured Rationale**: Every decision logs edge, friction, spread, staleness
//! 3. **Executable Pricing Only**: Edge computed from bid/ask, never mid
//! 4. **Cross-Market Symmetry**: TradeIntent works for India and future Crypto

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use quantlaxmi_paper::Strategy;

use super::fees_india::{InstrumentKind, Side};
use super::mapping::InstrumentMap;
use super::margin::{MarginGate, MarginOrderParams, MarginRejectReason};
use super::snapshot::{OptionsSnapshot, Right};

// =============================================================================
// TIME IN FORCE
// =============================================================================

/// Time-in-force for paper trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Immediate or cancel (paper v1 default)
    #[default]
    IOC,
    /// Good till cancelled
    GTC,
}

// =============================================================================
// INTENT TAG
// =============================================================================

/// Tag identifying the strategy/reason for the intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentTag {
    /// Calendar carry entry - short front, long back
    CalendarCarryEntry,
    /// Calendar carry exit
    CalendarCarryExit,
    /// Hedge adjustment
    HedgeAdjust,
    /// Manual/test intent
    Manual,
}

// =============================================================================
// TRADE INTENT (Venue-Agnostic)
// =============================================================================

/// Trade intent for paper trading.
///
/// Venue-agnostic design:
/// - India: instrument_token maps to NSE token
/// - Crypto: instrument_token could map to symbol hash
///
/// Note: Fee logic and pricing are fill model's job, not intent's.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeIntent {
    /// Instrument identifier (NSE token for India)
    pub instrument_token: u32,
    /// Trade side
    pub side: Side,
    /// Quantity in absolute contracts (always positive)
    pub qty: i32,
    /// Time in force
    pub tif: TimeInForce,
    /// Intent tag for audit/logging
    pub tag: IntentTag,
    // -------------------------------------------------------------------------
    // India-specific fields (needed for fee calculation, could be in metadata)
    // -------------------------------------------------------------------------
    /// Instrument kind (for fee calculation)
    pub kind: InstrumentKind,
    /// Lot size (for fee calculation)
    pub lot_size: i32,
}

impl TradeIntent {
    /// Create a new trade intent.
    pub fn new(
        instrument_token: u32,
        side: Side,
        qty: i32,
        tag: IntentTag,
        kind: InstrumentKind,
        lot_size: i32,
    ) -> Self {
        Self {
            instrument_token,
            side,
            qty: qty.abs(), // Always positive
            tif: TimeInForce::IOC,
            tag,
            kind,
            lot_size,
        }
    }
}

// Implement InstrumentIdentity for engine compatibility
impl quantlaxmi_paper::InstrumentIdentity for TradeIntent {
    type Key = u32;

    fn instrument_key(&self) -> u32 {
        self.instrument_token
    }
}

// =============================================================================
// RATIONALE (Structured Logging)
// =============================================================================

/// Gate status for audit logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateStatus {
    pub name: String,
    pub passed: bool,
    pub value: Option<i64>,
    pub threshold: Option<i64>,
    pub reason: Option<String>,
}

/// Structured rationale for strategy decisions.
///
/// Every field logged in paise (i64) for deterministic replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rationale {
    /// Timestamp of decision
    pub ts: DateTime<Utc>,
    /// Edge estimate in paise (expected profit from position)
    pub edge_estimate_paise: i64,
    /// Friction estimate in paise (expected round-trip costs)
    pub friction_estimate_paise: i64,
    /// Edge minus friction in paise
    pub edge_minus_friction_paise: i64,
    /// Spread cost in paise (half-spread on entry)
    pub spread_cost_paise: i64,
    /// Ratio of stale quotes (0.0 - 1.0, scaled to basis points)
    pub stale_quotes_ratio_bps: u32,
    /// Gate statuses
    pub gate_statuses: Vec<GateStatus>,
    /// Front leg spread in paise
    pub front_spread_paise: i64,
    /// Back leg spread in paise
    pub back_spread_paise: i64,
    /// Front leg mid price in paise
    pub front_mid_paise: i64,
    /// Back leg mid price in paise
    pub back_mid_paise: i64,
    /// Hedge ratio used
    pub hedge_ratio: f64,
    /// Whether hedge ratio was clamped
    pub hedge_ratio_clamped: bool,
    /// Underlying symbol
    pub underlying: String,
    /// Front expiry
    pub front_expiry: String,
    /// Back expiry
    pub back_expiry: String,
}

impl Default for Rationale {
    fn default() -> Self {
        Self {
            ts: Utc::now(),
            edge_estimate_paise: 0,
            friction_estimate_paise: 0,
            edge_minus_friction_paise: 0,
            spread_cost_paise: 0,
            stale_quotes_ratio_bps: 0,
            gate_statuses: Vec::new(),
            front_spread_paise: 0,
            back_spread_paise: 0,
            front_mid_paise: 0,
            back_mid_paise: 0,
            hedge_ratio: 1.0,
            hedge_ratio_clamped: false,
            underlying: String::new(),
            front_expiry: String::new(),
            back_expiry: String::new(),
        }
    }
}

// =============================================================================
// REFUSE REASON
// =============================================================================

/// Reason for refusing to trade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefuseReason {
    /// No executable quote available (bid/ask missing)
    NoExecutableQuote { leg: String, detail: String },
    /// Quote is stale
    StaleQuote {
        leg: String,
        age_ms: u32,
        threshold_ms: u32,
    },
    /// Spread too wide
    SpreadTooWide {
        leg: String,
        spread_bps: u32,
        max_bps: u32,
    },
    /// Insufficient edge
    InsufficientEdge { edge_paise: i64, min_paise: i64 },
    /// Gate failed
    GateFailed { gate: String, reason: String },
    /// Risk limit exceeded
    RiskLimit { detail: String },
    /// Session not active
    SessionNotActive { reason: String },
    /// No position to exit
    NoPositionToExit,
    /// Insufficient data
    InsufficientData { detail: String },
    /// Insufficient margin (SPAN)
    InsufficientMargin { detail: String },
}

// =============================================================================
// STRATEGY DECISION (Explicit Accept/Refuse)
// =============================================================================

/// Strategy decision: either accept with intents or refuse with reason.
///
/// Never use "empty intents" to mean refuse - be explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyDecisionExplicit {
    /// Accept: emit trade intents
    Accept {
        intents: Vec<TradeIntent>,
        rationale: Rationale,
    },
    /// Refuse: do not trade
    Refuse {
        reason: RefuseReason,
        rationale: Rationale,
    },
    /// Hold: maintain current position (no action)
    Hold { rationale: Rationale },
}

// =============================================================================
// CALENDAR CARRY CONFIG
// =============================================================================

/// Configuration for calendar carry adapter.
#[derive(Debug, Clone)]
pub struct CalendarCarryConfig {
    /// Underlying to trade (e.g., "NIFTY")
    pub underlying: String,
    /// Lot size
    pub lot_size: i32,
    /// Staleness threshold in ms
    pub staleness_threshold_ms: u32,
    /// Maximum spread in bps
    pub max_spread_bps: u32,
    /// Minimum edge in paise to enter
    pub entry_threshold_paise: i64,
    /// Minimum edge to hold (hysteresis)
    pub exit_threshold_paise: i64,
    /// Minutes before close to exit
    pub exit_minutes_before_close: u32,
    /// Hedge ratio bounds
    pub hedge_ratio_min: f64,
    pub hedge_ratio_max: f64,
}

impl Default for CalendarCarryConfig {
    fn default() -> Self {
        Self {
            underlying: "NIFTY".to_string(),
            lot_size: 25,
            staleness_threshold_ms: 5000,
            max_spread_bps: 50,
            entry_threshold_paise: 800, // ₹8
            exit_threshold_paise: 200,  // ₹2 (hysteresis)
            exit_minutes_before_close: 15,
            hedge_ratio_min: 0.5,
            hedge_ratio_max: 2.0,
        }
    }
}

// =============================================================================
// CALENDAR CARRY ADAPTER
// =============================================================================

/// Calendar carry strategy adapter for paper trading.
///
/// Adapts the existing CalendarCarryStrategy to:
/// - Use OptionsSnapshot for market data
/// - Emit explicit Accept/Refuse decisions
/// - Compute edge from executable quotes only
/// - Log structured rationale for every decision
pub struct CalendarCarryAdapter {
    config: CalendarCarryConfig,
    /// Current position state
    position: Option<CalendarPosition>,
    /// Decision counter
    decision_count: u64,
}

/// Active calendar carry position.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for position management in Phase 2
pub struct CalendarPosition {
    /// Front leg tokens (CE, PE)
    pub front_ce_token: u32,
    pub front_pe_token: u32,
    /// Back leg tokens (CE, PE)
    pub back_ce_token: u32,
    pub back_pe_token: u32,
    /// Entry prices (paise)
    pub entry_front_paise: i64,
    pub entry_back_paise: i64,
    /// Entry friction for exit threshold
    pub entry_friction_paise: i64,
    /// Entry timestamp
    pub entry_ts: DateTime<Utc>,
    /// Lots
    pub lots: i32,
    /// Hedge ratio
    pub hedge_ratio: f64,
}

impl CalendarCarryAdapter {
    /// Create a new calendar carry adapter.
    pub fn new(config: CalendarCarryConfig) -> Self {
        Self {
            config,
            position: None,
            decision_count: 0,
        }
    }

    /// Get current position (if any).
    pub fn position(&self) -> Option<&CalendarPosition> {
        self.position.as_ref()
    }

    /// Check if we have an open position.
    pub fn has_position(&self) -> bool {
        self.position.is_some()
    }

    /// Process a snapshot and return a decision.
    pub fn evaluate(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
    ) -> StrategyDecisionExplicit {
        self.decision_count += 1;

        // Build base rationale
        let mut rationale = Rationale {
            ts,
            underlying: self.config.underlying.clone(),
            ..Default::default()
        };

        // 1. Find ATM straddles (front and back expiry)
        let atm_data = match self.find_atm_straddles(snapshot, ts, &mut rationale) {
            Ok(data) => data,
            Err(reason) => {
                return StrategyDecisionExplicit::Refuse { reason, rationale };
            }
        };

        // 2. Validate quotes (not stale, has executable prices)
        if let Err(reason) = self.validate_quotes(&atm_data, ts, &mut rationale) {
            return StrategyDecisionExplicit::Refuse { reason, rationale };
        }

        // 3. Compute edge and friction from executable quotes
        self.compute_edge_and_friction(&atm_data, &mut rationale);

        // 4. Decide based on position state
        if self.has_position() {
            self.evaluate_exit(ts, &atm_data, rationale)
        } else {
            self.evaluate_entry(ts, &atm_data, rationale)
        }
    }

    /// Find ATM straddles for front and back expiry.
    fn find_atm_straddles(
        &self,
        snapshot: &OptionsSnapshot,
        _ts: DateTime<Utc>,
        rationale: &mut Rationale,
    ) -> Result<AtmStraddleData, RefuseReason> {
        // Get unique expiries sorted by date
        let mut expiries: Vec<&str> = snapshot
            .quotes
            .iter()
            .map(|q| q.expiry.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        expiries.sort();

        if expiries.len() < 2 {
            return Err(RefuseReason::InsufficientData {
                detail: format!("Need at least 2 expiries, found {}", expiries.len()),
            });
        }

        let front_expiry = expiries[0].to_string();
        let back_expiry = expiries[1].to_string();

        rationale.front_expiry = front_expiry.clone();
        rationale.back_expiry = back_expiry.clone();

        // Find ATM strike (highest OI or closest to spot)
        // For now, use the most common strike in the front expiry
        let front_strikes: Vec<i32> = snapshot
            .quotes
            .iter()
            .filter(|q| q.expiry == front_expiry)
            .map(|q| q.strike)
            .collect();

        if front_strikes.is_empty() {
            return Err(RefuseReason::InsufficientData {
                detail: "No quotes for front expiry".to_string(),
            });
        }

        // Find the strike that appears most frequently (has both CE and PE)
        let mut strike_counts: std::collections::HashMap<i32, u32> =
            std::collections::HashMap::new();
        for s in &front_strikes {
            *strike_counts.entry(*s).or_insert(0) += 1;
        }

        let atm_strike = strike_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2) // Has both CE and PE
            .max_by_key(|(_, count)| *count)
            .map(|(strike, _)| strike)
            .ok_or_else(|| RefuseReason::InsufficientData {
                detail: "No complete straddle (CE+PE) found".to_string(),
            })?;

        // Get front straddle quotes
        let front_ce = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == front_expiry && q.strike == atm_strike && q.right == Right::Call)
            .ok_or_else(|| RefuseReason::NoExecutableQuote {
                leg: "front_ce".to_string(),
                detail: format!("No CE at strike {}", atm_strike),
            })?;

        let front_pe = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == front_expiry && q.strike == atm_strike && q.right == Right::Put)
            .ok_or_else(|| RefuseReason::NoExecutableQuote {
                leg: "front_pe".to_string(),
                detail: format!("No PE at strike {}", atm_strike),
            })?;

        // Get back straddle quotes (same or nearest strike)
        let back_ce = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == back_expiry && q.strike == atm_strike && q.right == Right::Call)
            .ok_or_else(|| RefuseReason::NoExecutableQuote {
                leg: "back_ce".to_string(),
                detail: format!("No back CE at strike {}", atm_strike),
            })?;

        let back_pe = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == back_expiry && q.strike == atm_strike && q.right == Right::Put)
            .ok_or_else(|| RefuseReason::NoExecutableQuote {
                leg: "back_pe".to_string(),
                detail: format!("No back PE at strike {}", atm_strike),
            })?;

        Ok(AtmStraddleData {
            front_ce_token: front_ce.instrument_token,
            front_pe_token: front_pe.instrument_token,
            back_ce_token: back_ce.instrument_token,
            back_pe_token: back_pe.instrument_token,
            // Front straddle: sell = bid prices
            front_ce_bid: front_ce.bid.map(|b| (b.price * 100.0) as i64),
            front_ce_ask: front_ce.ask.map(|a| (a.price * 100.0) as i64),
            front_pe_bid: front_pe.bid.map(|b| (b.price * 100.0) as i64),
            front_pe_ask: front_pe.ask.map(|a| (a.price * 100.0) as i64),
            // Back straddle: buy = ask prices
            back_ce_bid: back_ce.bid.map(|b| (b.price * 100.0) as i64),
            back_ce_ask: back_ce.ask.map(|a| (a.price * 100.0) as i64),
            back_pe_bid: back_pe.bid.map(|b| (b.price * 100.0) as i64),
            back_pe_ask: back_pe.ask.map(|a| (a.price * 100.0) as i64),
            // Staleness
            front_ce_age_ms: front_ce.age_ms,
            front_pe_age_ms: front_pe.age_ms,
            back_ce_age_ms: back_ce.age_ms,
            back_pe_age_ms: back_pe.age_ms,
            // Strike
            strike: atm_strike,
            front_expiry,
            back_expiry,
        })
    }

    /// Validate quotes (not stale, has executable prices).
    fn validate_quotes(
        &self,
        data: &AtmStraddleData,
        _ts: DateTime<Utc>,
        rationale: &mut Rationale,
    ) -> Result<(), RefuseReason> {
        let threshold = self.config.staleness_threshold_ms;

        // Check staleness
        let stale_count = [
            data.front_ce_age_ms,
            data.front_pe_age_ms,
            data.back_ce_age_ms,
            data.back_pe_age_ms,
        ]
        .iter()
        .filter(|&&age| age > threshold)
        .count();

        rationale.stale_quotes_ratio_bps = ((stale_count as f64 / 4.0) * 10000.0) as u32;

        // Any stale quote is a rejection
        if data.front_ce_age_ms > threshold {
            return Err(RefuseReason::StaleQuote {
                leg: "front_ce".to_string(),
                age_ms: data.front_ce_age_ms,
                threshold_ms: threshold,
            });
        }
        if data.front_pe_age_ms > threshold {
            return Err(RefuseReason::StaleQuote {
                leg: "front_pe".to_string(),
                age_ms: data.front_pe_age_ms,
                threshold_ms: threshold,
            });
        }
        if data.back_ce_age_ms > threshold {
            return Err(RefuseReason::StaleQuote {
                leg: "back_ce".to_string(),
                age_ms: data.back_ce_age_ms,
                threshold_ms: threshold,
            });
        }
        if data.back_pe_age_ms > threshold {
            return Err(RefuseReason::StaleQuote {
                leg: "back_pe".to_string(),
                age_ms: data.back_pe_age_ms,
                threshold_ms: threshold,
            });
        }

        // Check executable prices (bid/ask present)
        // Front leg: need bids (we're selling)
        if data.front_ce_bid.is_none() || data.front_ce_bid == Some(0) {
            return Err(RefuseReason::NoExecutableQuote {
                leg: "front_ce".to_string(),
                detail: "No bid".to_string(),
            });
        }
        if data.front_pe_bid.is_none() || data.front_pe_bid == Some(0) {
            return Err(RefuseReason::NoExecutableQuote {
                leg: "front_pe".to_string(),
                detail: "No bid".to_string(),
            });
        }

        // Back leg: need asks (we're buying)
        if data.back_ce_ask.is_none() || data.back_ce_ask == Some(0) {
            return Err(RefuseReason::NoExecutableQuote {
                leg: "back_ce".to_string(),
                detail: "No ask".to_string(),
            });
        }
        if data.back_pe_ask.is_none() || data.back_pe_ask == Some(0) {
            return Err(RefuseReason::NoExecutableQuote {
                leg: "back_pe".to_string(),
                detail: "No ask".to_string(),
            });
        }

        Ok(())
    }

    /// Compute edge and friction from executable quotes.
    ///
    /// Calendar carry:
    /// - Sell front straddle at bid
    /// - Buy back straddle at ask
    /// - Edge = premium_received - premium_paid (positive = profit)
    fn compute_edge_and_friction(&self, data: &AtmStraddleData, rationale: &mut Rationale) {
        // Front straddle: sell at bid
        let front_bid = data.front_ce_bid.unwrap_or(0) + data.front_pe_bid.unwrap_or(0);
        let front_ask = data.front_ce_ask.unwrap_or(0) + data.front_pe_ask.unwrap_or(0);
        let front_mid = (front_bid + front_ask) / 2;
        let front_spread = front_ask - front_bid;

        // Back straddle: buy at ask
        let back_bid = data.back_ce_bid.unwrap_or(0) + data.back_pe_bid.unwrap_or(0);
        let back_ask = data.back_ce_ask.unwrap_or(0) + data.back_pe_ask.unwrap_or(0);
        let back_mid = (back_bid + back_ask) / 2;
        let back_spread = back_ask - back_bid;

        // Hedge ratio (simplified: 1.0 for now)
        // In full implementation, compute from vega ratio
        let h = 1.0_f64.clamp(self.config.hedge_ratio_min, self.config.hedge_ratio_max);
        let h_clamped = h == self.config.hedge_ratio_min || h == self.config.hedge_ratio_max;

        // Executable edge (what we actually get):
        // Sell front at bid, buy back at ask
        // edge = front_bid - h * back_ask
        let edge_paise = front_bid - (h * back_ask as f64) as i64;

        // Friction = half-spread on entry + half-spread on exit = spread on round-trip
        // Entry: half of front spread (selling) + half of back spread (buying)
        // Exit: same
        let friction_entry = (front_spread / 2) + ((h * back_spread as f64) as i64 / 2);
        let friction_round = 2 * friction_entry;

        rationale.edge_estimate_paise = edge_paise;
        rationale.friction_estimate_paise = friction_round;
        rationale.edge_minus_friction_paise = edge_paise - friction_round;
        rationale.spread_cost_paise = friction_entry;
        rationale.front_spread_paise = front_spread;
        rationale.back_spread_paise = back_spread;
        rationale.front_mid_paise = front_mid;
        rationale.back_mid_paise = back_mid;
        rationale.hedge_ratio = h;
        rationale.hedge_ratio_clamped = h_clamped;

        // Log gate statuses
        rationale.gate_statuses.push(GateStatus {
            name: "EDGE".to_string(),
            passed: edge_paise >= self.config.entry_threshold_paise,
            value: Some(edge_paise),
            threshold: Some(self.config.entry_threshold_paise),
            reason: None,
        });

        rationale.gate_statuses.push(GateStatus {
            name: "FRICTION".to_string(),
            passed: edge_paise > friction_round,
            value: Some(friction_round),
            threshold: Some(edge_paise),
            reason: None,
        });
    }

    /// Evaluate entry decision.
    fn evaluate_entry(
        &mut self,
        ts: DateTime<Utc>,
        data: &AtmStraddleData,
        rationale: Rationale,
    ) -> StrategyDecisionExplicit {
        // Check if edge meets threshold
        if rationale.edge_minus_friction_paise < self.config.entry_threshold_paise {
            return StrategyDecisionExplicit::Refuse {
                reason: RefuseReason::InsufficientEdge {
                    edge_paise: rationale.edge_minus_friction_paise,
                    min_paise: self.config.entry_threshold_paise,
                },
                rationale,
            };
        }

        // Build trade intents
        // INVARIANT: qty is in CONTRACTS (not lots)
        // Convention A: qty = lots * lot_size
        let lot_size = self.config.lot_size;
        let lots = 1; // Start with 1 lot
        let contracts = lots * lot_size; // Convert to contracts

        // Front leg: SELL straddle (sell CE + sell PE)
        let front_ce_intent = TradeIntent::new(
            data.front_ce_token,
            Side::Sell,
            contracts,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            lot_size,
        );

        let front_pe_intent = TradeIntent::new(
            data.front_pe_token,
            Side::Sell,
            contracts,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            lot_size,
        );

        // Back leg: BUY straddle (buy CE + buy PE)
        let back_ce_intent = TradeIntent::new(
            data.back_ce_token,
            Side::Buy,
            contracts,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            lot_size,
        );

        let back_pe_intent = TradeIntent::new(
            data.back_pe_token,
            Side::Buy,
            contracts,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            lot_size,
        );

        // Record position (track lots for exit)
        self.position = Some(CalendarPosition {
            front_ce_token: data.front_ce_token,
            front_pe_token: data.front_pe_token,
            back_ce_token: data.back_ce_token,
            back_pe_token: data.back_pe_token,
            entry_front_paise: data.front_ce_bid.unwrap_or(0) + data.front_pe_bid.unwrap_or(0),
            entry_back_paise: data.back_ce_ask.unwrap_or(0) + data.back_pe_ask.unwrap_or(0),
            entry_friction_paise: rationale.friction_estimate_paise,
            entry_ts: ts,
            lots,
            hedge_ratio: rationale.hedge_ratio,
        });

        info!(
            edge_paise = rationale.edge_estimate_paise,
            friction_paise = rationale.friction_estimate_paise,
            edge_minus_friction = rationale.edge_minus_friction_paise,
            "[STRATEGY] Calendar carry ENTER"
        );

        StrategyDecisionExplicit::Accept {
            intents: vec![
                front_ce_intent,
                front_pe_intent,
                back_ce_intent,
                back_pe_intent,
            ],
            rationale,
        }
    }

    /// Evaluate exit decision.
    fn evaluate_exit(
        &mut self,
        _ts: DateTime<Utc>,
        data: &AtmStraddleData,
        rationale: Rationale,
    ) -> StrategyDecisionExplicit {
        if self.position.is_none() {
            return StrategyDecisionExplicit::Refuse {
                reason: RefuseReason::NoPositionToExit,
                rationale,
            };
        }

        // Exit if edge dropped below threshold (hysteresis)
        if rationale.edge_minus_friction_paise < self.config.exit_threshold_paise {
            return self.execute_exit(data, rationale, "EDGE_BELOW_THRESHOLD");
        }

        // Hold position
        debug!(
            edge_paise = rationale.edge_estimate_paise,
            "[STRATEGY] Calendar carry HOLD"
        );

        StrategyDecisionExplicit::Hold { rationale }
    }

    /// Execute exit.
    fn execute_exit(
        &mut self,
        data: &AtmStraddleData,
        rationale: Rationale,
        reason: &str,
    ) -> StrategyDecisionExplicit {
        let position = self.position.take().unwrap();
        let lot_size = self.config.lot_size;
        // INVARIANT: qty is in CONTRACTS (not lots)
        let contracts = position.lots * lot_size;

        // Exit: reverse the position
        // Buy back front (was short) -> BUY at ask
        let front_ce_intent = TradeIntent::new(
            data.front_ce_token,
            Side::Buy,
            contracts,
            IntentTag::CalendarCarryExit,
            InstrumentKind::IndexOption,
            lot_size,
        );

        let front_pe_intent = TradeIntent::new(
            data.front_pe_token,
            Side::Buy,
            contracts,
            IntentTag::CalendarCarryExit,
            InstrumentKind::IndexOption,
            lot_size,
        );

        // Sell back (was long) -> SELL at bid
        let back_ce_intent = TradeIntent::new(
            data.back_ce_token,
            Side::Sell,
            contracts,
            IntentTag::CalendarCarryExit,
            InstrumentKind::IndexOption,
            lot_size,
        );

        let back_pe_intent = TradeIntent::new(
            data.back_pe_token,
            Side::Sell,
            contracts,
            IntentTag::CalendarCarryExit,
            InstrumentKind::IndexOption,
            lot_size,
        );

        info!(
            reason = reason,
            edge_paise = rationale.edge_estimate_paise,
            "[STRATEGY] Calendar carry EXIT"
        );

        StrategyDecisionExplicit::Accept {
            intents: vec![
                front_ce_intent,
                front_pe_intent,
                back_ce_intent,
                back_pe_intent,
            ],
            rationale,
        }
    }
}

/// ATM straddle data extracted from snapshot.
#[allow(dead_code)] // Some fields reserved for Phase 2 position management
struct AtmStraddleData {
    front_ce_token: u32,
    front_pe_token: u32,
    back_ce_token: u32,
    back_pe_token: u32,
    front_ce_bid: Option<i64>,
    front_ce_ask: Option<i64>,
    front_pe_bid: Option<i64>,
    front_pe_ask: Option<i64>,
    back_ce_bid: Option<i64>,
    back_ce_ask: Option<i64>,
    back_pe_bid: Option<i64>,
    back_pe_ask: Option<i64>,
    front_ce_age_ms: u32,
    front_pe_age_ms: u32,
    back_ce_age_ms: u32,
    back_pe_age_ms: u32,
    strike: i32,
    front_expiry: String,
    back_expiry: String,
}

// =============================================================================
// STRATEGY TRAIT ADAPTER
// =============================================================================

/// Wrapper to adapt CalendarCarryAdapter to quantlaxmi-paper Strategy trait.
pub struct CalendarCarryStrategyWrapper {
    adapter: CalendarCarryAdapter,
    /// Optional margin gate for SPAN margin checking (P0 requirement)
    margin_gate: Option<Arc<Mutex<MarginGate>>>,
    /// Instrument map for canonical symbol lookup (avoids snapshot symbol collisions)
    instrument_map: Option<InstrumentMap>,
    /// Last margin reserved for current position (for release on exit)
    last_margin_reserved: f64,
    /// Pending margin side-effect to apply only after observing actual fills.
    pending_margin_action: Option<PendingMarginAction>,
}

/// Pending margin action to apply only after engine confirms AllFilled.
#[derive(Debug, Clone, Copy)]
enum PendingMarginAction {
    Reserve(f64),
    Release,
}

impl CalendarCarryStrategyWrapper {
    pub fn new(config: CalendarCarryConfig) -> Self {
        Self {
            adapter: CalendarCarryAdapter::new(config),
            margin_gate: None,
            instrument_map: None,
            last_margin_reserved: 0.0,
            pending_margin_action: None,
        }
    }

    /// Set the margin gate for SPAN margin checking.
    ///
    /// When set, all entry trades will be validated against actual SPAN margins
    /// from Zerodha API before being accepted.
    pub fn with_margin_gate(mut self, gate: Arc<Mutex<MarginGate>>) -> Self {
        self.margin_gate = Some(gate);
        self
    }

    /// Set the instrument map for canonical symbol lookup.
    pub fn with_instrument_map(mut self, map: InstrumentMap) -> Self {
        self.instrument_map = Some(map);
        self
    }

    /// Check margin for a set of intents using Zerodha SPAN API.
    /// Returns the margin requirement if accepted (for reserve_margin call).
    async fn check_margin(&self, intents: &[TradeIntent]) -> Result<f64, MarginRejectReason> {
        let gate = match &self.margin_gate {
            Some(g) => g,
            None => return Ok(0.0), // No margin gate = no margin check, no margin to reserve
        };

        // Build margin order params from intents
        let mut orders: Vec<MarginOrderParams> = Vec::new();

        for intent in intents {
            // Use InstrumentMap for canonical symbol (safer than snapshot)
            let symbol = self
                .instrument_map
                .as_ref()
                .and_then(|m| m.get(intent.instrument_token))
                .map(|meta| meta.tradingsymbol.clone())
                .unwrap_or_else(|| format!("TOKEN_{}", intent.instrument_token));

            let transaction_type = match intent.side {
                Side::Buy => "BUY",
                Side::Sell => "SELL",
            };

            orders.push(MarginOrderParams {
                exchange: "NFO".to_string(),
                tradingsymbol: symbol,
                transaction_type: transaction_type.to_string(),
                variety: "regular".to_string(),
                product: "NRML".to_string(),
                order_type: "MARKET".to_string(),
                quantity: intent.qty,
                price: None,
            });
        }

        // Check margin via API (gate uses its internal available_cash)
        let mut gate_lock = gate.lock().await;
        let result = gate_lock.check_basket_entry(orders).await;

        match result {
            Ok(req) => {
                info!(
                    total = req.total,
                    per_order_span = req.per_order_span,
                    per_order_exposure = req.per_order_exposure,
                    from_cache = req.from_cache,
                    "[MARGIN] SPAN check passed (using final_margin.total)"
                );
                Ok(req.total)
            }
            Err(e) => {
                warn!(reason = %e, "[MARGIN] SPAN check failed");
                Err(e)
            }
        }
    }

    /// Reserve margin (engine-confirmed AllFilled).
    async fn reserve_margin_confirmed(&mut self, margin: f64) {
        if let Some(gate) = &self.margin_gate {
            let mut gate_lock = gate.lock().await;
            gate_lock.reserve_margin(margin);
            self.last_margin_reserved = margin;
        }
    }

    /// Release margin (engine-confirmed AllFilled).
    async fn release_margin_confirmed(&mut self) {
        if let Some(gate) = &self.margin_gate
            && self.last_margin_reserved > 0.0
        {
            let mut gate_lock = gate.lock().await;
            gate_lock.release_margin(self.last_margin_reserved);
            self.last_margin_reserved = 0.0;
        }
    }
}

/// Convert Rationale to venue-agnostic DecisionMetrics.
fn rationale_to_metrics(rationale: &Rationale) -> quantlaxmi_paper::DecisionMetrics {
    quantlaxmi_paper::DecisionMetrics {
        edge_estimate: rationale.edge_estimate_paise,
        friction_estimate: rationale.friction_estimate_paise,
        spread_cost: rationale.spread_cost_paise,
        stale_quotes_ratio_bps: rationale.stale_quotes_ratio_bps,
        strategy_name: "CalendarCarry".to_string(),
    }
}

/// Build StrategyView for TUI display.
fn build_strategy_view(
    rationale: &Rationale,
    snapshot: &OptionsSnapshot,
    config: &CalendarCarryConfig,
    decision_type: &str,
    decision_reason: &str,
    positions: Vec<quantlaxmi_paper::PositionView>,
) -> quantlaxmi_paper::StrategyView {
    // Find front and back leg quotes for display
    let front_leg = snapshot
        .quotes
        .iter()
        .find(|q| q.expiry == rationale.front_expiry && q.right == Right::Call)
        .map(|q| quantlaxmi_paper::OptionLegView {
            symbol: q.tradingsymbol.clone(),
            strike: q.strike,
            right: q.right.to_zerodha().to_string(),
            expiry: q.expiry.clone(),
            bid: q.bid.map(|b| b.price),
            ask: q.ask.map(|a| a.price),
            mid: q.mid(),
            age_ms: q.age_ms,
        });

    let back_leg = snapshot
        .quotes
        .iter()
        .find(|q| q.expiry == rationale.back_expiry && q.right == Right::Call)
        .map(|q| quantlaxmi_paper::OptionLegView {
            symbol: q.tradingsymbol.clone(),
            strike: q.strike,
            right: q.right.to_zerodha().to_string(),
            expiry: q.expiry.clone(),
            bid: q.bid.map(|b| b.price),
            ask: q.ask.map(|a| a.price),
            mid: q.mid(),
            age_ms: q.age_ms,
        });

    quantlaxmi_paper::StrategyView {
        name: "Calendar Carry".to_string(),
        underlying: rationale.underlying.clone(),
        spot: snapshot.spot,
        futures: None, // Not tracking futures price currently
        edge_rupees: rationale.edge_estimate_paise as f64 / 100.0,
        friction_rupees: rationale.friction_estimate_paise as f64 / 100.0,
        net_edge_rupees: rationale.edge_minus_friction_paise as f64 / 100.0,
        entry_threshold_rupees: config.entry_threshold_paise as f64 / 100.0,
        exit_threshold_rupees: config.exit_threshold_paise as f64 / 100.0,
        front_leg,
        back_leg,
        positions,
        decision_type: decision_type.to_string(),
        decision_reason: decision_reason.to_string(),
    }
}

#[async_trait]
impl Strategy<OptionsSnapshot, TradeIntent> for CalendarCarryStrategyWrapper {
    async fn on_snapshot(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
    ) -> Result<quantlaxmi_paper::StrategyDecision<TradeIntent>> {
        let decision = self.adapter.evaluate(ts, snapshot);

        // Convert to quantlaxmi-paper StrategyDecision with metrics
        match decision {
            StrategyDecisionExplicit::Accept { intents, rationale } => {
                let metrics = rationale_to_metrics(&rationale);

                // Determine if this is an entry or exit based on intent tags
                let is_entry = intents
                    .first()
                    .map(|i| matches!(i.tag, IntentTag::CalendarCarryEntry))
                    .unwrap_or(false);

                if is_entry {
                    // P0 MARGIN GATE: Check SPAN margin before accepting entry
                    match self.check_margin(&intents).await {
                        Ok(margin_required) => {
                            // IMPORTANT: Do NOT reserve margin here.
                            // Only reserve after the engine confirms `AllFilled`.
                            self.pending_margin_action =
                                Some(PendingMarginAction::Reserve(margin_required));

                            let reason = format!(
                                "ENTER edge={} friction={} net={} margin=₹{:.0}",
                                rationale.edge_estimate_paise,
                                rationale.friction_estimate_paise,
                                rationale.edge_minus_friction_paise,
                                margin_required
                            );

                            let strategy_view = build_strategy_view(
                                &rationale,
                                snapshot,
                                &self.adapter.config,
                                "Accept",
                                &reason,
                                vec![],
                            );

                            Ok(quantlaxmi_paper::StrategyDecision {
                                ts,
                                accepted: true,
                                reason,
                                intents,
                                decision_type: quantlaxmi_paper::DecisionType::Accept,
                                metrics: Some(metrics),
                                strategy_view: Some(strategy_view),
                            })
                        }
                        Err(margin_err) => {
                            // Margin check failed - refuse the trade
                            let reason = format!("MARGIN_REJECTED: {}", margin_err);
                            let strategy_view = build_strategy_view(
                                &rationale,
                                snapshot,
                                &self.adapter.config,
                                "Refuse",
                                &reason,
                                vec![],
                            );

                            Ok(quantlaxmi_paper::StrategyDecision {
                                ts,
                                accepted: false,
                                reason,
                                intents: vec![],
                                decision_type: quantlaxmi_paper::DecisionType::Refuse,
                                metrics: Some(metrics),
                                strategy_view: Some(strategy_view),
                            })
                        }
                    }
                } else {
                    // IMPORTANT: Do NOT release margin here.
                    // Only release after the engine confirms `AllFilled` for the exit.
                    self.pending_margin_action = Some(PendingMarginAction::Release);

                    let reason = format!(
                        "EXIT edge={} friction={} net={}",
                        rationale.edge_estimate_paise,
                        rationale.friction_estimate_paise,
                        rationale.edge_minus_friction_paise
                    );

                    let strategy_view = build_strategy_view(
                        &rationale,
                        snapshot,
                        &self.adapter.config,
                        "Accept",
                        &reason,
                        vec![], // TODO: Include current positions on exit
                    );

                    Ok(quantlaxmi_paper::StrategyDecision {
                        ts,
                        accepted: true,
                        reason,
                        intents,
                        decision_type: quantlaxmi_paper::DecisionType::Accept,
                        metrics: Some(metrics),
                        strategy_view: Some(strategy_view),
                    })
                }
            }
            StrategyDecisionExplicit::Refuse { reason, rationale } => {
                let metrics = rationale_to_metrics(&rationale);
                let reason_str = format!("{:?} edge={}", reason, rationale.edge_estimate_paise);

                let strategy_view = build_strategy_view(
                    &rationale,
                    snapshot,
                    &self.adapter.config,
                    "Refuse",
                    &reason_str,
                    vec![],
                );

                Ok(quantlaxmi_paper::StrategyDecision {
                    ts,
                    accepted: false,
                    reason: reason_str,
                    intents: vec![],
                    decision_type: quantlaxmi_paper::DecisionType::Refuse,
                    metrics: Some(metrics),
                    strategy_view: Some(strategy_view),
                })
            }
            StrategyDecisionExplicit::Hold { rationale } => {
                let metrics = rationale_to_metrics(&rationale);
                let reason_str = format!("HOLD edge={}", rationale.edge_estimate_paise);

                let strategy_view = build_strategy_view(
                    &rationale,
                    snapshot,
                    &self.adapter.config,
                    "Hold",
                    &reason_str,
                    vec![], // TODO: Include current positions
                );

                Ok(quantlaxmi_paper::StrategyDecision {
                    ts,
                    accepted: true,
                    reason: reason_str,
                    intents: vec![],
                    decision_type: quantlaxmi_paper::DecisionType::Hold,
                    metrics: Some(metrics),
                    strategy_view: Some(strategy_view),
                })
            }
        }
    }

    async fn on_outcome(
        &mut self,
        _decision: &quantlaxmi_paper::StrategyDecision<TradeIntent>,
        outcome: quantlaxmi_paper::FillOutcome,
    ) -> Result<()> {
        // Apply margin side-effects only on fully successful decisions.
        let action = self.pending_margin_action.take();
        if action.is_none() {
            return Ok(());
        }

        match (outcome, action) {
            (quantlaxmi_paper::FillOutcome::AllFilled, Some(PendingMarginAction::Reserve(m))) => {
                self.reserve_margin_confirmed(m).await;
            }
            (quantlaxmi_paper::FillOutcome::AllFilled, Some(PendingMarginAction::Release)) => {
                self.release_margin_confirmed().await;
            }
            // Partial/Rejected => do nothing (no reserve/release); pending action already cleared.
            _ => {}
        }

        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::snapshot::{OptQuote, PriceQty};

    fn make_test_snapshot() -> OptionsSnapshot {
        let mut snapshot = OptionsSnapshot::new("NIFTY".into(), "2026-02-06".into());

        // Front expiry: 2026-02-06
        let mut front_ce = OptQuote::new(111, "NIFTY26FEB23400CE".into(), 23400, Right::Call);
        front_ce.bid = Some(PriceQty::new(150.0, 100));
        front_ce.ask = Some(PriceQty::new(152.0, 100));
        front_ce.age_ms = 100;
        front_ce.expiry = "2026-02-06".to_string();
        snapshot.quotes.push(front_ce);

        let mut front_pe = OptQuote::new(112, "NIFTY26FEB23400PE".into(), 23400, Right::Put);
        front_pe.bid = Some(PriceQty::new(145.0, 100));
        front_pe.ask = Some(PriceQty::new(147.0, 100));
        front_pe.age_ms = 100;
        front_pe.expiry = "2026-02-06".to_string();
        snapshot.quotes.push(front_pe);

        // Back expiry: 2026-02-13
        let mut back_ce = OptQuote::new(211, "NIFTY26FEB23400CE".into(), 23400, Right::Call);
        back_ce.bid = Some(PriceQty::new(180.0, 100));
        back_ce.ask = Some(PriceQty::new(183.0, 100));
        back_ce.age_ms = 100;
        back_ce.expiry = "2026-02-13".to_string();
        snapshot.quotes.push(back_ce);

        let mut back_pe = OptQuote::new(212, "NIFTY26FEB23400PE".into(), 23400, Right::Put);
        back_pe.bid = Some(PriceQty::new(175.0, 100));
        back_pe.ask = Some(PriceQty::new(178.0, 100));
        back_pe.age_ms = 100;
        back_pe.expiry = "2026-02-13".to_string();
        snapshot.quotes.push(back_pe);

        snapshot
    }

    #[test]
    fn test_find_atm_straddles() {
        let config = CalendarCarryConfig::default();
        let mut adapter = CalendarCarryAdapter::new(config);
        let snapshot = make_test_snapshot();
        let ts = Utc::now();

        let decision = adapter.evaluate(ts, &snapshot);

        // Should get a decision (either accept or refuse based on edge)
        match decision {
            StrategyDecisionExplicit::Accept { rationale, .. }
            | StrategyDecisionExplicit::Refuse { rationale, .. }
            | StrategyDecisionExplicit::Hold { rationale } => {
                assert_eq!(rationale.front_expiry, "2026-02-06");
                assert_eq!(rationale.back_expiry, "2026-02-13");
            }
        }
    }

    #[test]
    fn test_edge_computation() {
        let config = CalendarCarryConfig {
            entry_threshold_paise: 0, // Accept any edge for testing
            ..Default::default()
        };
        let mut adapter = CalendarCarryAdapter::new(config);
        let snapshot = make_test_snapshot();
        let ts = Utc::now();

        let decision = adapter.evaluate(ts, &snapshot);

        match decision {
            StrategyDecisionExplicit::Accept { rationale, .. } => {
                // Front: sell at bid = 150 + 145 = 295 (29500 paise)
                // Back: buy at ask = 183 + 178 = 361 (36100 paise)
                // Edge = 29500 - 36100 = -6600 paise (negative!)
                // This is expected for calendar carry when back is more expensive
                assert!(rationale.edge_estimate_paise < 0);
                assert!(rationale.front_mid_paise > 0);
                assert!(rationale.back_mid_paise > 0);
            }
            StrategyDecisionExplicit::Refuse { rationale, .. } => {
                // Edge might be negative, which is fine
                assert!(rationale.front_mid_paise > 0);
            }
            _ => {}
        }
    }

    #[test]
    fn test_stale_quote_rejection() {
        let config = CalendarCarryConfig::default();
        let mut adapter = CalendarCarryAdapter::new(config);
        let mut snapshot = make_test_snapshot();

        // Make one quote stale
        snapshot.quotes[0].age_ms = 10000; // 10 seconds

        let ts = Utc::now();
        let decision = adapter.evaluate(ts, &snapshot);

        match decision {
            StrategyDecisionExplicit::Refuse { reason, .. } => {
                assert!(matches!(reason, RefuseReason::StaleQuote { .. }));
            }
            _ => panic!("Expected stale quote rejection"),
        }
    }

    #[test]
    fn test_missing_bid_rejection() {
        let config = CalendarCarryConfig::default();
        let mut adapter = CalendarCarryAdapter::new(config);
        let mut snapshot = make_test_snapshot();

        // Remove bid from front CE
        snapshot.quotes[0].bid = None;

        let ts = Utc::now();
        let decision = adapter.evaluate(ts, &snapshot);

        match decision {
            StrategyDecisionExplicit::Refuse { reason, .. } => {
                assert!(matches!(reason, RefuseReason::NoExecutableQuote { .. }));
            }
            _ => panic!("Expected no executable quote rejection"),
        }
    }

    #[test]
    fn test_trade_intent_creation() {
        let intent = TradeIntent::new(
            12345,
            Side::Buy,
            2,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            25,
        );

        assert_eq!(intent.instrument_token, 12345);
        assert_eq!(intent.side, Side::Buy);
        assert_eq!(intent.qty, 2);
        assert_eq!(intent.tif, TimeInForce::IOC);
        assert_eq!(intent.tag, IntentTag::CalendarCarryEntry);
        assert_eq!(intent.lot_size, 25);
    }
}
