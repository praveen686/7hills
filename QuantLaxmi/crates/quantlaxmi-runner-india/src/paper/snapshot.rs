//! India F&O Options Snapshot Schema
//!
//! Defines the snapshot structure for paper trading with:
//! - Staleness tracking (age_ms per quote) - DETERMINISTIC via nanosecond timestamps
//! - Fixed-point pricing compatibility (mantissa/exponent for future)
//! - Full option chain representation (calls and puts)
//!
//! ## Determinism Guarantee
//!
//! All staleness computation uses nanosecond timestamps, NOT wall-clock.
//! This ensures identical replay behavior regardless of execution speed.

use quantlaxmi_paper::TopOfBookProvider;
use serde::{Deserialize, Serialize};

/// Option right (call or put).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Right {
    Call,
    Put,
}

impl Right {
    /// Parse from Zerodha instrument_type ("CE" or "PE").
    pub fn from_zerodha(s: &str) -> Option<Self> {
        match s {
            "CE" => Some(Right::Call),
            "PE" => Some(Right::Put),
            _ => None,
        }
    }

    /// Convert to Zerodha instrument_type string.
    pub fn to_zerodha(&self) -> &'static str {
        match self {
            Right::Call => "CE",
            Right::Put => "PE",
        }
    }
}

/// Price and quantity pair for bid/ask.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PriceQty {
    /// Price in rupees (float for now, consider mantissa for precision).
    pub price: f64,
    /// Quantity available at this price level.
    pub qty: u32,
}

impl PriceQty {
    pub fn new(price: f64, qty: u32) -> Self {
        Self { price, qty }
    }

    /// Check if this level is valid (non-zero price and qty).
    pub fn is_valid(&self) -> bool {
        self.price > 0.0 && self.qty > 0
    }
}

/// Individual option quote with staleness tracking.
///
/// ## Staleness Model
///
/// `last_update_ns` stores the nanosecond timestamp of the last tick.
/// `age_ms` is computed deterministically as:
/// ```text
/// age_ms = ((snapshot_ts_ns - last_update_ns) / 1_000_000).max(0) as u32
/// ```
///
/// This ensures replay produces identical staleness values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptQuote {
    /// Zerodha instrument token for this option.
    pub instrument_token: u32,
    /// Trading symbol (e.g., "NIFTY2510223400CE").
    pub tradingsymbol: String,
    /// Strike price in index points (e.g., 23400 for NIFTY 23400).
    /// INVARIANT: Always in index points, never scaled/paise.
    pub strike: i32,
    /// Option right (Call or Put).
    pub right: Right,
    /// Expiry date in canonical format (YYYY-MM-DD).
    pub expiry: String,
    /// Best bid (if available). None = no bid, NOT zero.
    pub bid: Option<PriceQty>,
    /// Best ask (if available). None = no ask, NOT zero.
    pub ask: Option<PriceQty>,
    /// Last traded price (if available).
    pub last: Option<PriceQty>,
    /// Age of this quote in milliseconds since last tick update.
    /// Computed deterministically from nanosecond timestamps.
    /// Used for staleness detection (e.g., > 5000ms = stale).
    pub age_ms: u32,
    /// Nanosecond timestamp of last update (for deterministic staleness).
    /// This is the tick's timestamp, not wall-clock.
    pub last_update_ns: i64,
}

impl OptQuote {
    /// Create a new quote from tick data.
    pub fn new(instrument_token: u32, tradingsymbol: String, strike: i32, right: Right) -> Self {
        Self {
            instrument_token,
            tradingsymbol,
            strike,
            right,
            expiry: String::new(), // Will be set from instrument metadata
            bid: None,
            ask: None,
            last: None,
            age_ms: 0,
            last_update_ns: 0, // Will be set on first tick
        }
    }

    /// Create a new quote with expiry.
    pub fn with_expiry(
        instrument_token: u32,
        tradingsymbol: String,
        strike: i32,
        right: Right,
        expiry: String,
    ) -> Self {
        Self {
            instrument_token,
            tradingsymbol,
            strike,
            right,
            expiry,
            bid: None,
            ask: None,
            last: None,
            age_ms: 0,
            last_update_ns: 0,
        }
    }

    /// Update quote from parsed tick with explicit timestamp.
    ///
    /// `tick_ts_ns`: Nanosecond timestamp from the tick source (NOT wall-clock).
    pub fn update_from_tick(
        &mut self,
        tick_ts_ns: i64,
        bid: Option<PriceQty>,
        ask: Option<PriceQty>,
        ltp: Option<f64>,
        volume: Option<u32>,
    ) {
        self.bid = bid;
        self.ask = ask;
        if let Some(p) = ltp {
            self.last = Some(PriceQty::new(p, volume.unwrap_or(0)));
        }
        self.last_update_ns = tick_ts_ns;
        self.age_ms = 0; // Will be recomputed on snapshot emit
    }

    /// Compute staleness deterministically from snapshot timestamp.
    ///
    /// `snapshot_ts_ns`: The snapshot's nanosecond timestamp.
    ///
    /// This MUST be called during snapshot building, not during tick processing.
    pub fn compute_age(&mut self, snapshot_ts_ns: i64) {
        if self.last_update_ns == 0 {
            // Never updated - mark as maximally stale
            self.age_ms = u32::MAX;
        } else {
            let delta_ns = snapshot_ts_ns.saturating_sub(self.last_update_ns);
            self.age_ms = (delta_ns / 1_000_000).clamp(0, u32::MAX as i64) as u32;
        }
    }

    /// Check if quote is stale (above threshold).
    pub fn is_stale(&self, threshold_ms: u32) -> bool {
        self.age_ms > threshold_ms
    }

    /// Get mid price if both bid and ask are available.
    pub fn mid(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) if b.is_valid() && a.is_valid() => Some((b.price + a.price) / 2.0),
            _ => None,
        }
    }

    /// Get spread in rupees if both bid and ask are available.
    pub fn spread(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) if b.is_valid() && a.is_valid() => Some(a.price - b.price),
            _ => None,
        }
    }
}

/// Snapshot provenance for audit and replay diagnostics.
///
/// Every snapshot carries provenance to explain its origin and quality.
/// This is non-negotiable for:
/// - Diagnosing feed dropouts
/// - Explaining bad paper PnL
/// - Regulatory/audit-grade replay
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SnapshotProvenance {
    /// Source identifier (e.g., "zerodha_ws", "binance_ws", "replay_file").
    pub source: String,
    /// Connection ID (increments on reconnect).
    pub connection_id: u64,
    /// Snapshot sequence number (monotonic within connection).
    pub seq: u64,
    /// Ticks processed since last snapshot.
    pub ticks_since_last: u32,
    /// Number of subscribed tokens.
    pub subscribed_tokens: u32,
    /// Ticks dropped due to unknown tokens or parse errors.
    pub dropped_ticks: u32,
    /// Quotes that are stale (age_ms > threshold).
    pub stale_quotes: u32,
}

/// Full options snapshot for paper trading.
///
/// Represents the current state of an option chain slice (one underlying,
/// one expiry, multiple strikes). Updated periodically from tick stream.
///
/// ## Determinism
///
/// All timestamps are nanoseconds. Staleness is computed from these,
/// never from wall-clock. This ensures identical replay behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsSnapshot {
    /// Snapshot timestamp (nanoseconds, monotonic within session).
    pub ts_ns: i64,
    /// Underlying name (e.g., "NIFTY", "BANKNIFTY").
    pub underlying: String,
    /// Expiry date as canonical string (YYYY-MM-DD format, e.g., "2025-10-02").
    /// INVARIANT: Always YYYY-MM-DD, never parsed differently downstream.
    pub expiry: String,
    /// Spot price of underlying (if available).
    pub spot: Option<f64>,
    /// All option quotes in this snapshot.
    pub quotes: Vec<OptQuote>,
    /// Provenance for audit trail.
    pub provenance: SnapshotProvenance,
}

impl TopOfBookProvider for OptionsSnapshot {
    fn best_bid_ask(&self, token: u32) -> Option<(f64, f64)> {
        self.get_quote(token).and_then(|q| match (q.bid, q.ask) {
            (Some(b), Some(a)) if b.is_valid() && a.is_valid() => Some((b.price, a.price)),
            _ => None,
        })
    }
}

impl OptionsSnapshot {
    /// Create a new empty snapshot.
    pub fn new(underlying: String, expiry: String) -> Self {
        Self {
            ts_ns: 0,
            underlying,
            expiry,
            spot: None,
            quotes: Vec::new(),
            provenance: SnapshotProvenance::default(),
        }
    }

    /// Finalize snapshot for emission.
    ///
    /// - Sets timestamp
    /// - Computes all quote ages deterministically
    /// - Updates provenance
    pub fn finalize(&mut self, ts_ns: i64, provenance: SnapshotProvenance) {
        self.ts_ns = ts_ns;
        self.provenance = provenance;

        // Compute staleness for all quotes deterministically
        for q in &mut self.quotes {
            q.compute_age(ts_ns);
        }
    }

    /// Get quote by instrument token.
    pub fn get_quote(&self, token: u32) -> Option<&OptQuote> {
        self.quotes.iter().find(|q| q.instrument_token == token)
    }

    /// Get mutable quote by instrument token.
    pub fn get_quote_mut(&mut self, token: u32) -> Option<&mut OptQuote> {
        self.quotes.iter_mut().find(|q| q.instrument_token == token)
    }

    /// Get quote by strike and right.
    pub fn get_by_strike(&self, strike: i32, right: Right) -> Option<&OptQuote> {
        self.quotes
            .iter()
            .find(|q| q.strike == strike && q.right == right)
    }

    /// Count stale quotes (above threshold).
    pub fn stale_count(&self, threshold_ms: u32) -> usize {
        self.quotes
            .iter()
            .filter(|q| q.is_stale(threshold_ms))
            .count()
    }

    /// Check if all quotes are fresh (below threshold).
    pub fn is_all_fresh(&self, threshold_ms: u32) -> bool {
        self.quotes.iter().all(|q| !q.is_stale(threshold_ms))
    }

    /// Get all call quotes sorted by strike.
    pub fn calls(&self) -> Vec<&OptQuote> {
        let mut calls: Vec<_> = self
            .quotes
            .iter()
            .filter(|q| q.right == Right::Call)
            .collect();
        calls.sort_by_key(|q| q.strike);
        calls
    }

    /// Get all put quotes sorted by strike.
    pub fn puts(&self) -> Vec<&OptQuote> {
        let mut puts: Vec<_> = self
            .quotes
            .iter()
            .filter(|q| q.right == Right::Put)
            .collect();
        puts.sort_by_key(|q| q.strike);
        puts
    }

    /// Find ATM strike (nearest to spot).
    pub fn atm_strike(&self) -> Option<i32> {
        let spot = self.spot?;
        self.quotes
            .iter()
            .map(|q| q.strike)
            .min_by_key(|&s| ((s as f64 - spot).abs() * 100.0) as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_right_parsing() {
        assert_eq!(Right::from_zerodha("CE"), Some(Right::Call));
        assert_eq!(Right::from_zerodha("PE"), Some(Right::Put));
        assert_eq!(Right::from_zerodha("FUT"), None);
    }

    #[test]
    fn test_deterministic_staleness() {
        let mut quote = OptQuote::new(123456, "NIFTY2510223400CE".into(), 23400, Right::Call);

        // Simulate tick at t=1000ms (1_000_000_000 ns)
        let tick_ts = 1_000_000_000i64;
        quote.update_from_tick(tick_ts, None, None, Some(100.0), Some(10));

        // Snapshot at t=7000ms (7_000_000_000 ns)
        let snapshot_ts = 7_000_000_000i64;
        quote.compute_age(snapshot_ts);

        // Age should be exactly 6000ms
        assert_eq!(quote.age_ms, 6000);
        assert!(quote.is_stale(5000));
        assert!(!quote.is_stale(7000));
    }

    #[test]
    fn test_never_updated_quote_is_max_stale() {
        let mut quote = OptQuote::new(123456, "NIFTY2510223400CE".into(), 23400, Right::Call);
        quote.compute_age(1_000_000_000);

        assert_eq!(quote.age_ms, u32::MAX);
        assert!(quote.is_stale(0));
    }

    #[test]
    fn test_snapshot_quotes() {
        let mut snap = OptionsSnapshot::new("NIFTY".into(), "2025-10-02".into());
        snap.spot = Some(23500.0);

        snap.quotes.push(OptQuote::new(
            1,
            "NIFTY2510223400CE".into(),
            23400,
            Right::Call,
        ));
        snap.quotes.push(OptQuote::new(
            2,
            "NIFTY2510223400PE".into(),
            23400,
            Right::Put,
        ));
        snap.quotes.push(OptQuote::new(
            3,
            "NIFTY2510223500CE".into(),
            23500,
            Right::Call,
        ));

        assert_eq!(snap.calls().len(), 2);
        assert_eq!(snap.puts().len(), 1);
        assert_eq!(snap.atm_strike(), Some(23500));
    }

    #[test]
    fn test_provenance_default() {
        let prov = SnapshotProvenance::default();
        assert_eq!(prov.connection_id, 0);
        assert_eq!(prov.seq, 0);
        assert!(prov.source.is_empty());
    }
}
