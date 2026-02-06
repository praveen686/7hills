//! Instrument Token Mapping
//!
//! Maps Zerodha instrument tokens to option metadata.
//! Used by MarketFeedZerodha to route ticks to the correct quote slots.
//!
//! ## Invariants
//!
//! ### Strike Normalization
//! - Strikes are ALWAYS in index points (e.g., 19600, 23400)
//! - NEVER scaled to paise or any other unit
//! - NEVER mixed with price mantissas
//!
//! ### Expiry Canonicalization
//! - Expiry is stored as `NaiveDate` (source of truth)
//! - `expiry_canonical` provides YYYY-MM-DD string (computed once)
//! - Downstream code MUST NOT re-parse expiry strings

use chrono::NaiveDate;
use std::collections::HashMap;
use tracing::info;

use crate::paper::snapshot::Right;

/// Metadata for an instrument token.
///
/// ## Invariants
///
/// - `strike`: Always in index points (e.g., 23400), never paise
/// - `expiry`: Canonical NaiveDate, never re-parsed downstream
/// - `expiry_canonical`: Pre-computed YYYY-MM-DD string for serialization
#[derive(Debug, Clone)]
pub struct InstrumentMeta {
    /// Zerodha instrument token.
    pub token: u32,
    /// Trading symbol (e.g., "NIFTY2510223400CE").
    pub tradingsymbol: String,
    /// Underlying name (e.g., "NIFTY").
    pub underlying: String,
    /// Expiry date (source of truth).
    pub expiry: NaiveDate,
    /// Canonical expiry string (YYYY-MM-DD). Computed once, never re-parsed.
    pub expiry_canonical: String,
    /// Strike price in index points (e.g., 23400 for NIFTY 23400).
    /// INVARIANT: Always index points, never scaled/paise.
    pub strike: i32,
    /// Option right (Call or Put).
    pub right: Right,
    /// Lot size for position sizing.
    pub lot_size: u32,
}

impl InstrumentMeta {
    /// Create new metadata with canonical expiry string computed automatically.
    pub fn new(
        token: u32,
        tradingsymbol: String,
        underlying: String,
        expiry: NaiveDate,
        strike: i32,
        right: Right,
        lot_size: u32,
    ) -> Self {
        Self {
            token,
            tradingsymbol,
            underlying,
            expiry_canonical: expiry.format("%Y-%m-%d").to_string(),
            expiry,
            strike,
            right,
            lot_size,
        }
    }
}

/// Bidirectional mapping between instrument tokens and metadata.
///
/// All invariants are enforced at insertion time:
/// - Strikes normalized to index points
/// - Expiry canonicalized to YYYY-MM-DD
#[derive(Debug, Clone, Default)]
pub struct InstrumentMap {
    /// Token → Metadata lookup.
    by_token: HashMap<u32, InstrumentMeta>,
    /// Tradingsymbol → Token lookup.
    by_symbol: HashMap<String, u32>,
    /// (Strike, Right) → Token lookup for quick access.
    by_strike: HashMap<(i32, Right), u32>,
    /// Canonical expiry string (set on first insert, asserted same for all).
    canonical_expiry: Option<String>,
    /// Underlying (set on first insert, asserted same for all).
    canonical_underlying: Option<String>,
}

impl InstrumentMap {
    /// Create a new empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an instrument into the map.
    ///
    /// Allows multiple expiries for calendar spread strategies.
    pub fn insert(&mut self, meta: InstrumentMeta) {
        // Enforce single underlying per map
        if let Some(ref u) = self.canonical_underlying {
            debug_assert_eq!(
                u, &meta.underlying,
                "InstrumentMap: mixed underlyings not allowed"
            );
        } else {
            self.canonical_underlying = Some(meta.underlying.clone());
        }

        // Track first expiry seen (for backward compatibility)
        // Note: Calendar spreads require multiple expiries, so we don't enforce single expiry
        if self.canonical_expiry.is_none() {
            self.canonical_expiry = Some(meta.expiry_canonical.clone());
        }

        self.by_symbol
            .insert(meta.tradingsymbol.clone(), meta.token);
        self.by_strike.insert((meta.strike, meta.right), meta.token);
        self.by_token.insert(meta.token, meta);
    }

    /// Build from a list of UniverseInstruments (from Zerodha auto-discovery).
    pub fn from_universe(
        instruments: &[quantlaxmi_connectors_zerodha::UniverseInstrument],
    ) -> Self {
        let mut map = Self::new();

        for ins in instruments {
            if let Some(right) = Right::from_zerodha(&ins.instrument_type) {
                let meta = InstrumentMeta::new(
                    ins.instrument_token,
                    ins.tradingsymbol.clone(),
                    ins.underlying.clone(),
                    ins.expiry,
                    ins.strike as i32,
                    right,
                    ins.lot_size,
                );
                map.insert(meta);
            }
        }

        info!("[MAPPING] Built instrument map with {} entries", map.len());
        map
    }

    /// Build from (tradingsymbol, token) pairs and NFO instruments lookup.
    pub fn from_tokens_and_instruments(
        tokens: &[(String, u32)],
        instruments: &[quantlaxmi_connectors_zerodha::NfoInstrument],
    ) -> Self {
        let mut map = Self::new();

        // Build lookup by token
        let ins_by_token: HashMap<u32, &quantlaxmi_connectors_zerodha::NfoInstrument> = instruments
            .iter()
            .map(|i| (i.instrument_token, i))
            .collect();

        for (symbol, token) in tokens {
            if let Some(ins) = ins_by_token.get(token)
                && let Some(right) = Right::from_zerodha(&ins.instrument_type)
            {
                let meta = InstrumentMeta::new(
                    *token,
                    symbol.clone(),
                    ins.name.clone(),
                    ins.expiry,
                    ins.strike as i32,
                    right,
                    ins.lot_size,
                );
                map.insert(meta);
            } else {
                let reason = if !ins_by_token.contains_key(token) {
                    "token not found in NFO instruments"
                } else {
                    "instrument_type not CE/PE"
                };
                eprintln!("[MAPPING] SKIP: {} token={} — {}", symbol, token, reason);
            }
        }

        eprintln!(
            "[MAPPING] Built instrument map with {} entries from {} tokens",
            map.len(),
            tokens.len()
        );
        map
    }

    /// Get metadata by token.
    pub fn get(&self, token: u32) -> Option<&InstrumentMeta> {
        self.by_token.get(&token)
    }

    /// Get token by trading symbol.
    pub fn get_token(&self, symbol: &str) -> Option<u32> {
        self.by_symbol.get(symbol).copied()
    }

    /// Get token by strike and right.
    pub fn get_by_strike(&self, strike: i32, right: Right) -> Option<u32> {
        self.by_strike.get(&(strike, right)).copied()
    }

    /// Check if token exists.
    pub fn contains(&self, token: u32) -> bool {
        self.by_token.contains_key(&token)
    }

    /// Number of instruments in the map.
    pub fn len(&self) -> usize {
        self.by_token.len()
    }

    /// Check if map is empty.
    pub fn is_empty(&self) -> bool {
        self.by_token.is_empty()
    }

    /// Iterate over all tokens.
    pub fn tokens(&self) -> impl Iterator<Item = u32> + '_ {
        self.by_token.keys().copied()
    }

    /// Iterate over all metadata.
    pub fn instruments(&self) -> impl Iterator<Item = &InstrumentMeta> + '_ {
        self.by_token.values()
    }

    /// Get all tokens as a Vec (for WebSocket subscription).
    pub fn token_list(&self) -> Vec<u32> {
        self.by_token.keys().copied().collect()
    }

    /// Get canonical expiry string (YYYY-MM-DD).
    /// Returns None if map is empty.
    pub fn expiry_str(&self) -> Option<&str> {
        self.canonical_expiry.as_deref()
    }

    /// Get underlying name.
    /// Returns None if map is empty.
    pub fn underlying(&self) -> Option<&str> {
        self.canonical_underlying.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_meta_new() {
        let meta = InstrumentMeta::new(
            123456,
            "NIFTY2510223400CE".into(),
            "NIFTY".into(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
            23400,
            Right::Call,
            25,
        );

        assert_eq!(meta.expiry_canonical, "2025-10-02");
        assert_eq!(meta.strike, 23400);
    }

    #[test]
    fn test_instrument_map() {
        let mut map = InstrumentMap::new();

        map.insert(InstrumentMeta::new(
            123456,
            "NIFTY2510223400CE".into(),
            "NIFTY".into(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
            23400,
            Right::Call,
            25,
        ));

        assert_eq!(map.len(), 1);
        assert!(map.contains(123456));
        assert_eq!(map.get_token("NIFTY2510223400CE"), Some(123456));
        assert_eq!(map.get_by_strike(23400, Right::Call), Some(123456));
        assert_eq!(map.get_by_strike(23400, Right::Put), None);
        assert_eq!(map.expiry_str(), Some("2025-10-02"));
        assert_eq!(map.underlying(), Some("NIFTY"));
    }

    #[test]
    fn test_canonical_strings_set_once() {
        let mut map = InstrumentMap::new();

        map.insert(InstrumentMeta::new(
            1,
            "NIFTY2510223400CE".into(),
            "NIFTY".into(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
            23400,
            Right::Call,
            25,
        ));

        map.insert(InstrumentMeta::new(
            2,
            "NIFTY2510223400PE".into(),
            "NIFTY".into(),
            NaiveDate::from_ymd_opt(2025, 10, 2).unwrap(),
            23400,
            Right::Put,
            25,
        ));

        assert_eq!(map.len(), 2);
        assert_eq!(map.expiry_str(), Some("2025-10-02"));
        assert_eq!(map.underlying(), Some("NIFTY"));
    }
}
