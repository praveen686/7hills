//! # NullObserverStrategy
//!
//! A diagnostic strategy that observes option market data without trading.
//!
//! ## Purpose
//! - Validate market coverage
//! - Validate time alignment
//! - Validate option chain completeness
//! - Log strikes, expiries, bid/ask, mid, spreads
//!
//! ## Design
//! This strategy is a diagnostic instrument, not alpha generation.
//! It generates zero trades and only emits heartbeat signals for pipeline continuity.

use crate::{EventBus, Strategy};
use kubera_models::{MarketEvent, MarketPayload, OrderEvent};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// Parsed option symbol components
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedOption {
    pub underlying: String,      // NIFTY, BANKNIFTY
    pub expiry: String,          // 26JAN (day + month)
    pub strike: u32,             // 25300
    pub option_type: OptionType, // CE or PE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    Call,
    Put,
}

impl std::fmt::Display for OptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionType::Call => write!(f, "CE"),
            OptionType::Put => write!(f, "PE"),
        }
    }
}

/// Observation record for a single option
#[derive(Debug, Clone, Default)]
pub struct OptionObservation {
    pub tick_count: u64,
    pub first_seen: Option<chrono::DateTime<chrono::Utc>>,
    pub last_seen: Option<chrono::DateTime<chrono::Utc>>,
    pub last_bid: f64,
    pub last_ask: f64,
    pub last_mid: f64,
    pub min_spread_bps: f64,
    pub max_spread_bps: f64,
    pub avg_spread_bps: f64,
    pub spread_samples: u64,
}

/// Strike ladder analysis for one expiry
#[derive(Debug, Clone, Default)]
pub struct StrikeLadder {
    pub expiry: String,
    pub strikes: BTreeSet<u32>,
    pub call_strikes: BTreeSet<u32>,
    pub put_strikes: BTreeSet<u32>,
    pub missing_calls: Vec<u32>,
    pub missing_puts: Vec<u32>,
}

/// NullObserverStrategy - observes options, logs diagnostics, zero trades
pub struct NullObserverStrategy {
    name: String,
    bus: Option<Arc<EventBus>>,

    // Observation state
    observations: HashMap<String, OptionObservation>, // symbol -> observation
    parsed_options: HashMap<String, ParsedOption>,    // symbol -> parsed

    // Strike ladder tracking
    ladders: HashMap<String, StrikeLadder>, // expiry -> ladder

    // Heartbeat
    last_heartbeat_ms: u64,
    heartbeat_interval_ms: u64,
    total_events: u64,

    // ATM tracking (for spread analysis)
    underlying_prices: HashMap<String, f64>, // NIFTY -> spot price estimate
}

impl NullObserverStrategy {
    pub fn new() -> Self {
        Self {
            name: "NullObserverStrategy".to_string(),
            bus: None,
            observations: HashMap::new(),
            parsed_options: HashMap::new(),
            ladders: HashMap::new(),
            last_heartbeat_ms: 0,
            heartbeat_interval_ms: 10_000, // 10 second heartbeats
            total_events: 0,
            underlying_prices: HashMap::new(),
        }
    }

    /// Parse Indian option symbol: NIFTY26JAN25300CE -> ParsedOption
    fn parse_symbol(symbol: &str) -> Option<ParsedOption> {
        // Pattern: UNDERLYING + EXPIRY(DDMMM) + STRIKE + TYPE(CE/PE)
        // Examples: NIFTY26JAN25300CE, BANKNIFTY26JAN59100PE

        let symbol = symbol.to_uppercase();

        // Determine option type
        let option_type = if symbol.ends_with("CE") {
            OptionType::Call
        } else if symbol.ends_with("PE") {
            OptionType::Put
        } else {
            return None;
        };

        // Remove option type suffix
        let without_type = &symbol[..symbol.len() - 2];

        // Find where the strike starts (first digit after expiry month)
        // Expiry format: DDMMM (e.g., 26JAN)
        let underlying;
        let expiry;
        let strike;

        if without_type.starts_with("BANKNIFTY") {
            underlying = "BANKNIFTY".to_string();
            let rest = &without_type[9..]; // After "BANKNIFTY"
            // Format: 26JAN59100
            if rest.len() < 6 {
                return None;
            }
            expiry = rest[..5].to_string(); // 26JAN
            strike = rest[5..].parse::<u32>().ok()?;
        } else if without_type.starts_with("NIFTY") {
            underlying = "NIFTY".to_string();
            let rest = &without_type[5..]; // After "NIFTY"
            if rest.len() < 6 {
                return None;
            }
            expiry = rest[..5].to_string(); // 26JAN
            strike = rest[5..].parse::<u32>().ok()?;
        } else if without_type.starts_with("FINNIFTY") {
            underlying = "FINNIFTY".to_string();
            let rest = &without_type[8..];
            if rest.len() < 6 {
                return None;
            }
            expiry = rest[..5].to_string();
            strike = rest[5..].parse::<u32>().ok()?;
        } else {
            return None;
        }

        Some(ParsedOption {
            underlying,
            expiry,
            strike,
            option_type,
        })
    }

    /// Calculate spread in basis points
    fn spread_bps(bid: f64, ask: f64) -> f64 {
        if bid <= 0.0 || ask <= 0.0 {
            return 0.0;
        }
        let mid = (bid + ask) / 2.0;
        if mid <= 0.0 {
            return 0.0;
        }
        ((ask - bid) / mid) * 10_000.0
    }

    /// Update observation for a symbol
    fn observe(&mut self, symbol: &str, ts: chrono::DateTime<chrono::Utc>, bid: f64, ask: f64) {
        let mid = (bid + ask) / 2.0;
        let spread = Self::spread_bps(bid, ask);

        let obs = self.observations.entry(symbol.to_string()).or_default();
        obs.tick_count += 1;

        if obs.first_seen.is_none() {
            obs.first_seen = Some(ts);
        }
        obs.last_seen = Some(ts);
        obs.last_bid = bid;
        obs.last_ask = ask;
        obs.last_mid = mid;

        if spread > 0.0 {
            if obs.spread_samples == 0 {
                obs.min_spread_bps = spread;
                obs.max_spread_bps = spread;
            } else {
                obs.min_spread_bps = obs.min_spread_bps.min(spread);
                obs.max_spread_bps = obs.max_spread_bps.max(spread);
            }
            // Running average
            obs.avg_spread_bps = (obs.avg_spread_bps * obs.spread_samples as f64 + spread)
                                 / (obs.spread_samples + 1) as f64;
            obs.spread_samples += 1;
        }

        // Parse and track in ladder
        if !self.parsed_options.contains_key(symbol) {
            if let Some(parsed) = Self::parse_symbol(symbol) {
                self.parsed_options.insert(symbol.to_string(), parsed.clone());

                // Add to ladder
                let ladder = self.ladders.entry(parsed.expiry.clone()).or_insert_with(|| {
                    StrikeLadder {
                        expiry: parsed.expiry.clone(),
                        ..Default::default()
                    }
                });

                ladder.strikes.insert(parsed.strike);
                match parsed.option_type {
                    OptionType::Call => { ladder.call_strikes.insert(parsed.strike); }
                    OptionType::Put => { ladder.put_strikes.insert(parsed.strike); }
                }
            }
        }

        // Estimate underlying price from ATM options
        if let Some(parsed) = self.parsed_options.get(symbol) {
            // If this is near ATM (mid price roughly equals strike), track as underlying estimate
            // This is a rough heuristic: ATM options have premium ~ 0
            let intrinsic_estimate = mid; // Premium-based estimate
            if intrinsic_estimate > 50.0 && intrinsic_estimate < 1000.0 {
                // Reasonable option premium range - likely near ATM
                self.underlying_prices.insert(
                    parsed.underlying.clone(),
                    parsed.strike as f64
                );
            }
        }
    }

    /// Analyze strike ladder for completeness
    fn analyze_ladders(&mut self) {
        for ladder in self.ladders.values_mut() {
            if ladder.strikes.len() < 2 {
                continue;
            }

            // Detect strike interval
            let strikes: Vec<u32> = ladder.strikes.iter().cloned().collect();
            let intervals: Vec<u32> = strikes.windows(2)
                .map(|w| w[1] - w[0])
                .collect();

            // Most common interval
            let mut interval_counts: HashMap<u32, usize> = HashMap::new();
            for &i in &intervals {
                *interval_counts.entry(i).or_insert(0) += 1;
            }

            let expected_interval = interval_counts.into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(interval, _)| interval)
                .unwrap_or(50);

            // Find missing strikes
            if let (Some(&min_strike), Some(&max_strike)) = (strikes.first(), strikes.last()) {
                let mut expected_strike = min_strike;
                while expected_strike <= max_strike {
                    if !ladder.call_strikes.contains(&expected_strike) {
                        ladder.missing_calls.push(expected_strike);
                    }
                    if !ladder.put_strikes.contains(&expected_strike) {
                        ladder.missing_puts.push(expected_strike);
                    }
                    expected_strike += expected_interval;
                }
            }
        }
    }

    /// Generate diagnostic report
    pub fn generate_report(&mut self) -> OptionChainReport {
        self.analyze_ladders();

        let mut expiries = Vec::new();
        for (expiry, ladder) in &self.ladders {
            expiries.push(ExpiryReport {
                expiry: expiry.clone(),
                total_strikes: ladder.strikes.len(),
                call_strikes: ladder.call_strikes.len(),
                put_strikes: ladder.put_strikes.len(),
                missing_calls: ladder.missing_calls.len(),
                missing_puts: ladder.missing_puts.len(),
                strike_range: if ladder.strikes.is_empty() {
                    (0, 0)
                } else {
                    (*ladder.strikes.first().unwrap(), *ladder.strikes.last().unwrap())
                },
            });
        }

        // ATM spread analysis (strikes closest to underlying)
        let mut atm_spreads = Vec::new();
        for (symbol, obs) in &self.observations {
            if let Some(parsed) = self.parsed_options.get(symbol) {
                if let Some(&underlying_price) = self.underlying_prices.get(&parsed.underlying) {
                    let moneyness = (parsed.strike as f64 - underlying_price).abs() / underlying_price * 100.0;
                    if moneyness < 2.0 { // Within 2% of ATM
                        atm_spreads.push(AtmSpreadReport {
                            symbol: symbol.clone(),
                            strike: parsed.strike,
                            option_type: parsed.option_type,
                            avg_spread_bps: obs.avg_spread_bps,
                            min_spread_bps: obs.min_spread_bps,
                            max_spread_bps: obs.max_spread_bps,
                            tick_count: obs.tick_count,
                        });
                    }
                }
            }
        }

        OptionChainReport {
            total_events: self.total_events,
            unique_symbols: self.observations.len(),
            expiries,
            atm_spreads,
            underlying_estimates: self.underlying_prices.clone(),
        }
    }
}

impl Default for NullObserverStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for NullObserverStrategy {
    fn on_start(&mut self, bus: Arc<EventBus>) {
        tracing::info!("[{}] Starting - observation mode, zero trades", self.name);
        self.bus = Some(bus);
    }

    fn on_tick(&mut self, event: &MarketEvent) {
        self.total_events += 1;

        // Extract bid/ask from L2 snapshot
        let (bid, ask) = match &event.payload {
            MarketPayload::L2Snapshot(snap) => {
                let bid = snap.bids.first().map(|l| l.price).unwrap_or(0.0);
                let ask = snap.asks.first().map(|l| l.price).unwrap_or(0.0);
                (bid, ask)
            }
            MarketPayload::L2Update(update) => {
                let bid = update.bids.iter().find(|l| l.size > 0.0).map(|l| l.price).unwrap_or(0.0);
                let ask = update.asks.iter().find(|l| l.size > 0.0).map(|l| l.price).unwrap_or(0.0);
                (bid, ask)
            }
            MarketPayload::Tick { price, .. } => (*price, *price), // Tick has single price, use as both bid/ask
            _ => return,
        };

        self.observe(&event.symbol, event.exchange_time, bid, ask);
    }

    fn on_bar(&mut self, _event: &MarketEvent) {
        // Options don't use bars
    }

    fn on_fill(&mut self, _fill: &OrderEvent) {
        // No trades, no fills
    }

    fn on_signal_timer(&mut self, elapsed_ms: u64) {
        if elapsed_ms - self.last_heartbeat_ms >= self.heartbeat_interval_ms {
            self.last_heartbeat_ms = elapsed_ms;
            tracing::debug!(
                "[{}] Heartbeat: {} events, {} symbols observed",
                self.name, self.total_events, self.observations.len()
            );
        }
    }

    fn on_stop(&mut self) {
        tracing::info!("[{}] Stopped. Total events: {}", self.name, self.total_events);
    }

    fn name(&self) -> &str { &self.name }
}

/// Report structures for option chain analysis
#[derive(Debug, Clone)]
pub struct OptionChainReport {
    pub total_events: u64,
    pub unique_symbols: usize,
    pub expiries: Vec<ExpiryReport>,
    pub atm_spreads: Vec<AtmSpreadReport>,
    pub underlying_estimates: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ExpiryReport {
    pub expiry: String,
    pub total_strikes: usize,
    pub call_strikes: usize,
    pub put_strikes: usize,
    pub missing_calls: usize,
    pub missing_puts: usize,
    pub strike_range: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct AtmSpreadReport {
    pub symbol: String,
    pub strike: u32,
    pub option_type: OptionType,
    pub avg_spread_bps: f64,
    pub min_spread_bps: f64,
    pub max_spread_bps: f64,
    pub tick_count: u64,
}

impl std::fmt::Display for OptionChainReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Option Chain Analysis ===")?;
        writeln!(f, "Total Events: {}", self.total_events)?;
        writeln!(f, "Unique Symbols: {}", self.unique_symbols)?;
        writeln!(f)?;

        writeln!(f, "EXPIRIES:")?;
        for exp in &self.expiries {
            writeln!(f, "  {} | Strikes: {} (CE:{}, PE:{}) | Range: {}-{} | Missing: CE={}, PE={}",
                exp.expiry, exp.total_strikes, exp.call_strikes, exp.put_strikes,
                exp.strike_range.0, exp.strike_range.1,
                exp.missing_calls, exp.missing_puts)?;
        }
        writeln!(f)?;

        writeln!(f, "ATM SPREAD BEHAVIOR:")?;
        for atm in &self.atm_spreads {
            writeln!(f, "  {} | Avg: {:.1} bps | Min: {:.1} | Max: {:.1} | Ticks: {}",
                atm.symbol, atm.avg_spread_bps, atm.min_spread_bps, atm.max_spread_bps, atm.tick_count)?;
        }
        writeln!(f)?;

        writeln!(f, "UNDERLYING ESTIMATES:")?;
        for (underlying, price) in &self.underlying_estimates {
            writeln!(f, "  {}: ~{:.0}", underlying, price)?;
        }

        Ok(())
    }
}
