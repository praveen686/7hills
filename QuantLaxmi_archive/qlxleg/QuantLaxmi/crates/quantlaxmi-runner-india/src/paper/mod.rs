//! India F&O Paper Trading Module
//!
//! Wires the venue-agnostic `quantlaxmi-paper` spine to:
//! - Zerodha WebSocket feed (market data)
//! - India F&O fee model (STT, brokerage, exchange charges)
//! - Options strategies (calendar carry, etc.)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PaperEngine<F, S, M>                         │
//! │  (from quantlaxmi-paper, venue-agnostic orchestration)          │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//! ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
//! │MarketFeedZero-│   │   Strategy    │   │IndiaFnoFill-  │
//! │dha (this mod) │   │(calendar carry│   │Model (STT,    │
//! │               │   │ etc.)         │   │ brokerage)    │
//! └───────────────┘   └───────────────┘   └───────────────┘
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────────────┐
//! │ Zerodha Kite WebSocket (184-byte full mode L2 depth)  │
//! └───────────────────────────────────────────────────────┘
//! ```

pub mod feed_zerodha;
pub mod fees_india;
pub mod fill_model_india;
pub mod mapping;
pub mod margin;
pub mod snapshot;
pub mod strategy_adapter;

// Re-exports for convenience
pub use feed_zerodha::{FeedConfig, MarketFeedZerodha};
pub use fees_india::{
    BrokerageModel, FeesBreakdown, FillRejected, IndiaFnoFeeCalculator, IndiaFnoFeeConfig,
    InstrumentKind, Side,
};
pub use fill_model_india::{IndiaFill, IndiaFnoFillModel};
pub use mapping::{InstrumentMap, InstrumentMeta};
pub use margin::{MarginGate, MarginOrderParams, MarginRejectReason, MarginRequirement};
pub use snapshot::{OptQuote, OptionsSnapshot, PriceQty, Right, SnapshotProvenance};
pub use strategy_adapter::{
    CalendarCarryAdapter, CalendarCarryConfig, CalendarCarryStrategyWrapper, CalendarPosition,
    GateStatus, IntentTag, Rationale, RefuseReason, StrategyDecisionExplicit, TimeInForce,
    TradeIntent,
};

use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::{Mutex, watch};
use tracing::info;

use quantlaxmi_connectors_zerodha::ZerodhaAutoDiscovery;
use quantlaxmi_paper::{EngineConfig, PaperEngine, PaperState};

/// Paper trading configuration for India F&O.
#[derive(Debug, Clone)]
pub struct IndiaPaperConfig {
    /// Underlying to trade (e.g., "NIFTY", "BANKNIFTY").
    pub underlying: String,
    /// Number of strikes around ATM to include.
    pub strikes_around_atm: u32,
    /// Initial capital in INR.
    pub initial_capital: f64,
    /// Feed configuration.
    pub feed_config: FeedConfig,
    /// Strategy configuration.
    pub strategy_config: CalendarCarryConfig,
    /// Fee configuration.
    pub fee_config: IndiaFnoFeeConfig,
    /// Duration in seconds (0 = run until interrupted).
    pub duration_secs: u64,
    /// Enable decision logging.
    pub log_decisions: bool,
}

impl Default for IndiaPaperConfig {
    fn default() -> Self {
        Self {
            underlying: "NIFTY".to_string(),
            strikes_around_atm: 5,
            initial_capital: 1_000_000.0, // 10 lakh INR
            feed_config: FeedConfig::default(),
            strategy_config: CalendarCarryConfig::default(),
            fee_config: IndiaFnoFeeConfig::default(),
            duration_secs: 120, // 2 minutes default
            log_decisions: true,
        }
    }
}

/// Read Zerodha credentials without Python.
///
/// Reads `ZERODHA_API_KEY` from environment (loaded from `.env`)
/// and the access token from `~/.zerodha_access_token` (cached by auth script).
fn read_zerodha_credentials() -> Result<(String, String)> {
    // Load .env so ZERODHA_API_KEY is available
    dotenvy::dotenv().ok();

    let api_key = std::env::var("ZERODHA_API_KEY")
        .map_err(|_| anyhow::anyhow!("ZERODHA_API_KEY not set in environment or .env"))?;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ubuntu".to_string());
    let token_path = std::path::PathBuf::from(home).join(".zerodha_access_token");

    let access_token = std::fs::read_to_string(&token_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Cannot read {}: {}. Run scripts/zerodha_auth.py once to cache the token.",
                token_path.display(),
                e
            )
        })?
        .trim()
        .to_string();

    if access_token.is_empty() {
        anyhow::bail!("Access token file {} is empty", token_path.display());
    }

    Ok((api_key, access_token))
}

/// Paper trading entrypoint (venue-specific wiring).
///
/// This function:
/// 1. Reads Zerodha credentials from env + cached token file
/// 2. Discovers ATM options for the configured underlying
/// 3. Initializes the MarketFeedZerodha
/// 4. Creates the IndiaFnoFillModel
/// 5. Creates the CalendarCarryStrategy
/// 6. Runs the PaperEngine loop
pub async fn run(config: &IndiaPaperConfig) -> Result<()> {
    eprintln!(
        "[PAPER] Starting for {} with ₹{:.0} capital",
        config.underlying, config.initial_capital
    );

    // Step 1: Read Zerodha credentials (no Python needed)
    eprintln!("[PAPER] Step 1: Reading Zerodha credentials...");
    let (api_key, access_token) = read_zerodha_credentials()?;
    let discovery = ZerodhaAutoDiscovery::new(api_key.clone(), access_token.clone());
    eprintln!(
        "[PAPER] Auth OK (api_key={}..., token={}...)",
        &api_key[..4],
        &access_token[..8]
    );

    // Step 2: Discover ATM options
    eprintln!(
        "[PAPER] Step 2: Discovering ATM options for {}...",
        config.underlying
    );

    // For calendar carry, we need 2 expiries (front and back week)
    // Discover this week's options
    let disc_config = quantlaxmi_connectors_zerodha::AutoDiscoveryConfig {
        underlying: config.underlying.clone(),
        strikes_around_atm: config.strikes_around_atm,
        strike_interval: if config.underlying == "BANKNIFTY" {
            100.0
        } else {
            50.0
        },
        weekly_only: true,
    };

    let symbols_this_week = discovery.discover_symbols(&disc_config).await?;
    eprintln!(
        "[PAPER] Discovered {} symbols for this week",
        symbols_this_week.len()
    );

    // Also discover next week's options (for calendar spread back leg)
    let next_week_symbols = discovery
        .discover_next_week_symbols(&disc_config)
        .await
        .unwrap_or_else(|e| {
            eprintln!("[PAPER] Could not discover next week symbols: {}", e);
            vec![]
        });
    eprintln!(
        "[PAPER] Discovered {} symbols for next week",
        next_week_symbols.len()
    );

    // Merge both weeks
    let mut symbols = symbols_this_week;
    symbols.extend(next_week_symbols);
    eprintln!(
        "[PAPER] Total: {} option symbols across 2 expiries",
        symbols.len()
    );

    if symbols.is_empty() {
        anyhow::bail!("No options discovered for {}", config.underlying);
    }

    // Step 3: Fetch NFO instruments for metadata
    eprintln!("[PAPER] Step 3: Fetching NFO instruments...");
    let nfo_instruments = discovery.fetch_nfo_instruments().await?;
    eprintln!("[PAPER] Fetched {} NFO instruments", nfo_instruments.len());

    // Step 4: Build instrument map
    let instrument_map = InstrumentMap::from_tokens_and_instruments(&symbols, &nfo_instruments);
    eprintln!(
        "[PAPER] Step 4: Built instrument map with {} entries",
        instrument_map.len()
    );

    // Step 5: Create the feed (reuse credentials from Step 1)
    let feed = MarketFeedZerodha::new(
        api_key.clone(),
        access_token.clone(),
        instrument_map.clone(),
        config.feed_config.clone(),
    );
    eprintln!("[PAPER] Step 5: MarketFeedZerodha initialized");

    // Step 6: Create margin gate (SPAN margins via Zerodha API)
    let mut margin_gate = MarginGate::new(api_key.clone(), access_token.clone());
    // Initialize with starting capital (updated each step from ledger)
    margin_gate.set_available_cash(config.initial_capital);
    let margin_gate = Arc::new(Mutex::new(margin_gate));
    info!("[PAPER] MarginGate initialized (SPAN via Zerodha API, uses final_margin.total)");

    // Step 8: Create the strategy with margin gate and instrument map
    let strategy = CalendarCarryStrategyWrapper::new(config.strategy_config.clone())
        .with_margin_gate(margin_gate.clone())
        .with_instrument_map(instrument_map.clone());
    info!(
        "[PAPER] CalendarCarryStrategy initialized (entry_threshold={}p, exit_threshold={}p)",
        config.strategy_config.entry_threshold_paise, config.strategy_config.exit_threshold_paise
    );

    // Step 9: Create the fill model
    let fill_model = IndiaFnoFillModel::with_config(config.fee_config.clone());
    info!("[PAPER] IndiaFnoFillModel initialized (Budget 2026 STT rates)");

    // Step 10: Create the engine
    let engine_config = EngineConfig {
        initial_capital: config.initial_capital,
        log_decisions: config.log_decisions,
        state_tx: None,
    };

    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);
    eprintln!("[PAPER] PaperEngine initialized, entering loop...");

    // Step 11: Run the engine loop
    eprintln!(
        "[PAPER] Starting paper trading loop (duration={}s)...",
        config.duration_secs
    );

    let start_time = Utc::now();
    let duration_limit = if config.duration_secs > 0 {
        Some(chrono::Duration::seconds(config.duration_secs as i64))
    } else {
        None
    };

    let mut step_count = 0u64;
    loop {
        // Check duration limit
        if let Some(limit) = duration_limit {
            let elapsed = Utc::now().signed_duration_since(start_time);
            if elapsed >= limit {
                info!("[PAPER] Duration limit reached ({}s)", config.duration_secs);
                break;
            }
        }

        // Update margin gate's available_cash from engine state before step
        // This ensures margin checks use current cash, not stale initial capital
        {
            let mut gate = margin_gate.lock().await;
            gate.set_available_cash(engine.state().cash);
        }

        // Run one step
        if step_count == 0 {
            eprintln!("[PAPER] Waiting for first market event...");
        }
        match engine.step().await {
            Ok(()) => {
                step_count += 1;
                if step_count == 1 {
                    eprintln!("[PAPER] First event received! Engine is running.");
                }
                if step_count.is_multiple_of(100) {
                    eprintln!("[PAPER] Processed {} steps", step_count);
                }
            }
            Err(e) => {
                // Check if it's just end-of-stream
                let err_str = e.to_string();
                eprintln!("[PAPER] Engine step error: {}", err_str);
                if err_str.contains("disconnected") || err_str.contains("closed") {
                    eprintln!("[PAPER] Feed disconnected, stopping");
                    break;
                }
                return Err(e);
            }
        }
    }

    // Print session summary with truthful equity (not just cash)
    let state = engine.state();
    let ledger = engine.ledger();

    info!("[PAPER] ═══════════════════════════════════════════════════════");
    info!("[PAPER] SESSION SUMMARY");
    info!("[PAPER] ═══════════════════════════════════════════════════════");
    info!(
        "[PAPER] Duration: {}s",
        Utc::now().signed_duration_since(start_time).num_seconds()
    );
    info!("[PAPER] Initial Capital: ₹{:.2}", config.initial_capital);
    info!("[PAPER] ─────────────────────────────────────────────────────────");
    info!("[PAPER] Cash:           ₹{:.2}", state.cash);
    info!(
        "[PAPER] Unrealized PnL: ₹{:.2} (conservative MTM)",
        state.unrealized_pnl
    );
    info!("[PAPER] Realized PnL:   ₹{:.2}", state.realized_pnl);
    info!("[PAPER] ─────────────────────────────────────────────────────────");
    info!(
        "[PAPER] EQUITY:         ₹{:.2} (cash + unrealized)",
        state.equity
    );
    info!(
        "[PAPER] Total PnL:      ₹{:.2} (realized + unrealized)",
        state.total_pnl
    );
    info!("[PAPER] ─────────────────────────────────────────────────────────");
    info!("[PAPER] Total Fees:     ₹{:.2}", state.fees_paid);
    info!("[PAPER] Fills:          {}", ledger.fees.fill_count);
    info!("[PAPER] Rejections:     {}", ledger.fees.reject_count);
    info!("[PAPER] Open Positions: {}", state.open_positions);
    info!("[PAPER] ═══════════════════════════════════════════════════════");

    Ok(())
}

/// Run with default configuration and specified capital.
///
/// NOTE: Config file loading is not yet implemented.
/// This function uses IndiaPaperConfig defaults with the specified capital.
pub async fn run_with_defaults(initial_capital: f64) -> Result<()> {
    let config = IndiaPaperConfig {
        initial_capital,
        duration_secs: 120, // 2 minute default
        log_decisions: true,
        ..Default::default()
    };

    run(&config).await
}

/// Create a watch channel for state broadcasting to TUI.
pub fn make_state_channel() -> (watch::Sender<PaperState>, watch::Receiver<PaperState>) {
    watch::channel(PaperState::default())
}

/// Run paper trading with a state channel for TUI updates.
///
/// This variant broadcasts PaperState to the provided channel after each snapshot.
pub async fn run_with_channel(
    config: &IndiaPaperConfig,
    state_tx: watch::Sender<PaperState>,
) -> Result<()> {
    info!(
        "[PAPER] Starting for {} with ₹{:.0} capital (TUI mode)",
        config.underlying, config.initial_capital
    );

    // Step 1: Read Zerodha credentials (no Python needed)
    info!("[PAPER] Reading Zerodha credentials...");
    let (api_key, access_token) = read_zerodha_credentials()?;
    let discovery = ZerodhaAutoDiscovery::new(api_key.clone(), access_token.clone());
    info!("[PAPER] Auth successful (api_key={}...)", &api_key[..4]);

    // Step 2: Discover ATM options
    info!(
        "[INDIA-PAPER] Discovering ATM options for {}...",
        config.underlying
    );

    // For calendar carry, we need 2 expiries (front and back week)
    let disc_config = quantlaxmi_connectors_zerodha::AutoDiscoveryConfig {
        underlying: config.underlying.clone(),
        strikes_around_atm: config.strikes_around_atm,
        strike_interval: if config.underlying == "BANKNIFTY" {
            100.0
        } else {
            50.0
        },
        weekly_only: true,
    };

    let symbols_this_week = discovery.discover_symbols(&disc_config).await?;
    info!(
        "[PAPER] Discovered {} symbols for this week",
        symbols_this_week.len()
    );

    let next_week_symbols = discovery
        .discover_next_week_symbols(&disc_config)
        .await
        .unwrap_or_else(|e| {
            info!("[PAPER] Could not discover next week symbols: {}", e);
            vec![]
        });
    info!(
        "[PAPER] Discovered {} symbols for next week",
        next_week_symbols.len()
    );

    let mut symbols = symbols_this_week;
    symbols.extend(next_week_symbols);
    info!(
        "[PAPER] Total: {} option symbols across 2 expiries",
        symbols.len()
    );

    if symbols.is_empty() {
        anyhow::bail!("No options discovered for {}", config.underlying);
    }

    // Step 3: Fetch NFO instruments for metadata
    let nfo_instruments = discovery.fetch_nfo_instruments().await?;

    // Step 4: Build instrument map
    let instrument_map = InstrumentMap::from_tokens_and_instruments(&symbols, &nfo_instruments);
    info!(
        "[INDIA-PAPER] Built instrument map with {} entries",
        instrument_map.len()
    );

    // Step 5: Create the feed (reuse credentials from Step 1)
    let feed = MarketFeedZerodha::new(
        api_key.clone(),
        access_token.clone(),
        instrument_map.clone(),
        config.feed_config.clone(),
    );
    info!("[PAPER] MarketFeedZerodha initialized");

    // Step 6: Create margin gate
    let mut margin_gate = MarginGate::new(api_key.clone(), access_token.clone());
    margin_gate.set_available_cash(config.initial_capital);
    let margin_gate = Arc::new(Mutex::new(margin_gate));
    info!("[PAPER] MarginGate initialized");

    // Step 8: Create the strategy
    let strategy = CalendarCarryStrategyWrapper::new(config.strategy_config.clone())
        .with_margin_gate(margin_gate.clone())
        .with_instrument_map(instrument_map.clone());
    info!("[PAPER] CalendarCarryStrategy initialized");

    // Step 9: Create the fill model
    let fill_model = IndiaFnoFillModel::with_config(config.fee_config.clone());
    info!("[PAPER] IndiaFnoFillModel initialized");

    // Step 10: Create the engine with state channel
    let engine_config = EngineConfig {
        initial_capital: config.initial_capital,
        log_decisions: config.log_decisions,
        state_tx: Some(state_tx),
    };

    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);
    info!("[PAPER] PaperEngine initialized (TUI mode)");

    // Step 11: Run the engine loop
    let start_time = Utc::now();
    let duration_limit = if config.duration_secs > 0 {
        Some(chrono::Duration::seconds(config.duration_secs as i64))
    } else {
        None
    };

    let mut step_count = 0u64;
    loop {
        // Check duration limit
        if let Some(limit) = duration_limit {
            let elapsed = Utc::now().signed_duration_since(start_time);
            if elapsed >= limit {
                info!("[PAPER] Duration limit reached ({}s)", config.duration_secs);
                break;
            }
        }

        // Update margin gate's available_cash from engine state
        {
            let mut gate = margin_gate.lock().await;
            gate.set_available_cash(engine.state().cash);
        }

        // Run one step
        match engine.step().await {
            Ok(()) => {
                step_count += 1;
                if step_count.is_multiple_of(1000) {
                    info!("[PAPER] Processed {} steps", step_count);
                }
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("disconnected") || err_str.contains("closed") {
                    info!("[PAPER] Feed disconnected, stopping");
                    break;
                }
                tracing::error!(error = %e, "[INDIA-PAPER] Engine step error");
                return Err(e);
            }
        }
    }

    // Mark finished for TUI shutdown
    engine.mark_finished();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IndiaPaperConfig::default();
        assert_eq!(config.underlying, "NIFTY");
        assert_eq!(config.strikes_around_atm, 5);
        assert_eq!(config.initial_capital, 1_000_000.0);
    }
}
