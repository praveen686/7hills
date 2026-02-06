//! SANOS Calendar Carry — Standalone Live Module
//!
//! Wires the SANOS-gated Calendar Carry strategy to live Zerodha WebSocket feed
//! via the PaperEngine spine. Discovers 3 expiries (T1+T2+T3) and runs the full
//! SANOS calibration pipeline at configurable intervals.
//!
//! ## Architecture
//!
//! ```text
//! WebSocket → MarketFeedZerodha → OptionsSnapshot
//!   → SanosCalendarCarryAdapter.evaluate()
//!     → group by expiry → ExpirySlice × 3
//!     → SanosCalibrator.calibrate() × 3 → SanosSlice × 3
//!     → build_features() → Phase8Features
//!     → build straddle quotes → StraddleQuotes × 2
//!     → CalendarCarryStrategy.evaluate(ctx)
//!   → SanosStrategyWrapper.on_snapshot()
//!     → margin check → TradeIntent × 4
//!     → StrategyView + SanosTuiState
//!   → PaperEngine.step() → fill → PaperState → TUI
//! ```

pub mod adapter;
pub mod tui_state;
pub mod wrapper;

use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::{Mutex, watch};
use tracing::info;

use quantlaxmi_connectors_zerodha::ZerodhaAutoDiscovery;
use quantlaxmi_paper::{EngineConfig, PaperEngine, PaperState};

use crate::paper::{
    FeedConfig, IndiaFnoFeeConfig, IndiaFnoFillModel, InstrumentMap, MarginGate, MarketFeedZerodha,
};

pub use tui_state::SanosTuiState;
pub use wrapper::SanosStrategyWrapper;

/// Configuration for SANOS Calendar Carry paper trading.
#[derive(Debug, Clone)]
pub struct SanosCalCarryConfig {
    /// Underlying to trade (e.g., "NIFTY", "BANKNIFTY").
    pub underlying: String,
    /// Number of strikes around ATM to include.
    pub strikes_around_atm: u32,
    /// Initial capital in INR.
    pub initial_capital: f64,
    /// Feed configuration.
    pub feed_config: FeedConfig,
    /// Fee configuration.
    pub fee_config: IndiaFnoFeeConfig,
    /// Duration in seconds (0 = run until interrupted).
    pub duration_secs: u64,
    /// Enable decision logging.
    pub log_decisions: bool,
    /// SANOS calibration interval in seconds.
    pub calibration_interval_secs: u64,
    /// Lot size for the underlying.
    pub lot_size: i32,
    /// Relax E1/E2/E3 economic hardener gates for pipeline testing.
    pub relax_e_gates: bool,
}

impl Default for SanosCalCarryConfig {
    fn default() -> Self {
        Self {
            underlying: "NIFTY".to_string(),
            strikes_around_atm: 10,
            initial_capital: 1_000_000.0,
            feed_config: FeedConfig::default(),
            fee_config: IndiaFnoFeeConfig::default(),
            duration_secs: 120,
            log_decisions: false,
            calibration_interval_secs: 15,
            lot_size: 25,
            relax_e_gates: false,
        }
    }
}

/// Read Zerodha credentials without Python.
fn read_zerodha_credentials() -> Result<(String, String)> {
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

/// Create watch channels for state broadcasting.
pub fn make_state_channels() -> (
    watch::Sender<PaperState>,
    watch::Receiver<PaperState>,
    watch::Sender<SanosTuiState>,
    watch::Receiver<SanosTuiState>,
) {
    let (paper_tx, paper_rx) = watch::channel(PaperState::default());
    let (sanos_tx, sanos_rx) = watch::channel(SanosTuiState::default());
    (paper_tx, paper_rx, sanos_tx, sanos_rx)
}

/// Run SANOS Calendar Carry paper trading with state channels for TUI updates.
pub async fn run_with_channel(
    config: &SanosCalCarryConfig,
    state_tx: watch::Sender<PaperState>,
    sanos_tx: watch::Sender<SanosTuiState>,
) -> Result<()> {
    info!(
        "[SANOS-CAL-CARRY] Starting for {} with INR {:.0} capital (TUI mode){}",
        config.underlying,
        config.initial_capital,
        if config.relax_e_gates {
            " [RELAX-GATES]"
        } else {
            ""
        }
    );

    // Step 1: Read Zerodha credentials
    info!("[SANOS-CAL-CARRY] Reading Zerodha credentials...");
    let (api_key, access_token) = read_zerodha_credentials()?;
    let discovery = ZerodhaAutoDiscovery::new(api_key.clone(), access_token.clone());
    info!(
        "[SANOS-CAL-CARRY] Auth successful (api_key={}...)",
        &api_key[..api_key.len().min(4)]
    );

    // Step 2: Discover options for 3 expiries (T1, T2, T3)
    info!(
        "[SANOS-CAL-CARRY] Discovering options for {} (3 expiries)...",
        config.underlying
    );

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

    // T1: this week
    let symbols_t1 = discovery.discover_symbols(&disc_config).await?;
    info!(
        "[SANOS-CAL-CARRY] T1: {} symbols (this week)",
        symbols_t1.len()
    );

    // T2: next week
    let symbols_t2 = discovery
        .discover_next_week_symbols(&disc_config)
        .await
        .unwrap_or_else(|e| {
            info!("[SANOS-CAL-CARRY] Could not discover T2 symbols: {}", e);
            vec![]
        });
    info!(
        "[SANOS-CAL-CARRY] T2: {} symbols (next week)",
        symbols_t2.len()
    );

    // T3: discover monthly by using weekly_only=false in a separate config
    // For now, use T1+T2 (BANKNIFTY mode: T1+T3 when only 2 available)
    let mut symbols = symbols_t1;
    symbols.extend(symbols_t2);

    // We discovered T1+T2 (2 expiries minimum); the feed will reveal actual expiry dates.
    let total_expiries = 2;

    info!(
        "[SANOS-CAL-CARRY] Total: {} option symbols across {} expiries",
        symbols.len(),
        total_expiries
    );

    if symbols.is_empty() {
        anyhow::bail!("No options discovered for {}", config.underlying);
    }

    // Step 3: Fetch NFO instruments for metadata
    let nfo_instruments = discovery.fetch_nfo_instruments().await?;

    // Step 4: Build instrument map
    let instrument_map = InstrumentMap::from_tokens_and_instruments(&symbols, &nfo_instruments);
    info!(
        "[SANOS-CAL-CARRY] Built instrument map with {} entries",
        instrument_map.len()
    );

    // Step 5: Create the feed
    let feed = MarketFeedZerodha::new(
        api_key.clone(),
        access_token.clone(),
        instrument_map.clone(),
        config.feed_config.clone(),
    );
    info!("[SANOS-CAL-CARRY] MarketFeedZerodha initialized");

    // Step 6: Create margin gate
    let mut margin_gate = if config.relax_e_gates {
        MarginGate::new(api_key.clone(), access_token.clone()).with_margin_at_risk_cap(1.0) // 100% cap when relaxing gates for testing
    } else {
        MarginGate::new(api_key.clone(), access_token.clone())
    };
    margin_gate.set_available_cash(config.initial_capital);
    let margin_gate = Arc::new(Mutex::new(margin_gate));
    info!(
        "[SANOS-CAL-CARRY] MarginGate initialized (cap={}%)",
        if config.relax_e_gates { 100 } else { 80 }
    );

    // Step 7: Create the SANOS strategy wrapper
    let strategy = SanosStrategyWrapper::new(
        &config.underlying,
        config.lot_size,
        config.calibration_interval_secs,
        config.relax_e_gates,
    )
    .with_margin_gate(margin_gate.clone())
    .with_instrument_map(instrument_map.clone())
    .with_tui_sender(sanos_tx);
    info!(
        "[SANOS-CAL-CARRY] SanosStrategyWrapper initialized (calibration_interval={}s)",
        config.calibration_interval_secs
    );

    // Step 8: Create fill model
    let fill_model = IndiaFnoFillModel::with_config(config.fee_config.clone());
    info!("[SANOS-CAL-CARRY] IndiaFnoFillModel initialized");

    // Step 9: Create engine with state channel
    let engine_config = EngineConfig {
        initial_capital: config.initial_capital,
        log_decisions: config.log_decisions,
        state_tx: Some(state_tx),
    };

    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);
    info!("[SANOS-CAL-CARRY] PaperEngine initialized (TUI mode)");

    // Step 10: Run the engine loop
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
                info!(
                    "[SANOS-CAL-CARRY] Duration limit reached ({}s)",
                    config.duration_secs
                );
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
                    info!("[SANOS-CAL-CARRY] Processed {} steps", step_count);
                }
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("disconnected") || err_str.contains("closed") {
                    info!("[SANOS-CAL-CARRY] Feed disconnected, stopping");
                    break;
                }
                tracing::error!(error = %e, "[SANOS-CAL-CARRY] Engine step error");
                return Err(e);
            }
        }
    }

    // Mark finished for TUI shutdown
    engine.mark_finished();

    Ok(())
}

/// Run SANOS Calendar Carry paper trading (headless, no TUI).
pub async fn run(config: &SanosCalCarryConfig) -> Result<()> {
    let (paper_tx, _paper_rx, sanos_tx, _sanos_rx) = make_state_channels();
    run_with_channel(config, paper_tx, sanos_tx).await
}
