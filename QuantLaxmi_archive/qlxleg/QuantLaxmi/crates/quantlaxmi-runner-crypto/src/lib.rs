//! # QuantLaxmi Runner Crypto
//!
//! Crypto-specific trading runner for Binance (and future exchanges).
//!
//! ## Isolation Guarantee
//! This crate has NO dependency on Zerodha or any India-specific code:
//! - NO Zerodha imports
//! - NO KiteSim imports
//! - NO NSE specs imports
//! - NO options ledger imports
//!
//! ## Commands
//! - `capture-binance` - Capture bookTicker stream
//! - `capture-sbe-depth` - Capture SBE depth stream (certified)
//! - `exchange-info` - Fetch exchange info for symbols
//! - `paper` - Run paper trading
//! - `live` - Run live trading

pub mod backtest;
pub mod binance_capture;

// New paper trading mode using unified executor
pub mod deterministic_ids;
pub mod modes;
// Simple paper trading module with TelemetryBus (watch channel pattern)
pub mod paper;
// Phase 2 unified simulator (shared by backtest + paper trading)
pub mod binance_exchange_info;
pub mod binance_funding_capture;
pub mod binance_perp_capture;
pub mod binance_perp_execution;
pub mod binance_perp_session;
pub mod binance_sbe_depth_capture;
pub mod binance_trades_capture;
pub mod experiment;
pub mod features;
pub mod replay;
pub mod segment_manifest;
pub mod session_capture;
pub mod tournament;
pub mod ws_resilient;

// GPU batch scoring for tournament acceleration
pub mod gpu_batch;

// Two-pass tournament mode (wide scan + deep validation)
pub mod two_pass;

// Tournament sharding for distributed execution
pub mod shard;

use anyhow::Context;
use clap::{Parser, Subcommand};
use tracing::{error, info};

pub use quantlaxmi_runner_common::{
    AppState, RunnerConfig, SymbolState,
    artifact::{ArtifactBuilder, ArtifactFamily, FileHash, RunManifest, RunProfile},
    circuit_breakers::TradingCircuitBreakers,
    create_runtime, init_observability, report, tui,
    web_server::{ServerState, start_server},
};

/// Merge mode for shard merging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum MergeMode {
    /// Fail merge if any tasks are incomplete (default)
    #[default]
    Complete,
    /// Allow partial merges, write missing_tasks.txt
    Partial,
}

#[derive(Parser, Debug)]
#[command(name = "quantlaxmi-crypto")]
#[command(about = "QuantLaxmi Crypto - Cryptocurrency Trading via Binance")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)] // CLI enum parsed once, size difference acceptable
pub enum Commands {
    /// Capture Binance Spot bookTicker stream into QuoteEvent JSONL
    CaptureBinance {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/BINANCE/BTCUSDT/quotes.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,
    },

    /// Capture Binance SBE depth stream into DepthEvent JSONL (Phase-2A authoritative)
    CaptureSbeDepth {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_depth.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,

        /// Strict sequencing mode (default: true). Fails capture if sequence gaps detected.
        /// Gaps break deterministic replay, so certified captures must have strict=true.
        #[arg(long, default_value_t = true)]
        strict: bool,
    },

    /// Capture Binance SBE trades stream into TradeEvent JSONL (certified)
    CaptureTrades {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_trades.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,
    },

    /// Capture multi-symbol session (depth + trades) for arbitrage research
    CaptureSession {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,BNBUSDT
        #[arg(long)]
        symbols: String,

        /// Output directory for session data
        #[arg(long)]
        out_dir: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 7200)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,

        /// Include trades capture (in addition to depth)
        #[arg(long, default_value_t = true)]
        include_trades: bool,

        /// Strict sequencing mode (default: true). Fails capture if sequence gaps detected.
        #[arg(long, default_value_t = true)]
        strict: bool,
    },

    /// Capture Binance Futures perp bookTicker stream
    CapturePerpTicker {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_perp.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,
    },

    /// Capture Binance Futures perp depth stream (L2 order book)
    CapturePerpDepth {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_perp_depth.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,
    },

    /// Capture Binance Futures funding rate stream (markPrice)
    CaptureFunding {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_funding.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,
    },

    /// Capture combined perp session (Spot + Perp + Funding) for funding arbitrage
    CapturePerpSession {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT
        #[arg(long)]
        symbols: String,

        /// Output directory for session data
        #[arg(long, default_value = "data/perp_sessions")]
        out_dir: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 3600)]
        duration_secs: u64,

        /// Include spot capture for basis calculation
        #[arg(long, default_value_t = true)]
        include_spot: bool,

        /// Include perp depth (L2) instead of just bookTicker
        #[arg(long, default_value_t = false)]
        include_depth: bool,

        /// Price exponent for depth capture
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for depth capture
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,
    },

    /// Fetch exchange info for symbols
    ExchangeInfo {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT
        #[arg(long)]
        symbols: String,
    },

    /// Run paper trading mode with live capture + WAL + unified executor
    Paper {
        /// Symbol to trade, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output directory for WAL and session data
        #[arg(long, default_value = "data/paper")]
        out_dir: String,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,

        /// Run in headless mode (no TUI)
        #[arg(long, default_value_t = true)]
        headless: bool,
    },

    /// Run live trading mode
    Live {
        /// Path to configuration file
        #[arg(short, long, default_value = "configs/live.toml")]
        config: String,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
    },

    /// Retroactively finalize a crashed/incomplete segment
    ///
    /// For segments that were killed ungracefully (SIGHUP, power loss, etc.),
    /// this command computes digests from the raw data files and updates
    /// the segment manifest to FINALIZED_RETRO state.
    ///
    /// For segments without any manifest, use --create-bootstrap first
    /// to create a minimal manifest with default configuration.
    FinalizeSegment {
        /// Path to segment directory (e.g., data/perp_sessions/perp_20260125_120000)
        #[arg(long)]
        segment_dir: String,

        /// Force re-finalization even if already finalized
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Override stop reason (default: UNKNOWN)
        #[arg(long)]
        stop_reason: Option<String>,

        /// Create bootstrap manifest for orphaned segments that have no manifest
        #[arg(long, default_value_t = false)]
        create_bootstrap: bool,
    },

    /// Extract features from captured segment (Phase 2C.1)
    ///
    /// Computes FeatureSet v1 from quote streams and produces a deterministic
    /// features.jsonl file with full audit trail in run_manifest.json.
    ExtractFeatures {
        /// Path to segment directory (must have segment_manifest.json)
        #[arg(long)]
        segment_dir: String,

        /// Output directory for run (default: runs/)
        #[arg(long, default_value = "runs")]
        out_dir: String,

        /// Update rate window in milliseconds
        #[arg(long, default_value_t = 1000)]
        update_rate_window_ms: u64,

        /// Volatility EWMA decay factor (0-1)
        #[arg(long, default_value_t = 0.1)]
        vol_decay: f64,
    },

    /// Replay a captured segment (print stats or stream events)
    ///
    /// Reads all JSONL streams from a segment, merges by timestamp,
    /// and emits a unified event stream.
    ReplaySegment {
        /// Path to segment directory
        #[arg(long)]
        segment_dir: String,

        /// Output mode: "stats" (default) or "stream" (prints events)
        #[arg(long, default_value = "stats")]
        output: String,

        /// Limit number of events (0 = unlimited)
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },

    /// Run backtest on a captured segment
    ///
    /// Replays a segment through a strategy with paper execution.
    Backtest {
        /// Path to segment directory
        #[arg(long)]
        segment_dir: String,

        /// Strategy to run: "funding_bias" or "basis_capture"
        /// With --use-sdk, this selects from the Phase 2 Strategy Registry.
        #[arg(long, default_value = "basis_capture")]
        strategy: String,

        /// Path to strategy config TOML file (Phase 2 SDK only).
        /// If not provided, default config is used.
        #[arg(long)]
        strategy_config: Option<String>,

        /// Use Phase 2 Strategy SDK (recommended).
        /// When enabled, the strategy is loaded from the StrategyRegistry.
        #[arg(long, default_value_t = false)]
        use_sdk: bool,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,

        /// Fee in basis points (e.g., 10 = 0.1%)
        #[arg(long, default_value_t = 10.0)]
        fee_bps: f64,

        /// Position size for strategy (non-SDK strategies only)
        #[arg(long, default_value_t = 0.1)]
        position_size: f64,

        /// Basis threshold in bps (for basis_capture strategy, non-SDK only)
        #[arg(long, default_value_t = 5.0)]
        threshold_bps: f64,

        /// Pacing mode: "fast" (no delays, default) or "real" (real-time pacing)
        #[arg(long, default_value = "fast")]
        pace: String,

        /// Output results to JSON file
        #[arg(long)]
        output_json: Option<String>,

        /// Output decision trace to JSON file (for replay parity verification)
        #[arg(long)]
        output_trace: Option<String>,

        /// Run ID for correlation context (auto-generated if not provided)
        #[arg(long)]
        run_id: Option<String>,

        /// Phase 19D: Enforce admission decisions from WAL during replay.
        /// Instead of re-evaluating admission, follow WAL decisions exactly.
        /// Fails if WAL is missing entries or outcomes mismatch.
        #[arg(long, default_value_t = false)]
        enforce_admission_from_wal: bool,

        /// Policy when admission enforcement detects mismatch: "fail" (default) or "warn"
        #[arg(long, default_value = "fail")]
        admission_mismatch_policy: String,

        /// Write segment_admission_summary.json after backtest completion
        #[arg(long, default_value_t = true)]
        write_admission_summary: bool,

        // Phase 22E: Enforcement configuration
        /// Path to strategies_manifest.json (required with --require-promotion)
        #[arg(long)]
        strategies_manifest: Option<String>,

        /// Path to signals_manifest.json (required with --require-promotion)
        #[arg(long)]
        signals_manifest: Option<String>,

        /// Path to promotion directory root (required with --require-promotion)
        #[arg(long)]
        promotion_root: Option<String>,

        /// Require signal promotion status check before execution.
        /// When set, signals must be promoted to execute.
        #[arg(long, default_value_t = false)]
        require_promotion: bool,

        /// Override WAL output directory (default: segment_dir/wal)
        #[arg(long)]
        wal_dir: Option<String>,

        // Phase 25A/25B: Cost model and latency
        /// Path to cost model JSON file (Phase 25A)
        #[arg(long)]
        cost_model_path: Option<String>,

        /// Latency in ticks (Phase 25B). 0 = immediate execution.
        #[arg(long, default_value_t = 0)]
        latency_ticks: u32,

        /// Force close any open position at end of session (evaluation mode only).
        /// Converts MTM to realized PnL for proper profitability measurement.
        #[arg(long, default_value_t = false)]
        flatten_on_end: bool,

        /// Use spot prices for execution instead of perp prices.
        /// Useful when perp depth data is incomplete but spot quotes are available.
        #[arg(long, default_value_t = false)]
        use_spot_prices: bool,
    },

    /// Verify replay parity between two decision traces
    ///
    /// Compares two trace files and reports whether they match or diverge.
    VerifyTrace {
        /// Path to original trace file
        #[arg(long)]
        original: String,

        /// Path to replay trace file
        #[arg(long)]
        replay: String,
    },

    /// Run parameter grid tournament across multiple segments (P1)
    ///
    /// Expands a TOML parameter grid into configs, runs backtests across
    /// multiple segments, and produces results.jsonl, leaderboard.json,
    /// and promotion_candidates.json.
    Tournament {
        /// Root directory containing segment directories
        #[arg(long)]
        segments_root: String,

        /// Comma-separated list of glob patterns matched against segment dir names
        /// Example: "perp_2026*,smoke_*"
        #[arg(long)]
        segments: String,

        /// Strategy name (from registry)
        #[arg(long)]
        strategy: String,

        /// Path to grid TOML file
        #[arg(long)]
        grid: String,

        /// Output directory for tournament results
        #[arg(long)]
        out_dir: String,

        /// Initial capital for every run
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,

        /// If set, also emit P0 traces (fills/equity_curve) per run
        #[arg(long, default_value_t = false)]
        emit_traces: bool,

        /// Max number of runs (safety)
        #[arg(long, default_value_t = 5000)]
        max_runs: usize,

        /// Number of parallel workers (0 = auto-detect, uses half of CPU cores)
        #[arg(long, default_value_t = 0)]
        parallel: usize,

        // === Two-Pass Tournament Mode ===
        /// Enable two-pass mode: Pass 1 wide scan, Pass 2 deep validation
        #[arg(long, default_value_t = false)]
        two_pass: bool,

        /// Pass 1: fraction of segments to sample (0.0-1.0)
        #[arg(long, default_value_t = 0.30)]
        pass1_segment_fraction: f64,

        /// Pass 1: number of bins for stratified sampling
        #[arg(long, default_value_t = 10)]
        pass1_bins: usize,

        /// Number of top configs to promote from Pass 1 to Pass 2
        #[arg(long, default_value_t = 50)]
        select_top_k: usize,

        /// Refuse rate threshold - configs above this are excluded from Pass 2
        #[arg(long, default_value_t = 0.50)]
        refuse_threshold: f64,

        /// Run only Pass 2 using previously selected configs
        #[arg(long, default_value_t = false)]
        pass2_only: bool,

        /// Path to Pass 1 output dir (required for --pass2-only)
        #[arg(long)]
        pass1_dir: Option<String>,

        // === Sharding for Distributed Execution ===
        /// Shard index (0-indexed). Use with --shard-count for distributed execution.
        #[arg(long)]
        shard_index: Option<usize>,

        /// Total number of shards. Use with --shard-index for distributed execution.
        #[arg(long)]
        shard_count: Option<usize>,

        /// Merge shard results from multiple directories.
        /// Pass comma-separated list of shard output directories.
        #[arg(long)]
        merge_shards: Option<String>,

        /// Merge mode: 'complete' (default) fails if any tasks missing,
        /// 'partial' allows incomplete merges and writes missing_tasks.txt.
        #[arg(long, value_enum, default_value_t = MergeMode::Complete)]
        merge_mode: MergeMode,
    },
}

/// Main entry point for the Crypto runner
pub fn run() -> anyhow::Result<()> {
    let rt = create_runtime()?;
    rt.block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize observability - guard must be held for process lifetime
    let _tracing_guards = init_observability("quantlaxmi-crypto");

    match cli.command {
        Commands::CaptureBinance {
            symbol,
            out,
            duration_secs,
        } => run_capture_binance(&symbol, &out, duration_secs).await,
        Commands::CaptureSbeDepth {
            symbol,
            out,
            duration_secs,
            price_exponent,
            qty_exponent,
            strict,
        } => {
            run_capture_sbe_depth(
                &symbol,
                &out,
                duration_secs,
                price_exponent,
                qty_exponent,
                strict,
            )
            .await
        }
        Commands::CaptureTrades {
            symbol,
            out,
            duration_secs,
            price_exponent,
            qty_exponent,
        } => run_capture_trades(&symbol, &out, duration_secs, price_exponent, qty_exponent).await,
        Commands::CaptureSession {
            symbols,
            out_dir,
            duration_secs,
            price_exponent,
            qty_exponent,
            include_trades,
            strict,
        } => {
            run_capture_session(
                &symbols,
                &out_dir,
                duration_secs,
                price_exponent,
                qty_exponent,
                include_trades,
                strict,
            )
            .await
        }
        Commands::ExchangeInfo { symbols } => run_exchange_info(&symbols).await,
        Commands::Paper {
            symbol,
            out_dir,
            price_exponent,
            qty_exponent,
            initial_capital,
            headless,
        } => {
            run_paper_mode(
                &symbol,
                &out_dir,
                price_exponent,
                qty_exponent,
                initial_capital,
                headless,
            )
            .await
        }
        Commands::Live {
            config,
            initial_capital,
        } => run_live_mode(&config, initial_capital).await,
        Commands::CapturePerpTicker {
            symbol,
            out,
            duration_secs,
        } => run_capture_perp_ticker(&symbol, &out, duration_secs).await,
        Commands::CapturePerpDepth {
            symbol,
            out,
            duration_secs,
            price_exponent,
            qty_exponent,
        } => {
            run_capture_perp_depth(&symbol, &out, duration_secs, price_exponent, qty_exponent).await
        }
        Commands::CaptureFunding {
            symbol,
            out,
            duration_secs,
        } => run_capture_funding(&symbol, &out, duration_secs).await,
        Commands::CapturePerpSession {
            symbols,
            out_dir,
            duration_secs,
            include_spot,
            include_depth,
            price_exponent,
            qty_exponent,
        } => {
            run_capture_perp_session(
                &symbols,
                &out_dir,
                duration_secs,
                include_spot,
                include_depth,
                price_exponent,
                qty_exponent,
            )
            .await
        }
        Commands::FinalizeSegment {
            segment_dir,
            force,
            stop_reason,
            create_bootstrap,
        } => {
            run_finalize_segment(
                &segment_dir,
                force,
                stop_reason.as_deref(),
                create_bootstrap,
            )
            .await
        }
        Commands::ExtractFeatures {
            segment_dir,
            out_dir,
            update_rate_window_ms,
            vol_decay,
        } => run_extract_features(&segment_dir, &out_dir, update_rate_window_ms, vol_decay).await,
        Commands::ReplaySegment {
            segment_dir,
            output,
            limit,
        } => run_replay_segment(&segment_dir, &output, limit).await,
        Commands::Backtest {
            segment_dir,
            strategy,
            strategy_config,
            use_sdk,
            initial_capital,
            fee_bps,
            position_size,
            threshold_bps,
            pace,
            output_json,
            output_trace,
            run_id,
            enforce_admission_from_wal,
            admission_mismatch_policy,
            write_admission_summary,
            strategies_manifest,
            signals_manifest,
            promotion_root,
            require_promotion,
            wal_dir,
            cost_model_path,
            latency_ticks,
            flatten_on_end,
            use_spot_prices,
        } => {
            run_backtest(
                &segment_dir,
                &strategy,
                strategy_config.as_deref(),
                use_sdk,
                initial_capital,
                fee_bps,
                position_size,
                threshold_bps,
                &pace,
                output_json.as_deref(),
                output_trace.as_deref(),
                run_id.as_deref(),
                enforce_admission_from_wal,
                &admission_mismatch_policy,
                write_admission_summary,
                strategies_manifest.as_deref(),
                signals_manifest.as_deref(),
                promotion_root.as_deref(),
                require_promotion,
                wal_dir.as_deref(),
                cost_model_path.as_deref(),
                latency_ticks,
                flatten_on_end,
                use_spot_prices,
            )
            .await
        }
        Commands::VerifyTrace { original, replay } => run_verify_trace(&original, &replay).await,
        Commands::Tournament {
            segments_root,
            segments,
            strategy,
            grid,
            out_dir,
            initial_capital,
            emit_traces,
            max_runs,
            parallel,
            two_pass,
            pass1_segment_fraction,
            pass1_bins,
            select_top_k,
            refuse_threshold,
            pass2_only,
            pass1_dir,
            shard_index,
            shard_count,
            merge_shards,
            merge_mode,
        } => {
            // Handle merge mode separately
            if let Some(ref shard_dirs_str) = merge_shards {
                let shard_dirs: Vec<std::path::PathBuf> = shard_dirs_str
                    .split(',')
                    .map(|s| std::path::PathBuf::from(s.trim()))
                    .collect();
                let shard_dir_refs: Vec<&std::path::Path> =
                    shard_dirs.iter().map(|p| p.as_path()).collect();

                let two_pass_config = if two_pass {
                    Some(two_pass::TwoPassConfig {
                        pass1_segment_fraction,
                        pass1_bins,
                        select_top_k,
                        refuse_threshold,
                        max_per_family: None,
                    })
                } else {
                    None
                };

                let require_complete = merge_mode == MergeMode::Complete;
                return shard::merge_shards(
                    &shard_dir_refs,
                    std::path::Path::new(&out_dir),
                    &strategy,
                    two_pass_config.as_ref(),
                    require_complete,
                );
            }

            // Build shard config if sharding enabled
            let shard_config = match (shard_index, shard_count) {
                (Some(idx), Some(count)) => Some(shard::ShardConfig::new(idx, count)?),
                (None, None) => None,
                _ => {
                    anyhow::bail!(
                        "Both --shard-index and --shard-count must be specified together"
                    );
                }
            };

            let two_pass_config = if two_pass || pass2_only {
                Some(two_pass::TwoPassConfig {
                    pass1_segment_fraction,
                    pass1_bins,
                    select_top_k,
                    refuse_threshold,
                    max_per_family: None,
                })
            } else {
                None
            };

            tournament::run_tournament_grid_cli(
                &segments_root,
                &segments,
                &strategy,
                &grid,
                &out_dir,
                initial_capital,
                emit_traces,
                max_runs,
                parallel,
                two_pass_config,
                pass2_only,
                pass1_dir.as_deref(),
                shard_config,
            )
            .await
        }
    }
}

async fn run_capture_binance(symbol: &str, out: &str, duration_secs: u64) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    tracing::info!(
        "Capturing Binance {} bookTicker for {} seconds...",
        symbol,
        duration_secs
    );
    let stats = binance_capture::capture_book_ticker_jsonl(symbol, out_path, duration_secs).await?;
    tracing::info!("Capture complete: {} ({})", out, stats);

    // Emit RunManifest (Research profile - bookTicker is not the authoritative path)
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_bookticker_manifest(manifest_dir, out_path, symbol, &stats.stats)?;

    Ok(())
}

/// Emit RunManifest for bookTicker capture.
/// BookTicker uses Research profile (not Certified - SBE depth is authoritative).
fn emit_bookticker_manifest(
    manifest_dir: &std::path::Path,
    quotes_path: &std::path::Path,
    symbol: &str,
    stats: &binance_capture::CaptureStats,
) -> anyhow::Result<()> {
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, RunProfile::Research);

    // Record input hash (the captured quotes file)
    manifest.inputs.quotes = Some(FileHash::from_file(quotes_path)?);

    // NOT certified - bookTicker is JSON WebSocket, not SBE
    manifest.determinism.certified = false;

    // Compute input hash
    manifest.compute_input_hash();

    // Add capture metadata
    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "events_written": stats.events_written,
                "integrity_tier": "Uncertified",
                "source": "binance_bookticker_capture",
                "note": "Use capture-sbe-depth for certified replay"
            }),
        },
    );

    // Finish and write manifest
    manifest.finish();
    let manifest_path = manifest_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    tracing::info!("Manifest written: {:?}", manifest_path);
    tracing::info!("  Run ID: {}", manifest.run_id);
    tracing::info!(
        "  ⚠️  Profile: Research (not Certified - use capture-sbe-depth for deterministic replay)"
    );

    Ok(())
}

async fn run_capture_sbe_depth(
    symbol: &str,
    out: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
    strict: bool,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    tracing::info!(
        "Capturing Binance SBE {} depth stream for {} seconds (price_exp={}, qty_exp={}, strict={})...",
        symbol,
        duration_secs,
        price_exponent,
        qty_exponent,
        strict
    );

    let stats = binance_sbe_depth_capture::capture_sbe_depth_jsonl(
        symbol,
        out_path,
        duration_secs,
        price_exponent,
        qty_exponent,
        &api_key,
    )
    .await?;

    tracing::info!("Capture complete: {} ({})", out, stats);

    // In strict mode (default), fail if any sequence gaps detected
    // Gaps break deterministic replay, so certified captures must have no gaps
    if strict && stats.gaps_detected > 0 {
        // Clean up the output file since it's not usable for certified replay
        let _ = std::fs::remove_file(out_path);
        return Err(anyhow::anyhow!(
            "STRICT MODE FAILURE: {} sequence gaps detected. \
             Deterministic replay requires continuous sequencing. \
             Re-run capture or use --strict=false for research-only data.",
            stats.gaps_detected
        ));
    }

    // Always emit RunManifest for audit-grade reproducibility
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_capture_manifest(
        manifest_dir,
        out_path,
        symbol,
        &stats,
        price_exponent,
        qty_exponent,
        strict,
    )?;

    Ok(())
}

/// Emit RunManifest for a capture operation.
/// SBE depth captures with strict=true are Certified profile.
fn emit_capture_manifest(
    manifest_dir: &std::path::Path,
    depth_events_path: &std::path::Path,
    symbol: &str,
    stats: &binance_sbe_depth_capture::CaptureStats,
    price_exponent: i8,
    qty_exponent: i8,
    strict: bool,
) -> anyhow::Result<()> {
    // Profile depends on strict mode: Certified if strict, Research otherwise
    let profile = if strict {
        RunProfile::Certified
    } else {
        RunProfile::Research
    };
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, profile);

    // Record input hash (the captured depth events file)
    manifest.inputs.depth_events = Some(FileHash::from_file(depth_events_path)?);

    // Mark as certified only if strict mode (no gaps allowed)
    manifest.determinism.certified = strict;

    // Compute input hash for reproducibility tracking
    manifest.compute_input_hash();

    // Add capture metadata to diagnostics
    manifest.diagnostics.context_validation.validation_passed = stats.snapshot_written;
    manifest.diagnostics.context_validation.validation_errors = if stats.gaps_detected > 0 {
        vec![format!(
            "Sequence gaps detected: {} (replay may fail)",
            stats.gaps_detected
        )]
    } else {
        vec![]
    };

    // Add custom metadata as JSON in regime_transitions (repurposed for capture metadata)
    let integrity_tier = if strict { "Certified" } else { "Research" };
    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "events_written": stats.events_written,
                "snapshot_written": stats.snapshot_written,
                "gaps_detected": stats.gaps_detected,
                "price_exponent": price_exponent,
                "qty_exponent": qty_exponent,
                "strict_mode": strict,
                "integrity_tier": integrity_tier,
                "source": "binance_sbe_depth_capture"
            }),
        },
    );

    // Finish and write manifest
    manifest.finish();
    let manifest_path = manifest_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    tracing::info!("Manifest written: {:?}", manifest_path);
    tracing::info!("  Run ID: {}", manifest.run_id);
    tracing::info!("  Watermark: {}", manifest.watermark);
    if let Some(ref hash) = manifest.determinism.input_hash {
        tracing::info!("  Input hash: {}", &hash[..16]);
    }

    Ok(())
}

async fn run_capture_trades(
    symbol: &str,
    out: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    tracing::info!(
        "Capturing Binance SBE {} trades stream for {} seconds (price_exp={}, qty_exp={})...",
        symbol,
        duration_secs,
        price_exponent,
        qty_exponent
    );

    let stats = binance_trades_capture::capture_sbe_trades_jsonl(
        symbol,
        out_path,
        duration_secs,
        price_exponent,
        qty_exponent,
        &api_key,
    )
    .await?;

    tracing::info!("Capture complete: {} ({})", out, stats);

    // Emit manifest for trades capture
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_trades_manifest(
        manifest_dir,
        out_path,
        symbol,
        &stats,
        price_exponent,
        qty_exponent,
    )?;

    Ok(())
}

/// Emit RunManifest for trades capture.
fn emit_trades_manifest(
    manifest_dir: &std::path::Path,
    trades_path: &std::path::Path,
    symbol: &str,
    stats: &binance_trades_capture::TradesCaptureStats,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, RunProfile::Certified);

    // Record trades file hash (using quotes field as generic input)
    manifest.inputs.quotes = Some(FileHash::from_file(trades_path)?);

    manifest.determinism.certified = true;
    manifest.compute_input_hash();

    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "trades_written": stats.trades_written,
                "buy_count": stats.buy_count,
                "sell_count": stats.sell_count,
                "total_volume_mantissa": stats.total_volume_mantissa,
                "price_exponent": price_exponent,
                "qty_exponent": qty_exponent,
                "integrity_tier": "Certified",
                "source": "binance_sbe_trades_capture"
            }),
        },
    );

    manifest.finish();
    let manifest_path = manifest_dir.join("trades_manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    tracing::info!("Trades manifest written: {:?}", manifest_path);
    tracing::info!("  Run ID: {}", manifest.run_id);

    Ok(())
}

async fn run_capture_session(
    symbols: &str,
    out_dir: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
    include_trades: bool,
    strict: bool,
) -> anyhow::Result<()> {
    // Parse symbols
    let symbol_list: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BTCUSDT,ETHUSDT,BNBUSDT"
        ));
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    tracing::info!("=== Session Capture ===");
    tracing::info!("Symbols: {:?}", symbol_list);
    tracing::info!("Duration: {} seconds", duration_secs);
    tracing::info!("Include trades: {}", include_trades);
    tracing::info!("Strict mode: {}", strict);
    tracing::info!("Output: {}", out_dir);

    let config = session_capture::SessionCaptureConfig {
        symbols: symbol_list,
        out_dir: std::path::PathBuf::from(out_dir),
        duration_secs,
        price_exponent,
        qty_exponent,
        include_trades,
        strict,
        api_key,
        price_exponents: None, // Use default price_exponent for all symbols
    };

    let stats = session_capture::capture_session(config).await?;

    tracing::info!("\n=== Session Summary ===");
    tracing::info!("Session ID: {}", stats.session_id);
    tracing::info!("Duration: {:.1}s", stats.duration_secs);
    tracing::info!("Total depth events: {}", stats.total_depth_events);
    tracing::info!("Total trades: {}", stats.total_trades);
    tracing::info!("Total gaps: {}", stats.total_gaps);
    tracing::info!(
        "Certified: {}",
        if stats.all_symbols_clean { "YES" } else { "NO" }
    );

    Ok(())
}

async fn run_exchange_info(symbols: &str) -> anyhow::Result<()> {
    let symbol_list: std::collections::HashSet<String> = symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BTCUSDT,ETHUSDT"
        ));
    }

    tracing::info!(
        "Fetching exchange info for {} symbols...",
        symbol_list.len()
    );

    match binance_exchange_info::fetch_spot_specs(&symbol_list) {
        Ok(specs) => {
            tracing::info!("\n=== Binance Exchange Info ===");
            for (sym, (tick_size, qty_scale)) in specs {
                tracing::info!(
                    "  {}: tick_size={}, qty_scale={}",
                    sym,
                    tick_size,
                    qty_scale
                );
            }
        }
        Err(e) => {
            error!("Failed to fetch exchange info: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

async fn run_paper_mode(
    symbol: &str,
    out_dir: &str,
    price_exponent: i8,
    qty_exponent: i8,
    initial_capital: f64,
    headless: bool,
) -> anyhow::Result<()> {
    use futures_util::{SinkExt, StreamExt};
    use quantlaxmi_models::depth::{DepthEvent, DepthLevel, IntegrityTier};
    use quantlaxmi_sbe::{BinanceSbeDecoder, SBE_HEADER_SIZE, SbeHeader};
    use tokio::sync::mpsc;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message;
    use url::Url;

    info!("QuantLaxmi Crypto - Paper Trading Mode (unified executor)");
    info!("Symbol: {}", symbol);
    info!("Initial capital: ${:.2}", initial_capital);

    if headless {
        tui::print_headless_banner("QuantLaxmi Crypto Paper Trading", initial_capital);
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    // Create depth event channel
    let (depth_tx, depth_rx) = mpsc::channel::<DepthEvent>(1024);

    // Configure paper mode
    let paper_cfg = modes::paper::PaperModeConfig {
        base_dir: std::path::PathBuf::from(out_dir),
        symbol: symbol.to_uppercase(),
        initial_capital,
        fee_bps_maker: 2.0,
        fee_bps_taker: 10.0,
        http_bind: "127.0.0.1:8080".to_string(),
    };

    // Spawn the paper trading engine (uses modes::paper::run_paper_mode)
    let paper_handle = tokio::spawn(modes::paper::run_paper_mode(paper_cfg, depth_rx));

    // Connect to SBE stream
    let sym_lower = symbol.to_lowercase();
    let url_str = "wss://stream-sbe.binance.com:9443/stream";
    info!("Connecting to Binance SBE stream: {}", url_str);

    let url = Url::parse(url_str)?;
    let mut request = url.into_client_request()?;
    request
        .headers_mut()
        .insert("X-MBX-APIKEY", api_key.parse()?);
    request
        .headers_mut()
        .insert("Sec-WebSocket-Protocol", "binance-sbe".parse()?);

    let (ws_stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .context("connect SBE websocket")?;

    info!("Connected to SBE stream");

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to depth stream
    let sub = serde_json::json!({
        "method": "SUBSCRIBE",
        "params": [format!("{}@depth", sym_lower)],
        "id": 1
    });
    write.send(Message::Text(sub.to_string())).await?;

    // Fetch initial snapshot
    info!("Fetching depth snapshot...");
    let snapshot_url = format!(
        "https://api.binance.com/api/v3/depth?symbol={}&limit=1000",
        symbol.to_uppercase()
    );
    let snapshot_resp: serde_json::Value = reqwest::get(&snapshot_url).await?.json().await?;
    let snapshot_last_id = snapshot_resp["lastUpdateId"].as_u64().unwrap_or(0);
    info!("Snapshot lastUpdateId: {}", snapshot_last_id);

    // Create snapshot event
    let snapshot_bids: Vec<DepthLevel> = snapshot_resp["bids"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|b| {
            let price_str = b[0].as_str()?;
            let qty_str = b[1].as_str()?;
            let price =
                binance_sbe_depth_capture::parse_to_mantissa_from_str(price_str, price_exponent)
                    .ok()?;
            let qty = binance_sbe_depth_capture::parse_to_mantissa_from_str(qty_str, qty_exponent)
                .ok()?;
            Some(DepthLevel { price, qty })
        })
        .collect();

    let snapshot_asks: Vec<DepthLevel> = snapshot_resp["asks"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|a| {
            let price_str = a[0].as_str()?;
            let qty_str = a[1].as_str()?;
            let price =
                binance_sbe_depth_capture::parse_to_mantissa_from_str(price_str, price_exponent)
                    .ok()?;
            let qty = binance_sbe_depth_capture::parse_to_mantissa_from_str(qty_str, qty_exponent)
                .ok()?;
            Some(DepthLevel { price, qty })
        })
        .collect();

    let snapshot_event = DepthEvent {
        ts: chrono::Utc::now(),
        tradingsymbol: symbol.to_uppercase(),
        first_update_id: snapshot_last_id,
        last_update_id: snapshot_last_id,
        price_exponent,
        qty_exponent,
        bids: snapshot_bids,
        asks: snapshot_asks,
        is_snapshot: true,
        integrity_tier: IntegrityTier::Certified,
        source: Some("sbe_stream".to_string()),
    };

    // Send snapshot to paper engine
    depth_tx.send(snapshot_event).await?;
    info!("Snapshot sent to paper engine");

    info!("Paper trading active. Press Ctrl+C to stop.");
    info!("HTTP order endpoint: POST http://127.0.0.1:8080/order");

    // Process WebSocket messages and forward to paper engine
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received");
                break;
            }
            msg = tokio::time::timeout(std::time::Duration::from_secs(30), read.next()) => {
                match msg {
                    Ok(Some(Ok(Message::Binary(bin)))) => {
                        if bin.len() >= SBE_HEADER_SIZE
                            && let Ok(header) = SbeHeader::decode(&bin[..SBE_HEADER_SIZE])
                            && header.template_id == 10003
                            && let Ok(update) = BinanceSbeDecoder::decode_depth_update(
                                &header,
                                &bin[SBE_HEADER_SIZE..],
                            )
                        {
                            let event = binance_sbe_depth_capture::sbe_depth_to_event_pub(
                                &update,
                                symbol,
                                price_exponent,
                                qty_exponent,
                            );
                            if depth_tx.send(event).await.is_err() {
                                error!("Paper engine channel closed");
                                break;
                            }
                        }
                    }
                    Ok(Some(Ok(Message::Ping(p)))) => {
                        let _ = write.send(Message::Pong(p)).await;
                    }
                    Ok(Some(Ok(_))) => {}
                    Ok(Some(Err(e))) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    Ok(None) => {
                        info!("WebSocket closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue waiting
                    }
                }
            }
        }
    }

    // Drop sender to signal shutdown
    drop(depth_tx);

    // Wait for paper engine to finish
    match paper_handle.await {
        Ok(Ok(())) => info!("Paper engine completed successfully"),
        Ok(Err(e)) => error!("Paper engine error: {}", e),
        Err(e) => error!("Paper engine task error: {}", e),
    }

    Ok(())
}

async fn run_live_mode(config_path: &str, initial_capital: f64) -> anyhow::Result<()> {
    info!("QuantLaxmi Crypto - LIVE Trading Mode");
    info!("Loading configuration from: {}", config_path);

    let config = RunnerConfig::load(config_path)?;

    info!("Live trading mode initialized");
    info!("Symbols: {:?}", config.mode.symbols);
    info!("Initial capital: ${:.2}", initial_capital);
    info!("⚠️  LIVE MODE - Real orders will be placed!");
    info!("Press Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;
    info!("Shutting down...");

    Ok(())
}

// ============================================================================
// Perp Capture Functions (Phase 1: Crypto Calendar-Carry)
// ============================================================================

async fn run_capture_perp_ticker(
    symbol: &str,
    out: &str,
    duration_secs: u64,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    tracing::info!(
        "Capturing Binance Futures {} bookTicker for {} seconds...",
        symbol,
        duration_secs
    );

    let stats =
        binance_perp_capture::capture_perp_bookticker_jsonl(symbol, out_path, duration_secs)
            .await?;

    tracing::info!("Capture complete: {} ({})", out, stats);
    Ok(())
}

async fn run_capture_perp_depth(
    symbol: &str,
    out: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    tracing::info!(
        "Capturing Binance Futures {} depth for {} seconds (price_exp={}, qty_exp={})...",
        symbol,
        duration_secs,
        price_exponent,
        qty_exponent
    );

    let stats = binance_perp_capture::capture_perp_depth_jsonl(
        symbol,
        out_path,
        duration_secs,
        price_exponent,
        qty_exponent,
    )
    .await?;

    tracing::info!("Capture complete: {} ({})", out, stats);

    if stats.stats.sequence_gaps > 0 {
        tracing::info!(
            "⚠️  Warning: {} sequence gaps detected. Replay may have issues.",
            stats.stats.sequence_gaps
        );
    }

    if stats.total_reconnects > 0 {
        tracing::info!(
            "ℹ️  {} reconnections occurred, {} gaps recorded",
            stats.total_reconnects,
            stats.connection_gaps.len()
        );
    }

    Ok(())
}

async fn run_capture_funding(symbol: &str, out: &str, duration_secs: u64) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    tracing::info!(
        "Capturing Binance Futures {} funding rate for {} seconds...",
        symbol,
        duration_secs
    );

    let stats =
        binance_funding_capture::capture_funding_jsonl(symbol, out_path, duration_secs).await?;

    tracing::info!("Capture complete: {} ({})", out, stats);
    Ok(())
}

async fn run_capture_perp_session(
    symbols: &str,
    out_dir: &str,
    duration_secs: u64,
    include_spot: bool,
    include_depth: bool,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    use segment_manifest::{CaptureConfig, EventCounts, ManagedSegment, StopReason};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, Ordering};

    // Parse symbols
    let symbol_list: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BTCUSDT,ETHUSDT"
        ));
    }

    let out_path = std::path::Path::new(out_dir);
    std::fs::create_dir_all(out_path)?;

    // Build capture config for manifest
    let capture_config = CaptureConfig {
        include_spot,
        include_depth,
        price_exponent: price_exponent as i32,
        qty_exponent: qty_exponent as i32,
    };

    // Start managed segment with manifest
    let mut managed = ManagedSegment::start(
        out_path,
        &symbol_list,
        "capture-perp-session",
        capture_config,
    )
    .await?;
    let segment_dir = managed.segment_dir().to_path_buf();
    let segment_id = managed.segment_id().await;

    tracing::info!("=== Perp Session Capture ===");
    tracing::info!("Segment ID: {}", segment_id);
    tracing::info!("Symbols: {:?}", symbol_list);
    tracing::info!("Duration: {} seconds", duration_secs);
    tracing::info!("Include spot: {}", include_spot);
    tracing::info!("Include depth: {}", include_depth);
    tracing::info!("Output: {:?}", segment_dir);

    // Set up signal handling
    // 0 = running, 1 = SIGINT, 2 = SIGTERM, 3 = SIGHUP
    let signal_received = Arc::new(AtomicU8::new(0));
    let signal_for_int = Arc::clone(&signal_received);
    let signal_for_term = Arc::clone(&signal_received);
    let signal_for_hup = Arc::clone(&signal_received);

    // SIGINT handler (Ctrl+C)
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_for_int.store(1, Ordering::SeqCst);
        }
    });

    // SIGTERM handler
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate())?;
        tokio::spawn(async move {
            sigterm.recv().await;
            signal_for_term.store(2, Ordering::SeqCst);
        });

        let mut sighup = signal(SignalKind::hangup())?;
        tokio::spawn(async move {
            sighup.recv().await;
            signal_for_hup.store(3, Ordering::SeqCst);
        });
    }

    // Run capture with signal monitoring
    // Capture runs directly in the managed segment directory
    // Load API key from env (optional - will fall back to public WS if not available)
    let api_key = std::env::var("BINANCE_API_KEY_ED25519").ok();
    if api_key.is_some() {
        tracing::info!("BINANCE_API_KEY_ED25519 found, will use SBE stream for trades");
    } else {
        tracing::info!("No API key found, will use public aggTrades WS stream");
    }

    let capture_config = binance_perp_session::PerpSessionConfig {
        symbols: symbol_list.clone(),
        out_dir: segment_dir.clone(), // Not used by capture_to_segment, but kept for config
        duration_secs,
        include_spot,
        include_depth,
        include_trades: true, // Always capture trades for toxicity calibration
        price_exponent,
        qty_exponent,
        api_key,
    };

    // Run capture with signal checking
    let signal_check = Arc::clone(&signal_received);
    let capture_result = tokio::select! {
        result = binance_perp_session::capture_to_segment(&segment_dir, &capture_config) => {
            result
        }
        _ = async {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if signal_check.load(Ordering::SeqCst) != 0 {
                    break;
                }
            }
        } => {
            // Signal received - will handle below
            Err(anyhow::anyhow!("Signal received"))
        }
    };

    // Determine stop reason and collect stats
    let signal_val = signal_received.load(Ordering::SeqCst);
    let (stop_reason, stats) = match (signal_val, &capture_result) {
        (0, Ok(stats)) => (StopReason::NormalCompletion, Some(stats.clone())),
        (1, _) => {
            tracing::info!("SIGINT received - finalizing segment...");
            (StopReason::UserInterrupt, None)
        }
        (2, _) => {
            tracing::info!("SIGTERM received - finalizing segment...");
            (StopReason::ExternalKillSigterm, None)
        }
        (3, _) => {
            tracing::info!("SIGHUP received - finalizing segment...");
            (StopReason::ExternalKillSighup, None)
        }
        (_, Err(e)) => {
            tracing::error!("Capture error: {}", e);
            (StopReason::NetworkError, None)
        }
        (_, Ok(stats)) => {
            // Unexpected signal value but capture succeeded
            tracing::warn!("Unexpected signal value: {}", signal_val);
            (StopReason::NormalCompletion, Some(stats.clone()))
        }
    };

    // Collect event counts
    let events = if let Some(ref s) = stats {
        EventCounts {
            spot_quotes: s.total_spot_events,
            perp_quotes: s.total_perp_events,
            funding: s.total_funding_events,
            depth: 0,
        }
    } else {
        // Count events from files if capture was interrupted
        let mut events = EventCounts::default();
        for symbol in &symbol_list {
            let sym_dir = segment_dir.join(symbol.to_uppercase());
            if let Ok(content) = std::fs::read_to_string(sym_dir.join("spot_quotes.jsonl")) {
                events.spot_quotes += content.lines().count();
            }
            if let Ok(content) = std::fs::read_to_string(sym_dir.join("perp_quotes.jsonl")) {
                events.perp_quotes += content.lines().count();
            }
            if let Ok(content) = std::fs::read_to_string(sym_dir.join("funding.jsonl")) {
                events.funding += content.lines().count();
            }
        }
        events
    };

    // Finalize segment manifest
    managed.finalize(stop_reason, events.clone()).await?;

    tracing::info!("\n=== Session Summary ===");
    tracing::info!("Segment ID: {}", segment_id);
    tracing::info!("Stop reason: {}", stop_reason);
    tracing::info!("Total spot events: {}", events.spot_quotes);
    tracing::info!("Total perp events: {}", events.perp_quotes);
    tracing::info!("Total funding events: {}", events.funding);

    if let Some(stats) = stats {
        tracing::info!("Duration: {:.1}s", stats.duration_secs);
        for sym_stat in &stats.symbols {
            tracing::info!(
                "  {}: basis={:.2}bps, funding={:.4}%",
                sym_stat.symbol,
                sym_stat.basis_bps,
                sym_stat.last_funding_rate * 100.0
            );
        }
    }

    Ok(())
}

/// Retroactively finalize a crashed/incomplete segment.
///
/// This command is used to create valid manifests for segments that were
/// killed ungracefully (SIGHUP, power loss, etc.) and only have raw data files.
async fn run_finalize_segment(
    segment_dir: &str,
    force: bool,
    stop_reason_override: Option<&str>,
    create_bootstrap: bool,
) -> anyhow::Result<()> {
    use segment_manifest::{
        CaptureConfig, EventCounts, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest, StopReason,
        compute_binary_hash, compute_segment_digests,
    };

    let segment_path = std::path::Path::new(segment_dir);
    if !segment_path.exists() {
        return Err(anyhow::anyhow!(
            "Segment directory not found: {}",
            segment_dir
        ));
    }

    // Check for existing manifest
    let manifest_path = segment_path.join("segment_manifest.json");

    // Create bootstrap manifest for legacy segments if requested
    if create_bootstrap && !manifest_path.exists() {
        tracing::info!("Creating bootstrap manifest for legacy segment...");

        // Infer segment ID from directory name
        let segment_id = segment_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Detect symbols from subdirectories
        let mut symbols: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(segment_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.chars().all(|c| c.is_alphanumeric()) && !name.is_empty() {
                    symbols.push(name.to_string());
                }
            }
        }

        if symbols.is_empty() {
            return Err(anyhow::anyhow!(
                "No symbol directories found in segment. Cannot create bootstrap manifest."
            ));
        }

        // Detect if depth was captured
        let first_sym = &symbols[0];
        let include_depth = segment_path
            .join(first_sym)
            .join("perp_depth.jsonl")
            .exists();
        let include_spot = segment_path
            .join(first_sym)
            .join("spot_quotes.jsonl")
            .exists();

        // Create session family ID
        let date_part = segment_id
            .split('_')
            .nth(1)
            .unwrap_or("unknown")
            .chars()
            .take(8)
            .collect::<String>();
        let session_family_id = format!("perp_{}_{}", symbols[0], date_part);

        let binary_hash = compute_binary_hash().unwrap_or_else(|_| "LEGACY".to_string());

        let manifest = SegmentManifest::new(
            session_family_id,
            segment_id.clone(),
            symbols.clone(),
            "capture-perp-session".to_string(),
            binary_hash,
            CaptureConfig {
                include_spot,
                include_depth,
                price_exponent: -2,
                qty_exponent: -8,
            },
        );

        manifest.write(segment_path)?;
        tracing::info!(
            "Bootstrap manifest created for legacy segment: {} (symbols: {:?})",
            segment_id,
            symbols
        );
    }

    if manifest_path.exists() {
        let manifest = SegmentManifest::load(segment_path)?;

        if manifest.is_finalized() && !force {
            tracing::info!(
                "Segment already finalized (state: {:?}). Use --force to re-finalize.",
                manifest.state
            );
            return Ok(());
        }

        // Validate schema version
        if manifest.schema_version != SEGMENT_MANIFEST_SCHEMA_VERSION {
            return Err(anyhow::anyhow!(
                "Schema version mismatch: manifest has v{}, expected v{}. Cannot retro-finalize.",
                manifest.schema_version,
                SEGMENT_MANIFEST_SCHEMA_VERSION
            ));
        }

        tracing::info!("Retro-finalizing existing manifest...");

        // Compute digests
        let digests = compute_segment_digests(segment_path)?;

        // Count events from digests
        let events = EventCounts {
            spot_quotes: digests.spot.as_ref().map_or(0, |d| d.event_count),
            perp_quotes: digests.perp.as_ref().map_or(0, |d| d.event_count),
            funding: digests.funding.as_ref().map_or(0, |d| d.event_count),
            depth: digests.depth.as_ref().map_or(0, |d| d.event_count),
        };

        // Update manifest
        let mut manifest = manifest;

        // Apply stop reason override if provided
        if let Some(reason_str) = stop_reason_override {
            manifest.stop_reason = match reason_str.to_uppercase().as_str() {
                "NORMAL_COMPLETION" => StopReason::NormalCompletion,
                "USER_INTERRUPT" => StopReason::UserInterrupt,
                "EXTERNAL_KILL_SIGTERM" => StopReason::ExternalKillSigterm,
                "EXTERNAL_KILL_SIGHUP" => StopReason::ExternalKillSighup,
                "PANIC" => StopReason::Panic,
                "NETWORK_ERROR" => StopReason::NetworkError,
                _ => StopReason::Unknown,
            };
        }

        manifest.retro_finalize(events.clone(), digests);
        manifest.write(segment_path)?;

        tracing::info!("=== Retro-finalization Complete ===");
        tracing::info!("Segment: {}", manifest.segment_id);
        tracing::info!("State: {:?}", manifest.state);
        tracing::info!(
            "Events: spot={}, perp={}, funding={}, depth={}",
            events.spot_quotes,
            events.perp_quotes,
            events.funding,
            events.depth
        );
        if let Some(ref digests) = manifest.digests
            && let Some(ref perp) = digests.perp
        {
            tracing::info!("Perp digest: {}...", &perp.sha256[..16]);
        }
    } else {
        // No manifest exists - this is an orphaned segment
        // We can't retro-finalize without knowing the capture configuration
        return Err(anyhow::anyhow!(
            "No segment_manifest.json found in {}. Cannot retro-finalize without bootstrap manifest.",
            segment_dir
        ));
    }

    Ok(())
}

/// Extract features from a captured segment (Phase 2C.1)
async fn run_extract_features(
    segment_dir: &str,
    out_dir: &str,
    update_rate_window_ms: u64,
    vol_decay: f64,
) -> anyhow::Result<()> {
    use experiment::{RunManifest, SegmentInput, generate_run_dir};
    use features::{FeatureConfig, extract_features};
    use segment_manifest::SegmentManifest;

    let segment_path = std::path::Path::new(segment_dir);
    if !segment_path.exists() {
        return Err(anyhow::anyhow!(
            "Segment directory not found: {}",
            segment_dir
        ));
    }

    // Load segment manifest
    let manifest = SegmentManifest::load(segment_path)?;
    tracing::info!("Loaded segment: {}", manifest.segment_id);
    tracing::info!("State: {:?}", manifest.state);

    // Warn if not finalized
    if !manifest.is_finalized() {
        tracing::warn!(
            "Segment is not finalized (state: {:?}). Consider running finalize-segment first.",
            manifest.state
        );
    }

    // Create segment input reference
    let segment_input = SegmentInput::from_manifest(&manifest, segment_path);

    // Build feature config
    let feature_config = FeatureConfig {
        update_rate_window_ms,
        vol_ewma_decay: vol_decay,
        min_spread: 0,
        max_spread: 0,
    };

    // Generate run directory
    let base_out = std::path::Path::new(out_dir);
    let run_dir = generate_run_dir(base_out, std::slice::from_ref(&segment_input));
    std::fs::create_dir_all(&run_dir)?;

    tracing::info!("Run directory: {:?}", run_dir);

    // Create run manifest
    let mut run_manifest = RunManifest::new(&run_dir, vec![segment_input], feature_config.clone());
    run_manifest.write(&run_dir)?;

    // Find input files - try per-symbol structure first, then flat structure
    let perp_path = {
        // Try flat structure: segment_dir/perp.jsonl
        let flat = segment_path.join("perp.jsonl");
        if flat.exists() {
            flat
        } else {
            // Try per-symbol structure: segment_dir/BTCUSDT/perp_quotes.jsonl
            let symbol = manifest
                .symbols
                .first()
                .map(|s| s.as_str())
                .unwrap_or("BTCUSDT");
            let per_symbol = segment_path.join(symbol).join("perp_quotes.jsonl");
            if per_symbol.exists() {
                per_symbol
            } else {
                return Err(anyhow::anyhow!(
                    "No perp quote file found. Tried:\n  {:?}\n  {:?}",
                    flat,
                    per_symbol
                ));
            }
        }
    };

    tracing::info!("Input: {:?}", perp_path);

    // Extract features
    let output_path = run_dir.join("features.jsonl");
    let result = extract_features(&perp_path, &output_path, &feature_config)?;

    tracing::info!("=== Feature Extraction Complete ===");
    tracing::info!("Input events: {}", result.input_events);
    tracing::info!("Output events: {}", result.output_events);
    tracing::info!("Output digest: {}...", &result.output_digest[..16]);

    // Update run manifest
    run_manifest.add_feature_result(&result);
    run_manifest.complete();
    run_manifest.write(&run_dir)?;

    tracing::info!("Run manifest: {:?}", run_dir.join("run_manifest.json"));

    // Print summary
    println!("\n=== Run Summary ===");
    println!("Run ID: {}", run_manifest.run_id);
    println!("Input: {} ({} events)", segment_dir, result.input_events);
    println!(
        "Output: {:?} ({} events)",
        output_path, result.output_events
    );
    println!("Output digest: {}", result.output_digest);
    println!("Run manifest: {:?}", run_dir.join("run_manifest.json"));

    Ok(())
}

/// Replay a captured segment (stats or stream mode).
async fn run_replay_segment(segment_dir: &str, output: &str, limit: usize) -> anyhow::Result<()> {
    use replay::{ReplayStats, SegmentReplayAdapter};

    let segment_path = std::path::Path::new(segment_dir);
    if !segment_path.exists() {
        return Err(anyhow::anyhow!(
            "Segment directory not found: {}",
            segment_dir
        ));
    }

    match output {
        "stats" => {
            tracing::info!("Computing replay stats for {:?}...", segment_path);
            let mut adapter = SegmentReplayAdapter::open(segment_path)?;
            let stats = ReplayStats::from_adapter(&mut adapter)?;

            println!("\n=== Segment Replay Stats ===");
            println!("Path: {}", segment_dir);
            println!("Symbols: {:?}", stats.symbols);
            println!("Total events: {}", stats.total_events);
            println!("  Spot quotes: {}", stats.spot_events);
            println!("  Perp quotes: {}", stats.perp_events);
            println!("  Funding: {}", stats.funding_events);
            if let Some(first) = stats.first_ts {
                println!("First event: {}", first);
            }
            if let Some(last) = stats.last_ts {
                println!("Last event: {}", last);
            }
            println!(
                "Duration: {:.1}s ({:.1}h)",
                stats.duration_secs,
                stats.duration_secs / 3600.0
            );
        }
        "stream" => {
            let mut adapter = SegmentReplayAdapter::open(segment_path)?;
            let mut count = 0usize;

            while let Some(event) = adapter.next_event()? {
                println!(
                    "{} {} {:?} {}",
                    event.ts.format("%H:%M:%S%.3f"),
                    event.symbol,
                    event.kind,
                    serde_json::to_string(&event.payload)?
                );

                count += 1;
                if limit > 0 && count >= limit {
                    tracing::info!("Reached limit of {} events", limit);
                    break;
                }
            }

            tracing::info!("Streamed {} events", count);
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown output mode: {}. Use 'stats' or 'stream'",
                output
            ));
        }
    }

    Ok(())
}

/// Run backtest on a captured segment.
#[allow(clippy::too_many_arguments)]
async fn run_backtest(
    segment_dir: &str,
    strategy_name: &str,
    strategy_config: Option<&str>,
    use_sdk: bool,
    initial_capital: f64,
    fee_bps: f64,
    position_size: f64,
    threshold_bps: f64,
    pace: &str,
    output_json: Option<&str>,
    output_trace: Option<&str>,
    run_id: Option<&str>,
    enforce_admission_from_wal: bool,
    admission_mismatch_policy: &str,
    write_admission_summary: bool,
    // Phase 22E: Enforcement configuration
    strategies_manifest: Option<&str>,
    signals_manifest: Option<&str>,
    promotion_root: Option<&str>,
    require_promotion: bool,
    wal_dir: Option<&str>,
    // Phase 25A/25B: Cost model and latency
    cost_model_path: Option<&str>,
    latency_ticks: u32,
    // Evaluation-only: flatten position at end of session
    flatten_on_end: bool,
    use_spot_prices: bool,
) -> anyhow::Result<()> {
    use backtest::{
        BacktestConfig, BacktestEngine, BasisCaptureStrategy, EnforcementConfig, ExchangeConfig,
        FundingBiasStrategy, PaceMode,
    };
    use quantlaxmi_strategy::StrategyRegistry;
    use quantlaxmi_wal::{SegmentAdmissionSummary, WalReader};

    let segment_path = std::path::Path::new(segment_dir);
    if !segment_path.exists() {
        return Err(anyhow::anyhow!(
            "Segment directory not found: {}",
            segment_dir
        ));
    }

    let pace_mode = match pace {
        "real" => PaceMode::RealTime,
        _ => PaceMode::Fast,
    };

    // Phase 22E: Build enforcement config
    let enforcement = EnforcementConfig {
        strategies_manifest_path: strategies_manifest.map(std::path::PathBuf::from),
        signals_manifest_path: signals_manifest.map(std::path::PathBuf::from),
        promotion_root: promotion_root.map(std::path::PathBuf::from),
        require_promotion,
        wal_dir: wal_dir.map(std::path::PathBuf::from),
    };

    // Validate enforcement config
    enforcement
        .validate_for_production()
        .map_err(|e| anyhow::anyhow!(e))?;

    // Emit dev mode warning if applicable
    enforcement.emit_dev_mode_warning();

    let config = BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps,
            initial_cash: initial_capital,
            use_perp_prices: !use_spot_prices,
        },
        log_interval: 500_000,
        pace: pace_mode,
        output_trace: output_trace.map(|s| s.to_string()),
        run_id: run_id.map(|s| s.to_string()),
        params_json: None, // CLI single-run mode
        enforce_admission_from_wal,
        admission_mismatch_policy: admission_mismatch_policy.to_string(),
        strategy_spec: None, // Phase 22C: CLI doesn't set strategy_spec yet
        enforcement,
        cost_model_path: cost_model_path.map(std::path::PathBuf::from),
        latency_ticks,
        flatten_on_end,
    };

    let engine = BacktestEngine::new(config);

    tracing::info!("=== Starting Backtest ===");
    tracing::info!("Segment: {}", segment_dir);
    tracing::info!("Strategy: {}", strategy_name);
    tracing::info!("Initial capital: ${:.2}", initial_capital);
    tracing::info!("Fee: {:.1} bps", fee_bps);
    tracing::info!("Pace: {}", pace);
    tracing::info!("Latency ticks: {}", latency_ticks);
    tracing::info!(
        "Price source: {}",
        if use_spot_prices { "spot" } else { "perp" }
    );
    if let Some(ref cm_path) = cost_model_path {
        tracing::info!("Cost model: {}", cm_path);
    }

    // Run backtest with either SDK strategy or legacy strategy
    let result = if use_sdk {
        // Phase 2 SDK: Create strategy from registry
        let registry = StrategyRegistry::with_builtins();
        let config_path = strategy_config.map(std::path::Path::new);

        let strategy = registry
            .create(strategy_name, config_path)
            .map_err(|e| anyhow::anyhow!("Failed to create strategy '{}': {}", strategy_name, e))?;

        tracing::info!(
            "Using Phase 2 SDK strategy: {} (config: {:?})",
            strategy.short_id(),
            strategy_config
        );

        let (result, strategy_binding) = engine
            .run_with_strategy(segment_path, strategy, config_path)
            .await?;

        // Log strategy binding for manifest
        if let Some(ref binding) = strategy_binding {
            tracing::info!("Strategy binding: {}", binding.strategy_id);
        }

        result
    } else {
        // Legacy strategies (Phase 1)
        match strategy_name {
            "funding_bias" => {
                let strategy = FundingBiasStrategy::new(position_size, threshold_bps / 10_000.0);
                engine.run(segment_path, strategy).await?
            }
            "basis_capture" => {
                let strategy = BasisCaptureStrategy::new(threshold_bps, position_size);
                engine.run(segment_path, strategy).await?
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown strategy: {}. Available: funding_bias, basis_capture. Use --use-sdk for Phase 2 strategies.",
                    strategy_name
                ));
            }
        }
    };

    // Print results
    println!("\n=== Backtest Results ===");
    println!("Strategy: {}", result.strategy_name);
    println!("Segment: {}", result.segment_path);
    println!("Events processed: {}", result.total_events);
    println!(
        "Duration: {:.1}s ({:.1}h)",
        result.duration_secs,
        result.duration_secs / 3600.0
    );

    // Trade metrics
    let m = &result.metrics;
    println!();
    println!("--- Trade Statistics ---");
    println!(
        "Trades: {} ({}W / {}L)",
        m.total_trades, m.winning_trades, m.losing_trades
    );
    println!("Win Rate: {:.1}%", m.win_rate);
    println!("Avg Win: ${:.2} | Avg Loss: ${:.2}", m.avg_win, m.avg_loss);
    if m.profit_factor.is_finite() {
        println!("Profit Factor: {:.2}", m.profit_factor);
    } else {
        println!("Profit Factor: ∞ (no losses)");
    }
    println!("Expectancy: ${:.2} per trade", m.expectancy);
    println!(
        "Largest Win: ${:.2} | Largest Loss: ${:.2}",
        m.largest_win, m.largest_loss
    );

    // Risk metrics
    println!();
    println!("--- Risk Metrics ---");
    println!(
        "Max Drawdown: ${:.2} ({:.2}%)",
        m.max_drawdown, m.max_drawdown_pct
    );
    if m.sharpe_ratio.is_finite() {
        println!("Sharpe Ratio: {:.2}", m.sharpe_ratio);
    } else {
        println!("Sharpe Ratio: N/A");
    }
    if m.sortino_ratio.is_finite() {
        println!("Sortino Ratio: {:.2}", m.sortino_ratio);
    } else {
        println!("Sortino Ratio: N/A");
    }
    println!("Total Fees: ${:.2}", m.total_fees);
    println!("Avg Trade Duration: {:.1}s", m.avg_trade_duration_secs);

    // Capital summary
    println!();
    println!("--- Capital Summary ---");
    println!("Initial Capital: ${:.2}", result.initial_cash);
    println!("Final Cash: ${:.2}", result.final_cash);
    println!("Realized PnL: ${:.2}", result.realized_pnl);
    println!("Unrealized PnL: ${:.2}", result.unrealized_pnl);
    println!(
        "Total PnL: ${:.2} ({:.2}%)",
        result.total_pnl, result.return_pct
    );

    // Trace info (for replay parity)
    println!();
    println!("--- Decision Trace ---");
    println!("Decisions: {}", result.total_decisions);
    println!("Trace hash: {}", result.trace_hash);
    if let Some(ref path) = result.trace_path {
        println!("Trace saved to: {}", path);
    }

    // Write JSON output if requested
    if let Some(path) = output_json {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(path, json)?;
        println!("\nResults written to: {}", path);
    }

    // Phase 19D: Write admission summary
    if write_admission_summary {
        let wal_reader = WalReader::open(segment_path)?;
        let admission_decisions = wal_reader.read_admission_decisions()?;

        if !admission_decisions.is_empty() {
            let session_id = run_id.unwrap_or("backtest");
            let summary = SegmentAdmissionSummary::from_decisions(session_id, &admission_decisions);

            let summary_path = segment_path.join("segment_admission_summary.json");
            summary.write(&summary_path)?;

            println!();
            println!("--- Admission Summary (Phase 19D) ---");
            println!("Evaluated events: {}", summary.evaluated_events);
            println!("Admitted events: {}", summary.admitted_events);
            println!("Refused events: {}", summary.refused_events);
            println!(
                "Admission rate: {:.1}%",
                if summary.evaluated_events > 0 {
                    100.0 * summary.admitted_events as f64 / summary.evaluated_events as f64
                } else {
                    0.0
                }
            );
            println!("Summary written to: {}", summary_path.display());
        }
    }

    // Phase 19D: Enforcement is handled by BacktestEngine::run_with_strategy() via AdmissionMode::EnforceFromWal
    // Logging happens inside the engine when enforcement is active

    Ok(())
}

/// Verify replay parity between two decision traces.
async fn run_verify_trace(original_path: &str, replay_path: &str) -> anyhow::Result<()> {
    use quantlaxmi_events::{DecisionTrace, ReplayParityResult, verify_replay_parity};
    use std::path::Path;

    println!("=== Replay Parity Verification ===");
    println!("Original: {}", original_path);
    println!("Replay: {}", replay_path);
    println!();

    // Load traces
    let original = DecisionTrace::load(Path::new(original_path))
        .map_err(|e| anyhow::anyhow!("Failed to load original trace: {}", e))?;
    let replay = DecisionTrace::load(Path::new(replay_path))
        .map_err(|e| anyhow::anyhow!("Failed to load replay trace: {}", e))?;

    println!(
        "Original: {} decisions, hash={}...",
        original.len(),
        &original.hash_hex()[..16]
    );
    println!(
        "Replay: {} decisions, hash={}...",
        replay.len(),
        &replay.hash_hex()[..16]
    );
    println!();

    // Verify parity
    let result = verify_replay_parity(&original, &replay);

    match result {
        ReplayParityResult::Match => {
            println!("PARITY_MATCH");
            println!(
                "Traces are identical - all {} decisions match.",
                original.len()
            );
            Ok(())
        }
        ReplayParityResult::Divergence {
            index,
            original: orig_decision,
            replay: replay_decision,
            reason,
        } => {
            println!("PARITY_DIVERGENCE");
            println!();
            println!("First divergence at index: {}", index);
            println!("Reason: {}", reason);
            println!();
            println!("Original decision:");
            println!("  ts: {}", orig_decision.ts);
            println!("  decision_id: {}", orig_decision.decision_id);
            println!("  direction: {}", orig_decision.direction);
            println!("  qty: {}", orig_decision.target_qty_mantissa);
            println!();
            println!("Replay decision:");
            println!("  ts: {}", replay_decision.ts);
            println!("  decision_id: {}", replay_decision.decision_id);
            println!("  direction: {}", replay_decision.direction);
            println!("  qty: {}", replay_decision.target_qty_mantissa);
            Err(anyhow::anyhow!(
                "Replay parity check failed: divergence at index {}",
                index
            ))
        }
        ReplayParityResult::LengthMismatch {
            original_len,
            replay_len,
        } => {
            println!("PARITY_LENGTH_MISMATCH");
            println!();
            println!("Original length: {}", original_len);
            println!("Replay length: {}", replay_len);
            Err(anyhow::anyhow!(
                "Replay parity check failed: length mismatch ({} vs {})",
                original_len,
                replay_len
            ))
        }
    }
}

// =============================================================================
// Public CLI API for External Callers
// =============================================================================

/// Run backtest from CLI arguments (Phase 2 unified entry point).
///
/// This provides a simple API for running backtests that can be called from
/// external crates (e.g., quantlaxmi-crypto CLI binary).
///
/// # Arguments
/// * `session_dir` - Path to a captured session directory (contains segment_manifest.json)
/// * `strategy` - Strategy name registered in quantlaxmi-strategy
/// * `config_path` - Path to backtest config (TOML/JSON)
/// * `out_dir` - Output directory for run manifest + metrics
///
/// # Returns
/// Result indicating success or failure with error details.
pub fn run_backtest_cli(
    session_dir: &str,
    strategy: &str,
    config_path: &str,
    out_dir: &str,
) -> anyhow::Result<()> {
    // Create runtime for async backtest execution
    let rt = create_runtime()?;

    rt.block_on(async {
        use backtest::{BacktestConfig, BacktestEngine, ExchangeConfig};
        use quantlaxmi_strategy::StrategyRegistry;

        // 1) Load config (if provided, otherwise use defaults)
        let exchange_config = if std::path::Path::new(config_path).exists() {
            let content =
                std::fs::read_to_string(config_path).context("Failed to read backtest config")?;
            toml::from_str(&content).unwrap_or_else(|_| ExchangeConfig::default())
        } else {
            ExchangeConfig::default()
        };

        let config = BacktestConfig {
            exchange: exchange_config,
            ..Default::default()
        };

        let engine = BacktestEngine::new(config);

        // 2) Load strategy from registry
        let registry = StrategyRegistry::with_builtins();
        let strategy_config_path = if std::path::Path::new(config_path).exists() {
            Some(std::path::Path::new(config_path))
        } else {
            None
        };

        let strategy_box = registry
            .create(strategy, strategy_config_path)
            .map_err(|e| anyhow::anyhow!("Failed to create strategy '{}': {}", strategy, e))?;

        // 3) Run backtest
        let segment_path = std::path::Path::new(session_dir);
        let (result, _strategy_binding) = engine
            .run_with_strategy(segment_path, strategy_box, strategy_config_path)
            .await?;

        // 4) Write outputs (using v1 schemas with deterministic serialization)
        let out_path = std::path::Path::new(out_dir);
        std::fs::create_dir_all(out_path)?;

        // Write metrics.json (v1 schema)
        let metrics_v1 = backtest::BacktestMetricsV1::from_metrics(&result.metrics);
        let metrics_path = out_path.join("metrics.json");
        let metrics_json = serde_json::to_string_pretty(&metrics_v1)?;
        std::fs::write(&metrics_path, metrics_json)?;

        // Write run_manifest.json (v1 schema)
        let run_manifest_v1 = backtest::BacktestRunManifestV1::from_result(&result);
        let run_manifest_path = out_path.join("run_manifest.json");
        let manifest_json = serde_json::to_string_pretty(&run_manifest_v1)?;
        std::fs::write(&run_manifest_path, manifest_json)?;

        // Write equity_curve.jsonl (P0 audit-grade trace)
        let equity_curve_path = out_path.join("equity_curve.jsonl");
        backtest::write_equity_curve_jsonl(&equity_curve_path, &result.equity_curve)
            .context("Failed to write equity_curve.jsonl")?;

        // Write fills.jsonl (P0 audit-grade trace)
        let fills_path = out_path.join("fills.jsonl");
        backtest::write_fills_jsonl(&fills_path, &result.fills)
            .context("Failed to write fills.jsonl")?;

        info!(
            "Backtest complete: {} events, {} fills, {:.2}% return",
            result.total_events, result.total_fills, result.return_pct
        );
        info!("Metrics written to: {}", metrics_path.display());
        info!("Manifest written to: {}", run_manifest_path.display());
        info!("Equity curve written to: {}", equity_curve_path.display());
        info!("Fills written to: {}", fills_path.display());

        Ok(())
    })
}
