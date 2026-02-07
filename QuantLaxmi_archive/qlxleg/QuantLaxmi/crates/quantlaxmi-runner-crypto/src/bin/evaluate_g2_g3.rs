//! G2/G3 Anti-Overfit Evaluation Harness
//!
//! Evaluates strategy robustness and stability:
//! - G2: Time-shift degradation, cost sensitivity, random baseline
//! - G3: Walk-forward stability across chronological folds
//!
//! ## Usage
//! ```bash
//! # Run full G2/G3 evaluation
//! cargo run --bin evaluate_g2_g3 -- \
//!     --segment-dir data/segments/perp_20260125_100000 \
//!     --strategy funding_bias \
//!     --strategy-config config/funding_bias.json
//!
//! # Run only G2 (robustness)
//! cargo run --bin evaluate_g2_g3 -- \
//!     --segment-dir data/segments/perp_20260125_100000 \
//!     --strategy funding_bias \
//!     --g2-only
//!
//! # Run only G3 (walk-forward)
//! cargo run --bin evaluate_g2_g3 -- \
//!     --segment-dir data/segments/perp_20260125_100000 \
//!     --strategy funding_bias \
//!     --g3-only \
//!     --num-folds 5
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use quantlaxmi_models::{
    AlphaScoreV1, AttributionSummary, CostSensitivityResult, G2Gate, G2Report, G2Thresholds,
    G3Gate, G3Report, G3Thresholds, RandomBaselineResult, TimeShiftResult, WalkForwardFold,
};
use quantlaxmi_runner_crypto::backtest::{
    BacktestConfig, BacktestEngine, BacktestResult, BasisCaptureStrategy, ExchangeConfig, Fill,
    FundingBiasStrategy, OrderIntent, PaceMode, Side, Strategy,
};
use quantlaxmi_runner_crypto::replay::{EventKind, ReplayEvent};
use quantlaxmi_runner_crypto::segment_manifest::SegmentManifest;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::PathBuf;
use uuid::Uuid;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "evaluate_g2_g3")]
#[command(about = "G2/G3 Anti-Overfit Evaluation Harness")]
struct Args {
    /// Path to segment directory
    #[arg(long)]
    segment_dir: PathBuf,

    /// Strategy name (funding_bias, basis_capture)
    #[arg(long)]
    strategy: String,

    /// Path to strategy config JSON (optional)
    #[arg(long)]
    strategy_config: Option<PathBuf>,

    /// Run only G2 evaluation
    #[arg(long, default_value = "false")]
    g2_only: bool,

    /// Run only G3 evaluation
    #[arg(long, default_value = "false")]
    g3_only: bool,

    /// Number of folds for G3 walk-forward (default: 5)
    #[arg(long, default_value = "5")]
    num_folds: u32,

    /// Time-shift values to test (comma-separated, default: 1,3,5)
    #[arg(long, default_value = "1,3,5")]
    shift_k_values: String,

    /// Cost multipliers to test (comma-separated, default: 1,2,5)
    #[arg(long, default_value = "1,2,5")]
    cost_multipliers: String,

    /// Base fee in basis points (default: 10)
    #[arg(long, default_value = "10.0")]
    base_fee_bps: f64,

    /// Initial cash for backtest
    #[arg(long, default_value = "10000.0")]
    initial_cash: f64,

    /// Skip manifest binding (dry run)
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Verbose output
    #[arg(long, short, default_value = "false")]
    verbose: bool,

    /// Position size for strategy (default: 0.01 BTC)
    #[arg(long, default_value = "0.01")]
    position_size: f64,

    /// Threshold in basis points for strategy signals (default: 10)
    #[arg(long, default_value = "10.0")]
    threshold_bps: f64,
}

// =============================================================================
// Time-Shifted Strategy Wrapper
// =============================================================================

/// Wraps a strategy to delay order execution by k events.
/// This tests whether strategy alpha is robust to execution timing.
struct TimeShiftedStrategy<S: Strategy> {
    inner: S,
    shift_k: usize,
    order_queue: VecDeque<Vec<OrderIntent>>,
}

impl<S: Strategy> TimeShiftedStrategy<S> {
    fn new(inner: S, shift_k: usize) -> Self {
        Self {
            inner,
            shift_k,
            order_queue: VecDeque::new(),
        }
    }
}

impl<S: Strategy> Strategy for TimeShiftedStrategy<S> {
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        // Get orders from inner strategy
        let orders = self.inner.on_event(event);

        // Queue the orders
        self.order_queue.push_back(orders);

        // Return orders from k events ago (if available)
        if self.order_queue.len() > self.shift_k {
            self.order_queue.pop_front().unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.inner.on_fill(fill);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// =============================================================================
// Time-Bounded Strategy Wrapper
// =============================================================================

/// Wraps a strategy to only trade within a time window (for fold evaluation).
struct TimeBoundedStrategy<S: Strategy> {
    inner: S,
    start_ts_ns: i64,
    end_ts_ns: i64,
}

impl<S: Strategy> TimeBoundedStrategy<S> {
    fn new(inner: S, start_ts_ns: i64, end_ts_ns: i64) -> Self {
        Self {
            inner,
            start_ts_ns,
            end_ts_ns,
        }
    }
}

impl<S: Strategy> Strategy for TimeBoundedStrategy<S> {
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        let event_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);

        // Only process events within time window
        if event_ts_ns < self.start_ts_ns || event_ts_ns >= self.end_ts_ns {
            return Vec::new();
        }

        self.inner.on_event(event)
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.inner.on_fill(fill);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// =============================================================================
// Random Baseline Strategy
// =============================================================================

/// Deterministic random baseline strategy.
/// Makes random entry/exit decisions with same constraints as target strategy.
struct RandomBaselineStrategy {
    #[allow(dead_code)] // Retained for debugging/documentation
    seed: u64,
    rng_state: u64,
    position: f64,
    min_hold_events: usize,
    events_since_entry: usize,
    target_decisions: usize,
    decisions_made: usize,
}

impl RandomBaselineStrategy {
    fn new(seed: u64, target_decisions: usize) -> Self {
        Self {
            seed,
            rng_state: seed,
            position: 0.0,
            min_hold_events: 10, // Minimum events between decisions
            events_since_entry: 0,
            target_decisions,
            decisions_made: 0,
        }
    }

    /// Simple deterministic PRNG (xorshift64)
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn should_trade(&mut self) -> bool {
        if self.decisions_made >= self.target_decisions {
            return false;
        }
        // ~1% chance per event to make a decision
        (self.next_random() % 100) < 1
    }
}

impl Strategy for RandomBaselineStrategy {
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        self.events_since_entry += 1;

        // Only consider trading on quote events
        if !matches!(event.kind, EventKind::PerpQuote) {
            return Vec::new();
        }

        let symbol = event.symbol.clone();

        // If flat, maybe enter
        if self.position == 0.0 && self.should_trade() {
            let side = if self.next_random().is_multiple_of(2) {
                Side::Buy
            } else {
                Side::Sell
            };
            self.position = if side == Side::Buy { 1.0 } else { -1.0 };
            self.events_since_entry = 0;
            self.decisions_made += 1;
            return vec![OrderIntent::market(symbol, side, 0.01)];
        }

        // If in position and held long enough, maybe exit
        if self.position != 0.0
            && self.events_since_entry >= self.min_hold_events
            && self.should_trade()
        {
            let side = if self.position > 0.0 {
                Side::Sell
            } else {
                Side::Buy
            };
            self.position = 0.0;
            self.decisions_made += 1;
            return vec![OrderIntent::market(symbol, side, 0.01)];
        }

        Vec::new()
    }

    fn on_fill(&mut self, _fill: &Fill) {}

    fn name(&self) -> &str {
        "random_baseline"
    }
}

// =============================================================================
// Strategy Enum for Dynamic Dispatch
// =============================================================================

/// Strategy enum to avoid dynamic dispatch issues with BacktestEngine.
enum StrategyKind {
    FundingBias(FundingBiasStrategy),
    BasisCapture(BasisCaptureStrategy),
}

impl Strategy for StrategyKind {
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        match self {
            StrategyKind::FundingBias(s) => s.on_event(event),
            StrategyKind::BasisCapture(s) => s.on_event(event),
        }
    }

    fn on_fill(&mut self, fill: &Fill) {
        match self {
            StrategyKind::FundingBias(s) => s.on_fill(fill),
            StrategyKind::BasisCapture(s) => s.on_fill(fill),
        }
    }

    fn name(&self) -> &str {
        match self {
            StrategyKind::FundingBias(s) => s.name(),
            StrategyKind::BasisCapture(s) => s.name(),
        }
    }
}

/// Creates a strategy instance by name with parameters.
fn create_strategy(name: &str, args: &Args) -> Result<StrategyKind> {
    match name {
        "funding_bias" => {
            // FundingBias uses threshold_bps converted to decimal (e.g., 10 bps = 0.001)
            let threshold = args.threshold_bps / 10_000.0;
            Ok(StrategyKind::FundingBias(FundingBiasStrategy::new(
                args.position_size,
                threshold,
            )))
        }
        "basis_capture" => Ok(StrategyKind::BasisCapture(BasisCaptureStrategy::new(
            args.threshold_bps,
            args.position_size,
        ))),
        _ => anyhow::bail!(
            "Unknown strategy: {}. Supported: funding_bias, basis_capture",
            name
        ),
    }
}

// =============================================================================
// Backtest Helpers
// =============================================================================

fn create_backtest_config(args: &Args, run_id: &str, fee_multiplier: f64) -> BacktestConfig {
    BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps: args.base_fee_bps * fee_multiplier,
            initial_cash: args.initial_cash,
            use_perp_prices: true,
        },
        log_interval: 1_000_000,
        pace: PaceMode::Fast,
        output_trace: None,
        run_id: Some(run_id.to_string()),
        ..Default::default()
    }
}

/// Build AttributionSummary from BacktestResult.
fn build_summary_from_result(
    result: &BacktestResult,
    strategy_name: &str,
    run_id: &str,
) -> AttributionSummary {
    let winning = result.trades.iter().filter(|t| t.pnl > 0.0).count() as u32;
    let losing = result.trades.iter().filter(|t| t.pnl <= 0.0).count() as u32;
    let total = winning + losing;

    let win_rate_bps = if total > 0 {
        (winning * 10000) / total
    } else {
        0
    };

    let max_loss = result
        .trades
        .iter()
        .map(|t| if t.pnl < 0.0 { -t.pnl } else { 0.0 })
        .fold(0.0f64, |a, b| a.max(b));

    let total_holding_ns: i64 = result
        .trades
        .iter()
        .map(|t| (t.duration_secs * 1_000_000_000.0) as i64)
        .sum();

    AttributionSummary {
        strategy_id: format!("{}:1.0.0:test", strategy_name),
        run_id: run_id.to_string(),
        symbols: vec!["BTCUSDT".to_string()],
        generated_ts_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        total_decisions: result.total_decisions as u32,
        total_fills: result.total_fills as u32,
        winning_decisions: winning,
        losing_decisions: losing,
        round_trips: (result.total_fills / 2) as u32, // Approximate: fills / 2
        total_gross_pnl_mantissa: result.realized_pnl_mantissa + result.total_fees_mantissa,
        total_fees_mantissa: result.total_fees_mantissa,
        total_net_pnl_mantissa: result.realized_pnl_mantissa,
        pnl_exponent: result.pnl_exponent,
        win_rate_bps,
        avg_pnl_per_decision_mantissa: if result.total_decisions > 0 {
            result.realized_pnl_mantissa / result.total_decisions as i128
        } else {
            0
        },
        total_slippage_mantissa: 0,
        slippage_exponent: result.pnl_exponent,
        max_loss_mantissa: (max_loss * 100_000_000.0) as i128,
        total_holding_time_ns: total_holding_ns,
    }
}

/// Compute SHA-256 hash of AttributionSummary.
fn compute_summary_hash(summary: &AttributionSummary) -> String {
    let json = serde_json::to_string(summary).unwrap_or_default();
    let hash = Sha256::digest(json.as_bytes());
    hex::encode(hash)
}

/// Compute deterministic seed from segment path and run ID.
fn compute_baseline_seed(segment_dir: &std::path::Path, run_id: &str) -> u64 {
    let input = format!("{}:{}", segment_dir.display(), run_id);
    let hash = Sha256::digest(input.as_bytes());
    u64::from_be_bytes(hash[0..8].try_into().unwrap())
}

// =============================================================================
// G2 Evaluation
// =============================================================================

/// Run G2 robustness evaluation.
async fn run_g2_evaluation(
    args: &Args,
    base_summary: &AttributionSummary,
    base_alpha: &AlphaScoreV1,
) -> Result<G2Report> {
    let gate = G2Gate::default();
    let run_id = Uuid::new_v4().to_string();

    println!("\n=== G2 Robustness Evaluation ===");
    println!(
        "Base score: {} (mantissa) / 10^{}",
        base_alpha.score_mantissa, base_alpha.score_exponent
    );

    // Parse shift values
    let shift_values: Vec<u32> = args
        .shift_k_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Parse cost multipliers
    let cost_mults: Vec<u32> = args
        .cost_multipliers
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // === Time-Shift Tests ===
    println!("\n--- Time-Shift Tests ---");
    let mut time_shift_results = Vec::new();

    for &k in &shift_values {
        println!("  Running shift k={}...", k);

        let strategy = create_strategy(&args.strategy, args)?;
        let shifted = TimeShiftedStrategy::new(strategy, k as usize);

        let config = create_backtest_config(args, &format!("{}_shift_{}", run_id, k), 1.0);
        let engine = BacktestEngine::new(config);
        let result = engine.run(&args.segment_dir, shifted).await?;

        let summary = build_summary_from_result(&result, &args.strategy, &run_id);
        let alpha = AlphaScoreV1::from_summary(&summary);

        let degradation_ratio_bps = if base_alpha.score_mantissa > 0 {
            ((alpha.score_mantissa * 10000) / base_alpha.score_mantissa) as u32
        } else {
            0
        };

        let summary_hash = compute_summary_hash(&summary);

        println!(
            "    Score: {} ({}% of base)",
            alpha.score_mantissa,
            degradation_ratio_bps / 100
        );

        time_shift_results.push(TimeShiftResult {
            shift_k: k,
            score_mantissa: alpha.score_mantissa,
            score_exponent: alpha.score_exponent,
            win_rate_bps: summary.win_rate_bps,
            total_decisions: summary.total_decisions,
            net_pnl_mantissa: summary.total_net_pnl_mantissa,
            pnl_exponent: summary.pnl_exponent,
            degradation_ratio_bps,
            summary_sha256: summary_hash,
        });
    }

    let (time_shift_passed, time_shift_reasons) =
        gate.evaluate_time_shift(base_alpha.score_mantissa, &time_shift_results);
    println!(
        "  Time-shift: {}",
        if time_shift_passed { "PASS" } else { "FAIL" }
    );

    // === Cost Sensitivity Tests ===
    println!("\n--- Cost Sensitivity Tests ---");
    let mut cost_sensitivity_results = Vec::new();

    for &mult in &cost_mults {
        println!("  Running cost {}x...", mult);

        let strategy = create_strategy(&args.strategy, args)?;
        let config =
            create_backtest_config(args, &format!("{}_cost_{}x", run_id, mult), mult as f64);
        let engine = BacktestEngine::new(config);
        let result = engine.run(&args.segment_dir, strategy).await?;

        let summary = build_summary_from_result(&result, &args.strategy, &run_id);
        let alpha = AlphaScoreV1::from_summary(&summary);

        let retention_ratio_bps = if base_alpha.score_mantissa > 0 {
            ((alpha.score_mantissa * 10000) / base_alpha.score_mantissa) as u32
        } else {
            0
        };

        let summary_hash = compute_summary_hash(&summary);

        println!(
            "    Score: {} ({}% retention)",
            alpha.score_mantissa,
            retention_ratio_bps / 100
        );

        cost_sensitivity_results.push(CostSensitivityResult {
            fee_multiplier: mult,
            slippage_multiplier: mult,
            score_mantissa: alpha.score_mantissa,
            score_exponent: alpha.score_exponent,
            win_rate_bps: summary.win_rate_bps,
            net_pnl_mantissa: summary.total_net_pnl_mantissa,
            pnl_exponent: summary.pnl_exponent,
            retention_ratio_bps,
            summary_sha256: summary_hash,
        });
    }

    let (cost_sensitivity_passed, cost_sensitivity_reasons) =
        gate.evaluate_cost_sensitivity(base_alpha.score_mantissa, &cost_sensitivity_results);
    println!(
        "  Cost sensitivity: {}",
        if cost_sensitivity_passed {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // === Random Baseline Test ===
    println!("\n--- Random Baseline Test ---");

    let seed = compute_baseline_seed(&args.segment_dir, &run_id);
    let seed_hex = format!("{:016x}", seed);
    println!("  Seed: {}", seed_hex);

    let baseline_strategy =
        RandomBaselineStrategy::new(seed, base_summary.total_decisions as usize);
    let config = create_backtest_config(args, &format!("{}_baseline", run_id), 1.0);
    let engine = BacktestEngine::new(config);
    let result = engine.run(&args.segment_dir, baseline_strategy).await?;

    let baseline_summary = build_summary_from_result(&result, "random_baseline", &run_id);
    let baseline_alpha = AlphaScoreV1::from_summary(&baseline_summary);

    let edge_ratio_bps = if baseline_alpha.score_mantissa > 0 {
        ((base_alpha.score_mantissa * 10000) / baseline_alpha.score_mantissa) as u32
    } else if base_alpha.score_mantissa > 0 {
        u32::MAX
    } else {
        10000
    };

    let baseline_hash = compute_summary_hash(&baseline_summary);

    println!(
        "  Baseline score: {}, Strategy score: {}",
        baseline_alpha.score_mantissa, base_alpha.score_mantissa
    );
    println!("  Edge ratio: {}%", edge_ratio_bps / 100);

    let baseline_result = RandomBaselineResult {
        seed_hex,
        baseline_score_mantissa: baseline_alpha.score_mantissa,
        baseline_score_exponent: baseline_alpha.score_exponent,
        baseline_win_rate_bps: baseline_summary.win_rate_bps,
        baseline_net_pnl_mantissa: baseline_summary.total_net_pnl_mantissa,
        strategy_score_mantissa: base_alpha.score_mantissa,
        edge_ratio_bps,
        absolute_edge_mantissa: base_alpha.score_mantissa - baseline_alpha.score_mantissa,
        baseline_summary_sha256: baseline_hash,
    };

    let (baseline_passed, baseline_reasons) = gate.evaluate_baseline(&baseline_result);
    println!(
        "  Baseline comparison: {}",
        if baseline_passed { "PASS" } else { "FAIL" }
    );

    // === Build Report ===
    let passed = time_shift_passed && cost_sensitivity_passed && baseline_passed;
    let mut all_reasons = Vec::new();
    all_reasons.extend(time_shift_reasons.clone());
    all_reasons.extend(cost_sensitivity_reasons.clone());
    all_reasons.extend(baseline_reasons.clone());

    println!(
        "\n=== G2 Result: {} ===",
        if passed { "PASS" } else { "FAIL" }
    );

    Ok(G2Report {
        version: G2Report::VERSION.to_string(),
        generated_ts_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        strategy_id: base_summary.strategy_id.clone(),
        run_id: run_id.clone(),
        base_summary_sha256: compute_summary_hash(base_summary),
        base_score_mantissa: base_alpha.score_mantissa,
        base_score_exponent: base_alpha.score_exponent,
        time_shift_results,
        time_shift_passed,
        time_shift_reasons,
        cost_sensitivity_results,
        cost_sensitivity_passed,
        cost_sensitivity_reasons,
        baseline_result,
        baseline_passed,
        baseline_reasons,
        thresholds: G2Thresholds::default(),
        passed,
        all_reasons,
    })
}

// =============================================================================
// G3 Evaluation
// =============================================================================

/// Run G3 walk-forward stability evaluation.
async fn run_g3_evaluation(args: &Args, base_summary: &AttributionSummary) -> Result<G3Report> {
    let gate = G3Gate::default();
    let run_id = Uuid::new_v4().to_string();

    println!("\n=== G3 Walk-Forward Stability Evaluation ===");
    println!("Number of folds: {}", args.num_folds);

    // Get segment time bounds
    let manifest =
        SegmentManifest::load(&args.segment_dir).context("Failed to load segment manifest")?;

    let start_ts_ns = manifest.start_ts.timestamp_nanos_opt().unwrap_or(0);
    let end_ts_ns = manifest
        .end_ts
        .map(|t| t.timestamp_nanos_opt().unwrap_or(0))
        .unwrap_or(start_ts_ns + 3_600_000_000_000);

    let duration_ns = end_ts_ns - start_ts_ns;
    let fold_duration_ns = duration_ns / args.num_folds as i64;

    println!(
        "Segment duration: {:.2} hours",
        duration_ns as f64 / 3_600_000_000_000.0
    );
    println!(
        "Fold duration: {:.2} minutes",
        fold_duration_ns as f64 / 60_000_000_000.0
    );

    // Run each fold
    let mut folds = Vec::new();

    for fold_idx in 0..args.num_folds {
        let fold_start = start_ts_ns + (fold_idx as i64 * fold_duration_ns);
        let fold_end = if fold_idx == args.num_folds - 1 {
            end_ts_ns
        } else {
            fold_start + fold_duration_ns
        };

        println!("\n--- Fold {} ---", fold_idx);

        let strategy = create_strategy(&args.strategy, args)?;
        let bounded = TimeBoundedStrategy::new(strategy, fold_start, fold_end);

        let config = create_backtest_config(args, &format!("{}_fold_{}", run_id, fold_idx), 1.0);
        let engine = BacktestEngine::new(config);
        let result = engine.run(&args.segment_dir, bounded).await?;

        let summary = build_summary_from_result(&result, &args.strategy, &run_id);
        let alpha = AlphaScoreV1::from_summary(&summary);

        println!(
            "  Decisions: {}, Score: {}, Win rate: {}%",
            summary.total_decisions,
            alpha.score_mantissa,
            summary.win_rate_bps / 100
        );

        folds.push(WalkForwardFold {
            fold_index: fold_idx,
            start_ts_ns: fold_start,
            end_ts_ns: fold_end,
            num_decisions: summary.total_decisions,
            score_mantissa: alpha.score_mantissa,
            score_exponent: alpha.score_exponent,
            win_rate_bps: summary.win_rate_bps,
            net_pnl_mantissa: summary.total_net_pnl_mantissa,
            pnl_exponent: summary.pnl_exponent,
            max_loss_mantissa: summary.max_loss_mantissa,
            summary_sha256: compute_summary_hash(&summary),
        });
    }

    // Compute stability metrics
    let metrics = gate.compute_stability_metrics(&folds);

    println!("\n--- Stability Metrics ---");
    println!("  Median score: {}", metrics.median_score_mantissa);
    println!("  Min score: {}", metrics.min_score_mantissa);
    println!("  Max score: {}", metrics.max_score_mantissa);
    println!("  Dispersion: {}", metrics.score_dispersion_mantissa);
    println!(
        "  Consistency: {}% ({}/{} profitable)",
        metrics.consistency_ratio_bps / 100,
        metrics.profitable_folds,
        metrics.total_folds
    );

    // Evaluate gate
    let (passed, reasons) = gate.evaluate(&metrics);

    println!(
        "\n=== G3 Result: {} ===",
        if passed { "PASS" } else { "FAIL" }
    );

    Ok(G3Report {
        version: G3Report::VERSION.to_string(),
        generated_ts_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        strategy_id: base_summary.strategy_id.clone(),
        run_id,
        segment_start_ts_ns: start_ts_ns,
        segment_end_ts_ns: end_ts_ns,
        folds,
        stability_metrics: metrics,
        thresholds: G3Thresholds::default(),
        passed,
        reasons,
    })
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("quantlaxmi=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    println!("G2/G3 Anti-Overfit Evaluation");
    println!("==============================");
    println!("Segment: {:?}", args.segment_dir);
    println!("Strategy: {}", args.strategy);

    // Verify segment exists
    if !args.segment_dir.exists() {
        anyhow::bail!("Segment directory not found: {:?}", args.segment_dir);
    }

    // Run base backtest to get baseline metrics
    println!("\n=== Running Base Backtest ===");
    let strategy = create_strategy(&args.strategy, &args)?;
    let run_id = Uuid::new_v4().to_string();
    let config = create_backtest_config(&args, &run_id, 1.0);
    let engine = BacktestEngine::new(config);
    let base_result = engine.run(&args.segment_dir, strategy).await?;

    let base_summary = build_summary_from_result(&base_result, &args.strategy, "base");
    let base_alpha = AlphaScoreV1::from_summary(&base_summary);

    println!(
        "Base result: {} decisions, {} fills, PnL: {}",
        base_result.total_decisions, base_result.total_fills, base_result.realized_pnl
    );
    println!(
        "Alpha score: {} (mantissa) / 10^{}",
        base_alpha.score_mantissa, base_alpha.score_exponent
    );

    // Run evaluations
    let run_g2 = !args.g3_only;
    let run_g3 = !args.g2_only;

    let g2_report = if run_g2 {
        Some(run_g2_evaluation(&args, &base_summary, &base_alpha).await?)
    } else {
        None
    };

    let g3_report = if run_g3 {
        Some(run_g3_evaluation(&args, &base_summary).await?)
    } else {
        None
    };

    // Bind to manifest (unless dry run)
    if !args.dry_run {
        println!("\n=== Binding Reports to Manifest ===");

        let mut manifest = SegmentManifest::load(&args.segment_dir)
            .context("Failed to load manifest for binding")?;

        if let Some(ref report) = g2_report {
            manifest.bind_g2_report(report, &args.segment_dir)?;
            println!("  Bound G2 report: g2_report.json");
        }

        if let Some(ref report) = g3_report {
            manifest.bind_g3_report(report, &args.segment_dir)?;
            println!("  Bound G3 report: g3_walkforward.json");
        }

        manifest.write(&args.segment_dir)?;
        println!("  Updated segment_manifest.json");
    } else {
        println!("\n=== Dry Run - Reports Not Saved ===");
    }

    // Final summary
    println!("\n=== Final Summary ===");
    if let Some(ref report) = g2_report {
        println!(
            "G2 Robustness: {}",
            if report.passed {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        );
        if !report.all_reasons.is_empty() {
            for reason in &report.all_reasons {
                println!("  - {}", reason);
            }
        }
    }
    if let Some(ref report) = g3_report {
        println!(
            "G3 Stability:  {}",
            if report.passed {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        );
        if !report.reasons.is_empty() {
            for reason in &report.reasons {
                println!("  - {}", reason);
            }
        }
    }

    // Overall promotion decision
    let g2_passed = g2_report.as_ref().map(|r| r.passed).unwrap_or(true);
    let g3_passed = g3_report.as_ref().map(|r| r.passed).unwrap_or(true);

    if g2_passed && g3_passed {
        println!("\n✓ Strategy PROMOTED - passed all anti-overfit checks");
    } else {
        println!("\n✗ Strategy NOT PROMOTED - failed anti-overfit validation");
    }

    Ok(())
}
