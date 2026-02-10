//! Tournament Runner for Alpha Discovery (Phase 12.2)
//!
//! Evaluates multiple strategies against multiple segments, producing:
//! - Per-run manifests with decision traces
//! - Attribution summaries and alpha scores
//! - G1 gate evaluation results
//! - Tournament leaderboard ranked by alpha score
//!
//! ## Determinism Contract
//! - Tournament ID derived from inputs (preset, segments, strategies)
//! - All timestamps derived from segment data (no wall-clock)
//! - Stable sort: segments by ID, strategies by name
//! - Canonical JSON serialization (typed structs only)

use crate::backtest::{
    BacktestConfig, BacktestEngine, EnforcementConfig, ExchangeConfig, PaceMode,
};
use crate::segment_manifest::SegmentManifest;
use anyhow::{Context, Result};
use quantlaxmi_models::tournament::{
    ArtifactDigest, InputSegment, LEADERBOARD_SCHEMA, LeaderboardRow, LeaderboardV1,
    RUN_MANIFEST_SCHEMA, RunManifestV1, RunRecord, RunResultPaths, TOURNAMENT_MANIFEST_SCHEMA,
    TournamentManifestV1, TournamentPreset, compare_rows, compute_bundle_digest, generate_run_id,
    generate_run_key, generate_tournament_id, is_meaningful_run,
};
use quantlaxmi_models::{
    AlphaScoreV1, AttributionSummary, AttributionSummaryBuilder, G1PromotionGate, G1PromotionResult,
};
use quantlaxmi_strategy::StrategyRegistry;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// =============================================================================
// Tournament Configuration
// =============================================================================

/// Tournament configuration (CLI args normalized).
#[derive(Debug, Clone)]
pub struct TournamentConfig {
    /// Preset name (e.g., "baseline_v1")
    pub preset: String,
    /// Paths to segment directories
    pub segment_dirs: Vec<PathBuf>,
    /// Strategy names to evaluate (from registry)
    pub strategy_names: Vec<String>,
    /// Output directory for tournament results
    pub output_dir: PathBuf,
    /// Symbols filter (empty = all symbols)
    pub symbols: Vec<String>,
    /// Initial capital for backtests
    pub initial_capital_f64: f64,
    /// Fee in basis points
    pub fee_bps_f64: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            preset: "baseline_v1".to_string(),
            segment_dirs: Vec::new(),
            strategy_names: Vec::new(),
            output_dir: PathBuf::from("tournaments"),
            symbols: Vec::new(),
            initial_capital_f64: 10000.0,
            fee_bps_f64: 10.0,
            verbose: false,
        }
    }
}

// =============================================================================
// Tournament Result
// =============================================================================

/// Result of tournament execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentResult {
    /// Tournament ID (deterministic)
    pub tournament_id: String,
    /// Preset used
    pub preset: String,
    /// Number of segments evaluated
    pub segment_count: usize,
    /// Number of strategies evaluated
    pub strategy_count: usize,
    /// Total runs completed
    pub total_runs: usize,
    /// Meaningful runs (met activity threshold)
    pub meaningful_runs: usize,
    /// Output directory
    pub output_dir: String,
    /// Path to tournament manifest
    pub manifest_path: String,
    /// Path to leaderboard JSON
    pub leaderboard_json_path: String,
    /// Path to leaderboard markdown
    pub leaderboard_md_path: String,
}

// =============================================================================
// Single Run Result (internal)
// =============================================================================

/// Result of a single strategy/segment run.
#[derive(Debug, Clone)]
struct SingleRunResult {
    /// Run ID
    run_id: String,
    /// Run key (filesystem-safe)
    run_key: String,
    /// Segment ID
    segment_id: String,
    /// Strategy ID (name:version:hash)
    strategy_id: String,
    /// Strategy name only
    strategy_name: String,
    /// Attribution summary
    summary: AttributionSummary,
    /// Alpha score
    alpha_score: AlphaScoreV1,
    /// G1 gate result
    g1_result: G1PromotionResult,
    /// Decision trace hash
    trace_hash: String,
    /// Paths to result artifacts (relative to run dir)
    result_paths: RunResultPaths,
    /// Artifact digests
    artifact_digests: BTreeMap<String, ArtifactDigest>,
}

// =============================================================================
// Tournament Runner
// =============================================================================

/// Run a tournament evaluation.
///
/// This is the main entry point for Phase 12.2 tournament execution.
/// It evaluates each strategy against each segment and produces
/// tournament artifacts with deterministic IDs.
pub async fn run_tournament(config: TournamentConfig) -> Result<TournamentResult> {
    info!("=== Tournament Runner ===");
    info!("Preset: {}", config.preset);
    info!("Segments: {}", config.segment_dirs.len());
    info!("Strategies: {:?}", config.strategy_names);

    // Load preset
    let preset = match config.preset.as_str() {
        "baseline_v1" => TournamentPreset::baseline_v1(),
        other => {
            return Err(anyhow::anyhow!(
                "Unknown preset: '{}'. Available: baseline_v1",
                other
            ));
        }
    };

    // Load segment manifests and sort by segment_id
    let mut segments: Vec<(PathBuf, SegmentManifest)> = Vec::new();
    for dir in &config.segment_dirs {
        let manifest = SegmentManifest::load(dir)
            .with_context(|| format!("load segment manifest: {:?}", dir))?;
        segments.push((dir.clone(), manifest));
    }
    segments.sort_by(|a, b| a.1.segment_id.cmp(&b.1.segment_id));

    // Create input segment records
    let input_segments: Vec<InputSegment> = segments
        .iter()
        .map(|(dir, m)| {
            let manifest_sha256 = compute_file_sha256(&dir.join("segment_manifest.json"))
                .unwrap_or_else(|_| "UNKNOWN".to_string());
            InputSegment {
                segment_id: m.segment_id.clone(),
                path: dir.to_string_lossy().to_string(),
                manifest_sha256,
            }
        })
        .collect();

    // Sort strategy names for determinism
    let mut strategy_names = config.strategy_names.clone();
    strategy_names.sort();

    // Generate deterministic tournament ID
    let segment_digests: Vec<String> = input_segments
        .iter()
        .map(|s| s.manifest_sha256.clone())
        .collect();
    let tournament_id = generate_tournament_id(&config.preset, &segment_digests, &strategy_names);

    info!("Tournament ID: {}...", &tournament_id[..16]);

    // Create output directory structure
    let tournament_dir = config.output_dir.join(&tournament_id[..16]);
    std::fs::create_dir_all(&tournament_dir)?;
    let runs_dir = tournament_dir.join("runs");
    std::fs::create_dir_all(&runs_dir)?;

    // Initialize strategy registry
    let registry = StrategyRegistry::with_builtins();

    // Validate all strategies exist before running
    for name in &strategy_names {
        if !registry.contains(name) {
            return Err(anyhow::anyhow!(
                "Unknown strategy: '{}'. Available: {:?}",
                name,
                registry.list()
            ));
        }
    }

    // Run all strategy/segment combinations
    let mut run_results: Vec<SingleRunResult> = Vec::new();
    let mut artifact_digests: BTreeMap<String, ArtifactDigest> = BTreeMap::new();

    for (segment_dir, segment_manifest) in &segments {
        for strategy_name in &strategy_names {
            info!(
                "Running {} on segment {}...",
                strategy_name, segment_manifest.segment_id
            );

            match run_single_evaluation(
                segment_dir,
                segment_manifest,
                strategy_name,
                &registry,
                &runs_dir,
                &config,
            )
            .await
            {
                Ok(result) => {
                    // Collect artifact digests
                    for (name, digest) in &result.artifact_digests {
                        let full_name = format!("runs/{}/{}", result.run_key, name);
                        artifact_digests.insert(full_name, digest.clone());
                    }
                    run_results.push(result);
                }
                Err(e) => {
                    warn!(
                        "Failed to run {} on {}: {}",
                        strategy_name, segment_manifest.segment_id, e
                    );
                }
            }
        }
    }

    // Sort runs by run_key for determinism
    run_results.sort_by(|a, b| a.run_key.cmp(&b.run_key));

    // Convert to RunRecords
    let runs: Vec<RunRecord> = run_results
        .iter()
        .map(|r| RunRecord {
            run_id: r.run_id.clone(),
            run_key: r.run_key.clone(),
            segment_id: r.segment_id.clone(),
            strategy_id: r.strategy_id.clone(),
            strategy_name: r.strategy_name.clone(),
            result_paths: r.result_paths.clone(),
            alpha_score_mantissa: r.alpha_score.score_mantissa as i64,
            alpha_score_exponent: r.alpha_score.score_exponent,
            decisions: r.summary.total_decisions,
            fills: r.summary.total_fills,
            round_trips: r.summary.round_trips,
            win_rate_bps: r.summary.win_rate_bps,
            net_pnl_mantissa: r.summary.total_net_pnl_mantissa,
            pnl_exponent: r.summary.pnl_exponent,
            g1_passed: r.g1_result.passed,
            g1_reasons: r.g1_result.reasons.clone(),
            g2_passed: None,
            g2_reasons: Vec::new(),
            g3_passed: None,
            g3_reasons: Vec::new(),
            run_manifest_sha256: r.trace_hash.clone(),
        })
        .collect();

    // Compute bundle digest
    let bundle_digest = compute_bundle_digest(&artifact_digests);

    // Derive timestamp from max segment end time
    let created_ts_ns = segments
        .iter()
        .filter_map(|(_, m)| m.end_ts.map(|t| t.timestamp_nanos_opt().unwrap_or(0)))
        .max()
        .unwrap_or(0);

    // Build tournament manifest
    let manifest = TournamentManifestV1 {
        schema_version: TOURNAMENT_MANIFEST_SCHEMA.to_string(),
        tournament_id: tournament_id.clone(),
        created_ts_ns,
        preset: config.preset.clone(),
        input_segments,
        strategies: strategy_names.clone(),
        symbols: config.symbols.clone(),
        runs,
        artifact_digests,
        bundle_digest,
    };

    // Build leaderboard
    let leaderboard = build_leaderboard(&manifest, &run_results, &preset);

    // Write outputs
    let manifest_path = tournament_dir.join("tournament_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, &manifest_json)?;

    let leaderboard_json_path = tournament_dir.join("leaderboard.json");
    let leaderboard_json = serde_json::to_string_pretty(&leaderboard)?;
    std::fs::write(&leaderboard_json_path, &leaderboard_json)?;

    let leaderboard_md_path = tournament_dir.join("leaderboard.md");
    let leaderboard_md = render_leaderboard_markdown(&leaderboard);
    std::fs::write(&leaderboard_md_path, &leaderboard_md)?;

    info!("Tournament complete: {}", tournament_id);
    info!("  Runs: {}", run_results.len());
    info!("  Meaningful: {}", leaderboard.meaningful_runs);
    info!("  Output: {:?}", tournament_dir);

    Ok(TournamentResult {
        tournament_id,
        preset: config.preset,
        segment_count: segments.len(),
        strategy_count: strategy_names.len(),
        total_runs: run_results.len(),
        meaningful_runs: leaderboard.meaningful_runs as usize,
        output_dir: tournament_dir.to_string_lossy().to_string(),
        manifest_path: manifest_path.to_string_lossy().to_string(),
        leaderboard_json_path: leaderboard_json_path.to_string_lossy().to_string(),
        leaderboard_md_path: leaderboard_md_path.to_string_lossy().to_string(),
    })
}

// =============================================================================
// Single Evaluation
// =============================================================================

/// Run a single strategy evaluation on a segment.
async fn run_single_evaluation(
    segment_dir: &Path,
    segment_manifest: &SegmentManifest,
    strategy_name: &str,
    registry: &StrategyRegistry,
    runs_dir: &Path,
    config: &TournamentConfig,
) -> Result<SingleRunResult> {
    // Create strategy instance
    let strategy = registry.create(strategy_name, None)?;
    let strategy_id = strategy.strategy_id();

    // Generate deterministic IDs
    let run_id = generate_run_id(&segment_manifest.segment_id, &strategy_id);
    let run_key = generate_run_key(&segment_manifest.segment_id, strategy_name);

    // Create run output directory
    let run_dir = runs_dir.join(&run_key);
    std::fs::create_dir_all(&run_dir)?;

    // Configure backtest
    let backtest_config = BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps: config.fee_bps_f64,
            initial_cash: config.initial_capital_f64,
            use_perp_prices: true,
        },
        log_interval: 1_000_000,
        pace: PaceMode::Fast,
        output_trace: Some(
            run_dir
                .join("decision_trace.json")
                .to_string_lossy()
                .to_string(),
        ),
        run_id: Some(run_id.clone()),
        params_json: None, // Single evaluation, no grid params
        enforce_admission_from_wal: false,
        admission_mismatch_policy: "fail".to_string(),
        strategy_spec: None, // Phase 22C: No permission gating in tournament mode
        enforcement: EnforcementConfig::default(), // Phase 22E: Dev mode for tournaments
        cost_model_path: None, // Phase 25A: No cost model for tournaments (use fill.fee as-is)
        latency_ticks: 0,    // Phase 25B: No latency in tournament mode (immediate execution)
        flatten_on_end: false, // Tournament tracks MTM, not realized
    };

    let engine = BacktestEngine::new(backtest_config);

    // Run backtest
    let (result, _strategy_binding) = engine
        .run_with_strategy(segment_dir, strategy, None)
        .await?;

    // Build attribution summary
    let pnl_exponent = -8i8;
    let mut builder = AttributionSummaryBuilder::new(
        strategy_id.clone(),
        run_id.clone(),
        segment_manifest.symbols.clone(),
        pnl_exponent,
    );

    // Add metrics from backtest result
    builder.add_metrics(quantlaxmi_models::BacktestMetrics {
        decisions: result.total_decisions as u32,
        fills: result.total_fills as u32,
        winning: result.metrics.winning_trades as u32,
        losing: (result.metrics.total_trades - result.metrics.winning_trades) as u32,
        net_pnl_mantissa: (result.realized_pnl * 100_000_000.0) as i128,
        max_loss_mantissa: (result.metrics.largest_loss.abs() * 100_000_000.0) as i128,
        round_trips: (result.total_fills / 2) as u32,
    });

    let generated_ts_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let summary = builder.build(generated_ts_ns);

    // Compute alpha score
    let alpha_score = AlphaScoreV1::from_summary(&summary);

    // Evaluate G1 gate
    let g1_gate = G1PromotionGate::new();
    let g1_result = g1_gate.evaluate(&summary, &alpha_score);

    // Write attribution summary
    let summary_path = run_dir.join("attribution_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, &summary_json)?;

    // Write alpha score
    let alpha_path = run_dir.join("alpha_score.json");
    let alpha_json = serde_json::to_string_pretty(&alpha_score)?;
    std::fs::write(&alpha_path, &alpha_json)?;

    // Write G1 result
    let g1_path = run_dir.join("g1_result.json");
    let g1_json = serde_json::to_string_pretty(&g1_result)?;
    std::fs::write(&g1_path, &g1_json)?;

    // Write run manifest
    let run_manifest = RunManifestV1 {
        schema_version: RUN_MANIFEST_SCHEMA.to_string(),
        run_id: run_id.clone(),
        run_key: run_key.clone(),
        strategy_id: strategy_id.clone(),
        segment_id: segment_manifest.segment_id.clone(),
        segment_path: segment_dir.to_string_lossy().to_string(),
        symbols: segment_manifest.symbols.clone(),
        decisions: summary.total_decisions,
        fills: summary.total_fills,
        round_trips: summary.round_trips,
        win_rate_bps: summary.win_rate_bps,
        net_pnl_mantissa: summary.total_net_pnl_mantissa,
        pnl_exponent: summary.pnl_exponent,
        alpha_score_mantissa: alpha_score.score_mantissa as i64,
        alpha_score_exponent: alpha_score.score_exponent,
        g1_passed: g1_result.passed,
        g1_reasons: g1_result.reasons.clone(),
        g2_passed: None,
        g2_reasons: Vec::new(),
        g3_passed: None,
        g3_reasons: Vec::new(),
        decision_trace_sha256: result.trace_hash.clone(),
        attribution_summary_sha256: compute_string_sha256(&summary_json),
        alpha_score_sha256: compute_string_sha256(&alpha_json),
        g1_result_sha256: compute_string_sha256(&g1_json),
        g2_result_sha256: None,
        g3_result_sha256: None,
        derived_ts_ns: generated_ts_ns,
    };

    let run_manifest_path = run_dir.join("run_manifest.json");
    let run_manifest_json = serde_json::to_string_pretty(&run_manifest)?;
    std::fs::write(&run_manifest_path, &run_manifest_json)?;

    // Compute artifact digests
    let mut artifact_digests = BTreeMap::new();
    for (name, path) in [
        ("decision_trace.json", run_dir.join("decision_trace.json")),
        ("attribution_summary.json", summary_path.clone()),
        ("alpha_score.json", alpha_path.clone()),
        ("g1_result.json", g1_path.clone()),
        ("run_manifest.json", run_manifest_path.clone()),
    ] {
        if path.exists() {
            let data = std::fs::read(&path)?;
            artifact_digests.insert(name.to_string(), ArtifactDigest::from_bytes(&data));
        }
    }

    // Build result paths (relative to run dir)
    let result_paths = RunResultPaths {
        decision_trace: "decision_trace.json".to_string(),
        attribution_summary: "attribution_summary.json".to_string(),
        alpha_score: "alpha_score.json".to_string(),
        g1_result: "g1_result.json".to_string(),
        g2_result: None,
        g3_result: None,
        run_manifest: "run_manifest.json".to_string(),
    };

    Ok(SingleRunResult {
        run_id,
        run_key,
        segment_id: segment_manifest.segment_id.clone(),
        strategy_id,
        strategy_name: strategy_name.to_string(),
        summary,
        alpha_score,
        g1_result,
        trace_hash: result.trace_hash,
        result_paths,
        artifact_digests,
    })
}

// =============================================================================
// Leaderboard Building
// =============================================================================

/// Build leaderboard from tournament results.
fn build_leaderboard(
    manifest: &TournamentManifestV1,
    results: &[SingleRunResult],
    preset: &TournamentPreset,
) -> LeaderboardV1 {
    // Create leaderboard rows
    let mut rows: Vec<LeaderboardRow> = results
        .iter()
        .filter(|r| {
            is_meaningful_run(
                r.summary.total_decisions,
                r.summary.total_fills,
                r.summary.round_trips,
                preset,
            )
        })
        .map(|r| LeaderboardRow {
            rank: 0, // Will be set after sorting
            strategy_id: r.strategy_id.clone(),
            strategy_name: r.strategy_name.clone(),
            segment_id: r.segment_id.clone(),
            run_key: r.run_key.clone(),
            alpha_score_mantissa: r.alpha_score.score_mantissa as i64,
            alpha_score_exponent: r.alpha_score.score_exponent,
            alpha_score_f64: r.alpha_score.score_f64(),
            g1_passed: r.g1_result.passed,
            g2_passed: None,
            g3_passed: None,
            decisions: r.summary.total_decisions,
            fills: r.summary.total_fills,
            round_trips: r.summary.round_trips,
            win_rate_bps: r.summary.win_rate_bps,
            net_pnl_mantissa: r.summary.total_net_pnl_mantissa,
            pnl_exponent: r.summary.pnl_exponent,
        })
        .collect();

    // Sort by ranking criteria
    rows.sort_by(compare_rows);

    // Assign ranks
    for (i, row) in rows.iter_mut().enumerate() {
        row.rank = (i + 1) as u32;
    }

    LeaderboardV1 {
        schema_version: LEADERBOARD_SCHEMA.to_string(),
        tournament_id: manifest.tournament_id.clone(),
        rows,
        total_runs: manifest.runs.len() as u32,
        meaningful_runs: results
            .iter()
            .filter(|r| {
                is_meaningful_run(
                    r.summary.total_decisions,
                    r.summary.total_fills,
                    r.summary.round_trips,
                    preset,
                )
            })
            .count() as u32,
        created_ts_ns: manifest.created_ts_ns,
    }
}

/// Render leaderboard as markdown table.
fn render_leaderboard_markdown(leaderboard: &LeaderboardV1) -> String {
    let mut md = String::new();

    md.push_str(&format!(
        "# Tournament Leaderboard\n\n**Tournament ID:** `{}`\n\n",
        &leaderboard.tournament_id[..16]
    ));

    md.push_str(&format!(
        "**Total Runs:** {} | **Meaningful Runs:** {}\n\n",
        leaderboard.total_runs, leaderboard.meaningful_runs
    ));

    // Header
    md.push_str("| Rank | Strategy | Segment | Alpha | Win% | G1 | Decisions | Fills | PnL |\n");
    md.push_str("|------|----------|---------|-------|------|----|-----------:|------:|----:|\n");

    // Rows
    for row in &leaderboard.rows {
        let g1_status = if row.g1_passed { "✓" } else { "✗" };
        let win_pct = row.win_rate_bps as f64 / 100.0;
        let pnl_f64 = row.net_pnl_mantissa as f64 * 10f64.powi(row.pnl_exponent as i32);

        md.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.1}% | {} | {} | {} | ${:.2} |\n",
            row.rank,
            row.strategy_name,
            &row.segment_id[..8.min(row.segment_id.len())],
            row.alpha_score_f64,
            win_pct,
            g1_status,
            row.decisions,
            row.fills,
            pnl_f64,
        ));
    }

    md
}

// =============================================================================
// Validation
// =============================================================================

/// Validation result for tournament artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Tournament ID
    pub tournament_id: String,
    /// Validation errors (empty if valid)
    pub errors: Vec<String>,
    /// Warnings (non-fatal issues)
    pub warnings: Vec<String>,
}

/// Validate a tournament directory.
///
/// Checks:
/// - Manifest integrity
/// - Artifact digests match
/// - No undeclared files
/// - Leaderboard consistency
pub fn validate_tournament(tournament_dir: &Path) -> Result<ValidationResult> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Load manifest
    let manifest_path = tournament_dir.join("tournament_manifest.json");
    if !manifest_path.exists() {
        return Ok(ValidationResult {
            valid: false,
            tournament_id: "UNKNOWN".to_string(),
            errors: vec!["tournament_manifest.json not found".to_string()],
            warnings: Vec::new(),
        });
    }

    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: TournamentManifestV1 =
        serde_json::from_str(&manifest_content).with_context(|| "parse tournament manifest")?;

    // Verify schema version
    if manifest.schema_version != TOURNAMENT_MANIFEST_SCHEMA {
        errors.push(format!(
            "Schema version mismatch: expected {}, got {}",
            TOURNAMENT_MANIFEST_SCHEMA, manifest.schema_version
        ));
    }

    // Verify artifact digests
    for (artifact_path, expected_digest) in &manifest.artifact_digests {
        let full_path = tournament_dir.join(artifact_path);
        if !full_path.exists() {
            errors.push(format!("Missing artifact: {}", artifact_path));
            continue;
        }

        let data = std::fs::read(&full_path)?;
        let actual_digest = ArtifactDigest::from_bytes(&data);

        if actual_digest.sha256 != expected_digest.sha256 {
            errors.push(format!(
                "Digest mismatch for {}: expected {}, got {}",
                artifact_path,
                &expected_digest.sha256[..16],
                &actual_digest.sha256[..16]
            ));
        }

        if actual_digest.bytes != expected_digest.bytes {
            errors.push(format!(
                "Size mismatch for {}: expected {} bytes, got {} bytes",
                artifact_path, expected_digest.bytes, actual_digest.bytes
            ));
        }
    }

    // Check for undeclared files in runs/
    let runs_dir = tournament_dir.join("runs");
    if runs_dir.exists() {
        for entry in walkdir::WalkDir::new(&runs_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let rel_path = entry
                .path()
                .strip_prefix(tournament_dir)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();

            if !manifest.artifact_digests.contains_key(&rel_path) {
                warnings.push(format!("Undeclared file: {}", rel_path));
            }
        }
    }

    // Verify bundle digest
    let computed_bundle = compute_bundle_digest(&manifest.artifact_digests);
    if computed_bundle != manifest.bundle_digest {
        errors.push(format!(
            "Bundle digest mismatch: expected {}, got {}",
            &manifest.bundle_digest[..16],
            &computed_bundle[..16]
        ));
    }

    // Verify leaderboard
    let leaderboard_path = tournament_dir.join("leaderboard.json");
    if leaderboard_path.exists() {
        let leaderboard_content = std::fs::read_to_string(&leaderboard_path)?;
        let leaderboard: LeaderboardV1 = serde_json::from_str(&leaderboard_content)?;

        if leaderboard.tournament_id != manifest.tournament_id {
            errors.push(format!(
                "Leaderboard tournament_id mismatch: expected {}, got {}",
                &manifest.tournament_id[..16],
                &leaderboard.tournament_id[..16]
            ));
        }
    } else {
        warnings.push("leaderboard.json not found".to_string());
    }

    Ok(ValidationResult {
        valid: errors.is_empty(),
        tournament_id: manifest.tournament_id,
        errors,
        warnings,
    })
}

// =============================================================================
// Helpers
// =============================================================================

/// Compute SHA-256 of a file.
fn compute_file_sha256(path: &Path) -> Result<String> {
    let data = std::fs::read(path)?;
    let hash = Sha256::digest(&data);
    Ok(hex::encode(hash))
}

/// Compute SHA-256 of a string.
fn compute_string_sha256(s: &str) -> String {
    let hash = Sha256::digest(s.as_bytes());
    hex::encode(hash)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tournament_id_determinism() {
        let id1 = generate_tournament_id(
            "baseline_v1",
            &["digest1".to_string(), "digest2".to_string()],
            &["funding_bias".to_string(), "micro_breakout".to_string()],
        );
        let id2 = generate_tournament_id(
            "baseline_v1",
            &["digest1".to_string(), "digest2".to_string()],
            &["funding_bias".to_string(), "micro_breakout".to_string()],
        );
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_run_key_format() {
        let key = generate_run_key("perp_20260125_120000", "funding_bias");
        assert_eq!(key, "perp_20260125_120000__funding_bias");
    }

    #[test]
    fn test_leaderboard_ranking() {
        let _preset = TournamentPreset::baseline_v1();

        // Create test rows with different alpha scores
        let row_high = LeaderboardRow {
            rank: 0,
            strategy_id: "high:1.0:abc".to_string(),
            strategy_name: "high".to_string(),
            segment_id: "seg1".to_string(),
            run_key: "seg1__high".to_string(),
            alpha_score_mantissa: 5000,
            alpha_score_exponent: -4,
            alpha_score_f64: 0.5,
            g1_passed: true,
            g2_passed: None,
            g3_passed: None,
            decisions: 100,
            fills: 50,
            round_trips: 10,
            win_rate_bps: 6000,
            net_pnl_mantissa: 1000000,
            pnl_exponent: -8,
        };

        let row_low = LeaderboardRow {
            alpha_score_mantissa: 1000,
            alpha_score_f64: 0.1,
            strategy_name: "low".to_string(),
            run_key: "seg1__low".to_string(),
            ..row_high.clone()
        };

        // High alpha should rank first (Less in sort order)
        assert_eq!(compare_rows(&row_high, &row_low), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_meaningful_run_filter() {
        let preset = TournamentPreset::baseline_v1();

        // Meaningful: meets all thresholds
        assert!(is_meaningful_run(10, 5, 2, &preset));

        // Not meaningful: no decisions
        assert!(!is_meaningful_run(0, 5, 2, &preset));

        // Not meaningful: no fills
        assert!(!is_meaningful_run(10, 0, 2, &preset));
    }
}

// =============================================================================
// P1: Grid-Based Tournament Runner
// =============================================================================

/// Parameter value in grid (normalized for deterministic iteration).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{}", v),
            ParamValue::Float(v) => write!(f, "{}", v),
            ParamValue::Bool(v) => write!(f, "{}", v),
            ParamValue::String(v) => write!(f, "{}", v),
        }
    }
}

// =============================================================================
// P1.1: Diagnostics for Zero-Trade Investigation
// =============================================================================

/// Event counts by type for diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCounts {
    pub depth: usize,
    pub trades: usize,
    pub funding: usize,
    pub spot: usize,
    pub total: usize,
}

/// Signal counts for diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalCounts {
    /// Number of signals generated by strategy
    pub generated: usize,
    /// Number of signals admitted (passed gates)
    pub admitted: usize,
    /// Number of signals refused (blocked by gates)
    pub refused: usize,
}

/// Order/fill counts for diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderCounts {
    /// Number of orders submitted to exchange
    pub submitted: usize,
    /// Number of orders filled
    pub filled: usize,
}

/// Reason why a run produced zero trades.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Default)]
pub enum ZeroTradeReason {
    /// Run was not zero-trade (had fills)
    NotZeroTrade,
    /// No events of required type (e.g., no funding events for funding_bias)
    NoRelevantEvents,
    /// Events exist but strategy generated no signals
    NoSignals,
    /// Signals generated but all refused by admission gates
    AllSignalsRefused,
    /// Signals admitted but no orders filled (liquidity issue)
    NoFills,
    /// Unknown reason (needs investigation)
    #[default]
    Unknown,
}

// =============================================================================
// P2: Run Summary for SLR Sensitivity Analysis
// =============================================================================

/// Compact run summary for sensitivity analysis (run_summary.json).
/// Contains all key metrics in one file for easy comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Schema version
    pub schema_version: String,
    /// Parameter values for this run
    pub params: BTreeMap<String, ParamValue>,
    /// Number of round-trips (trades closed)
    pub trades_closed: usize,
    /// Number of decisions (signals)
    pub decisions: usize,
    /// Gross PnL before fees
    pub gross_pnl: f64,
    /// Total fees paid
    pub fees: f64,
    /// Net PnL after fees
    pub net_pnl: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Win rate percentage (0-100)
    pub win_rate: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Trace hash (for audit)
    pub wal_digest: String,
    /// Guardrail: passed minimum trades?
    pub guardrail_min_trades_ok: bool,
    /// Guardrail: passed max drawdown?
    pub guardrail_max_dd_ok: bool,
    /// Overall guardrails passed?
    pub guardrails_passed: bool,
}

/// Guardrails configuration for sensitivity analysis.
#[derive(Debug, Clone)]
pub struct Guardrails {
    /// Minimum number of trades to be considered valid
    pub min_trades: usize,
    /// Maximum drawdown percentage (0.0-1.0) to be considered valid
    pub max_drawdown_pct: f64,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            min_trades: 10,
            max_drawdown_pct: 0.05, // 5%
        }
    }
}

/// Diagnostic data for a single run (P1.1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunDiagnostics {
    /// Event counts by type
    pub events: EventCounts,
    /// Signal counts
    pub signals: SignalCounts,
    /// Order/fill counts
    pub orders: OrderCounts,
    /// Reason for zero trades (if applicable)
    pub zero_trade_reason: ZeroTradeReason,
    /// Exit reason breakdown (e.g., "exit_long_stop_loss": 2)
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub exit_reasons: BTreeMap<String, usize>,
    /// Strategy-specific notes (e.g., "funding rate never exceeded threshold")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl RunDiagnostics {
    /// Determine zero-trade reason from counts.
    pub fn compute_zero_trade_reason(&mut self) {
        if self.orders.filled > 0 {
            self.zero_trade_reason = ZeroTradeReason::NotZeroTrade;
        } else if self.events.total == 0 {
            self.zero_trade_reason = ZeroTradeReason::NoRelevantEvents;
        } else if self.signals.generated == 0 {
            self.zero_trade_reason = ZeroTradeReason::NoSignals;
        } else if self.signals.admitted == 0 {
            self.zero_trade_reason = ZeroTradeReason::AllSignalsRefused;
        } else if self.orders.submitted > 0 && self.orders.filled == 0 {
            self.zero_trade_reason = ZeroTradeReason::NoFills;
        } else {
            self.zero_trade_reason = ZeroTradeReason::Unknown;
        }
    }
}

/// Grid tournament configuration.
#[derive(Debug, Clone)]
pub struct GridTournamentConfig {
    pub segments_root: PathBuf,
    pub segment_patterns: Vec<String>,
    pub strategy: String,
    pub grid_path: PathBuf,
    pub out_dir: PathBuf,
    pub initial_capital: f64,
    pub emit_traces: bool,
    pub max_runs: usize,
}

/// Single run result for grid tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridRunResult {
    pub segment: String,
    pub strategy: String,
    pub param_hash: String,
    pub params: BTreeMap<String, ParamValue>,
    pub run_dir: String,
    pub metrics: crate::backtest::BacktestMetricsV1,
    pub manifest: crate::backtest::BacktestRunManifestV1,
    /// P1.1: Diagnostic data for debugging zero-trade runs
    pub diagnostics: RunDiagnostics,
    /// P2: Run summary for sensitivity analysis
    pub summary: RunSummary,
}

/// Leaderboard entry aggregated across segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub param_hash: String,
    pub params: BTreeMap<String, ParamValue>,
    pub score: f64,
    pub mean_total_return: f64,
    pub median_total_return: f64,
    pub mean_max_drawdown: f64,
    pub segments: usize,
    pub positive_fraction: f64,
}

/// Promotion candidates output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCandidates {
    pub generated_at_rfc3339: String,
    pub strategy: String,
    pub candidates: Vec<LeaderboardEntry>,
}

/// Tournament manifest for audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridTournamentManifest {
    pub schema_version: String,
    pub strategy: String,
    pub segments_root: String,
    pub matched_segments: Vec<String>,
    pub grid_file_hash: String,
    pub total_runs: usize,
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
}

/// Parse grid TOML and expand to parameter combinations.
///
/// Supports nested tables via `[params.subtable]` which become dot-notated keys
/// (e.g., `[params.grassmann]` with `enabled = [true]` becomes `grassmann.enabled`).
fn parse_and_expand_grid(grid_path: &Path) -> Result<Vec<BTreeMap<String, ParamValue>>> {
    let content = std::fs::read_to_string(grid_path)
        .with_context(|| format!("Failed to read grid file: {:?}", grid_path))?;

    let parsed: toml::Value = toml::from_str(&content)
        .with_context(|| format!("Failed to parse grid TOML: {:?}", grid_path))?;

    let params_table = parsed
        .get("params")
        .and_then(|v| v.as_table())
        .ok_or_else(|| anyhow::anyhow!("Grid TOML must have [params] table"))?;

    // Recursively collect all parameters (handles nested tables)
    let mut param_values: Vec<(String, Vec<ParamValue>)> = Vec::new();
    collect_params_recursive(params_table, "", &mut param_values)?;

    // Sort for determinism
    param_values.sort_by(|a, b| a.0.cmp(&b.0));

    // Cartesian product expansion
    let mut combinations: Vec<BTreeMap<String, ParamValue>> = vec![BTreeMap::new()];

    for (name, values) in param_values {
        let mut new_combinations = Vec::new();
        for combo in &combinations {
            for value in &values {
                let mut new_combo = combo.clone();
                new_combo.insert(name.clone(), value.clone());
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
    }

    Ok(combinations)
}

/// Recursively collect parameters from a TOML table, handling nested tables.
fn collect_params_recursive(
    table: &toml::map::Map<String, toml::Value>,
    prefix: &str,
    out: &mut Vec<(String, Vec<ParamValue>)>,
) -> Result<()> {
    for (name, value) in table {
        let full_key = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{}.{}", prefix, name)
        };

        match value {
            toml::Value::Array(arr) => {
                // This is a parameter with values to sweep
                let parsed_values: Vec<ParamValue> = arr
                    .iter()
                    .map(|v| match v {
                        toml::Value::Integer(i) => ParamValue::Int(*i),
                        toml::Value::Float(f) => ParamValue::Float(*f),
                        toml::Value::Boolean(b) => ParamValue::Bool(*b),
                        toml::Value::String(s) => ParamValue::String(s.clone()),
                        _ => ParamValue::String(v.to_string()),
                    })
                    .collect();
                out.push((full_key, parsed_values));
            }
            toml::Value::Table(subtable) => {
                // Recurse into nested table
                collect_params_recursive(subtable, &full_key, out)?;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Parameter '{}' must be an array or table, found: {:?}",
                    full_key,
                    value.type_str()
                ));
            }
        }
    }
    Ok(())
}

/// Compute SHA256 hash for parameter set (deterministic).
fn compute_param_hash(params: &BTreeMap<String, ParamValue>) -> String {
    let json = serde_json::to_string(params).unwrap_or_default();
    let hash = Sha256::digest(json.as_bytes());
    hex::encode(&hash[..8]) // Use first 8 bytes = 16 hex chars
}

/// Discover segments matching glob patterns.
fn discover_segments(segments_root: &Path, patterns: &[String]) -> Result<Vec<PathBuf>> {
    let mut matched: Vec<PathBuf> = Vec::new();

    // Read directory entries
    let entries: Vec<_> = std::fs::read_dir(segments_root)
        .with_context(|| format!("Failed to read segments_root: {:?}", segments_root))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();

    for entry in entries {
        let dir_name = entry.file_name().to_string_lossy().to_string();

        // Check if dir name matches any pattern
        for pattern in patterns {
            if glob::Pattern::new(pattern)
                .map(|p| p.matches(&dir_name))
                .unwrap_or(false)
            {
                let full_path = entry.path();
                // Verify segment_manifest.json exists
                if full_path.join("segment_manifest.json").exists() {
                    matched.push(full_path);
                    break;
                }
            }
        }
    }

    // Sort lexicographically for determinism
    matched.sort();

    if matched.is_empty() {
        return Err(anyhow::anyhow!(
            "No segments matched patterns {:?} in {:?}",
            patterns,
            segments_root
        ));
    }

    Ok(matched)
}

/// Write strategy config TOML for a parameter set.
///
/// Supports nested config via dot notation (e.g., `grassmann.enabled` becomes `[grassmann]\nenabled = ...`).
/// Fields without dots are written at root level.
fn write_param_config(
    config_dir: &Path,
    param_hash: &str,
    params: &BTreeMap<String, ParamValue>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(config_dir)?;

    let config_path = config_dir.join(format!("{}.toml", param_hash));

    // Separate root-level and nested keys
    let mut root_params: BTreeMap<String, &ParamValue> = BTreeMap::new();
    let mut nested_params: BTreeMap<String, BTreeMap<String, &ParamValue>> = BTreeMap::new();

    for (key, value) in params {
        if let Some(dot_pos) = key.find('.') {
            let table_name = &key[..dot_pos];
            let field_name = &key[dot_pos + 1..];
            nested_params
                .entry(table_name.to_string())
                .or_default()
                .insert(field_name.to_string(), value);
        } else {
            root_params.insert(key.clone(), value);
        }
    }

    // Build TOML content
    let mut content = String::new();

    // Write root-level fields first
    for (key, value) in &root_params {
        write_toml_value(&mut content, key, value);
    }

    // Write nested tables
    for (table_name, fields) in &nested_params {
        content.push_str(&format!("\n[{}]\n", table_name));
        for (key, value) in fields {
            write_toml_value(&mut content, key, value);
        }
    }

    std::fs::write(&config_path, content)?;
    Ok(config_path)
}

/// Write a single TOML key = value line.
fn write_toml_value(content: &mut String, key: &str, value: &ParamValue) {
    match value {
        ParamValue::Int(v) => content.push_str(&format!("{} = {}\n", key, v)),
        ParamValue::Float(v) => content.push_str(&format!("{} = {}\n", key, v)),
        ParamValue::Bool(v) => content.push_str(&format!("{} = {}\n", key, v)),
        ParamValue::String(v) => content.push_str(&format!("{} = \"{}\"\n", key, v)),
    }
}

/// Type alias for run task tuple (segment_path, segment_name, param_hash, params, config_path).
type RunTask = (
    PathBuf,
    String,
    String,
    BTreeMap<String, ParamValue>,
    PathBuf,
);

/// Run grid-based tournament CLI.
#[allow(clippy::too_many_arguments)]
pub async fn run_tournament_grid_cli(
    segments_root: &str,
    segments_pattern: &str,
    strategy: &str,
    grid_path: &str,
    out_dir: &str,
    initial_capital: f64,
    emit_traces: bool,
    max_runs: usize,
    parallel: usize,
) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Determine number of parallel workers
    let num_workers = if parallel == 0 {
        // Auto-detect: use half of CPU cores to leave headroom
        (num_cpus::get() / 2).max(1)
    } else {
        parallel
    };

    info!("=== P1 Grid Tournament (Parallel) ===");
    info!("Segments root: {}", segments_root);
    info!("Strategy: {}", strategy);
    info!("Grid: {}", grid_path);
    info!("Parallel workers: {}", num_workers);

    let segments_root_path = PathBuf::from(segments_root);
    let grid_path_buf = PathBuf::from(grid_path);
    let out_dir_path = PathBuf::from(out_dir);

    // Parse segment patterns
    let patterns: Vec<String> = segments_pattern
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Discover segments
    let segments = discover_segments(&segments_root_path, &patterns)?;
    info!("Matched {} segments", segments.len());

    // Parse and expand grid
    let param_sets = parse_and_expand_grid(&grid_path_buf)?;
    info!(
        "Grid expanded to {} parameter combinations",
        param_sets.len()
    );

    // Safety check: max_runs
    let total_runs = segments.len() * param_sets.len();
    if total_runs > max_runs {
        return Err(anyhow::anyhow!(
            "Total runs ({} segments × {} params = {}) exceeds max_runs ({}). \
             Reduce grid size or increase --max-runs",
            segments.len(),
            param_sets.len(),
            total_runs,
            max_runs
        ));
    }

    info!("Total runs to execute: {}", total_runs);

    // Create output directories
    std::fs::create_dir_all(&out_dir_path)?;
    let configs_dir = out_dir_path.join("configs");
    let runs_dir = out_dir_path.join("runs");
    std::fs::create_dir_all(&configs_dir)?;
    std::fs::create_dir_all(&runs_dir)?;

    // Generate config files for each param set
    let mut param_configs: Vec<(String, BTreeMap<String, ParamValue>, PathBuf)> = Vec::new();
    for params in &param_sets {
        let param_hash = compute_param_hash(params);
        let config_path = write_param_config(&configs_dir, &param_hash, params)?;
        param_configs.push((param_hash, params.clone(), config_path));
    }

    // Compute grid file hash for manifest
    let grid_content = std::fs::read(&grid_path_buf)?;
    let grid_hash = hex::encode(Sha256::digest(&grid_content));

    // Get git info (optional)
    let (git_commit, git_dirty) = get_git_info();

    // Collect segment names
    let segment_names: Vec<String> = segments
        .iter()
        .map(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        })
        .collect();

    // Validate strategy exists (use a temporary registry for validation)
    {
        let registry = quantlaxmi_strategy::StrategyRegistry::with_builtins();
        if !registry.contains(strategy) {
            return Err(anyhow::anyhow!(
                "Unknown strategy: '{}'. Available: {:?}",
                strategy,
                registry.list()
            ));
        }
    }

    // Configure rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()
        .context("Failed to create rayon thread pool")?;

    // Build list of all run tasks
    let mut run_tasks: Vec<RunTask> = Vec::new();
    for segment_path in &segments {
        let segment_name = segment_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        for (param_hash, params, config_path) in &param_configs {
            run_tasks.push((
                segment_path.clone(),
                segment_name.clone(),
                param_hash.clone(),
                params.clone(),
                config_path.clone(),
            ));
        }
    }

    // Create all output directories upfront (sequential - fast)
    for (_, segment_name, param_hash, _, _) in &run_tasks {
        let run_dir = runs_dir.join(segment_name).join(param_hash);
        std::fs::create_dir_all(&run_dir)?;
    }

    // P2: Create guardrails for sensitivity analysis
    let guardrails = Guardrails::default();

    // Progress tracking
    let completed = AtomicUsize::new(0);
    let results_mutex: Mutex<Vec<GridRunResult>> = Mutex::new(Vec::new());

    // Shared config for closures
    let strategy_str = strategy.to_string();
    let emit_traces_val = emit_traces;
    let initial_capital_val = initial_capital;
    let runs_dir_clone = runs_dir.clone();

    // Run all backtests in parallel
    info!(
        "Starting {} runs across {} workers...",
        total_runs, num_workers
    );

    pool.install(|| {
        run_tasks.par_iter().for_each(
            |(segment_path, segment_name, param_hash, params, config_path)| {
                let run_dir = runs_dir_clone.join(segment_name).join(param_hash);

                // Create a mini tokio runtime for this thread's async work
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime");

                // Each thread gets its own registry (they're cheap to create)
                let registry = quantlaxmi_strategy::StrategyRegistry::with_builtins();

                let result = rt.block_on(async {
                    run_single_grid_backtest(
                        segment_path,
                        &strategy_str,
                        config_path,
                        &run_dir,
                        initial_capital_val,
                        emit_traces_val,
                        &registry,
                        params,
                        &guardrails,
                    )
                    .await
                });

                // Update progress
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(5) || done == total_runs {
                    info!("[{}/{}] completed", done, total_runs);
                }

                // Collect result
                match result {
                    Ok((metrics, manifest, diagnostics, summary)) => {
                        let relative_run_dir = format!("runs/{}/{}", segment_name, param_hash);
                        let grid_result = GridRunResult {
                            segment: segment_name.clone(),
                            strategy: strategy_str.clone(),
                            param_hash: param_hash.clone(),
                            params: params.clone(),
                            run_dir: relative_run_dir,
                            metrics,
                            manifest,
                            diagnostics,
                            summary,
                        };
                        results_mutex.lock().unwrap().push(grid_result);
                    }
                    Err(e) => {
                        warn!("Run failed: {} / {}: {}", segment_name, param_hash, e);
                    }
                }
            },
        );
    });

    // Extract results
    let results = results_mutex.into_inner().unwrap();
    info!("Completed {} / {} runs", results.len(), total_runs);

    // Write results.jsonl
    let results_path = out_dir_path.join("results.jsonl");
    write_results_jsonl(&results_path, &results)?;
    info!("Wrote {}", results_path.display());

    // Build and write leaderboard
    let leaderboard = build_grid_leaderboard(&results);
    let leaderboard_path = out_dir_path.join("leaderboard.json");
    let leaderboard_json = serde_json::to_string_pretty(&leaderboard)?;
    std::fs::write(&leaderboard_path, leaderboard_json)?;
    info!("Wrote {}", leaderboard_path.display());

    // Build and write promotion candidates
    let candidates = select_promotion_candidates(&leaderboard, strategy, 10);
    let candidates_path = out_dir_path.join("promotion_candidates.json");
    let candidates_json = serde_json::to_string_pretty(&candidates)?;
    std::fs::write(&candidates_path, candidates_json)?;
    info!("Wrote {}", candidates_path.display());

    // Write tournament manifest
    let manifest = GridTournamentManifest {
        schema_version: "v1".to_string(),
        strategy: strategy.to_string(),
        segments_root: segments_root.to_string(),
        matched_segments: segment_names,
        grid_file_hash: grid_hash,
        total_runs: results.len(),
        git_commit,
        git_dirty,
    };
    let manifest_path = out_dir_path.join("tournament_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;
    info!("Wrote {}", manifest_path.display());

    info!("=== Tournament Complete ===");
    info!("  Runs: {}", results.len());
    info!("  Leaderboard entries: {}", leaderboard.len());
    info!("  Promotion candidates: {}", candidates.candidates.len());
    info!("  Output: {}", out_dir_path.display());

    // P2: Print topline ranking table
    print_topline_ranking(&results, &guardrails);

    Ok(())
}

/// Run single backtest for grid tournament.
#[allow(clippy::too_many_arguments)]
async fn run_single_grid_backtest(
    segment_dir: &Path,
    strategy: &str,
    config_path: &Path,
    run_dir: &Path,
    initial_capital: f64,
    emit_traces: bool,
    registry: &quantlaxmi_strategy::StrategyRegistry,
    params: &BTreeMap<String, ParamValue>,
    guardrails: &Guardrails,
) -> Result<(
    crate::backtest::BacktestMetricsV1,
    crate::backtest::BacktestRunManifestV1,
    RunDiagnostics,
    RunSummary,
)> {
    use crate::backtest::{
        BacktestConfig, BacktestEngine, BacktestMetricsV1, BacktestRunManifestV1,
        EnforcementConfig, ExchangeConfig, PaceMode,
    };
    use crate::replay::{ReplayStats, SegmentReplayAdapter};

    // P1.1: Collect event stats from segment (before backtest consumes the iterator)
    let replay_stats = {
        let mut adapter = SegmentReplayAdapter::open(segment_dir)?;
        ReplayStats::from_adapter(&mut adapter)?
    };

    // Create strategy instance
    let strategy_box = registry.create(strategy, Some(config_path))?;

    // Configure backtest
    // Serialize params to canonical JSON for deterministic run_id computation
    let params_json = serde_json::to_string(params).ok();
    let config = BacktestConfig {
        exchange: ExchangeConfig {
            fee_bps: 10.0,
            initial_cash: initial_capital,
            use_perp_prices: true,
        },
        log_interval: 0, // Quiet mode
        pace: PaceMode::Fast,
        output_trace: None,
        run_id: None,
        params_json, // Grid params for deterministic run_id
        enforce_admission_from_wal: false,
        admission_mismatch_policy: "fail".to_string(),
        strategy_spec: None,
        enforcement: EnforcementConfig::default(),
        cost_model_path: None,
        latency_ticks: 0,
        flatten_on_end: false,
    };

    let engine = BacktestEngine::new(config);

    // Run backtest
    let (result, _) = engine
        .run_with_strategy(segment_dir, strategy_box, Some(config_path))
        .await?;

    // Convert to V1 schemas
    let metrics_v1 = BacktestMetricsV1::from_metrics(&result.metrics);
    let manifest_v1 = BacktestRunManifestV1::from_result(&result);

    // P1.1: Build diagnostics
    // Count exit reasons from fill tags
    let mut exit_reasons: BTreeMap<String, usize> = BTreeMap::new();
    for fill in &result.fills {
        if let Some(tag) = &fill.tag
            && tag.starts_with("exit_")
        {
            *exit_reasons.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let mut diagnostics = RunDiagnostics {
        events: EventCounts {
            depth: replay_stats.perp_events, // PerpQuote/PerpDepth
            trades: 0,                       // Not tracked separately yet
            funding: replay_stats.funding_events,
            spot: replay_stats.spot_events,
            total: replay_stats.total_events,
        },
        signals: SignalCounts {
            // total_decisions = signals generated by strategy
            generated: result.total_decisions,
            // In current implementation, all generated signals are admitted
            // (no admission gating in grid tournament mode)
            admitted: result.total_decisions,
            refused: 0,
        },
        orders: OrderCounts {
            // Each decision leads to an order attempt
            submitted: result.total_decisions,
            filled: result.total_fills,
        },
        zero_trade_reason: ZeroTradeReason::Unknown,
        exit_reasons,
        notes: None,
    };

    // Compute zero-trade reason
    diagnostics.compute_zero_trade_reason();

    // Add strategy-specific notes for funding_bias
    if strategy == "funding_bias" && result.total_fills == 0 {
        if replay_stats.funding_events == 0 {
            diagnostics.notes = Some("No funding events in segment".to_string());
        } else if result.total_decisions == 0 {
            diagnostics.notes =
                Some("Funding events present but rate never exceeded threshold".to_string());
        }
    }

    // Write metrics.json
    let metrics_path = run_dir.join("metrics.json");
    let metrics_json = serde_json::to_string_pretty(&metrics_v1)?;
    std::fs::write(&metrics_path, metrics_json)?;

    // Write run_manifest.json
    let manifest_path = run_dir.join("run_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest_v1)?;
    std::fs::write(&manifest_path, manifest_json)?;

    // P1.1: Write diagnostics.json
    let diagnostics_path = run_dir.join("diagnostics.json");
    let diagnostics_json = serde_json::to_string_pretty(&diagnostics)?;
    std::fs::write(&diagnostics_path, diagnostics_json)?;

    // Write P0 traces if requested
    if emit_traces {
        let equity_curve_path = run_dir.join("equity_curve.jsonl");
        crate::backtest::write_equity_curve_jsonl(&equity_curve_path, &result.equity_curve)?;

        let fills_path = run_dir.join("fills.jsonl");
        crate::backtest::write_fills_jsonl(&fills_path, &result.fills)?;
    }

    // P2: Build run summary with guardrails
    // Note: metrics.max_drawdown_pct is already in percentage form (e.g., 19.05 = 19.05%)
    // Convert to decimal for comparison with guardrails (which uses decimal form, e.g., 0.05 = 5%)
    let trades_closed = metrics_v1.total_trades;
    let max_dd_decimal = metrics_v1.max_drawdown_pct / 100.0; // Convert 19.05 -> 0.1905
    let guardrail_min_trades_ok = trades_closed >= guardrails.min_trades;
    let guardrail_max_dd_ok = max_dd_decimal <= guardrails.max_drawdown_pct;
    let guardrails_passed = guardrail_min_trades_ok && guardrail_max_dd_ok;

    let summary = RunSummary {
        schema_version: "v1".to_string(),
        params: params.clone(),
        trades_closed,
        decisions: result.total_decisions,
        gross_pnl: metrics_v1.gross_profit - metrics_v1.gross_loss.abs(),
        fees: metrics_v1.total_fees,
        net_pnl: metrics_v1.net_pnl,
        max_drawdown_pct: max_dd_decimal, // Store as decimal (0.1905 = 19.05%)
        win_rate: metrics_v1.win_rate,
        avg_win: metrics_v1.avg_win,
        avg_loss: metrics_v1.avg_loss,
        wal_digest: manifest_v1.trace_hash.clone(),
        guardrail_min_trades_ok,
        guardrail_max_dd_ok,
        guardrails_passed,
    };

    // Write run_summary.json
    let summary_path = run_dir.join("run_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, summary_json)?;

    Ok((metrics_v1, manifest_v1, diagnostics, summary))
}

/// Write results.jsonl
fn write_results_jsonl(path: &Path, results: &[GridRunResult]) -> Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    for result in results {
        let line = serde_json::to_string(result)?;
        writeln!(writer, "{}", line)?;
    }

    Ok(())
}

/// P2: Print topline ranking table sorted by net_pnl, then drawdown, then trades.
/// Applies guardrails filtering. Includes profit_factor and expectancy columns.
fn print_topline_ranking(results: &[GridRunResult], guardrails: &Guardrails) {
    // Collect summaries + metrics from all runs
    let mut ranked: Vec<_> = results
        .iter()
        .map(|r| (&r.summary, &r.metrics, &r.param_hash, &r.segment))
        .collect();

    // Sort: passed guardrails first, then by net_pnl descending, then drawdown ascending
    ranked.sort_by(|a, b| {
        // Guardrails passed comes first
        let pass_cmp = b.0.guardrails_passed.cmp(&a.0.guardrails_passed);
        if pass_cmp != std::cmp::Ordering::Equal {
            return pass_cmp;
        }
        // Then by net_pnl descending
        let pnl_cmp =
            b.0.net_pnl
                .partial_cmp(&a.0.net_pnl)
                .unwrap_or(std::cmp::Ordering::Equal);
        if pnl_cmp != std::cmp::Ordering::Equal {
            return pnl_cmp;
        }
        // Then by drawdown ascending
        let dd_cmp =
            a.0.max_drawdown_pct
                .partial_cmp(&b.0.max_drawdown_pct)
                .unwrap_or(std::cmp::Ordering::Equal);
        if dd_cmp != std::cmp::Ordering::Equal {
            return dd_cmp;
        }
        // Then by trades descending
        b.0.trades_closed.cmp(&a.0.trades_closed)
    });

    // Count how many passed guardrails and rejection reasons
    let passed_count = ranked
        .iter()
        .filter(|(s, _, _, _)| s.guardrails_passed)
        .count();
    let failed_min_trades = ranked
        .iter()
        .filter(|(s, _, _, _)| !s.guardrail_min_trades_ok)
        .count();
    let failed_max_dd = ranked
        .iter()
        .filter(|(s, _, _, _)| !s.guardrail_max_dd_ok)
        .count();
    let total_rejected = ranked.len() - passed_count;

    println!(
        "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                                         TOPLINE RANKING (net_pnl desc)                                                      ║"
    );
    println!(
        "║  Guardrails: min_trades >= {}, max_dd <= {:.1}%                                                                              ║",
        guardrails.min_trades,
        guardrails.max_drawdown_pct * 100.0
    );
    println!(
        "║  PASSED: {:>3} | REJECTED: {:>3} (min_trades: {:>3}, max_dd: {:>3})                                                              ║",
        passed_count, total_rejected, failed_min_trades, failed_max_dd
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ {:>4} │ {:>16} │ {:>6} │ {:>9} │ {:>5} │ {:>6} │ {:>7} │ {:>8} │ {:>6} │ {:>6} │ {:>4} ║",
        "Rank",
        "param_hash",
        "Trades",
        "Net PnL",
        "DD%",
        "WinRt",
        "ProfFac",
        "Expect",
        "AvgWin",
        "AvgLos",
        "Pass"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Print top 20 (or all if fewer)
    let display_count = ranked.len().min(20);
    for (i, (summary, metrics, param_hash, _segment)) in
        ranked.iter().take(display_count).enumerate()
    {
        let pass_str = if summary.guardrails_passed {
            "  Y "
        } else {
            "  N "
        };

        // Compute profit_factor and expectancy
        let profit_factor = if metrics.gross_loss.abs() > 0.001 {
            metrics.gross_profit / metrics.gross_loss.abs()
        } else if metrics.gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let expectancy = if summary.trades_closed > 0 {
            summary.net_pnl / summary.trades_closed as f64
        } else {
            0.0
        };

        // Format profit factor (cap display at 99.9)
        let pf_str = if profit_factor.is_infinite() {
            "   inf".to_string()
        } else {
            format!("{:>6.2}", profit_factor.min(99.99))
        };

        println!(
            "║ {:>4} │ {:>16} │ {:>6} │ {:>9.2} │ {:>4.1}% │ {:>5.1}% │ {} │ {:>8.2} │ {:>6.2} │ {:>6.2} │{} ║",
            i + 1,
            param_hash,
            summary.trades_closed,
            summary.net_pnl,
            summary.max_drawdown_pct * 100.0,
            summary.win_rate,
            pf_str,
            expectancy,
            summary.avg_win,
            summary.avg_loss,
            pass_str,
        );
    }

    if ranked.len() > display_count {
        println!(
            "║ {:>4} │ {:>16} │ {:>6} │ {:>9} │ {:>5} │ {:>6} │ {:>7} │ {:>8} │ {:>6} │ {:>6} │ {:>4} ║",
            "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."
        );
    }

    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Print rejection summary
    if total_rejected > 0 {
        println!("Rejected runs summary:");
        println!(
            "  - Failed min_trades (< {}): {} runs",
            guardrails.min_trades, failed_min_trades
        );
        println!(
            "  - Failed max_drawdown (> {:.1}%): {} runs",
            guardrails.max_drawdown_pct * 100.0,
            failed_max_dd
        );
        println!();
    }

    // Print parameter summary for top 5 passing runs
    if passed_count > 0 {
        println!("Top 5 parameter sets (passed guardrails):");
        for (i, (summary, metrics, param_hash, _)) in ranked
            .iter()
            .filter(|(s, _, _, _)| s.guardrails_passed)
            .take(5)
            .enumerate()
        {
            // Compute expectancy for display
            let expectancy = if summary.trades_closed > 0 {
                summary.net_pnl / summary.trades_closed as f64
            } else {
                0.0
            };
            let profit_factor = if metrics.gross_loss.abs() > 0.001 {
                metrics.gross_profit / metrics.gross_loss.abs()
            } else {
                0.0
            };

            // Extract key params
            let params = &summary.params;
            let d_perp = params
                .get("min_d_perp_mantissa")
                .map(|v| v.to_string())
                .unwrap_or_default();
            let tau_dir = params
                .get("tau_dir_enter")
                .map(|v| v.to_string())
                .unwrap_or_default();
            let u_agg = params
                .get("u_aggressive_mantissa")
                .map(|v| v.to_string())
                .unwrap_or_default();
            let hold = params
                .get("min_hold_ms")
                .map(|v| v.to_string())
                .unwrap_or_default();

            println!(
                "  {}. {} | d_perp={}, tau_dir={}, u_agg={}, hold={}ms",
                i + 1,
                param_hash,
                d_perp,
                tau_dir,
                u_agg,
                hold
            );
            println!(
                "     PnL=${:.2}, DD={:.2}%, Trades={}, Expect=${:.2}/trade, PF={:.2}",
                summary.net_pnl,
                summary.max_drawdown_pct * 100.0,
                summary.trades_closed,
                expectancy,
                profit_factor
            );
        }
        println!();
    }
}

/// Build leaderboard aggregated by param_hash.
fn build_grid_leaderboard(results: &[GridRunResult]) -> Vec<LeaderboardEntry> {
    // Group by param_hash
    let mut by_hash: BTreeMap<String, Vec<&GridRunResult>> = BTreeMap::new();
    for r in results {
        by_hash.entry(r.param_hash.clone()).or_default().push(r);
    }

    // Aggregate each group
    let mut entries: Vec<LeaderboardEntry> = Vec::new();

    for (param_hash, runs) in &by_hash {
        if runs.is_empty() {
            continue;
        }

        let params = runs[0].params.clone();

        // Extract metrics
        let returns: Vec<f64> = runs
            .iter()
            .map(|r| {
                // Use net_pnl / initial_capital as return approximation
                // (manifest has return_pct but it's percentage)
                r.manifest.return_pct / 100.0
            })
            .collect();

        let drawdowns: Vec<f64> = runs.iter().map(|r| r.metrics.max_drawdown_pct).collect();

        let positive_count = returns.iter().filter(|&&r| r > 0.0).count();

        // Compute aggregates
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let median_return = {
            let mut sorted = returns.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) && sorted.len() > 1 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        };
        let mean_drawdown = drawdowns.iter().sum::<f64>() / drawdowns.len() as f64;
        let positive_fraction = positive_count as f64 / runs.len() as f64;

        // Composite score (v0): mean_return - 2.0 * mean_drawdown
        let score = mean_return - 2.0 * mean_drawdown;

        entries.push(LeaderboardEntry {
            param_hash: param_hash.clone(),
            params,
            score,
            mean_total_return: mean_return,
            median_total_return: median_return,
            mean_max_drawdown: mean_drawdown,
            segments: runs.len(),
            positive_fraction,
        });
    }

    // Sort descending by score, then by param_hash for stability
    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.param_hash.cmp(&b.param_hash))
    });

    entries
}

/// Select top K promotion candidates.
fn select_promotion_candidates(
    leaderboard: &[LeaderboardEntry],
    strategy: &str,
    top_k: usize,
) -> PromotionCandidates {
    let candidates: Vec<LeaderboardEntry> = leaderboard
        .iter()
        .filter(|e| {
            e.mean_total_return > 0.0
                && e.positive_fraction >= 0.60
                && e.mean_max_drawdown <= 0.20
                && e.segments >= 1 // Relax for small tournaments
        })
        .take(top_k)
        .cloned()
        .collect();

    PromotionCandidates {
        generated_at_rfc3339: chrono::Utc::now().to_rfc3339(),
        strategy: strategy.to_string(),
        candidates,
    }
}

/// Get git commit and dirty flag (best effort).
fn get_git_info() -> (Option<String>, Option<bool>) {
    // Try to get commit hash
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    // Try to get dirty status
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty());

    (commit, dirty)
}

// =============================================================================
// Grid Tournament Tests
// =============================================================================

#[cfg(test)]
mod grid_tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_param_hash_determinism() {
        let mut params1 = BTreeMap::new();
        params1.insert("lookback".to_string(), ParamValue::Int(100));
        params1.insert("entry_z".to_string(), ParamValue::Float(2.0));

        let mut params2 = BTreeMap::new();
        params2.insert("entry_z".to_string(), ParamValue::Float(2.0));
        params2.insert("lookback".to_string(), ParamValue::Int(100));

        // Same params (regardless of insertion order) should produce same hash
        assert_eq!(compute_param_hash(&params1), compute_param_hash(&params2));
    }

    #[test]
    fn test_grid_expansion() {
        let dir = tempdir().unwrap();
        let grid_path = dir.path().join("grid.toml");

        std::fs::write(
            &grid_path,
            r#"
[params]
lookback = [50, 100]
entry_z = [1.5, 2.0]
"#,
        )
        .unwrap();

        let combos = parse_and_expand_grid(&grid_path).unwrap();
        assert_eq!(combos.len(), 4); // 2 × 2

        // Verify all combinations present
        let hashes: Vec<String> = combos.iter().map(compute_param_hash).collect();
        assert_eq!(hashes.len(), 4);
        // All unique
        let unique: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_leaderboard_ranking() {
        let results = vec![
            GridRunResult {
                segment: "seg1".to_string(),
                strategy: "test".to_string(),
                param_hash: "aaa".to_string(),
                params: BTreeMap::new(),
                run_dir: "runs/seg1/aaa".to_string(),
                metrics: crate::backtest::BacktestMetricsV1 {
                    schema_version: "v1".to_string(),
                    total_trades: 10,
                    winning_trades: 6,
                    losing_trades: 4,
                    win_rate: 60.0,
                    gross_profit: 100.0,
                    gross_loss: 50.0,
                    net_pnl: 50.0,
                    profit_factor: 2.0,
                    expectancy: 5.0,
                    avg_win: 16.67,
                    avg_loss: 12.5,
                    avg_win_loss_ratio: 1.33,
                    largest_win: 30.0,
                    largest_loss: 20.0,
                    max_drawdown: 100.0,
                    max_drawdown_pct: 0.05, // 5%
                    sharpe_ratio: 1.0,
                    sortino_ratio: 1.5,
                    avg_trade_duration_secs: 3600.0,
                    total_fees: 10.0,
                },
                manifest: crate::backtest::BacktestRunManifestV1 {
                    schema_version: "v1".to_string(),
                    run_id: "run1".to_string(),
                    strategy: "test".to_string(),
                    segment_path: "/test".to_string(),
                    total_events: 100,
                    total_fills: 20,
                    realized_pnl: 50.0,
                    return_pct: 5.0, // 5% return
                    trace_hash: "hash".to_string(),
                },
                diagnostics: RunDiagnostics {
                    events: EventCounts {
                        depth: 90,
                        trades: 0,
                        funding: 10,
                        spot: 0,
                        total: 100,
                    },
                    signals: SignalCounts {
                        generated: 20,
                        admitted: 20,
                        refused: 0,
                    },
                    orders: OrderCounts {
                        submitted: 20,
                        filled: 20,
                    },
                    zero_trade_reason: ZeroTradeReason::NotZeroTrade,
                    exit_reasons: BTreeMap::new(),
                    notes: None,
                },
                summary: RunSummary {
                    schema_version: "v1".to_string(),
                    params: BTreeMap::new(),
                    trades_closed: 10,
                    decisions: 20,
                    gross_pnl: 50.0,
                    fees: 10.0,
                    net_pnl: 50.0,
                    max_drawdown_pct: 0.05,
                    win_rate: 60.0,
                    avg_win: 16.67,
                    avg_loss: 12.5,
                    wal_digest: "hash".to_string(),
                    guardrail_min_trades_ok: true,
                    guardrail_max_dd_ok: true,
                    guardrails_passed: true,
                },
            },
            GridRunResult {
                segment: "seg1".to_string(),
                strategy: "test".to_string(),
                param_hash: "bbb".to_string(),
                params: BTreeMap::new(),
                run_dir: "runs/seg1/bbb".to_string(),
                metrics: crate::backtest::BacktestMetricsV1 {
                    schema_version: "v1".to_string(),
                    total_trades: 10,
                    winning_trades: 3,
                    losing_trades: 7,
                    win_rate: 30.0,
                    gross_profit: 50.0,
                    gross_loss: 100.0,
                    net_pnl: -50.0,
                    profit_factor: 0.5,
                    expectancy: -5.0,
                    avg_win: 16.67,
                    avg_loss: 14.29,
                    avg_win_loss_ratio: 1.17,
                    largest_win: 30.0,
                    largest_loss: 25.0,
                    max_drawdown: 200.0,
                    max_drawdown_pct: 0.10, // 10%
                    sharpe_ratio: -0.5,
                    sortino_ratio: -0.3,
                    avg_trade_duration_secs: 3600.0,
                    total_fees: 10.0,
                },
                manifest: crate::backtest::BacktestRunManifestV1 {
                    schema_version: "v1".to_string(),
                    run_id: "run2".to_string(),
                    strategy: "test".to_string(),
                    segment_path: "/test".to_string(),
                    total_events: 100,
                    total_fills: 20,
                    realized_pnl: -50.0,
                    return_pct: -5.0, // -5% return
                    trace_hash: "hash".to_string(),
                },
                diagnostics: RunDiagnostics {
                    events: EventCounts {
                        depth: 90,
                        trades: 0,
                        funding: 10,
                        spot: 0,
                        total: 100,
                    },
                    signals: SignalCounts {
                        generated: 20,
                        admitted: 20,
                        refused: 0,
                    },
                    orders: OrderCounts {
                        submitted: 20,
                        filled: 20,
                    },
                    zero_trade_reason: ZeroTradeReason::NotZeroTrade,
                    exit_reasons: BTreeMap::new(),
                    notes: None,
                },
                summary: RunSummary {
                    schema_version: "v1".to_string(),
                    params: BTreeMap::new(),
                    trades_closed: 10,
                    decisions: 20,
                    gross_pnl: -50.0,
                    fees: 10.0,
                    net_pnl: -50.0,
                    max_drawdown_pct: 0.10,
                    win_rate: 30.0,
                    avg_win: 16.67,
                    avg_loss: 14.29,
                    wal_digest: "hash".to_string(),
                    guardrail_min_trades_ok: true,
                    guardrail_max_dd_ok: false, // 10% > 5% limit
                    guardrails_passed: false,
                },
            },
        ];

        let leaderboard = build_grid_leaderboard(&results);
        assert_eq!(leaderboard.len(), 2);

        // "aaa" should rank first (positive return, lower drawdown)
        assert_eq!(leaderboard[0].param_hash, "aaa");
        assert!(leaderboard[0].score > leaderboard[1].score);
    }

    #[test]
    fn test_promotion_filter() {
        let leaderboard = vec![
            LeaderboardEntry {
                param_hash: "good".to_string(),
                params: BTreeMap::new(),
                score: 0.10,
                mean_total_return: 0.05,
                median_total_return: 0.04,
                mean_max_drawdown: 0.02,
                segments: 5,
                positive_fraction: 0.80,
            },
            LeaderboardEntry {
                param_hash: "bad_return".to_string(),
                params: BTreeMap::new(),
                score: -0.05,
                mean_total_return: -0.01,
                median_total_return: -0.02,
                mean_max_drawdown: 0.02,
                segments: 5,
                positive_fraction: 0.40,
            },
            LeaderboardEntry {
                param_hash: "bad_drawdown".to_string(),
                params: BTreeMap::new(),
                score: 0.02,
                mean_total_return: 0.06,
                median_total_return: 0.05,
                mean_max_drawdown: 0.25, // Too high
                segments: 5,
                positive_fraction: 0.70,
            },
        ];

        let candidates = select_promotion_candidates(&leaderboard, "test", 10);
        assert_eq!(candidates.candidates.len(), 1);
        assert_eq!(candidates.candidates[0].param_hash, "good");
    }

    // =========================================================================
    // P1.1 Diagnostics Tests
    // =========================================================================

    #[test]
    fn test_zero_trade_reason_not_zero_trade() {
        let mut diag = RunDiagnostics {
            events: EventCounts {
                depth: 100,
                trades: 0,
                funding: 10,
                spot: 0,
                total: 110,
            },
            signals: SignalCounts {
                generated: 5,
                admitted: 5,
                refused: 0,
            },
            orders: OrderCounts {
                submitted: 5,
                filled: 5,
            },
            zero_trade_reason: ZeroTradeReason::Unknown,
            exit_reasons: BTreeMap::new(),
            notes: None,
        };
        diag.compute_zero_trade_reason();
        assert_eq!(diag.zero_trade_reason, ZeroTradeReason::NotZeroTrade);
    }

    #[test]
    fn test_zero_trade_reason_no_events() {
        let mut diag = RunDiagnostics {
            events: EventCounts {
                depth: 0,
                trades: 0,
                funding: 0,
                spot: 0,
                total: 0,
            },
            signals: SignalCounts::default(),
            orders: OrderCounts::default(),
            zero_trade_reason: ZeroTradeReason::Unknown,
            exit_reasons: BTreeMap::new(),
            notes: None,
        };
        diag.compute_zero_trade_reason();
        assert_eq!(diag.zero_trade_reason, ZeroTradeReason::NoRelevantEvents);
    }

    #[test]
    fn test_zero_trade_reason_no_signals() {
        let mut diag = RunDiagnostics {
            events: EventCounts {
                depth: 100,
                trades: 0,
                funding: 10,
                spot: 0,
                total: 110,
            },
            signals: SignalCounts {
                generated: 0,
                admitted: 0,
                refused: 0,
            },
            orders: OrderCounts::default(),
            zero_trade_reason: ZeroTradeReason::Unknown,
            exit_reasons: BTreeMap::new(),
            notes: None,
        };
        diag.compute_zero_trade_reason();
        assert_eq!(diag.zero_trade_reason, ZeroTradeReason::NoSignals);
    }

    #[test]
    fn test_zero_trade_reason_all_refused() {
        let mut diag = RunDiagnostics {
            events: EventCounts {
                depth: 100,
                trades: 0,
                funding: 10,
                spot: 0,
                total: 110,
            },
            signals: SignalCounts {
                generated: 5,
                admitted: 0,
                refused: 5,
            },
            orders: OrderCounts::default(),
            zero_trade_reason: ZeroTradeReason::Unknown,
            exit_reasons: BTreeMap::new(),
            notes: None,
        };
        diag.compute_zero_trade_reason();
        assert_eq!(diag.zero_trade_reason, ZeroTradeReason::AllSignalsRefused);
    }

    #[test]
    fn test_zero_trade_reason_no_fills() {
        let mut diag = RunDiagnostics {
            events: EventCounts {
                depth: 100,
                trades: 0,
                funding: 10,
                spot: 0,
                total: 110,
            },
            signals: SignalCounts {
                generated: 5,
                admitted: 5,
                refused: 0,
            },
            orders: OrderCounts {
                submitted: 5,
                filled: 0,
            },
            zero_trade_reason: ZeroTradeReason::Unknown,
            exit_reasons: BTreeMap::new(),
            notes: None,
        };
        diag.compute_zero_trade_reason();
        assert_eq!(diag.zero_trade_reason, ZeroTradeReason::NoFills);
    }

    #[test]
    fn test_diagnostics_serialization() {
        let diag = RunDiagnostics {
            events: EventCounts {
                depth: 100,
                trades: 0,
                funding: 10,
                spot: 50,
                total: 160,
            },
            signals: SignalCounts {
                generated: 5,
                admitted: 3,
                refused: 2,
            },
            orders: OrderCounts {
                submitted: 3,
                filled: 2,
            },
            zero_trade_reason: ZeroTradeReason::NotZeroTrade,
            exit_reasons: BTreeMap::new(),
            notes: Some("Test note".to_string()),
        };

        let json = serde_json::to_string(&diag).unwrap();
        assert!(json.contains("\"depth\":100"));
        assert!(json.contains("\"funding\":10"));
        assert!(json.contains("\"generated\":5"));
        assert!(json.contains("\"NOT_ZERO_TRADE\""));
        assert!(json.contains("Test note"));

        // Verify it deserializes back
        let parsed: RunDiagnostics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.events.depth, 100);
        assert_eq!(parsed.signals.generated, 5);
        assert_eq!(parsed.zero_trade_reason, ZeroTradeReason::NotZeroTrade);
    }
}
