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

use crate::backtest::{BacktestConfig, BacktestEngine, ExchangeConfig, PaceMode};
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
        enforce_admission_from_wal: false,
        admission_mismatch_policy: "fail".to_string(),
        strategy_spec: None, // Phase 22C: No permission gating in tournament mode
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
