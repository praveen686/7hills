//! gate-check CLI — Signal promotion gate validator.
//!
//! Phase 20C/20D: CLI for running G0/G1/G2 signal promotion gates.
//!
//! ## Usage
//!
//! ```bash
//! # Validate manifest schema (G0)
//! gate-check g0 --manifest config/signals_manifest.json
//!
//! # Check WAL parity between live and replay (G1)
//! gate-check g1 --live wal/live.jsonl --replay wal/replay.jsonl
//!
//! # Check admission coverage (G2)
//! gate-check g2 --wal wal/signals_admission.jsonl --threshold 0.9
//!
//! # Run all gates
//! gate-check all --manifest config/signals_manifest.json \
//!                --live wal/live.jsonl \
//!                --replay wal/replay.jsonl \
//!                --wal wal/signals_admission.jsonl
//!
//! # Phase 20D: Promote with artifact writing
//! gate-check promote --segment-dir ./segment \
//!                    --manifest config/signals_manifest.json \
//!                    --session-dir ./session \
//!                    --replay-dir ./replay \
//!                    --min-coverage 0.9
//! ```
//!
//! ## Exit Codes
//! - 0: All gates passed (promotion_record written for promote)
//! - 1: One or more gates failed (no promotion_record)
//! - 2: Error (missing files, invalid arguments, etc.)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use quantlaxmi_gates::promotion_pipeline::{
    GateOutcome, GatesSummary, PromotionRecord, g0_output_path, g1_output_path, g2_output_path,
    gates_dir, gates_summary_path, get_git_branch, get_git_clean, get_git_commit, get_hostname,
    promotion_dir, promotion_record_path, session_wal_path, sha256_hex,
};
use quantlaxmi_gates::g4_admission_determinism::G4AdmissionDeterminismGate;
use quantlaxmi_gates::signal_gates::{
    G0SchemaGate, G1DeterminismGate, G2DataIntegrityGate, G3ExecutionContractGate,
    SignalGatesResult, check_names,
};
use quantlaxmi_gates::signals_manifest::SignalsManifest;

/// gate-check: Signal promotion gate validator.
///
/// Validates signal admission integrity across G0 (schema), G1 (determinism),
/// and G2 (coverage) gates.
#[derive(Parser)]
#[command(name = "gate-check")]
#[command(version = "0.1.0")]
#[command(about = "Signal promotion gate validator for QuantLaxmi")]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format: text (default) or json
    #[arg(long, global = true, default_value = "text")]
    format: OutputFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// G0: Validate signals_manifest.json schema
    G0 {
        /// Path to signals_manifest.json
        #[arg(long, short = 'm')]
        manifest: PathBuf,
    },

    /// G1: Check WAL parity between live and replay
    G1 {
        /// Path to live WAL file (signals_admission.jsonl)
        #[arg(long, short = 'l')]
        live: PathBuf,

        /// Path to replay WAL file
        #[arg(long, short = 'r')]
        replay: PathBuf,
    },

    /// G2: Check admission event coverage
    G2 {
        /// Path to WAL file (signals_admission.jsonl)
        #[arg(long, short = 'w')]
        wal: PathBuf,

        /// Minimum coverage threshold (0.0 to 1.0)
        #[arg(long, short = 't', default_value = "0.0")]
        threshold: f64,

        /// Minimum number of events required
        #[arg(long, short = 'n', default_value = "1")]
        min_events: usize,
    },

    /// G3: Validate strategies_manifest.json execution contracts
    G3 {
        /// Path to strategies_manifest.json
        #[arg(long, short = 's')]
        strategies: PathBuf,

        /// Path to signals_manifest.json (optional, enables signal binding validation)
        #[arg(long, short = 'm')]
        signals: Option<PathBuf>,

        /// Path to promotion directory (optional, enables promotion status check)
        #[arg(long)]
        promotion_root: Option<PathBuf>,
    },

    /// G4: Strategy admission determinism gate (compares strategy_admission WALs)
    G4 {
        /// Path to live WAL file (strategy_admission.jsonl)
        #[arg(long, short = 'l')]
        live: PathBuf,

        /// Path to replay WAL file (strategy_admission.jsonl)
        #[arg(long, short = 'r')]
        replay: PathBuf,
    },

    /// Run all gates (without artifact writing)
    All {
        /// Path to signals_manifest.json
        #[arg(long, short = 'm')]
        manifest: PathBuf,

        /// Path to live WAL file (optional, skips G1 if not provided)
        #[arg(long, short = 'l')]
        live: Option<PathBuf>,

        /// Path to replay WAL file (required if live is provided)
        #[arg(long, short = 'r')]
        replay: Option<PathBuf>,

        /// Path to WAL file for G2 (uses live if not provided)
        #[arg(long, short = 'w')]
        wal: Option<PathBuf>,

        /// Minimum coverage threshold for G2
        #[arg(long, short = 't', default_value = "0.0")]
        threshold: f64,

        /// Minimum events for G2
        #[arg(long, short = 'n', default_value = "1")]
        min_events: usize,
    },

    /// Phase 20D: Run gates and write promotion artifacts
    Promote {
        /// Segment directory (required, artifacts written here)
        #[arg(long)]
        segment_dir: PathBuf,

        /// Path to signals_manifest.json (required)
        #[arg(long, short = 'm')]
        manifest: PathBuf,

        /// Session directory (optional; enables G2, enables G1 if replay also provided)
        #[arg(long)]
        session_dir: Option<PathBuf>,

        /// Replay directory (optional; enables G1 if session-dir also present)
        #[arg(long)]
        replay_dir: Option<PathBuf>,

        /// Minimum coverage threshold for G2 (0.0 to 1.0)
        #[arg(long, short = 't', default_value = "0.0")]
        min_coverage: f64,

        /// Minimum events for G2
        #[arg(long, short = 'n', default_value = "1")]
        min_events: usize,

        /// Skip G1 even if replay-dir provided
        #[arg(long, default_value = "false")]
        skip_g1: bool,

        /// Dry run: validate but don't write artifacts
        #[arg(long, default_value = "false")]
        dry_run: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(passed) => {
            if passed {
                ExitCode::from(0)
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> Result<bool, Box<dyn std::error::Error>> {
    match cli.command {
        Commands::G0 { manifest } => run_g0(&manifest, cli.format),
        Commands::G1 { live, replay } => run_g1(&live, &replay, cli.format),
        Commands::G2 {
            wal,
            threshold,
            min_events,
        } => run_g2(&wal, threshold, min_events, cli.format),
        Commands::G3 {
            strategies,
            signals,
            promotion_root,
        } => run_g3(
            &strategies,
            signals.as_deref(),
            promotion_root.as_deref(),
            cli.format,
        ),
        Commands::G4 { live, replay } => run_g4(&live, &replay, cli.format),
        Commands::All {
            manifest,
            live,
            replay,
            wal,
            threshold,
            min_events,
        } => run_all(
            &manifest,
            live.as_deref(),
            replay.as_deref(),
            wal.as_deref(),
            threshold,
            min_events,
            cli.format,
        ),
        Commands::Promote {
            segment_dir,
            manifest,
            session_dir,
            replay_dir,
            min_coverage,
            min_events,
            skip_g1,
            dry_run,
        } => run_promote(
            &segment_dir,
            &manifest,
            session_dir.as_deref(),
            replay_dir.as_deref(),
            min_coverage,
            min_events,
            skip_g1,
            dry_run,
            cli.format,
        ),
    }
}

fn run_g0(manifest: &Path, format: OutputFormat) -> Result<bool, Box<dyn std::error::Error>> {
    let result = G0SchemaGate::validate(manifest);

    match format {
        OutputFormat::Text => {
            println!("[{}] {}", result.check_name, result.message);
            if let Some(ref hash) = result.manifest_hash {
                println!("  Manifest hash: {}", hex::encode(hash));
            }
            if let Some(count) = result.signal_count {
                println!("  Signal count: {}", count);
            }
            if let Some(ref version) = result.manifest_version {
                println!("  Manifest version: {}", version);
            }
            if let Some(ref err) = result.error {
                println!("  Error: {}", err);
            }
            println!("\nG0 {}", if result.passed { "PASSED" } else { "FAILED" });
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(result.passed)
}

fn run_g1(
    live: &Path,
    replay: &Path,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let result = G1DeterminismGate::compare(live, replay)?;

    match format {
        OutputFormat::Text => {
            println!("[{}] {}", result.check_name, result.message);
            println!(
                "  Live entries: {}, Replay entries: {}, Matched: {}",
                result.live_entry_count, result.replay_entry_count, result.matched_count
            );

            if !result.mismatches.is_empty() {
                println!("\nMismatches ({}):", result.mismatches.len());
                for (i, m) in result.mismatches.iter().enumerate() {
                    if i >= 10 {
                        println!("  ... and {} more", result.mismatches.len() - 10);
                        break;
                    }
                    println!("  - {}", m.description());
                }
            }

            println!("\nG1 {}", if result.passed { "PASSED" } else { "FAILED" });
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(result.passed)
}

fn run_g2(
    wal: &Path,
    threshold: f64,
    min_events: usize,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let result = G2DataIntegrityGate::validate(wal, min_events, threshold)?;

    match format {
        OutputFormat::Text => {
            println!("[{}] {}", result.check_name, result.message);
            println!(
                "  Total events: {}, Admitted: {}, Refused: {}",
                result.total_events, result.admitted_count, result.refused_count
            );
            println!(
                "  Coverage: {:.2}% (threshold: {:.2}%)",
                result.coverage_ratio * 100.0,
                result.required_threshold * 100.0
            );
            if result.parse_error_count > 0 {
                println!("  Parse errors: {}", result.parse_error_count);
            }

            println!("\nG2 {}", if result.passed { "PASSED" } else { "FAILED" });
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(result.passed)
}

fn run_g3(
    strategies: &Path,
    signals: Option<&Path>,
    promotion_root: Option<&Path>,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let result = match (signals, promotion_root) {
        (Some(s), Some(p)) => G3ExecutionContractGate::validate_full(strategies, s, Some(p)),
        (Some(s), None) => G3ExecutionContractGate::validate_bindings(strategies, s),
        (None, _) => G3ExecutionContractGate::validate_schema(strategies),
    };

    match format {
        OutputFormat::Text => {
            println!("=== G3 Execution Contract Gate ===\n");

            for check in &result.checks {
                println!(
                    "[{}] {}",
                    check.name,
                    if check.passed { "PASS" } else { "FAIL" }
                );
                println!("  {}", check.message);
            }

            if let Some(ref hash) = result.strategies_manifest_hash {
                println!("\nStrategies manifest hash: {}", hex::encode(&hash[..8]));
            }
            if let Some(count) = result.strategy_count {
                println!("Strategy count: {}", count);
            }
            if let Some(count) = result.signal_binding_count {
                println!("Signal binding count: {}", count);
            }
            if let Some(ref version) = result.strategies_manifest_version {
                println!("Manifest version: {}", version);
            }

            if !result.violations.is_empty() {
                println!("\nViolations ({}):", result.violations.len());
                for v in &result.violations {
                    println!("  - {}", v.description());
                }
            }

            if let Some(ref err) = result.error {
                println!("\nError: {}", err);
            }

            println!("\nG3 {}", if result.passed { "PASSED" } else { "FAILED" });
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(result.passed)
}

fn run_g4(
    live: &Path,
    replay: &Path,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let result = G4AdmissionDeterminismGate::compare(live, replay)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    match format {
        OutputFormat::Text => {
            println!("[{}] {}", result.check_name, result.message);
            println!(
                "  Live entries: {}, Replay entries: {}, Matched: {}",
                result.live_entry_count, result.replay_entry_count, result.matched_count
            );

            if !result.mismatches.is_empty() {
                println!("\nMismatches ({}):", result.mismatches.len());
                for (i, m) in result.mismatches.iter().enumerate() {
                    if i >= 10 {
                        println!("  ... and {} more", result.mismatches.len() - 10);
                        break;
                    }
                    println!("  - {}", m.description());
                }
            }

            if !result.parse_errors.is_empty() {
                println!("\nParse errors ({}):", result.parse_errors.len());
                for (i, e) in result.parse_errors.iter().enumerate() {
                    if i >= 5 {
                        println!("  ... and {} more", result.parse_errors.len() - 5);
                        break;
                    }
                    println!("  - {}", e);
                }
            }

            println!("\nG4 {}", if result.passed { "PASSED" } else { "FAILED" });
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(result.passed)
}

fn run_all(
    manifest: &Path,
    live: Option<&Path>,
    replay: Option<&Path>,
    wal: Option<&Path>,
    threshold: f64,
    min_events: usize,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut combined = SignalGatesResult::new();

    // G0: Always run
    let g0 = G0SchemaGate::validate(manifest);
    combined = combined.with_g0(g0.clone());

    // G1: Run if both live and replay are provided
    let g1 = match (live, replay) {
        (Some(l), Some(r)) => Some(G1DeterminismGate::compare(l, r)?),
        (Some(_), None) => {
            return Err("G1 requires both --live and --replay".into());
        }
        (None, Some(_)) => {
            return Err("G1 requires both --live and --replay".into());
        }
        (None, None) => None,
    };

    if let Some(ref r) = g1 {
        combined = combined.with_g1(r.clone());
    }

    // G2: Use provided wal, or live if available
    let g2_wal = wal.or(live);
    let g2 = if let Some(w) = g2_wal {
        Some(G2DataIntegrityGate::validate(w, min_events, threshold)?)
    } else {
        None
    };

    if let Some(ref r) = g2 {
        combined = combined.with_g2(r.clone());
    }

    let combined = combined.finalize();

    match format {
        OutputFormat::Text => {
            println!("=== Gate Check Results ===\n");

            // G0
            println!(
                "[{}] {}",
                check_names::G0_SCHEMA_VALID,
                if g0.passed { "PASS" } else { "FAIL" }
            );
            println!("  {}", g0.message);

            // G1
            if let Some(ref r) = g1 {
                println!(
                    "\n[{}] {}",
                    check_names::G1_DECISION_PARITY,
                    if r.passed { "PASS" } else { "FAIL" }
                );
                println!("  {}", r.message);
                if !r.mismatches.is_empty() {
                    println!("  Mismatches: {}", r.mismatches.len());
                }
            } else {
                println!(
                    "\n[{}] SKIPPED (no WAL files provided)",
                    check_names::G1_DECISION_PARITY
                );
            }

            // G2
            if let Some(ref r) = g2 {
                println!(
                    "\n[{}] {}",
                    r.check_name,
                    if r.passed { "PASS" } else { "FAIL" }
                );
                println!("  {}", r.message);
            } else {
                println!(
                    "\n[{}] SKIPPED (no WAL file)",
                    check_names::G2_COVERAGE_THRESHOLD
                );
            }

            println!("\n=== {} ===", combined.summary);
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&combined)?);
        }
    }

    Ok(combined.passed)
}

/// Phase 20D: Run gates with artifact writing.
#[allow(clippy::too_many_arguments)]
fn run_promote(
    segment_dir: &Path,
    manifest_path: &Path,
    session_dir: Option<&Path>,
    replay_dir: Option<&Path>,
    min_coverage: f64,
    min_events: usize,
    skip_g1: bool,
    dry_run: bool,
    format: OutputFormat,
) -> Result<bool, Box<dyn std::error::Error>> {
    // 1. Validate manifest exists and is readable
    if !manifest_path.exists() {
        return Err(format!("Manifest not found: {}", manifest_path.display()).into());
    }

    // 2. Create directories if not dry run
    if !dry_run {
        fs::create_dir_all(gates_dir(segment_dir))?;
        fs::create_dir_all(promotion_dir(segment_dir))?;
    }

    // 3. Load manifest for metadata
    let manifest = SignalsManifest::load_validated(manifest_path)?;
    let manifest_hash = manifest.compute_version_hash();
    let manifest_version = manifest.manifest_version.clone();

    // 4. Run G0 (always)
    let g0_result = G0SchemaGate::validate(manifest_path);
    let g0_json = serde_json::to_vec_pretty(&g0_result)?;
    let g0_digest = sha256_hex(&g0_json);

    if !dry_run {
        fs::write(g0_output_path(segment_dir), &g0_json)?;
    }

    let mut summary = GatesSummary::new();
    summary = summary.with_g0(GateOutcome::new(
        "g0",
        g0_result.passed,
        "gates/g0_manifest.json",
        &g0_digest,
    ));

    let mut gate_digests = std::collections::BTreeMap::new();
    gate_digests.insert("g0".to_string(), g0_digest);

    // 5. Run G2 if session_dir provided
    let g2_result = if let Some(sess_dir) = session_dir {
        let wal_path = session_wal_path(sess_dir);
        if !wal_path.exists() {
            return Err(format!("Session WAL not found: {}", wal_path.display()).into());
        }

        let result = G2DataIntegrityGate::validate(&wal_path, min_events, min_coverage)?;
        let g2_json = serde_json::to_vec_pretty(&result)?;
        let g2_digest = sha256_hex(&g2_json);

        if !dry_run {
            fs::write(g2_output_path(segment_dir), &g2_json)?;
        }

        summary = summary.with_g2(GateOutcome::new(
            "g2",
            result.passed,
            "gates/g2_integrity.json",
            &g2_digest,
        ));
        gate_digests.insert("g2".to_string(), g2_digest);

        Some(result)
    } else {
        None
    };

    // 6. Run G1 if both session_dir and replay_dir provided (and not skipped)
    let g1_result = if let (Some(sess_dir), Some(rep_dir)) = (session_dir, replay_dir) {
        if skip_g1 {
            None
        } else {
            let live_wal = session_wal_path(sess_dir);
            let replay_wal = session_wal_path(rep_dir);

            if !live_wal.exists() {
                return Err(format!("Live WAL not found: {}", live_wal.display()).into());
            }
            if !replay_wal.exists() {
                return Err(format!("Replay WAL not found: {}", replay_wal.display()).into());
            }

            let result = G1DeterminismGate::compare(&live_wal, &replay_wal)?;
            let g1_json = serde_json::to_vec_pretty(&result)?;
            let g1_digest = sha256_hex(&g1_json);

            if !dry_run {
                fs::write(g1_output_path(segment_dir), &g1_json)?;
            }

            summary = summary.with_g1(GateOutcome::new(
                "g1",
                result.passed,
                "gates/g1_determinism.json",
                &g1_digest,
            ));
            gate_digests.insert("g1".to_string(), g1_digest);

            Some(result)
        }
    } else {
        None
    };

    // 7. Write gates_summary.json
    let summary_json = summary.to_json_bytes();
    let summary_digest = sha256_hex(&summary_json);

    if !dry_run {
        fs::write(gates_summary_path(segment_dir), &summary_json)?;
    }

    // 8. Output results
    match format {
        OutputFormat::Text => {
            println!("=== Promote Gate Results ===\n");
            println!("Segment: {}", segment_dir.display());
            println!(
                "Manifest: {} (v{})",
                manifest_path.display(),
                manifest_version
            );
            println!("Manifest hash: {}", hex::encode(&manifest_hash[..8]));
            println!();

            // G0
            println!(
                "[G0] {} - {}",
                if g0_result.passed { "PASS" } else { "FAIL" },
                g0_result.message
            );

            // G2
            if let Some(ref r) = g2_result {
                println!(
                    "[G2] {} - {}",
                    if r.passed { "PASS" } else { "FAIL" },
                    r.message
                );
            } else {
                println!("[G2] SKIPPED (no --session-dir)");
            }

            // G1
            if let Some(ref r) = g1_result {
                println!(
                    "[G1] {} - {}",
                    if r.passed { "PASS" } else { "FAIL" },
                    r.message
                );
            } else if skip_g1 {
                println!("[G1] SKIPPED (--skip-g1)");
            } else if replay_dir.is_none() {
                println!("[G1] SKIPPED (no --replay-dir)");
            } else {
                println!("[G1] SKIPPED (no --session-dir)");
            }

            println!();
        }
        OutputFormat::Json => {
            // Will output full summary at the end
        }
    }

    // 9. If failed, exit without promotion record
    if !summary.passed {
        match format {
            OutputFormat::Text => {
                println!("=== PROMOTION FAILED ===");
                println!("Gate failures detected. No promotion_record written.");
            }
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            }
        }
        return Ok(false);
    }

    // 10. Build and write promotion record
    let git_commit = get_git_commit();
    let git_branch = get_git_branch();
    let git_clean = get_git_clean();
    let hostname = get_hostname();

    let record = PromotionRecord::builder()
        .git_info(git_commit, git_branch, git_clean)
        .manifest(
            &manifest_path.display().to_string(),
            manifest_hash,
            &manifest_version,
        )
        .segment_dir(&segment_dir.display().to_string())
        .session_dir(session_dir.map(|p| p.display().to_string()))
        .replay_dir(replay_dir.map(|p| p.display().to_string()))
        .thresholds(min_coverage, min_events)
        .gates_passed(true)
        .gates_summary_digest(&summary_digest)
        .add_gate_digest("g0", gate_digests.get("g0").unwrap())
        .hostname(hostname)
        .build();

    // Add remaining gate digests
    let mut record_builder = PromotionRecord::builder()
        .timestamp(&record.timestamp)
        .promotion_id(&record.promotion_id)
        .git_info(
            record.git_commit.clone(),
            record.git_branch.clone(),
            record.git_clean,
        )
        .manifest(
            &record.manifest_path,
            record.manifest_hash,
            &record.manifest_version,
        )
        .segment_dir(&record.segment_dir)
        .session_dir(record.session_dir.clone())
        .replay_dir(record.replay_dir.clone())
        .thresholds(record.min_coverage, record.min_events)
        .gates_passed(true)
        .gates_summary_digest(&summary_digest)
        .hostname(record.hostname.clone());

    for (gate, digest) in &gate_digests {
        record_builder = record_builder.add_gate_digest(gate, digest);
    }

    let final_record = record_builder.build();

    if !dry_run {
        let record_json = final_record.to_json_bytes();
        fs::write(promotion_record_path(segment_dir), &record_json)?;
    }

    match format {
        OutputFormat::Text => {
            println!("=== PROMOTION SUCCESSFUL ===");
            println!("Promotion ID: {}", final_record.promotion_id);
            println!("Digest: {}...", &final_record.digest[..16]);
            if dry_run {
                println!("\n(Dry run: no files written)");
            } else {
                println!("\nArtifacts written:");
                println!("  - {}", g0_output_path(segment_dir).display());
                if g2_result.is_some() {
                    println!("  - {}", g2_output_path(segment_dir).display());
                }
                if g1_result.is_some() {
                    println!("  - {}", g1_output_path(segment_dir).display());
                }
                println!("  - {}", gates_summary_path(segment_dir).display());
                println!("  - {}", promotion_record_path(segment_dir).display());
            }
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&final_record)?);
        }
    }

    Ok(true)
}
