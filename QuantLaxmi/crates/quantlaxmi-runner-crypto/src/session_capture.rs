//! Multi-Symbol Session Capture for Arbitrage-Ready Data Collection.
//!
//! Captures depth and trades for multiple symbols in parallel, producing
//! a unified session with consistent timestamps and per-symbol manifests.
//!
//! ## Output Structure
//! ```text
//! data/sessions/{tag}/
//! ├── session_manifest.json    # Meta-manifest for entire session
//! ├── BTCUSDT/
//! │   ├── depth.jsonl
//! │   ├── trades.jsonl
//! │   └── manifest.json
//! ├── ETHUSDT/
//! │   ├── depth.jsonl
//! │   ├── trades.jsonl
//! │   └── manifest.json
//! └── ...
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

use crate::binance_sbe_depth_capture::{
    CaptureStats as DepthCaptureStats, capture_sbe_depth_jsonl,
};
use crate::binance_trades_capture::{TradesCaptureStats, capture_sbe_trades_jsonl};
use quantlaxmi_runner_common::artifact::{ArtifactFamily, FileHash, RunManifest, RunProfile};

/// Per-symbol capture statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCaptureStats {
    pub symbol: String,
    pub depth: DepthStats,
    pub trades: Option<TradesStats>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthStats {
    pub snapshot_written: bool,
    pub events_written: usize,
    pub gaps_detected: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradesStats {
    pub trades_written: usize,
    pub buy_count: usize,
    pub sell_count: usize,
    pub total_volume_mantissa: i64,
}

/// Session-level capture statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCaptureStats {
    pub session_id: String,
    pub symbols: Vec<String>,
    pub symbol_stats: HashMap<String, SymbolCaptureStats>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: f64,
    pub total_depth_events: usize,
    pub total_trades: usize,
    pub total_gaps: usize,
    pub all_symbols_clean: bool,
}

/// Session manifest for multi-symbol capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    pub session_id: String,
    pub watermark: String,
    pub family: String,
    pub profile: String,
    pub symbols: Vec<String>,
    pub captures: Vec<SymbolCapture>,
    pub determinism: SessionDeterminism,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCapture {
    pub symbol: String,
    pub manifest_path: String,
    pub depth_file: String,
    pub depth_hash: Option<String>,
    pub trades_file: Option<String>,
    pub trades_hash: Option<String>,
    pub events_written: usize,
    pub trades_written: usize,
    pub gaps_detected: usize,
    pub snapshot_written: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDeterminism {
    pub certified: bool,
    pub all_symbols_clean: bool,
    pub symbol_hashes: HashMap<String, SymbolHashes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolHashes {
    pub depth_hash: Option<String>,
    pub trades_hash: Option<String>,
}

/// Configuration for session capture.
#[derive(Debug, Clone)]
pub struct SessionCaptureConfig {
    pub symbols: Vec<String>,
    pub out_dir: PathBuf,
    pub duration_secs: u64,
    pub price_exponent: i8,
    pub qty_exponent: i8,
    pub include_trades: bool,
    pub strict: bool,
    pub api_key: String,
}

/// Capture a multi-symbol session.
///
/// Spawns parallel capture tasks for each symbol, collecting both depth and
/// optionally trades data. Produces per-symbol manifests and a session manifest.
pub async fn capture_session(config: SessionCaptureConfig) -> Result<SessionCaptureStats> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let start_time = Utc::now();

    println!(
        "Starting session capture: {} symbols, {} seconds",
        config.symbols.len(),
        config.duration_secs
    );
    println!("Session ID: {}", session_id);
    println!("Output directory: {:?}", config.out_dir);

    // Create session directory
    tokio::fs::create_dir_all(&config.out_dir).await?;

    // Spawn capture tasks for each symbol
    let mut depth_handles: Vec<(String, JoinHandle<Result<DepthCaptureStats>>)> = Vec::new();
    let mut trades_handles: Vec<(String, JoinHandle<Result<TradesCaptureStats>>)> = Vec::new();

    for symbol in &config.symbols {
        let symbol_dir = config.out_dir.join(symbol.to_uppercase());
        tokio::fs::create_dir_all(&symbol_dir).await?;

        // Depth capture
        let depth_path = symbol_dir.join("depth.jsonl");
        let sym = symbol.clone();
        let duration = config.duration_secs;
        let price_exp = config.price_exponent;
        let qty_exp = config.qty_exponent;
        let api_key = config.api_key.clone();

        let depth_handle = tokio::spawn(async move {
            capture_sbe_depth_jsonl(&sym, &depth_path, duration, price_exp, qty_exp, &api_key).await
        });
        depth_handles.push((symbol.clone(), depth_handle));

        // Trades capture (if enabled)
        if config.include_trades {
            let trades_path = symbol_dir.join("trades.jsonl");
            let sym = symbol.clone();
            let api_key = config.api_key.clone();

            let trades_handle = tokio::spawn(async move {
                capture_sbe_trades_jsonl(&sym, &trades_path, duration, price_exp, qty_exp, &api_key)
                    .await
            });
            trades_handles.push((symbol.clone(), trades_handle));
        }
    }

    println!(
        "Capturing {} depth streams + {} trades streams...",
        depth_handles.len(),
        trades_handles.len()
    );

    // Wait for all captures to complete
    let mut symbol_stats: HashMap<String, SymbolCaptureStats> = HashMap::new();
    let mut total_depth_events = 0;
    let mut total_trades = 0;
    let mut total_gaps = 0;
    let mut all_clean = true;

    // Collect depth results
    for (symbol, handle) in depth_handles {
        let symbol_start = start_time; // Approximate
        let symbol_end = Utc::now();
        let symbol_dir = config.out_dir.join(symbol.to_uppercase());
        let depth_path = symbol_dir.join("depth.jsonl");

        // Handle task completion (may panic or complete normally)
        let result = match handle.await {
            Ok(r) => r,
            Err(join_err) => {
                println!("ERROR: {} depth task panicked: {}", symbol, join_err);
                all_clean = false;
                // Try to recover stats from file
                Err(anyhow::anyhow!("Task panicked: {}", join_err))
            }
        };

        // Extract stats - either from result or from file as fallback
        let (events_written, gaps_detected, snapshot_written) = match &result {
            Ok(stats) => {
                (stats.events_written, stats.gaps_detected, stats.snapshot_written)
            }
            Err(e) => {
                println!("WARNING: {} depth capture error: {}", symbol, e);
                // Fallback: count lines in file if it exists
                let line_count = if depth_path.exists() {
                    count_file_lines(&depth_path).unwrap_or(0)
                } else {
                    0
                };
                // If file has data, bootstrap likely succeeded initially
                let snapshot_written = line_count > 0;
                println!(
                    "  Recovered from file: {} lines, snapshot={}",
                    line_count, snapshot_written
                );
                (line_count, 0, snapshot_written)
            }
        };

        if gaps_detected > 0 {
            all_clean = false;
            if config.strict {
                println!(
                    "WARNING: {} had {} sequence gaps (strict mode)",
                    symbol, gaps_detected
                );
            }
        }

        total_depth_events += events_written;
        total_gaps += gaps_detected;

        symbol_stats.insert(
            symbol.clone(),
            SymbolCaptureStats {
                symbol: symbol.clone(),
                depth: DepthStats {
                    snapshot_written,
                    events_written,
                    gaps_detected,
                },
                trades: None,
                start_time: symbol_start,
                end_time: symbol_end,
                duration_secs: (symbol_end - symbol_start).num_milliseconds() as f64 / 1000.0,
            },
        );

        println!(
            "{}: depth={} events, {} gaps",
            symbol, events_written, gaps_detected
        );
    }

    // Collect trades results
    for (symbol, handle) in trades_handles {
        let symbol_dir = config.out_dir.join(symbol.to_uppercase());
        let trades_path = symbol_dir.join("trades.jsonl");

        // Handle task completion (may panic or complete normally)
        let result = match handle.await {
            Ok(r) => r,
            Err(join_err) => {
                println!("ERROR: {} trades task panicked: {}", symbol, join_err);
                Err(anyhow::anyhow!("Task panicked: {}", join_err))
            }
        };

        // Extract stats - either from result or from file as fallback
        let (trades_written, buy_count, sell_count, total_volume_mantissa) = match &result {
            Ok(stats) => (
                stats.trades_written,
                stats.buy_count,
                stats.sell_count,
                stats.total_volume_mantissa,
            ),
            Err(e) => {
                println!("WARNING: {} trades capture error: {}", symbol, e);
                // Fallback: count lines in file if it exists
                let line_count = if trades_path.exists() {
                    count_file_lines(&trades_path).unwrap_or(0)
                } else {
                    0
                };
                println!("  Recovered from file: {} lines", line_count);
                (line_count, 0, 0, 0)
            }
        };

        total_trades += trades_written;

        if let Some(sym_stats) = symbol_stats.get_mut(&symbol) {
            sym_stats.trades = Some(TradesStats {
                trades_written,
                buy_count,
                sell_count,
                total_volume_mantissa,
            });
        }

        println!(
            "{}: trades={} (buys={}, sells={})",
            symbol, trades_written, buy_count, sell_count
        );
    }

    let end_time = Utc::now();
    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

    // Generate session manifest
    let session_manifest =
        generate_session_manifest(&config, &session_id, &symbol_stats, start_time, end_time)
            .await?;

    let manifest_path = config.out_dir.join("session_manifest.json");
    let json = serde_json::to_string_pretty(&session_manifest)?;
    tokio::fs::write(&manifest_path, json).await?;
    println!("Session manifest written: {:?}", manifest_path);

    // Generate per-symbol manifests
    for symbol in &config.symbols {
        if let Some(stats) = symbol_stats.get(symbol) {
            let symbol_dir = config.out_dir.join(symbol.to_uppercase());
            generate_symbol_manifest(&symbol_dir, symbol, stats, &config).await?;
        }
    }

    // Strict mode check
    if config.strict && total_gaps > 0 {
        println!(
            "\nSTRICT MODE FAILURE: {} total gaps across all symbols",
            total_gaps
        );
        return Err(anyhow::anyhow!(
            "Session capture failed strict mode: {} gaps detected",
            total_gaps
        ));
    }

    let stats = SessionCaptureStats {
        session_id,
        symbols: config.symbols.clone(),
        symbol_stats,
        start_time,
        end_time,
        duration_secs,
        total_depth_events,
        total_trades,
        total_gaps,
        all_symbols_clean: all_clean,
    };

    println!("\n=== Session Capture Complete ===");
    println!("  Duration: {:.1}s", stats.duration_secs);
    println!("  Total depth events: {}", stats.total_depth_events);
    println!("  Total trades: {}", stats.total_trades);
    println!("  Total gaps: {}", stats.total_gaps);
    println!(
        "  Status: {}",
        if stats.all_symbols_clean {
            "CLEAN"
        } else {
            "GAPS DETECTED"
        }
    );

    Ok(stats)
}

async fn generate_session_manifest(
    config: &SessionCaptureConfig,
    session_id: &str,
    symbol_stats: &HashMap<String, SymbolCaptureStats>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
) -> Result<SessionManifest> {
    let mut captures = Vec::new();
    let mut symbol_hashes = HashMap::new();
    let mut all_clean = true;

    for symbol in &config.symbols {
        let symbol_upper = symbol.to_uppercase();
        let symbol_dir = config.out_dir.join(&symbol_upper);

        let depth_file = format!("{}/depth.jsonl", symbol_upper);
        let depth_path = symbol_dir.join("depth.jsonl");
        let depth_hash = if depth_path.exists() {
            FileHash::from_file(&depth_path).ok().map(|h| h.sha256)
        } else {
            None
        };

        let (trades_file, trades_hash) = if config.include_trades {
            let trades_path = symbol_dir.join("trades.jsonl");
            let hash = if trades_path.exists() {
                FileHash::from_file(&trades_path).ok().map(|h| h.sha256)
            } else {
                None
            };
            (Some(format!("{}/trades.jsonl", symbol_upper)), hash)
        } else {
            (None, None)
        };

        let stats = symbol_stats.get(symbol);
        let (events_written, trades_written, gaps_detected, snapshot_written) = stats
            .map(|s| {
                (
                    s.depth.events_written,
                    s.trades.as_ref().map(|t| t.trades_written).unwrap_or(0),
                    s.depth.gaps_detected,
                    s.depth.snapshot_written,
                )
            })
            .unwrap_or((0, 0, 0, false));

        if gaps_detected > 0 {
            all_clean = false;
        }

        captures.push(SymbolCapture {
            symbol: symbol_upper.clone(),
            manifest_path: format!("{}/manifest.json", symbol_upper),
            depth_file,
            depth_hash: depth_hash.clone(),
            trades_file,
            trades_hash: trades_hash.clone(),
            events_written,
            trades_written,
            gaps_detected,
            snapshot_written,
        });

        symbol_hashes.insert(
            symbol_upper,
            SymbolHashes {
                depth_hash,
                trades_hash,
            },
        );
    }

    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;
    let watermark = format!(
        "QuantLaxmi-crypto-session-{}-{}",
        start_time.format("%Y%m%d-%H%M%S"),
        &session_id[..8]
    );

    Ok(SessionManifest {
        session_id: session_id.to_string(),
        watermark,
        family: "crypto".to_string(),
        profile: if config.strict && all_clean {
            "certified".to_string()
        } else {
            "research".to_string()
        },
        symbols: config.symbols.iter().map(|s| s.to_uppercase()).collect(),
        captures,
        determinism: SessionDeterminism {
            certified: config.strict && all_clean,
            all_symbols_clean: all_clean,
            symbol_hashes,
        },
        started_at: start_time,
        finished_at: end_time,
        duration_secs,
    })
}

async fn generate_symbol_manifest(
    symbol_dir: &Path,
    symbol: &str,
    stats: &SymbolCaptureStats,
    config: &SessionCaptureConfig,
) -> Result<()> {
    let profile = if config.strict && stats.depth.gaps_detected == 0 {
        RunProfile::Certified
    } else {
        RunProfile::Research
    };

    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, profile);

    // Record depth file hash
    let depth_path = symbol_dir.join("depth.jsonl");
    if depth_path.exists() {
        manifest.inputs.depth_events = Some(FileHash::from_file(&depth_path)?);
    }

    // Mark certification status
    manifest.determinism.certified = config.strict && stats.depth.gaps_detected == 0;
    manifest.compute_input_hash();

    // Add capture metadata
    manifest.diagnostics.context_validation.validation_passed = stats.depth.snapshot_written;
    if stats.depth.gaps_detected > 0 {
        manifest.diagnostics.context_validation.validation_errors = vec![format!(
            "Sequence gaps detected: {} (replay may fail)",
            stats.depth.gaps_detected
        )];
    }

    let integrity_tier = if manifest.determinism.certified {
        "Certified"
    } else {
        "Research"
    };

    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: stats.end_time,
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol.to_uppercase(),
                "depth_events_written": stats.depth.events_written,
                "trades_written": stats.trades.as_ref().map(|t| t.trades_written).unwrap_or(0),
                "snapshot_written": stats.depth.snapshot_written,
                "gaps_detected": stats.depth.gaps_detected,
                "price_exponent": config.price_exponent,
                "qty_exponent": config.qty_exponent,
                "strict_mode": config.strict,
                "integrity_tier": integrity_tier,
                "source": "binance_session_capture"
            }),
        },
    );

    manifest.finish();

    let manifest_path = symbol_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!(
        "  {} manifest written: {:?}",
        symbol.to_uppercase(),
        manifest_path
    );

    Ok(())
}

/// Count lines in a file (for fallback stats recovery).
fn count_file_lines(path: &Path) -> std::io::Result<usize> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_manifest_serialization() {
        let manifest = SessionManifest {
            session_id: "test-123".to_string(),
            watermark: "QuantLaxmi-crypto-session-20260122-120000-test1234".to_string(),
            family: "crypto".to_string(),
            profile: "certified".to_string(),
            symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            captures: vec![],
            determinism: SessionDeterminism {
                certified: true,
                all_symbols_clean: true,
                symbol_hashes: HashMap::new(),
            },
            started_at: Utc::now(),
            finished_at: Utc::now(),
            duration_secs: 300.0,
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        assert!(json.contains("session_id"));
        assert!(json.contains("BTCUSDT"));
    }
}
