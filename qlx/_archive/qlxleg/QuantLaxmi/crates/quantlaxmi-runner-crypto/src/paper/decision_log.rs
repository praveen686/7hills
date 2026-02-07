//! Decision WAL (Write-Ahead Log) for SLRT Paper Trading
//!
//! JSONL format for quantitative analysis of sniper gate behavior.
//! Captures every decision tick with full context for offline analysis.

use serde::Serialize;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Book state snapshot for logging.
#[derive(Debug, Clone, Serialize)]
pub struct BookLog {
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread_bps: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
    pub imb: f64,
}

/// FTI state for logging.
/// No silent poisoning: numeric fields are Option to log null when absent.
#[derive(Debug, Clone, Serialize)]
pub struct FTILog {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thresh: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elev: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persist: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated: Option<bool>,
}

/// Model/regime state for logging.
/// No silent poisoning: all metrics are Option<f64> to log null when absent.
#[derive(Debug, Clone, Serialize)]
pub struct StateLog {
    pub regime: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d_perp: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fragility: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toxicity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tox_persist: Option<f64>,
    pub fti: FTILog,
}

/// Single gate evaluation result.
#[derive(Debug, Clone, Serialize)]
pub struct GateLog {
    pub name: String,
    pub pass: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Edge calculation breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct EdgeLog {
    pub est: f64,
    pub req: f64,
    pub fees: f64,
    pub spread: f64,
    pub buffer: f64,
}

/// Sniper evaluation summary.
#[derive(Debug, Clone, Serialize)]
pub struct SniperLog {
    pub eligible: bool,
    pub final_gate: String,
    pub gates: Vec<GateLog>,
    pub edge: EdgeLog,
    pub cooldown_ms: u64,
    pub entries_hour: u32,
    pub entries_session: u32,
}

/// Proposed/accepted intent.
#[derive(Debug, Clone, Serialize)]
pub struct IntentLog {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed: Option<ProposedIntent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted: Option<ProposedIntent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProposedIntent {
    pub side: String,
    pub qty: f64,
    #[serde(rename = "type")]
    pub order_type: String,
    pub id: String,
}

/// Execution outcome (only when trade happens).
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionLog {
    pub fill_px: f64,
    pub fill_qty: f64,
    pub fee_usd: f64,
    pub position_after: f64,
    pub pnl_after: f64,
}

/// Complete decision record.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionRecord {
    pub ts_ms: u64,
    pub tick: u64,
    pub symbol: String,
    pub book: BookLog,
    pub state: StateLog,
    pub sniper: SniperLog,
    pub intent: IntentLog,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionLog>,
}

/// Gate failure counters for live analysis.
#[derive(Debug, Default)]
pub struct GateCounters {
    pub total_ticks: AtomicU64,
    pub warmup_ticks: AtomicU64,
    pub eligible_ticks: AtomicU64,
    pub accepted_ticks: AtomicU64,
    pub fail_counts: parking_lot::RwLock<HashMap<String, u64>>,
}

impl GateCounters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_tick(
        &self,
        is_warmup: bool,
        is_eligible: bool,
        is_accepted: bool,
        failed_gates: &[String],
    ) {
        self.total_ticks.fetch_add(1, Ordering::Relaxed);

        if is_warmup {
            self.warmup_ticks.fetch_add(1, Ordering::Relaxed);
        }
        if is_eligible {
            self.eligible_ticks.fetch_add(1, Ordering::Relaxed);
        }
        if is_accepted {
            self.accepted_ticks.fetch_add(1, Ordering::Relaxed);
        }

        let mut counts = self.fail_counts.write();
        for gate in failed_gates {
            *counts.entry(gate.clone()).or_insert(0) += 1;
        }
    }

    /// Get top N failing gates.
    pub fn top_failing_gates(&self, n: usize) -> Vec<(String, u64, f64)> {
        let total = self.total_ticks.load(Ordering::Relaxed).max(1);
        let counts = self.fail_counts.read();

        let mut sorted: Vec<_> = counts
            .iter()
            .map(|(k, v)| (k.clone(), *v, *v as f64 / total as f64 * 100.0))
            .collect();

        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Get summary stats.
    pub fn summary(&self) -> (u64, u64, u64, u64) {
        (
            self.total_ticks.load(Ordering::Relaxed),
            self.warmup_ticks.load(Ordering::Relaxed),
            self.eligible_ticks.load(Ordering::Relaxed),
            self.accepted_ticks.load(Ordering::Relaxed),
        )
    }
}

/// Async decision logger with bounded channel.
pub struct DecisionLogger {
    tx: mpsc::Sender<DecisionRecord>,
    pub counters: Arc<GateCounters>,
}

impl DecisionLogger {
    /// Create a new decision logger.
    ///
    /// - `log_dir`: Directory for log files (created if needed)
    /// - `run_id`: Unique run identifier
    /// - `buffer_size`: Channel buffer size (backpressure)
    pub fn new(log_dir: PathBuf, run_id: &str, buffer_size: usize) -> std::io::Result<Self> {
        // Create log directory
        std::fs::create_dir_all(&log_dir)?;

        // Create log file path
        let log_file = log_dir.join(format!("{}_decisions.jsonl", run_id));
        let manifest_file = log_dir.join(format!("{}_manifest.json", run_id));

        // Write manifest
        let manifest = serde_json::json!({
            "run_id": run_id,
            "start_time": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            "log_file": log_file.to_string_lossy(),
        });
        std::fs::write(&manifest_file, serde_json::to_string_pretty(&manifest)?)?;

        // Open log file for append
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)?;

        let (tx, rx) = mpsc::channel(buffer_size);
        let counters = Arc::new(GateCounters::new());

        // Spawn writer task
        let writer_counters = counters.clone();
        tokio::spawn(async move {
            Self::writer_task(rx, file, writer_counters).await;
        });

        tracing::info!("[DECISION_LOG] Logging to {:?}", log_file);

        Ok(Self { tx, counters })
    }

    /// Writer task - batches writes for efficiency.
    async fn writer_task(
        mut rx: mpsc::Receiver<DecisionRecord>,
        file: File,
        _counters: Arc<GateCounters>,
    ) {
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        let mut batch_count = 0u64;
        let flush_interval = tokio::time::interval(std::time::Duration::from_millis(500));
        tokio::pin!(flush_interval);

        loop {
            tokio::select! {
                Some(record) = rx.recv() => {
                    // Serialize and write
                    if let Ok(json) = serde_json::to_string(&record) {
                        let _ = writeln!(writer, "{}", json);
                        batch_count += 1;

                        // Flush every 100 records
                        if batch_count >= 100 {
                            let _ = writer.flush();
                            batch_count = 0;
                        }
                    }
                }
                _ = flush_interval.tick() => {
                    // Periodic flush
                    if batch_count > 0 {
                        let _ = writer.flush();
                        batch_count = 0;
                    }
                }
                else => break,
            }
        }

        // Final flush
        let _ = writer.flush();
    }

    /// Log a decision record (non-blocking).
    pub fn log(&self, record: DecisionRecord) {
        // Extract failed gates for counters
        let failed_gates: Vec<String> = record
            .sniper
            .gates
            .iter()
            .filter(|g| !g.pass)
            .map(|g| g.name.clone())
            .collect();

        let is_warmup = failed_gates.iter().any(|g| g == "WARMUP");
        let is_eligible = record.sniper.eligible;
        let is_accepted = record.intent.accepted.is_some();

        self.counters
            .record_tick(is_warmup, is_eligible, is_accepted, &failed_gates);

        // Non-blocking send (drop if full)
        let _ = self.tx.try_send(record);
    }
}

/// Builder to construct DecisionRecord from engine state.
pub struct DecisionRecordBuilder {
    tick: u64,
    ts_ms: u64,
    symbol: String,
}

impl DecisionRecordBuilder {
    pub fn new(tick: u64, symbol: impl Into<String>) -> Self {
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            tick,
            ts_ms,
            symbol: symbol.into(),
        }
    }

    pub fn build(
        self,
        book: BookLog,
        state: StateLog,
        sniper: SniperLog,
        intent: IntentLog,
        execution: Option<ExecutionLog>,
    ) -> DecisionRecord {
        DecisionRecord {
            ts_ms: self.ts_ms,
            tick: self.tick,
            symbol: self.symbol,
            book,
            state,
            sniper,
            intent,
            execution,
        }
    }
}

/// Helper to extract gate info from refusal reasons.
pub fn extract_gates_from_refusals(
    refusal_reasons: &[crate::paper::state::RefusalReason],
    sniper_config: &crate::paper::sniper::SniperConfig,
    metrics: &crate::paper::state::DecisionMetrics,
    edge_est: f64,
    edge_req: f64,
) -> Vec<GateLog> {
    let mut gates = Vec::new();

    // Track which gates failed
    let mut failed_codes: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for reason in refusal_reasons {
        failed_codes.insert(reason.code);
    }

    // WARMUP
    let warmup_failed = failed_codes.iter().any(|c| c.starts_with("WARMUP"));
    gates.push(GateLog {
        name: "WARMUP".to_string(),
        pass: !warmup_failed,
        lhs: None,
        rhs: None,
        detail: if warmup_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code.starts_with("WARMUP"))
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    // REGIME
    let regime_failed = failed_codes.contains(&"GATE1_REGIME");
    gates.push(GateLog {
        name: "REGIME".to_string(),
        pass: !regime_failed,
        lhs: None,
        rhs: None,
        detail: if regime_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code == "GATE1_REGIME")
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    // CONFIDENCE - No silent poisoning: log None when absent
    let conf_absent = failed_codes.contains(&"GATE1_CONFIDENCE_ABSENT");
    let conf_failed = failed_codes.contains(&"GATE1_CONFIDENCE") || conf_absent;
    gates.push(GateLog {
        name: "CONFIDENCE".to_string(),
        pass: !conf_failed,
        lhs: metrics.confidence, // None if absent, Some(val) if present
        rhs: Some(sniper_config.confidence_min),
        detail: if conf_absent {
            Some("METRIC_ABSENT".to_string())
        } else {
            None
        },
    });

    // D_PERP - REMOVED from sniper gates (R3 requires high d_perp by definition)
    // Log for diagnostics only, always passes. No silent poisoning: log None when absent.
    gates.push(GateLog {
        name: "D_PERP".to_string(),
        pass: true,          // Gate removed - always passes
        lhs: metrics.d_perp, // None if absent, Some(val) if present
        rhs: None,           // No threshold - informational only
        detail: Some("(diagnostic only - gate removed)".to_string()),
    });

    // TOXICITY - No silent poisoning: log None when absent
    let tox_absent = failed_codes.contains(&"GATE2_TOXICITY_ABSENT");
    let tox_failed = failed_codes.contains(&"GATE2_TOXICITY") || tox_absent;
    gates.push(GateLog {
        name: "TOXICITY".to_string(),
        pass: !tox_failed,
        lhs: metrics.toxicity, // None if absent, Some(val) if present
        rhs: Some(sniper_config.toxicity_max),
        detail: if tox_absent {
            Some("METRIC_ABSENT".to_string())
        } else {
            None
        },
    });

    // SPREAD
    let spread_failed = failed_codes.contains(&"GATE2_SPREAD");
    gates.push(GateLog {
        name: "SPREAD".to_string(),
        pass: !spread_failed,
        lhs: None,
        rhs: Some(sniper_config.spread_max_bps),
        detail: if spread_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code == "GATE2_SPREAD")
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    // EDGE
    let edge_failed = failed_codes.contains(&"GATE3_EDGE");
    gates.push(GateLog {
        name: "EDGE".to_string(),
        pass: !edge_failed,
        lhs: Some(edge_est),
        rhs: Some(edge_req),
        detail: None,
    });

    // COOLDOWN
    let cooldown_failed = failed_codes.contains(&"GATE4_COOLDOWN");
    gates.push(GateLog {
        name: "COOLDOWN".to_string(),
        pass: !cooldown_failed,
        lhs: None,
        rhs: None,
        detail: if cooldown_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code == "GATE4_COOLDOWN")
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    // HOURLY_LIMIT
    let hourly_failed = failed_codes.contains(&"GATE4_HOURLY");
    gates.push(GateLog {
        name: "HOURLY_LIMIT".to_string(),
        pass: !hourly_failed,
        lhs: None,
        rhs: None,
        detail: None,
    });

    // IMBALANCE
    let imb_failed = failed_codes.contains(&"GATE5_IMBALANCE");
    gates.push(GateLog {
        name: "IMBALANCE".to_string(),
        pass: !imb_failed,
        lhs: None,
        rhs: Some(sniper_config.imbalance_min_abs),
        detail: if imb_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code == "GATE5_IMBALANCE")
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    // FTI_PERSIST - No silent poisoning: log None when absent
    let fti_failed = failed_codes.contains(&"GATE6_FTI_PERSIST");
    gates.push(GateLog {
        name: "FTI_PERSIST".to_string(),
        pass: !fti_failed,
        lhs: metrics.fti_persist, // None if absent, Some(val) if present
        rhs: Some(sniper_config.fti_persist_min),
        detail: if fti_failed {
            refusal_reasons
                .iter()
                .find(|r| r.code == "GATE6_FTI_PERSIST")
                .map(|r| r.detail.clone())
        } else {
            None
        },
    });

    gates
}
