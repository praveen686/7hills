//! WAL-style structured logging and digests.
//!
//! Spec: Section 9.0 AUDITING AND REPLAYABILITY
//!
//! WAL Events:
//! - FeatureBatchComputed
//! - RegimeComputed
//! - SignalAdmitted / SignalRefused
//! - IntentEmitted
//! - ExecutionReport

use crate::features::{SnapshotFeatures, TradeFlowFeatures};
use crate::fragility::FragilityScore;
use crate::fti::FTIMetrics;
use crate::regime::{Regime, RegimeClassification};
use crate::sealed::STATE_DIM;
use crate::subspace::RegimeMetrics;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// WAL event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WalEvent {
    /// Features computed for a batch/tick.
    FeatureBatchComputed(FeatureBatchEvent),
    /// Regime computed.
    RegimeComputed(RegimeComputedEvent),
    /// Signal admitted for processing.
    SignalAdmitted(SignalEvent),
    /// Signal refused (data quality issue).
    SignalRefused(SignalRefusedEvent),
}

/// Feature batch computed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBatchEvent {
    /// Timestamp in nanoseconds
    pub ts_ns: i64,
    /// Symbol
    pub symbol: String,
    /// Sequence number
    pub seq: u64,
    /// Snapshot features
    pub snapshot: SnapshotFeaturesRecord,
    /// Trade flow features
    pub trade_flow: TradeFlowFeaturesRecord,
    /// FTI metrics
    pub fti: FTIMetricsRecord,
    /// Fragility score
    pub fragility: f64,
    /// Raw state vector (pre-normalization)
    pub raw_state: [f64; STATE_DIM],
    /// Normalized state vector
    pub normalized_state: [f64; STATE_DIM],
}

/// Snapshot features for WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotFeaturesRecord {
    pub mid: f64,
    pub microprice: f64,
    pub imbalance_10: f64,
    pub depth_slope_bid: f64,
    pub depth_slope_ask: f64,
    pub gap_risk: f64,
    pub spread_ticks: f64,
}

impl From<&SnapshotFeatures> for SnapshotFeaturesRecord {
    fn from(sf: &SnapshotFeatures) -> Self {
        Self {
            mid: sf.mid,
            microprice: sf.microprice,
            imbalance_10: sf.imbalance_10,
            depth_slope_bid: sf.depth_slope_bid,
            depth_slope_ask: sf.depth_slope_ask,
            gap_risk: sf.gap_risk,
            spread_ticks: sf.spread_ticks,
        }
    }
}

/// Trade flow features for WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeFlowFeaturesRecord {
    pub signed_volume: f64,
    pub elasticity: f64,
    pub depth_collapse_rate: f64,
}

impl From<&TradeFlowFeatures> for TradeFlowFeaturesRecord {
    fn from(tf: &TradeFlowFeatures) -> Self {
        Self {
            signed_volume: tf.signed_volume,
            elasticity: tf.elasticity,
            depth_collapse_rate: tf.depth_collapse_rate,
        }
    }
}

/// FTI metrics for WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FTIMetricsRecord {
    pub fti_level: f64,
    pub fti_slope: f64,
    pub fti_persist: f64,
}

impl From<&FTIMetrics> for FTIMetricsRecord {
    fn from(fti: &FTIMetrics) -> Self {
        Self {
            fti_level: fti.fti_level,
            fti_slope: fti.fti_slope,
            fti_persist: fti.fti_persist,
        }
    }
}

/// Regime computed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeComputedEvent {
    /// Timestamp in nanoseconds
    pub ts_ns: i64,
    /// Symbol
    pub symbol: String,
    /// Sequence number
    pub seq: u64,
    /// Classified regime
    pub regime: String,
    /// Regime metrics
    pub metrics: RegimeMetricsRecord,
    /// Classification details
    pub classification: RegimeClassificationRecord,
}

/// Regime metrics for WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeMetricsRecord {
    pub d_perp: f64,
    pub v_para: f64,
    pub rho: f64,
}

impl From<&RegimeMetrics> for RegimeMetricsRecord {
    fn from(rm: &RegimeMetrics) -> Self {
        Self {
            d_perp: rm.d_perp,
            v_para: rm.v_para,
            rho: rm.rho,
        }
    }
}

/// Regime classification for WAL record.
/// v1.2: Added confidence breakdown and degraded_reasons for audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeClassificationRecord {
    /// Effective confidence after all penalties (used for sizing)
    pub confidence: f64,
    /// Raw confidence before normalization penalties (always 1.0 per frame)
    pub raw_confidence: f64,
    /// Normalization penalty factor (confidence = raw_confidence * normalization_penalty)
    /// 1.0 = no penalty, 0.68 = DegradedHigh penalty, etc.
    pub normalization_penalty: f64,
    /// Bitmask of degradation reasons (v1.2)
    pub degraded_reasons: u32,
    pub d_perp: f64,
    pub fragility: f64,
    pub fti_persist: f64,
    pub toxicity: f64,
    pub toxicity_persist: f64,
    /// Whether frame was refused (RefuseFrame only)
    pub refused: bool,
}

impl From<&RegimeClassification> for RegimeClassificationRecord {
    fn from(rc: &RegimeClassification) -> Self {
        Self {
            confidence: rc.confidence,
            raw_confidence: 1.0, // Always 1.0 - reset at start of each frame
            normalization_penalty: rc.normalization_penalty,
            degraded_reasons: rc.degraded_reasons,
            d_perp: rc.d_perp,
            fragility: rc.fragility,
            fti_persist: rc.fti_persist,
            toxicity: rc.toxicity,
            toxicity_persist: rc.toxicity_persist,
            refused: rc.refused,
        }
    }
}

/// Signal event (admitted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEvent {
    pub ts_ns: i64,
    pub symbol: String,
    pub seq: u64,
    pub regime: String,
    pub confidence: f64,
}

/// Signal refused event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRefusedEvent {
    pub ts_ns: i64,
    pub symbol: String,
    pub seq: u64,
    pub reason: String,
}

/// Parameters for write_feature_batch (refactored from 9 args).
pub struct FeatureBatchParams<'a> {
    pub ts_ns: i64,
    pub symbol: &'a str,
    pub snapshot: &'a SnapshotFeatures,
    pub trade_flow: &'a TradeFlowFeatures,
    pub fti: &'a FTIMetrics,
    pub fragility: &'a FragilityScore,
    pub raw_state: [f64; STATE_DIM],
    pub normalized_state: [f64; STATE_DIM],
}

/// WAL writer for structured logging.
pub struct WalWriter {
    /// Events buffer
    events: Vec<WalEvent>,
    /// Running digest
    hasher: Sha256,
    /// Sequence counter
    seq: u64,
}

impl WalWriter {
    /// Create a new WAL writer.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            hasher: Sha256::new(),
            seq: 0,
        }
    }

    /// Get next sequence number.
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Write an event.
    pub fn write(&mut self, event: WalEvent) {
        // Update digest with event JSON
        let json = serde_json::to_string(&event).unwrap_or_default();
        self.hasher.update(json.as_bytes());

        self.events.push(event);
    }

    /// Write feature batch event.
    pub fn write_feature_batch(&mut self, params: FeatureBatchParams<'_>) {
        let seq = self.next_seq();
        self.write(WalEvent::FeatureBatchComputed(FeatureBatchEvent {
            ts_ns: params.ts_ns,
            symbol: params.symbol.to_string(),
            seq,
            snapshot: params.snapshot.into(),
            trade_flow: params.trade_flow.into(),
            fti: params.fti.into(),
            fragility: params.fragility.value,
            raw_state: params.raw_state,
            normalized_state: params.normalized_state,
        }));
    }

    /// Write regime computed event.
    pub fn write_regime(
        &mut self,
        ts_ns: i64,
        symbol: &str,
        regime: Regime,
        metrics: &RegimeMetrics,
        classification: &RegimeClassification,
    ) {
        let seq = self.next_seq();
        self.write(WalEvent::RegimeComputed(RegimeComputedEvent {
            ts_ns,
            symbol: symbol.to_string(),
            seq,
            regime: regime.as_str().to_string(),
            metrics: metrics.into(),
            classification: classification.into(),
        }));
    }

    /// Write signal admitted event.
    pub fn write_signal_admitted(
        &mut self,
        ts_ns: i64,
        symbol: &str,
        regime: Regime,
        confidence: f64,
    ) {
        let seq = self.next_seq();
        self.write(WalEvent::SignalAdmitted(SignalEvent {
            ts_ns,
            symbol: symbol.to_string(),
            seq,
            regime: regime.as_str().to_string(),
            confidence,
        }));
    }

    /// Write signal refused event.
    pub fn write_signal_refused(&mut self, ts_ns: i64, symbol: &str, reason: &str) {
        let seq = self.next_seq();
        self.write(WalEvent::SignalRefused(SignalRefusedEvent {
            ts_ns,
            symbol: symbol.to_string(),
            seq,
            reason: reason.to_string(),
        }));
    }

    /// Get all events.
    pub fn events(&self) -> &[WalEvent] {
        &self.events
    }

    /// Get current sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Finalize and get digest.
    pub fn finalize_digest(&self) -> String {
        let hasher = self.hasher.clone();
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Export events as JSON lines.
    pub fn to_jsonl(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for WalWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Run manifest for audit trail.
/// Spec: Section 9.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// Feature schema version
    pub feature_schema_version: String,
    /// Threshold values used (static config)
    pub thresholds: ThresholdManifest,
    /// Calibrated values (computed from warmup data)
    pub calibration: CalibrationManifest,
    /// Session metadata
    pub session: SessionManifest,
    /// Final digest
    pub digest: String,
}

/// Threshold values in manifest (static config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdManifest {
    pub tau_d_perp: f64,
    pub tau_fragility: f64,
    pub tau_fti_persist: f64,
    pub tau_toxicity_persist: f64,
    pub tau_confidence: f64,
}

/// Calibrated values computed from warmup data.
/// These must be deterministic across replays of the same session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationManifest {
    /// Toxicity bucket size in quote units (USDT)
    /// Source: p50(notional/sec) * 10 from warmup period
    pub bucket_size_usdt: f64,
    /// FTI persist threshold (p95 of FTI_level during warmup)
    pub fti_persist_threshold: f64,
    /// Toxicity persist threshold (configured, not calibrated)
    pub toxicity_persist_threshold: f64,
    /// Number of warmup seconds used for bucket_size calibration
    pub warmup_seconds: f64,
    /// Number of samples used for FTI threshold calibration
    pub fti_calibration_samples: usize,
}

/// Session metadata in manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    pub session_id: String,
    pub symbol: String,
    pub start_ts_ns: i64,
    pub end_ts_ns: i64,
    pub total_ticks: u64,
    pub total_regimes: RegimeCountManifest,
}

/// Regime counts in manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeCountManifest {
    pub r0: u64,
    pub r1: u64,
    pub r2: u64,
    pub r3: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_writer() {
        let mut writer = WalWriter::new();

        writer.write_signal_admitted(1000, "BTCUSDT", Regime::R0, 1.0);
        writer.write_signal_refused(2000, "BTCUSDT", "crossed_book");

        assert_eq!(writer.events().len(), 2);
        assert_eq!(writer.seq(), 2);
    }

    #[test]
    fn test_digest() {
        let mut writer = WalWriter::new();
        writer.write_signal_admitted(1000, "TEST", Regime::R0, 1.0);

        let digest = writer.finalize_digest();
        assert_eq!(digest.len(), 64); // SHA-256 hex = 64 chars
    }
}
