//! WAL-compatible regime events.
//!
//! All events are designed for deterministic serialization and replay parity.

use crate::canonical::SubspaceDigest;
use crate::cpd::ShiftDirection;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Event emitted when a new subspace is computed.
///
/// This is the primary regime state event, emitted at each update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeSubspaceEvent {
    /// Exchange timestamp
    pub ts: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Rolling window length used
    pub window_len: usize,
    /// Subspace dimension (k in Gr(k,n))
    pub k: usize,
    /// Ambient dimension (n in Gr(k,n))
    pub n: usize,
    /// SHA-256 digest of canonical subspace representation
    pub subspace_digest: SubspaceDigest,
    /// Distance to previous subspace (mantissa)
    pub distance_to_prev_mantissa: i64,
    /// Distance exponent
    pub distance_exponent: i8,
}

impl RegimeSubspaceEvent {
    /// Compute canonical digest of this event.
    pub fn canonical_digest(&self) -> String {
        let mut hasher = Sha256::new();

        // Include all fields in canonical order
        hasher.update(self.ts.timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        hasher.update(self.symbol.as_bytes());
        hasher.update((self.window_len as u32).to_le_bytes());
        hasher.update((self.k as u32).to_le_bytes());
        hasher.update((self.n as u32).to_le_bytes());
        hasher.update(self.subspace_digest.as_str().as_bytes());
        hasher.update(self.distance_to_prev_mantissa.to_le_bytes());
        hasher.update([self.distance_exponent as u8]);

        hex::encode(hasher.finalize())
    }
}

/// Event emitted when a regime shift is detected.
///
/// This is a "hard signal" indicating the regime has changed,
/// independent of label assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeShiftEvent {
    /// Detection timestamp
    pub ts: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Distance that triggered detection (mantissa)
    pub distance_mantissa: i64,
    /// Distance exponent
    pub distance_exponent: i8,
    /// CUSUM statistic at detection (mantissa)
    pub cusum_stat_mantissa: i64,
    /// Detection threshold (mantissa)
    pub threshold_mantissa: i64,
    /// Observations since last shift
    pub observations_since_last: usize,
    /// Direction of shift
    pub direction: ShiftDirection,
}

impl RegimeShiftEvent {
    /// Compute canonical digest of this event.
    pub fn canonical_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.ts.timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        hasher.update(self.symbol.as_bytes());
        hasher.update(self.distance_mantissa.to_le_bytes());
        hasher.update([self.distance_exponent as u8]);
        hasher.update(self.cusum_stat_mantissa.to_le_bytes());
        hasher.update(self.threshold_mantissa.to_le_bytes());
        hasher.update((self.observations_since_last as u32).to_le_bytes());
        hasher.update([match self.direction {
            ShiftDirection::Increasing => 1u8,
            ShiftDirection::Decreasing => 0u8,
        }]);

        hex::encode(hasher.finalize())
    }
}

/// Event emitted when a regime label is assigned.
///
/// Labels are assigned via prototype matching after subspace computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeLabelEvent {
    /// Timestamp
    pub ts: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Assigned regime label
    pub regime_id: String,
    /// Confidence (margin between best and second-best, mantissa with exp=-4)
    pub confidence_mantissa: i64,
    /// Distance to best prototype (mantissa)
    pub distance_best_mantissa: i64,
    /// Distance to second-best prototype (mantissa)
    pub distance_second_mantissa: i64,
    /// Distance exponent
    pub distance_exponent: i8,
    /// Method used for classification
    pub method: ClassificationMethod,
    /// Subspace digest (for correlation)
    pub subspace_digest: SubspaceDigest,
}

/// Classification method used for regime labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationMethod {
    /// Nearest prototype matching
    Prototype,
    /// Change-point detection
    Cpd,
    /// Hybrid (CPD for shift detection, prototype for labeling)
    Hybrid,
    /// Heuristic-based (using feature statistics)
    Heuristic,
}

impl RegimeLabelEvent {
    /// Compute canonical digest of this event.
    pub fn canonical_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.ts.timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        hasher.update(self.symbol.as_bytes());
        hasher.update(self.regime_id.as_bytes());
        hasher.update(self.confidence_mantissa.to_le_bytes());
        hasher.update(self.distance_best_mantissa.to_le_bytes());
        hasher.update(self.distance_second_mantissa.to_le_bytes());
        hasher.update([self.distance_exponent as u8]);
        hasher.update([match self.method {
            ClassificationMethod::Prototype => 0u8,
            ClassificationMethod::Cpd => 1u8,
            ClassificationMethod::Hybrid => 2u8,
            ClassificationMethod::Heuristic => 3u8,
        }]);
        hasher.update(self.subspace_digest.as_str().as_bytes());

        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subspace_event_digest_deterministic() {
        let event1 = RegimeSubspaceEvent {
            ts: DateTime::parse_from_rfc3339("2026-01-30T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            symbol: "NIFTY".to_string(),
            window_len: 64,
            k: 3,
            n: 6,
            subspace_digest: SubspaceDigest("abc123".to_string()),
            distance_to_prev_mantissa: 1500,
            distance_exponent: -4,
        };

        let event2 = event1.clone();

        assert_eq!(event1.canonical_digest(), event2.canonical_digest());
    }

    #[test]
    fn test_shift_event_serialization() {
        let event = RegimeShiftEvent {
            ts: Utc::now(),
            symbol: "BANKNIFTY".to_string(),
            distance_mantissa: 2500,
            distance_exponent: -4,
            cusum_stat_mantissa: 5100,
            threshold_mantissa: 5000,
            observations_since_last: 42,
            direction: ShiftDirection::Increasing,
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: RegimeShiftEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(event.distance_mantissa, deserialized.distance_mantissa);
        assert_eq!(event.direction, deserialized.direction);
    }
}
