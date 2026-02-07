//! Regime prototypes for classification.
//!
//! Prototypes are reference subspaces representing known regimes.
//! Classification finds the nearest prototype to the current subspace.

use crate::canonical::CanonicalSubspace;
use crate::events::{ClassificationMethod, RegimeLabelEvent};
use crate::grassmann::grassmann_distance;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Standard regime labels for microstructure trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeLabel {
    /// Low volatility, tight spreads, stable book
    Quiet,
    /// Directional drift with widening spreads
    TrendImpulse,
    /// High-frequency mean-reversion microstructure
    MeanReversionChop,
    /// Wide spreads, unstable book, low liquidity
    LiquidityDrought,
    /// Sharp volatility spike, possible news event
    EventShock,
    /// Unknown/unclassified regime
    Unknown,
}

impl RegimeLabel {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::TrendImpulse => "trend_impulse",
            Self::MeanReversionChop => "mean_reversion_chop",
            Self::LiquidityDrought => "liquidity_drought",
            Self::EventShock => "event_shock",
            Self::Unknown => "unknown",
        }
    }

    /// Parse from string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "quiet" => Self::Quiet,
            "trend_impulse" | "trend" => Self::TrendImpulse,
            "mean_reversion_chop" | "chop" | "mean_reversion" => Self::MeanReversionChop,
            "liquidity_drought" | "drought" => Self::LiquidityDrought,
            "event_shock" | "shock" | "event" => Self::EventShock,
            _ => Self::Unknown,
        }
    }
}

/// A regime prototype: a reference subspace representing a known regime.
#[derive(Debug, Clone)]
pub struct RegimePrototype {
    /// Human-readable label
    pub label: RegimeLabel,
    /// Unique ID for this prototype
    pub id: String,
    /// Reference subspace
    pub subspace: CanonicalSubspace,
    /// Description
    pub description: String,
    /// Source (e.g., "manual", "learned_from_session_xyz")
    pub source: String,
}

impl RegimePrototype {
    /// Create a new prototype.
    pub fn new(
        label: RegimeLabel,
        id: impl Into<String>,
        subspace: CanonicalSubspace,
        description: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            label,
            id: id.into(),
            subspace,
            description: description.into(),
            source: source.into(),
        }
    }
}

/// Bank of regime prototypes for classification.
#[derive(Debug, Default)]
pub struct PrototypeBank {
    prototypes: Vec<RegimePrototype>,
    /// Distance exponent for comparisons
    distance_exponent: i8,
    /// Minimum confidence margin for label assignment (mantissa)
    min_confidence_mantissa: i64,
}

impl PrototypeBank {
    /// Create a new empty prototype bank.
    pub fn new() -> Self {
        Self {
            prototypes: Vec::new(),
            distance_exponent: -4,
            min_confidence_mantissa: 2000, // 0.2 margin
        }
    }

    /// Create with custom settings.
    pub fn with_settings(distance_exponent: i8, min_confidence_mantissa: i64) -> Self {
        Self {
            prototypes: Vec::new(),
            distance_exponent,
            min_confidence_mantissa,
        }
    }

    /// Add a prototype.
    pub fn add(&mut self, prototype: RegimePrototype) {
        self.prototypes.push(prototype);
    }

    /// Check if the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }

    /// Number of prototypes.
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Classify a subspace by finding nearest prototype.
    ///
    /// Returns a label event if confidence is above threshold.
    pub fn classify(
        &self,
        ts: DateTime<Utc>,
        symbol: &str,
        subspace: &CanonicalSubspace,
    ) -> Option<RegimeLabelEvent> {
        if self.prototypes.is_empty() {
            return None;
        }

        // Compute distances to all prototypes
        let mut distances: Vec<(usize, i64)> = self
            .prototypes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dist = grassmann_distance(&p.subspace, subspace, self.distance_exponent);
                (i, dist)
            })
            .collect();

        // Sort by distance ascending
        distances.sort_by_key(|&(_, d)| d);

        let (best_idx, best_distance) = distances[0];
        let best_prototype = &self.prototypes[best_idx];

        // Get second-best distance (or max if only one prototype)
        let second_distance = if distances.len() > 1 {
            distances[1].1
        } else {
            i64::MAX
        };

        // Compute confidence as margin between best and second-best
        let confidence = if second_distance == i64::MAX {
            10000 // Max confidence if only one prototype
        } else {
            second_distance.saturating_sub(best_distance)
        };

        // Only emit label if confidence exceeds threshold
        if confidence < self.min_confidence_mantissa {
            return None;
        }

        Some(RegimeLabelEvent {
            ts,
            symbol: symbol.to_string(),
            regime_id: best_prototype.label.as_str().to_string(),
            confidence_mantissa: confidence,
            distance_best_mantissa: best_distance,
            distance_second_mantissa: second_distance,
            distance_exponent: self.distance_exponent,
            method: ClassificationMethod::Prototype,
            subspace_digest: subspace.digest(),
        })
    }

    /// Get all prototypes.
    pub fn prototypes(&self) -> &[RegimePrototype] {
        &self.prototypes
    }

    /// Find prototype by ID.
    pub fn find(&self, id: &str) -> Option<&RegimePrototype> {
        self.prototypes.iter().find(|p| p.id == id)
    }
}

/// Builder for creating regime prototypes from observed data.
pub struct PrototypeBuilder {
    /// Collected subspaces for this regime
    subspaces: Vec<CanonicalSubspace>,
    /// Label for this regime
    label: RegimeLabel,
    /// ID for the prototype
    id: String,
    /// Description
    description: String,
}

impl PrototypeBuilder {
    /// Create a new builder.
    pub fn new(label: RegimeLabel, id: impl Into<String>) -> Self {
        Self {
            subspaces: Vec::new(),
            label,
            id: id.into(),
            description: String::new(),
        }
    }

    /// Add a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add an observed subspace.
    pub fn add_observation(&mut self, subspace: CanonicalSubspace) {
        self.subspaces.push(subspace);
    }

    /// Build the prototype using centroid of observations.
    ///
    /// For Grassmann manifolds, we use Karcher mean (iterative).
    /// For simplicity, we use the observation with minimum total distance
    /// to all others (medoid).
    pub fn build(mut self, source: impl Into<String>) -> Option<RegimePrototype> {
        if self.subspaces.is_empty() {
            return None;
        }

        if self.subspaces.len() == 1 {
            // Safe: just checked len() == 1, so pop() will succeed
            return Some(RegimePrototype::new(
                self.label,
                self.id,
                self.subspaces.pop().expect("len checked == 1"),
                self.description,
                source,
            ));
        }

        // Find medoid: subspace with minimum total distance to all others
        let mut best_idx = 0;
        let mut best_total_dist = i64::MAX;

        for i in 0..self.subspaces.len() {
            let total_dist: i64 = self
                .subspaces
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, s)| grassmann_distance(&self.subspaces[i], s, -4))
                .sum();

            if total_dist < best_total_dist {
                best_total_dist = total_dist;
                best_idx = i;
            }
        }

        // Safe: best_idx was computed from valid indices via enumerate()
        Some(RegimePrototype::new(
            self.label,
            self.id,
            self.subspaces.swap_remove(best_idx),
            self.description,
            source,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_subspace(basis: Vec<f64>, k: usize) -> CanonicalSubspace {
        let n = basis.len() / k;
        let eigenvalues = vec![1.0; k];
        CanonicalSubspace::from_basis(basis, eigenvalues, n, k)
    }

    #[test]
    fn test_prototype_bank_empty() {
        let bank = PrototypeBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
    }

    #[test]
    fn test_classification_single_prototype() {
        let mut bank = PrototypeBank::new();

        let proto_subspace = make_subspace(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2);
        bank.add(RegimePrototype::new(
            RegimeLabel::Quiet,
            "quiet_01",
            proto_subspace,
            "Test quiet regime",
            "test",
        ));

        let test_subspace = make_subspace(vec![0.99, 0.1, 0.0, -0.1, 0.99, 0.0], 2);

        let result = bank.classify(Utc::now(), "TEST", &test_subspace);
        assert!(result.is_some());

        let event = result.unwrap();
        assert_eq!(event.regime_id, "quiet"); // Uses label, not prototype ID
    }

    #[test]
    fn test_classification_nearest_prototype() {
        let mut bank = PrototypeBank::with_settings(-4, 0); // Zero confidence threshold for testing

        // Add two prototypes - more orthogonal for clearer distinction
        let quiet_subspace = make_subspace(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2);
        bank.add(RegimePrototype::new(
            RegimeLabel::Quiet,
            "quiet",
            quiet_subspace,
            "Quiet",
            "test",
        ));

        // 45-degree rotation to make clearly different
        let trend_subspace = make_subspace(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 2);
        bank.add(RegimePrototype::new(
            RegimeLabel::TrendImpulse,
            "trend",
            trend_subspace,
            "Trend",
            "test",
        ));

        // Test subspace very close to quiet (nearly identity)
        let test_near_quiet = make_subspace(vec![0.999, 0.01, 0.0, -0.01, 0.999, 0.0], 2);
        let result = bank.classify(Utc::now(), "TEST", &test_near_quiet);

        assert!(result.is_some());
        assert_eq!(result.unwrap().regime_id, "quiet");
    }

    #[test]
    fn test_regime_label_parsing() {
        assert_eq!(RegimeLabel::parse("quiet"), RegimeLabel::Quiet);
        assert_eq!(RegimeLabel::parse("QUIET"), RegimeLabel::Quiet);
        assert_eq!(RegimeLabel::parse("trend"), RegimeLabel::TrendImpulse);
        assert_eq!(RegimeLabel::parse("unknown_xyz"), RegimeLabel::Unknown);
    }
}
