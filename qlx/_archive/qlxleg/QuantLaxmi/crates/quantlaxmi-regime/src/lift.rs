//! RegimeLift: Geometric lifting from features to Grassmann manifold.
//!
//! Pipeline: features → rolling covariance → SVD → canonical subspace U_t ∈ Gr(k,n)

use crate::canonical::CanonicalSubspace;
use crate::features::FeatureVector;
use nalgebra::{DMatrix, SVD};
use std::collections::VecDeque;

/// Configuration for the regime lift.
#[derive(Debug, Clone)]
pub struct RegimeLiftConfig {
    /// Number of features (n in Gr(k,n))
    pub n_features: usize,
    /// Subspace dimension (k in Gr(k,n))
    pub subspace_dim: usize,
    /// Rolling window size for covariance
    pub window_size: usize,
}

impl Default for RegimeLiftConfig {
    fn default() -> Self {
        Self {
            n_features: 6,
            subspace_dim: 3,
            window_size: 64,
        }
    }
}

/// Regime lift engine.
///
/// Maintains a rolling window of feature vectors and computes the
/// covariance matrix, then extracts the top-k eigenvectors as a
/// point on the Grassmann manifold.
pub struct RegimeLift {
    config: RegimeLiftConfig,
    /// Rolling window of feature vectors (as f64 for matrix ops)
    window: VecDeque<[f64; 6]>,
    /// Running sum for mean calculation
    sum: [f64; 6],
    /// Running sum of squares for covariance (flattened upper triangle)
    sum_sq: Vec<f64>,
}

impl RegimeLift {
    /// Create a new regime lift with the given configuration.
    pub fn new(config: RegimeLiftConfig) -> Self {
        let n = config.n_features;
        let window_cap = config.window_size + 1;
        Self {
            window: VecDeque::with_capacity(window_cap),
            sum: [0.0; 6],
            sum_sq: vec![0.0; n * n],
            config,
        }
    }

    /// Update with a new feature vector and return the subspace if ready.
    pub fn update(&mut self, features: &FeatureVector) -> Option<CanonicalSubspace> {
        let x = features.to_dense();

        // Add to rolling sums
        for (i, xi) in x.iter().enumerate() {
            self.sum[i] += xi;
        }
        for (i, xi) in x.iter().enumerate() {
            for (j, xj) in x.iter().enumerate() {
                self.sum_sq[i * 6 + j] += xi * xj;
            }
        }

        self.window.push_back(x);

        // If window is full, remove oldest and update sums
        if self.window.len() > self.config.window_size {
            if let Some(old) = self.window.pop_front() {
                for (i, oi) in old.iter().enumerate() {
                    self.sum[i] -= oi;
                }
                for (i, oi) in old.iter().enumerate() {
                    for (j, oj) in old.iter().enumerate() {
                        self.sum_sq[i * 6 + j] -= oi * oj;
                    }
                }
            }
        }

        // Only compute subspace when window is full
        if self.window.len() < self.config.window_size {
            return None;
        }

        // Compute covariance matrix
        let cov = self.compute_covariance();

        // Compute SVD and extract top-k
        self.compute_subspace(&cov)
    }

    /// Compute covariance matrix from running sums.
    fn compute_covariance(&self) -> DMatrix<f64> {
        let n = self.window.len() as f64;
        let mut cov = DMatrix::zeros(6, 6);

        for i in 0..6 {
            for j in 0..6 {
                // Cov(X,Y) = E[XY] - E[X]E[Y]
                let e_xy = self.sum_sq[i * 6 + j] / n;
                let e_x = self.sum[i] / n;
                let e_y = self.sum[j] / n;
                cov[(i, j)] = e_xy - e_x * e_y;
            }
        }

        cov
    }

    /// Compute canonical subspace from covariance matrix via SVD.
    fn compute_subspace(&self, cov: &DMatrix<f64>) -> Option<CanonicalSubspace> {
        // SVD of covariance gives eigenvectors as singular vectors
        let svd = SVD::new(cov.clone(), true, true);

        let u = svd.u?;
        let singular_values = svd.singular_values;

        // Extract top-k columns
        let k = self.config.subspace_dim.min(6);
        let mut basis = Vec::with_capacity(k * 6);

        for col in 0..k {
            let column = u.column(col);
            for row in 0..6 {
                basis.push(column[row]);
            }
        }

        // Get eigenvalues (singular values of symmetric matrix)
        let eigenvalues: Vec<f64> = singular_values.iter().take(k).cloned().collect();

        Some(CanonicalSubspace::from_basis(basis, eigenvalues, 6, k))
    }

    /// Reset state for a new symbol/session.
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = [0.0; 6];
        self.sum_sq.fill(0.0);
    }

    /// Check if the lift has enough data.
    pub fn is_ready(&self) -> bool {
        self.window.len() >= self.config.window_size
    }

    /// Get current window fill level.
    pub fn fill_level(&self) -> (usize, usize) {
        (self.window.len(), self.config.window_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature_vector(mid_return: i64, imbalance: i64) -> FeatureVector {
        FeatureVector::new(mid_return, imbalance, 20, 15, 10000, 5000)
    }

    #[test]
    fn test_lift_needs_full_window() {
        let config = RegimeLiftConfig {
            n_features: 6,
            subspace_dim: 3,
            window_size: 10,
        };
        let mut lift = RegimeLift::new(config);

        // First 9 updates should return None
        for i in 0..9 {
            let fv = make_feature_vector(i as i64 * 10, 0);
            assert!(lift.update(&fv).is_none());
        }

        // 10th update should return a subspace
        let fv = make_feature_vector(100, 0);
        let result = lift.update(&fv);
        assert!(result.is_some());
    }

    #[test]
    fn test_lift_deterministic() {
        let config = RegimeLiftConfig {
            n_features: 6,
            subspace_dim: 3,
            window_size: 10,
        };

        let mut lift1 = RegimeLift::new(config.clone());
        let mut lift2 = RegimeLift::new(config);

        // Feed identical data
        for i in 0..15 {
            let fv = make_feature_vector(i as i64 * 10 + 5, (i as i64 - 7) * 100);
            let r1 = lift1.update(&fv);
            let r2 = lift2.update(&fv);

            match (r1, r2) {
                (Some(s1), Some(s2)) => {
                    // Digests should match
                    assert_eq!(s1.digest(), s2.digest());
                }
                (None, None) => {}
                _ => panic!("Lift results diverged"),
            }
        }
    }

    #[test]
    fn test_subspace_dimensions() {
        let config = RegimeLiftConfig {
            n_features: 6,
            subspace_dim: 2, // Only top-2
            window_size: 10,
        };
        let mut lift = RegimeLift::new(config);

        for i in 0..10 {
            let fv = make_feature_vector(i as i64 * 10, i as i64 * 5);
            lift.update(&fv);
        }

        let fv = make_feature_vector(100, 50);
        let subspace = lift.update(&fv).unwrap();

        assert_eq!(subspace.n(), 6);
        assert_eq!(subspace.k(), 2);
    }
}
