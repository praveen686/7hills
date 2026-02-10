//! Subspace tracking using exponentially weighted covariance and eigendecomposition.
//!
//! Spec: Section 6.3 Subspace Tracking (Default, SEALED)
//!
//! - Rank r = 4
//! - EW covariance decay λ = 0.995
//! - Eigensolve cadence = every 250ms OR 500 batches
//! - Diagonal jitter = 1e-6

use crate::sealed::{
    DIAGONAL_JITTER, EIGENSOLVE_CADENCE_BATCHES, EIGENSOLVE_CADENCE_MS, EW_DECAY_LAMBDA, STATE_DIM,
    SUBSPACE_RANK,
};

/// Subspace tracking state.
/// Maintains EW covariance matrix and principal subspace.
pub struct SubspaceTracker {
    /// Exponentially weighted mean
    mean: [f64; STATE_DIM],
    /// Exponentially weighted covariance (upper triangular stored as flat array)
    covariance: [[f64; STATE_DIM]; STATE_DIM],
    /// Principal subspace basis (columns are eigenvectors)
    /// U is STATE_DIM x SUBSPACE_RANK
    basis: [[f64; SUBSPACE_RANK]; STATE_DIM],
    /// Previous basis for rotation calculation
    prev_basis: [[f64; SUBSPACE_RANK]; STATE_DIM],
    /// Eigenvalues (top r)
    eigenvalues: [f64; SUBSPACE_RANK],
    /// EW decay factor
    lambda: f64,
    /// Sample count for initialization
    sample_count: u64,
    /// Batch count since last eigensolve
    batch_count: usize,
    /// Last eigensolve timestamp (ns)
    last_eigensolve_ns: i64,
    /// Eigensolve cadence (ns)
    eigensolve_cadence_ns: i64,
}

impl SubspaceTracker {
    /// Create a new subspace tracker.
    pub fn new() -> Self {
        // Initialize basis to identity-like structure
        let mut basis = [[0.0; SUBSPACE_RANK]; STATE_DIM];
        for (i, row) in basis
            .iter_mut()
            .enumerate()
            .take(SUBSPACE_RANK.min(STATE_DIM))
        {
            row[i] = 1.0;
        }

        Self {
            mean: [0.0; STATE_DIM],
            covariance: [[0.0; STATE_DIM]; STATE_DIM],
            basis,
            prev_basis: basis,
            eigenvalues: [1.0; SUBSPACE_RANK],
            lambda: EW_DECAY_LAMBDA,
            sample_count: 0,
            batch_count: 0,
            last_eigensolve_ns: 0,
            eigensolve_cadence_ns: EIGENSOLVE_CADENCE_MS as i64 * 1_000_000,
        }
    }

    /// Update the EW covariance with a new sample.
    pub fn update(&mut self, ts_ns: i64, x: &[f64; STATE_DIM]) {
        self.sample_count += 1;
        self.batch_count += 1;

        // Update EW mean
        for (mean_elem, &x_elem) in self.mean.iter_mut().zip(x.iter()) {
            *mean_elem = self.lambda * *mean_elem + (1.0 - self.lambda) * x_elem;
        }

        // Compute centered sample
        let mut centered = [0.0; STATE_DIM];
        for i in 0..STATE_DIM {
            centered[i] = x[i] - self.mean[i];
        }

        // Update EW covariance: C = λC + (1-λ)xxᵀ
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                self.covariance[i][j] = self.lambda * self.covariance[i][j]
                    + (1.0 - self.lambda) * centered[i] * centered[j];
            }
        }

        // Add diagonal jitter for numerical stability
        for i in 0..STATE_DIM {
            self.covariance[i][i] += DIAGONAL_JITTER;
        }

        // Check if eigensolve is needed
        let elapsed_ns = ts_ns - self.last_eigensolve_ns;
        if elapsed_ns >= self.eigensolve_cadence_ns
            || self.batch_count >= EIGENSOLVE_CADENCE_BATCHES
        {
            self.eigensolve();
            self.last_eigensolve_ns = ts_ns;
            self.batch_count = 0;
        }
    }

    /// Perform eigendecomposition of covariance matrix.
    /// Uses power iteration for simplicity (correctness over speed).
    fn eigensolve(&mut self) {
        // Save previous basis for rotation calculation
        self.prev_basis = self.basis;

        // Power iteration to find top r eigenvectors
        // This is a simple implementation for correctness
        let mut work_matrix = self.covariance;

        for k in 0..SUBSPACE_RANK {
            // Initialize random-ish vector
            let mut v = [0.0; STATE_DIM];
            for (i, v_elem) in v.iter_mut().enumerate() {
                *v_elem = 1.0 / ((i + k + 1) as f64).sqrt();
            }
            Self::normalize_vector(&mut v);

            // Power iteration
            for _ in 0..100 {
                // Maximum iterations for convergence
                let mut new_v = [0.0; STATE_DIM];

                // Matrix-vector multiply
                for i in 0..STATE_DIM {
                    for j in 0..STATE_DIM {
                        new_v[i] += work_matrix[i][j] * v[j];
                    }
                }

                // Normalize
                let norm = Self::normalize_vector(&mut new_v);

                // Check convergence
                let mut diff = 0.0;
                for i in 0..STATE_DIM {
                    diff += (new_v[i] - v[i]).powi(2);
                }

                v = new_v;

                if diff.sqrt() < 1e-10 {
                    break;
                }

                // Store eigenvalue estimate
                if k < SUBSPACE_RANK {
                    self.eigenvalues[k] = norm;
                }
            }

            // Store eigenvector
            for (basis_row, &v_elem) in self.basis.iter_mut().zip(v.iter()) {
                basis_row[k] = v_elem;
            }

            // Deflate: remove this component from work matrix
            // A = A - λvvᵀ
            let eigenvalue = self.eigenvalues[k];
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    work_matrix[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
    }

    /// Normalize a vector in place, return the original norm.
    fn normalize_vector(v: &mut [f64; STATE_DIM]) -> f64 {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        norm
    }

    /// Project a vector onto the subspace (parallel component).
    /// x_para = U * Uᵀ * x
    pub fn project_parallel(&self, x: &[f64; STATE_DIM]) -> [f64; STATE_DIM] {
        // First compute Uᵀx (r-dimensional)
        let mut coeffs = [0.0; SUBSPACE_RANK];
        for (k, coeff) in coeffs.iter_mut().enumerate() {
            for (basis_row, &x_val) in self.basis.iter().zip(x.iter()) {
                *coeff += basis_row[k] * x_val;
            }
        }

        // Then compute U * coeffs
        let mut result = [0.0; STATE_DIM];
        for (i, res) in result.iter_mut().enumerate() {
            for (k, &coeff) in coeffs.iter().enumerate() {
                *res += self.basis[i][k] * coeff;
            }
        }

        result
    }

    /// Get perpendicular component (off-manifold).
    /// x_perp = x - x_para
    pub fn project_perpendicular(&self, x: &[f64; STATE_DIM]) -> [f64; STATE_DIM] {
        let para = self.project_parallel(x);
        let mut perp = [0.0; STATE_DIM];
        for (p, (&x_val, &para_val)) in perp.iter_mut().zip(x.iter().zip(para.iter())) {
            *p = x_val - para_val;
        }
        perp
    }

    /// Compute off-manifold distance.
    /// Spec: Section 6.4 - d_perp = ||x_perp||₂
    pub fn off_manifold_distance(&self, x: &[f64; STATE_DIM]) -> f64 {
        let perp = self.project_perpendicular(x);
        perp.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Compute tangent speed.
    /// Spec: Section 6.4 - v_para = ||x_para(t) - x_para(t-Δ)||₂
    pub fn tangent_speed(&self, x_current: &[f64; STATE_DIM], x_prev: &[f64; STATE_DIM]) -> f64 {
        let para_current = self.project_parallel(x_current);
        let para_prev = self.project_parallel(x_prev);

        para_current
            .iter()
            .zip(para_prev.iter())
            .map(|(&c, &p)| (c - p).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute subspace rotation.
    /// Spec: Section 6.4 - ρ = ||U(t)ᵀ U(t-Δ) - I||_F
    pub fn subspace_rotation(&self) -> f64 {
        // Compute UᵀU_prev (r x r matrix)
        let mut product = [[0.0; SUBSPACE_RANK]; SUBSPACE_RANK];

        for (i, row) in product.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for (basis_row, prev_row) in self.basis.iter().zip(self.prev_basis.iter()) {
                    *cell += basis_row[i] * prev_row[j];
                }
            }
        }

        // Compute ||product - I||_F
        let frobenius_sq: f64 = product
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter().enumerate().map(move |(j, &val)| {
                    let identity_val = if i == j { 1.0 } else { 0.0 };
                    (val - identity_val).powi(2)
                })
            })
            .sum();

        frobenius_sq.sqrt()
    }

    /// Get current eigenvalues.
    pub fn eigenvalues(&self) -> &[f64; SUBSPACE_RANK] {
        &self.eigenvalues
    }

    /// Get sample count.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

impl Default for SubspaceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Regime metrics computed from subspace tracking.
/// Spec: Section 6.4
#[derive(Debug, Clone, Default)]
pub struct RegimeMetrics {
    /// Off-manifold distance: d_perp = ||x_perp||₂
    pub d_perp: f64,
    /// Tangent speed: v_para = ||x_para(t) - x_para(t-Δ)||₂
    pub v_para: f64,
    /// Subspace rotation: ρ = ||U(t)ᵀ U(t-Δ) - I||_F
    pub rho: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subspace_basic() {
        let tracker = SubspaceTracker::new();
        assert_eq!(tracker.sample_count(), 0);
    }

    #[test]
    fn test_projection() {
        let tracker = SubspaceTracker::new();

        // With identity-like initial basis, projection should preserve first r dims
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let para = tracker.project_parallel(&x);

        // First SUBSPACE_RANK dimensions should be preserved
        for i in 0..SUBSPACE_RANK {
            assert!((para[i] - x[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_perpendicular() {
        let tracker = SubspaceTracker::new();

        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let perp = tracker.project_perpendicular(&x);

        // With identity-like initial basis, perpendicular should be dims > r
        for p in perp.iter().take(SUBSPACE_RANK) {
            assert!(p.abs() < 1e-10);
        }
        for i in SUBSPACE_RANK..STATE_DIM {
            assert!((perp[i] - x[i]).abs() < 1e-10);
        }
    }
}
