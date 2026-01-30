//! Canonical subspace representation for Grassmann manifold.
//!
//! Ensures deterministic representation by:
//! 1. Fixing sign ambiguity (largest-magnitude element positive per column)
//! 2. Ordering by eigenvalue descending
//! 3. Quantizing to fixed-point int32

use sha2::{Digest, Sha256};

/// Quantization scale for basis vectors (10^6)
const QUANT_SCALE: f64 = 1_000_000.0;

/// Canonical digest of a subspace.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubspaceDigest(pub String);

impl SubspaceDigest {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Canonical representation of a subspace U ∈ Gr(k,n).
///
/// The subspace is stored as a k×n matrix (k columns of n-dimensional vectors),
/// with deterministic sign/ordering and quantized to int32.
#[derive(Debug, Clone)]
pub struct CanonicalSubspace {
    /// Quantized basis vectors (column-major: [col0_row0, col0_row1, ..., col0_rowN, col1_row0, ...])
    basis_quantized: Vec<i32>,
    /// Eigenvalues (descending order)
    eigenvalues: Vec<f64>,
    /// Original dimension (n)
    n: usize,
    /// Subspace dimension (k)
    k: usize,
    /// Precomputed digest
    digest: SubspaceDigest,
}

impl CanonicalSubspace {
    /// Create a canonical subspace from a raw basis.
    ///
    /// # Arguments
    /// * `basis` - Raw basis vectors as f64, column-major (k columns of n elements)
    /// * `eigenvalues` - Eigenvalues corresponding to each column
    /// * `n` - Original space dimension
    /// * `k` - Subspace dimension
    pub fn from_basis(mut basis: Vec<f64>, eigenvalues: Vec<f64>, n: usize, k: usize) -> Self {
        assert_eq!(basis.len(), n * k, "Basis size mismatch");
        assert_eq!(eigenvalues.len(), k, "Eigenvalue count mismatch");

        // Step 1: Fix sign ambiguity per column
        // For each column, ensure the largest-magnitude element is positive
        for col in 0..k {
            let col_start = col * n;
            let col_end = col_start + n;

            // Find largest-magnitude element value
            let max_val = basis[col_start..col_end]
                .iter()
                .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                .copied()
                .unwrap_or(0.0);

            // If negative, flip the entire column
            if max_val < 0.0 {
                for val in basis[col_start..col_end].iter_mut() {
                    *val = -*val;
                }
            }

            // Tie-breaker for sign: if max is zero or very small, use lexicographic
            if max_val.abs() < 1e-10 {
                // Find first non-zero and make it positive
                let should_flip = basis[col_start..col_end]
                    .iter()
                    .find(|v| v.abs() > 1e-10)
                    .map(|&v| v < 0.0)
                    .unwrap_or(false);

                if should_flip {
                    for val in basis[col_start..col_end].iter_mut() {
                        *val = -*val;
                    }
                }
            }
        }

        // Step 2: Order by eigenvalue descending (already done by SVD typically)
        // If eigenvalues are equal within epsilon, use lexicographic tie-break
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| {
            let ev_cmp = eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal);
            if ev_cmp != std::cmp::Ordering::Equal {
                return ev_cmp;
            }
            // Tie-break by lexicographic order of quantized column
            let col_a: Vec<i32> = basis[a * n..(a + 1) * n]
                .iter()
                .map(|&v| (v * QUANT_SCALE) as i32)
                .collect();
            let col_b: Vec<i32> = basis[b * n..(b + 1) * n]
                .iter()
                .map(|&v| (v * QUANT_SCALE) as i32)
                .collect();
            col_a.cmp(&col_b)
        });

        // Reorder basis and eigenvalues
        let mut ordered_basis = vec![0.0; n * k];
        let mut ordered_eigenvalues = vec![0.0; k];
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for row in 0..n {
                ordered_basis[new_idx * n + row] = basis[old_idx * n + row];
            }
            ordered_eigenvalues[new_idx] = eigenvalues[old_idx];
        }

        // Step 3: Quantize to int32
        let basis_quantized: Vec<i32> = ordered_basis
            .iter()
            .map(|&v| (v * QUANT_SCALE).round() as i32)
            .collect();

        // Compute digest
        let digest = Self::compute_digest(&basis_quantized, n, k);

        Self {
            basis_quantized,
            eigenvalues: ordered_eigenvalues,
            n,
            k,
            digest,
        }
    }

    /// Compute SHA-256 digest of the quantized basis.
    fn compute_digest(basis: &[i32], n: usize, k: usize) -> SubspaceDigest {
        let mut hasher = Sha256::new();

        // Include dimensions
        hasher.update(&(n as u32).to_le_bytes());
        hasher.update(&(k as u32).to_le_bytes());

        // Include quantized basis
        for &val in basis {
            hasher.update(&val.to_le_bytes());
        }

        SubspaceDigest(hex::encode(hasher.finalize()))
    }

    /// Get the subspace digest.
    pub fn digest(&self) -> SubspaceDigest {
        self.digest.clone()
    }

    /// Get the original space dimension (n).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the subspace dimension (k).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the eigenvalues.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Get a column vector as f64 (for distance calculations).
    pub fn column_f64(&self, col: usize) -> Vec<f64> {
        assert!(col < self.k);
        let start = col * self.n;
        self.basis_quantized[start..start + self.n]
            .iter()
            .map(|&v| v as f64 / QUANT_SCALE)
            .collect()
    }

    /// Get the full basis as f64 matrix (for distance calculations).
    pub fn basis_f64(&self) -> Vec<f64> {
        self.basis_quantized
            .iter()
            .map(|&v| v as f64 / QUANT_SCALE)
            .collect()
    }

    /// Get the quantized basis (for WAL serialization).
    pub fn basis_quantized(&self) -> &[i32] {
        &self.basis_quantized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_canonicalization() {
        // Two bases that differ only by sign should produce same canonical form
        let basis1 = vec![0.5, 0.5, 0.707, -0.5, 0.5, 0.707];
        let basis2 = vec![-0.5, -0.5, -0.707, 0.5, -0.5, -0.707]; // Flipped

        let eigenvalues = vec![1.0, 0.5];

        let s1 = CanonicalSubspace::from_basis(basis1, eigenvalues.clone(), 3, 2);
        let s2 = CanonicalSubspace::from_basis(basis2, eigenvalues, 3, 2);

        assert_eq!(s1.digest(), s2.digest());
    }

    #[test]
    fn test_ordering_by_eigenvalue() {
        // Eigenvalues out of order should be sorted
        let basis = vec![
            0.1, 0.2, 0.3, // Column 0 (smaller eigenvalue)
            0.4, 0.5, 0.6, // Column 1 (larger eigenvalue)
        ];
        let eigenvalues = vec![0.3, 0.7]; // Column 1 has larger eigenvalue

        let s = CanonicalSubspace::from_basis(basis, eigenvalues, 3, 2);

        // First eigenvalue should now be the larger one
        assert!(s.eigenvalues()[0] > s.eigenvalues()[1]);
    }

    #[test]
    fn test_digest_deterministic() {
        let basis = vec![0.5, 0.5, 0.707, 0.5, -0.5, 0.0];
        let eigenvalues = vec![1.0, 0.5];

        let s1 = CanonicalSubspace::from_basis(basis.clone(), eigenvalues.clone(), 3, 2);
        let s2 = CanonicalSubspace::from_basis(basis, eigenvalues, 3, 2);

        assert_eq!(s1.digest(), s2.digest());
    }

    #[test]
    fn test_column_extraction() {
        let basis = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let eigenvalues = vec![1.0, 0.5];

        let s = CanonicalSubspace::from_basis(basis, eigenvalues, 3, 2);

        let col0 = s.column_f64(0);
        let col1 = s.column_f64(1);

        assert_eq!(col0.len(), 3);
        assert_eq!(col1.len(), 3);
    }
}
