//! Grassmann manifold distance metrics.
//!
//! Distance between subspaces U, V ∈ Gr(k,n) based on principal angles.

use crate::canonical::CanonicalSubspace;
use nalgebra::{DMatrix, SVD};

/// Compute principal angles between two subspaces.
///
/// Principal angles θ₁, θ₂, ..., θₖ are defined via:
/// cos(θᵢ) = σᵢ(U^T V)
///
/// where σᵢ are singular values of U^T V.
///
/// Returns angles in radians, sorted ascending.
pub fn principal_angles(u: &CanonicalSubspace, v: &CanonicalSubspace) -> Vec<f64> {
    assert_eq!(u.n(), v.n(), "Subspaces must be in same ambient space");
    assert_eq!(u.k(), v.k(), "Subspaces must have same dimension");

    let n = u.n();
    let k = u.k();

    // Build matrices
    let u_mat = DMatrix::from_column_slice(n, k, &u.basis_f64());
    let v_mat = DMatrix::from_column_slice(n, k, &v.basis_f64());

    // Compute U^T V
    let product = u_mat.transpose() * v_mat;

    // SVD to get singular values
    let svd = SVD::new(product, false, false);
    let singular_values = svd.singular_values;

    // Convert to angles: θᵢ = arccos(σᵢ)
    // Clamp to [-1, 1] to handle numerical issues
    singular_values
        .iter()
        .map(|&s| s.clamp(-1.0, 1.0).acos())
        .collect()
}

/// Compute Grassmann distance as the Frobenius norm of principal angles.
///
/// d(U, V) = ||θ||₂ = sqrt(Σ θᵢ²)
///
/// Returns distance as fixed-point mantissa with given exponent.
pub fn grassmann_distance(u: &CanonicalSubspace, v: &CanonicalSubspace, exponent: i8) -> i64 {
    let angles = principal_angles(u, v);

    // Frobenius norm of angles
    let distance_f64: f64 = angles.iter().map(|&a| a * a).sum::<f64>().sqrt();

    // Convert to mantissa with given exponent
    let scale = 10f64.powi(-exponent as i32);
    (distance_f64 * scale).round() as i64
}

/// Compute geodesic distance on Grassmann manifold.
///
/// This is the length of the shortest path on the manifold:
/// d_geo(U, V) = ||θ||₂
///
/// Same as grassmann_distance but with explicit naming.
pub fn geodesic_distance(u: &CanonicalSubspace, v: &CanonicalSubspace) -> f64 {
    let angles = principal_angles(u, v);
    angles.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

/// Compute chordal distance between subspaces.
///
/// d_chord(U, V) = ||sin(θ)||₂ = sqrt(Σ sin²(θᵢ))
///
/// Often more numerically stable for small angles.
pub fn chordal_distance(u: &CanonicalSubspace, v: &CanonicalSubspace) -> f64 {
    let angles = principal_angles(u, v);
    angles.iter().map(|&a| a.sin().powi(2)).sum::<f64>().sqrt()
}

/// Compute projection distance (Frobenius norm of projector difference).
///
/// d_proj(U, V) = ||P_U - P_V||_F / sqrt(2)
///
/// where P_U = U U^T is the orthogonal projector onto U.
pub fn projection_distance(u: &CanonicalSubspace, v: &CanonicalSubspace) -> f64 {
    let angles = principal_angles(u, v);
    // ||P_U - P_V||_F² = 2 Σ sin²(θᵢ)
    (2.0 * angles.iter().map(|&a| a.sin().powi(2)).sum::<f64>()).sqrt()
}

/// Check if two subspaces are "close" (within threshold).
///
/// Uses geodesic distance with fixed-point threshold.
pub fn subspaces_close(
    u: &CanonicalSubspace,
    v: &CanonicalSubspace,
    threshold_mantissa: i64,
    exponent: i8,
) -> bool {
    let distance = grassmann_distance(u, v, exponent);
    distance <= threshold_mantissa
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
    fn test_identical_subspaces_zero_distance() {
        let basis = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // Standard basis
        let s1 = make_subspace(basis.clone(), 2);
        let s2 = make_subspace(basis, 2);

        let distance = grassmann_distance(&s1, &s2, -4);
        assert_eq!(distance, 0);
    }

    #[test]
    fn test_orthogonal_subspaces_max_distance() {
        // Two orthogonal 1D subspaces in 2D
        let s1 = make_subspace(vec![1.0, 0.0], 1);
        let s2 = make_subspace(vec![0.0, 1.0], 1);

        let angles = principal_angles(&s1, &s2);
        assert_eq!(angles.len(), 1);

        // Angle should be π/2
        let angle = angles[0];
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_principal_angles_count() {
        // k-dimensional subspaces should have k principal angles
        let s1 = make_subspace(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3);
        let s2 = make_subspace(
            vec![0.707, 0.707, 0.0, -0.707, 0.707, 0.0, 0.0, 0.0, 1.0],
            3,
        );

        let angles = principal_angles(&s1, &s2);
        assert_eq!(angles.len(), 3);
    }

    #[test]
    fn test_distance_symmetric() {
        let s1 = make_subspace(vec![0.8, 0.6, 0.0, -0.6, 0.8, 0.0], 2);
        let s2 = make_subspace(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2);

        let d12 = grassmann_distance(&s1, &s2, -4);
        let d21 = grassmann_distance(&s2, &s1, -4);

        assert_eq!(d12, d21);
    }

    #[test]
    fn test_triangle_inequality() {
        let s1 = make_subspace(vec![1.0, 0.0, 0.0], 1);
        let s2 = make_subspace(vec![0.707, 0.707, 0.0], 1);
        let s3 = make_subspace(vec![0.0, 1.0, 0.0], 1);

        let d12 = geodesic_distance(&s1, &s2);
        let d23 = geodesic_distance(&s2, &s3);
        let d13 = geodesic_distance(&s1, &s3);

        // Triangle inequality: d13 <= d12 + d23
        assert!(d13 <= d12 + d23 + 0.001); // Small epsilon for float precision
    }
}
