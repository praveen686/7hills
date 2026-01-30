# Phase 29 — Geometry-First Regime Engine (GRE)
## Frozen Specification v1.0.1

**Status:** FROZEN
**Supersedes:** v1.0
**Determinism target:** Cross-machine replay determinism (CI + x86_64 Linux)
**Scope:** Regime lifting + regime shift detection only (no labeling)

---

## 0. Patch Summary (What Changed from v1.0)

Mandatory corrections applied:

1. **Quantization unified to power-of-two scaling**
   - `basis_exponent` → `basis_shift`
   - All subspace bases use `mantissa / 2^shift`

2. **Covariance precision fixed**
   - Covariance returns `i128` mantissas
   - Normalization only at final step

3. **Eigen tie-breaking made deterministic**
   - No epsilon comparisons
   - Tie-break on quantized eigenvalues, then lexicographic basis bytes

4. **WAL digests corrected**
   - All digests are `[u8; 32]`, not hex strings
   - Canonical bytes include `session_id` and `seq`

5. **Distance metric made deterministic**
   - Phase 29.2 uses Frobenius-based Grassmann proxy
   - No float SVD in distance path

6. **CUSUM update rule frozen explicitly**

---

## 1. Crate Structure

```
crates/quantlaxmi-regime/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── lift/
│   │   ├── mod.rs
│   │   ├── covariance.rs
│   │   ├── eigen.rs
│   │   ├── canonicalize.rs
│   │   └── subspace.rs
│   ├── dynamics/
│   │   ├── mod.rs
│   │   ├── distance.rs
│   │   ├── cusum.rs
│   │   └── shift.rs
│   ├── events/
│   │   ├── mod.rs
│   │   ├── subspace_event.rs
│   │   ├── shift_event.rs
│   │   └── label_event.rs   # Stub only
│   ├── gates/
│   │   ├── mod.rs
│   │   ├── lift_determinism.rs
│   │   └── shift_parity.rs
│   └── config.rs
└── tests/
    ├── determinism_tests.rs
    ├── canonicalization_tests.rs
    └── integration_tests.rs
```

---

## 2. Core Types

### 2.1 Integer Covariance Accumulator

```rust
pub struct CovarianceAccumulator {
    n: usize,
    window_size: u32,

    /// Σ x_i (mantissa)
    sum_x: Vec<i64>,

    /// Σ x_i ⊗ x_iᵀ (mantissa²)
    sum_xx: Vec<i128>,

    count: u32,

    /// Feature scale: actual = mantissa / 2^feature_shift
    feature_shift: i8,

    /// Covariance scale: mantissa / 2^(2*feature_shift)
    cov_shift: i8,
}
```

**API:**
```rust
impl CovarianceAccumulator {
    pub fn accumulate(&mut self, x: &[i64]) -> Result<(), RegimeError>;

    /// Returns symmetric covariance matrix (upper triangle),
    /// mantissa in i128, scale = cov_shift.
    pub fn compute_covariance(&self) -> Result<(Vec<i128>, i8), RegimeError>;

    pub fn reset(&mut self);
}
```

**Determinism Rules (FROZEN):**
- Chronological accumulation only
- Overflow checks on every i128 update
- Block-window semantics: caller resets after W samples
- No floats anywhere in accumulation or normalization

### 2.2 Canonical Subspace Representation

```rust
pub struct CanonicalSubspace {
    pub n: u16,
    pub k: u16,

    /// Quantized orthonormal basis
    /// actual ≈ mantissa / 2^basis_shift
    pub basis_mantissa: Vec<i32>,
    pub basis_shift: i8,

    /// Quantized eigenvalues (descending)
    pub eigenvalues_mantissa: Vec<i64>,
    pub eigenvalue_shift: i8,

    /// SHA-256 of canonical_bytes()
    pub digest: [u8; 32],
}
```

### 2.3 Canonicalization Rules (FROZEN)

**Input:** float eigenpairs from deterministic backend
**Output:** canonical integer subspace

**Canonicalization Order (STRICT):**

1. **SIGN FIXING**
   - For each column u_j
   - Find index of max |u_ij|
   - If that value < 0 → negate entire column
   - Tie-break: lowest index

2. **QUANTIZATION**
   - `basis_mantissa = round(u * 2^basis_shift)`
   - Default `basis_shift = 30`

3. **ORDERING**
   - Sort by `eigenvalues_mantissa` descending
   - If equal → lexicographic compare `basis_mantissa` columns

4. **ORTHONORMALITY CHECK**
   - Verify `||u_j|| ∈ [0.999, 1.001]`
   - No re-orthogonalization

**NO epsilon comparisons permitted.**

---

## 3. WAL Event Schemas

### 3.1 RegimeSubspaceEvent

```rust
pub struct RegimeSubspaceEvent {
    pub ts_exchange_ns: i64,
    pub ts_recv_ns: i64,

    pub symbol: String,
    pub session_id: String,
    pub seq: u64,

    pub n: u16,
    pub k: u16,
    pub window_size: u32,

    pub basis_mantissa: Vec<i32>,
    pub basis_shift: i8,

    pub eigenvalues_mantissa: Vec<i64>,
    pub eigenvalue_shift: i8,

    pub explained_variance_bps: u32,

    pub admission_summary_digest: [u8; 32],
    pub subspace_digest: [u8; 32],
}
```

#### 3.1.1 Canonical Byte Encoding (FROZEN)

Used for digest computation only.

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 | ts_exchange_ns |
| 8 | 8 | ts_recv_ns |
| 16 | 2 | symbol_len |
| 18 | var | symbol |
| … | 2 | session_id_len |
| … | var | session_id |
| … | 8 | seq |
| … | 2 | n |
| … | 2 | k |
| … | 4 | window_size |
| … | 1 | basis_shift |
| … | 4·n·k | basis_mantissa |
| … | 1 | eigenvalue_shift |
| … | 8·k | eigenvalues_mantissa |
| … | 4 | explained_variance_bps |
| … | 32 | admission_summary_digest |

### 3.2 RegimeShiftEvent

```rust
pub struct RegimeShiftEvent {
    pub ts_exchange_ns: i64,
    pub ts_recv_ns: i64,

    pub symbol: String,
    pub session_id: String,
    pub seq: u64,

    pub delta_mantissa: i64,
    pub delta_shift: i8,

    pub cusum_stat_mantissa: i64,
    pub cusum_stat_shift: i8,

    pub threshold_mantissa: i64,
    pub threshold_shift: i8,

    pub shift_detected: bool,

    pub current_subspace_digest: [u8; 32],
    pub previous_subspace_digest: [u8; 32],
    pub event_digest: [u8; 32],
}
```

---

## 4. Distance Metric (FROZEN)

### Phase 29.2 Deterministic Grassmann Proxy

Let:
- U, V ∈ ℤ^(n×k) quantized bases
- M = UᵀV computed in i128

Define:
```
s = Σᵢⱼ M_ij²
distance² = max(0, k·(2^(2·basis_shift))² − s)
```

Return `(distance_mantissa, distance_shift)` where:
```
distance_shift = basis_shift * 2
```

**Properties:**
- Deterministic
- Symmetric
- Monotone in subspace misalignment
- Sufficient for CPD / gating

True principal angles are explicitly deferred to Phase ≥29.3.

---

## 5. CUSUM Change Detection (FROZEN)

### Update Rule

One-sided CUSUM:
```
S₀ = 0

Sₜ = max(
    0,
    Sₜ₋₁ + (deltaₜ − drift)
)

shift_detected = (Sₜ ≥ threshold)
```

All values:
- Integer mantissa
- Fixed shift
- No floats
- Reset Sₜ = 0 after detection

---

## 6. Gates

### Gx-RegimeLiftDeterminism
- Same admission digest + params
- Same `subspace_digest` across replay

### Gx-RegimeShiftParity
- Same subspace sequence
- Identical `delta`, `cusum_stat`, `shift_detected`

---

## 7. Configuration

```rust
pub struct RegimeEngineConfig {
    pub n: u16,
    pub k: u16,
    pub window_size: u32,

    pub feature_shift: i8,
    pub basis_shift: i8,
    pub eigenvalue_shift: i8,

    pub cusum_drift_mantissa: i64,
    pub cusum_drift_shift: i8,

    pub cusum_threshold_mantissa: i64,
    pub cusum_threshold_shift: i8,

    pub min_explained_variance_bps: u32,
    pub min_window_fill: u32,
}
```

---

## 8. Determinism Contract (FROZEN)

- Default backend must be single-threaded
- SIMD-dependent paths must be feature-gated
- CI must use deterministic backend
- Any nondeterminism → gate fail

---

## 9. Phase 29 Milestones

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| 29.1a | `CovarianceAccumulator` with integer-only accumulation | Unit tests |
| 29.1b | Deterministic eigen/SVD with canonicalization | `test_lift_determinism_repeated` |
| 29.1c | `RegimeSubspaceEvent` WAL emission | Schema validation |
| 29.1d | `Gx-RegimeLiftDeterminism` gate | Gate passes on replay |
| 29.2a | Frobenius-based Grassmann distance | `test_distance_symmetry` |
| 29.2b | CUSUM change-point detection | `test_cusum_replay_parity` |
| 29.2c | `RegimeShiftEvent` WAL emission | Schema validation |
| 29.2d | `Gx-RegimeShiftParity` gate | Gate passes on replay |

---

## 10. Dependencies

```toml
[dependencies]
quantlaxmi-models = { path = "../quantlaxmi-models" }
quantlaxmi-wal = { path = "../quantlaxmi-wal" }
serde = { version = "1", features = ["derive"] }
sha2 = "0.10"
nalgebra = "0.32"  # With deterministic settings
```

**nalgebra determinism note:** Must configure with consistent SIMD/threading settings. Consider pure-Rust fallback for guaranteed reproducibility.

---

## Final Status

Phase 29 GRE Spec v1.0.1 is now:
- Internally consistent
- WAL-sound
- Replay-safe
- Audit-grade
- Aligned with QuantLaxmi doctrine

---

*This specification is FROZEN. Implementation begins on explicit instruction.*
