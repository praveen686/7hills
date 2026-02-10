# Alpha Factory v1 - Implementation Plan

**Objective:** Build canonical trading pipeline with determinism, replay parity, and audit-grade event correlation.

**Alignment:** QUANTLAXMI_CHARTER_v2_1.md + PRODUCTION_CONTRACT.md

---

## Phase 1: Canonical Events Integration (Priority 1)

### 1.1 Create `quantlaxmi-events` crate (new)
Consolidate all canonical event types in one place with re-exports.

**Files:**
- `crates/quantlaxmi-events/Cargo.toml`
- `crates/quantlaxmi-events/src/lib.rs` - Re-exports from quantlaxmi-models
- `crates/quantlaxmi-events/src/trace.rs` - DecisionTrace + hashing

**Types to export:**
```rust
// From quantlaxmi-models
pub use quantlaxmi_models::{
    DecisionEvent, RiskEvent, OrderEvent, FillEvent,
    QuoteEvent, DepthEvent, MarketSnapshot, CorrelationContext,
};

// New: Decision trace for replay parity
pub struct DecisionTrace {
    pub decisions: Vec<DecisionEvent>,
    pub trace_hash: [u8; 32],  // SHA-256
}

impl DecisionTrace {
    pub fn compute_hash(&self) -> [u8; 32];
    pub fn verify_parity(&self, other: &DecisionTrace) -> bool;
}
```

### 1.2 Refactor `backtest.rs` to use canonical events

**Current:** Local `Fill`, `OrderIntent` types with f64
**Target:** Use `FillEvent`, `OrderEvent` from quantlaxmi-models with mantissa

**Changes:**
- Replace `Fill` with `FillEvent` (add mantissa fields)
- Replace `OrderIntent` with `OrderEvent::New` payload
- Add `CorrelationContext` to all events
- Track `DecisionEvent` emissions for trace hashing

### 1.3 Add fixed-point PnL calculation

**New module:** `crates/quantlaxmi-runner-crypto/src/fixed_point.rs`

```rust
/// Fixed-point arithmetic for deterministic PnL
pub struct FixedPoint {
    pub mantissa: i64,
    pub exponent: i8,
}

impl FixedPoint {
    pub fn add(&self, other: &Self) -> Self;
    pub fn sub(&self, other: &Self) -> Self;
    pub fn mul(&self, other: &Self) -> Self;
    pub fn div(&self, other: &Self) -> Self;
    pub fn to_f64(&self) -> f64;  // Display only
}

/// PnL accumulator using fixed-point
pub struct PnLAccumulator {
    realized: FixedPoint,
    unrealized: FixedPoint,
    fees: FixedPoint,
}
```

---

## Phase 2: Strategy SDK + Registry (Priority 2)

### 2.1 Create `quantlaxmi-strategy-sdk` crate (new)

**Files:**
- `crates/quantlaxmi-strategy-sdk/Cargo.toml`
- `crates/quantlaxmi-strategy-sdk/src/lib.rs`
- `crates/quantlaxmi-strategy-sdk/src/registry.rs`
- `crates/quantlaxmi-strategy-sdk/src/config.rs`

**Core trait:**
```rust
pub trait CryptoStrategy: Send + Sync {
    /// Stable strategy identifier (for registry lookup)
    fn strategy_id(&self) -> &str;

    /// Deterministic config hash
    fn config_hash(&self) -> [u8; 32];

    /// Process market event, emit DecisionEvent if action needed
    fn on_market(&mut self, event: &ReplayEvent, ctx: &CorrelationContext)
        -> Option<DecisionEvent>;

    /// Handle fill confirmation
    fn on_fill(&mut self, fill: &FillEvent);

    /// Current position (for risk checks)
    fn position(&self) -> FixedPoint;
}

pub struct StrategyRegistry {
    strategies: HashMap<String, Box<dyn CryptoStrategy>>,
}

impl StrategyRegistry {
    pub fn register(&mut self, strategy: Box<dyn CryptoStrategy>);
    pub fn get(&self, id: &str) -> Option<&dyn CryptoStrategy>;
    pub fn list(&self) -> Vec<&str>;
}
```

### 2.2 Migrate BasisCaptureStrategy

Move from `backtest.rs` to separate module using SDK trait.

**Files:**
- `crates/quantlaxmi-runner-crypto/src/strategies/mod.rs`
- `crates/quantlaxmi-runner-crypto/src/strategies/basis_capture.rs`

**Changes:**
- Implement `CryptoStrategy` trait
- Emit `DecisionEvent` with rationale hash
- Use fixed-point internally
- Add attribution fields (spread_at_decision, quote_age_ms)

---

## Phase 3: Crypto Alpha v1 (Priority 3)

### 3.1 Enhanced BasisCapture with feasibility filters

**New fields in strategy config:**
```rust
pub struct BasisCaptureConfig {
    // Existing
    pub threshold_bps: FixedPoint,
    pub position_size: FixedPoint,
    pub exit_threshold_bps: FixedPoint,

    // Feasibility filters (new)
    pub max_spread_bps: FixedPoint,
    pub max_quote_age_ms: u64,
    pub min_bid_qty: FixedPoint,
    pub min_ask_qty: FixedPoint,

    // Cooldown/churn control (new)
    pub cooldown_ms: u64,
    pub max_trades_per_hour: u32,
}
```

### 3.2 Attribution events

**New struct:**
```rust
pub struct TradeAttribution {
    pub decision_id: Uuid,
    pub fill_id: Uuid,

    // Edge decomposition
    pub gross_edge_bps: FixedPoint,
    pub fee_drag_bps: FixedPoint,
    pub slippage_bps: FixedPoint,  // reference_price vs fill_price
    pub realized_edge_bps: FixedPoint,

    // Context at decision
    pub spread_at_decision_bps: FixedPoint,
    pub quote_age_at_decision_ms: u64,
    pub book_depth_at_decision: FixedPoint,
}
```

---

## Phase 4: India Alpha v1 (Priority 4)

### 4.1 Create options strategy scaffold

**Files:**
- `crates/quantlaxmi-runner-india/src/strategies/mod.rs`
- `crates/quantlaxmi-runner-india/src/strategies/calendar_carry.rs`

**Trait extension for India:**
```rust
pub trait IndiaOptionsStrategy: Send + Sync {
    fn strategy_id(&self) -> &str;
    fn config_hash(&self) -> [u8; 32];

    /// Emit multi-leg order intent
    fn on_market(&mut self, event: &IndiaMarketEvent, ctx: &CorrelationContext)
        -> Option<MultiLegOrderIntent>;

    fn on_fill(&mut self, fill: &FillEvent);
}

pub struct MultiLegOrderIntent {
    pub decision_id: Uuid,
    pub legs: Vec<LegIntent>,
    pub strategy_type: OptionsStrategyType,  // Calendar, Spread, etc.
}

pub struct LegIntent {
    pub symbol: String,
    pub side: Side,
    pub qty: FixedPoint,
    pub order_type: OrderType,
    pub limit_price: Option<FixedPoint>,
}
```

### 4.2 Surface feature filters

```rust
pub struct OptionsSurfaceFilters {
    pub min_iv: f64,
    pub max_iv: f64,
    pub skew_threshold: f64,
    pub term_structure_slope: f64,
    pub min_open_interest: u64,
    pub max_bid_ask_spread_pct: f64,
}
```

---

## Phase 5: G2/G3 Anti-Overfit Harness (Priority 5)

### 5.1 Create anti-overfit test suite

**Files:**
- `crates/quantlaxmi-gates/src/g2_antioverfit.rs`
- `crates/quantlaxmi-gates/src/g3_robustness.rs`

**G2 Tests:**
```rust
pub struct G2AntiOverfitSuite {
    pub time_shift_results: Vec<TimeShiftResult>,
    pub random_entry_baseline: RandomEntryResult,
    pub permutation_results: Vec<PermutationResult>,
    pub ablation_results: Vec<AblationResult>,
}

impl G2AntiOverfitSuite {
    /// Time-shift: run strategy on shifted time windows
    pub fn run_time_shift(&self, strategy: &dyn CryptoStrategy,
                          segment: &Path, shifts: &[Duration]) -> Vec<TimeShiftResult>;

    /// Random entry: replace strategy signals with random
    pub fn run_random_entry(&self, segment: &Path, seed: u64) -> RandomEntryResult;

    /// Permutation: shuffle labels within day/regime
    pub fn run_permutation(&self, segment: &Path, n_perms: usize) -> Vec<PermutationResult>;

    /// Ablation: remove features one by one
    pub fn run_ablation(&self, strategy: &dyn CryptoStrategy,
                        segment: &Path) -> Vec<AblationResult>;
}
```

**G3 Tests:**
```rust
pub struct G3RobustnessSuite {
    pub walk_forward: WalkForwardResult,
    pub regime_slicing: Vec<RegimeSliceResult>,
    pub cost_sensitivity: CostSensitivityResult,
}

impl G3RobustnessSuite {
    /// Walk-forward: train on window, test on next
    pub fn run_walk_forward(&self, segment: &Path,
                            window_size: Duration) -> WalkForwardResult;

    /// Regime slicing: test on bull/bear/chop regimes
    pub fn run_regime_slicing(&self, segment: &Path) -> Vec<RegimeSliceResult>;

    /// Cost sensitivity: sweep fee/slippage parameters
    pub fn run_cost_sensitivity(&self, segment: &Path,
                                 fee_range: Range<f64>) -> CostSensitivityResult;
}
```

### 5.2 CLI command for gate harness

**Add to `quantlaxmi-crypto` CLI:**
```
quantlaxmi-crypto run-gates \
  --segment-dir <path> \
  --strategy <id> \
  --gates g2,g3 \
  --output-report <path>
```

### 5.3 Gate report binding

```rust
pub struct GateReport {
    pub run_id: Uuid,
    pub strategy_id: String,
    pub segment_id: String,
    pub timestamp: DateTime<Utc>,

    pub g2_passed: bool,
    pub g2_details: G2AntiOverfitSuite,

    pub g3_passed: bool,
    pub g3_details: G3RobustnessSuite,

    pub report_hash: [u8; 32],  // For manifest binding
}
```

---

## Phase 6: Decision Trace Hashing (Priority 6)

### 6.1 Implement trace computation

**In `quantlaxmi-events/src/trace.rs`:**

```rust
use sha2::{Sha256, Digest};

pub struct DecisionTraceBuilder {
    decisions: Vec<DecisionEvent>,
    hasher: Sha256,
}

impl DecisionTraceBuilder {
    pub fn new() -> Self;

    pub fn record(&mut self, decision: &DecisionEvent) {
        // Canonical serialization
        let bytes = decision.canonical_bytes();
        self.hasher.update(&bytes);
        self.decisions.push(decision.clone());
    }

    pub fn finalize(self) -> DecisionTrace {
        let hash = self.hasher.finalize();
        DecisionTrace {
            decisions: self.decisions,
            trace_hash: hash.into(),
        }
    }
}

impl DecisionEvent {
    /// Stable byte representation for hashing
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Fixed field order, no optional fields unless present
        let mut buf = Vec::new();
        buf.extend_from_slice(self.decision_id.as_bytes());
        buf.extend_from_slice(self.strategy_id.as_bytes());
        buf.extend_from_slice(&self.direction.to_le_bytes());
        buf.extend_from_slice(&self.target_qty_mantissa.to_le_bytes());
        buf.extend_from_slice(&self.reference_price_mantissa.to_le_bytes());
        // ... more fields
        buf
    }
}
```

### 6.2 Replay parity test

```rust
pub fn verify_replay_parity(
    original_trace: &DecisionTrace,
    replay_trace: &DecisionTrace,
) -> ReplayParityResult {
    if original_trace.trace_hash == replay_trace.trace_hash {
        ReplayParityResult::Match
    } else {
        // Find first divergence
        for (i, (orig, replay)) in original_trace.decisions.iter()
            .zip(replay_trace.decisions.iter()).enumerate()
        {
            if orig.canonical_bytes() != replay.canonical_bytes() {
                return ReplayParityResult::Divergence {
                    index: i,
                    original: orig.clone(),
                    replay: replay.clone(),
                };
            }
        }
        ReplayParityResult::LengthMismatch {
            original_len: original_trace.decisions.len(),
            replay_len: replay_trace.decisions.len(),
        }
    }
}
```

---

## Implementation Order

| Phase | Deliverable | Estimated Effort | Dependencies |
|-------|-------------|------------------|--------------|
| 1.1 | quantlaxmi-events crate | 2h | None |
| 1.2 | Refactor backtest.rs | 4h | 1.1 |
| 1.3 | Fixed-point PnL | 2h | 1.1 |
| 2.1 | Strategy SDK | 3h | 1.1, 1.3 |
| 2.2 | Migrate BasisCapture | 2h | 2.1 |
| 3.1 | Feasibility filters | 2h | 2.2 |
| 3.2 | Attribution events | 2h | 3.1 |
| 4.1 | India strategy scaffold | 3h | 2.1 |
| 4.2 | Surface filters | 2h | 4.1 |
| 5.1 | G2 anti-overfit | 4h | 2.2, 4.1 |
| 5.2 | G3 robustness | 3h | 5.1 |
| 5.3 | Gate CLI + binding | 2h | 5.1, 5.2 |
| 6.1 | Trace hashing | 2h | 1.1 |
| 6.2 | Parity test | 1h | 6.1 |

**Total: ~34 hours**

---

## Success Criteria

1. **Replay Parity Test:** Run backtest, save trace hash. Replay from WAL, compute trace hash. Hashes match.

2. **G2 Gate Pass:** Strategy PnL on random-entry baseline is statistically indistinguishable from zero.

3. **Fixed-Point Consistency:** PnL computed via fixed-point matches f64 display value within 1 basis point.

4. **Attribution Completeness:** Every fill has TradeAttribution with fee/slippage/edge decomposition.

5. **Strategy Isolation:** Strategies loaded via registry, not compiled into backtest.rs.

---

## Test Plan

### Test 1: Crypto Replay Parity
```bash
# Run backtest, save trace
quantlaxmi-crypto backtest --segment-dir $SEG --strategy basis_capture \
  --output-trace /tmp/trace_original.json

# Replay from same segment
quantlaxmi-crypto replay --segment-dir $SEG --strategy basis_capture \
  --output-trace /tmp/trace_replay.json

# Compare hashes
quantlaxmi-crypto verify-parity \
  --original /tmp/trace_original.json \
  --replay /tmp/trace_replay.json
# Expected: PARITY_MATCH
```

### Test 2: India Replay Parity
```bash
quantlaxmi-india backtest --session-dir $SESS --strategy calendar_carry \
  --output-trace /tmp/india_trace_original.json

quantlaxmi-india replay --session-dir $SESS --strategy calendar_carry \
  --output-trace /tmp/india_trace_replay.json

quantlaxmi-india verify-parity \
  --original /tmp/india_trace_original.json \
  --replay /tmp/india_trace_replay.json
# Expected: PARITY_MATCH
```

### Test 3: G2 Anti-Overfit
```bash
quantlaxmi-crypto run-gates --segment-dir $SEG --strategy basis_capture \
  --gates g2 --output-report /tmp/g2_report.json

# Check report
jq '.g2_passed' /tmp/g2_report.json
# Expected: true (strategy beats random baseline)
```
