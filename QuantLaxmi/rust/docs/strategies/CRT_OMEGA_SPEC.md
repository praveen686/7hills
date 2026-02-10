# CRT-Ω (Causal Reversal Transport - Omega) Strategy Specification

**Version**: 1.0.0
**Status**: Experimental
**Author**: QuantLaxmi Research
**Date**: 2026-02-09

---

## Executive Summary

CRT-Ω is a regime-conditioned microstructure mean-reversion strategy that trades the *residual* between where price *should* have gone (given order flow + liquidity state) and where it actually went. The core innovation is trading causal mispricing, not price or momentum.

Unlike traditional mean-reversion strategies that trade price z-scores, CRT-Ω:
1. Builds a **transport map** predicting expected return from microstructure state
2. Computes the **residual overshoot** (actual - predicted)
3. Uses **SPRT (Sequential Probability Ratio Test)** to gate trading per regime cell
4. Only trades when there is **statistical evidence of edge after costs**

---

## Theoretical Foundation

### 1. The Transport Map

Price moves in response to order flow and liquidity conditions. We model the expected next return as a conservative transport functional:

```
r̂_{t→t+Δ} = a·φ_t + b·ψ_t - c·κ_t·sign(r_{t-1})
```

Where:
- **φ (phi)**: Signed aggressor flow from trades (buyer-initiated minus seller-initiated volume)
- **ψ (psi)**: Depth imbalance from L2 book = (bid_qty - ask_qty) / (bid_qty + ask_qty)
- **κ (kappa)**: Stiffness/fragility proxy = spread / weighted_depth
- **sign(r_{t-1})**: Previous return direction (for anti-momentum term)

The coefficients (a, b, c) are structural, not curve-fit:
- `a`: Response to aggressor flow (small, flow is noisy)
- `b`: Response to standing liquidity pressure
- `-c·κ·sign(·)`: Anti-momentum term when book is fragile (overshoots more likely)

### 2. The Tradable Residual

The edge signal is the **excess drift residual**:

```
ε_t = r_{t→t+Δ} - r̂_{t→t+Δ}
```

Interpretation:
- `ε << 0`: Price fell more than microstructure justified → overshoot down → **BUY** (mean reversion)
- `ε >> 0`: Price rose more than justified → overshoot up → **SELL** (mean reversion)

We compute a robust z-score of the residual using rolling median and MAD (Median Absolute Deviation):

```
z_t = (ε_t - median(ε)) / (1.4826 · MAD(ε))
```

### 3. Why This Has Edge (Mechanism, Not Mythology)

Markets frequently show **impact overshoot**: aggressive flow pushes price beyond the level justified by contemporaneous liquidity. That overshoot mean-reverts as:

1. Passive liquidity refills
2. Aggressive flow decays
3. Spread normalizes

CRT-Ω explicitly separates:
- **Explained move**: What order flow + book predicts
- **Excess move**: Mispricing vs microstructure state

The excess move is the tradable object.

---

## SPRT Evidence Gate

### Why Evidence Gating?

Most strategies assume edge is present. CRT-Ω **demands statistical proof**.

We use a **Sequential Probability Ratio Test (SPRT)** per regime cell to decide:
- **H₀**: Mean residual payoff ≤ 0 (no edge after costs)
- **H₁**: Mean residual payoff ≥ δ (positive edge)

Trading is only enabled when H₁ is accepted with confidence bounds.

### Regime Cells

The state space is discretized into cells based on:
- **κ (kappa)**: Stiffness buckets (3 levels)
- **spread**: Spread buckets (3 levels)
- **premium**: Mark-mid premium buckets (3 levels)

This gives 27 regime cells by default. Each cell maintains its own SPRT state.

### Hysteresis

To prevent flip-flopping:
- **Activate** cell when P(edge > 0) > `prob_on` (default 60%)
- **Deactivate** when P(edge > 0) < `prob_off` (default 45%)

---

## Position Sizing

### Kelly Criterion with Constraints

Position size is computed using fractional Kelly:

```
size = base_size × kelly_multiplier
kelly_multiplier = min(|mean_edge / variance| × kelly_frac, max_multiplier)
```

### Drawdown Stop

If equity drawdown exceeds `dd_stop` threshold (default -6%), all trading stops and positions are flattened.

---

## Entry/Exit Logic

### Entry Conditions

Enter **LONG** when ALL conditions met:
- `z < -z_in` (residual suggests downward overshoot)
- Regime cell SPRT is ON (evidence gate passed)
- Not in drawdown stop

Enter **SHORT** when ALL conditions met:
- `z > +z_in` (residual suggests upward overshoot)
- Regime cell SPRT is ON
- Not in drawdown stop

### Exit Conditions

Exit when ANY condition met:
- `|z| <= z_out` (mean reversion complete)
- Hold time exceeds `max_hold_ticks`
- Evidence gate turns OFF
- Drawdown stop triggered

---

## Configuration Reference

### Transport Map Weights

| Parameter | Default | Exponent | Description |
|-----------|---------|----------|-------------|
| `flow_weight_mantissa` | 1 | -6 | Flow responsiveness |
| `psi_weight_mantissa` | 200 | -4 | Depth imbalance weight |
| `kappa_weight_mantissa` | 1000 | -4 | Anti-momentum weight |

### Z-Score Parameters

| Parameter | Default | Exponent | Description |
|-----------|---------|----------|-------------|
| `z_window` | 900 | - | Rolling window (ticks) |
| `z_in_mantissa` | 300 | -2 | Entry threshold (3.0σ) |
| `z_out_mantissa` | 50 | -2 | Exit threshold (0.5σ) |
| `max_hold_ticks` | 180 | - | Max hold duration |

### SPRT Evidence Gate

| Parameter | Default | Exponent | Description |
|-----------|---------|----------|-------------|
| `min_obs_per_cell` | 60 | - | Min observations before trading |
| `sprt_delta_mantissa` | 1 | -8 | Minimum edge to detect |
| `sprt_alpha_mantissa` | 500 | -4 | Type I error (5%) |
| `sprt_beta_mantissa` | 1000 | -4 | Type II error (10%) |

### Position Sizing

| Parameter | Default | Exponent | Description |
|-----------|---------|----------|-------------|
| `position_size_mantissa` | 1,000,000 | -8 | Base size (0.01 BTC) |
| `kelly_frac_mantissa` | 2500 | -4 | Kelly fraction (25%) |
| `max_pos_multiplier_mantissa` | 10000 | -4 | Max size multiplier |
| `dd_stop_mantissa` | -600 | -4 | Drawdown stop (-6%) |

### Cost Model

| Parameter | Default | Exponent | Description |
|-----------|---------|----------|-------------|
| `maker_fee_bps_mantissa` | 2 | -2 | Maker fee (0.02 bps) |
| `taker_fee_bps_mantissa` | 10 | -2 | Taker fee (0.10 bps) |

---

## Usage

### Backtest

```bash
# With default config
cargo run -p quantlaxmi-crypto -- backtest \
    --segment-dir data/perp_sessions/perp_20260130_211347/BTCUSDT \
    --strategy crt_omega \
    --use-sdk \
    --flatten-on-end \
    --output-json results/crt_omega_backtest.json

# With custom config
cargo run -p quantlaxmi-crypto -- backtest \
    --segment-dir data/perp_sessions/perp_20260130_211347/BTCUSDT \
    --strategy crt_omega \
    --strategy-config configs/strategies/crt_omega.toml \
    --use-sdk \
    --flatten-on-end
```

### Tournament Grid Sweep

```bash
cargo run -p quantlaxmi-crypto -- tournament \
    --grid-config configs/grids/crt_omega_grid.toml \
    --segment-dir data/perp_sessions \
    --output-dir results/crt_omega_tournament \
    --num-workers 16
```

---

## Diagnostics

The strategy provides diagnostics via `print_diagnostics()`:

```
=== CRT-Ω Diagnostics ===
Total signals:   1234
Signals gated:   892 (72.3%)    # Blocked by SPRT evidence gate
Trades taken:    342
SPRT cells:      27
Cells with edge: 8              # Only 8/27 cells show evidence of edge
Current tick:    15000
```

High gating rate (70%+) is expected and healthy—it means the strategy is only trading when evidence supports it.

---

## Failure Modes

| Condition | Behavior | Mitigation |
|-----------|----------|------------|
| Trending market | Residual mean-reversion fails | SPRT gate should deactivate |
| Low volatility | Few extreme z-scores | Normal—low turnover expected |
| Liquidity drought | Kappa explodes, spreads widen | Gate checks spread bounds |
| Data gaps | Missing quotes/trades | Warmup period resets |

---

## Validation Checklist

Before promoting to production:

- [ ] G0: Data integrity verified (no gaps, sequence valid)
- [ ] G1: Replay parity confirmed (same trace hash)
- [ ] G2: Backtest with realistic costs (taker fees, slippage)
- [ ] G3: Walk-forward validation (no lookahead)
- [ ] G4: Stress test (widen spreads 2x, add latency)
- [ ] G5: Multi-segment consistency

---

## References

1. **Impact Overshoot**: Bouchaud, J.P. et al. "Price Impact" (2009)
2. **SPRT**: Wald, A. "Sequential Analysis" (1947)
3. **Kelly Criterion**: Kelly, J.L. "A New Interpretation of Information Rate" (1956)
4. **Microstructure**: Hasbrouck, J. "Empirical Market Microstructure" (2007)

---

## Backtest Results (2026-02-09)

### Summary

| Metric | 1-min Klines (7d) | Tick-level (1d) |
|--------|-------------------|-----------------|
| Events | 10,080 | 204,000 |
| Trades | 8,283 | 1,312 |
| Win Rate | 0% | 0% |
| Net PnL | -$65.52 (-0.66%) | -$2.78 (-0.03%) |
| Max DD | 2.96% | 0.40% |
| Sharpe | -20.50 | -99.00 |

### Key Finding

The strategy is **not profitable with synthetic depth data**. The core issue is that CRT-Ω requires **real L2 order book data** to function:

1. **Order Flow (φ)**: ✅ Can approximate from `is_buyer_maker` flag on trades
2. **Imbalance (ψ)**: ❌ Requires real bid/ask queue sizes (synthetic depth has fixed ratios)
3. **Kyle's λ (κ)**: ❌ Requires real spread and depth dynamics

### Next Steps

1. **Capture real L2 depth** via Binance websocket (SBE or JSON streams)
2. **Use QuantLaxmi's live capture** mode for real data collection:
   ```bash
   cargo run -p quantlaxmi-crypto -- capture --symbol BTCUSDT --duration 3600
   ```
3. Re-run backtest with real L2 data once sufficient history is accumulated

---

## Changelog

### v1.0.0 (2026-02-09)
- Initial implementation
- Transport map: φ + ψ - κ·sign(r)
- SPRT evidence gating per regime cell
- Kelly sizing with drawdown stop
- Fixed-point arithmetic throughout
- Initial backtest with synthetic depth (negative results expected)
