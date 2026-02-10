"""CLRS Signal Generator -- 4 composite trading signals.

Signal A: Enhanced Carry
    VPIN-filtered funding arbitrage.  Only enter carry when VPIN < threshold
    (low probability of adverse selection).  Exit immediately on VPIN spike.

Signal B: Residual Carry
    PCA-identified mispriced funding.  Funding rate has a large common
    factor (PC1 = market-wide demand for leverage).  Symbols with high
    RESIDUAL funding (after removing common factors) are locally mispriced.

Signal C: Cascade Direction
    During liquidation cascades (Hawkes intensity ratio > 3), trade
    in the direction of the cascade (momentum).

Signal D: Post-Cascade Reversion
    After cascade intensity drops below 1.5 but VPIN remains elevated,
    fade the move.  Markets systematically overshoot during cascades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlaxmi.strategies.s26_crypto_flow.features import (
    RegimeSnapshot,
    VPINState,
    calibrate_hawkes,
    compute_vpin_series,
    funding_pca,
    kyles_lambda,
)
from quantlaxmi.strategies.s26_crypto_flow.scanner import (
    FundingMatrixBuilder,
    SymbolSnapshot,
    fetch_recent_klines,
    fetch_recent_trades,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for all 4 CLRS signals.

    Current state (2026-02):
      - Signal A (carry): PROVEN. +5.29% avg / 90d, Sharpe 11.86, 9/10 profitable.
        Kline VPIN disabled (noise). Tick VPIN enabled: entry < 0.80, exit > 0.90.
        Tick-level VPIN from LiveRegimeTracker uses proper BVC on individual trades.
      - Signal B (PCA residual): KILLED. z-scores ~ 0, no predictive value at 8h scale.
      - Signal C (cascade): DISABLED. max_positions=0. Has tick Hawkes but unvalidated.
      - Signal D (reversion): DISABLED. max_positions=0. Same as C.

    Volume threshold: $10M (not $50M). funding_paper misses 9/12 high-funding symbols
    at $50M. $10M captures them with max DD still only 1.34%.
    """

    # Signal A: Carry (the only proven signal)
    carry_vpin_max: float = 0.99        # Kline VPIN: DISABLED (noise)
    carry_vpin_exit: float = 0.99       # Kline VPIN: DISABLED
    carry_tick_vpin_max: float = 0.80   # Tick VPIN: block entry when > 0.80 (toxic flow)
    carry_tick_vpin_exit: float = 0.90  # Tick VPIN: exit when > 0.90 (extreme toxicity)
    carry_min_ann_funding: float = 20.0 # enter when smoothed ann funding > 20%
    carry_exit_funding: float = 3.0     # exit when < 3%

    # Signal B: Residual Carry -- KILLED (z ~ 0, no predictive value)
    residual_entry_z: float = 1.5
    residual_exit_z: float = 0.3

    # Signal C: Cascade Direction -- UNPROVEN (needs tick data)
    cascade_hawkes_entry: float = 3.0
    cascade_ofi_min: float = 0.3
    cascade_hold_bars: int = 6

    # Signal D: Post-Cascade Reversion -- UNPROVEN (needs tick data)
    revert_hawkes_max: float = 1.5
    revert_vpin_min: float = 0.30
    revert_ofi_min: float = 0.3
    revert_hold_bars: int = 12

    # Portfolio limits
    max_carry_positions: int = 10
    max_residual_positions: int = 0  # DISABLED -- PCA z-scores ~ 0, no alpha
    max_cascade_positions: int = 0   # DISABLED -- needs tick-level Hawkes
    max_reversion_positions: int = 0  # DISABLED -- needs tick-level data
    cost_per_leg_bps: float = 8.0

    # Universe -- $10M not $50M (captures 9 extra profitable symbols)
    min_volume_usd: float = 10_000_000
    vpin_bucket_size: float = 50_000
    vpin_n_buckets: int = 50
    kline_interval: str = "5m"
    kline_limit: int = 200


@dataclass(frozen=True)
class TradeSignal:
    """A trade decision from CLRS analysis."""

    symbol: str
    signal_type: str        # "carry_A", "residual_B", "cascade_C", "revert_D"
    direction: str          # "long" or "short"
    strength: float         # 0-1 signal strength
    reason: str
    vpin: float = 0.0
    hawkes_ratio: float = 0.0
    ofi: float = 0.0
    funding_residual: float = 0.0


def compute_regime(
    symbol: str,
    snapshot: SymbolSnapshot,
    funding_matrix: pd.DataFrame | None = None,
    config: SignalConfig | None = None,
) -> RegimeSnapshot:
    """Compute full regime assessment for one symbol.

    Makes REST calls to fetch recent klines and trades.
    """
    if config is None:
        config = SignalConfig()

    vpin_val = 0.0
    ofi_val = 0.0
    hawkes_ratio = 1.0
    lambda_val = 0.0
    funding_res = 0.0

    # Fetch klines for VPIN + Kyle's Lambda
    try:
        klines = fetch_recent_klines(
            symbol, interval=config.kline_interval, limit=config.kline_limit
        )
        if len(klines) > 30:
            prices = klines["Close"].values
            volumes_usd = klines["QuoteVolume"].values

            # VPIN
            vpin_series = compute_vpin_series(
                prices, volumes_usd,
                bucket_size=config.vpin_bucket_size,
                n_buckets=config.vpin_n_buckets,
            )
            vpin_val = vpin_series[~np.isnan(vpin_series)][-1] if np.any(~np.isnan(vpin_series)) else 0.0

            # Kyle's Lambda
            lam_series = kyles_lambda(prices, volumes_usd, window=50)
            valid_lam = lam_series[~np.isnan(lam_series)]
            lambda_val = valid_lam[-1] if len(valid_lam) > 0 else 0.0

            # Simple OFI proxy from klines (tick-level is better but needs WS)
            # Use taker buy ratio as proxy
            taker_buy = klines.get("Taker_buy_quote")
            if taker_buy is not None and len(taker_buy) > 0:
                buy_ratio = taker_buy.iloc[-1] / max(volumes_usd[-1], 1e-10)
                ofi_val = (buy_ratio - 0.5) * 2  # normalize to [-1, 1]
            else:
                # Fallback: use price direction as OFI proxy
                if len(prices) >= 5:
                    recent_ret = prices[-1] / prices[-5] - 1
                    ofi_val = np.clip(recent_ret * 20, -1, 1)

    except Exception as e:
        logger.warning("Kline fetch failed for %s: %s", symbol, e)

    # Hawkes intensity from trade arrivals
    try:
        timestamps = fetch_recent_trades(symbol, limit=1000)
        if len(timestamps) > 20:
            mu, alpha, beta = calibrate_hawkes(timestamps)
            from quantlaxmi.strategies.s26_crypto_flow.features import HawkesState
            state = HawkesState(mu=mu, alpha=alpha, beta=beta)
            for t in timestamps:
                state.event(t)
            hawkes_ratio = state.intensity_ratio(timestamps[-1])
    except Exception as e:
        logger.warning("Trade fetch failed for %s: %s", symbol, e)

    # Funding PCA residual
    if funding_matrix is not None and symbol in funding_matrix.columns and len(funding_matrix) >= 10:
        try:
            _, _, residuals = funding_pca(funding_matrix)
            col_idx = list(funding_matrix.columns).index(symbol)
            funding_res = residuals[-1, col_idx]
        except Exception as e:
            logger.debug("PCA failed for %s: %s", symbol, e)

    return RegimeSnapshot(
        symbol=symbol,
        vpin=vpin_val,
        ofi=ofi_val,
        hawkes_ratio=hawkes_ratio,
        kyles_lambda=lambda_val,
        funding_residual=funding_res,
    )


def generate_carry_signals(
    regimes: dict[str, RegimeSnapshot],
    snapshots: dict[str, SymbolSnapshot],
    active_carry: set[str],
    config: SignalConfig,
) -> list[TradeSignal]:
    """Signal A: Enhanced carry -- fund arb filtered by VPIN."""
    signals = []

    # Exits first
    for sym in list(active_carry):
        regime = regimes.get(sym)
        snap = snapshots.get(sym)
        if regime is None or snap is None:
            signals.append(TradeSignal(
                symbol=sym, signal_type="carry_A", direction="short",
                strength=0.0, reason="Symbol data missing",
            ))
            continue

        # VPIN exit: use tick threshold for tick-sourced, kline for kline-sourced
        vpin_exit_thresh = (config.carry_tick_vpin_exit
                           if regime.source == "tick" and regime.vpin > 0
                           else config.carry_vpin_exit)

        if regime.vpin > vpin_exit_thresh:
            signals.append(TradeSignal(
                symbol=sym, signal_type="carry_A", direction="exit",
                strength=0.0,
                reason=f"VPIN {regime.vpin:.2f} > {vpin_exit_thresh} ({regime.source}) exit",
                vpin=regime.vpin,
            ))
        elif snap.ann_funding_pct < config.carry_exit_funding:
            signals.append(TradeSignal(
                symbol=sym, signal_type="carry_A", direction="exit",
                strength=0.0, reason=f"Funding {snap.ann_funding_pct:.1f}% < {config.carry_exit_funding}%",
                vpin=regime.vpin,
            ))

    # Entries
    slots = config.max_carry_positions - len(active_carry)
    if slots <= 0:
        return signals

    candidates = []
    for sym, snap in snapshots.items():
        if sym in active_carry:
            continue
        regime = regimes.get(sym)
        if regime is None:
            continue

        # VPIN entry: use tick threshold for tick-sourced, kline for kline-sourced
        # Tick VPIN = 0.0 means not yet warmed up -- don't filter on it
        vpin_entry_thresh = (config.carry_tick_vpin_max
                            if regime.source == "tick" and regime.vpin > 0
                            else config.carry_vpin_max)

        if (snap.ann_funding_pct > config.carry_min_ann_funding
                and regime.vpin < vpin_entry_thresh
                and snap.volume_24h_usd >= config.min_volume_usd):
            # Strength: higher funding + lower VPIN = stronger signal
            strength = min(1.0, snap.ann_funding_pct / 50.0) * (1 - regime.vpin)
            candidates.append((sym, snap, regime, strength))

    candidates.sort(key=lambda x: x[3], reverse=True)
    for sym, snap, regime, strength in candidates[:slots]:
        vpin_thresh = (config.carry_tick_vpin_max
                       if regime.source == "tick" and regime.vpin > 0
                       else config.carry_vpin_max)
        signals.append(TradeSignal(
            symbol=sym, signal_type="carry_A", direction="long",
            strength=strength,
            reason=(f"Funding {snap.ann_funding_pct:+.1f}%, "
                    f"VPIN {regime.vpin:.2f} < {vpin_thresh} ({regime.source})"),
            vpin=regime.vpin,
        ))

    return signals


def generate_residual_signals(
    regimes: dict[str, RegimeSnapshot],
    snapshots: dict[str, SymbolSnapshot],
    active_residual: set[str],
    funding_matrix: pd.DataFrame,
    config: SignalConfig,
) -> list[TradeSignal]:
    """Signal B: Residual carry -- PCA-identified mispriced funding."""
    signals = []

    if len(funding_matrix) < 10 or len(funding_matrix.columns) < 5:
        return signals

    try:
        _, _, residuals = funding_pca(funding_matrix)
    except Exception as e:
        logger.debug("Funding PCA failed in residual signal generation: %s", e)
        return signals

    # Z-score the residuals
    res_last = residuals[-1]
    res_mean = np.nanmean(residuals, axis=0)
    res_std = np.nanstd(residuals, axis=0)
    res_std = np.where(res_std > 0, res_std, 1e-10)
    z_scores = (res_last - res_mean) / res_std

    sym_list = list(funding_matrix.columns)

    # Exits
    for sym in list(active_residual):
        if sym in sym_list:
            idx = sym_list.index(sym)
            if abs(z_scores[idx]) < config.residual_exit_z:
                signals.append(TradeSignal(
                    symbol=sym, signal_type="residual_B", direction="exit",
                    strength=0.0,
                    reason=f"Residual z={z_scores[idx]:+.2f} normalized",
                    funding_residual=res_last[idx],
                ))
        else:
            signals.append(TradeSignal(
                symbol=sym, signal_type="residual_B", direction="exit",
                strength=0.0, reason="Symbol dropped from matrix",
            ))

    # Entries
    slots = config.max_residual_positions - len(active_residual)
    if slots <= 0:
        return signals

    candidates = []
    for i, sym in enumerate(sym_list):
        if sym in active_residual:
            continue
        snap = snapshots.get(sym)
        if snap is None or snap.volume_24h_usd < config.min_volume_usd:
            continue

        if z_scores[i] > config.residual_entry_z:
            # Positive residual = symbol pays more than common factors explain
            strength = min(1.0, z_scores[i] / 3.0)
            candidates.append((sym, strength, z_scores[i], res_last[i]))

    candidates.sort(key=lambda x: x[1], reverse=True)
    for sym, strength, z, res in candidates[:slots]:
        signals.append(TradeSignal(
            symbol=sym, signal_type="residual_B", direction="long",
            strength=strength,
            reason=f"Funding residual z={z:+.2f}, raw={res:+.6f}",
            funding_residual=res,
        ))

    return signals


def generate_cascade_signals(
    regimes: dict[str, RegimeSnapshot],
    active_cascade: set[str],
    config: SignalConfig,
) -> list[TradeSignal]:
    """Signal C: Cascade direction -- momentum during liquidation cascades."""
    signals = []

    # Exits: cascade dissipated
    for sym in list(active_cascade):
        regime = regimes.get(sym)
        if regime is None or regime.hawkes_ratio < 2.0:
            signals.append(TradeSignal(
                symbol=sym, signal_type="cascade_C", direction="exit",
                strength=0.0,
                reason=f"Cascade over, Hawkes ratio={regime.hawkes_ratio:.1f}" if regime else "No data",
                hawkes_ratio=regime.hawkes_ratio if regime else 0.0,
            ))

    # Entries
    slots = config.max_cascade_positions - len(active_cascade)
    if slots <= 0:
        return signals

    candidates = []
    for sym, regime in regimes.items():
        if sym in active_cascade:
            continue
        if regime.hawkes_ratio > config.cascade_hawkes_entry and abs(regime.ofi) > config.cascade_ofi_min:
            direction = "long" if regime.ofi > 0 else "short"
            strength = min(1.0, regime.hawkes_ratio / 6.0) * min(1.0, abs(regime.ofi))
            candidates.append((sym, regime, direction, strength))

    candidates.sort(key=lambda x: x[3], reverse=True)
    for sym, regime, direction, strength in candidates[:slots]:
        signals.append(TradeSignal(
            symbol=sym, signal_type="cascade_C", direction=direction,
            strength=strength,
            reason=(f"Cascade: Hawkes={regime.hawkes_ratio:.1f}, "
                    f"OFI={regime.ofi:+.2f} -> {direction}"),
            hawkes_ratio=regime.hawkes_ratio,
            ofi=regime.ofi,
        ))

    return signals


def generate_reversion_signals(
    regimes: dict[str, RegimeSnapshot],
    active_reversion: set[str],
    config: SignalConfig,
) -> list[TradeSignal]:
    """Signal D: Post-cascade reversion -- fade the overshoot."""
    signals = []

    # Exits: reversion complete (VPIN normalized)
    for sym in list(active_reversion):
        regime = regimes.get(sym)
        if regime is None or regime.vpin < 0.15:
            signals.append(TradeSignal(
                symbol=sym, signal_type="revert_D", direction="exit",
                strength=0.0,
                reason=f"Reversion done, VPIN={regime.vpin:.2f}" if regime else "No data",
                vpin=regime.vpin if regime else 0.0,
            ))

    # Entries
    slots = config.max_reversion_positions - len(active_reversion)
    if slots <= 0:
        return signals

    candidates = []
    for sym, regime in regimes.items():
        if sym in active_reversion:
            continue
        # Post-cascade state: intensity dropping but VPIN still high
        if (regime.hawkes_ratio < config.revert_hawkes_max
                and regime.vpin > config.revert_vpin_min
                and abs(regime.ofi) > config.revert_ofi_min):
            # Fade the flow direction
            direction = "long" if regime.ofi < 0 else "short"
            strength = min(1.0, regime.vpin) * (1 - regime.hawkes_ratio / 3)
            candidates.append((sym, regime, direction, strength))

    candidates.sort(key=lambda x: x[3], reverse=True)
    for sym, regime, direction, strength in candidates[:slots]:
        signals.append(TradeSignal(
            symbol=sym, signal_type="revert_D", direction=direction,
            strength=strength,
            reason=(f"Post-cascade: VPIN={regime.vpin:.2f}, "
                    f"Hawkes={regime.hawkes_ratio:.1f}, "
                    f"OFI={regime.ofi:+.2f} -> fade to {direction}"),
            vpin=regime.vpin,
            hawkes_ratio=regime.hawkes_ratio,
            ofi=regime.ofi,
        ))

    return signals
