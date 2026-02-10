"""CLRS Strategy -- execution logic for all 4 signals.

Orchestrates:
  1. Regime computation (VPIN, OFI, Hawkes, Lambda) for top symbols
  2. Signal generation for all 4 channels
  3. Trade execution against portfolio state
  4. Funding settlement crediting for carry positions

Also provides ``S26CryptoFlowStrategy`` — a StrategyProtocol-compliant
wrapper so that S26 can be auto-discovered by the strategy registry.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from quantlaxmi.strategies.s26_crypto_flow.scanner import (
    FundingMatrixBuilder,
    SymbolSnapshot,
    scan_all_symbols,
)
from quantlaxmi.strategies.s26_crypto_flow.signals import (
    SignalConfig,
    TradeSignal,
    compute_regime,
    generate_carry_signals,
    generate_cascade_signals,
    generate_residual_signals,
    generate_reversion_signals,
)
from quantlaxmi.strategies.s26_crypto_flow.state import Position, PortfolioState

logger = logging.getLogger(__name__)


def scan_and_compute(
    state: PortfolioState,
    config: SignalConfig,
    tick_regimes: dict | None = None,
) -> tuple[list[SymbolSnapshot], dict]:
    """Phase 1: Scan market and compute regimes.

    Parameters
    ----------
    tick_regimes : dict, optional
        Pre-computed regimes from LiveRegimeTracker (tick-level features).
        If provided, these override kline-based regime computation for
        symbols that have tick data. Symbols without tick data fall back
        to the kline-based approach.

    Returns (snapshots, context_dict).
    """
    # Scan all symbols
    snapshots = scan_all_symbols(min_volume_usd=config.min_volume_usd)
    snap_map = {s.symbol: s for s in snapshots}

    # Build/update funding matrix
    fm_builder = FundingMatrixBuilder.from_dict(
        state.funding_matrix_state, max_rows=90
    )
    fm_builder.add_observation(snapshots)
    state.funding_matrix_state = fm_builder.to_dict()
    funding_matrix = fm_builder.to_dataframe()

    # Determine which symbols need regime computation
    symbols_to_evaluate = set()

    # All held positions
    for pos_dict in [state.carry_positions, state.residual_positions,
                     state.cascade_positions, state.reversion_positions]:
        symbols_to_evaluate.update(pos_dict.keys())

    # Top 20 by absolute funding rate
    top_funding = sorted(snapshots, key=lambda s: abs(s.ann_funding_pct), reverse=True)[:20]
    symbols_to_evaluate.update(s.symbol for s in top_funding)

    # Top 10 by volume (cascade candidates)
    top_volume = sorted(snapshots, key=lambda s: s.volume_24h_usd, reverse=True)[:10]
    symbols_to_evaluate.update(s.symbol for s in top_volume)

    logger.info("Evaluating %d symbols for regime computation", len(symbols_to_evaluate))

    regimes = {}

    # Use tick-level regimes where available
    if tick_regimes:
        for sym in symbols_to_evaluate:
            if sym in tick_regimes:
                regimes[sym] = tick_regimes[sym]
        tick_count = len(regimes)
        logger.info("Using %d tick-level regimes, computing %d from klines",
                     tick_count, len(symbols_to_evaluate) - tick_count)

    # Fall back to kline-based for the rest
    for sym in symbols_to_evaluate:
        if sym in regimes:
            continue
        snap = snap_map.get(sym)
        if snap is None:
            continue
        try:
            regime = compute_regime(sym, snap, funding_matrix, config)
            regimes[sym] = regime
        except Exception as e:
            logger.warning("Regime computation failed for %s: %s", sym, e)

    return snapshots, {"regimes": regimes, "snap_map": snap_map, "funding_matrix": funding_matrix}


def generate_all_signals(
    state: PortfolioState,
    regimes: dict,
    snap_map: dict,
    funding_matrix,
    config: SignalConfig,
) -> list[TradeSignal]:
    """Phase 2: Generate signals from all 4 channels."""
    all_signals = []

    # Signal A: Enhanced Carry
    carry_signals = generate_carry_signals(
        regimes, snap_map,
        set(state.carry_positions.keys()),
        config,
    )
    all_signals.extend(carry_signals)

    # Signal B: Residual Carry
    residual_signals = generate_residual_signals(
        regimes, snap_map,
        set(state.residual_positions.keys()),
        funding_matrix,
        config,
    )
    all_signals.extend(residual_signals)

    # Signal C: Cascade Direction
    cascade_signals = generate_cascade_signals(
        regimes,
        set(state.cascade_positions.keys()),
        config,
    )
    all_signals.extend(cascade_signals)

    # Signal D: Post-Cascade Reversion
    reversion_signals = generate_reversion_signals(
        regimes,
        set(state.reversion_positions.keys()),
        config,
    )
    all_signals.extend(reversion_signals)

    return all_signals


def execute_signals(
    state: PortfolioState,
    signals: list[TradeSignal],
    snap_map: dict[str, SymbolSnapshot],
    config: SignalConfig,
) -> list[str]:
    """Phase 3: Apply trade signals to portfolio state.

    Modifies state in-place. Returns log messages.
    """
    messages = []
    cost_frac = config.cost_per_leg_bps / 10_000
    now = datetime.now(timezone.utc).isoformat()

    for signal in signals:
        positions = state.positions_by_type(signal.signal_type)

        if signal.direction == "exit":
            pos = positions.pop(signal.symbol, None)
            if pos is not None:
                state.equity *= (1 - cost_frac)
                state.total_costs_paid += cost_frac
                state.total_exits += 1
                if pos.net_pnl > 0:
                    state.total_wins += 1

                state.log_trade(
                    signal.symbol, signal.signal_type, "exit",
                    pos.direction, signal.reason, pos.net_pnl,
                )
                msg = (f"EXIT  [{signal.signal_type}] {signal.symbol:12s}  "
                       f"pnl={pos.net_pnl:+.4f}  {signal.reason}")
                messages.append(msg)
                logger.info(msg)

        elif signal.direction in ("long", "short"):
            # Entry
            snap = snap_map.get(signal.symbol)
            price = snap.mark_price if snap else 0.0

            state.equity *= (1 - cost_frac)
            state.total_costs_paid += cost_frac
            state.total_entries += 1

            # Equal weight within signal type
            n_in_pool = len(positions) + 1
            weight = 1.0 / max(n_in_pool, 1)

            positions[signal.symbol] = Position(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                direction=signal.direction,
                entry_time=now,
                entry_price=price,
                notional_weight=weight,
                strength=signal.strength,
                accumulated_cost=cost_frac,
                reason=signal.reason,
            )

            state.log_trade(
                signal.symbol, signal.signal_type, "enter",
                signal.direction, signal.reason,
            )
            msg = (f"ENTER [{signal.signal_type}] {signal.symbol:12s}  "
                   f"{signal.direction:5s}  str={signal.strength:.2f}  "
                   f"{signal.reason}")
            messages.append(msg)
            logger.info(msg)

    # Rebalance weights within each pool
    for pool in [state.carry_positions, state.residual_positions,
                 state.cascade_positions, state.reversion_positions]:
        if pool:
            w = 1.0 / len(pool)
            for pos in pool.values():
                pos.notional_weight = w

    # Record equity on any trade
    had_trades = any(s.direction in ("long", "short", "exit") for s in signals)
    if had_trades:
        state.record_equity("trade")

    state.last_scan_time = now
    return messages


def record_funding_settlement(
    state: PortfolioState,
    snap_map: dict[str, SymbolSnapshot],
) -> list[str]:
    """Credit funding to carry positions (Signal A + B) on settlement."""
    messages = []

    # Only carry-type positions earn funding
    for pool_name, pool in [("carry_A", state.carry_positions),
                            ("residual_B", state.residual_positions)]:
        for sym, pos in pool.items():
            snap = snap_map.get(sym)
            if snap is None:
                continue

            # For short perp: positive funding rate = we EARN
            # (funding flows from longs to shorts when rate is positive)
            funding_earned = snap.funding_rate * pos.notional_weight
            pos.accumulated_pnl += funding_earned
            pos.n_settlements += 1
            state.total_funding_earned += funding_earned
            state.equity *= (1 + funding_earned)

            msg = (f"FUND  [{pool_name}] {sym:12s}  "
                   f"rate={snap.funding_rate:+.6f}  "
                   f"earned={funding_earned:+.6f}")
            messages.append(msg)
            logger.debug(msg)

    # Increment hold bars for time-limited positions (cascade, reversion)
    for pool in [state.cascade_positions, state.reversion_positions]:
        for pos in pool.values():
            pos.hold_bars += 1

    if state.carry_positions or state.residual_positions:
        state.record_equity("settlement")

    return messages


def check_ttl_exits(
    state: PortfolioState,
    config: SignalConfig,
) -> list[TradeSignal]:
    """Check cascade/reversion positions that have exceeded their hold TTL."""
    exits = []

    for sym, pos in list(state.cascade_positions.items()):
        if pos.hold_bars >= config.cascade_hold_bars:
            exits.append(TradeSignal(
                symbol=sym, signal_type="cascade_C", direction="exit",
                strength=0.0,
                reason=f"TTL expired ({pos.hold_bars} bars)",
            ))

    for sym, pos in list(state.reversion_positions.items()):
        if pos.hold_bars >= config.revert_hold_bars:
            exits.append(TradeSignal(
                symbol=sym, signal_type="revert_D", direction="exit",
                strength=0.0,
                reason=f"TTL expired ({pos.hold_bars} bars)",
            ))

    return exits


# ---------------------------------------------------------------------------
# StrategyProtocol-compliant wrapper
# ---------------------------------------------------------------------------

class S26CryptoFlowStrategy:
    """StrategyProtocol wrapper for the CLRS crypto funding carry strategy.

    This adapter lets S26 participate in the unified strategy registry and
    orchestrator alongside India FnO strategies.  Because CLRS is a crypto
    strategy that scans Binance funding rates (not NSE data via
    MarketDataStore), the ``scan()`` method issues live REST calls and
    does not use the ``store`` parameter.  An empty signal list is returned
    if the Binance API is unreachable.
    """

    def __init__(self, config: SignalConfig | None = None):
        self._config = config or SignalConfig()

    @property
    def strategy_id(self) -> str:
        return "s26_crypto_flow"

    def warmup_days(self) -> int:
        # Crypto strategy uses live data, no historical warmup needed
        return 0

    def scan(self, d: date, store=None) -> list:
        """Scan Binance funding rates and return StrategyProtocol Signals.

        Parameters
        ----------
        d : date
            The scan date (used for logging; actual data is live).
        store : MarketDataStore, optional
            Not used — CLRS scans Binance REST APIs directly.

        Returns
        -------
        list[Signal]
            Zero or more Signal objects conforming to StrategyProtocol.
        """
        from quantlaxmi.strategies.protocol import Signal

        try:
            snapshots = scan_all_symbols(min_volume_usd=self._config.min_volume_usd)
        except Exception as e:
            logger.warning("S26 Binance scan failed: %s", e)
            return []

        if not snapshots:
            return []

        signals = []
        for snap in snapshots:
            if snap.ann_funding_pct > self._config.carry_min_ann_funding:
                # Positive funding = short perp to earn carry
                strength = min(1.0, snap.ann_funding_pct / 50.0)
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=snap.symbol,
                    direction="short",
                    conviction=strength,
                    instrument_type="FUT",
                    ttl_bars=24,  # ~8h funding intervals
                    metadata={
                        "ann_funding_pct": round(snap.ann_funding_pct, 2),
                        "funding_rate": snap.funding_rate,
                        "volume_24h_usd": snap.volume_24h_usd,
                        "mark_price": snap.mark_price,
                    },
                ))

        if signals:
            logger.info(
                "[s26_crypto_flow] %s: %d carry signal(s)",
                d.isoformat(), len(signals),
            )
        return signals


def create_strategy() -> S26CryptoFlowStrategy:
    """Factory for registry auto-discovery."""
    return S26CryptoFlowStrategy()
