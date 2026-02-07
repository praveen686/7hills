"""Equity curve comparator for E2E determinism verification.

Compares day-by-day equity curves, drawdowns, position counts, and
trade histories between two replay runs with floating-point tolerance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Floating-point tolerance (same as event comparator)
_FP_RTOL = 1e-10
_FP_ATOL = 1e-12


@dataclass(frozen=True)
class EquityDiff:
    """Single difference between two equity curves."""

    index: int
    field: str
    value_a: object
    value_b: object


@dataclass
class EquityCurveComparison:
    """Result of comparing two equity curves."""

    identical: bool = True
    total_compared: int = 0
    diffs: list[EquityDiff] = field(default_factory=list)
    equity_points_a: int = 0
    equity_points_b: int = 0
    trades_a: int = 0
    trades_b: int = 0

    def summary(self) -> str:
        lines = [
            f"Equity Curve Parity: {'PASS' if self.identical else 'FAIL'}",
            f"  Points: a={self.equity_points_a} b={self.equity_points_b}",
            f"  Trades: a={self.trades_a} b={self.trades_b}",
            f"  Diffs: {len(self.diffs)}",
        ]
        for d in self.diffs[:10]:
            lines.append(
                f"    [{d.index}] {d.field}: {d.value_a!r} vs {d.value_b!r}"
            )
        return "\n".join(lines)


def _values_equal(a: object, b: object) -> bool:
    """Compare two values with FP tolerance."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return a == b
        if abs(a) < _FP_ATOL and abs(b) < _FP_ATOL:
            return True
        return abs(a - b) <= _FP_RTOL * max(abs(a), abs(b))
    return a == b


def compare_equity_curves(
    state_a,
    state_b,
) -> EquityCurveComparison:
    """Compare two BrahmastraState objects for equity curve parity.

    Checks:
      - equity_history (day-by-day equity, drawdown)
      - closed_trades (count, pnl_pct, entry/exit dates)
      - final equity, peak_equity, position count

    Parameters
    ----------
    state_a, state_b : BrahmastraState
        States to compare (typically from two independent replay runs).

    Returns
    -------
    EquityCurveComparison with parity result.
    """
    result = EquityCurveComparison()

    hist_a = state_a.equity_history
    hist_b = state_b.equity_history
    result.equity_points_a = len(hist_a)
    result.equity_points_b = len(hist_b)
    result.trades_a = len(state_a.closed_trades)
    result.trades_b = len(state_b.closed_trades)

    # 1. Equity history length
    if len(hist_a) != len(hist_b):
        result.identical = False
        result.diffs.append(EquityDiff(
            index=0,
            field="equity_history_length",
            value_a=len(hist_a),
            value_b=len(hist_b),
        ))

    # 2. Day-by-day equity comparison
    n = min(len(hist_a), len(hist_b))
    for i in range(n):
        result.total_compared += 1
        ha, hb = hist_a[i], hist_b[i]

        for fld in ("date", "equity", "drawdown", "day_pnl"):
            va = ha.get(fld)
            vb = hb.get(fld)
            if not _values_equal(va, vb):
                result.identical = False
                result.diffs.append(EquityDiff(
                    index=i, field=f"equity_history.{fld}",
                    value_a=va, value_b=vb,
                ))

    # 3. Final state comparison
    for fld in ("equity", "peak_equity"):
        va = getattr(state_a, fld)
        vb = getattr(state_b, fld)
        if not _values_equal(va, vb):
            result.identical = False
            result.diffs.append(EquityDiff(
                index=-1, field=f"final.{fld}",
                value_a=va, value_b=vb,
            ))
        result.total_compared += 1

    # 4. Position count
    pa = len(state_a.active_positions())
    pb = len(state_b.active_positions())
    if pa != pb:
        result.identical = False
        result.diffs.append(EquityDiff(
            index=-1, field="position_count",
            value_a=pa, value_b=pb,
        ))
    result.total_compared += 1

    # 5. Trade history parity
    if len(state_a.closed_trades) != len(state_b.closed_trades):
        result.identical = False
        result.diffs.append(EquityDiff(
            index=-1, field="trade_count",
            value_a=len(state_a.closed_trades),
            value_b=len(state_b.closed_trades),
        ))
    else:
        for i, (ta, tb) in enumerate(
            zip(state_a.closed_trades, state_b.closed_trades)
        ):
            result.total_compared += 1
            for fld in ("strategy_id", "symbol", "direction",
                         "entry_date", "exit_date", "pnl_pct",
                         "entry_price", "exit_price"):
                va = getattr(ta, fld)
                vb = getattr(tb, fld)
                if not _values_equal(va, vb):
                    result.identical = False
                    result.diffs.append(EquityDiff(
                        index=i, field=f"trade.{fld}",
                        value_a=va, value_b=vb,
                    ))

    return result
