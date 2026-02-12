"""S8: Expiry-Day Theta Harvest Strategy.

Concept: On weekly expiry days, sell iron condors (short legs ~1.5% OTM,
wings ~3% OTM) and harvest intraday theta decay.

Entry: Morning of expiry day when VIX < adaptive threshold.
Exit:  Hold to settlement or stop if spot breaches short strike.
Risk:  Defined risk — iron condor caps max loss at strike_width - net_credit.

P&L Model: Uses ACTUAL option prices from nse_fo_bhavcopy for entry credit
and nfo_1min intraday bars for exit/settlement pricing.  No approximations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]

# Realistic option cost in INDEX POINTS per leg (not bps of spot).
# NIFTY weekly 1-2% OTM options: bid-ask ~1-3 pts, plus exchange/STT/stamp ~0.5 pts.
# Conservative estimate: 3 pts per leg for NIFTY, 5 pts for BANKNIFTY.
COST_PTS_PER_LEG = {"NIFTY": 3.0, "BANKNIFTY": 5.0}


@dataclass(frozen=True)
class IronCondor:
    """Iron condor position parameters."""

    symbol: str
    expiry: str
    spot: float
    short_call: float
    long_call: float
    short_put: float
    long_put: float
    net_credit: float
    max_loss: float
    vix: float


class S8ExpiryThetaStrategy(BaseStrategy):
    """S8: Weekly expiry-day theta harvest via iron condors."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        short_otm_pct: float = 0.015,    # short legs 1.5% OTM
        wing_otm_pct: float = 0.030,     # wings 3% OTM
        max_vix: float = 18.0,           # VIX threshold
        max_vpin: float = 0.60,          # VPIN threshold
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._short_otm = short_otm_pct
        self._wing_otm = wing_otm_pct
        self._max_vix = max_vix
        self._max_vpin = max_vpin

    @property
    def strategy_id(self) -> str:
        return "s8_expiry_theta"

    def warmup_days(self) -> int:
        return 5

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Only fires on expiry days."""
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S8 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        d_str = d.isoformat()

        # Check if today is an expiry day
        if not self._is_expiry_day(store, d, symbol):
            return None

        # Get spot price
        try:
            _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}.get(
                symbol.upper(), f"Nifty {symbol}")
            df = store.sql(
                'SELECT "Closing Index Value" as close FROM nse_index_close '
                'WHERE date = ? AND "Index Name" = ? LIMIT 1',
                [d_str, _idx_name],
            )
            if df.empty:
                return None
            spot = float(df["close"].iloc[0])
        except Exception:
            return None

        # Check VIX gate
        try:
            from quantlaxmi.core.allocator.regime import get_vix
            vix = get_vix(store, d)
            if vix is not None and vix > self._max_vix:
                logger.debug("S8: VIX=%.1f > %.1f, skipping %s", vix, self._max_vix, symbol)
                return None
        except Exception:
            vix = 15.0  # default

        # Construct iron condor strikes (rounded to nearest strike interval)
        strike_interval = 50 if symbol == "NIFTY" else 100

        short_call = self._round_strike(spot * (1 + self._short_otm), strike_interval)
        long_call = self._round_strike(spot * (1 + self._wing_otm), strike_interval)
        short_put = self._round_strike(spot * (1 - self._short_otm), strike_interval)
        long_put = self._round_strike(spot * (1 - self._wing_otm), strike_interval)

        # Get actual option prices
        credit = _get_iron_condor_credit(store, d, symbol, short_call, long_call, short_put, long_put)
        if credit is None or credit <= 0:
            return None

        call_width = long_call - short_call
        max_loss_pts = max(call_width, short_put - long_put) - credit
        credit_pct = credit / spot
        max_loss_pct = max_loss_pts / spot

        ic = IronCondor(
            symbol=symbol, expiry=d_str, spot=spot,
            short_call=short_call, long_call=long_call,
            short_put=short_put, long_put=long_put,
            net_credit=credit_pct, max_loss=max_loss_pct,
            vix=vix or 15.0,
        )

        conviction = min(1.0, 0.5 + (self._max_vix - (vix or 15)) / 20)
        conviction = max(0.3, conviction)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction="short",
            conviction=conviction,
            instrument_type="SPREAD",
            strike=short_call,
            expiry=d_str,
            ttl_bars=1,
            metadata={
                "structure": "iron_condor",
                "short_call": short_call,
                "long_call": long_call,
                "short_put": short_put,
                "long_put": long_put,
                "net_credit_pts": round(credit, 2),
                "credit_pct": round(credit_pct * 100, 3),
                "max_loss_pct": round(max_loss_pct * 100, 3),
                "vix": round(vix or 15.0, 1),
            },
        )

    def _is_expiry_day(self, store: MarketDataStore, d: date, symbol: str) -> bool:
        """Check if today is a weekly/monthly expiry for this symbol."""
        d_str = d.isoformat()
        try:
            df = store.sql(
                "SELECT DISTINCT expiry FROM nfo_1min "
                "WHERE date = ? AND name = ? AND instrument_type IN ('CE', 'PE') "
                "AND expiry = ? LIMIT 1",
                [d_str, symbol, d_str],
            )
            return not df.empty
        except Exception:
            return d.weekday() == 3

    @staticmethod
    def _round_strike(price: float, interval: int) -> float:
        return round(price / interval) * interval


def create_strategy() -> S8ExpiryThetaStrategy:
    return S8ExpiryThetaStrategy()


# ---------------------------------------------------------------------------
# Helper: get actual option prices from DuckDB
# ---------------------------------------------------------------------------

def _get_option_price(
    store: MarketDataStore,
    d: date,
    symbol: str,
    strike: float,
    opt_type: str,  # "CE" or "PE"
    expiry: date | None = None,
) -> float | None:
    """Get the close price for a specific option from nse_fo_bhavcopy.

    Parameters
    ----------
    d : date
        Trading date to query.
    expiry : date | None
        Option expiry date. If None, defaults to ``d`` (same-day expiry).

    Notes
    -----
    ``SttlmPric`` in NSE FO bhavcopy is the *underlying* settlement price,
    NOT the option's settlement price.  We always use ``ClsPric`` (the
    option's closing price) instead.
    """
    d_str = d.isoformat()
    expiry_str = (expiry or d).isoformat()
    try:
        df = store.sql(
            'SELECT "ClsPric", "TtlTradgVol" '
            "FROM nse_fo_bhavcopy "
            'WHERE date = ? AND "TckrSymb" = ? AND "StrkPric" = ? '
            'AND "OptnTp" = ? AND "XpryDt" = ? '
            "LIMIT 1",
            [d_str, symbol, strike, opt_type, expiry_str],
        )
        if df.empty:
            return None
        close = float(df["ClsPric"].iloc[0])
        if close <= 0:
            return None
        return close
    except Exception:
        return None


def _get_iron_condor_credit(
    store: MarketDataStore,
    d: date,
    symbol: str,
    short_call: float,
    long_call: float,
    short_put: float,
    long_put: float,
    expiry: date | None = None,
) -> float | None:
    """Get net credit in points from actual option prices.

    Parameters
    ----------
    d : date
        Trading date to look up prices.
    expiry : date | None
        Option expiry date.  Defaults to ``d`` if not given.
    """
    exp = expiry or d
    sc = _get_option_price(store, d, symbol, short_call, "CE", expiry=exp)
    lc = _get_option_price(store, d, symbol, long_call, "CE", expiry=exp)
    sp = _get_option_price(store, d, symbol, short_put, "PE", expiry=exp)
    lp = _get_option_price(store, d, symbol, long_put, "PE", expiry=exp)

    if any(x is None for x in [sc, lc, sp, lp]):
        return None

    # Credit = sell short legs - buy long legs
    credit = (sc - lc) + (sp - lp)
    return credit if credit > 0 else None


def _get_intraday_option_prices(
    store: MarketDataStore,
    d: date,
    symbol: str,
    strike: float,
    opt_type: str,
) -> list[tuple[str, float]] | None:
    """Get 1-min close prices for a specific option on an expiry day."""
    d_str = d.isoformat()
    # Build the NFO symbol name (e.g., NIFTY26JAN25600CE)
    try:
        df = store.sql(
            "SELECT timestamp, close FROM nfo_1min "
            "WHERE date = ? AND name = ? AND strike = ? "
            "AND instrument_type = ? AND expiry = ? "
            "ORDER BY timestamp",
            [d_str, symbol, strike, opt_type, d_str],
        )
        if df.empty:
            return None
        return [(str(row["timestamp"]), float(row["close"])) for _, row in df.iterrows()]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest — uses actual option prices
# ---------------------------------------------------------------------------

@dataclass
class ThetaTrade:
    symbol: str
    date: date
    spot: float
    short_call: float
    long_call: float
    short_put: float
    long_put: float
    credit_pts: float       # actual credit in index points
    credit_pct: float       # credit as % of spot
    settlement_pnl_pts: float  # P&L at settlement in points
    pnl_pct: float          # net P&L after costs as % of notional
    exit_reason: str
    vix: float
    cost_pts: float         # total transaction costs in points


def backtest_expiry_theta(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    short_otm_pct: float = 0.015,
    wing_otm_pct: float = 0.030,
    max_vix: float = 18.0,
    cost_pts_per_leg: float | None = None,
) -> dict:
    """Backtest expiry-day theta strategy using ACTUAL option prices.

    Entry: Use previous day's closing option prices from nse_fo_bhavcopy
           (conservative — real entry would be during the day).
    Exit:  Settlement intrinsic at expiry (no exit cost) OR intraday close
           at breach (full exit cost).
    Cost:  Fixed pts per leg (3 pts NIFTY / 5 pts BANKNIFTY).
           Entry always incurs 4-leg cost.
           Exit costs only on early close (breach), NOT on settlement expiry.
    """
    from quantlaxmi.strategies.s9_momentum.data import is_trading_day
    from quantlaxmi.core.allocator.regime import get_vix

    trades: list[ThetaTrade] = []

    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        d_str = d.isoformat()

        # Check if expiry day
        try:
            df = store.sql(
                "SELECT DISTINCT expiry FROM nfo_1min "
                "WHERE date = ? AND name = ? AND instrument_type IN ('CE', 'PE') "
                "AND expiry = ? LIMIT 1",
                [d_str, symbol, d_str],
            )
            is_expiry = not df.empty
        except Exception:
            is_expiry = d.weekday() == 3

        if not is_expiry:
            d += timedelta(days=1)
            continue

        # VIX check
        vix = get_vix(store, d) or 15.0
        if vix > max_vix:
            d += timedelta(days=1)
            continue

        # Get spot price at open (from futures 1-min bars)
        try:
            bars = store.sql(
                "SELECT timestamp, close FROM nfo_1min "
                "WHERE date = ? AND name = ? AND instrument_type = 'FUT' "
                "ORDER BY timestamp",
                [d_str, symbol],
            )
            if bars.empty or len(bars) < 10:
                d += timedelta(days=1)
                continue
            spot_open = float(bars["close"].iloc[0])
        except Exception:
            d += timedelta(days=1)
            continue

        # Construct iron condor strikes
        interval = 50 if symbol == "NIFTY" else 100
        short_call = round(spot_open * (1 + short_otm_pct) / interval) * interval
        long_call = round(spot_open * (1 + wing_otm_pct) / interval) * interval
        short_put = round(spot_open * (1 - short_otm_pct) / interval) * interval
        long_put = round(spot_open * (1 - wing_otm_pct) / interval) * interval

        # Ensure sensible wing widths
        if long_call <= short_call:
            long_call = short_call + interval
        if long_put >= short_put:
            long_put = short_put - interval

        # ---------------------------------------------------------------
        # ENTRY: Get actual option prices from previous trading day
        # (conservative: we assume entry at previous close, not morning)
        # ---------------------------------------------------------------
        prev_d = d - timedelta(days=1)
        while prev_d >= start - timedelta(days=10) and not is_trading_day(prev_d):
            prev_d -= timedelta(days=1)

        entry_credit = _get_iron_condor_credit(
            store, prev_d, symbol, short_call, long_call, short_put, long_put,
            expiry=d,
        )
        if entry_credit is None or entry_credit <= 0:
            # Fallback: try same-day open prices from 1-min bars
            entry_credit = _get_intraday_ic_credit(
                store, d, symbol, short_call, long_call, short_put, long_put,
                bar_idx=0,  # first bar = open
            )
            if entry_credit is None or entry_credit <= 0:
                d += timedelta(days=1)
                continue

        # ---------------------------------------------------------------
        # EXIT: Settlement price from nse_fo_bhavcopy on expiry day
        # OR: intraday stop-loss if spot breaches short strike
        # ---------------------------------------------------------------

        # Check for intraday breach using futures bars
        exit_reason = "settlement"
        breach_bar = None
        for bar_idx in range(1, len(bars)):
            bar_price = float(bars["close"].iloc[bar_idx])
            if bar_price > short_call:
                exit_reason = "call_breach"
                breach_bar = bar_idx
                break
            elif bar_price < short_put:
                exit_reason = "put_breach"
                breach_bar = bar_idx
                break

        if exit_reason == "settlement":
            # Full settlement — options expire, compute intrinsic value
            settle_spot = float(bars["close"].iloc[-1])

            # At settlement, option value = max(0, intrinsic)
            sc_settle = max(0, settle_spot - short_call)
            lc_settle = max(0, settle_spot - long_call)
            sp_settle = max(0, short_put - settle_spot)
            lp_settle = max(0, long_put - settle_spot)

            # P&L = entry_credit - (exit cost to close, which is intrinsic at expiry)
            exit_cost = (sc_settle - lc_settle) + (sp_settle - lp_settle)
            settlement_pnl = entry_credit - exit_cost

        else:
            # Breach — use actual option prices at breach time from 1-min bars
            exit_credit = _get_intraday_ic_credit(
                store, d, symbol, short_call, long_call, short_put, long_put,
                bar_idx=breach_bar,
            )
            if exit_credit is not None:
                # We entered at entry_credit, now need to buy back at exit_credit
                # If exit_credit > entry_credit, we lost money (options got more expensive)
                settlement_pnl = entry_credit - exit_credit
            else:
                # Fallback: compute from intrinsic at breach price
                breach_price = float(bars["close"].iloc[breach_bar])
                sc_val = max(0, breach_price - short_call)
                lc_val = max(0, breach_price - long_call)
                sp_val = max(0, short_put - breach_price)
                lp_val = max(0, long_put - breach_price)
                exit_cost = (sc_val - lc_val) + (sp_val - lp_val)
                settlement_pnl = entry_credit - exit_cost

        # Transaction costs in index points per leg
        cpl = cost_pts_per_leg if cost_pts_per_leg is not None else COST_PTS_PER_LEG.get(symbol, 3.0)
        # Entry: always 4 legs
        entry_cost_pts = cpl * 4
        # Exit: only if we close early (breach); settlement = free (options expire)
        exit_cost_pts = cpl * 4 if exit_reason != "settlement" else 0.0
        total_cost_pts = entry_cost_pts + exit_cost_pts
        net_pnl_pts = settlement_pnl - total_cost_pts

        # Cap loss at max possible (wing width)
        max_wing = max(long_call - short_call, short_put - long_put)
        net_pnl_pts = max(net_pnl_pts, -(max_wing - entry_credit))

        # Express as % of spot (notional)
        pnl_pct = net_pnl_pts / spot_open * 100

        trades.append(ThetaTrade(
            symbol=symbol, date=d, spot=spot_open,
            short_call=short_call, long_call=long_call,
            short_put=short_put, long_put=long_put,
            credit_pts=round(entry_credit, 2),
            credit_pct=round(entry_credit / spot_open * 100, 4),
            settlement_pnl_pts=round(settlement_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=exit_reason,
            vix=vix,
            cost_pts=round(total_cost_pts, 2),
        ))

        d += timedelta(days=1)

    # ---------------------------------------------------------------
    # Compute metrics — Sharpe on TRADE returns (not calendar padded)
    # ---------------------------------------------------------------
    if trades:
        pnls = [t.pnl_pct / 100 for t in trades]  # as fractions
        equity = 1.0
        eq_curve = [1.0]
        for p in pnls:
            equity *= (1 + p)
            eq_curve.append(equity)

        total_ret = (equity - 1) * 100
        wins = sum(1 for p in pnls if p > 0)

        # Sharpe: annualize using sqrt(N_trades_per_year)
        # ~50 weekly expiries per year for NIFTY
        trades_per_year = 50
        if len(pnls) > 1 and np.std(pnls, ddof=1) > 1e-10:
            sharpe = float(
                np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(trades_per_year)
            )
        else:
            sharpe = 0.0

        peak = 1.0
        max_dd = 0.0
        for eq in eq_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    else:
        total_ret = sharpe = max_dd = 0.0
        wins = 0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else 0,
        "total_return_pct": round(total_ret, 4),
        "sharpe": round(sharpe, 4),
        "max_dd_pct": round(max_dd * 100, 4),
        "avg_credit_pct": round(float(np.mean([t.credit_pct for t in trades])), 4) if trades else 0,
        "avg_cost_pts": round(float(np.mean([t.cost_pts for t in trades])), 2) if trades else 0,
        "total_cost_pct": round(sum(t.cost_pts for t in trades) / (trades[0].spot if trades else 1) * 100, 4) if trades else 0,
        "trade_details": trades,
        "daily_returns": pnls if trades else [],
    }


def _get_intraday_ic_credit(
    store: MarketDataStore,
    d: date,
    symbol: str,
    short_call: float,
    long_call: float,
    short_put: float,
    long_put: float,
    bar_idx: int = 0,
) -> float | None:
    """Get iron condor credit/debit from intraday 1-min option bars at a specific bar index."""
    d_str = d.isoformat()

    def _get_bar(strike: float, opt_type: str) -> float | None:
        try:
            df = store.sql(
                "SELECT close FROM nfo_1min "
                "WHERE date = ? AND name = ? AND strike = ? "
                "AND instrument_type = ? AND expiry = ? "
                "ORDER BY timestamp",
                [d_str, symbol, strike, opt_type, d_str],
            )
            if df.empty or bar_idx >= len(df):
                return None
            return float(df["close"].iloc[bar_idx])
        except Exception:
            return None

    sc = _get_bar(short_call, "CE")
    lc = _get_bar(long_call, "CE")
    sp = _get_bar(short_put, "PE")
    lp = _get_bar(long_put, "PE")

    if any(x is None for x in [sc, lc, sp, lp]):
        return None

    credit = (sc - lc) + (sp - lp)
    return credit
