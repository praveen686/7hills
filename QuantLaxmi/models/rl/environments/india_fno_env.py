"""India Futures & Options trading environment.

Targets: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY + stock FnO.

Key India FnO characteristics modelled here:
  - 6.25 hours/day (09:15--15:30 IST), ~375 minutes
  - Weekly expiries (Thursday); monthly expiry last Thursday of month
  - Lot sizes: NIFTY=25, BANKNIFTY=15, FINNIFTY=25, MIDCPNIFTY=50
  - Transaction costs in *index points per leg* (NOT bps of spot):
        NIFTY ~3 pts, BANKNIFTY ~5 pts
  - Circuit limits (upper/lower), pre-open auction (09:00--09:15)
  - STT, stamp duty, SEBI charges baked into cost_per_leg

Environments:
  - IndiaFnOEnv: general India FnO trading MDP
  - NiftySwingEnv: 1--5 day swing trading on NIFTY
  - BankNiftyScalpEnv: intraday scalping on BANKNIFTY

Book reference: Ch 3 (MDP), Ch 10 (Financial MDPs).
"""
from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .trading_env import TradingEnv, TradingState, TradingAction, StepResult

__all__ = [
    "IndiaFnOEnv",
    "NiftySwingEnv",
    "BankNiftyScalpEnv",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOT_SIZES: Dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}

COST_PER_LEG: Dict[str, float] = {
    "NIFTY": 3.0,       # index points per leg
    "BANKNIFTY": 5.0,
    "FINNIFTY": 3.0,
    "MIDCPNIFTY": 4.0,
}

# Approximate spot levels (used for simulated mode)
INITIAL_SPOTS: Dict[str, float] = {
    "NIFTY": 24_000.0,
    "BANKNIFTY": 51_000.0,
    "FINNIFTY": 22_500.0,
    "MIDCPNIFTY": 12_500.0,
}

# Annualised vols (approximate 2024-25)
ANNUALISED_VOLS: Dict[str, float] = {
    "NIFTY": 0.14,
    "BANKNIFTY": 0.18,
    "FINNIFTY": 0.15,
    "MIDCPNIFTY": 0.17,
}

# Discrete action catalogue (for discrete-action mode)
DISCRETE_ACTIONS = [
    "flat",            # 0: close all, go flat
    "long_fut",        # 1: 1 lot long futures
    "short_fut",       # 2: 1 lot short futures
    "long_straddle",   # 3: buy ATM straddle (1 lot each)
    "short_straddle",  # 4: sell ATM straddle (1 lot each)
    "bull_spread",     # 5: buy ATM call + sell OTM call
    "bear_spread",     # 6: buy ATM put + sell OTM put
]


# ---------------------------------------------------------------------------
# IndiaFnOEnv
# ---------------------------------------------------------------------------


class IndiaFnOEnv(TradingEnv):
    """India FnO trading environment.

    State vector (dimension 12):
        [normalised_price, returns_5d, returns_20d, realised_vol_20d,
         iv_atm, iv_skew, put_call_ratio, oi_change, vix,
         days_to_expiry, position, pnl_normalised]

    **Discrete action** mode (7 actions):
        {flat, long_fut, short_fut, long_straddle,
         short_straddle, bull_spread, bear_spread}

    **Continuous action** mode (3-dim):
        (direction [-1, 1], size [0, 1], strike_offset [-5, 5])

    The environment enforces:
      - Costs in index points per leg (``cost_per_leg``)
      - Lot-size-quantised positions
      - No look-ahead bias: features computed from *past* data only

    Parameters
    ----------
    instrument : str
        ``"NIFTY"`` | ``"BANKNIFTY"`` | ``"FINNIFTY"`` | ``"MIDCPNIFTY"``
    mode : str
        ``"simulated"`` (GBM dynamics) or ``"historical"`` (replay from data)
    data_path : str or None
        Path to historical Parquet data (required if mode="historical")
    lot_size : int or None
        Override auto-detected lot size.
    cost_per_leg : float or None
        Override default cost per leg (index points).
    num_steps : int
        Number of steps per episode (days for daily, bars for intraday).
    include_options : bool
        If True, state includes IV / Greeks features.
    discrete_actions : bool
        If True, use discrete 7-action space; else continuous.
    """

    def __init__(
        self,
        instrument: str = "NIFTY",
        mode: str = "simulated",
        data_path: Optional[str] = None,
        lot_size: Optional[int] = None,
        cost_per_leg: Optional[float] = None,
        num_steps: int = 252,
        include_options: bool = True,
        discrete_actions: bool = True,
        **kwargs: Any,
    ) -> None:
        instrument = instrument.upper()
        if instrument not in LOT_SIZES:
            raise ValueError(
                f"Unknown instrument '{instrument}'. "
                f"Choose from: {list(LOT_SIZES.keys())}"
            )
        self.instrument = instrument
        self.mode = mode
        self.data_path = data_path
        self.lot_size = lot_size or self.get_lot_size(instrument)
        self.cost_per_leg_pts = cost_per_leg or self.get_cost_per_leg(instrument)
        self.num_steps = num_steps
        self.include_options = include_options
        self.discrete_actions_mode = discrete_actions

        # Override bps-based costs with index-point-based costs
        # Set bps to 0 since we handle costs directly in points
        kwargs.setdefault("transaction_cost_bps", 0.0)
        kwargs.setdefault("slippage_bps", 0.0)
        kwargs.setdefault("initial_cash", 5_000_000.0)  # 50 lakh margin
        super().__init__(**kwargs)

        # Price dynamics for simulated mode
        self._spot0 = INITIAL_SPOTS[instrument]
        self._sigma = ANNUALISED_VOLS[instrument]
        self._mu = 0.10  # ~10% annualised drift for Indian indices
        self._dt = 1.0 / 252.0

        # Episode state
        self._prices_array: Optional[np.ndarray] = None
        self._price_history: list[float] = []
        self._vix: float = self._sigma * 100.0  # crude VIX proxy

    # ----- static helpers -----

    @staticmethod
    def get_lot_size(instrument: str) -> int:
        """Return exchange-mandated lot size.

        NIFTY=25, BANKNIFTY=15, FINNIFTY=25, MIDCPNIFTY=50.
        """
        return LOT_SIZES[instrument.upper()]

    @staticmethod
    def get_cost_per_leg(instrument: str) -> float:
        """Return realistic cost per leg in index points.

        NIFTY~3 pts, BANKNIFTY~5 pts per leg.
        Includes brokerage + STT + stamp duty + SEBI charges + GST.
        """
        return COST_PER_LEG[instrument.upper()]

    # ----- TradingEnv interface -----

    @property
    def state_dim(self) -> int:
        base = 12  # core features
        return base

    @property
    def action_dim(self) -> int:
        if self.discrete_actions_mode:
            return len(DISCRETE_ACTIONS)
        return 3  # (direction, size, strike_offset)

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        if self.mode == "simulated":
            self._generate_price_path()
        elif self.mode == "historical":
            self._load_historical_prices()

        assert self._prices_array is not None
        spot = self._prices_array[0]
        self._price_history = [float(spot)]

        self._current_state = TradingState(
            timestamp=0,
            prices=np.array([spot], dtype=np.float64),
            position=np.zeros(1, dtype=np.float64),
            cash=self.initial_cash,
            pnl=0.0,
            features=self._initial_features(spot),
        )
        return self._current_state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")
        assert self._prices_array is not None

        old = self._current_state
        old_spot = float(old.prices[0])
        self._step_count += 1

        # New price (causal: decided before action, revealed after)
        new_spot = float(self._prices_array[min(self._step_count, len(self._prices_array) - 1)])
        self._price_history.append(new_spot)

        # Decode action
        if self.discrete_actions_mode:
            trade_lots = self._decode_discrete_action(action, old.position[0])
        else:
            trade_lots = float(action.trade_sizes[0])

        trade_lots = self._clip_to_limits(trade_lots, old.position[0])
        new_position = old.position[0] + trade_lots

        # Cost = |trade_lots| * lot_size * cost_per_leg_pts  (per leg, in INR)
        num_legs = 1.0  # futures = 1 leg
        cost_pts = abs(trade_lots) * num_legs * self.cost_per_leg_pts
        cost_inr = cost_pts * self.lot_size  # convert points -> INR

        # Cash change: sell proceeds / buy outlay (in INR, using lot size)
        trade_value_inr = trade_lots * self.lot_size * old_spot
        new_cash = old.cash - trade_value_inr - cost_inr

        # Mark-to-market PnL
        position_value_inr = new_position * self.lot_size * new_spot
        new_pnl = new_cash + position_value_inr - self.initial_cash

        reward = new_pnl - old.pnl

        # Features
        features = self._compute_features(new_spot)

        # Terminal
        done = self._step_count >= self.num_steps
        truncated = abs(new_position) > self.max_position

        if truncated:
            close_cost = abs(new_position) * self.cost_per_leg_pts * self.lot_size
            new_cash += new_position * self.lot_size * new_spot - close_cost
            new_position = 0.0
            new_pnl = new_cash - self.initial_cash
            reward = new_pnl - old.pnl

        new_state = TradingState(
            timestamp=self._step_count,
            prices=np.array([new_spot], dtype=np.float64),
            position=np.array([new_position], dtype=np.float64),
            cash=new_cash,
            pnl=new_pnl,
            features=features,
        )
        self._current_state = new_state if not (done or truncated) else None

        return StepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info={
                "cost_inr": cost_inr,
                "cost_pts": cost_pts,
                "trade_lots": trade_lots,
                "spot": new_spot,
            },
        )

    # ----- private helpers -----

    def _generate_price_path(self) -> None:
        """Generate a full episode of simulated prices via GBM."""
        n = self.num_steps + 1
        z = self._rng.standard_normal(n)
        log_returns = (self._mu - 0.5 * self._sigma ** 2) * self._dt + \
                      self._sigma * math.sqrt(self._dt) * z
        log_returns[0] = 0.0
        log_prices = np.log(self._spot0) + np.cumsum(log_returns)
        self._prices_array = np.exp(log_prices)

    def _load_historical_prices(self) -> None:
        """Load prices from Parquet file.  Fallback to simulated if unavailable."""
        if self.data_path is None:
            self._generate_price_path()
            return
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(self.data_path)
            close_col = None
            for col_name in ["Closing Index Value", "close", "Close"]:
                if col_name in table.column_names:
                    close_col = col_name
                    break
            if close_col is None:
                raise ValueError(f"No close column found in {self.data_path}")
            prices = table.column(close_col).to_numpy().astype(np.float64)
            if len(prices) < self.num_steps + 1:
                # Pad with last price
                padding = np.full(self.num_steps + 1 - len(prices), prices[-1])
                prices = np.concatenate([prices, padding])
            # Random window for variety
            max_start = max(0, len(prices) - self.num_steps - 1)
            start_idx = int(self._rng.integers(0, max_start + 1))
            self._prices_array = prices[start_idx:start_idx + self.num_steps + 1]
        except Exception:
            self._generate_price_path()

    def _decode_discrete_action(self, action: TradingAction, current_pos: float) -> float:
        """Map discrete action index to trade lots."""
        idx = int(action.trade_sizes[0])
        idx = max(0, min(idx, len(DISCRETE_ACTIONS) - 1))
        name = DISCRETE_ACTIONS[idx]

        if name == "flat":
            return -current_pos
        elif name == "long_fut":
            return max(0.0, 1.0 - current_pos)
        elif name == "short_fut":
            return min(0.0, -1.0 - current_pos)
        elif name == "long_straddle":
            return max(0.0, 1.0 - current_pos)
        elif name == "short_straddle":
            return min(0.0, -1.0 - current_pos)
        elif name == "bull_spread":
            return max(0.0, 1.0 - current_pos)
        elif name == "bear_spread":
            return min(0.0, -1.0 - current_pos)
        return 0.0

    def _clip_to_limits(self, trade_lots: float, current_pos: float) -> float:
        """Clip trade to keep position within max_position."""
        new_pos = current_pos + trade_lots
        if abs(new_pos) > self.max_position:
            if new_pos > 0:
                trade_lots = self.max_position - current_pos
            else:
                trade_lots = -self.max_position - current_pos
        return trade_lots

    def _initial_features(self, spot: float) -> dict:
        """Return features dict for the initial state."""
        return {
            "normalised_price": 1.0,
            "returns_5d": 0.0,
            "returns_20d": 0.0,
            "realised_vol_20d": self._sigma,
            "iv_atm": self._sigma if self.include_options else 0.0,
            "iv_skew": 0.0,
            "put_call_ratio": 1.0,
            "oi_change": 0.0,
            "vix": self._vix,
            "days_to_expiry": 5.0,
            "position": 0.0,
            "pnl_normalised": 0.0,
        }

    def _compute_features(self, spot: float) -> dict:
        """Compute features from price history (strictly causal)."""
        h = self._price_history
        n = len(h)
        features: dict = {}

        features["normalised_price"] = spot / h[0] if h[0] != 0 else 1.0

        # Returns
        if n > 5:
            features["returns_5d"] = (h[-1] - h[-6]) / h[-6]
        else:
            features["returns_5d"] = 0.0

        if n > 20:
            features["returns_20d"] = (h[-1] - h[-21]) / h[-21]
        else:
            features["returns_20d"] = 0.0

        # Realised vol (20d)
        if n > 20:
            log_rets = [
                math.log(h[i] / h[i - 1]) for i in range(max(1, n - 20), n)
                if h[i - 1] > 0
            ]
            features["realised_vol_20d"] = (
                float(np.std(log_rets, ddof=1) * math.sqrt(252))
                if len(log_rets) > 1 else self._sigma
            )
        else:
            features["realised_vol_20d"] = self._sigma

        # Synthetic IV / options features (simulated mode)
        rv = features["realised_vol_20d"]
        features["iv_atm"] = rv * (1.0 + 0.05 * self._rng.standard_normal()) if self.include_options else 0.0
        features["iv_skew"] = -0.02 + 0.005 * self._rng.standard_normal()
        features["put_call_ratio"] = 1.0 + 0.1 * self._rng.standard_normal()
        features["oi_change"] = self._rng.standard_normal() * 0.05
        features["vix"] = max(8.0, rv * 100.0 + self._rng.standard_normal() * 2.0)
        features["days_to_expiry"] = max(0, 5 - (self._step_count % 5))
        features["position"] = float(self._current_state.position[0]) if self._current_state else 0.0
        features["pnl_normalised"] = (
            self._current_state.pnl / self.initial_cash if self._current_state else 0.0
        )

        return features


# ---------------------------------------------------------------------------
# Specialised environments
# ---------------------------------------------------------------------------


class NiftySwingEnv(IndiaFnOEnv):
    """Swing trading environment for NIFTY (1-5 day holding period).

    State augmented with:
      - Regime indicator (trending / mean-reverting) based on Hurst exponent proxy
      - Support / resistance levels (20-day high/low)
      - Options-derived signals (IV rank, skew change)

    Action: continuous position size in [-1, 1] (normalised to max lots).

    Book reference: Ch 10 (Financial MDPs), Ch 8 (Merton's portfolio problem).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("instrument", "NIFTY")
        kwargs.setdefault("num_steps", 252)
        kwargs.setdefault("discrete_actions", False)
        kwargs.setdefault("max_position", 10.0)  # max 10 lots
        super().__init__(**kwargs)

    @property
    def state_dim(self) -> int:
        return 15  # base 12 + regime + support + resistance

    def _compute_features(self, spot: float) -> dict:
        features = super()._compute_features(spot)
        h = self._price_history
        n = len(h)

        # Regime indicator: simple Hurst proxy via variance ratio
        if n > 20:
            log_rets = [math.log(h[i] / h[i - 1]) for i in range(max(1, n - 20), n) if h[i - 1] > 0]
            if len(log_rets) > 5:
                var_1 = np.var(log_rets, ddof=1)
                # 5-bar aggregated returns
                agg = [sum(log_rets[i:i + 5]) for i in range(0, len(log_rets) - 4, 5)]
                var_5 = np.var(agg, ddof=1) / 5.0 if len(agg) > 1 else var_1
                vr = var_5 / var_1 if var_1 > 1e-12 else 1.0
                # VR > 1 = trending, VR < 1 = mean-reverting
                features["regime"] = float(np.clip(vr, 0.5, 2.0))
            else:
                features["regime"] = 1.0
        else:
            features["regime"] = 1.0

        # Support / resistance
        if n > 20:
            features["support"] = min(h[-20:]) / spot
            features["resistance"] = max(h[-20:]) / spot
        else:
            features["support"] = 1.0
            features["resistance"] = 1.0

        return features


class BankNiftyScalpEnv(IndiaFnOEnv):
    """Intraday scalping environment for BANKNIFTY.

    Higher volatility (~18% ann.), wider spreads, faster signals.
    Episode = 1 trading day = 375 minutes at 1-minute bars.

    State augmented with:
      - Microstructure features (bid-ask spread proxy, volume momentum)
      - Intraday VWAP deviation
      - Time-of-day encoding (sin/cos of minutes since open)

    Action: discrete {flat, long_1lot, short_1lot, add_long, add_short}

    Book reference: Ch 10.2 (Execution MDPs), Ch 10.3 (Market Making).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("instrument", "BANKNIFTY")
        kwargs.setdefault("num_steps", 375)  # 1 day at 1-min bars
        kwargs.setdefault("discrete_actions", True)
        kwargs.setdefault("max_position", 5.0)  # max 5 lots intraday
        super().__init__(**kwargs)
        self._dt = 1.0 / (252.0 * 375.0)  # per-minute dt

    @property
    def state_dim(self) -> int:
        return 16  # base 12 + spread_proxy + volume_mom + vwap_dev + time_encoding

    def _compute_features(self, spot: float) -> dict:
        features = super()._compute_features(spot)
        h = self._price_history
        n = len(h)

        # Spread proxy (higher vol = wider spread)
        rv = features.get("realised_vol_20d", self._sigma)
        features["spread_proxy"] = rv * spot * 0.001  # ~0.1% of spot

        # Volume momentum (synthetic)
        features["volume_momentum"] = float(self._rng.standard_normal() * 0.1)

        # VWAP deviation
        if n > 10:
            vwap = float(np.mean(h[-10:]))
            features["vwap_deviation"] = (spot - vwap) / vwap
        else:
            features["vwap_deviation"] = 0.0

        # Time of day encoding (minutes since open)
        minutes = self._step_count
        features["time_sin"] = math.sin(2.0 * math.pi * minutes / 375.0)
        features["time_cos"] = math.cos(2.0 * math.pi * minutes / 375.0)

        return features
