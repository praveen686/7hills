"""Cryptocurrency trading environment.

Targets: BTC, ETH, SOL on Binance perpetual futures.

Key crypto characteristics modelled here:
  - 24/7 trading = 5.6x more episodes than India FnO per calendar year
  - Funding rate every 8 hours (carry trade opportunity)
  - Higher leverage available (up to 20x, capped by exchange)
  - SBE binary feed (~270ns via Rust) -- latency not simulated here
  - More liquid LOB = tighter spreads, better for market-making
  - Maker/taker fee asymmetry

Environments:
  - CryptoEnv: general Binance crypto trading MDP (spot + perps)
  - CryptoFundingEnv: funding rate carry / cash-and-carry arbitrage
  - CryptoLeadLagEnv: cross-market lead-lag (BTC/ETH → NIFTY)

Book reference: Ch 3 (MDP), Ch 10 (Financial MDPs).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .trading_env import TradingEnv, TradingState, TradingAction, StepResult

__all__ = [
    "CryptoEnv",
    "CryptoFundingEnv",
    "CryptoLeadLagEnv",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRYPTO_DEFAULTS: Dict[str, dict] = {
    "BTCUSDT": {
        "price": 65_000.0,
        "sigma": 0.55,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "funding_rate": 0.0001,  # 1 bp per 8h
    },
    "ETHUSDT": {
        "price": 3_200.0,
        "sigma": 0.65,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "funding_rate": 0.00008,
    },
    "SOLUSDT": {
        "price": 140.0,
        "sigma": 0.80,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "funding_rate": 0.00012,
    },
}


# ---------------------------------------------------------------------------
# CryptoEnv
# ---------------------------------------------------------------------------


class CryptoEnv(TradingEnv):
    """Binance crypto trading environment (spot + perpetual futures).

    State vector (dimension 11):
        [price, returns, volatility, funding_rate, open_interest,
         volume_24h, spread, depth_imbalance, btc_dominance,
         position, unrealised_pnl]

    Action: continuous position change in [-max_trade, +max_trade].

    Supports both spot and perps.  Perps accrue funding every 8h
    (every ``funding_interval`` steps).

    Parameters
    ----------
    symbol : str
        ``"BTCUSDT"`` | ``"ETHUSDT"`` | ``"SOLUSDT"``
    mode : str
        ``"simulated"`` or ``"historical"``
    data_path : str or None
        Path to historical OHLCV data.
    leverage : float
        Initial leverage (1.0 = no leverage).
    max_leverage : float
        Maximum allowed leverage.
    funding_rate : float
        Per-8h funding rate (positive = longs pay shorts).
    maker_fee, taker_fee : float
        Fee as fraction of notional.
    num_steps : int
        Steps per episode (1440 = 1 day at 1-min bars).
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        mode: str = "simulated",
        data_path: Optional[str] = None,
        leverage: float = 1.0,
        max_leverage: float = 10.0,
        funding_rate: Optional[float] = None,
        maker_fee: Optional[float] = None,
        taker_fee: Optional[float] = None,
        num_steps: int = 1440,
        funding_interval: int = 480,  # steps between funding (8h = 480 min)
        **kwargs: Any,
    ) -> None:
        symbol = symbol.upper()
        defaults = CRYPTO_DEFAULTS.get(symbol, CRYPTO_DEFAULTS["BTCUSDT"])

        self.symbol = symbol
        self.mode = mode
        self.data_path = data_path
        self.leverage = leverage
        self.max_leverage = max_leverage
        self.funding_rate = funding_rate or defaults["funding_rate"]
        self.maker_fee = maker_fee or defaults["maker_fee"]
        self.taker_fee = taker_fee or defaults["taker_fee"]
        self.num_steps = num_steps
        self.funding_interval = funding_interval

        self._price0 = defaults["price"]
        self._sigma = defaults["sigma"]
        self._mu = 0.0  # crypto: assume zero drift for risk-neutral

        # Override base class costs to 0 (we handle fees directly)
        kwargs.setdefault("transaction_cost_bps", 0.0)
        kwargs.setdefault("slippage_bps", 0.0)
        kwargs.setdefault("initial_cash", 100_000.0)  # 100k USDT
        super().__init__(**kwargs)

        # Per-step dt (1-min bars, 24/7 = 525600 min/year)
        self._dt = 1.0 / 525_600.0

        # Internal state
        self._prices_array: Optional[np.ndarray] = None
        self._price_history: list[float] = []
        self._cumulative_funding: float = 0.0

    # ----- TradingEnv interface -----

    @property
    def state_dim(self) -> int:
        return 11

    @property
    def action_dim(self) -> int:
        return 1  # signed position change

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._cumulative_funding = 0.0

        self._generate_price_path()
        assert self._prices_array is not None
        p0 = float(self._prices_array[0])
        self._price_history = [p0]

        self._current_state = TradingState(
            timestamp=0,
            prices=np.array([p0], dtype=np.float64),
            position=np.zeros(1, dtype=np.float64),
            cash=self.initial_cash,
            pnl=0.0,
            features=self._initial_features(),
        )
        return self._current_state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")
        assert self._prices_array is not None

        old = self._current_state
        old_price = float(old.prices[0])
        self._step_count += 1

        new_price = float(
            self._prices_array[min(self._step_count, len(self._prices_array) - 1)]
        )
        self._price_history.append(new_price)

        # Trade execution
        trade_size = float(action.trade_sizes[0])

        # Check leverage constraint
        new_position = old.position[0] + trade_size
        notional = abs(new_position) * new_price
        if notional > self.max_leverage * old.cash:
            # Scale down to max leverage
            max_pos = self.max_leverage * old.cash / new_price
            if new_position > 0:
                trade_size = max_pos - old.position[0]
            else:
                trade_size = -max_pos - old.position[0]
            new_position = old.position[0] + trade_size

        # Fee: use taker_fee for market orders
        trade_notional = abs(trade_size) * old_price
        fee = trade_notional * self.taker_fee

        # Cash change
        cash_delta = -trade_size * old_price - fee
        new_cash = old.cash + cash_delta

        # Funding payment (perps)
        funding_payment = 0.0
        if (self._step_count % self.funding_interval) == 0 and self._step_count > 0:
            # Longs pay shorts when funding > 0
            funding_payment = new_position * new_price * self.funding_rate
            new_cash -= funding_payment
            self._cumulative_funding += funding_payment

        # Mark-to-market
        position_value = new_position * new_price
        new_pnl = new_cash + position_value - self.initial_cash
        reward = new_pnl - old.pnl

        features = self._compute_features(new_price)

        done = self._step_count >= self.num_steps
        truncated = False

        # Liquidation check: if equity < 5% of notional
        equity = new_cash + new_position * new_price
        if equity < 0.05 * abs(new_position * new_price) and abs(new_position) > 1e-12:
            truncated = True
            # Force liquidation
            liq_fee = abs(new_position) * new_price * self.taker_fee
            new_cash += new_position * new_price - liq_fee
            new_position = 0.0
            new_pnl = new_cash - self.initial_cash
            reward = new_pnl - old.pnl

        new_state = TradingState(
            timestamp=self._step_count,
            prices=np.array([new_price], dtype=np.float64),
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
                "fee": fee,
                "funding_payment": funding_payment,
                "cumulative_funding": self._cumulative_funding,
                "leverage_used": abs(new_position * new_price) / max(new_cash, 1e-6),
            },
        )

    # ----- internal -----

    def _generate_price_path(self) -> None:
        """Generate GBM price path."""
        n = self.num_steps + 1
        z = self._rng.standard_normal(n)
        log_rets = (self._mu - 0.5 * self._sigma ** 2) * self._dt + \
                   self._sigma * math.sqrt(self._dt) * z
        log_rets[0] = 0.0
        log_prices = math.log(self._price0) + np.cumsum(log_rets)
        self._prices_array = np.exp(log_prices)

    def _initial_features(self) -> dict:
        return {
            "returns": 0.0,
            "volatility": self._sigma,
            "funding_rate": self.funding_rate,
            "open_interest": 1.0,
            "volume_24h": 1.0,
            "spread": self._price0 * 0.0001,
            "depth_imbalance": 0.0,
            "btc_dominance": 0.5,
            "unrealised_pnl": 0.0,
        }

    def _compute_features(self, price: float) -> dict:
        h = self._price_history
        n = len(h)
        features: dict = {}

        # Returns (1-step)
        features["returns"] = (h[-1] - h[-2]) / h[-2] if n > 1 and h[-2] != 0 else 0.0

        # Rolling vol (60-step ~ 1 hour)
        if n > 60:
            log_rets = [math.log(h[i] / h[i - 1]) for i in range(max(1, n - 60), n) if h[i - 1] > 0]
            features["volatility"] = (
                float(np.std(log_rets, ddof=1) * math.sqrt(525_600))
                if len(log_rets) > 1 else self._sigma
            )
        else:
            features["volatility"] = self._sigma

        features["funding_rate"] = self.funding_rate
        features["open_interest"] = 1.0 + 0.01 * self._rng.standard_normal()
        features["volume_24h"] = max(0.0, 1.0 + 0.1 * self._rng.standard_normal())
        features["spread"] = price * max(0.00005, 0.0001 + 0.00002 * self._rng.standard_normal())
        features["depth_imbalance"] = float(np.clip(self._rng.standard_normal() * 0.3, -1, 1))
        features["btc_dominance"] = 0.50 + 0.02 * self._rng.standard_normal()
        features["unrealised_pnl"] = (
            float(self._current_state.position[0] * (price - self._current_state.prices[0]))
            if self._current_state is not None else 0.0
        )

        return features


# ---------------------------------------------------------------------------
# CryptoFundingEnv
# ---------------------------------------------------------------------------


class CryptoFundingEnv(CryptoEnv):
    """Funding rate carry trade environment.

    Strategy: earn funding by positioning in the direction of the funding
    rate (short when funding > 0 to receive payment; long when funding < 0).

    Cash-and-carry arbitrage: long spot + short perp collects positive funding.

    State augmented with:
      - funding_rate (current)
      - predicted_funding (EMA forecast)
      - basis (perp - spot spread)
      - inventory
      - time_to_next_funding

    Book reference: Ch 7 (Utility Theory) -- risk-adjusted carry.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("num_steps", 4320)  # 3 days at 1-min
        kwargs.setdefault("leverage", 2.0)
        super().__init__(**kwargs)
        self._funding_history: list[float] = []
        self._basis: float = 0.0

    @property
    def state_dim(self) -> int:
        return 14  # base 11 + predicted_funding + basis + time_to_next_funding

    def reset(self, seed: Optional[int] = None) -> TradingState:
        self._funding_history = [self.funding_rate]
        self._basis = 0.0
        return super().reset(seed)

    def _compute_features(self, price: float) -> dict:
        features = super()._compute_features(price)

        # Simulated time-varying funding rate
        noise = self._rng.standard_normal() * 0.00003
        current_fr = self.funding_rate + noise
        self._funding_history.append(current_fr)

        # EMA-predicted funding
        alpha = 0.1
        if len(self._funding_history) > 1:
            ema = self._funding_history[-1]
            for fr in reversed(self._funding_history[-20:]):
                ema = alpha * fr + (1 - alpha) * ema
            features["predicted_funding"] = ema
        else:
            features["predicted_funding"] = current_fr

        features["funding_rate"] = current_fr

        # Basis (perp premium over synthetic spot)
        self._basis = current_fr * 3.0 * 365.0  # annualised
        features["basis"] = self._basis

        # Time to next funding (in steps)
        features["time_to_next_funding"] = float(
            self.funding_interval - (self._step_count % self.funding_interval)
        )

        return features


# ---------------------------------------------------------------------------
# CryptoLeadLagEnv
# ---------------------------------------------------------------------------


class CryptoLeadLagEnv(TradingEnv):
    """Cross-market lead-lag environment.

    Exploits the empirical observation that BTC/ETH overnight returns have
    predictive power for NIFTY intraday direction (via global risk sentiment).

    State:
        [btc_overnight_return, eth_overnight_return, nifty_gap,
         vix, correlation_regime, position, pnl]

    Action: discrete {long_nifty, flat, short_nifty} based on crypto signal.

    The episode spans one India trading day.  The agent observes crypto
    returns from the preceding night session (UTC 15:30 to 03:45 IST)
    and decides a NIFTY position for the day.

    Book reference: Ch 15 (Multi-Armed Bandits) -- contextual decision.
    """

    def __init__(
        self,
        num_episodes: int = 252,
        btc_sigma: float = 0.55,
        eth_sigma: float = 0.65,
        nifty_sigma: float = 0.14,
        lead_lag_beta: float = 0.10,  # BTC return → NIFTY return coefficient
        nifty_lot_size: int = 25,
        cost_per_leg: float = 3.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("initial_cash", 5_000_000.0)
        kwargs.setdefault("transaction_cost_bps", 0.0)
        kwargs.setdefault("slippage_bps", 0.0)
        super().__init__(**kwargs)

        self.num_episodes_total = num_episodes
        self.btc_sigma = btc_sigma
        self.eth_sigma = eth_sigma
        self.nifty_sigma = nifty_sigma
        self.lead_lag_beta = lead_lag_beta
        self.nifty_lot_size = nifty_lot_size
        self.cost_per_leg = cost_per_leg

        # Each episode is 1 day (single-step)
        self.num_steps = 1
        self._episode_idx = 0
        self._nifty_spot = 24_000.0
        self._dt = 1.0 / 252.0

    @property
    def state_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 3  # long, flat, short

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_idx += 1

        # Generate overnight crypto returns
        btc_overnight = self._rng.normal(0, self.btc_sigma * math.sqrt(self._dt))
        eth_overnight = self._rng.normal(0, self.eth_sigma * math.sqrt(self._dt))

        # NIFTY gap influenced by crypto
        nifty_gap = (
            self.lead_lag_beta * btc_overnight
            + self._rng.normal(0, self.nifty_sigma * math.sqrt(self._dt) * 0.3)
        )

        # Correlation regime (rolling 20d correlation between BTC & NIFTY)
        corr_regime = float(np.clip(0.15 + self._rng.standard_normal() * 0.1, -0.5, 0.8))

        vix = max(8.0, 14.0 + self._rng.standard_normal() * 3.0)

        # Store for use in step()
        self._btc_return = btc_overnight
        self._eth_return = eth_overnight
        self._nifty_gap = nifty_gap

        self._current_state = TradingState(
            timestamp=0,
            prices=np.array([self._nifty_spot], dtype=np.float64),
            position=np.zeros(1, dtype=np.float64),
            cash=self.initial_cash,
            pnl=0.0,
            features={
                "btc_overnight_return": btc_overnight,
                "eth_overnight_return": eth_overnight,
                "nifty_gap": nifty_gap,
                "vix": vix,
                "correlation_regime": corr_regime,
            },
        )
        return self._current_state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")

        old = self._current_state
        self._step_count += 1

        # Decode action: 0=long, 1=flat, 2=short
        act_idx = int(action.trade_sizes[0])
        if act_idx == 0:
            position_lots = 1.0
        elif act_idx == 2:
            position_lots = -1.0
        else:
            position_lots = 0.0

        # NIFTY intraday return = gap + idiosyncratic
        nifty_intraday = (
            self._nifty_gap
            + self._rng.normal(0, self.nifty_sigma * math.sqrt(self._dt) * 0.7)
        )
        new_nifty = self._nifty_spot * (1.0 + nifty_intraday)

        # PnL = lots * lot_size * point_change - cost
        point_change = new_nifty - self._nifty_spot
        pnl_gross = position_lots * self.nifty_lot_size * point_change
        cost = abs(position_lots) * self.cost_per_leg * self.nifty_lot_size * 2.0  # entry + exit
        pnl_net = pnl_gross - cost

        reward = pnl_net

        new_state = TradingState(
            timestamp=self._step_count,
            prices=np.array([new_nifty], dtype=np.float64),
            position=np.array([0.0], dtype=np.float64),  # day trade = flat at close
            cash=self.initial_cash + pnl_net,
            pnl=pnl_net,
            features=old.features,
        )
        self._current_state = None  # single-step episode

        return StepResult(
            state=new_state,
            reward=reward,
            done=True,
            truncated=False,
            info={
                "nifty_return": nifty_intraday,
                "btc_return": self._btc_return,
                "position_lots": position_lots,
                "cost": cost,
                "pnl_gross": pnl_gross,
            },
        )
