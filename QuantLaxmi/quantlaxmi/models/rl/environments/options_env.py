"""Options-specific trading environment.

For Deep Hedging, Gamma Scalping, Theta Strategies, IV Mean Reversion.

Environments:
  - OptionsEnv: base options MDP with Greeks in the state
  - GammaScalpEnv: long options, dynamically hedge delta (S10 strategy)
  - ThetaDecayEnv: short iron condors, manage risk (S8 strategy)
  - IVMeanReversionEnv: trade options on IV deviations (S4 strategy)

Key modelling choices:
  - Greeks via Black-Scholes (European; adequate for index options)
  - Cost in index points per leg, NOT bps of spot
  - Discrete hedging intervals (5-min, hourly, daily)
  - Simulated IV dynamics: OU around realised vol

Book reference: Ch 9 (Derivatives Pricing & Hedging), Ch 10 (Financial MDPs).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .trading_env import TradingEnv, TradingState, TradingAction, StepResult

__all__ = [
    "OptionsEnv",
    "GammaScalpEnv",
    "ThetaDecayEnv",
    "IVMeanReversionEnv",
]

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _bs_d1(S: float, K: float, tau: float, sigma: float, r: float) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))


def _bs_d2(d1: float, sigma: float, tau: float) -> float:
    if tau <= 0 or sigma <= 0:
        return 0.0
    return d1 - sigma * math.sqrt(tau)


def bs_call_price(S: float, K: float, tau: float, sigma: float, r: float) -> float:
    """Black-Scholes European call price."""
    if tau <= 1e-12:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, tau, sigma, r)
    d2 = _bs_d2(d1, sigma, tau)
    return S * _norm_cdf(d1) - K * math.exp(-r * tau) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, tau: float, sigma: float, r: float) -> float:
    """Black-Scholes European put price."""
    if tau <= 1e-12:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, tau, sigma, r)
    d2 = _bs_d2(d1, sigma, tau)
    return K * math.exp(-r * tau) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def compute_greeks(
    S: float, K: float, tau: float, sigma: float, r: float
) -> dict:
    """Compute all Greeks using Black-Scholes.

    Returns dict with keys: delta_call, delta_put, gamma, theta_call,
    theta_put, vega, rho_call, rho_put.
    """
    if tau <= 1e-12 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta_call": 1.0 if S > K else 0.0,
            "delta_put": -1.0 if S < K else 0.0,
            "gamma": 0.0,
            "theta_call": 0.0,
            "theta_put": 0.0,
            "vega": 0.0,
            "rho_call": 0.0,
            "rho_put": 0.0,
        }

    d1 = _bs_d1(S, K, tau, sigma, r)
    d2 = _bs_d2(d1, sigma, tau)
    sqrt_tau = math.sqrt(tau)

    delta_call = _norm_cdf(d1)
    delta_put = delta_call - 1.0

    gamma = _norm_pdf(d1) / (S * sigma * sqrt_tau)

    theta_call = (
        -S * _norm_pdf(d1) * sigma / (2.0 * sqrt_tau)
        - r * K * math.exp(-r * tau) * _norm_cdf(d2)
    ) / 365.0  # per calendar day

    theta_put = (
        -S * _norm_pdf(d1) * sigma / (2.0 * sqrt_tau)
        + r * K * math.exp(-r * tau) * _norm_cdf(-d2)
    ) / 365.0

    vega = S * _norm_pdf(d1) * sqrt_tau / 100.0  # per 1% vol move

    rho_call = K * tau * math.exp(-r * tau) * _norm_cdf(d2) / 100.0
    rho_put = -K * tau * math.exp(-r * tau) * _norm_cdf(-d2) / 100.0

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "vega": vega,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


# ---------------------------------------------------------------------------
# OptionsEnv
# ---------------------------------------------------------------------------


class OptionsEnv(TradingEnv):
    """Options trading environment with Greeks in the state.

    State vector (dimension 14):
        [spot, strike, time_to_expiry, iv, delta, gamma, theta, vega,
         realised_vol, position_delta, position_gamma, position_theta,
         position_vega, pnl]

    Action (2-dim): (underlying_hedge_delta, option_trade)
        - underlying_hedge_delta: shares of underlying to trade
        - option_trade: option contracts to trade (signed)

    The environment tracks a single option (call or put) and the
    underlying.  Multi-leg strategies compose multiple OptionsEnv
    instances or use the specialised sub-environments.

    Parameters
    ----------
    spot_init : float
        Initial spot price.
    strikes : np.ndarray or None
        Available strikes.  None = ATM only.
    expiry_days : int
        Days to expiry at episode start.
    sigma : float
        Annualised implied volatility.
    risk_free_rate : float
        Continuous risk-free rate.
    num_steps_per_day : int
        Hedging frequency (1=daily, 78=5-min for NIFTY, 375=1-min).
    """

    def __init__(
        self,
        spot_init: float = 100.0,
        strikes: Optional[np.ndarray] = None,
        expiry_days: int = 30,
        sigma: float = 0.20,
        risk_free_rate: float = 0.05,
        num_steps_per_day: int = 1,
        option_type: str = "call",  # "call" or "put"
        cost_per_trade: float = 0.5,  # cost in price units per option
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("initial_cash", 1_000_000.0)
        kwargs.setdefault("transaction_cost_bps", 0.0)
        kwargs.setdefault("slippage_bps", 0.0)
        super().__init__(**kwargs)

        self.spot_init = spot_init
        self.strikes = strikes if strikes is not None else np.array([spot_init])
        self.expiry_days = expiry_days
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.num_steps_per_day = num_steps_per_day
        self.option_type = option_type
        self.cost_per_trade = cost_per_trade

        self.num_steps = expiry_days * num_steps_per_day
        self._dt = 1.0 / (252.0 * num_steps_per_day)

        # State
        self._spot: float = spot_init
        self._strike: float = float(self.strikes[0])
        self._tau: float = expiry_days / 365.0
        self._iv: float = sigma
        self._option_position: float = 0.0  # number of option contracts
        self._underlying_position: float = 0.0
        self._spot_history: list[float] = []
        self._rv: float = sigma

    @property
    def state_dim(self) -> int:
        return 14

    @property
    def action_dim(self) -> int:
        return 2  # (underlying_hedge, option_trade)

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self._spot = self.spot_init
        self._strike = float(self.strikes[0])
        self._tau = self.expiry_days / 365.0
        self._iv = self.sigma
        self._option_position = 0.0
        self._underlying_position = 0.0
        self._spot_history = [self._spot]
        self._rv = self.sigma

        state = self._build_state(0.0)
        self._current_state = state
        return state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")

        old = self._current_state
        self._step_count += 1

        # 1) Execute trades (causal: before price move)
        hedge_delta = float(action.trade_sizes[0])
        option_trade = float(action.trade_sizes[1]) if len(action.trade_sizes) > 1 else 0.0

        # Option trade cost
        option_cost = abs(option_trade) * self.cost_per_trade
        # Underlying hedge cost (small slippage)
        underlying_cost = abs(hedge_delta) * self._spot * 0.0001

        # Option premium paid/received
        if self.option_type == "call":
            opt_price = bs_call_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        else:
            opt_price = bs_put_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)

        premium_flow = -option_trade * opt_price  # buy: pay; sell: receive

        self._option_position += option_trade
        self._underlying_position += hedge_delta

        cash_delta = (
            premium_flow
            - hedge_delta * self._spot
            - option_cost
            - underlying_cost
        )
        new_cash = old.cash + cash_delta

        # 2) Advance spot price (GBM)
        z = self._rng.standard_normal()
        self._spot = self._spot * math.exp(
            (self.risk_free_rate - 0.5 * self._rv ** 2) * self._dt
            + self._rv * math.sqrt(self._dt) * z
        )
        self._spot_history.append(self._spot)

        # 3) Advance time
        self._tau -= self._dt
        self._tau = max(self._tau, 0.0)

        # 4) Update IV (OU dynamics around realised vol)
        iv_speed = 5.0
        iv_noise = self._rng.standard_normal() * 0.01
        self._iv += iv_speed * (self._rv - self._iv) * self._dt + iv_noise
        self._iv = max(self._iv, 0.01)

        # 5) Update realised vol
        if len(self._spot_history) > 20:
            log_rets = [
                math.log(self._spot_history[i] / self._spot_history[i - 1])
                for i in range(max(1, len(self._spot_history) - 20), len(self._spot_history))
                if self._spot_history[i - 1] > 0
            ]
            if len(log_rets) > 1:
                self._rv = float(np.std(log_rets, ddof=1)) * math.sqrt(252.0 * self.num_steps_per_day)
                self._rv = max(self._rv, 0.01)

        # 6) Mark-to-market
        if self.option_type == "call":
            new_opt_price = bs_call_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        else:
            new_opt_price = bs_put_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)

        portfolio_value = (
            new_cash
            + self._underlying_position * self._spot
            + self._option_position * new_opt_price
        )
        new_pnl = portfolio_value - self.initial_cash
        reward = new_pnl - old.pnl

        # Terminal: expiry or time exhausted
        done = self._step_count >= self.num_steps or self._tau <= 1e-12
        truncated = False

        state = self._build_state(new_pnl, new_cash)
        self._current_state = state if not (done or truncated) else None

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            truncated=truncated,
            info={
                "spot": self._spot,
                "iv": self._iv,
                "rv": self._rv,
                "tau": self._tau,
                "option_price": new_opt_price,
                "option_cost": option_cost,
                "underlying_cost": underlying_cost,
            },
        )

    def _build_state(self, pnl: float, cash: Optional[float] = None) -> TradingState:
        greeks = compute_greeks(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)

        # Portfolio Greeks
        opt_pos = self._option_position
        und_pos = self._underlying_position
        pos_delta = opt_pos * greeks["delta_call" if self.option_type == "call" else "delta_put"] + und_pos
        pos_gamma = opt_pos * greeks["gamma"]
        pos_theta = opt_pos * greeks["theta_call" if self.option_type == "call" else "theta_put"]
        pos_vega = opt_pos * greeks["vega"]

        features = {
            "strike": self._strike,
            "time_to_expiry": self._tau,
            "iv": self._iv,
            "delta": greeks["delta_call" if self.option_type == "call" else "delta_put"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta_call" if self.option_type == "call" else "theta_put"],
            "vega": greeks["vega"],
            "realised_vol": self._rv,
            "position_delta": pos_delta,
            "position_gamma": pos_gamma,
            "position_theta": pos_theta,
            "position_vega": pos_vega,
        }

        return TradingState(
            timestamp=self._step_count,
            prices=np.array([self._spot], dtype=np.float64),
            position=np.array([self._option_position, self._underlying_position], dtype=np.float64),
            cash=cash if cash is not None else self.initial_cash,
            pnl=pnl,
            features=features,
        )


# ---------------------------------------------------------------------------
# GammaScalpEnv — long gamma, hedge delta
# ---------------------------------------------------------------------------


class GammaScalpEnv(OptionsEnv):
    """Gamma scalping environment (long options, hedge delta).

    Used by S10 Gamma Scalp strategy.

    The agent starts long a straddle (1 call + 1 put at ATM) and must
    dynamically hedge delta.  Profit comes from realised vol > implied vol.

    Action: scalar delta hedge (underlying shares to trade).

    Reward: daily PnL = gamma scalping profit - theta decay - costs.

    Book reference: Ch 9.2 (Deep Hedging).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("expiry_days", 30)
        kwargs.setdefault("num_steps_per_day", 78)  # 5-min intervals
        super().__init__(**kwargs)

    @property
    def action_dim(self) -> int:
        return 1  # just delta hedge

    def reset(self, seed: Optional[int] = None) -> TradingState:
        state = super().reset(seed)
        # Start with long straddle: +1 call, +1 put
        self._option_position = 1.0
        self._call_pos = 1.0
        self._put_pos = 1.0

        # Pay straddle premium
        call_price = bs_call_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        put_price = bs_put_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        straddle_cost = call_price + put_price + 2.0 * self.cost_per_trade

        state = self._build_state(-straddle_cost, self.initial_cash - straddle_cost)
        state.pnl = -straddle_cost
        self._current_state = state
        return state.copy()

    def step(self, action: TradingAction) -> StepResult:
        # Route through parent with option_trade=0
        full_action = TradingAction(
            trade_sizes=np.array([float(action.trade_sizes[0]), 0.0])
        )
        return super().step(full_action)


# ---------------------------------------------------------------------------
# ThetaDecayEnv — short options
# ---------------------------------------------------------------------------


class ThetaDecayEnv(OptionsEnv):
    """Theta decay environment (short options / iron condor).

    Used by S8 Expiry Theta strategy.

    The agent starts short an iron condor (sell OTM put + call, buy
    further OTM wings) and manages risk.  Profit from theta decay;
    risk from large spot moves or IV expansion.

    Action: binary {hold, close_position} or continuous hedge.

    Book reference: Ch 9 (Derivatives Pricing), market lesson from
    MEMORY.md: "S8 Expiry Theta is break-even after realistic costs."
    """

    def __init__(self, wing_width: float = 5.0, **kwargs: Any) -> None:
        kwargs.setdefault("expiry_days", 7)  # weekly expiry
        kwargs.setdefault("num_steps_per_day", 1)
        kwargs.setdefault("spot_init", 24_000.0)  # NIFTY
        kwargs.setdefault("cost_per_trade", 3.0)  # NIFTY cost per leg
        super().__init__(**kwargs)
        self.wing_width = wing_width  # percentage of spot
        self._condor_position: float = 0.0

    @property
    def action_dim(self) -> int:
        return 1  # hold (0) or close (1)

    def reset(self, seed: Optional[int] = None) -> TradingState:
        state = super().reset(seed)

        # Setup iron condor legs
        spot = self._spot
        self._call_strike_short = spot * (1.0 + self.wing_width / 100.0)
        self._put_strike_short = spot * (1.0 - self.wing_width / 100.0)
        self._call_strike_long = spot * (1.0 + 2.0 * self.wing_width / 100.0)
        self._put_strike_long = spot * (1.0 - 2.0 * self.wing_width / 100.0)

        # Net premium received
        tau = self._tau
        iv = self._iv
        r = self.risk_free_rate

        premium = (
            bs_call_price(spot, self._call_strike_short, tau, iv, r)
            - bs_call_price(spot, self._call_strike_long, tau, iv, r)
            + bs_put_price(spot, self._put_strike_short, tau, iv, r)
            - bs_put_price(spot, self._put_strike_long, tau, iv, r)
        )
        cost = 4.0 * self.cost_per_trade  # 4 legs
        net_premium = premium - cost

        self._condor_position = -1.0  # short 1 condor
        self._initial_premium = net_premium

        state = self._build_state(net_premium, self.initial_cash + net_premium)
        state.pnl = net_premium
        self._current_state = state
        return state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")

        old = self._current_state
        self._step_count += 1

        act = int(action.trade_sizes[0])  # 0 = hold, 1 = close

        # Advance spot
        z = self._rng.standard_normal()
        self._spot = self._spot * math.exp(
            (self.risk_free_rate - 0.5 * self._rv ** 2) * self._dt
            + self._rv * math.sqrt(self._dt) * z
        )
        self._spot_history.append(self._spot)
        self._tau -= self._dt
        self._tau = max(self._tau, 0.0)

        # Revalue condor
        s = self._spot
        tau = self._tau
        iv = self._iv
        r = self.risk_free_rate

        condor_value = (
            bs_call_price(s, self._call_strike_short, tau, iv, r)
            - bs_call_price(s, self._call_strike_long, tau, iv, r)
            + bs_put_price(s, self._put_strike_short, tau, iv, r)
            - bs_put_price(s, self._put_strike_long, tau, iv, r)
        )

        # We are short: liability = condor_value
        portfolio_value = self.initial_cash + self._initial_premium - condor_value
        new_pnl = portfolio_value - self.initial_cash

        done = self._step_count >= self.num_steps or self._tau <= 1e-12

        if act == 1 and self._condor_position != 0.0:
            # Close position: pay current condor value + costs
            close_cost = 4.0 * self.cost_per_trade
            new_pnl -= close_cost
            self._condor_position = 0.0
            done = True

        reward = new_pnl - old.pnl

        state = TradingState(
            timestamp=self._step_count,
            prices=np.array([self._spot], dtype=np.float64),
            position=np.array([self._condor_position], dtype=np.float64),
            cash=self.initial_cash + new_pnl,
            pnl=new_pnl,
            features={
                "time_to_expiry": self._tau,
                "iv": self._iv,
                "condor_value": condor_value,
                "premium_received": self._initial_premium,
                "spot_move_pct": (self._spot - self.spot_init) / self.spot_init,
            },
        )
        self._current_state = state if not done else None

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            truncated=False,
            info={"condor_value": condor_value, "spot": self._spot, "tau": self._tau},
        )


# ---------------------------------------------------------------------------
# IVMeanReversionEnv
# ---------------------------------------------------------------------------


class IVMeanReversionEnv(OptionsEnv):
    """IV mean reversion environment.

    Used by S4 IV MR strategy.

    The agent trades options when IV deviates significantly from
    historical realised vol.  Buy vol (long straddle) when IV is low;
    sell vol (short straddle) when IV is high.

    State: (spot, iv, rv, iv_rank, iv_zscore, position, pnl)
    Action: continuous in [-1, 1] where -1 = max short vol, +1 = max long vol.

    Book reference: Ch 9 (Pricing), market lesson: "S4 IV MR, Sharpe 3.07."
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("expiry_days", 30)
        kwargs.setdefault("num_steps_per_day", 1)
        super().__init__(**kwargs)
        self._iv_history: list[float] = []

    @property
    def state_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 1  # continuous [-1, 1]

    def reset(self, seed: Optional[int] = None) -> TradingState:
        state = super().reset(seed)
        self._iv_history = [self._iv]
        return state

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")

        old = self._current_state
        self._step_count += 1

        vol_signal = float(np.clip(action.trade_sizes[0], -1.0, 1.0))

        # Desired option position: +1 = long straddle, -1 = short straddle
        desired_pos = vol_signal
        trade = desired_pos - self._option_position
        self._option_position = desired_pos

        # Pay/receive premium
        if self.option_type == "call":
            opt_price = bs_call_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        else:
            opt_price = bs_put_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)

        trade_cost = abs(trade) * (opt_price * 0.01 + self.cost_per_trade)

        # Advance dynamics
        z = self._rng.standard_normal()
        self._spot = self._spot * math.exp(
            (self.risk_free_rate - 0.5 * self._rv ** 2) * self._dt
            + self._rv * math.sqrt(self._dt) * z
        )
        self._spot_history.append(self._spot)
        self._tau -= self._dt
        self._tau = max(self._tau, 0.0)

        # IV dynamics (OU with mean = realised vol)
        iv_speed = 3.0
        iv_noise = self._rng.standard_normal() * 0.02
        self._iv += iv_speed * (self._rv - self._iv) * self._dt + iv_noise * math.sqrt(self._dt)
        self._iv = max(self._iv, 0.05)
        self._iv_history.append(self._iv)

        # Update RV
        if len(self._spot_history) > 20:
            log_rets = [
                math.log(self._spot_history[i] / self._spot_history[i - 1])
                for i in range(max(1, len(self._spot_history) - 20), len(self._spot_history))
                if self._spot_history[i - 1] > 0
            ]
            if len(log_rets) > 1:
                self._rv = float(np.std(log_rets, ddof=1)) * math.sqrt(252.0 * self.num_steps_per_day)
                self._rv = max(self._rv, 0.01)

        # Revalue
        if self.option_type == "call":
            new_opt_price = bs_call_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)
        else:
            new_opt_price = bs_put_price(self._spot, self._strike, self._tau, self._iv, self.risk_free_rate)

        portfolio_value = old.cash - trade_cost + self._option_position * new_opt_price
        new_pnl = portfolio_value - self.initial_cash
        reward = new_pnl - old.pnl

        # IV rank / z-score
        iv_arr = np.array(self._iv_history)
        iv_rank = float((self._iv - iv_arr.min()) / max(iv_arr.max() - iv_arr.min(), 1e-8))
        iv_zscore = float((self._iv - iv_arr.mean()) / max(iv_arr.std(ddof=1), 1e-8)) if len(iv_arr) > 1 else 0.0

        done = self._step_count >= self.num_steps or self._tau <= 1e-12

        state = TradingState(
            timestamp=self._step_count,
            prices=np.array([self._spot], dtype=np.float64),
            position=np.array([self._option_position], dtype=np.float64),
            cash=old.cash - trade_cost,
            pnl=new_pnl,
            features={
                "iv": self._iv,
                "rv": self._rv,
                "iv_rank": iv_rank,
                "iv_zscore": iv_zscore,
            },
        )
        self._current_state = state if not done else None

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            truncated=False,
            info={
                "iv": self._iv,
                "rv": self._rv,
                "iv_rank": iv_rank,
                "iv_zscore": iv_zscore,
                "option_price": new_opt_price,
                "trade_cost": trade_cost,
            },
        )
