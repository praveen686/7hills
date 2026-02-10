"""Order execution environment with limit order book simulation.

Implements the Optimal Execution MDP from Chapter 10.2 of
"Foundations of RL with Applications in Finance" (Rao & Jelvis).

The agent must execute a large parent order (e.g. buy 1000 shares or
100 lots of NIFTY futures) over a fixed time horizon, minimising
market impact (implementation shortfall).

Classes:
  - LOBSimulator: lightweight limit order book with market impact
  - ExecutionEnv: MDP where the agent splits a parent order optimally

Market impact model (Almgren-Chriss, 2001):
  - Permanent impact: price moves by alpha_perm * trade_size (persists)
  - Temporary impact: execution price shifted by beta_temp * trade_rate
  - Price volatility: GBM micro-noise between steps

Book reference: Ch 10.2 (Optimal Execution), Ch 5 (Backward Induction).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from .trading_env import TradingEnv, TradingState, TradingAction, StepResult

__all__ = [
    "ExecutionEnv",
    "LOBSimulator",
]


# ---------------------------------------------------------------------------
# LOBSimulator
# ---------------------------------------------------------------------------


class LOBSimulator:
    """Simple Limit Order Book simulator.

    Simulates:
      - Bid/ask queues with configurable depth at each price level
      - Market impact (temporary + permanent, Almgren-Chriss)
      - Fill probability for limit orders (exponential model)
      - Queue priority (FIFO within price level)

    This is a *reduced-form* LOB -- not a full agent-based model.
    It's designed to be fast enough for RL training (millions of steps).

    Parameters
    ----------
    price : float
        Initial mid-price.
    spread : float
        Bid-ask spread (in price units).
    num_levels : int
        Number of price levels on each side.
    depth_mean : float
        Mean queue depth at each level (shares).
    fill_rate_k : float
        Exponential fill rate parameter for limit orders.
        P(fill in dt) = 1 - exp(-k * depth_ratio * dt).
    alpha_perm : float
        Permanent impact coefficient (price shift per share).
    beta_temp : float
        Temporary impact coefficient (execution price shift per share/step).
    sigma : float
        Per-step price volatility.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        price: float = 100.0,
        spread: float = 0.01,
        num_levels: int = 5,
        depth_mean: float = 100.0,
        fill_rate_k: float = 1.5,
        alpha_perm: float = 0.001,
        beta_temp: float = 0.01,
        sigma: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.mid_price = price
        self.spread = spread
        self.num_levels = num_levels
        self.depth_mean = depth_mean
        self.fill_rate_k = fill_rate_k
        self.alpha_perm = alpha_perm
        self.beta_temp = beta_temp
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

        # Order book state
        self._bid_depth = self._rng.poisson(depth_mean, num_levels).astype(np.float64)
        self._ask_depth = self._rng.poisson(depth_mean, num_levels).astype(np.float64)

        # Pending limit orders: list of (order_id, price, remaining_size, side)
        self._next_order_id = 0
        self._pending_orders: list[dict] = []

    @property
    def best_bid(self) -> float:
        return self.mid_price - self.spread / 2.0

    @property
    def best_ask(self) -> float:
        return self.mid_price + self.spread / 2.0

    @property
    def bid_depth(self) -> np.ndarray:
        return self._bid_depth.copy()

    @property
    def ask_depth(self) -> np.ndarray:
        return self._ask_depth.copy()

    def depth_imbalance(self) -> float:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth) at top level."""
        total = self._bid_depth[0] + self._ask_depth[0]
        if total < 1e-12:
            return 0.0
        return float((self._bid_depth[0] - self._ask_depth[0]) / total)

    def submit_market_order(self, size: int) -> tuple[float, float]:
        """Execute a market order immediately.

        Parameters
        ----------
        size : int
            Signed: positive = buy, negative = sell.

        Returns
        -------
        (average_fill_price, filled_qty)
        """
        if size == 0:
            return self.mid_price, 0.0

        abs_size = abs(size)
        is_buy = size > 0

        # Walk the book
        remaining = abs_size
        total_cost = 0.0
        depth = self._ask_depth if is_buy else self._bid_depth
        tick = self.spread / self.num_levels

        for level in range(self.num_levels):
            level_price = (
                self.best_ask + level * tick if is_buy
                else self.best_bid - level * tick
            )
            available = depth[level]
            fill = min(remaining, available)
            total_cost += fill * level_price
            depth[level] -= fill
            remaining -= fill
            if remaining <= 0:
                break

        filled = abs_size - remaining

        # Temporary impact on fill price
        if filled > 0:
            temp_impact = self.beta_temp * filled
            if is_buy:
                total_cost += temp_impact * filled
            else:
                total_cost -= temp_impact * filled

        avg_price = total_cost / max(filled, 1.0)

        # Permanent impact on mid-price
        self.mid_price += self.alpha_perm * size

        return avg_price, float(filled)

    def submit_limit_order(self, price: float, size: int) -> int:
        """Submit a limit order. Returns order_id.

        Parameters
        ----------
        price : float
            Limit price.
        size : int
            Signed: positive = buy, negative = sell.
        """
        oid = self._next_order_id
        self._next_order_id += 1
        self._pending_orders.append({
            "id": oid,
            "price": price,
            "remaining": abs(size),
            "side": "buy" if size > 0 else "sell",
        })
        return oid

    def step(self, dt: float = 1.0) -> list[dict]:
        """Advance LOB by one time step.

        - Replenish depth (Poisson arrivals)
        - Random walk mid-price
        - Check limit order fills

        Parameters
        ----------
        dt : float
            Time step (normalised; 1.0 = one step).

        Returns
        -------
        List of filled orders: [{"id", "fill_price", "fill_qty"}, ...]
        """
        # 1) Mid-price random walk
        self.mid_price *= math.exp(self._rng.normal(0, self.sigma * math.sqrt(dt)))

        # 2) Replenish depth
        for level in range(self.num_levels):
            arrival = self._rng.poisson(self.depth_mean * 0.1 * dt)
            self._bid_depth[level] = min(self._bid_depth[level] + arrival, self.depth_mean * 3)
            arrival = self._rng.poisson(self.depth_mean * 0.1 * dt)
            self._ask_depth[level] = min(self._ask_depth[level] + arrival, self.depth_mean * 3)

        # 3) Check limit order fills
        fills: list[dict] = []
        surviving: list[dict] = []

        for order in self._pending_orders:
            if order["side"] == "buy" and order["price"] >= self.best_ask:
                # Fill at limit price
                fills.append({
                    "id": order["id"],
                    "fill_price": order["price"],
                    "fill_qty": order["remaining"],
                })
            elif order["side"] == "sell" and order["price"] <= self.best_bid:
                fills.append({
                    "id": order["id"],
                    "fill_price": order["price"],
                    "fill_qty": order["remaining"],
                })
            else:
                # Probabilistic fill based on distance from mid
                if order["side"] == "buy":
                    distance = self.best_ask - order["price"]
                else:
                    distance = order["price"] - self.best_bid
                if distance < 0:
                    distance = 0.0
                # Exponential fill probability
                fill_prob = 1.0 - math.exp(
                    -self.fill_rate_k * max(0, 1.0 - distance / self.spread) * dt
                )
                if self._rng.random() < fill_prob:
                    fills.append({
                        "id": order["id"],
                        "fill_price": order["price"],
                        "fill_qty": order["remaining"],
                    })
                else:
                    surviving.append(order)

        self._pending_orders = surviving
        return fills

    def cancel_all(self) -> None:
        """Cancel all pending limit orders."""
        self._pending_orders.clear()


# ---------------------------------------------------------------------------
# ExecutionEnv
# ---------------------------------------------------------------------------


class ExecutionEnv(TradingEnv):
    """Order execution MDP environment.

    The agent must sell ``total_shares`` over ``num_steps`` time steps,
    minimising implementation shortfall (deviation from arrival price).

    State vector (dimension 8):
        [time_remaining/T, remaining_shares/total, mid_price_change,
         spread, bid_depth_imbalance, recent_volume, volatility, urgency]

    Action (2-dim):
        [trade_fraction, aggressiveness]
        - trade_fraction: fraction of remaining to trade this step [0, 1]
        - aggressiveness: -1 (passive limit) to +1 (aggressive market)

    Reward: negative implementation shortfall per step.

    Benchmark: TWAP (uniform execution), Almgren-Chriss (optimal deterministic).

    Parameters
    ----------
    total_shares : int
        Total shares to execute.
    num_steps : int
        Number of execution intervals.
    price_init : float
        Initial mid-price.
    sigma : float
        Per-step price volatility.
    spread : float
        Bid-ask spread.
    depth_mean : float
        Average LOB depth at each level.
    alpha_perm : float
        Permanent impact coefficient.
    beta_temp : float
        Temporary impact coefficient.
    side : str
        ``"sell"`` or ``"buy"``.

    Book reference: Ch 10.2 (Optimal Execution), Almgren & Chriss (2001).
    """

    def __init__(
        self,
        total_shares: int = 1000,
        num_steps: int = 100,
        price_init: float = 100.0,
        sigma: float = 0.001,
        spread: float = 0.01,
        depth_mean: float = 100.0,
        alpha_perm: float = 0.001,
        beta_temp: float = 0.01,
        side: str = "sell",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("initial_cash", 0.0)
        kwargs.setdefault("transaction_cost_bps", 0.0)
        kwargs.setdefault("slippage_bps", 0.0)
        super().__init__(**kwargs)

        self.total_shares = total_shares
        self.num_steps = num_steps
        self.price_init = price_init
        self._sigma = sigma
        self._spread = spread
        self._depth_mean = depth_mean
        self._alpha_perm = alpha_perm
        self._beta_temp = beta_temp
        self.side = side

        # Will be initialised in reset()
        self._lob: Optional[LOBSimulator] = None
        self._remaining: int = 0
        self._arrival_price: float = 0.0
        self._total_cost: float = 0.0
        self._price_history: list[float] = []

    @property
    def state_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return 2  # (trade_fraction, aggressiveness)

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._remaining = self.total_shares
        self._total_cost = 0.0

        self._lob = LOBSimulator(
            price=self.price_init,
            spread=self._spread,
            depth_mean=self._depth_mean,
            alpha_perm=self._alpha_perm,
            beta_temp=self._beta_temp,
            sigma=self._sigma,
            seed=self._rng.integers(0, 2**31),
        )
        self._arrival_price = self._lob.mid_price
        self._price_history = [self._arrival_price]

        state = self._build_state(0.0)
        self._current_state = state
        return state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None or self._lob is None:
            raise RuntimeError("Call reset() before step()")

        old = self._current_state
        self._step_count += 1

        # Decode action
        trade_fraction = float(np.clip(action.trade_sizes[0], 0.0, 1.0))
        aggressiveness = float(np.clip(
            action.trade_sizes[1] if len(action.trade_sizes) > 1 else 0.5,
            -1.0, 1.0,
        ))

        # Compute trade size
        trade_size = max(1, int(round(trade_fraction * self._remaining)))
        trade_size = min(trade_size, self._remaining)

        if trade_size == 0 and self._remaining > 0 and self._step_count >= self.num_steps:
            trade_size = self._remaining  # force complete at end

        # Execute based on aggressiveness
        sign = -1 if self.side == "sell" else 1
        fill_price = 0.0
        filled_qty = 0.0

        if aggressiveness > 0.0:
            # Market order
            fill_price, filled_qty = self._lob.submit_market_order(sign * trade_size)
        elif aggressiveness < -0.5:
            # Passive limit order (behind mid)
            offset = abs(aggressiveness) * self._spread
            limit_price = (
                self._lob.best_bid - offset if self.side == "sell"
                else self._lob.best_ask + offset
            )
            oid = self._lob.submit_limit_order(limit_price, sign * trade_size)
            fills = self._lob.step(1.0)
            for f in fills:
                if f["id"] == oid:
                    fill_price = f["fill_price"]
                    filled_qty = f["fill_qty"]
                    break
            if filled_qty == 0:
                fill_price = self._lob.mid_price
        else:
            # Limit at mid / slightly aggressive
            limit_price = (
                self._lob.best_bid + aggressiveness * self._spread * 0.5
                if self.side == "sell"
                else self._lob.best_ask - aggressiveness * self._spread * 0.5
            )
            fill_price, filled_qty = self._lob.submit_market_order(sign * trade_size)

        filled_qty = abs(filled_qty)

        # Update state
        self._remaining -= int(filled_qty)
        self._remaining = max(0, self._remaining)

        # Implementation shortfall for this step
        if self.side == "sell":
            shortfall = (self._arrival_price - fill_price) * filled_qty
        else:
            shortfall = (fill_price - self._arrival_price) * filled_qty
        self._total_cost += shortfall

        # Advance LOB
        if aggressiveness > 0.0:
            self._lob.step(1.0)
        self._price_history.append(self._lob.mid_price)

        # Reward = negative shortfall (lower is better)
        reward = -shortfall

        done = self._step_count >= self.num_steps or self._remaining <= 0
        truncated = False

        # Force-complete if time is up but shares remain
        if done and self._remaining > 0:
            fp, fq = self._lob.submit_market_order(sign * self._remaining)
            extra_shortfall = (
                (self._arrival_price - fp) * abs(fq) if self.side == "sell"
                else (fp - self._arrival_price) * abs(fq)
            )
            self._total_cost += extra_shortfall
            reward -= extra_shortfall
            self._remaining = 0

        new_pnl = -self._total_cost
        state = self._build_state(new_pnl)
        self._current_state = state if not done else None

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            truncated=truncated,
            info={
                "fill_price": fill_price,
                "filled_qty": filled_qty,
                "remaining": self._remaining,
                "shortfall_step": shortfall,
                "total_shortfall": self._total_cost,
                "mid_price": self._lob.mid_price,
            },
        )

    def _build_state(self, pnl: float) -> TradingState:
        assert self._lob is not None
        h = self._price_history
        n = len(h)

        time_remaining = max(0.0, 1.0 - self._step_count / self.num_steps)
        shares_remaining = self._remaining / max(self.total_shares, 1)
        price_change = (h[-1] - h[0]) / h[0] if n > 1 and h[0] != 0 else 0.0

        # Recent volatility
        if n > 10:
            rets = [h[i] / h[i - 1] - 1.0 for i in range(max(1, n - 10), n) if h[i - 1] != 0]
            vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else self._sigma
        else:
            vol = self._sigma

        # Urgency = remaining / time_remaining
        urgency = shares_remaining / max(time_remaining, 0.01)

        features = {
            "time_remaining": time_remaining,
            "shares_remaining_frac": shares_remaining,
            "price_change": price_change,
            "spread": self._lob.spread,
            "depth_imbalance": self._lob.depth_imbalance(),
            "recent_volume": float(self._depth_mean),
            "volatility": vol,
            "urgency": urgency,
        }

        return TradingState(
            timestamp=self._step_count,
            prices=np.array([self._lob.mid_price], dtype=np.float64),
            position=np.array([float(self._remaining)], dtype=np.float64),
            cash=pnl,
            pnl=pnl,
            features=features,
        )

    def twap_schedule(self) -> np.ndarray:
        """Return TWAP schedule: uniform trade_fraction at each step.

        Returns array of shape (num_steps,) with constant values.
        """
        per_step = self.total_shares / self.num_steps
        return np.full(self.num_steps, per_step / self.total_shares)

    def almgren_chriss_schedule(self, risk_aversion: float = 1e-6) -> np.ndarray:
        """Compute Almgren-Chriss optimal deterministic schedule.

        The optimal strategy for selling X shares over T steps with
        permanent impact alpha, temporary impact beta, volatility sigma,
        and risk aversion lambda is:

            x_k = X * sinh(kappa * (T-k)) / sinh(kappa * T)
            where kappa = arccosh(1 + (lambda * sigma^2) / (2 * beta))

        Returns array of trade fractions at each step.

        Book reference: Ch 10.2, Almgren & Chriss (2001).
        """
        T = self.num_steps
        X = self.total_shares
        lam = risk_aversion
        sig = self._sigma
        beta = self._beta_temp

        # kappa
        arg = 1.0 + lam * sig ** 2 / (2.0 * beta)
        arg = max(arg, 1.0 + 1e-12)
        kappa = math.acosh(arg)

        if kappa < 1e-12:
            # Risk-neutral: TWAP
            return np.full(T, 1.0 / T)

        inventory = np.zeros(T + 1)
        inventory[0] = X
        for k in range(T):
            inventory[k + 1] = X * math.sinh(kappa * (T - k - 1)) / math.sinh(kappa * T)

        trades = np.diff(inventory)
        # Normalise to fractions of initial
        fractions = -trades / X  # negative because inventory decreases
        return np.clip(fractions, 0.0, 1.0)
