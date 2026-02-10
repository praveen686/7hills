"""RL Market-Making Agent.

Primary: BTC/ETH perps on Binance (270ns via Rust SBE feed).
Secondary: India FnO options (wider spreads, less liquid).

Combines the analytical Avellaneda-Stoikov (A-S) market-making solution
with Actor-Critic RL for adaptation to real market conditions.

The A-S framework provides a principled starting point:
    reservation_price = S - q * gamma * sigma^2 * (T - t)
    optimal_spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

where q = inventory, gamma = risk aversion, sigma = volatility,
T-t = time remaining, k = fill rate.

The RL agent then fine-tunes these for:
  - Non-exponential fill distributions (real LOBs)
  - Adverse selection (toxic flow detection)
  - Time-of-day patterns
  - Funding rate dynamics (crypto perps)

Book reference: Ch 10.3 (Market-Making), Avellaneda & Stoikov (2008).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    "MarketMakingAgent",
]


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov analytical solution
# ---------------------------------------------------------------------------


def avellaneda_stoikov_quotes(
    mid_price: float,
    inventory: int,
    sigma: float,
    gamma: float,
    T_remaining: float,
    fill_rate_k: float = 1.5,
) -> dict:
    """Compute Avellaneda-Stoikov optimal bid/ask quotes.

    Reservation price (indifference price):
        r = S - q * gamma * sigma^2 * (T - t)

    Optimal spread around reservation price:
        delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    bid = r - delta/2
    ask = r + delta/2

    Parameters
    ----------
    mid_price : float
    inventory : int (signed)
    sigma : float (per-step vol)
    gamma : float (risk aversion)
    T_remaining : float (fraction of session remaining)
    fill_rate_k : float (exponential fill rate parameter)

    Returns
    -------
    dict with bid_price, ask_price, reservation_price, optimal_spread.

    Book reference: Ch 10.3, Avellaneda & Stoikov (2008), Thm 1.
    """
    T = max(T_remaining, 1e-8)
    q = inventory

    # Reservation price
    reservation = mid_price - q * gamma * sigma ** 2 * T

    # Optimal spread
    spread_term1 = gamma * sigma ** 2 * T
    if gamma > 0 and fill_rate_k > 0:
        spread_term2 = (2.0 / gamma) * math.log(1.0 + gamma / fill_rate_k)
    else:
        spread_term2 = 0.0
    optimal_spread = spread_term1 + spread_term2

    bid = reservation - optimal_spread / 2.0
    ask = reservation + optimal_spread / 2.0

    return {
        "bid_price": bid,
        "ask_price": ask,
        "reservation_price": reservation,
        "optimal_spread": optimal_spread,
    }


# ---------------------------------------------------------------------------
# RL network
# ---------------------------------------------------------------------------


class _MMNetwork(nn.Module):
    """Market-making policy network.

    Input: state vector
    Output: (bid_offset, ask_offset, bid_size_frac, ask_size_frac)
    All outputs in [-1, 1] or [0, 1], scaled externally.
    """

    def __init__(self, state_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.shared = nn.Sequential(*layers)

        # Bid/ask offset: tanh for [-1, 1] range
        self.bid_offset = nn.Sequential(nn.Linear(in_dim, 1), nn.Tanh())
        self.ask_offset = nn.Sequential(nn.Linear(in_dim, 1), nn.Tanh())
        # Sizes: sigmoid for [0, 1]
        self.bid_size = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.ask_size = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return torch.cat([
            self.bid_offset(h),
            self.ask_offset(h),
            self.bid_size(h),
            self.ask_size(h),
        ], dim=-1)


# ---------------------------------------------------------------------------
# MarketMakingAgent
# ---------------------------------------------------------------------------


class MarketMakingAgent:
    """Production RL market maker combining A-S + Actor-Critic.

    Uses the analytical Avellaneda-Stoikov solution as initialisation
    / warmstart, then RL fine-tunes for real market conditions:
      - Non-exponential fill distributions
      - Adverse selection (informed traders)
      - Inventory penalties
      - Funding rate impact (crypto perps)

    State (dimension 10):
        [mid_price_change, inventory/max, spread, depth_imbalance,
         trade_flow_imbalance, volatility, funding_rate,
         time_of_day_sin, time_of_day_cos, pnl_normalised]

    Action (continuous, 4-dim):
        [bid_offset, ask_offset, bid_size, ask_size]

    Parameters
    ----------
    instrument : str
        ``"BTCUSDT"`` | ``"ETHUSDT"`` | ``"NIFTY"``
    max_inventory : int
        Maximum allowed inventory (absolute).
    sigma : float
        Estimated per-step volatility.
    gamma_risk : float
        Risk aversion parameter for A-S.
    fill_rate_k : float
        Exponential fill rate parameter.
    hidden_layers : Sequence[int]
        RL network hidden layer sizes.
    learning_rate : float
    warmstart_analytical : bool
        If True, initialise actions from A-S before RL takes over.
    device : str

    Book reference: Ch 10.3 (Market Making), Avellaneda & Stoikov (2008).
    """

    def __init__(
        self,
        instrument: str = "BTCUSDT",
        max_inventory: int = 10,
        sigma: float = 0.02,
        gamma_risk: float = 0.1,
        fill_rate_k: float = 1.5,
        hidden_layers: Sequence[int] = (256, 128, 64),
        learning_rate: float = 1e-4,
        warmstart_analytical: bool = True,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.instrument = instrument.upper()
        self.max_inventory = max_inventory
        self.sigma = sigma
        self.gamma_risk = gamma_risk
        self.fill_rate_k = fill_rate_k
        self.warmstart_analytical = warmstart_analytical
        self._rng = np.random.default_rng(seed)

        self._state_dim = 10
        self._action_dim = 4

        if _TORCH_AVAILABLE:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self._policy = _MMNetwork(self._state_dim, list(hidden_layers)).to(self.device)
            self._critic = nn.Sequential(
                *[layer for h in hidden_layers
                  for layer in [nn.Linear(self._state_dim if h == hidden_layers[0] else prev_h, h), nn.ReLU()]
                  for prev_h in [h]][:2 * len(hidden_layers)],
            )
            # Rebuild critic properly
            critic_layers: list[nn.Module] = []
            in_dim = self._state_dim
            for h in hidden_layers:
                critic_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
                in_dim = h
            critic_layers.append(nn.Linear(in_dim, 1))
            self._critic = nn.Sequential(*critic_layers).to(self.device)

            self._policy_optim = optim.Adam(self._policy.parameters(), lr=learning_rate)
            self._critic_optim = optim.Adam(self._critic.parameters(), lr=learning_rate * 3)
        else:
            self.device = None
            self._policy = None
            self._critic = None

        self._trained = False
        self._train_history: list[dict] = []
        self._total_pnl = 0.0
        self._inventory = 0

    def get_quotes(
        self,
        mid_price: float,
        inventory: int,
        market_state: dict,
    ) -> dict:
        """Compute bid/ask quotes for the current market state.

        If untrained or warmstart mode, uses Avellaneda-Stoikov.
        If trained, uses the RL policy with A-S as baseline.

        Parameters
        ----------
        mid_price : float
            Current mid-price.
        inventory : int
            Current inventory (signed).
        market_state : dict
            Keys: spread, depth_imbalance, trade_flow, volatility,
            funding_rate, time_fraction, pnl.

        Returns
        -------
        dict with bid_price, ask_price, bid_size, ask_size.
        """
        self._inventory = inventory

        # Analytical baseline
        T_remaining = market_state.get("time_fraction", 0.5)
        vol = market_state.get("volatility", self.sigma)
        as_quotes = avellaneda_stoikov_quotes(
            mid_price, inventory, vol, self.gamma_risk, T_remaining, self.fill_rate_k
        )

        if not self._trained or self._policy is None or self.warmstart_analytical:
            # Pure analytical
            spread = as_quotes["optimal_spread"]
            max_size = max(1, self.max_inventory - abs(inventory))

            return {
                "bid_price": as_quotes["bid_price"],
                "ask_price": as_quotes["ask_price"],
                "bid_size": max_size,
                "ask_size": max_size,
            }

        # RL-augmented quotes
        state = self._build_state(mid_price, inventory, market_state)
        self._policy.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw_action = self._policy(x).squeeze(0).cpu().numpy()

        # Decode actions: offsets are relative to A-S quotes
        spread = as_quotes["optimal_spread"]
        bid_offset = float(raw_action[0]) * spread * 0.5  # [-spread/2, +spread/2]
        ask_offset = float(raw_action[1]) * spread * 0.5
        bid_size_frac = float(raw_action[2])
        ask_size_frac = float(raw_action[3])

        max_size = max(1, self.max_inventory - abs(inventory))
        bid_size = max(1, int(round(bid_size_frac * max_size)))
        ask_size = max(1, int(round(ask_size_frac * max_size)))

        return {
            "bid_price": as_quotes["bid_price"] + bid_offset,
            "ask_price": as_quotes["ask_price"] + ask_offset,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    def train(self, env: Any, num_episodes: int = 50000) -> dict:
        """Train the RL market maker on the given environment.

        Uses Actor-Critic with experience collected from the environment.

        Parameters
        ----------
        env : TradingEnv
            Market-making environment.
        num_episodes : int
            Training episodes.

        Returns
        -------
        dict with avg_pnl, sharpe, max_drawdown, episode_pnls.
        """
        if self._policy is None:
            raise RuntimeError("PyTorch required for training")

        episode_pnls: list[float] = []
        gamma = 0.99

        for ep in range(num_episodes):
            state = env.reset()
            state_arr = self._state_from_trading_state(state)

            states, actions, rewards, log_probs, values = [], [], [], [], []
            done = False

            while not done:
                x = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)

                self._policy.train()
                raw = self._policy(x).squeeze(0)

                # Add exploration noise
                noise = torch.randn_like(raw) * 0.1
                action = (raw + noise).clamp(-1, 1)

                self._critic.eval()
                value = self._critic(x).item()

                # Map to trading action
                from .trading_env import TradingAction
                act = TradingAction(trade_sizes=action.detach().cpu().numpy())

                try:
                    result = env.step(act)
                except Exception as e:
                    logger.warning("Market maker env step failed: %s", e)
                    break

                states.append(state_arr)
                actions.append(action.detach())
                rewards.append(result.reward)
                values.append(value)

                done = result.done or result.truncated
                state_arr = self._state_from_trading_state(result.state)

            if len(rewards) == 0:
                continue

            ep_pnl = sum(rewards)
            episode_pnls.append(ep_pnl)

            # Compute returns
            returns = np.zeros(len(rewards))
            running = 0.0
            for t in reversed(range(len(rewards))):
                running = rewards[t] + gamma * running
                returns[t] = running

            # Update
            states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            values_t = torch.tensor(np.array(values), dtype=torch.float32, device=self.device)
            advantages = returns_t - values_t

            # Critic update
            self._critic.train()
            v_pred = self._critic(states_t).squeeze(-1)
            critic_loss = nn.functional.mse_loss(v_pred, returns_t)
            self._critic_optim.zero_grad()
            critic_loss.backward()
            self._critic_optim.step()

            # Actor update (policy gradient)
            self._policy.train()
            raw_actions = self._policy(states_t)
            # Simple REINFORCE-style: maximise E[A * log_pi]
            # For continuous, use negative MSE to saved actions as surrogate
            saved_actions = torch.stack(actions)
            actor_loss = -(raw_actions * advantages.unsqueeze(-1)).mean()

            self._policy_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
            self._policy_optim.step()

        self._trained = True

        if len(episode_pnls) > 0:
            pnls = np.array(episode_pnls)
            avg_pnl = float(pnls.mean())
            std_pnl = float(pnls.std(ddof=1)) if len(pnls) > 1 else 1e-8
            sharpe = avg_pnl / max(std_pnl, 1e-8) * math.sqrt(252)

            # Max drawdown
            cumulative = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative)
            dd = peak - cumulative
            max_dd = float(dd.max())
        else:
            avg_pnl = sharpe = max_dd = 0.0

        return {
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "episode_pnls": episode_pnls,
            "num_episodes": len(episode_pnls),
        }

    def evaluate(self, env: Any, num_episodes: int = 1000) -> dict:
        """Evaluate the market maker over multiple episodes.

        Returns
        -------
        dict with avg_pnl, sharpe, max_drawdown, avg_inventory,
        fill_rate, spread_captured.
        """
        pnls: list[float] = []
        inventories: list[float] = []
        fills = 0
        quotes = 0

        for ep in range(num_episodes):
            state = env.reset(seed=ep + 20000)
            ep_pnl = 0.0
            done = False

            while not done:
                state_arr = self._state_from_trading_state(state)

                if self._policy is not None:
                    self._policy.eval()
                    with torch.no_grad():
                        x = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
                        action = self._policy(x).squeeze(0).cpu().numpy()
                else:
                    action = np.array([0.0, 0.0, 0.5, 0.5])

                from .trading_env import TradingAction
                result = env.step(TradingAction(trade_sizes=action))

                ep_pnl += result.reward
                inventories.append(abs(state.position[0]) if len(state.position) > 0 else 0.0)
                quotes += 1
                if result.info.get("filled", False):
                    fills += 1

                done = result.done or result.truncated
                state = result.state

            pnls.append(ep_pnl)

        pnls_arr = np.array(pnls)
        avg_pnl = float(pnls_arr.mean())
        std_pnl = float(pnls_arr.std(ddof=1)) if len(pnls) > 1 else 1e-8
        sharpe = avg_pnl / max(std_pnl, 1e-8) * math.sqrt(252)

        cumulative = np.cumsum(pnls_arr)
        peak = np.maximum.accumulate(cumulative)
        max_dd = float((peak - cumulative).max())

        return {
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_inventory": float(np.mean(inventories)) if inventories else 0.0,
            "fill_rate": fills / max(quotes, 1),
            "num_episodes": len(pnls),
        }

    def _build_state(self, mid_price: float, inventory: int, market_state: dict) -> np.ndarray:
        """Build state from raw market data."""
        spread = market_state.get("spread", 0.01)
        depth_imb = market_state.get("depth_imbalance", 0.0)
        trade_flow = market_state.get("trade_flow", 0.0)
        vol = market_state.get("volatility", self.sigma)
        funding = market_state.get("funding_rate", 0.0)
        time_frac = market_state.get("time_fraction", 0.5)
        pnl = market_state.get("pnl", 0.0)
        price_change = market_state.get("price_change", 0.0)

        return np.array([
            price_change,
            inventory / max(self.max_inventory, 1),
            spread / max(mid_price * 0.001, 1e-8),
            depth_imb,
            trade_flow,
            vol / max(self.sigma, 1e-8),
            funding * 10000.0,  # scale to reasonable range
            math.sin(2.0 * math.pi * time_frac),
            math.cos(2.0 * math.pi * time_frac),
            pnl / max(mid_price, 1e-8),
        ], dtype=np.float32)

    def _state_from_trading_state(self, state: Any) -> np.ndarray:
        """Convert TradingState to numpy state vector."""
        features = state.features if hasattr(state, "features") else {}
        price = float(state.prices[0]) if hasattr(state, "prices") and len(state.prices) > 0 else 100.0
        pos = float(state.position[0]) if hasattr(state, "position") and len(state.position) > 0 else 0.0

        return np.array([
            features.get("price_change", features.get("returns", 0.0)),
            pos / max(self.max_inventory, 1),
            features.get("spread", 0.01) / max(price * 0.001, 1e-8),
            features.get("depth_imbalance", 0.0),
            features.get("trade_flow", features.get("volume_momentum", 0.0)),
            features.get("volatility", self.sigma) / max(self.sigma, 1e-8),
            features.get("funding_rate", 0.0) * 10000.0,
            math.sin(2.0 * math.pi * features.get("time_fraction", 0.5)),
            math.cos(2.0 * math.pi * features.get("time_fraction", 0.5)),
            state.pnl / max(price, 1e-8) if hasattr(state, "pnl") else 0.0,
        ], dtype=np.float32)
