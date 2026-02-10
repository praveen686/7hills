"""RL Optimal Execution Agent.

Handles large order execution for FnO + crypto, minimising market
impact using reinforcement learning.

The agent takes a parent order (e.g. "buy 100 lots NIFTY futures")
and splits it into child orders that trade off:
  - Market impact (large trades move prices)
  - Timing risk (slow execution exposes to adverse price moves)
  - Transaction costs (maker vs taker fees)

Three modes:
  1. ``"actor_critic"``: PPO-style continuous action RL
  2. ``"dqn"``: discretised action space (5 aggressiveness levels)
  3. ``"almgren_chriss"``: analytical benchmark (no learning)

Book reference: Ch 10.2 (Optimal Execution), Almgren & Chriss (2001),
                Bertsimas & Lo (1998).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    "OptimalExecutionAgent",
]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _ActorNetwork(nn.Module):
    """Actor: state -> (trade_fraction_mean, aggressiveness_mean)."""

    def __init__(self, state_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.fraction_head = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.aggress_head = nn.Sequential(nn.Linear(in_dim, 1), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2) - 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        frac = self.fraction_head(h)
        agg = self.aggress_head(h)
        mean = torch.cat([frac, agg], dim=-1)  # (B, 2)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class _CriticNetwork(nn.Module):
    """Critic: state -> V(state)."""

    def __init__(self, state_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# OptimalExecutionAgent
# ---------------------------------------------------------------------------


class OptimalExecutionAgent:
    """RL agent for optimal order execution.

    Takes a parent order and splits it into child orders that minimise
    implementation shortfall (difference between arrival price and
    average execution price).

    State (dimension 8):
        [time_remaining_frac, remaining_shares_frac, price_change,
         spread, depth_imbalance, recent_volume, volatility, urgency]

    Action (continuous, 2-dim):
        [trade_fraction, aggressiveness]
        - trade_fraction in [0, 1]: fraction of remaining to trade now
        - aggressiveness in [-1, 1]: passive limit (-1) to market order (+1)

    Reward: negative implementation shortfall per step.

    Parameters
    ----------
    instrument : str
        ``"NIFTY"`` | ``"BANKNIFTY"`` | ``"BTCUSDT"``
    algo : str
        ``"actor_critic"`` | ``"dqn"`` | ``"almgren_chriss"``
    hidden_layers : Sequence[int]
        Neural network hidden layer sizes.
    learning_rate : float
        Optimizer learning rate.
    risk_aversion : float
        Almgren-Chriss risk aversion parameter.
    device : str
        ``"auto"`` | ``"cuda"`` | ``"cpu"``

    Book reference: Ch 10.2 (Optimal Execution).
    """

    def __init__(
        self,
        instrument: str = "NIFTY",
        algo: str = "actor_critic",
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 3e-4,
        risk_aversion: float = 1e-6,
        gamma: float = 0.99,
        device: str = "auto",
    ) -> None:
        self.instrument = instrument.upper()
        self.algo = algo
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.risk_aversion = risk_aversion
        self.gamma = gamma

        self._state_dim = 8
        self._action_dim = 2

        if algo in ("actor_critic", "dqn") and _TORCH_AVAILABLE:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self._actor = _ActorNetwork(self._state_dim, self.hidden_layers).to(self.device)
            self._critic = _CriticNetwork(self._state_dim, self.hidden_layers).to(self.device)
            self._actor_optim = optim.Adam(self._actor.parameters(), lr=learning_rate)
            self._critic_optim = optim.Adam(self._critic.parameters(), lr=learning_rate)
        else:
            self.device = None
            self._actor = None
            self._critic = None

        self._train_history: list[dict] = []
        self._trained = False

    def execute_order(
        self,
        total_qty: int,
        side: str,
        time_horizon: int,
        market_state: dict,
    ) -> list[dict]:
        """Execute a parent order and return list of child orders.

        Parameters
        ----------
        total_qty : int
            Total quantity to execute.
        side : str
            ``"buy"`` or ``"sell"``
        time_horizon : int
            Number of execution intervals.
        market_state : dict
            Initial market state with keys: mid_price, spread, volatility, depth.

        Returns
        -------
        List of child order dicts, each containing:
            {time, qty, price, type, fill_price, slippage}
        """
        if self.algo == "almgren_chriss":
            return self._execute_almgren_chriss(total_qty, side, time_horizon, market_state)

        return self._execute_rl(total_qty, side, time_horizon, market_state)

    def _execute_rl(
        self,
        total_qty: int,
        side: str,
        time_horizon: int,
        market_state: dict,
    ) -> list[dict]:
        """Execute using trained RL policy."""
        if self._actor is None:
            raise RuntimeError("RL agent not initialised (PyTorch required)")

        mid_price = market_state.get("mid_price", 100.0)
        spread = market_state.get("spread", 0.01)
        vol = market_state.get("volatility", 0.001)
        depth = market_state.get("depth", 100.0)

        remaining = total_qty
        orders: list[dict] = []
        arrival_price = mid_price

        self._actor.eval()

        for t in range(time_horizon):
            if remaining <= 0:
                break

            # Build state
            time_frac = 1.0 - t / time_horizon
            remain_frac = remaining / max(total_qty, 1)
            price_change = (mid_price - arrival_price) / max(arrival_price, 1e-8)
            urgency = remain_frac / max(time_frac, 0.01)

            state = np.array([
                time_frac, remain_frac, price_change, spread,
                0.0, depth, vol, urgency,
            ], dtype=np.float32)

            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mean, std = self._actor(x)
                action = mean.squeeze(0).cpu().numpy()

            trade_frac = float(np.clip(action[0], 0.01, 1.0))
            aggress = float(np.clip(action[1], -1.0, 1.0))

            qty = max(1, int(round(trade_frac * remaining)))
            qty = min(qty, remaining)

            # Simulate execution
            slippage = vol * qty / max(depth, 1.0) * 0.5
            if side == "buy":
                fill_price = mid_price + spread / 2.0 + slippage
            else:
                fill_price = mid_price - spread / 2.0 - slippage

            orders.append({
                "time": t,
                "qty": qty,
                "price": mid_price,
                "type": "market" if aggress > 0 else "limit",
                "fill_price": fill_price,
                "slippage": slippage,
            })

            remaining -= qty
            # Price impact
            mid_price += 0.001 * qty * (1 if side == "buy" else -1)

        # Force-fill remainder
        if remaining > 0:
            orders.append({
                "time": time_horizon - 1,
                "qty": remaining,
                "price": mid_price,
                "type": "market",
                "fill_price": mid_price + (spread if side == "buy" else -spread),
                "slippage": spread,
            })

        return orders

    def _execute_almgren_chriss(
        self,
        total_qty: int,
        side: str,
        time_horizon: int,
        market_state: dict,
    ) -> list[dict]:
        """Execute using Almgren-Chriss optimal analytical schedule.

        The optimal trajectory for risk-averse execution:
            n_k = X * sinh(kappa*(T-k)) / sinh(kappa*T)
        where kappa = arccosh(1 + lambda*sigma^2/(2*eta)).
        """
        mid_price = market_state.get("mid_price", 100.0)
        vol = market_state.get("volatility", 0.001)
        spread = market_state.get("spread", 0.01)
        eta = market_state.get("temp_impact", 0.01)

        T = time_horizon
        X = total_qty
        lam = self.risk_aversion

        # Compute kappa
        arg = 1.0 + lam * vol ** 2 / (2.0 * max(eta, 1e-12))
        arg = max(arg, 1.0 + 1e-12)
        kappa = math.acosh(arg)

        orders: list[dict] = []
        remaining = X

        for k in range(T):
            if kappa < 1e-12:
                trade = X / T
            else:
                inv_now = X * math.sinh(kappa * (T - k)) / math.sinh(kappa * T)
                inv_next = X * math.sinh(kappa * (T - k - 1)) / math.sinh(kappa * T)
                trade = inv_now - inv_next

            qty = max(0, min(int(round(trade)), remaining))
            if qty == 0 and remaining > 0 and k == T - 1:
                qty = remaining

            slippage = vol * qty / 100.0 * 0.5
            if side == "buy":
                fill_price = mid_price + spread / 2.0 + slippage
            else:
                fill_price = mid_price - spread / 2.0 - slippage

            orders.append({
                "time": k,
                "qty": qty,
                "price": mid_price,
                "type": "limit",
                "fill_price": fill_price,
                "slippage": slippage,
            })

            remaining -= qty
            mid_price += 0.001 * qty * (1 if side == "buy" else -1)

        return orders

    def train(self, env: Any, num_episodes: int = 10000) -> dict:
        """Train the RL execution agent on the given environment.

        Uses Actor-Critic (A2C) with GAE for variance reduction.

        Parameters
        ----------
        env : ExecutionEnv
            Execution environment instance.
        num_episodes : int
            Number of training episodes.

        Returns
        -------
        dict with "avg_reward", "avg_shortfall", "episode_rewards".
        """
        if self._actor is None:
            raise RuntimeError("RL agent not initialised (PyTorch required)")

        episode_rewards: list[float] = []

        for ep in range(num_episodes):
            state = env.reset()
            state_arr = state.to_array()[:self._state_dim]

            states, actions, rewards, log_probs, values = [], [], [], [], []
            done = False

            while not done:
                x = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)

                self._actor.train()
                mean, std = self._actor(x)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

                self._critic.eval()
                value = self._critic(x)

                # Step environment
                from .trading_env import TradingAction
                act = TradingAction(trade_sizes=action.squeeze(0).detach().cpu().numpy())

                try:
                    result = env.step(act)
                except Exception:
                    break

                states.append(state_arr)
                actions.append(action.squeeze(0).detach().cpu().numpy())
                rewards.append(result.reward)
                log_probs.append(log_prob.item())
                values.append(value.item())

                done = result.done or result.truncated
                state_arr = result.state.to_array()[:self._state_dim]

            if len(rewards) == 0:
                continue

            ep_reward = sum(rewards)
            episode_rewards.append(ep_reward)

            # Compute returns and advantages (GAE)
            returns_arr = np.zeros(len(rewards))
            advantages = np.zeros(len(rewards))
            gae = 0.0
            next_value = 0.0

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * 0.95 * gae
                advantages[t] = gae
                returns_arr[t] = gae + values[t]
                next_value = values[t]

            # Normalise advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # Update actor
            states_t = torch.tensor(
                np.array(states), dtype=torch.float32, device=self.device
            )
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns_arr, dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(
                np.array(log_probs), dtype=torch.float32, device=self.device
            )

            self._actor.train()
            mean, std = self._actor(states_t)
            actions_t = torch.tensor(
                np.array(actions), dtype=torch.float32, device=self.device
            )
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (new_log_probs - old_log_probs).exp()
            actor_loss = -(ratio * adv_t).mean() - 0.01 * entropy

            self._actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
            self._actor_optim.step()

            # Update critic
            self._critic.train()
            v_pred = self._critic(states_t)
            critic_loss = nn.functional.mse_loss(v_pred, returns_t)

            self._critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 0.5)
            self._critic_optim.step()

            self._train_history.append({
                "episode": ep,
                "reward": ep_reward,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
            })

        self._trained = True

        return {
            "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "avg_shortfall": float(-np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episode_rewards": episode_rewards,
        }

    def benchmark_vs_twap(self, env: Any, num_episodes: int = 1000) -> dict:
        """Compare RL execution vs TWAP.

        Returns
        -------
        dict with rl_shortfall, twap_shortfall, improvement_pct.
        """
        rl_shortfalls = []
        twap_shortfalls = []

        for ep in range(num_episodes):
            seed = ep + 10000

            # RL execution
            state = env.reset(seed=seed)
            rl_pnl = 0.0
            done = False
            while not done:
                state_arr = state.to_array()[:self._state_dim]
                if self._actor is not None:
                    with torch.no_grad():
                        x = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
                        mean, _ = self._actor(x)
                        action_arr = mean.squeeze(0).cpu().numpy()
                else:
                    # Fallback: TWAP-like
                    action_arr = np.array([1.0 / max(env.num_steps, 1), 0.5])

                from .trading_env import TradingAction
                result = env.step(TradingAction(trade_sizes=action_arr))
                rl_pnl += result.reward
                done = result.done or result.truncated
                state = result.state

            rl_shortfalls.append(-rl_pnl)

            # TWAP execution
            state = env.reset(seed=seed)
            twap_pnl = 0.0
            done = False
            t = 0
            while not done:
                frac = 1.0 / max(env.num_steps - t, 1)
                from .trading_env import TradingAction
                result = env.step(TradingAction(trade_sizes=np.array([frac, 0.5])))
                twap_pnl += result.reward
                done = result.done or result.truncated
                state = result.state
                t += 1

            twap_shortfalls.append(-twap_pnl)

        rl_avg = float(np.mean(rl_shortfalls))
        twap_avg = float(np.mean(twap_shortfalls))
        improvement = (twap_avg - rl_avg) / max(abs(twap_avg), 1e-8) * 100.0

        return {
            "rl_shortfall": rl_avg,
            "twap_shortfall": twap_avg,
            "improvement_pct": improvement,
            "rl_std": float(np.std(rl_shortfalls, ddof=1)),
            "twap_std": float(np.std(twap_shortfalls, ddof=1)),
        }
