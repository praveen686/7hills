"""RL Trading Agent — Actor-Critic + KellySizer + ThompsonAllocator.

Combines three RL/decision components into a single multi-asset trading agent:

1. **Actor-Critic**: Gaussian policy (continuous actions per asset) with
   TD-bootstrapped value function (Ch 14.6 of Rao & Jelvis).
2. **KellySizer**: Drawdown-aware position magnitude constraints.
3. **ThompsonAllocator**: NIG Bayesian capital allocation across assets.

The agent receives a state vector composed of:
    [backbone_hidden(4×d_h), positions(4), norm_pnl(1), drawdown(1),
     heat(1), time_features(3)]

And outputs continuous position targets ∈ [-1, 1] per asset, constrained
by Kelly sizing and weighted by Thompson allocation.

References:
    - Policy Gradient Theorem: Ch 14.2, Sutton et al. (2000)
    - TD Actor-Critic: Ch 14.6, Rao & Jelvis
    - Kelly Criterion: Ch 8.1, Kelly (1956)
    - Thompson Sampling: Ch 15.5, Rao & Jelvis
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """RL hyperparameters for the integrated trading agent."""

    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_beta: float = 0.01
    num_episodes: int = 200
    max_steps_per_ep: int = 252
    reward_lambda_risk: float = 0.5
    reward_lambda_cost: float = 1.0
    max_grad_norm: float = 5.0


# ============================================================================
# PyTorch components
# ============================================================================

if _HAS_TORCH:

    class _ActorNetwork(nn.Module):
        """Maps state → Gaussian policy over per-asset position targets.

        Input: state_dim = 4×d_hidden + 10 (portfolio state features)
        Output: (n_assets × 2) — mean + log_std per asset
        Action sampling: Normal(mean, exp(log_std)), squashed through Tanh
        """

        def __init__(self, state_dim: int, n_assets: int) -> None:
            super().__init__()
            self.n_assets = n_assets
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ELU(),
            )
            self.mean_head = nn.Linear(128, n_assets)
            self.log_std_head = nn.Linear(128, n_assets)

        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return (mean, log_std) for Gaussian policy."""
            h = self.net(state)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-5.0, 2.0)
            return mean, log_std

    class _CriticNetwork(nn.Module):
        """Maps state → scalar value V(s)."""

        def __init__(self, state_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)

    class RLTradingAgent(nn.Module):
        """Full RL trading agent: Actor-Critic + KellySizer + ThompsonAllocator.

        Parameters
        ----------
        state_dim : int
            Dimension of the full state vector.
        n_assets : int
            Number of tradeable assets (e.g. 4 for India indices).
        d_hidden : int
            Backbone hidden dimension (for reference).
        kelly_cfg : dict
            Kwargs for KellySizer (mode, max_position_pct, etc.)
        thompson_names : list[str]
            Asset names for ThompsonAllocator.
        rl_cfg : RLConfig
            RL hyperparameters.
        """

        def __init__(
            self,
            state_dim: int,
            n_assets: int,
            d_hidden: int = 64,
            kelly_cfg: Optional[dict] = None,
            thompson_names: Optional[list[str]] = None,
            rl_cfg: Optional[RLConfig] = None,
        ) -> None:
            super().__init__()

            self.state_dim = state_dim
            self.n_assets = n_assets
            self.d_hidden = d_hidden
            self.rl_cfg = rl_cfg or RLConfig()

            # Actor-Critic networks
            self.actor = _ActorNetwork(state_dim, n_assets)
            self.critic = _CriticNetwork(state_dim)

            # Optimizers
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.rl_cfg.lr_actor
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.rl_cfg.lr_critic
            )

            # Kelly Sizer — position constraints
            from quantlaxmi.models.rl.agents.kelly_sizer import KellySizer

            kelly_params = kelly_cfg or {}
            kelly_params.setdefault("mode", "fractional_kelly")
            kelly_params.setdefault("max_position_pct", 0.25)
            kelly_params.setdefault("max_drawdown_pct", 0.20)
            self.kelly = KellySizer(**kelly_params)

            # Thompson Allocator — capital allocation across assets
            from quantlaxmi.models.rl.agents.thompson_allocator import ThompsonStrategyAllocator

            names = thompson_names or [f"asset_{i}" for i in range(n_assets)]
            self.thompson = ThompsonStrategyAllocator(
                strategy_names=names,
                context_dim=10,
                min_allocation=0.05,
                max_allocation=0.5,
            )

        def select_action(
            self,
            state_vec: np.ndarray,
            deterministic: bool = False,
            current_drawdown: float = 0.0,
            portfolio_heat: float = 0.0,
            expected_returns: Optional[np.ndarray] = None,
            volatilities: Optional[np.ndarray] = None,
        ) -> tuple[np.ndarray, float, float]:
            """Select multi-asset actions via Actor → Kelly → Thompson.

            Parameters
            ----------
            state_vec : (state_dim,) numpy array
            deterministic : bool
                If True, use mean (no sampling).
            current_drawdown : float
                Current portfolio drawdown fraction.
            portfolio_heat : float
                Current total absolute exposure.
            expected_returns : (n_assets,) optional
                Expected returns per asset for Kelly sizing.
            volatilities : (n_assets,) optional
                Volatility per asset for Kelly sizing.

            Returns
            -------
            actions : (n_assets,) — position targets ∈ [-1, 1]
            log_prob : float — log probability of the action
            value : float — critic's V(s) estimate
            """
            dev = next(self.actor.parameters()).device
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(dev)

            # 1. Actor outputs Gaussian policy parameters
            mean, log_std = self.actor(state_t)
            std = log_std.exp()

            if deterministic:
                raw_actions = torch.tanh(mean)
                log_prob_t = torch.tensor(0.0)
            else:
                dist = torch.distributions.Normal(mean, std)
                sample = dist.rsample()  # reparameterized
                raw_actions = torch.tanh(sample)

                # Log prob with Tanh squashing correction
                log_prob_t = dist.log_prob(sample) - torch.log(
                    1 - raw_actions.pow(2) + 1e-6
                )
                log_prob_t = log_prob_t.sum(dim=-1)

            # 2. Critic value estimate
            value = self.critic(state_t).squeeze()

            raw_np = raw_actions.squeeze(0).cpu().detach().numpy()

            # 3. Kelly sizing — constrain magnitude per asset
            constrained = np.zeros(self.n_assets)
            for i in range(self.n_assets):
                direction = np.sign(raw_np[i])
                magnitude = abs(raw_np[i])

                mu_i = expected_returns[i] if expected_returns is not None else 0.10
                vol_i = volatilities[i] if volatilities is not None else 0.15

                kelly_size = self.kelly.optimal_size(
                    expected_return=mu_i,
                    volatility=vol_i,
                    current_drawdown=current_drawdown,
                    portfolio_heat=portfolio_heat,
                )
                # Scale actor magnitude by Kelly constraint
                constrained_mag = min(magnitude, kelly_size)
                constrained[i] = direction * constrained_mag

            # 4. Thompson allocation — weight capital across assets
            context = np.zeros(10)  # minimal context
            context[0] = current_drawdown
            context[1] = portfolio_heat
            alloc_weights = self.thompson.select_allocation(context)
            alloc_array = np.array(
                [alloc_weights.get(n, 1.0 / self.n_assets)
                 for n in self.thompson.strategy_names]
            )

            # Apply allocation weights to constrained actions
            actions = constrained * alloc_array * self.n_assets  # scale so sum(alloc)=1 doesn't shrink
            actions = np.clip(actions, -1.0, 1.0)

            return actions, float(log_prob_t.item()), float(value.item())

        def update(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray,
        ) -> tuple[float, float]:
            """TD Actor-Critic update (Ch 14.6).

            δ_t = r + γ·V(s') − V(s)
            actor_loss = −log π(a|s) · δ_t − β·H(π)
            critic_loss = δ_t²

            Parameters
            ----------
            states : (batch, state_dim)
            actions : (batch, n_assets)
            rewards : (batch,)
            next_states : (batch, state_dim)
            dones : (batch,) — 1.0 if terminal

            Returns
            -------
            (actor_loss, critic_loss)
            """
            cfg = self.rl_cfg
            dev = next(self.actor.parameters()).device

            states_t = torch.FloatTensor(states).to(dev)
            actions_t = torch.FloatTensor(actions).to(dev)
            rewards_t = torch.FloatTensor(rewards).to(dev)
            next_states_t = torch.FloatTensor(next_states).to(dev)
            dones_t = torch.FloatTensor(dones).to(dev)

            # Critic values
            values = self.critic(states_t).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states_t).squeeze(-1)

            # TD target: r + γ·V(s')·(1 - done)
            td_targets = rewards_t + cfg.gamma * next_values * (1.0 - dones_t)
            advantages = (td_targets - values).detach()

            # Recompute log probs for the taken actions
            mean, log_std = self.actor(states_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            # Inverse tanh (atanh) to get pre-squash samples
            actions_clamped = actions_t.clamp(-0.999, 0.999)
            pre_squash = torch.atanh(actions_clamped)

            log_probs = dist.log_prob(pre_squash) - torch.log(
                1 - actions_clamped.pow(2) + 1e-6
            )
            log_probs = log_probs.sum(dim=-1)

            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1)

            # Actor loss: -E[log π · advantage + β · H(π)]
            actor_loss = -(log_probs * advantages + cfg.entropy_beta * entropy).mean()

            # Critic loss: MSE of TD error
            critic_loss = F.mse_loss(values, td_targets.detach())

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()

            return actor_loss.item(), critic_loss.item()

        def update_thompson(self, asset_returns: dict[str, float]) -> None:
            """Update Thompson posteriors with realized daily returns.

            Parameters
            ----------
            asset_returns : dict mapping asset_name → daily_return
            """
            context = np.zeros(10)
            for name, ret in asset_returns.items():
                self.thompson.update(name, ret, context)

else:
    class RLTradingAgent:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for RLTradingAgent")

    @dataclass
    class RLConfig:  # type: ignore[no-redef]
        pass
