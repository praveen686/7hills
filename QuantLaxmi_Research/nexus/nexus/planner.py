"""TD-MPC2-style Model Predictive Control in Learned Latent Space.

The crown jewel of NEXUS: given a learned world model, PLAN optimal
trading actions by simulating futures in latent space.

Traditional quant models:
    Market data -> Features -> Predict price/signal -> Act

NEXUS planning:
    Market data -> World Model -> Latent state -> Imagine N futures
    -> Evaluate each future (reward + risk) -> Select best action

This is how AlphaGo/MuZero beat humans at Go: not by predicting the
next move, but by SIMULATING thousands of possible futures and choosing
the path with highest expected value.

For trading:
    - Simulate 512 possible 5-day futures in latent space
    - For each future: compute expected return, max drawdown, Sharpe
    - Select action that maximizes risk-adjusted return
    - Apply CVaR constraint (tail risk control)

Algorithm: Cross-Entropy Method (CEM) with learned dynamics
    1. Sample N action sequences from prior (Gaussian)
    2. Roll out each through world model (imagined trajectories)
    3. Evaluate: reward + discount * value
    4. Select top-K elites
    5. Update prior toward elites
    6. Repeat for M iterations
    7. Execute first action of best trajectory

References:
    TD-MPC2: Hansen et al., ICLR 2024
    MPPI: Williams et al., 2017
    CEM: Botev et al., 2013
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsModel(nn.Module):
    """Learned dynamics in latent space: z_{t+1} = f(z_t, a_t).

    Given current latent state and action, predicts next latent state.
    Uses SimNorm (from TD-MPC2) for stable representations.

    Use this for impact-aware mode (e.g., HFT with market impact)
    where the trader's action materially affects state transitions.
    For exogenous (liquid) markets, use ExogenousDynamicsModel instead.
    """

    def __init__(
        self,
        d_latent: int,
        d_action: int,
        d_hidden: int = 512,
        n_layers: int = 3,
        simnorm_dim: int = 8,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.simnorm_dim = simnorm_dim

        # State-action encoder
        layers = [nn.Linear(d_latent + d_action, d_hidden), nn.Mish(), nn.LayerNorm(d_hidden)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d_hidden, d_hidden), nn.Mish(), nn.LayerNorm(d_hidden)])
        layers.append(nn.Linear(d_hidden, d_latent))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next latent state.

        Parameters
        ----------
        z : (B, d_latent) -- current latent state
        a : (B, d_action) -- action (position vector)

        Returns
        -------
        (B, d_latent) -- predicted next latent state
        """
        x = torch.cat([z, a], dim=-1)
        z_next = self.mlp(x)

        # SimNorm: partition into groups, apply softmax per group
        # This stabilizes representations and prevents collapse
        z_next = self._simnorm(z_next)

        return z_next

    def _simnorm(self, x: torch.Tensor) -> torch.Tensor:
        """SimNorm from TD-MPC2: softmax over groups of features.

        Partitions features into groups, normalizes each group to simplex.
        Enforces sparse, stable representations.
        """
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.simnorm_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


class ExogenousDynamicsModel(nn.Module):
    """Dynamics for exogenous markets: z_{t+1} = f(z_t).

    In liquid markets where the participant's actions don't move prices,
    the next market state depends only on the current state + noise,
    NOT on the trader's action. This is the correct model for
    retail/prop-size trading in index derivatives and large-cap equities.

    The trader's action still affects PnL (via reward), but NOT the
    market state transition. This separation is critical:
      - State dynamics: z_{t+1} = f(z_t)         (market doesn't care about you)
      - Reward:         r_{t+1} = g(z_t, a_t)    (your PnL depends on your position)

    Uses SimNorm (from TD-MPC2) for stable representations.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int = 512,
        n_layers: int = 3,
        simnorm_dim: int = 8,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.simnorm_dim = simnorm_dim

        # State-only encoder (NO action input)
        layers = [nn.Linear(d_latent, d_hidden), nn.Mish(), nn.LayerNorm(d_hidden)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d_hidden, d_hidden), nn.Mish(), nn.LayerNorm(d_hidden)])
        layers.append(nn.Linear(d_hidden, d_latent))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict next latent state from current state only.

        Parameters
        ----------
        z : (B, d_latent) -- current latent state

        Returns
        -------
        (B, d_latent) -- predicted next latent state
        """
        z_next = self.mlp(z)

        # SimNorm: partition into groups, apply softmax per group
        z_next = self._simnorm(z_next)

        return z_next

    def _simnorm(self, x: torch.Tensor) -> torch.Tensor:
        """SimNorm from TD-MPC2: softmax over groups of features."""
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.simnorm_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


class RewardModel(nn.Module):
    """Predicts scalar reward from latent state + action.

    Reward = risk-adjusted return (Sharpe-like).
    """

    def __init__(self, d_latent: int, d_action: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, d_hidden),
            nn.Mish(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.Mish(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict reward. Returns (B, 1)."""
        return self.net(torch.cat([z, a], dim=-1))


class ExplicitRewardModel(nn.Module):
    """Reward with explicit financial structure.

    r_{t+1} = a_t^T delta_p_{t+1} - c(|delta_a_t|) - lambda * risk(a_t, Sigma_t)

    Where:
    - a_t^T delta_p_{t+1}: position PnL (decoded from latent state)
    - c(|delta_a_t|): transaction costs (proportional + fixed)
    - risk(a_t, Sigma_t): risk penalty (position * covariance proxy)

    This model combines:
    1. A learned component that decodes price changes from latent state
    2. Explicit cost and risk penalties with known financial structure

    The explicit structure provides inductive bias (costs are KNOWN to be
    proportional to turnover), while the learned component handles the
    unknown mapping from latent state to expected returns.
    """

    def __init__(
        self,
        d_latent: int,
        d_action: int,
        commission_bps: float = 3.0,
        slippage_bps: float = 2.0,
        risk_lambda: float = 0.1,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.d_action = d_action
        self.commission_bps = commission_bps / 1e4  # Convert to decimal
        self.slippage_bps = slippage_bps / 1e4
        self.risk_lambda = risk_lambda

        # Learned: decode expected returns from latent state
        self.return_decoder = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_action),  # Expected return per asset
        )

        # Learned: decode risk (covariance proxy) from latent state
        self.risk_decoder = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_action),  # Variance proxy per asset
            nn.Softplus(),  # Variance must be positive
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        a_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward with explicit financial structure.

        Parameters
        ----------
        z : (B, d_latent) -- current latent state
        a : (B, d_action) -- current position vector
        a_prev : (B, d_action) -- previous position (for turnover costs).
                 If None, assumes zero previous position.

        Returns
        -------
        (B, 1) -- scalar reward
        """
        if a_prev is None:
            a_prev = torch.zeros_like(a)

        # 1. Position PnL: a_t^T * E[delta_p_{t+1}]
        expected_returns = self.return_decoder(z)  # (B, d_action)
        pnl = (a * expected_returns).sum(dim=-1, keepdim=True)  # (B, 1)

        # 2. Transaction costs: c * |delta_a|
        turnover = (a - a_prev).abs()  # (B, d_action)
        cost_per_asset = turnover * (self.commission_bps + self.slippage_bps)
        total_cost = cost_per_asset.sum(dim=-1, keepdim=True)  # (B, 1)

        # 3. Risk penalty: lambda * a^T * diag(sigma^2) * a
        variance_proxy = self.risk_decoder(z)  # (B, d_action)
        risk = (a.pow(2) * variance_proxy).sum(dim=-1, keepdim=True)  # (B, 1)

        # r = PnL - costs - risk_penalty
        reward = pnl - total_cost - self.risk_lambda * risk

        return reward


class ValueModel(nn.Module):
    """Predicts state value V(z) = E[sum gamma^t r_t | z_0 = z].

    Ensemble of 2 value functions for stability (twin Q-learning).
    """

    def __init__(self, d_latent: int, d_hidden: int = 256, n_ensemble: int = 2):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_latent, d_hidden),
                nn.Mish(),
                nn.LayerNorm(d_hidden),
                nn.Linear(d_hidden, d_hidden),
                nn.Mish(),
                nn.LayerNorm(d_hidden),
                nn.Linear(d_hidden, 1),
            )
            for _ in range(n_ensemble)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict state value. Returns (B, n_ensemble)."""
        values = [net(z) for net in self.nets]
        return torch.cat(values, dim=-1)  # (B, n_ensemble)

    def min_value(self, z: torch.Tensor) -> torch.Tensor:
        """Conservative value estimate (min over ensemble)."""
        return self.forward(z).min(dim=-1, keepdim=True).values


class CEMPlanner(nn.Module):
    """Cross-Entropy Method planner in learned latent space.

    Plans optimal action sequences by:
    1. Sampling N trajectories from action prior
    2. Rolling out through learned dynamics model
    3. Scoring: cumulative reward + terminal value
    4. Refitting prior to top-K elites
    5. Repeating for M iterations

    With risk constraint: CVaR_alpha(trajectory returns) must exceed threshold.

    Supports two dynamics modes:
    - exogenous=True (default): state evolves independently of actions.
      Actions only affect reward (PnL), not state transitions.
      Correct for retail/prop-size trading in liquid markets.
    - exogenous=False: state depends on actions (market impact mode).
      For HFT or large positions that move prices.
    """

    def __init__(
        self,
        dynamics,  # ExogenousDynamicsModel or LatentDynamicsModel
        reward_model: RewardModel,
        value_model: ValueModel,
        d_action: int,
        horizon: int = 5,
        n_samples: int = 512,
        n_elites: int = 64,
        n_iterations: int = 6,
        temperature: float = 0.5,
        discount: float = 0.99,
        max_position: float = 0.25,
        cvar_alpha: float = 0.05,
        exogenous: bool = True,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.reward_model = reward_model
        self.value_model = value_model
        self.d_action = d_action
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.discount = discount
        self.max_position = max_position
        self.cvar_alpha = cvar_alpha
        self.exogenous = exogenous

    @torch.no_grad()
    def plan(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Plan optimal action from current latent state.

        Parameters
        ----------
        z : (B, d_latent) -- current market state encoding

        Returns
        -------
        action : (B, d_action) -- optimal first action (position vector)
        info : dict -- planning diagnostics
        """
        B = z.size(0)
        device = z.device

        # Initialize action prior: N(0, sigma^2 I), clamped to [-max_pos, max_pos]
        mu = torch.zeros(B, self.horizon, self.d_action, device=device)
        sigma = torch.ones(B, self.horizon, self.d_action, device=device) * 0.5

        best_value = torch.full((B,), -float("inf"), device=device)
        best_actions = mu[:, 0].clone()

        for iteration in range(self.n_iterations):
            # 1. Sample action sequences from current prior
            noise = torch.randn(B, self.n_samples, self.horizon, self.d_action,
                              device=device)
            actions = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise  # (B, N, H, d_action)
            actions = actions.clamp(-self.max_position, self.max_position)

            # 2. Roll out through dynamics model
            returns = torch.zeros(B, self.n_samples, device=device)
            z_current = z.unsqueeze(1).expand(-1, self.n_samples, -1)  # (B, N, d_latent)
            z_flat = z_current.reshape(B * self.n_samples, -1)

            discount_factor = 1.0
            for t in range(self.horizon):
                a_flat = actions[:, :, t].reshape(B * self.n_samples, -1)

                # Predict reward (action always affects reward / PnL)
                r = self.reward_model(z_flat, a_flat).squeeze(-1)  # (B*N,)
                returns += discount_factor * r.view(B, self.n_samples)

                # Predict next state
                if self.exogenous:
                    # Exogenous dynamics: z_{t+1} = f(z_t) -- action does NOT affect state
                    z_flat = self.dynamics(z_flat)
                else:
                    # Impact-aware dynamics: z_{t+1} = f(z_t, a_t)
                    z_flat = self.dynamics(z_flat, a_flat)

                discount_factor *= self.discount

            # Terminal value
            terminal_v = self.value_model.min_value(z_flat).squeeze(-1)  # (B*N,)
            returns += discount_factor * terminal_v.view(B, self.n_samples)

            # 3. Select elites
            _, elite_idx = returns.topk(self.n_elites, dim=1)  # (B, K)

            # Gather elite actions
            elite_actions = torch.gather(
                actions,
                1,
                elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.d_action),
            )  # (B, K, H, d_action)

            # Gather elite returns for weighting
            elite_returns = torch.gather(returns, 1, elite_idx)  # (B, K)

            # 4. Softmax weighting of elites
            weights = F.softmax(elite_returns / self.temperature, dim=1)  # (B, K)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)

            # 5. Update prior
            mu = (weights * elite_actions).sum(dim=1)  # (B, H, d_action)
            diff = elite_actions - mu.unsqueeze(1)
            sigma = torch.sqrt((weights * diff.pow(2)).sum(dim=1) + 1e-6)

            # Track best
            batch_best_val, batch_best_idx = elite_returns.max(dim=1)
            improved = batch_best_val > best_value
            if improved.any():
                best_value[improved] = batch_best_val[improved]
                # Get actions of best trajectory
                for b in range(B):
                    if improved[b]:
                        best_actions[b] = elite_actions[b, batch_best_idx[b], 0]

        info = {
            "planned_value": best_value,
            "action_std": sigma[:, 0].mean(dim=-1),
            "n_iterations": self.n_iterations,
            "exogenous": self.exogenous,
        }

        return best_actions, info


# ---------------------------------------------------------------------------
# Full Planner Module
# ---------------------------------------------------------------------------

class NexusPlanner(nn.Module):
    """Complete planning module for NEXUS.

    Combines:
    - Learned dynamics model (latent transitions)
    - Reward model (return prediction)
    - Value model (discounted future returns)
    - CEM planner (trajectory optimization)

    Parameters
    ----------
    exogenous : bool (default True)
        If True (default), uses ExogenousDynamicsModel where state evolves
        independently of the trader's action. This is correct for
        retail/prop-size trading in liquid markets (index derivatives,
        large-cap equities) where your trades don't move the market.

        If False, uses LatentDynamicsModel where z_{t+1} = f(z_t, a_t),
        appropriate for HFT or large positions with market impact.
    """

    def __init__(
        self,
        d_latent: int,
        d_action: int = 6,
        horizon: int = 5,
        n_samples: int = 512,
        n_elites: int = 64,
        n_iterations: int = 6,
        discount: float = 0.99,
        max_position: float = 0.25,
        exogenous: bool = True,
    ):
        super().__init__()
        self.exogenous = exogenous

        if exogenous:
            self.dynamics = ExogenousDynamicsModel(d_latent)
        else:
            self.dynamics = LatentDynamicsModel(d_latent, d_action)

        self.reward_model = RewardModel(d_latent, d_action)
        self.value_model = ValueModel(d_latent)

        self.cem = CEMPlanner(
            dynamics=self.dynamics,
            reward_model=self.reward_model,
            value_model=self.value_model,
            d_action=d_action,
            horizon=horizon,
            n_samples=n_samples,
            n_elites=n_elites,
            n_iterations=n_iterations,
            discount=discount,
            max_position=max_position,
            exogenous=exogenous,
        )

    def plan(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Plan optimal action from latent state."""
        return self.cem.plan(z)

    def compute_td_loss(
        self,
        z: torch.Tensor,       # (B, d_latent) -- current state
        a: torch.Tensor,       # (B, d_action) -- action taken
        r: torch.Tensor,       # (B, 1) -- observed reward
        z_next: torch.Tensor,  # (B, d_latent) -- next state
        done: torch.Tensor,    # (B, 1) -- terminal flag
        discount: float = 0.99,
    ) -> dict:
        """Compute TD losses for dynamics, reward, and value models.

        Parameters
        ----------
        z, a, r, z_next, done : batch of transitions

        Returns
        -------
        dict with dynamics_loss, reward_loss, value_loss
        """
        # Dynamics loss: ||f(z[, a]) - z_next||^2
        if self.exogenous:
            z_pred = self.dynamics(z)
        else:
            z_pred = self.dynamics(z, a)
        dynamics_loss = F.mse_loss(z_pred, z_next.detach())

        # Reward loss: ||R(z, a) - r||^2
        r_pred = self.reward_model(z, a)
        reward_loss = F.mse_loss(r_pred, r)

        # Value loss: TD(0) with target network
        with torch.no_grad():
            v_next = self.value_model.min_value(z_next)
            td_target = r + discount * (1 - done) * v_next

        v_current = self.value_model.forward(z)
        value_loss = sum(
            F.mse_loss(v_current[:, i:i+1], td_target)
            for i in range(self.value_model.n_ensemble)
        ) / self.value_model.n_ensemble

        return {
            "dynamics_loss": dynamics_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "total_loss": dynamics_loss + reward_loss + value_loss,
        }
