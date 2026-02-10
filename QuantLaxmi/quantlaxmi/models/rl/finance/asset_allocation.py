"""Chapter 8: Dynamic Asset Allocation

Merton's Portfolio Problem solved with both analytical and RL methods.

The fundamental problem: An investor has wealth W_t and must choose portfolio
weights pi_t at each time step to maximize expected terminal utility E[U(W_T)].

Wealth dynamics (continuous-time, discretized):
    W_{t+1} = W_t * (1 + pi_t^T * r_{t+1} + (1 - pi_t^T * 1) * r_f * dt)

where:
    r_{t+1} ~ N(mu * dt, Sigma * dt)  for N risky assets
    r_f is the risk-free rate
    pi_t is the vector of portfolio weights

Analytical Solution (CRRA utility, GBM dynamics):
    Single asset:  pi* = (mu - r) / (gamma * sigma^2)
    Multi-asset:   pi* = (1/gamma) * Sigma^{-1} * (mu - r*1)

    Optimal value function (CRRA, U(W) = W^{1-gamma}/(1-gamma)):
    V(t, W) = -(W^{1-gamma}/(1-gamma)) * exp(-(1-gamma)*[r + (mu-r)^2/(2*gamma*sigma^2)]*(T-t))

References:
    Rao & Jelvis, Ch 8 -- "Dynamic Asset Allocation and Consumption"
    Merton (1969), "Lifetime Portfolio Selection Under Uncertainty"
    Merton (1971), "Optimum Consumption and Portfolio Rules in a Continuous-Time Model"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Graceful imports from sibling packages
# ---------------------------------------------------------------------------
try:
    from quantlaxmi.models.rl.core.markov_process import (
        MarkovDecisionProcess,
        NonTerminal,
        Terminal,
        State,
        Distribution,
        SampledDistribution,
        Categorical,
        Constant,
    )
except ImportError:
    pass

try:
    from quantlaxmi.models.rl.core.dynamic_programming import ValueFunction
except ImportError:
    pass

# PyTorch -- optional but required for RL-based solvers
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal as TorchNormal

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

__all__ = [
    "AssetAllocationMDP",
    "MertonSolution",
    "AssetAllocPG",
    "MultiStrategyAllocator",
]


# ============================================================================
# Utility functions
# ============================================================================


def _crra_utility(wealth: float, gamma: float) -> float:
    """CRRA (Constant Relative Risk Aversion) utility.

    U(W) = W^{1-gamma} / (1-gamma)   if gamma != 1
    U(W) = log(W)                      if gamma == 1

    CRRA parameter gamma is the coefficient of relative risk aversion:
        -W * U''(W) / U'(W) = gamma
    """
    if wealth <= 0:
        return float('-inf')  # ruin: utility is -∞
    if abs(gamma - 1.0) < 1e-8:
        return math.log(wealth)
    return (wealth ** (1.0 - gamma)) / (1.0 - gamma)


def _cara_utility(wealth: float, alpha: float) -> float:
    """CARA (Constant Absolute Risk Aversion) utility.

    U(W) = -exp(-alpha * W) / alpha

    CARA parameter alpha is the coefficient of absolute risk aversion:
        -U''(W) / U'(W) = alpha
    """
    return -math.exp(-alpha * wealth) / alpha


def _get_device(device: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# ============================================================================
# AssetAllocationMDP -- Chapter 8
# ============================================================================


@dataclass
class AssetAllocState:
    """State for the asset allocation MDP.

    Attributes:
        time_step: current time step t in [0, T]
        wealth: current wealth W_t
        features: optional market features (regime, vol, etc.)
    """
    time_step: int
    wealth: float
    features: Optional[np.ndarray] = None

    def __hash__(self) -> int:
        return hash((self.time_step, round(self.wealth, 6)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssetAllocState):
            return NotImplemented
        return (self.time_step == other.time_step
                and abs(self.wealth - other.wealth) < 1e-8)


class AssetAllocationMDP:
    """Chapter 8: Asset allocation as a Markov Decision Process.

    State: (time_step, wealth, [market_features])
    Action: portfolio weights vector pi in R^N (summing to <=1 for no leverage)
    Transition:
        r_{t+1} ~ N(mu * dt, Sigma * dt)
        W_{t+1} = W_t * (1 + pi^T * r_{t+1} + (1 - sum(pi)) * r_f * dt)
    Reward: 0 at intermediate steps, U(W_T) at terminal step

    The key insight from Ch 8: this MDP has continuous state and action spaces,
    so we need function approximation (neural networks) to solve it in general.

    Parameters
    ----------
    num_assets : int
        Number of risky assets.
    mu : np.ndarray, shape (num_assets,)
        Expected returns vector (annualized).
    sigma : np.ndarray, shape (num_assets, num_assets)
        Covariance matrix (annualized). For single asset, can pass shape (1,1).
    risk_free_rate : float
        Annualized risk-free rate.
    num_steps : int
        Number of trading periods (e.g. 252 for daily over 1 year).
    utility : str
        "crra" or "cara".
    gamma_risk : float
        Risk aversion coefficient (gamma for CRRA, alpha for CARA).
    dt : float
        Time step size in years (1/252 for daily).
    initial_wealth : float
        Starting wealth W_0.
    """

    def __init__(
        self,
        num_assets: int,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: float = 0.05,
        num_steps: int = 252,
        utility: str = "crra",
        gamma_risk: float = 2.0,
        dt: float = 1.0 / 252,
        initial_wealth: float = 1.0,
    ) -> None:
        self.num_assets = num_assets
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        assert self.mu.shape[0] == num_assets, (
            f"mu shape {self.mu.shape} doesn't match num_assets={num_assets}"
        )

        self.sigma = np.asarray(sigma, dtype=np.float64)
        if self.sigma.ndim == 1:
            # Single asset: sigma is a scalar variance, reshape to (1,1)
            self.sigma = self.sigma.reshape(1, 1)
        assert self.sigma.shape == (num_assets, num_assets), (
            f"sigma shape {self.sigma.shape} doesn't match ({num_assets},{num_assets})"
        )

        self.risk_free_rate = risk_free_rate
        self.num_steps = num_steps
        self.utility = utility
        self.gamma_risk = gamma_risk
        self.dt = dt
        self.initial_wealth = initial_wealth

        # Precompute Cholesky for correlated asset returns
        self._cholesky = np.linalg.cholesky(self.sigma)

    def utility_fn(self, wealth: float) -> float:
        """Compute utility of terminal wealth."""
        if self.utility == "crra":
            return _crra_utility(wealth, self.gamma_risk)
        elif self.utility == "cara":
            return _cara_utility(wealth, self.gamma_risk)
        else:
            raise ValueError(f"Unknown utility: {self.utility}")

    def sample_returns(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one-period asset returns from multivariate normal.

        r_{t+1} ~ N(mu * dt, Sigma * dt)
        """
        z = rng.standard_normal(self.num_assets)
        returns = self.mu * self.dt + math.sqrt(self.dt) * self._cholesky @ z
        return returns

    def step(
        self,
        state: AssetAllocState,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[AssetAllocState, float, bool]:
        """Execute one step of the MDP.

        Parameters
        ----------
        state : AssetAllocState
            Current state.
        weights : np.ndarray, shape (num_assets,)
            Portfolio weights for risky assets. (1 - sum(weights)) goes to risk-free.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        next_state : AssetAllocState
        reward : float
            0 for intermediate steps, U(W_T) at terminal.
        done : bool
            True if terminal step reached.
        """
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert weights.shape[0] == self.num_assets

        # Sample asset returns for this period
        asset_returns = self.sample_returns(rng)

        # Portfolio return: weighted sum of risky + risk-free
        # r_p = pi^T * r_risky + (1 - sum(pi)) * r_f * dt
        risky_return = weights @ asset_returns
        rf_return = (1.0 - weights.sum()) * self.risk_free_rate * self.dt
        portfolio_return = risky_return + rf_return

        # Wealth evolution: W_{t+1} = W_t * (1 + r_p)
        new_wealth = state.wealth * (1.0 + portfolio_return)
        new_wealth = max(new_wealth, 1e-10)  # prevent negative wealth

        next_time = state.time_step + 1
        done = next_time >= self.num_steps

        # Reward: 0 at intermediate, U(W_T) at terminal
        reward = self.utility_fn(new_wealth) if done else 0.0

        next_state = AssetAllocState(
            time_step=next_time,
            wealth=new_wealth,
            features=state.features,
        )
        return next_state, reward, done

    def simulate_episode(
        self,
        policy_fn: Callable[[AssetAllocState], np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[List[AssetAllocState], List[np.ndarray], List[float]]:
        """Simulate a full episode using the given policy.

        Returns
        -------
        states : list of AssetAllocState
        actions : list of weight vectors
        rewards : list of float
        """
        state = AssetAllocState(time_step=0, wealth=self.initial_wealth)
        states: List[AssetAllocState] = [state]
        actions: List[np.ndarray] = []
        rewards: List[float] = []

        done = False
        while not done:
            weights = policy_fn(state)
            next_state, reward, done = self.step(state, weights, rng)
            actions.append(weights)
            rewards.append(reward)
            states.append(next_state)
            state = next_state

        return states, actions, rewards

    def state_to_features(self, state: AssetAllocState) -> np.ndarray:
        """Convert state to feature vector for neural network input.

        Features: [time_remaining/T, log(wealth/W0)]
        Optionally appended with external market features.
        """
        time_frac = (self.num_steps - state.time_step) / self.num_steps
        log_wealth = math.log(max(state.wealth, 1e-10) / self.initial_wealth)
        base_features = np.array([time_frac, log_wealth], dtype=np.float64)

        if state.features is not None:
            return np.concatenate([base_features, state.features])
        return base_features


# ============================================================================
# MertonSolution -- Analytical benchmark
# ============================================================================


class MertonSolution:
    """Analytical solution for Merton's portfolio problem (benchmark).

    For CRRA utility U(W) = W^{1-gamma}/(1-gamma) with GBM asset dynamics:

    Single risky asset:
        pi* = (mu - r) / (gamma * sigma^2)

    Multiple risky assets:
        pi* = (1/gamma) * Sigma^{-1} * (mu - r*1)

    Optimal value function (CRRA, single asset, continuous-time):
        V(t, W) = (W^{1-gamma} / (1-gamma))
                   * exp((1-gamma) * [r + (mu-r)^2 / (2*gamma*sigma^2)] * (T-t))

    For CARA utility U(W) = -exp(-alpha*W) / alpha:
        pi* = (mu - r) / (alpha * sigma^2)  [single asset]

    References:
        Merton (1969), Eqs. 18-21
        Rao & Jelvis Ch 8, Theorem 8.1
    """

    @staticmethod
    def optimal_weights(
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_free_rate: float,
        gamma_risk: float,
        utility: str = "crra",
    ) -> np.ndarray:
        """Compute optimal portfolio weights.

        Parameters
        ----------
        mu : np.ndarray, shape (N,)
            Expected returns (annualized).
        sigma : np.ndarray, shape (N, N)
            Covariance matrix (annualized).
        risk_free_rate : float
            Risk-free rate.
        gamma_risk : float
            Risk aversion (gamma for CRRA, alpha for CARA).
        utility : str
            "crra" or "cara".

        Returns
        -------
        np.ndarray, shape (N,)
            Optimal portfolio weights for risky assets.
        """
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.ndim == 1:
            sigma = sigma.reshape(1, 1)

        n = mu.shape[0]
        excess_return = mu - risk_free_rate * np.ones(n)

        # pi* = (1/gamma) * Sigma^{-1} * (mu - r*1)
        sigma_inv = np.linalg.inv(sigma)
        weights = (1.0 / gamma_risk) * sigma_inv @ excess_return

        return weights

    @staticmethod
    def optimal_value(
        wealth: float,
        time: float,
        T: float,
        mu: float,
        sigma_scalar: float,
        r: float,
        gamma: float,
    ) -> float:
        """Optimal value function for single-asset CRRA case.

        V(t, W) = (W^{1-gamma} / (1-gamma))
                   * exp((1-gamma) * [r + (mu-r)^2 / (2*gamma*sigma^2)] * (T-t))

        Parameters
        ----------
        wealth : float
            Current wealth W.
        time : float
            Current time t.
        T : float
            Terminal time.
        mu : float
            Expected return (annualized).
        sigma_scalar : float
            Volatility (annualized standard deviation).
        r : float
            Risk-free rate.
        gamma : float
            CRRA coefficient.

        Returns
        -------
        float
            Optimal value V(t, W).
        """
        if wealth <= 0:
            return -1e10
        if abs(gamma - 1.0) < 1e-8:
            # Log utility: V(t,W) = log(W) + [r + (mu-r)^2/(2*sigma^2)]*(T-t)
            growth_rate = r + (mu - r) ** 2 / (2.0 * sigma_scalar ** 2)
            return math.log(wealth) + growth_rate * (T - time)

        tau = T - time
        sigma_sq = sigma_scalar ** 2
        exponent = (1.0 - gamma) * (r + (mu - r) ** 2 / (2.0 * gamma * sigma_sq)) * tau
        return (wealth ** (1.0 - gamma) / (1.0 - gamma)) * math.exp(exponent)

    @staticmethod
    def certainty_equivalent_return(
        mu: float,
        sigma_scalar: float,
        r: float,
        gamma: float,
    ) -> float:
        """Certainty equivalent excess return for the Merton portfolio.

        CE = r + (mu - r)^2 / (2 * gamma * sigma^2)

        This is the growth rate of certainty equivalent wealth.
        """
        return r + (mu - r) ** 2 / (2.0 * gamma * sigma_scalar ** 2)


# ============================================================================
# AssetAllocPG -- Policy Gradient solution (Ch 8 + Ch 14)
# ============================================================================


class AssetAllocPG:
    """Policy Gradient solution for Dynamic Asset Allocation.

    When the analytical Merton solution doesn't exist (regime changes,
    transaction costs, constraints, non-GBM dynamics), we use RL.

    Architecture:
        Policy network: state -> Gaussian(mu, sigma) for each asset weight
        Value network: state -> scalar (expected utility baseline)
        Algorithm: Actor-Critic with GAE

    The policy outputs parameters of a Gaussian distribution over weights.
    For long-only constraints, we apply a sigmoid/softmax transform.
    For unconstrained, we use tanh scaled by leverage_limit.

    Training uses the REINFORCE with baseline (advantage actor-critic):
        grad J = E[sum_t A_t * grad log pi(a_t|s_t)]
    where A_t = G_t - V(s_t) is the advantage.

    References:
        Rao & Jelvis Ch 8 (problem formulation) + Ch 14 (policy gradient)
        Buehler et al. (2019), "Deep Hedging" (architecture inspiration)
    """

    def __init__(
        self,
        num_assets: int,
        state_dim: int = 2,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        utility: str = "crra",
        gamma_risk: float = 2.0,
        leverage_limit: float = 1.0,
        device: str = "auto",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for AssetAllocPG")

        self.num_assets = num_assets
        self.state_dim = state_dim
        self.gamma = gamma
        self.utility = utility
        self.gamma_risk = gamma_risk
        self.leverage_limit = leverage_limit
        self.device = _get_device(device)

        # --- Policy network ---
        # Outputs: mean and log_std for each asset weight
        policy_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            policy_layers.append(nn.Linear(prev_dim, h))
            policy_layers.append(nn.ReLU())
            prev_dim = h
        self.policy_body = nn.Sequential(*policy_layers).to(self.device)
        self.policy_mean = nn.Linear(prev_dim, num_assets).to(self.device)
        self.policy_log_std = nn.Parameter(
            torch.zeros(num_assets, device=self.device) - 0.5
        )

        # --- Value network ---
        value_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            value_layers.append(nn.Linear(prev_dim, h))
            value_layers.append(nn.ReLU())
            prev_dim = h
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers).to(self.device)

        # Optimizers
        policy_params = (
            list(self.policy_body.parameters())
            + list(self.policy_mean.parameters())
            + [self.policy_log_std]
        )
        self.policy_optimizer = optim.Adam(policy_params, lr=learning_rate)
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )

        self._trained = False

    def _get_policy_distribution(
        self, state_tensor: torch.Tensor
    ) -> TorchNormal:
        """Get the Gaussian policy distribution for a state batch.

        Parameters
        ----------
        state_tensor : torch.Tensor, shape (batch, state_dim)

        Returns
        -------
        Normal distribution over weights, shape (batch, num_assets)
        """
        features = self.policy_body(state_tensor)
        mean = self.policy_mean(features)
        # Clamp mean to leverage_limit
        mean = torch.tanh(mean) * self.leverage_limit
        std = torch.exp(self.policy_log_std.clamp(-5, 2)).expand_as(mean)
        return TorchNormal(mean, std)

    def _select_action(self, state_features: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action and return (weights, log_prob)."""
        state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        dist = self._get_policy_distribution(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        weights = action.squeeze(0).detach().cpu().numpy()
        return weights, log_prob.squeeze(0)

    def train(
        self,
        env: AssetAllocationMDP,
        num_episodes: int = 10000,
        verbose: bool = True,
        seed: int = 42,
    ) -> dict:
        """Train the policy using Actor-Critic.

        Algorithm (per episode):
            1. Collect trajectory: (s_0, a_0, r_0, ..., s_T, r_T)
            2. Compute returns: G_t = sum_{k=t}^T gamma^{k-t} * r_k
            3. Compute advantages: A_t = G_t - V(s_t)
            4. Policy loss: -sum_t A_t * log pi(a_t|s_t)
            5. Value loss: sum_t (G_t - V(s_t))^2
            6. Update both networks

        Returns
        -------
        dict with keys: episode_utilities, policy_losses, value_losses
        """
        rng = np.random.default_rng(seed)
        history: dict[str, list] = {
            "episode_utilities": [],
            "policy_losses": [],
            "value_losses": [],
        }

        for ep in range(1, num_episodes + 1):
            # --- Collect trajectory ---
            state = AssetAllocState(time_step=0, wealth=env.initial_wealth)
            states_features: list[np.ndarray] = []
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            values: list[torch.Tensor] = []

            done = False
            while not done:
                features = env.state_to_features(state)
                states_features.append(features)

                # Get value estimate
                state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                value = self.value_net(state_t).squeeze()
                values.append(value)

                # Select action
                weights, log_prob = self._select_action(features)
                log_probs.append(log_prob)

                # Step environment
                next_state, reward, done = env.step(state, weights, rng)
                rewards.append(reward)
                state = next_state

            # --- Compute discounted returns ---
            T = len(rewards)
            returns_arr = np.zeros(T, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns_arr[t] = G

            returns_t = torch.FloatTensor(returns_arr).to(self.device)

            # --- Compute advantages ---
            values_t = torch.stack(values)
            advantages = returns_t - values_t.detach()

            # Normalize advantages for stability
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # --- Policy loss: -E[A_t * log pi(a_t|s_t)] ---
            log_probs_t = torch.stack(log_probs)
            policy_loss = -(log_probs_t * advantages).mean()

            # --- Value loss: E[(G_t - V(s_t))^2] ---
            value_loss = nn.functional.mse_loss(values_t, returns_t)

            # --- Update ---
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_body.parameters())
                + list(self.policy_mean.parameters())
                + [self.policy_log_std],
                1.0,
            )
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()

            # --- Logging ---
            terminal_utility = rewards[-1]  # only terminal reward is non-zero
            history["episode_utilities"].append(terminal_utility)
            history["policy_losses"].append(policy_loss.item())
            history["value_losses"].append(value_loss.item())

            if verbose and ep % max(1, num_episodes // 20) == 0:
                recent = history["episode_utilities"][-100:]
                mean_u = np.mean(recent)
                std_u = np.std(recent) if len(recent) > 1 else 0.0
                logger.info(
                    f"Episode {ep}/{num_episodes} | "
                    f"Mean utility: {mean_u:.4f} +/- {std_u:.4f} | "
                    f"Policy loss: {policy_loss.item():.6f}"
                )

        self._trained = True
        return history

    def get_weights(self, state: AssetAllocState, env: AssetAllocationMDP) -> np.ndarray:
        """Get portfolio weights for the given state using the trained policy.

        Parameters
        ----------
        state : AssetAllocState
        env : AssetAllocationMDP
            Needed for state_to_features conversion.

        Returns
        -------
        np.ndarray, shape (num_assets,)
        """
        features = env.state_to_features(state)
        state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self._get_policy_distribution(state_t)
            weights = dist.mean.squeeze(0).cpu().numpy()
        return weights

    def evaluate(
        self,
        env: AssetAllocationMDP,
        num_episodes: int = 1000,
        seed: int = 123,
    ) -> dict:
        """Evaluate the trained policy.

        Returns
        -------
        dict with keys:
            mean_utility, std_utility, sharpe, mean_terminal_wealth,
            std_terminal_wealth, weights_history
        """
        rng = np.random.default_rng(seed)
        utilities: list[float] = []
        terminal_wealths: list[float] = []
        all_weights: list[list[np.ndarray]] = []

        for _ in range(num_episodes):
            state = AssetAllocState(time_step=0, wealth=env.initial_wealth)
            episode_weights: list[np.ndarray] = []
            done = False
            while not done:
                weights = self.get_weights(state, env)
                episode_weights.append(weights)
                state, reward, done = env.step(state, weights, rng)

            utilities.append(reward)
            terminal_wealths.append(state.wealth)
            all_weights.append(episode_weights)

        utilities_arr = np.array(utilities)
        wealths_arr = np.array(terminal_wealths)
        returns_arr = wealths_arr / env.initial_wealth - 1.0

        sharpe = 0.0
        if len(returns_arr) > 1 and np.std(returns_arr, ddof=1) > 1e-10:
            sharpe = float(
                np.mean(returns_arr) / np.std(returns_arr, ddof=1)
                * math.sqrt(252 / env.num_steps)
            )

        return {
            "mean_utility": float(np.mean(utilities_arr)),
            "std_utility": float(np.std(utilities_arr, ddof=1)) if len(utilities_arr) > 1 else 0.0,
            "sharpe": sharpe,
            "mean_terminal_wealth": float(np.mean(wealths_arr)),
            "std_terminal_wealth": float(np.std(wealths_arr, ddof=1)) if len(wealths_arr) > 1 else 0.0,
            "weights_history": all_weights,
        }


# ============================================================================
# MultiStrategyAllocator -- Extension for BRAHMASTRA strategies
# ============================================================================


class MultiStrategyAllocator:
    """Allocate across multiple strategy return streams using RL.

    Each "asset" is a strategy's return stream (e.g., S1-S12 from BRAHMASTRA).
    The RL agent learns dynamic weights conditioned on regime, VIX,
    and strategy-specific signals.

    This extends Merton's problem to:
    - Non-Gaussian returns (strategies have fat tails, skew)
    - Regime-dependent correlations
    - Strategy-specific capacity constraints
    - Turnover penalties

    Architecture:
        Actor: (features) -> strategy weights via softmax (long-only)
        Critic: (features) -> expected portfolio utility

    References:
        Rao & Jelvis Ch 8
        Zhang et al. (2020), "Deep Reinforcement Learning for Portfolio Management"
    """

    def __init__(
        self,
        num_strategies: int,
        feature_dim: int,
        hidden_layers: Sequence[int] = (256, 128, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        utility: str = "crra",
        gamma_risk: float = 2.0,
        turnover_penalty: float = 0.001,
        device: str = "auto",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for MultiStrategyAllocator")

        self.num_strategies = num_strategies
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.utility = utility
        self.gamma_risk = gamma_risk
        self.turnover_penalty = turnover_penalty
        self.device = _get_device(device)

        # --- Actor network: features -> strategy weights (softmax) ---
        actor_layers: list[nn.Module] = []
        prev_dim = feature_dim
        for h in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, h))
            actor_layers.append(nn.LayerNorm(h))
            actor_layers.append(nn.ReLU())
            prev_dim = h
        actor_layers.append(nn.Linear(prev_dim, num_strategies))
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # --- Critic network: features -> scalar value ---
        critic_layers: list[nn.Module] = []
        prev_dim = feature_dim
        for h in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, h))
            critic_layers.append(nn.LayerNorm(h))
            critic_layers.append(nn.ReLU())
            prev_dim = h
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self._prev_weights: Optional[np.ndarray] = None

    def get_weights(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute strategy allocation weights.

        Parameters
        ----------
        features : np.ndarray, shape (feature_dim,)
            Market/strategy features (regime, VIX, rolling returns, etc.)
        temperature : float
            Softmax temperature. Lower = more concentrated.

        Returns
        -------
        np.ndarray, shape (num_strategies,)
            Normalized allocation weights (sum to 1).
        """
        feat_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(feat_t).squeeze(0) / temperature
            weights = torch.softmax(logits, dim=-1).cpu().numpy()
        return weights

    def train_step(
        self,
        features: np.ndarray,
        strategy_returns: np.ndarray,
    ) -> dict:
        """One training step with observed strategy returns.

        Parameters
        ----------
        features : np.ndarray, shape (feature_dim,)
            Features at time t.
        strategy_returns : np.ndarray, shape (num_strategies,)
            Realized returns of each strategy for this period.

        Returns
        -------
        dict with actor_loss, critic_loss, portfolio_return
        """
        feat_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        ret_t = torch.FloatTensor(strategy_returns).to(self.device)

        # Get allocation
        logits = self.actor(feat_t).squeeze(0)
        weights = torch.softmax(logits, dim=-1)

        # Portfolio return
        port_return = (weights * ret_t).sum()

        # Turnover cost
        if self._prev_weights is not None:
            prev_w = torch.FloatTensor(self._prev_weights).to(self.device)
            turnover = torch.abs(weights - prev_w).sum()
            port_return = port_return - self.turnover_penalty * turnover

        # Utility
        if self.utility == "crra":
            # Approximate: for small returns, utility ~ return - 0.5*gamma*return^2
            reward = port_return - 0.5 * self.gamma_risk * port_return ** 2
        else:
            reward = port_return

        # Critic value
        value = self.critic(feat_t).squeeze()

        # Advantage
        advantage = reward.detach() - value.detach()

        # Actor loss: policy gradient
        log_weights = torch.log_softmax(logits, dim=-1)
        # Use the log probability of the chosen allocation
        actor_loss = -(advantage * (log_weights * weights.detach()).sum())

        # Critic loss
        critic_loss = nn.functional.mse_loss(value, reward.detach())

        # Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self._prev_weights = weights.detach().cpu().numpy()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "portfolio_return": port_return.item(),
        }

    def train(
        self,
        features_series: np.ndarray,
        returns_series: np.ndarray,
        num_epochs: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Train on historical feature and return series.

        Parameters
        ----------
        features_series : np.ndarray, shape (T, feature_dim)
        returns_series : np.ndarray, shape (T, num_strategies)
        num_epochs : int
            Number of passes through the data.
        verbose : bool

        Returns
        -------
        dict with training history.
        """
        T = features_series.shape[0]
        history: dict[str, list] = {
            "epoch_returns": [],
            "epoch_actor_losses": [],
            "epoch_critic_losses": [],
        }

        for epoch in range(1, num_epochs + 1):
            self._prev_weights = None
            epoch_returns: list[float] = []
            epoch_actor: list[float] = []
            epoch_critic: list[float] = []

            for t in range(T):
                result = self.train_step(features_series[t], returns_series[t])
                epoch_returns.append(result["portfolio_return"])
                epoch_actor.append(result["actor_loss"])
                epoch_critic.append(result["critic_loss"])

            mean_ret = np.mean(epoch_returns)
            history["epoch_returns"].append(mean_ret)
            history["epoch_actor_losses"].append(np.mean(epoch_actor))
            history["epoch_critic_losses"].append(np.mean(epoch_critic))

            if verbose and epoch % max(1, num_epochs // 10) == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Mean return: {mean_ret:.6f}"
                )

        return history
