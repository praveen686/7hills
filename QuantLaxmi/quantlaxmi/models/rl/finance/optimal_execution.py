"""Chapter 10.2: Optimal Order Execution

Problem: Execute a large order of N shares over T periods, minimizing the
combined cost of market impact and execution risk.

The market microstructure reality:
- Large orders MOVE prices (market impact)
- Trading too fast => high impact cost
- Trading too slow => high risk (price may move against you)
- Optimal: balance impact vs risk

Two impact components:
1. Temporary impact: price moves during your trade, reverts after
   Cost = beta * n_t (proportional to trade size)
2. Permanent impact: your trade permanently shifts the price
   Cost = alpha * n_t (information content of your trade)

Bertsimas-Lo Model (1998):
    - Linear temporary and permanent impact
    - Risk-neutral optimal: N*_t = R_t/(T-t) = TWAP
    - With AR(1) price signal X_t:
      N*_t = R_t/(T-t) + h(t, beta, theta, rho) * X_t

Almgren-Chriss Model (2001):
    - Adds execution risk (variance of implementation shortfall)
    - Optimal trajectory: x_t = x_0 * sinh(kappa*(T-t)) / sinh(kappa*T)
      where kappa = sqrt(lambda * sigma^2 / eta)
    - lambda = risk aversion: higher => trade faster (front-loaded)
    - Efficient frontier: E[cost] vs Var[cost] parameterized by lambda

References:
    Rao & Jelvis, Ch 10.2 -- "Optimal Order Execution"
    Bertsimas & Lo (1998), "Optimal Control of Execution Costs"
    Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
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
    )
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

__all__ = [
    "OrderExecutionMDP",
    "BertsimasLoSolution",
    "AlmgrenChrissSolution",
    "RLExecutionAgent",
]


def _get_device(device: str = "auto") -> str:
    if device == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# ============================================================================
# OrderExecutionMDP -- Chapter 10.2
# ============================================================================


@dataclass
class ExecutionState:
    """State for the order execution MDP.

    Attributes:
        time_step: current period t in [0, T]
        remaining_shares: shares left to execute R_t (R_0 = N)
        mid_price: current mid price P_t
        spread: current bid-ask spread (optional)
        volatility: recent realized volatility estimate (optional)
    """
    time_step: int
    remaining_shares: float
    mid_price: float
    spread: float = 0.01
    volatility: float = 0.02

    def __hash__(self) -> int:
        return hash((self.time_step, round(self.remaining_shares, 2),
                      round(self.mid_price, 4)))


class OrderExecutionMDP:
    """Chapter 10.2: Order execution as MDP.

    State: (time_step, remaining_shares, mid_price, spread, volatility)
    Action: number of shares to trade this period, n_t in [0, R_t]
    Transition:
        - Temporary price impact: execution price = P_t + beta * n_t
        - Permanent price impact: P_{t+1} = P_t - alpha * n_t + sigma * Z_t
        - Remaining: R_{t+1} = R_t - n_t
    Reward: -execution_cost at each step

    Execution cost per step:
        C_t = n_t * (beta * n_t + 0.5 * spread)  [temporary impact + half-spread]
            + alpha * n_t^2                         [permanent impact]

    For BUY orders: we pay more than mid (positive impact).
    For SELL orders: we receive less than mid (negative impact).

    Implementation Shortfall (IS) = sum_t C_t + risk_penalty

    Parameters
    ----------
    total_shares : int
        Total number of shares to execute (positive for buy).
    num_steps : int
        Number of trading periods.
    price_init : float
        Initial mid price.
    sigma : float
        Price volatility per step (standard deviation of price change).
    alpha : float
        Permanent impact coefficient. Permanent shift = alpha * n_t.
    beta : float
        Temporary impact coefficient. Execution price = mid + beta * n_t.
    spread : float
        Bid-ask spread.
    gamma_risk : float
        Risk aversion for variance penalty. 0 = risk-neutral.
    dt : float
        Time per step in years.
    """

    def __init__(
        self,
        total_shares: int = 1000,
        num_steps: int = 20,
        price_init: float = 100.0,
        sigma: float = 0.02,
        alpha: float = 0.001,
        beta: float = 0.005,
        spread: float = 0.01,
        gamma_risk: float = 0.0,
        dt: float = 1.0 / 252,
    ) -> None:
        self.total_shares = total_shares
        self.num_steps = num_steps
        self.price_init = price_init
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.spread = spread
        self.gamma_risk = gamma_risk
        self.dt = dt

    def step(
        self,
        state: ExecutionState,
        trade_size: float,
        rng: np.random.Generator,
    ) -> Tuple[ExecutionState, float, bool]:
        """Execute one period of trading.

        Parameters
        ----------
        state : ExecutionState
        trade_size : float
            Number of shares to trade this period. Clamped to [0, remaining].
        rng : np.random.Generator

        Returns
        -------
        next_state, reward, done
        """
        # Clamp trade size
        trade_size = max(0.0, min(trade_size, state.remaining_shares))

        # Execution cost:
        # Temporary impact cost: n * beta * n = beta * n^2
        # Half-spread cost: n * spread / 2
        # Permanent impact: alpha * n (shifts the price for future trades)
        temp_impact_cost = self.beta * trade_size ** 2
        spread_cost = trade_size * state.spread / 2.0
        perm_impact_shift = self.alpha * trade_size

        execution_cost = temp_impact_cost + spread_cost

        # Risk penalty: gamma * sigma^2 * remaining^2 (penalty for holding risk)
        remaining_after = state.remaining_shares - trade_size
        risk_penalty = self.gamma_risk * self.sigma ** 2 * remaining_after ** 2

        # Reward = negative of total cost
        reward = -(execution_cost + risk_penalty)

        # Price evolution: random walk with permanent impact
        # P_{t+1} = P_t - alpha * n_t + sigma * Z_t
        # The permanent impact shifts the price AGAINST us (we're buying, price goes up)
        z = rng.standard_normal()
        new_price = state.mid_price + perm_impact_shift + self.sigma * z

        next_time = state.time_step + 1
        done = next_time >= self.num_steps or remaining_after <= 0

        # If we still have shares at the end, penalize heavily
        # (liquidation at unfavorable price)
        if done and remaining_after > 0:
            liquidation_cost = remaining_after * (self.beta * remaining_after + state.spread)
            reward -= liquidation_cost

        next_state = ExecutionState(
            time_step=next_time,
            remaining_shares=remaining_after,
            mid_price=new_price,
            spread=state.spread,
            volatility=state.volatility,
        )
        return next_state, reward, done

    def initial_state(self) -> ExecutionState:
        """Create initial execution state."""
        return ExecutionState(
            time_step=0,
            remaining_shares=float(self.total_shares),
            mid_price=self.price_init,
            spread=self.spread,
            volatility=self.sigma,
        )

    def state_to_features(self, state: ExecutionState) -> np.ndarray:
        """Convert state to feature vector for neural network.

        Features:
            [0] time_remaining / T
            [1] remaining_shares / total_shares
            [2] price_change = (mid - init_price) / init_price
            [3] spread (normalized)
            [4] volatility (normalized)
            [5] urgency = remaining / max(1, time_remaining) (shares per step needed)
        """
        time_remaining = max(self.num_steps - state.time_step, 1)
        return np.array([
            time_remaining / self.num_steps,
            state.remaining_shares / max(self.total_shares, 1),
            (state.mid_price - self.price_init) / max(self.price_init, 1e-10),
            state.spread / max(self.price_init, 1e-10),
            state.volatility,
            state.remaining_shares / time_remaining / max(self.total_shares, 1),
        ], dtype=np.float64)


# ============================================================================
# BertsimasLoSolution -- Analytical benchmark
# ============================================================================


class BertsimasLoSolution:
    """Analytical solution: Bertsimas-Lo optimal execution.

    For the linear impact model with risk-neutral agent:

    Optimal schedule (risk-neutral, no signal):
        N*_t = R_t / (T - t) = N / T  (uniform TWAP)

    This is because with linear costs, there's no benefit to front-loading
    or back-loading when there's no risk penalty.

    With AR(1) price signal X_t (autocorrelation rho):
        N*_t = R_t/(T-t) + h(t, beta, theta, rho) * X_t

    where h accounts for the price predictability. If the signal predicts
    prices will rise, trade more now; if fall, trade less now.

    TWAP (Time-Weighted Average Price):
        Trade equal amounts each period: n_t = N/T

    VWAP (Volume-Weighted Average Price):
        Trade proportional to expected volume: n_t = N * v_t / sum(v)
        where v_t is the expected volume at time t

    References:
        Bertsimas & Lo (1998), Proposition 1 and Theorem 1
        Rao & Jelvis Ch 10.2, Section "Bertsimas-Lo Model"
    """

    @staticmethod
    def optimal_schedule(
        total_shares: int,
        num_steps: int,
        risk_neutral: bool = True,
    ) -> np.ndarray:
        """Compute the Bertsimas-Lo optimal execution schedule.

        For risk-neutral (no price signal): uniform TWAP.

        Parameters
        ----------
        total_shares : int
        num_steps : int
        risk_neutral : bool
            If True, returns TWAP (the risk-neutral optimal).

        Returns
        -------
        np.ndarray, shape (num_steps,)
            Number of shares to trade at each step.
        """
        if risk_neutral:
            return BertsimasLoSolution.twap_schedule(total_shares, num_steps)
        else:
            # Without signal, risk-neutral is optimal
            return BertsimasLoSolution.twap_schedule(total_shares, num_steps)

    @staticmethod
    def twap_schedule(total_shares: int, num_steps: int) -> np.ndarray:
        """Time-Weighted Average Price schedule.

        n_t = N / T for all t.
        Minimizes market impact when impact is linear and no risk penalty.

        Parameters
        ----------
        total_shares : int
        num_steps : int

        Returns
        -------
        np.ndarray, shape (num_steps,)
        """
        per_step = total_shares / num_steps
        return np.full(num_steps, per_step, dtype=np.float64)

    @staticmethod
    def vwap_schedule(
        total_shares: int,
        volume_profile: np.ndarray,
    ) -> np.ndarray:
        """Volume-Weighted Average Price schedule.

        n_t = N * v_t / sum(v)

        Matches execution to expected market volume, reducing
        market impact by trading when liquidity is highest.

        Parameters
        ----------
        total_shares : int
        volume_profile : np.ndarray, shape (num_steps,)
            Expected volume at each step.

        Returns
        -------
        np.ndarray, shape (num_steps,)
        """
        volume_profile = np.asarray(volume_profile, dtype=np.float64)
        total_vol = volume_profile.sum()
        if total_vol <= 0:
            # Fallback to TWAP
            return BertsimasLoSolution.twap_schedule(total_shares, len(volume_profile))
        return total_shares * volume_profile / total_vol

    @staticmethod
    def expected_cost_twap(
        total_shares: int,
        num_steps: int,
        alpha: float,
        beta: float,
        spread: float = 0.0,
    ) -> float:
        """Expected execution cost for TWAP schedule.

        For TWAP with n_t = N/T each step:
            E[cost] = T * [beta * (N/T)^2 + (N/T) * spread/2]
                    = beta * N^2 / T + N * spread / 2

        The permanent impact cost is path-independent and equals alpha * N * P_avg.
        For the simplified model: perm_cost = T * alpha * (N/T)^2 = alpha * N^2 / T

        Total: (alpha + beta) * N^2 / T + N * spread / 2
        """
        n = total_shares / num_steps
        temp_cost = num_steps * beta * n ** 2
        spread_cost = total_shares * spread / 2.0
        return temp_cost + spread_cost

    @staticmethod
    def optimal_with_signal(
        remaining_shares: float,
        time_remaining: int,
        signal: float,
        beta: float,
        signal_sensitivity: float = 0.5,
    ) -> float:
        """Optimal trade size with a price signal.

        N*_t = R_t / (T-t) + h * X_t

        where h = signal_sensitivity / (2 * beta) is the responsiveness
        to the price signal.

        If signal > 0 (price expected to rise for a buy order):
            Trade more now to avoid higher future prices.
        If signal < 0 (price expected to fall):
            Trade less now and wait for lower prices.

        Parameters
        ----------
        remaining_shares : float
        time_remaining : int
        signal : float
            Price signal X_t (positive = expect price increase).
        beta : float
            Temporary impact coefficient.
        signal_sensitivity : float
            How responsive to be to the signal.

        Returns
        -------
        float
            Optimal trade size.
        """
        base = remaining_shares / max(time_remaining, 1)
        signal_adj = signal_sensitivity / (2.0 * max(beta, 1e-10)) * signal
        return max(0.0, base + signal_adj)


# ============================================================================
# AlmgrenChrissSolution -- Risk-averse optimal execution
# ============================================================================


class AlmgrenChrissSolution:
    """Almgren-Chriss framework with risk aversion.

    The key insight: there's a trade-off between execution cost and
    execution risk. Risk-averse agents trade faster (front-loaded).

    The optimization problem:
        min_n  E[IS] + lambda * Var[IS]

    where IS = Implementation Shortfall = sum of execution costs.

    Optimal trajectory (inventory at each time):
        x_t = x_0 * sinh(kappa * (T-t)) / sinh(kappa * T)

    where:
        kappa = sqrt(lambda * sigma^2 / eta)
        lambda = risk aversion
        sigma = price volatility (per step)
        eta = temporary impact coefficient

    Trade schedule:
        n_t = x_{t-1} - x_t

    Special cases:
        lambda = 0:  TWAP (risk-neutral), kappa -> 0
        lambda -> inf:  Immediate execution (risk-averse), kappa -> inf

    Expected cost:
        E[IS] = 0.5 * gamma_perm * x_0^2 + eta * sum_t n_t^2

    Cost variance:
        Var[IS] = sigma^2 * sum_t x_t^2

    Efficient frontier:
        Parameterized by lambda: (E[cost], Var[cost])

    References:
        Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
        Rao & Jelvis Ch 10.2
    """

    def __init__(
        self,
        total_shares: int,
        num_steps: int,
        sigma: float,
        eta: float,
        gamma_perm: float,
        risk_aversion: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        total_shares : int
            Total shares to execute (x_0 = total_shares).
        num_steps : int
            Number of trading periods T.
        sigma : float
            Price volatility per step.
        eta : float
            Temporary impact coefficient (cost = eta * n^2 per step).
        gamma_perm : float
            Permanent impact coefficient.
        risk_aversion : float
            Lambda in the mean-variance objective.
        """
        self.total_shares = total_shares
        self.num_steps = num_steps
        self.sigma = sigma
        self.eta = eta
        self.gamma_perm = gamma_perm
        self.risk_aversion = risk_aversion

        # kappa = sqrt(lambda * sigma^2 / eta)
        if eta > 0 and risk_aversion > 0:
            self.kappa = math.sqrt(risk_aversion * sigma ** 2 / eta)
        else:
            self.kappa = 0.0

    def optimal_trajectory(self) -> np.ndarray:
        """Compute the optimal inventory trajectory x_t.

        x_t = x_0 * sinh(kappa * (T-t)) / sinh(kappa * T)

        Returns
        -------
        np.ndarray, shape (T+1,)
            Inventory at each time: x[0] = total_shares, x[T] = 0.
        """
        T = self.num_steps
        x0 = float(self.total_shares)

        if self.kappa < 1e-10:
            # Risk-neutral limit: linear trajectory (TWAP)
            return x0 * np.linspace(1.0, 0.0, T + 1)

        trajectory = np.zeros(T + 1, dtype=np.float64)
        sinh_kT = math.sinh(self.kappa * T)

        for t in range(T + 1):
            trajectory[t] = x0 * math.sinh(self.kappa * (T - t)) / sinh_kT

        return trajectory

    def optimal_schedule(self) -> np.ndarray:
        """Compute the optimal trade schedule n_t = x_{t-1} - x_t.

        Returns
        -------
        np.ndarray, shape (T,)
            Shares to trade at each step.
        """
        trajectory = self.optimal_trajectory()
        return trajectory[:-1] - trajectory[1:]

    def expected_cost(self) -> float:
        """Expected implementation shortfall.

        E[IS] = 0.5 * gamma_perm * x_0^2
              + eta * sum_t n_t^2

        This ignores the stochastic component (which has zero expectation).
        """
        schedule = self.optimal_schedule()
        perm_cost = 0.5 * self.gamma_perm * self.total_shares ** 2
        temp_cost = self.eta * np.sum(schedule ** 2)
        return perm_cost + temp_cost

    def cost_variance(self) -> float:
        """Variance of implementation shortfall.

        Var[IS] = sigma^2 * sum_{t=0}^{T-1} x_t^2

        The variance comes from the random price movements while
        holding inventory. More inventory = more risk.
        """
        trajectory = self.optimal_trajectory()
        # x_t for t = 0, ..., T-1 (we hold inventory x_t during period t)
        return self.sigma ** 2 * float(np.sum(trajectory[:-1] ** 2))

    def efficient_frontier(
        self, lambdas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the efficient frontier: E[cost] vs Var[cost].

        Parameterized by risk aversion lambda.

        Parameters
        ----------
        lambdas : np.ndarray
            Array of risk aversion values to evaluate.

        Returns
        -------
        (expected_costs, cost_variances)
            Arrays of same shape as lambdas.
        """
        lambdas = np.asarray(lambdas, dtype=np.float64)
        expected_costs = np.zeros_like(lambdas)
        cost_variances = np.zeros_like(lambdas)

        for i, lam in enumerate(lambdas):
            solver = AlmgrenChrissSolution(
                total_shares=self.total_shares,
                num_steps=self.num_steps,
                sigma=self.sigma,
                eta=self.eta,
                gamma_perm=self.gamma_perm,
                risk_aversion=lam,
            )
            expected_costs[i] = solver.expected_cost()
            cost_variances[i] = solver.cost_variance()

        return expected_costs, cost_variances

    def urgency_factor(self) -> float:
        """Measure of how front-loaded the optimal schedule is.

        Returns the fraction of shares traded in the first half of the period.
        TWAP = 0.5, fully front-loaded = 1.0.
        """
        schedule = self.optimal_schedule()
        T = len(schedule)
        first_half = schedule[: T // 2].sum()
        return float(first_half / max(schedule.sum(), 1e-10))


# ============================================================================
# RLExecutionAgent -- RL-based execution (Ch 10.2 + Ch 13-14)
# ============================================================================


class RLExecutionAgent:
    """RL-based execution agent using Actor-Critic.

    Advantages over analytical solutions (Bertsimas-Lo, Almgren-Chriss):
    1. Handles NON-LINEAR market impact (concave, convex, or empirical)
    2. Adapts to CHANGING market conditions (regime shifts)
    3. Incorporates ORDER BOOK features (depth, imbalance)
    4. Works with EMPIRICAL dynamics (non-Gaussian price moves)
    5. Can learn to be OPPORTUNISTIC (trade more in favorable conditions)

    Architecture:
        Actor: state -> fraction of remaining to trade, output in [0, 1]
               via sigmoid (can't sell what you don't have, can't trade negative)
        Critic: state -> expected future cost (value function)

    State features:
        [0] time_remaining / T
        [1] remaining_shares / total_shares
        [2] price_change since start (normalized)
        [3] spread (normalized)
        [4] depth_imbalance (optional, from LOB)
        [5] volatility_estimate

    Action: fraction of remaining shares to trade this period
        actual_trade = action * remaining_shares

    Training objective: minimize total execution cost (maximize -cost reward).

    References:
        Rao & Jelvis Ch 10.2 (problem formulation)
        Ning et al. (2021), "Double Deep Q-Learning for Optimal Execution"
    """

    def __init__(
        self,
        state_dim: int = 6,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "auto",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for RLExecutionAgent")

        self.state_dim = state_dim
        self.gamma = gamma
        self.device = _get_device(device)

        # --- Actor network: state -> trade fraction in [0, 1] ---
        actor_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, h))
            actor_layers.append(nn.ReLU())
            prev_dim = h
        actor_layers.append(nn.Linear(prev_dim, 1))
        actor_layers.append(nn.Sigmoid())  # output in [0, 1]
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # --- Critic network: state -> value ---
        critic_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, h))
            critic_layers.append(nn.ReLU())
            prev_dim = h
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        # Exploration noise
        self._noise_std = 0.1

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self._trained = False

    def _select_action(
        self, state_features: np.ndarray, explore: bool = True
    ) -> Tuple[float, torch.Tensor]:
        """Select trade fraction and return log_prob for training."""
        state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        action_mean = self.actor(state_t).squeeze()

        if explore:
            # Add Gaussian noise for exploration, clamp to [0, 1]
            noise = torch.randn_like(action_mean) * self._noise_std
            action = (action_mean + noise).clamp(0.0, 1.0)
        else:
            action = action_mean

        # Log probability under Gaussian (for policy gradient)
        # This is approximate -- we use the mean action for the gradient
        log_prob = -0.5 * ((action - action_mean) / max(self._noise_std, 1e-8)) ** 2

        return action.item(), log_prob

    def train(
        self,
        env: OrderExecutionMDP,
        num_episodes: int = 10000,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """Train the execution agent.

        Algorithm: Advantage Actor-Critic (A2C)
        1. Collect episode trajectory
        2. Compute discounted returns
        3. Advantage = return - baseline (critic)
        4. Update actor with policy gradient
        5. Update critic with TD target

        Parameters
        ----------
        env : OrderExecutionMDP
        num_episodes : int
        seed : int
        verbose : bool

        Returns
        -------
        dict with training history.
        """
        rng = np.random.default_rng(seed)
        history: dict[str, list] = {
            "episode_costs": [],
            "actor_losses": [],
            "critic_losses": [],
        }

        for ep in range(1, num_episodes + 1):
            state = env.initial_state()
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []

            done = False
            while not done:
                features = env.state_to_features(state)

                # Critic value
                state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                value = self.critic(state_t).squeeze()
                values.append(value)

                # Actor action
                fraction, log_prob = self._select_action(features, explore=True)
                log_probs.append(log_prob)

                # Convert fraction to actual shares
                trade_size = fraction * state.remaining_shares
                next_state, reward, done = env.step(state, trade_size, rng)
                rewards.append(reward)
                state = next_state

            # Compute returns
            T = len(rewards)
            returns = np.zeros(T, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns[t] = G

            returns_t = torch.FloatTensor(returns).to(self.device)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)

            # Advantages
            advantages = returns_t - values_t.detach()
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # Actor loss
            actor_loss = -(log_probs_t * advantages).mean()

            # Critic loss
            critic_loss = nn.functional.mse_loss(values_t, returns_t)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Decay exploration noise
            self._noise_std = max(0.01, self._noise_std * 0.9999)

            total_cost = -sum(rewards)
            history["episode_costs"].append(total_cost)
            history["actor_losses"].append(actor_loss.item())
            history["critic_losses"].append(critic_loss.item())

            if verbose and ep % max(1, num_episodes // 20) == 0:
                recent_costs = history["episode_costs"][-100:]
                logger.info(
                    f"Episode {ep}/{num_episodes} | "
                    f"Mean cost: {np.mean(recent_costs):.4f} | "
                    f"Noise: {self._noise_std:.4f}"
                )

        self._trained = True
        return history

    def get_trade_size(
        self,
        state: ExecutionState,
        env: OrderExecutionMDP,
    ) -> float:
        """Get the number of shares to trade.

        Parameters
        ----------
        state : ExecutionState
        env : OrderExecutionMDP
            Needed for state_to_features.

        Returns
        -------
        float
            Number of shares to trade this period.
        """
        features = env.state_to_features(state)
        state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fraction = self.actor(state_t).item()
        return fraction * state.remaining_shares

    def evaluate(
        self,
        env: OrderExecutionMDP,
        num_episodes: int = 1000,
        benchmark: str = "twap",
        seed: int = 123,
    ) -> dict:
        """Evaluate the RL agent against a benchmark.

        Parameters
        ----------
        env : OrderExecutionMDP
        num_episodes : int
        benchmark : str
            "twap" or "ac" (Almgren-Chriss).
        seed : int

        Returns
        -------
        dict with keys:
            rl_cost_mean, rl_cost_std,
            benchmark_cost_mean, benchmark_cost_std,
            improvement_pct,
            rl_completion_rate (fraction of shares executed on time)
        """
        rng = np.random.default_rng(seed)

        # --- RL agent ---
        rl_costs: list[float] = []
        rl_completions: list[float] = []
        for _ in range(num_episodes):
            state = env.initial_state()
            total_cost = 0.0
            done = False
            while not done:
                trade_size = self.get_trade_size(state, env)
                state, reward, done = env.step(state, trade_size, rng)
                total_cost += -reward
            rl_costs.append(total_cost)
            rl_completions.append(
                1.0 - state.remaining_shares / max(env.total_shares, 1)
            )

        # --- Benchmark ---
        if benchmark == "twap":
            schedule = BertsimasLoSolution.twap_schedule(
                env.total_shares, env.num_steps
            )
        elif benchmark == "ac":
            ac = AlmgrenChrissSolution(
                total_shares=env.total_shares,
                num_steps=env.num_steps,
                sigma=env.sigma,
                eta=env.beta,
                gamma_perm=env.alpha,
                risk_aversion=env.gamma_risk if env.gamma_risk > 0 else 1e-6,
            )
            schedule = ac.optimal_schedule()
        else:
            schedule = BertsimasLoSolution.twap_schedule(
                env.total_shares, env.num_steps
            )

        bench_costs: list[float] = []
        for _ in range(num_episodes):
            state = env.initial_state()
            total_cost = 0.0
            for t in range(env.num_steps):
                trade_size = float(schedule[t]) if t < len(schedule) else 0.0
                state, reward, done = env.step(state, trade_size, rng)
                total_cost += -reward
                if done:
                    break
            bench_costs.append(total_cost)

        rl_mean = float(np.mean(rl_costs))
        rl_std = float(np.std(rl_costs, ddof=1)) if len(rl_costs) > 1 else 0.0
        bench_mean = float(np.mean(bench_costs))
        bench_std = float(np.std(bench_costs, ddof=1)) if len(bench_costs) > 1 else 0.0

        improvement = 0.0
        if bench_mean > 1e-10:
            improvement = (bench_mean - rl_mean) / bench_mean * 100.0

        return {
            "rl_cost_mean": rl_mean,
            "rl_cost_std": rl_std,
            "benchmark_cost_mean": bench_mean,
            "benchmark_cost_std": bench_std,
            "benchmark_type": benchmark,
            "improvement_pct": improvement,
            "rl_completion_rate": float(np.mean(rl_completions)),
        }
