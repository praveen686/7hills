"""Chapter 10.3: Avellaneda-Stoikov Market-Making

A market maker continuously quotes bid and ask prices around the mid price.
The core trade-off:
- Tighter spreads = more fills, more revenue, BUT more adverse selection risk
- Wider spreads = fewer fills, less revenue, BUT less adverse selection risk
- Inventory risk: holding inventory exposes the MM to directional price risk

The Avellaneda-Stoikov (2008) model solves the HJB equation for optimal
market-making quotes as a function of inventory, volatility, risk aversion,
and fill rate parameters.

Key Results:

1. Pseudo-mid (inventory-adjusted mid price):
    Q_t^(m) = S_t - I_t * gamma * sigma^2 * (T-t)

    Interpretation: When the MM is LONG (I_t > 0), the pseudo-mid is BELOW
    the true mid, biasing quotes to sell. This creates an inventory
    mean-reversion force.

2. Optimal total spread:
    delta* = gamma * sigma^2 * (T-t) + (2/gamma) * log(1 + gamma/k)

    The first term is the inventory risk premium.
    The second term is the adverse selection / fill rate component.

3. Individual quote offsets from mid:
    delta_bid* = delta*/2 + I_t * gamma * sigma^2 * (T-t)
    delta_ask* = delta*/2 - I_t * gamma * sigma^2 * (T-t)

    When long inventory: widen bid (less keen to buy more), tighten ask (keen to sell).
    When short inventory: tighten bid (keen to buy), widen ask (less keen to sell).

4. Fill probability (exponential model):
    P(fill) = A * exp(-k * delta)
    where delta is the distance from mid and k is the decay parameter.

Parameters:
- gamma: risk aversion (higher = wider spreads, faster inventory reversion)
- sigma: price volatility per step
- k: fill rate decay (higher k = fills decay faster with distance)
- T: trading horizon
- A: fill rate base intensity

References:
    Avellaneda & Stoikov (2008), "High-Frequency Trading in a Limit Order Book"
    Gueant, Lehalle & Fernandez-Tapia (2012), "Dealing with the Inventory Risk"
    Rao & Jelvis Ch 10.3 -- "Market-Making as an MDP"
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
    from models.rl.core.markov_process import (
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
    from torch.distributions import Normal as TorchNormal

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

__all__ = [
    "MarketMakingMDP",
    "AvellanedaStoikovSolution",
    "RLMarketMaker",
    "InventoryRiskManager",
]


def _get_device(device: str = "auto") -> str:
    if device == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# ============================================================================
# MarketMakingMDP -- Chapter 10.3
# ============================================================================


@dataclass
class MarketMakingState:
    """State for the market-making MDP.

    Attributes:
        time_step: current step t in [0, T]
        mid_price: current mid price S_t
        inventory: net position I_t (positive = long, negative = short)
        pnl: accumulated realized PnL
        spread: current market spread (observable)
        recent_vol: recent realized volatility estimate
        recent_flow: recent net order flow (positive = buy pressure)
    """
    time_step: int
    mid_price: float
    inventory: int
    pnl: float
    spread: float = 0.01
    recent_vol: float = 0.02
    recent_flow: float = 0.0

    def __hash__(self) -> int:
        return hash((self.time_step, round(self.mid_price, 4), self.inventory))


class MarketMakingMDP:
    """Chapter 10.3: Market-making as MDP.

    State: (time, mid_price, inventory, pnl, spread, vol, flow)
    Action: (bid_offset, ask_offset) -- distances from mid price
    Transition:
        1. Quote bid = mid - bid_offset, ask = mid + ask_offset
        2. Fills arrive stochastically: P(bid_fill) = A * exp(-k * bid_offset)
        3. Inventory updates: I_{t+1} = I_t + bid_fills - ask_fills
        4. Mid price evolves: S_{t+1} = S_t + sigma * Z_t
        5. PnL updates from filled orders
    Reward: PnL from filled orders - inventory risk penalty

    The fill probability model:
        P(fill at distance delta from mid) = A * exp(-k * delta)

    This exponential model captures the idea that:
    - Orders close to mid fill with high probability
    - Orders far from mid rarely fill
    - The parameter k controls the steepness of this decay

    Parameters
    ----------
    price_init : float
        Initial mid price.
    sigma : float
        Per-step price volatility.
    num_steps : int
        Trading horizon (number of steps).
    gamma_risk : float
        Risk aversion coefficient for inventory penalty.
    fill_rate_A : float
        Base fill intensity (probability scale).
    fill_rate_k : float
        Fill rate decay with distance from mid.
    max_inventory : int
        Maximum absolute inventory position.
    tick_size : float
        Minimum price increment.
    dt : float
        Time per step in years.
    """

    def __init__(
        self,
        price_init: float = 100.0,
        sigma: float = 0.02,
        num_steps: int = 1000,
        gamma_risk: float = 0.1,
        fill_rate_A: float = 1.0,
        fill_rate_k: float = 1.5,
        max_inventory: int = 10,
        tick_size: float = 0.01,
        dt: float = 1.0 / (252 * 6.5 * 60),  # 1 minute
    ) -> None:
        self.price_init = price_init
        self.sigma = sigma
        self.num_steps = num_steps
        self.gamma_risk = gamma_risk
        self.fill_rate_A = fill_rate_A
        self.fill_rate_k = fill_rate_k
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        self.dt = dt

    def fill_probability(self, offset: float) -> float:
        """Probability that an order at distance `offset` from mid gets filled.

        P(fill) = A * exp(-k * offset)

        Clamped to [0, 1].
        """
        if offset < 0:
            return min(self.fill_rate_A, 1.0)
        prob = self.fill_rate_A * math.exp(-self.fill_rate_k * offset)
        return min(max(prob, 0.0), 1.0)

    def step(
        self,
        state: MarketMakingState,
        bid_offset: float,
        ask_offset: float,
        rng: np.random.Generator,
    ) -> Tuple[MarketMakingState, float, bool]:
        """Execute one market-making step.

        Parameters
        ----------
        state : MarketMakingState
        bid_offset : float
            Distance below mid for bid quote (positive = below mid).
        ask_offset : float
            Distance above mid for ask quote (positive = above mid).
        rng : np.random.Generator

        Returns
        -------
        next_state, reward, done
        """
        # Enforce minimum spread (tick size)
        bid_offset = max(bid_offset, self.tick_size / 2)
        ask_offset = max(ask_offset, self.tick_size / 2)

        # Quote prices
        bid_price = state.mid_price - bid_offset
        ask_price = state.mid_price + ask_offset

        # Determine fills
        bid_fill_prob = self.fill_probability(bid_offset)
        ask_fill_prob = self.fill_probability(ask_offset)

        bid_filled = rng.random() < bid_fill_prob
        ask_filled = rng.random() < ask_fill_prob

        # Update inventory and PnL
        new_inventory = state.inventory
        step_pnl = 0.0

        if bid_filled and new_inventory < self.max_inventory:
            # Buy 1 unit at bid price
            new_inventory += 1
            step_pnl -= bid_price  # pay bid price

        if ask_filled and new_inventory > -self.max_inventory:
            # Sell 1 unit at ask price
            new_inventory -= 1
            step_pnl += ask_price  # receive ask price

        # Inventory risk penalty: gamma * I^2 * sigma^2
        # This penalizes holding large positions
        inv_penalty = self.gamma_risk * new_inventory ** 2 * self.sigma ** 2

        # Mid price evolution: arithmetic random walk
        z = rng.standard_normal()
        new_mid = state.mid_price + self.sigma * z

        # Mark-to-market PnL adjustment for inventory
        mtm_pnl = new_inventory * (new_mid - state.mid_price)

        # Total reward = realized PnL + unrealized PnL change - inventory penalty
        reward = step_pnl + mtm_pnl - inv_penalty

        next_time = state.time_step + 1
        done = next_time >= self.num_steps

        # At terminal, liquidate inventory at mid (with impact)
        if done and new_inventory != 0:
            # Liquidation cost: proportional to inventory
            liq_cost = abs(new_inventory) * self.sigma * 2
            reward -= liq_cost

        next_state = MarketMakingState(
            time_step=next_time,
            mid_price=new_mid,
            inventory=new_inventory,
            pnl=state.pnl + reward,
            spread=ask_offset + bid_offset,
            recent_vol=state.recent_vol,
            recent_flow=float(bid_filled) - float(ask_filled),
        )
        return next_state, reward, done

    def initial_state(self) -> MarketMakingState:
        """Create initial market-making state."""
        return MarketMakingState(
            time_step=0,
            mid_price=self.price_init,
            inventory=0,
            pnl=0.0,
            spread=0.01,
            recent_vol=self.sigma,
            recent_flow=0.0,
        )

    def state_to_features(self, state: MarketMakingState) -> np.ndarray:
        """Convert state to feature vector for neural network.

        Features:
            [0] time_remaining / T
            [1] inventory / max_inventory
            [2] mid_price_change = (mid - init) / init
            [3] spread (normalized)
            [4] imbalance (proxy for order flow)
            [5] recent_vol
            [6] recent_flow
            [7] inventory_risk = |inventory| / max_inventory
        """
        return np.array([
            (self.num_steps - state.time_step) / max(self.num_steps, 1),
            state.inventory / max(self.max_inventory, 1),
            (state.mid_price - self.price_init) / max(self.price_init, 1e-10),
            state.spread / max(self.price_init, 1e-10),
            0.0,  # placeholder for LOB imbalance
            state.recent_vol,
            state.recent_flow,
            abs(state.inventory) / max(self.max_inventory, 1),
        ], dtype=np.float64)


# ============================================================================
# AvellanedaStoikovSolution -- Analytical benchmark
# ============================================================================


class AvellanedaStoikovSolution:
    """Analytical solution for the Avellaneda-Stoikov market-making model.

    Solves the HJB equation:
        dV/dt + max_{delta_b, delta_a} {
            lambda_b(delta_b) * [delta_b + V(I+1) - V(I)]
          + lambda_a(delta_a) * [delta_a + V(I-1) - V(I)]
          - gamma * sigma^2 * I^2
        } = 0

    where lambda(delta) = A * exp(-k * delta) is the fill rate.

    First-order conditions yield the optimal quotes.

    Parameters
    ----------
    sigma : float
        Price volatility per step.
    gamma_risk : float
        Risk aversion coefficient.
    fill_rate_k : float
        Fill rate decay parameter.
    time_horizon : float
        Total trading time (in whatever units sigma is measured).
    fill_rate_A : float
        Base fill intensity.
    """

    def __init__(
        self,
        sigma: float,
        gamma_risk: float,
        fill_rate_k: float,
        time_horizon: float = 1.0,
        fill_rate_A: float = 1.0,
    ) -> None:
        self.sigma = sigma
        self.gamma_risk = gamma_risk
        self.fill_rate_k = fill_rate_k
        self.time_horizon = time_horizon
        self.fill_rate_A = fill_rate_A

    def pseudo_mid(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float,
    ) -> float:
        """Inventory-adjusted mid price.

        Q_t^(m) = S_t - I_t * gamma * sigma^2 * (T - t)

        When long (I > 0): pseudo-mid is BELOW actual mid.
            => Bid and ask are both shifted down => more likely to sell.
        When short (I < 0): pseudo-mid is ABOVE actual mid.
            => Bid and ask are both shifted up => more likely to buy.

        This creates an automatic inventory mean-reversion.

        Parameters
        ----------
        mid_price : float
            Current true mid price S_t.
        inventory : int
            Current inventory I_t.
        time_remaining : float
            Time left until horizon (T - t).

        Returns
        -------
        float
            Pseudo-mid price.
        """
        gamma = self.gamma_risk
        sigma_sq = self.sigma ** 2
        return mid_price - inventory * gamma * sigma_sq * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """Total optimal spread (bid_offset + ask_offset).

        delta* = gamma * sigma^2 * (T-t) + (2/gamma) * log(1 + gamma/k)

        Decomposition:
        - gamma * sigma^2 * (T-t): inventory risk premium
            Scales with volatility, risk aversion, and time remaining.
            As T-t -> 0, this term vanishes (less risk near end).
        - (2/gamma) * log(1 + gamma/k): adverse selection component
            Independent of time. Determined by fill rate decay and risk aversion.

        Parameters
        ----------
        time_remaining : float

        Returns
        -------
        float
            Total spread delta_bid + delta_ask.
        """
        gamma = self.gamma_risk
        sigma_sq = self.sigma ** 2
        k = self.fill_rate_k

        # Guard against gamma = 0
        if abs(gamma) < 1e-12:
            return 2.0 / max(k, 1e-12)

        risk_term = gamma * sigma_sq * time_remaining
        fill_term = (2.0 / gamma) * math.log(1.0 + gamma / k)

        return risk_term + fill_term

    def optimal_offsets(
        self,
        inventory: int,
        time_remaining: float,
    ) -> Tuple[float, float]:
        """Compute optimal bid and ask offsets from mid price.

        delta_bid* = delta*/2 + I_t * gamma * sigma^2 * (T-t)
        delta_ask* = delta*/2 - I_t * gamma * sigma^2 * (T-t)

        When I > 0 (long):
            delta_bid increases (wider bid, less keen to buy more)
            delta_ask decreases (tighter ask, keener to sell)
        When I < 0 (short):
            delta_bid decreases (tighter bid, keener to buy)
            delta_ask increases (wider ask, less keen to sell)

        Parameters
        ----------
        inventory : int
        time_remaining : float

        Returns
        -------
        (delta_bid, delta_ask)
            Offsets from mid price. Both positive.
        """
        gamma = self.gamma_risk
        sigma_sq = self.sigma ** 2
        half_spread = self.optimal_spread(time_remaining) / 2.0
        inventory_adj = inventory * gamma * sigma_sq * time_remaining

        delta_bid = half_spread + inventory_adj
        delta_ask = half_spread - inventory_adj

        # Ensure offsets are positive (can't post inside the spread)
        delta_bid = max(delta_bid, 1e-6)
        delta_ask = max(delta_ask, 1e-6)

        return delta_bid, delta_ask

    def optimal_quotes(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float,
    ) -> Tuple[float, float]:
        """Compute optimal bid and ask prices.

        bid = pseudo_mid - optimal_spread/2 = mid - delta_bid
        ask = pseudo_mid + optimal_spread/2 = mid + delta_ask

        Parameters
        ----------
        mid_price : float
        inventory : int
        time_remaining : float

        Returns
        -------
        (bid_price, ask_price)
        """
        delta_bid, delta_ask = self.optimal_offsets(inventory, time_remaining)
        bid = mid_price - delta_bid
        ask = mid_price + delta_ask
        return bid, ask

    def expected_pnl_per_step(
        self,
        inventory: int,
        time_remaining: float,
    ) -> float:
        """Expected PnL per step from optimal quoting.

        E[pnl] = lambda_b * delta_b + lambda_a * delta_a - gamma * sigma^2 * I^2

        where lambda(delta) = A * exp(-k * delta).
        """
        delta_bid, delta_ask = self.optimal_offsets(inventory, time_remaining)
        k = self.fill_rate_k
        A = self.fill_rate_A

        bid_fill_rate = A * math.exp(-k * delta_bid)
        ask_fill_rate = A * math.exp(-k * delta_ask)

        revenue = bid_fill_rate * delta_bid + ask_fill_rate * delta_ask
        inv_cost = self.gamma_risk * self.sigma ** 2 * inventory ** 2

        return revenue - inv_cost


# ============================================================================
# RLMarketMaker -- RL-enhanced market maker (Ch 10.3 + Ch 13-14)
# ============================================================================


class RLMarketMaker:
    """RL-enhanced market maker using Actor-Critic.

    Extends Avellaneda-Stoikov to handle:
    1. Non-exponential fill rates (empirical from LOB data)
    2. Adverse selection (informed traders who know price direction)
    3. Queue priority (our position in the order book queue)
    4. Multiple instruments (portfolio of market-making)
    5. Non-stationary volatility (regime changes)
    6. Asymmetric information (different fill rates on each side)

    Architecture:
        Actor: state -> (bid_offset, ask_offset) in R+^2
               Uses softplus to ensure positive offsets
        Critic: state -> expected total PnL (value function)

    State features:
        [0] time_remaining / T
        [1] inventory / max_inventory
        [2] mid_price_change
        [3] spread (normalized)
        [4] imbalance
        [5] recent_vol
        [6] recent_flow
        [7] inventory_risk

    Training objective: maximize total PnL (sum of per-step rewards)
    with entropy regularization for exploration.

    References:
        Rao & Jelvis Ch 10.3
        Spooner et al. (2018), "Market Making via Reinforcement Learning"
        Gueant & Manziuk (2019), "Deep RL for Market Making"
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 2,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        max_inventory: int = 10,
        entropy_coef: float = 0.01,
        device: str = "auto",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for RLMarketMaker")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_inventory = max_inventory
        self.entropy_coef = entropy_coef
        self.device = _get_device(device)

        # --- Actor: state -> (bid_offset_mean, ask_offset_mean, log_stds) ---
        actor_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, h))
            actor_layers.append(nn.ReLU())
            prev_dim = h
        self.actor_body = nn.Sequential(*actor_layers).to(self.device)
        # Mean outputs: one for bid offset, one for ask offset
        self.actor_mean = nn.Linear(prev_dim, action_dim).to(self.device)
        # Log standard deviation (learnable)
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim, device=self.device) - 1.0
        )

        # --- Critic: state -> value ---
        critic_layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, h))
            critic_layers.append(nn.ReLU())
            prev_dim = h
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        # Optimizers
        actor_params = (
            list(self.actor_body.parameters())
            + list(self.actor_mean.parameters())
            + [self.actor_log_std]
        )
        self.actor_optimizer = optim.Adam(actor_params, lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )
        self._trained = False

    def _get_action_distribution(
        self, state_tensor: torch.Tensor
    ) -> TorchNormal:
        """Get Gaussian action distribution for quote offsets."""
        features = self.actor_body(state_tensor)
        # Use softplus to ensure positive offsets
        mean = nn.functional.softplus(self.actor_mean(features))
        std = torch.exp(self.actor_log_std.clamp(-5, 2)).expand_as(mean)
        return TorchNormal(mean, std)

    def _select_action(
        self, state_features: np.ndarray, explore: bool = True
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select quote offsets.

        Returns (offsets, log_prob, entropy).
        """
        state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        dist = self._get_action_distribution(state_t)

        if explore:
            action = dist.sample()
        else:
            action = dist.mean

        # Ensure positive offsets
        action = torch.clamp(action, min=1e-4)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        offsets = action.squeeze(0).detach().cpu().numpy()
        return offsets, log_prob.squeeze(0), entropy.squeeze(0)

    def train(
        self,
        env: MarketMakingMDP,
        num_episodes: int = 50000,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """Train the market maker using A2C with entropy regularization.

        Training loop:
        1. Collect episode: at each step, quote bid/ask offsets
        2. Compute returns from PnL rewards
        3. Advantage = return - baseline
        4. Update actor with policy gradient + entropy bonus
        5. Update critic with value regression

        Parameters
        ----------
        env : MarketMakingMDP
        num_episodes : int
        seed : int
        verbose : bool

        Returns
        -------
        dict with training history.
        """
        rng = np.random.default_rng(seed)
        history: dict[str, list] = {
            "episode_pnl": [],
            "episode_spread": [],
            "max_inventory": [],
            "actor_losses": [],
            "critic_losses": [],
        }

        for ep in range(1, num_episodes + 1):
            state = env.initial_state()
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []
            spreads: list[float] = []
            max_inv = 0

            done = False
            while not done:
                features = env.state_to_features(state)

                # Critic
                state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                value = self.critic(state_t).squeeze()
                values.append(value)

                # Actor
                offsets, log_prob, entropy = self._select_action(features, explore=True)
                log_probs.append(log_prob)
                entropies.append(entropy)

                bid_offset, ask_offset = offsets[0], offsets[1]
                spreads.append(bid_offset + ask_offset)

                # Step environment
                next_state, reward, done = env.step(
                    state, bid_offset, ask_offset, rng
                )
                rewards.append(reward)
                max_inv = max(max_inv, abs(next_state.inventory))
                state = next_state

            # Compute returns
            T = len(rewards)
            if T == 0:
                continue

            returns = np.zeros(T, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns[t] = G

            returns_t = torch.FloatTensor(returns).to(self.device)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)

            # Advantages
            advantages = returns_t - values_t.detach()
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # Actor loss = -policy_gradient - entropy_bonus
            policy_loss = -(log_probs_t * advantages).mean()
            entropy_bonus = -self.entropy_coef * entropies_t.mean()
            actor_loss = policy_loss + entropy_bonus

            # Critic loss
            critic_loss = nn.functional.mse_loss(values_t, returns_t)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_body.parameters())
                + list(self.actor_mean.parameters())
                + [self.actor_log_std],
                1.0,
            )
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # History
            total_pnl = sum(rewards)
            history["episode_pnl"].append(total_pnl)
            history["episode_spread"].append(float(np.mean(spreads)) if spreads else 0.0)
            history["max_inventory"].append(max_inv)
            history["actor_losses"].append(actor_loss.item())
            history["critic_losses"].append(critic_loss.item())

            if verbose and ep % max(1, num_episodes // 20) == 0:
                recent_pnl = history["episode_pnl"][-100:]
                recent_inv = history["max_inventory"][-100:]
                logger.info(
                    f"Episode {ep}/{num_episodes} | "
                    f"PnL: {np.mean(recent_pnl):.4f} +/- {np.std(recent_pnl):.4f} | "
                    f"Max inv: {np.mean(recent_inv):.1f} | "
                    f"Spread: {np.mean(history['episode_spread'][-100:]):.6f}"
                )

        self._trained = True
        return history

    def get_quotes(
        self,
        state: MarketMakingState,
        env: MarketMakingMDP,
    ) -> Tuple[float, float]:
        """Get optimal bid and ask prices.

        Parameters
        ----------
        state : MarketMakingState
        env : MarketMakingMDP

        Returns
        -------
        (bid_price, ask_price)
        """
        features = env.state_to_features(state)
        state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self._get_action_distribution(state_t)
            offsets = dist.mean.squeeze(0).cpu().numpy()

        bid_offset = max(offsets[0], 1e-4)
        ask_offset = max(offsets[1], 1e-4)
        bid_price = state.mid_price - bid_offset
        ask_price = state.mid_price + ask_offset
        return bid_price, ask_price

    def get_offsets(
        self,
        state: MarketMakingState,
        env: MarketMakingMDP,
    ) -> Tuple[float, float]:
        """Get optimal bid and ask offsets from mid.

        Returns
        -------
        (bid_offset, ask_offset)
        """
        features = env.state_to_features(state)
        state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self._get_action_distribution(state_t)
            offsets = dist.mean.squeeze(0).cpu().numpy()
        return max(offsets[0], 1e-4), max(offsets[1], 1e-4)

    def evaluate(
        self,
        env: MarketMakingMDP,
        num_episodes: int = 1000,
        benchmark: Optional[AvellanedaStoikovSolution] = None,
        seed: int = 123,
    ) -> dict:
        """Compare RL market maker vs Avellaneda-Stoikov analytical.

        Parameters
        ----------
        env : MarketMakingMDP
        num_episodes : int
        benchmark : AvellanedaStoikovSolution or None
        seed : int

        Returns
        -------
        dict with keys:
            rl_pnl_mean, rl_pnl_std, rl_sharpe,
            as_pnl_mean, as_pnl_std, as_sharpe (if benchmark given),
            rl_max_inventory, rl_avg_spread, rl_fill_rate
        """
        rng = np.random.default_rng(seed)

        # --- RL agent ---
        rl_pnls: list[float] = []
        rl_max_invs: list[int] = []
        rl_spreads: list[float] = []
        rl_fills: list[int] = []

        for _ in range(num_episodes):
            state = env.initial_state()
            done = False
            ep_fills = 0
            ep_spreads: list[float] = []
            max_inv = 0

            while not done:
                bid_price, ask_price = self.get_quotes(state, env)
                bid_off = state.mid_price - bid_price
                ask_off = ask_price - state.mid_price
                ep_spreads.append(bid_off + ask_off)

                old_inv = state.inventory
                state, reward, done = env.step(state, bid_off, ask_off, rng)
                if state.inventory != old_inv:
                    ep_fills += 1
                max_inv = max(max_inv, abs(state.inventory))

            rl_pnls.append(state.pnl)
            rl_max_invs.append(max_inv)
            rl_spreads.append(float(np.mean(ep_spreads)) if ep_spreads else 0.0)
            rl_fills.append(ep_fills)

        rl_pnl_arr = np.array(rl_pnls)
        rl_sharpe = 0.0
        if len(rl_pnl_arr) > 1 and np.std(rl_pnl_arr, ddof=1) > 1e-10:
            rl_sharpe = float(np.mean(rl_pnl_arr) / np.std(rl_pnl_arr, ddof=1))

        result: dict[str, Any] = {
            "rl_pnl_mean": float(np.mean(rl_pnl_arr)),
            "rl_pnl_std": float(np.std(rl_pnl_arr, ddof=1)) if len(rl_pnl_arr) > 1 else 0.0,
            "rl_sharpe": rl_sharpe,
            "rl_max_inventory": float(np.mean(rl_max_invs)),
            "rl_avg_spread": float(np.mean(rl_spreads)),
            "rl_fill_rate": float(np.mean(rl_fills)),
        }

        # --- Avellaneda-Stoikov benchmark ---
        if benchmark is not None:
            as_pnls: list[float] = []
            for _ in range(num_episodes):
                state = env.initial_state()
                done = False
                total_steps = env.num_steps
                while not done:
                    t_rem = (total_steps - state.time_step) * env.dt
                    delta_bid, delta_ask = benchmark.optimal_offsets(
                        state.inventory, max(t_rem, 1e-10)
                    )
                    state, reward, done = env.step(
                        state, delta_bid, delta_ask, rng
                    )
                as_pnls.append(state.pnl)

            as_pnl_arr = np.array(as_pnls)
            as_sharpe = 0.0
            if len(as_pnl_arr) > 1 and np.std(as_pnl_arr, ddof=1) > 1e-10:
                as_sharpe = float(np.mean(as_pnl_arr) / np.std(as_pnl_arr, ddof=1))

            result["as_pnl_mean"] = float(np.mean(as_pnl_arr))
            result["as_pnl_std"] = float(np.std(as_pnl_arr, ddof=1)) if len(as_pnl_arr) > 1 else 0.0
            result["as_sharpe"] = as_sharpe

        return result


# ============================================================================
# InventoryRiskManager -- Risk management for market-making
# ============================================================================


class InventoryRiskManager:
    """Inventory risk management for market-making.

    Implements multiple layers of inventory risk control:

    1. Position limits (hard max inventory):
        Cannot accumulate beyond max_inventory in either direction.

    2. Skew quoting (soft inventory control):
        When long, widen bid (buy less) and tighten ask (sell more).
        When short, tighten bid (buy more) and widen ask (sell less).
        Skew amount = skew_factor * |inventory| / max_inventory * base_spread

    3. Emergency flattening:
        When |inventory| > emergency_threshold * max_inventory,
        send a market order to flatten immediately.
        This is expensive but prevents catastrophic losses.

    4. Greeks-based hedging (for options market-making):
        Hedge delta exposure with the underlying.
        Size = -delta * contracts * multiplier

    References:
        Gueant, Lehalle & Fernandez-Tapia (2012)
        Cartea, Jaimungal & Penalva (2015), "Algorithmic and High-Frequency Trading"
    """

    def __init__(
        self,
        max_inventory: int = 10,
        skew_factor: float = 0.5,
        emergency_threshold: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        max_inventory : int
            Hard limit on absolute inventory.
        skew_factor : float
            How aggressively to skew quotes based on inventory.
            0 = no skew, 1 = maximum skew.
        emergency_threshold : float
            Fraction of max_inventory that triggers emergency flattening.
            E.g., 0.9 means flatten when |I| >= 0.9 * max_inventory.
        """
        self.max_inventory = max_inventory
        self.skew_factor = skew_factor
        self.emergency_threshold = emergency_threshold

    def should_emergency_flatten(self, inventory: int) -> bool:
        """Check if inventory breaches emergency threshold.

        Parameters
        ----------
        inventory : int
            Current inventory position.

        Returns
        -------
        bool
            True if emergency flattening is needed.
        """
        return abs(inventory) >= self.emergency_threshold * self.max_inventory

    def emergency_flatten_size(self, inventory: int) -> int:
        """Compute emergency market order size to reduce inventory.

        Strategy: reduce inventory to 50% of max to give room to breathe,
        rather than flattening to zero (which is too aggressive).

        Parameters
        ----------
        inventory : int

        Returns
        -------
        int
            Number of units to trade (positive = sell, negative = buy).
        """
        target = int(self.max_inventory * 0.5) * (1 if inventory > 0 else -1)
        return inventory - target

    def adjust_quotes(
        self,
        bid_offset: float,
        ask_offset: float,
        inventory: int,
    ) -> Tuple[float, float]:
        """Apply inventory risk adjustments to raw quotes.

        The adjustment creates an asymmetry:
        - When long: widen bid, tighten ask (to reduce long exposure)
        - When short: tighten bid, widen ask (to reduce short exposure)

        The magnitude of the skew is proportional to |inventory|/max_inventory.

        Parameters
        ----------
        bid_offset : float
            Raw bid offset from mid (positive = below mid).
        ask_offset : float
            Raw ask offset from mid (positive = above mid).
        inventory : int
            Current inventory.

        Returns
        -------
        (adjusted_bid_offset, adjusted_ask_offset)
        """
        if self.max_inventory == 0:
            return bid_offset, ask_offset

        # Inventory fraction: -1 to +1
        inv_frac = inventory / self.max_inventory

        # Skew amount proportional to inventory
        base_spread = bid_offset + ask_offset
        skew = self.skew_factor * inv_frac * base_spread / 2.0

        # When long (inv_frac > 0): increase bid_offset (wider bid), decrease ask_offset (tighter ask)
        # When short (inv_frac < 0): decrease bid_offset (tighter bid), increase ask_offset (wider ask)
        adj_bid = bid_offset + skew
        adj_ask = ask_offset - skew

        # Ensure offsets remain positive
        min_offset = 1e-6
        adj_bid = max(adj_bid, min_offset)
        adj_ask = max(adj_ask, min_offset)

        # Hard limit: if at max inventory, prevent further accumulation
        if inventory >= self.max_inventory:
            adj_bid = float("inf")  # don't bid (can't buy more)
        elif inventory <= -self.max_inventory:
            adj_ask = float("inf")  # don't ask (can't sell more)

        return adj_bid, adj_ask

    def compute_delta_hedge(
        self,
        inventory: int,
        delta_per_unit: float,
        multiplier: float = 1.0,
    ) -> float:
        """Compute delta hedge size for options market-making.

        When market-making options, the inventory creates delta exposure.
        Hedge by trading the underlying:
            hedge_size = -inventory * delta_per_unit * multiplier

        Parameters
        ----------
        inventory : int
            Net option inventory (positive = long options).
        delta_per_unit : float
            BS delta per option contract.
        multiplier : float
            Contract multiplier (e.g., 75 for NIFTY options).

        Returns
        -------
        float
            Number of underlying units to trade (positive = buy, negative = sell).
        """
        return -inventory * delta_per_unit * multiplier

    def risk_metrics(
        self,
        inventory: int,
        mid_price: float,
        sigma: float,
    ) -> dict:
        """Compute current inventory risk metrics.

        Parameters
        ----------
        inventory : int
        mid_price : float
        sigma : float
            Per-step volatility.

        Returns
        -------
        dict with keys:
            inventory_value: dollar value of inventory
            var_1step: 1-step Value at Risk (95%)
            var_10step: 10-step VaR (95%)
            inventory_utilization: |I| / max_I
        """
        inv_value = inventory * mid_price
        # VaR(95%) ~ 1.645 * sigma * |position_value|
        var_1 = 1.645 * sigma * abs(inv_value)
        var_10 = 1.645 * sigma * math.sqrt(10) * abs(inv_value)
        utilization = abs(inventory) / max(self.max_inventory, 1)

        return {
            "inventory_value": inv_value,
            "var_1step": var_1,
            "var_10step": var_10,
            "inventory_utilization": utilization,
            "emergency_triggered": self.should_emergency_flatten(inventory),
        }
