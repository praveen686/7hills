"""Unified Decision Layer — Joint TFT + RL end-to-end training.

Provides three components for joint TFT feature extraction and RL policy learning:

1. **UnifiedDecisionLayer**: End-to-end differentiable module that pipes
   market state through a TFT encoder to produce features, then through
   an RL policy network (actor-critic) to produce trading actions.
   Supports curriculum learning (freeze TFT for N epochs) and
   differential LR scaling (TFT at 0.1x).

2. **JointTrainingPipeline**: Four-phase training protocol:
   Phase 1: Pre-train TFT encoder on historical data (Sharpe loss)
   Phase 2: Freeze TFT, train RL policy on environment
   Phase 3: Unfreeze TFT, joint fine-tuning with scaled LR
   Phase 4: Walk-forward OOS evaluation

3. **HierarchicalDecisionLayer**: Multi-timeframe hierarchical RL with
   goal-conditioned low-level policy. High-level (daily) sets regime +
   target allocation; low-level (per-minute) executes orders.

Architecture:
    market_state -> TFT_encoder -> features -> RL_policy -> action

References:
    - Lim et al. (2021), TFT for multi-horizon forecasting
    - Schulman et al. (2017), PPO
    - Nachum et al. (2018), Hierarchical RL with off-policy correction
    - Vezhnevets et al. (2017), FeUdal Networks for hierarchical RL
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


# ============================================================================
# UnifiedDecisionLayer
# ============================================================================

if _HAS_TORCH:

    class _PolicyNetwork(nn.Module):
        """Actor-Critic policy head for the unified decision layer.

        Takes TFT-encoded features and produces:
        - Action mean + log_std (Gaussian policy, Tanh-squashed)
        - Value estimate V(s)

        Parameters
        ----------
        feature_dim : int
            Dimension of input features from TFT encoder.
        action_dim : int
            Number of continuous action dimensions (e.g. 1 for single-asset).
        hidden_dim : int
            Width of the hidden layers.
        """

        def __init__(
            self,
            feature_dim: int,
            action_dim: int = 1,
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.action_dim = action_dim

            # Shared trunk
            self.shared = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
            )

            # Actor head: mean + log_std
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

            # Critic head: scalar value
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(
            self, features: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through the policy network.

            Parameters
            ----------
            features : (batch, feature_dim)

            Returns
            -------
            mean : (batch, action_dim) — Gaussian mean
            log_std : (batch, action_dim) — clamped log standard deviation
            value : (batch, 1) — state value estimate
            shared_h : (batch, hidden_dim) — shared hidden for analysis
            """
            h = self.shared(features)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(-5.0, 2.0)
            value = self.value_head(h)
            return mean, log_std, value, h

    class UnifiedDecisionLayer(nn.Module):
        """Joint TFT feature extractor + RL policy network.

        Architecture:
        market_state -> TFT_encoder -> features -> RL_policy -> action

        End-to-end differentiable: gradients flow from RL loss through TFT encoder.

        Parameters
        ----------
        tft_encoder : nn.Module
            Pre-trained TFT encoder (VSN + LSTM + attention layers).
            Must have a forward method accepting (batch, seq_len, n_features)
            and returning (batch, hidden_dim) or (batch, 1) with a method
            to extract hidden states.
        policy_network : nn.Module
            RL policy head (actor-critic). If None, a default _PolicyNetwork
            is created.
        freeze_tft_epochs : int
            Freeze TFT encoder for first N epochs (curriculum learning).
        tft_lr_scale : float
            Scale TFT learning rate relative to policy (0.1 = 10x slower).
        feature_dim : int
            Output dimension of the TFT encoder. If None, inferred from
            tft_encoder attributes (hidden_dim or d_hidden).
        action_dim : int
            Number of action dimensions. Default 1 (single-asset position).
        """

        def __init__(
            self,
            tft_encoder: nn.Module,
            policy_network: Optional[nn.Module] = None,
            freeze_tft_epochs: int = 5,
            tft_lr_scale: float = 0.1,
            feature_dim: Optional[int] = None,
            action_dim: int = 1,
        ) -> None:
            super().__init__()

            self.tft_encoder = tft_encoder
            self.freeze_tft_epochs = freeze_tft_epochs
            self.tft_lr_scale = tft_lr_scale
            self._current_epoch = 0

            # Infer feature dimension from encoder
            if feature_dim is not None:
                self._feature_dim = feature_dim
            elif hasattr(tft_encoder, "hidden_dim"):
                self._feature_dim = tft_encoder.hidden_dim
            elif hasattr(tft_encoder, "d_hidden"):
                self._feature_dim = tft_encoder.d_hidden
            else:
                raise ValueError(
                    "Cannot infer feature_dim from tft_encoder. "
                    "Pass feature_dim explicitly."
                )

            # Build default policy network if not provided
            if policy_network is not None:
                self.policy_network = policy_network
            else:
                self.policy_network = _PolicyNetwork(
                    feature_dim=self._feature_dim,
                    action_dim=action_dim,
                )

            # Apply initial freeze state
            self._update_freeze_state()

        @property
        def feature_dim(self) -> int:
            """Dimension of the TFT encoder output features."""
            return self._feature_dim

        def _update_freeze_state(self) -> None:
            """Freeze or unfreeze TFT encoder based on current epoch."""
            should_freeze = self._current_epoch < self.freeze_tft_epochs
            for param in self.tft_encoder.parameters():
                param.requires_grad = not should_freeze

        def set_epoch(self, epoch: int) -> None:
            """Set current epoch and update freeze state accordingly.

            Parameters
            ----------
            epoch : int
                Current training epoch (0-indexed).
            """
            self._current_epoch = epoch
            self._update_freeze_state()

        def is_tft_frozen(self) -> bool:
            """Check whether TFT encoder parameters are frozen."""
            for param in self.tft_encoder.parameters():
                return not param.requires_grad
            # No parameters means effectively frozen
            return True

        def encode(self, market_state: torch.Tensor) -> torch.Tensor:
            """Extract features via TFT encoder.

            Parameters
            ----------
            market_state : (batch, seq_len, n_features) or (batch, n_features)
                Raw market state input.

            Returns
            -------
            features : (batch, feature_dim) — encoded features.
            """
            raw_out = self.tft_encoder(market_state)

            # Handle different encoder output formats
            if isinstance(raw_out, tuple):
                # Some encoders return (output, attention_weights) or (mu, log_sigma)
                raw_out = raw_out[0]

            # If output is (batch, seq_len, hidden_dim), take last timestep
            if raw_out.dim() == 3:
                raw_out = raw_out[:, -1, :]

            # If output is (batch, 1), squeeze to (batch,) then unsqueeze back
            if raw_out.dim() == 2 and raw_out.shape[-1] == 1:
                # Encoder outputting scalar (e.g. position signal) -- need to
                # project to feature_dim. Use a linear projection if needed.
                if not hasattr(self, "_scalar_proj"):
                    self._scalar_proj = nn.Linear(1, self._feature_dim).to(
                        raw_out.device
                    )
                raw_out = self._scalar_proj(raw_out)

            return raw_out

        def forward(self, market_state: torch.Tensor) -> dict:
            """Full forward pass: TFT encode -> policy network -> action.

            Parameters
            ----------
            market_state : (batch, seq_len, n_features)
                Raw market state input.

            Returns
            -------
            dict with keys:
                action : (batch, action_dim) — trading action in [-1, 1]
                value : (batch, 1) — state value estimate
                features : (batch, feature_dim) — intermediate TFT features
                log_prob : (batch,) — action log probability (for PPO)
                entropy : (batch,) — policy entropy
            """
            # 1. Encode market state through TFT
            features = self.encode(market_state)

            # 2. Policy network forward pass
            mean, log_std, value, _ = self.policy_network(features)
            std = log_std.exp()

            # 3. Sample action from Gaussian, squash through Tanh
            dist = torch.distributions.Normal(mean, std)
            raw_sample = dist.rsample()  # reparameterized gradient
            action = torch.tanh(raw_sample)

            # 4. Log probability with Tanh squashing correction
            log_prob = dist.log_prob(raw_sample) - torch.log(
                1 - action.pow(2) + 1e-6
            )
            log_prob = log_prob.sum(dim=-1)

            # 5. Policy entropy
            entropy = dist.entropy().sum(dim=-1)

            return {
                "action": action,
                "value": value,
                "features": features.detach(),
                "log_prob": log_prob,
                "entropy": entropy,
            }

        def act_deterministic(self, market_state: torch.Tensor) -> dict:
            """Deterministic action (mean of policy, no sampling).

            Used during evaluation / inference.
            """
            features = self.encode(market_state)
            mean, log_std, value, _ = self.policy_network(features)
            action = torch.tanh(mean)

            return {
                "action": action,
                "value": value,
                "features": features.detach(),
            }

        def get_parameter_groups(self) -> list[dict]:
            """Return separate parameter groups for optimizer.

            TFT params get tft_lr_scale * base_lr.
            Policy params get base_lr (scale = 1.0).

            Returns
            -------
            list of dicts, each with 'params' and 'lr_scale' keys.
            """
            tft_params = list(self.tft_encoder.parameters())
            policy_params = list(self.policy_network.parameters())

            # Also include any auxiliary modules (e.g. _scalar_proj)
            aux_params = []
            if hasattr(self, "_scalar_proj"):
                aux_params = list(self._scalar_proj.parameters())

            groups = []
            if tft_params:
                groups.append({
                    "params": tft_params,
                    "lr_scale": self.tft_lr_scale,
                    "name": "tft_encoder",
                })
            if policy_params:
                groups.append({
                    "params": policy_params,
                    "lr_scale": 1.0,
                    "name": "policy_network",
                })
            if aux_params:
                groups.append({
                    "params": aux_params,
                    "lr_scale": 1.0,
                    "name": "auxiliary",
                })

            return groups

        def build_optimizer(
            self, base_lr: float = 3e-4, weight_decay: float = 1e-5
        ) -> torch.optim.Optimizer:
            """Build optimizer with differential learning rates.

            Parameters
            ----------
            base_lr : float
                Base learning rate (applied to policy network).
            weight_decay : float
                L2 regularization weight.

            Returns
            -------
            torch.optim.Adam optimizer with per-group LR.
            """
            param_groups = []
            for group in self.get_parameter_groups():
                if group["params"]:
                    param_groups.append({
                        "params": group["params"],
                        "lr": base_lr * group["lr_scale"],
                        "weight_decay": weight_decay,
                    })
            return torch.optim.Adam(param_groups)

    # ========================================================================
    # JointTrainingPipeline
    # ========================================================================

    class JointTrainingPipeline:
        """End-to-end training pipeline for TFT + RL.

        Training protocol:
        Phase 1: Pre-train TFT encoder on historical data (Sharpe loss)
        Phase 2: Freeze TFT, train RL policy on environment
        Phase 3: Unfreeze TFT, joint fine-tuning with scaled LR
        Phase 4: Walk-forward evaluation

        Parameters
        ----------
        unified_model : UnifiedDecisionLayer
            The joint TFT + RL model.
        env : object
            Training environment. Must support reset() -> state and
            step(action) -> (next_state, reward, done, info).
        config : dict, optional
            Configuration overrides. Supported keys:
            - base_lr (float): base learning rate (default 3e-4)
            - gamma (float): RL discount factor (default 0.99)
            - entropy_beta (float): entropy bonus coefficient (default 0.01)
            - ppo_clip (float): PPO clip ratio (default 0.2)
            - max_grad_norm (float): gradient clipping norm (default 5.0)
            - batch_size (int): mini-batch size (default 64)
            - gae_lambda (float): GAE lambda (default 0.95)
        """

        def __init__(
            self,
            unified_model: UnifiedDecisionLayer,
            env,
            config: Optional[dict] = None,
        ) -> None:
            self.model = unified_model
            self.env = env

            cfg = config or {}
            self.base_lr = cfg.get("base_lr", 3e-4)
            self.gamma = cfg.get("gamma", 0.99)
            self.entropy_beta = cfg.get("entropy_beta", 0.01)
            self.ppo_clip = cfg.get("ppo_clip", 0.2)
            self.max_grad_norm = cfg.get("max_grad_norm", 5.0)
            self.batch_size = cfg.get("batch_size", 64)
            self.gae_lambda = cfg.get("gae_lambda", 0.95)

            self.optimizer = unified_model.build_optimizer(
                base_lr=self.base_lr
            )
            self._phase_history: list[str] = []
            self._metrics: dict[str, dict] = {}

        @property
        def _device(self) -> torch.device:
            """Infer device from model parameters."""
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        @property
        def phase_history(self) -> list[str]:
            """Ordered list of completed phases."""
            return list(self._phase_history)

        def _collect_rollout(
            self, n_steps: int
        ) -> tuple[list, list, list, list, list, list]:
            """Collect a rollout of n_steps from the environment.

            Returns lists of: states, actions, rewards, log_probs, values, dones.
            """
            states_list = []
            actions_list = []
            rewards_list = []
            log_probs_list = []
            values_list = []
            dones_list = []

            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Gymnasium-style (obs, info)

            steps_done = 0
            while steps_done < n_steps:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    out = self.model(state_t)

                action_np = out["action"].squeeze(0).cpu().numpy()
                log_prob_np = out["log_prob"].cpu().item()
                value_np = out["value"].squeeze().cpu().item()

                result = self.env.step(action_np)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                elif len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    raise ValueError(
                        f"env.step returned {len(result)} values, expected 4 or 5"
                    )

                states_list.append(state)
                actions_list.append(action_np)
                rewards_list.append(float(reward))
                log_probs_list.append(log_prob_np)
                values_list.append(value_np)
                dones_list.append(float(done))

                if done:
                    state = self.env.reset()
                    if isinstance(state, tuple):
                        state = state[0]
                else:
                    state = next_state

                steps_done += 1

            return (
                states_list,
                actions_list,
                rewards_list,
                log_probs_list,
                values_list,
                dones_list,
            )

        def _compute_gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute Generalized Advantage Estimation (GAE).

            Parameters
            ----------
            rewards : (T,)
            values : (T,)
            dones : (T,)

            Returns
            -------
            advantages : (T,)
            returns : (T,)
            """
            T = len(rewards)
            advantages = np.zeros(T, dtype=np.float32)
            last_gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = 0.0
                else:
                    next_val = values[t + 1]

                delta = (
                    rewards[t]
                    + self.gamma * next_val * (1.0 - dones[t])
                    - values[t]
                )
                last_gae = (
                    delta
                    + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
                )
                advantages[t] = last_gae

            returns = advantages + values
            return advantages, returns

        def _ppo_update(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            old_log_probs: np.ndarray,
            advantages: np.ndarray,
            returns: np.ndarray,
            n_epochs: int = 4,
        ) -> dict:
            """Perform PPO update on collected data.

            Returns dict with actor_loss, critic_loss, entropy.
            """
            N = len(states)
            dev = self._device
            states_t = torch.FloatTensor(states).to(dev)
            actions_t = torch.FloatTensor(actions).to(dev)
            old_lp_t = torch.FloatTensor(old_log_probs).to(dev)
            adv_t = torch.FloatTensor(advantages).to(dev)
            ret_t = torch.FloatTensor(returns).to(dev)

            # Normalize advantages
            if adv_t.std() > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            n_updates = 0

            for _ in range(n_epochs):
                perm = torch.randperm(N, device=dev)
                for start in range(0, N, self.batch_size):
                    end = min(start + self.batch_size, N)
                    idx = perm[start:end]

                    batch_states = states_t[idx]
                    batch_actions = actions_t[idx]
                    batch_old_lp = old_lp_t[idx]
                    batch_adv = adv_t[idx]
                    batch_ret = ret_t[idx]

                    out = self.model(batch_states)
                    new_log_prob = out["log_prob"]
                    new_value = out["value"].squeeze(-1)
                    entropy = out["entropy"]

                    # PPO clipped surrogate
                    ratio = torch.exp(new_log_prob - batch_old_lp)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip
                    )
                    actor_loss = -torch.min(
                        ratio * batch_adv, clipped * batch_adv
                    ).mean()

                    # Value loss
                    critic_loss = F.mse_loss(new_value, batch_ret)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    loss = (
                        actor_loss
                        + 0.5 * critic_loss
                        + self.entropy_beta * entropy_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

            n_updates = max(n_updates, 1)
            return {
                "actor_loss": total_actor_loss / n_updates,
                "critic_loss": total_critic_loss / n_updates,
                "entropy": total_entropy / n_updates,
            }

        def phase1_pretrain_tft(
            self, data: np.ndarray, epochs: int = 30
        ) -> dict:
            """Pre-train TFT encoder on historical returns (Sharpe loss).

            Parameters
            ----------
            data : (n_samples, seq_len, n_features)
                Historical market data sequences. The last feature column is
                assumed to be the next-day return target if no separate target
                is provided.
            epochs : int
                Number of pre-training epochs.

            Returns
            -------
            dict with keys: losses, final_loss, best_epoch
            """
            self._phase_history.append("phase1_pretrain_tft")

            # Ensure TFT is trainable for pre-training
            for param in self.model.tft_encoder.parameters():
                param.requires_grad = True

            # Simple Sharpe-based pre-training using the encoder's forward
            dev = self._device
            data_t = torch.FloatTensor(data).to(dev)

            # Use all features as input, last column as return target
            if data_t.shape[-1] > 1:
                X = data_t[:, :, :-1]
                y = data_t[:, -1, -1]  # last timestep, last feature = return
            else:
                X = data_t
                y = torch.zeros(data_t.shape[0], device=dev)

            # Temporary optimizer for TFT-only pre-training
            tft_optimizer = torch.optim.Adam(
                self.model.tft_encoder.parameters(),
                lr=self.base_lr,
                weight_decay=1e-5,
            )

            losses = []
            best_loss = float("inf")
            best_epoch = 0
            batch_size = min(self.batch_size, len(X))

            self.model.tft_encoder.train()

            for epoch in range(epochs):
                perm = torch.randperm(len(X), device=dev)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(X), batch_size):
                    end = min(start + batch_size, len(X))
                    idx = perm[start:end]

                    batch_X = X[idx]
                    batch_y = y[idx]

                    # Forward through encoder to get positions
                    raw_out = self.model.tft_encoder(batch_X)
                    if isinstance(raw_out, tuple):
                        raw_out = raw_out[0]
                    if raw_out.dim() == 3:
                        raw_out = raw_out[:, -1, :]

                    # Reduce to (batch,) scalar positions for Sharpe
                    if raw_out.dim() == 2 and raw_out.shape[-1] > 1:
                        # Multi-dim encoder output: mean-pool to scalar
                        positions = raw_out.mean(dim=-1)
                    else:
                        positions = raw_out.squeeze(-1)
                    if positions.dim() == 0:
                        positions = positions.unsqueeze(0)

                    # Sharpe loss
                    strat_ret = positions * batch_y
                    mean_r = strat_ret.mean()
                    std_r = strat_ret.std(correction=1)
                    if std_r > 1e-8:
                        loss = -(mean_r / std_r) * math.sqrt(252)
                    else:
                        loss = torch.tensor(
                            0.0, device=dev, requires_grad=True
                        )

                    tft_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.tft_encoder.parameters(), 1.0
                    )
                    tft_optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch

                if epoch % 10 == 0:
                    logger.info(
                        "Phase1 epoch %d/%d: loss=%.4f", epoch, epochs, avg_loss
                    )

            # Restore freeze state for phase 2
            self.model.set_epoch(0)

            metrics = {
                "losses": losses,
                "final_loss": losses[-1] if losses else float("nan"),
                "best_epoch": best_epoch,
            }
            self._metrics["phase1"] = metrics
            return metrics

        def phase2_train_rl(self, n_steps: int = 100_000) -> dict:
            """Train RL policy with frozen TFT encoder.

            Parameters
            ----------
            n_steps : int
                Total environment steps for RL training.

            Returns
            -------
            dict with keys: actor_losses, critic_losses, entropies, episode_rewards
            """
            self._phase_history.append("phase2_train_rl")

            # Freeze TFT encoder
            self.model.set_epoch(0)  # epoch < freeze_tft_epochs => frozen

            rollout_len = min(2048, n_steps)
            actor_losses = []
            critic_losses = []
            entropies = []
            episode_rewards = []

            self.model.train()
            steps_done = 0

            while steps_done < n_steps:
                current_rollout = min(rollout_len, n_steps - steps_done)

                (
                    states,
                    actions,
                    rewards,
                    log_probs,
                    values,
                    dones,
                ) = self._collect_rollout(current_rollout)

                states_arr = np.array(states, dtype=np.float32)
                actions_arr = np.array(actions, dtype=np.float32)
                rewards_arr = np.array(rewards, dtype=np.float32)
                log_probs_arr = np.array(log_probs, dtype=np.float32)
                values_arr = np.array(values, dtype=np.float32)
                dones_arr = np.array(dones, dtype=np.float32)

                advantages, returns = self._compute_gae(
                    rewards_arr, values_arr, dones_arr
                )

                update_metrics = self._ppo_update(
                    states_arr,
                    actions_arr,
                    log_probs_arr,
                    advantages,
                    returns,
                )

                actor_losses.append(update_metrics["actor_loss"])
                critic_losses.append(update_metrics["critic_loss"])
                entropies.append(update_metrics["entropy"])

                # Track episode rewards
                ep_reward = 0.0
                for r, d in zip(rewards, dones):
                    ep_reward += r
                    if d:
                        episode_rewards.append(ep_reward)
                        ep_reward = 0.0

                steps_done += current_rollout

            metrics = {
                "actor_losses": actor_losses,
                "critic_losses": critic_losses,
                "entropies": entropies,
                "episode_rewards": episode_rewards,
                "total_steps": steps_done,
            }
            self._metrics["phase2"] = metrics
            return metrics

        def phase3_joint_finetune(self, n_steps: int = 50_000) -> dict:
            """Joint fine-tuning. TFT at tft_lr_scale * base_lr, policy at base_lr.

            Parameters
            ----------
            n_steps : int
                Total environment steps for joint fine-tuning.

            Returns
            -------
            dict with keys: actor_losses, critic_losses, entropies, episode_rewards
            """
            self._phase_history.append("phase3_joint_finetune")

            # Unfreeze TFT encoder
            self.model.set_epoch(self.model.freeze_tft_epochs)

            # Rebuild optimizer with differential LR
            self.optimizer = self.model.build_optimizer(
                base_lr=self.base_lr
            )

            rollout_len = min(2048, n_steps)
            actor_losses = []
            critic_losses = []
            entropies = []
            episode_rewards = []

            self.model.train()
            steps_done = 0

            while steps_done < n_steps:
                current_rollout = min(rollout_len, n_steps - steps_done)

                (
                    states,
                    actions,
                    rewards,
                    log_probs,
                    values,
                    dones,
                ) = self._collect_rollout(current_rollout)

                states_arr = np.array(states, dtype=np.float32)
                actions_arr = np.array(actions, dtype=np.float32)
                rewards_arr = np.array(rewards, dtype=np.float32)
                log_probs_arr = np.array(log_probs, dtype=np.float32)
                values_arr = np.array(values, dtype=np.float32)
                dones_arr = np.array(dones, dtype=np.float32)

                advantages, returns = self._compute_gae(
                    rewards_arr, values_arr, dones_arr
                )

                update_metrics = self._ppo_update(
                    states_arr,
                    actions_arr,
                    log_probs_arr,
                    advantages,
                    returns,
                )

                actor_losses.append(update_metrics["actor_loss"])
                critic_losses.append(update_metrics["critic_loss"])
                entropies.append(update_metrics["entropy"])

                ep_reward = 0.0
                for r, d in zip(rewards, dones):
                    ep_reward += r
                    if d:
                        episode_rewards.append(ep_reward)
                        ep_reward = 0.0

                steps_done += current_rollout

            metrics = {
                "actor_losses": actor_losses,
                "critic_losses": critic_losses,
                "entropies": entropies,
                "episode_rewards": episode_rewards,
                "total_steps": steps_done,
            }
            self._metrics["phase3"] = metrics
            return metrics

        def phase4_evaluate(self, test_data: np.ndarray) -> dict:
            """Walk-forward OOS evaluation.

            Parameters
            ----------
            test_data : (n_steps, seq_len, n_features)
                Out-of-sample market state sequences. The last feature column
                is treated as the actual return for Sharpe computation.

            Returns
            -------
            dict with keys: positions, returns, sharpe, total_return, max_drawdown
            """
            self._phase_history.append("phase4_evaluate")

            self.model.eval()
            dev = self._device
            test_t = torch.FloatTensor(test_data).to(dev)

            if test_t.shape[-1] > 1:
                X = test_t[:, :, :-1]
                actual_returns = test_t[:, -1, -1].cpu().numpy()
            else:
                X = test_t
                actual_returns = np.zeros(len(test_t))

            positions = []
            with torch.no_grad():
                # Process in batches
                bs = min(self.batch_size, len(X))
                for start in range(0, len(X), bs):
                    end = min(start + bs, len(X))
                    batch = X[start:end]
                    out = self.model.act_deterministic(batch)
                    pos = out["action"].squeeze(-1).cpu().numpy()
                    if pos.ndim == 0:
                        pos = np.array([pos.item()])
                    positions.append(pos)

            positions_arr = np.concatenate(positions)
            strategy_returns = positions_arr * actual_returns

            # Compute Sharpe
            if len(strategy_returns) > 1 and np.std(strategy_returns, ddof=1) > 1e-10:
                sharpe = (
                    np.mean(strategy_returns)
                    / np.std(strategy_returns, ddof=1)
                    * math.sqrt(252)
                )
            else:
                sharpe = 0.0

            # Total return (geometric)
            cumulative = np.cumprod(1.0 + strategy_returns) - 1.0
            total_return = cumulative[-1] if len(cumulative) > 0 else 0.0

            # Max drawdown
            equity = np.cumprod(1.0 + strategy_returns)
            running_max = np.maximum.accumulate(equity)
            drawdown = np.where(
                running_max > 0, (running_max - equity) / running_max, 0.0
            )
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

            metrics = {
                "positions": positions_arr,
                "returns": strategy_returns,
                "sharpe": sharpe,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "n_samples": len(positions_arr),
            }
            self._metrics["phase4"] = metrics
            return metrics

        def run_full_pipeline(
            self,
            train_data: np.ndarray,
            test_data: np.ndarray,
            pretrain_epochs: int = 30,
            rl_steps: int = 100_000,
            finetune_steps: int = 50_000,
        ) -> dict:
            """Run all 4 phases sequentially.

            Parameters
            ----------
            train_data : (n_samples, seq_len, n_features)
                Training sequences (last feature = return target).
            test_data : (n_samples, seq_len, n_features)
                Test sequences.
            pretrain_epochs : int
                Phase 1 epochs.
            rl_steps : int
                Phase 2 total steps.
            finetune_steps : int
                Phase 3 total steps.

            Returns
            -------
            dict with per-phase metrics and overall summary.
            """
            logger.info("Starting full pipeline: 4 phases")

            p1 = self.phase1_pretrain_tft(train_data, epochs=pretrain_epochs)
            logger.info(
                "Phase 1 complete: final_loss=%.4f, best_epoch=%d",
                p1["final_loss"],
                p1["best_epoch"],
            )

            p2 = self.phase2_train_rl(n_steps=rl_steps)
            logger.info(
                "Phase 2 complete: %d steps, %d episodes",
                p2["total_steps"],
                len(p2["episode_rewards"]),
            )

            p3 = self.phase3_joint_finetune(n_steps=finetune_steps)
            logger.info(
                "Phase 3 complete: %d steps, %d episodes",
                p3["total_steps"],
                len(p3["episode_rewards"]),
            )

            p4 = self.phase4_evaluate(test_data)
            logger.info(
                "Phase 4 complete: Sharpe=%.4f, Return=%.4f, MaxDD=%.4f",
                p4["sharpe"],
                p4["total_return"],
                p4["max_drawdown"],
            )

            return {
                "phase1": p1,
                "phase2": p2,
                "phase3": p3,
                "phase4": p4,
                "phase_history": list(self._phase_history),
            }

    # ========================================================================
    # HierarchicalDecisionLayer
    # ========================================================================

    class _HighLevelPolicy(nn.Module):
        """Daily regime detection + target allocation policy.

        Input: daily aggregated state (features, positions, vol, regime indicators)
        Output: target position, regime logits, confidence
        """

        def __init__(self, state_dim: int, n_regimes: int = 3) -> None:
            super().__init__()
            self.n_regimes = n_regimes
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ELU(),
            )
            # Target position (continuous in [-1, 1])
            self.position_head = nn.Sequential(
                nn.Linear(64, 1),
                nn.Tanh(),
            )
            # Regime classification
            self.regime_head = nn.Linear(64, n_regimes)
            # Confidence (sigmoid)
            self.confidence_head = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            # Value function
            self.value_head = nn.Linear(64, 1)

        def forward(self, state: torch.Tensor) -> dict:
            h = self.net(state)
            target_position = self.position_head(h)
            regime_logits = self.regime_head(h)
            confidence = self.confidence_head(h)
            value = self.value_head(h)
            return {
                "target_position": target_position,
                "regime_logits": regime_logits,
                "confidence": confidence,
                "value": value,
                "hidden": h,
            }

    class _LowLevelPolicy(nn.Module):
        """Intraday execution policy conditioned on high-level goal.

        Input: intraday state + goal (target position from high-level)
        Output: execution action (order timing/sizing)

        Goal-conditioned: the goal modulates the policy via FiLM conditioning
        (Feature-wise Linear Modulation).
        """

        def __init__(self, state_dim: int, goal_dim: int = 1) -> None:
            super().__init__()
            self.state_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.LayerNorm(64),
                nn.ELU(),
            )
            # FiLM conditioning: goal -> (scale, shift) for hidden features
            self.goal_scale = nn.Linear(goal_dim, 64)
            self.goal_shift = nn.Linear(goal_dim, 64)

            self.output_net = nn.Sequential(
                nn.Linear(64, 32),
                nn.ELU(),
            )
            # Action: order fraction of remaining target
            self.action_head = nn.Sequential(
                nn.Linear(32, 1),
                nn.Tanh(),
            )
            # Progress: estimated fraction of goal achieved
            self.progress_head = nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            # Value function
            self.value_head = nn.Linear(32, 1)

        def forward(self, state: torch.Tensor, goal: torch.Tensor) -> dict:
            h = self.state_net(state)

            # FiLM: modulate features with goal
            scale = self.goal_scale(goal)
            shift = self.goal_shift(goal)
            h = h * (1.0 + scale) + shift

            out = self.output_net(h)
            action = self.action_head(out)
            progress = self.progress_head(out)
            value = self.value_head(out)

            return {
                "action": action,
                "progress": progress,
                "value": value,
            }

    class HierarchicalDecisionLayer(nn.Module):
        """Multi-timeframe hierarchical RL.

        High-level policy: daily regime detection -> target allocation
        Low-level policy: intraday execution -> order timing/sizing

        Communication: high-level sets goal for low-level via
        goal-conditioned RL (FiLM conditioning).

        Parameters
        ----------
        high_level_policy : nn.Module, optional
            Daily policy. If None, a default _HighLevelPolicy is created.
        low_level_policy : nn.Module, optional
            Intraday policy. If None, a default _LowLevelPolicy is created.
        high_freq : int
            Decisions per day for high-level (1 = daily).
        low_freq : int
            Decisions per day for low-level (375 = per-minute for India).
        high_state_dim : int
            State dimension for high-level policy.
        low_state_dim : int
            State dimension for low-level policy.
        n_regimes : int
            Number of regime classes for high-level.
        """

        def __init__(
            self,
            high_level_policy: Optional[nn.Module] = None,
            low_level_policy: Optional[nn.Module] = None,
            high_freq: int = 1,
            low_freq: int = 375,
            high_state_dim: int = 64,
            low_state_dim: int = 32,
            n_regimes: int = 3,
        ) -> None:
            super().__init__()

            self.high_freq = high_freq
            self.low_freq = low_freq
            self.high_state_dim = high_state_dim
            self.low_state_dim = low_state_dim

            if high_level_policy is not None:
                self.high_level = high_level_policy
            else:
                self.high_level = _HighLevelPolicy(
                    state_dim=high_state_dim,
                    n_regimes=n_regimes,
                )

            if low_level_policy is not None:
                self.low_level = low_level_policy
            else:
                self.low_level = _LowLevelPolicy(
                    state_dim=low_state_dim,
                    goal_dim=1,
                )

            self._current_goal = None
            self._intraday_step = 0

        def high_level_step(self, daily_state: torch.Tensor) -> dict:
            """Daily regime + allocation decision.

            Parameters
            ----------
            daily_state : (batch, high_state_dim)
                Daily aggregated state features.

            Returns
            -------
            dict with keys:
                target_position : (batch, 1) -- target allocation in [-1, 1]
                regime_label : (batch,) -- predicted regime (argmax)
                confidence : (batch, 1) -- regime confidence
                value : (batch, 1) -- high-level value estimate
            """
            out = self.high_level(daily_state)

            regime_probs = torch.softmax(out["regime_logits"], dim=-1)
            regime_label = regime_probs.argmax(dim=-1)

            # Store goal for low-level policy
            self._current_goal = out["target_position"].detach()
            self._intraday_step = 0

            return {
                "target_position": out["target_position"],
                "regime_label": regime_label,
                "confidence": out["confidence"],
                "value": out["value"],
                "regime_probs": regime_probs,
            }

        def low_level_step(
            self,
            intraday_state: torch.Tensor,
            goal: Optional[torch.Tensor] = None,
        ) -> dict:
            """Intraday execution decision conditioned on daily goal.

            Parameters
            ----------
            intraday_state : (batch, low_state_dim)
                Per-minute intraday state features.
            goal : (batch, 1), optional
                Target position from high-level. If None, uses cached goal.

            Returns
            -------
            dict with keys:
                order_action : (batch, 1) -- fraction of remaining to execute
                execution_progress : (batch, 1) -- estimated goal completion
                value : (batch, 1) -- low-level value estimate
            """
            if goal is None:
                goal = self._current_goal
            if goal is None:
                raise ValueError(
                    "No goal available. Call high_level_step() first or "
                    "pass goal explicitly."
                )

            # Ensure goal has right batch dimension
            if goal.shape[0] != intraday_state.shape[0]:
                goal = goal.expand(intraday_state.shape[0], -1)

            out = self.low_level(intraday_state, goal)

            self._intraday_step += 1

            return {
                "order_action": out["action"],
                "execution_progress": out["progress"],
                "value": out["value"],
            }

        def forward(
            self,
            daily_state: torch.Tensor,
            intraday_state: torch.Tensor,
        ) -> dict:
            """Combined forward pass: high-level + low-level.

            Useful for training both levels together.

            Parameters
            ----------
            daily_state : (batch, high_state_dim)
            intraday_state : (batch, low_state_dim)

            Returns
            -------
            dict with all high-level and low-level outputs.
            """
            high_out = self.high_level_step(daily_state)
            low_out = self.low_level_step(
                intraday_state, goal=high_out["target_position"]
            )

            return {
                "high": high_out,
                "low": low_out,
            }

else:
    # Stubs for no-torch environments
    class UnifiedDecisionLayer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for UnifiedDecisionLayer")

    class JointTrainingPipeline:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for JointTrainingPipeline")

    class HierarchicalDecisionLayer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for HierarchicalDecisionLayer")
