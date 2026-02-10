"""End-to-end Integrated Pipeline: Data → Pretrain → RL Train → Evaluate.

Orchestrates the full X-Trend + RL integration:

Phase 1 (Supervised Pre-training):
    MegaFeatureBuilder → ~287 features × 4 assets → XTrendBackbone
    Train backbone via joint MLE+Sharpe loss per walk-forward fold.

Phase 2 (RL Fine-tuning):
    Frozen backbone hidden states → IntegratedTradingEnv → RLTradingAgent
    Train via TD Actor-Critic on IndiaFnOEnv with KellySizer + ThompsonAllocator.

Walk-forward protocol:
    For each fold: train 252d, test 63d, step 21d.
    Backbone pre-trained on train fold, RL agent trained on episodes from
    train fold, evaluated OOS on test fold.

Zero look-ahead bias: test data never enters training at any stage.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

from quantlaxmi.models.ml.tft.x_trend import XTrendConfig
from quantlaxmi.models.rl.environments.india_fno_env import INITIAL_SPOTS
from .backbone import XTrendBackbone, MegaFeatureAdapter
from .rl_trading_agent import RLTradingAgent, RLConfig
from .integrated_env import IntegratedTradingEnv


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IntegratedPipeline:
    """End-to-end pipeline: data → pretrain → RL train → evaluate.

    Parameters
    ----------
    symbols : list[str]
        Trading symbols (e.g. ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]).
    backbone_cfg : XTrendConfig
        Backbone hyperparameters (n_features will be overridden by mega count).
    rl_cfg : RLConfig
        RL hyperparameters.
    train_window : int
        Walk-forward training window in days.
    test_window : int
        Walk-forward test window in days.
    step_size : int
        Walk-forward step size in days.
    pretrain_epochs : int
        Number of epochs for backbone pre-training per fold.
    """

    def __init__(
        self,
        symbols: list[str],
        backbone_cfg: Optional[XTrendConfig] = None,
        rl_cfg: Optional[RLConfig] = None,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
        pretrain_epochs: int = 50,
    ) -> None:
        self.symbols = symbols
        self.backbone_cfg = backbone_cfg or XTrendConfig()
        self.rl_cfg = rl_cfg or RLConfig()
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.pretrain_epochs = pretrain_epochs

    def run(
        self,
        start: str,
        end: str,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        targets: Optional[np.ndarray] = None,
    ) -> dict:
        """Run the full integrated pipeline.

        Parameters
        ----------
        start, end : str
            Date range "YYYY-MM-DD".
        features : optional
            Pre-built feature tensor (n_days, n_assets, n_features).
            If None, MegaFeatureAdapter builds it from quantlaxmi.data.
        feature_names : optional
        dates : optional
        targets : optional
            Pre-built targets (n_days, n_assets). If None, computed from close prices.

        Returns
        -------
        dict with keys:
            positions : (n_days, n_assets) — OOS positions
            returns : (n_days, n_assets) — OOS strategy returns
            per_asset_sharpe : dict[str, float]
            total_sharpe : float
            feature_importance : dict[str, float]
            fold_metrics : list[dict]
        """
        # 1. Build features if not provided
        if features is None:
            adapter = MegaFeatureAdapter(self.symbols)
            features, feature_names, dates = adapter.build_multi_asset(start, end)

        n_days, n_assets, n_features = features.shape
        self.backbone_cfg.n_features = n_features
        self.backbone_cfg.n_assets = n_assets

        logger.info(
            "Pipeline: %d days, %d assets, %d features",
            n_days, n_assets, n_features,
        )

        # 2. Compute targets if not provided (vol-scaled next-day returns)
        if targets is None:
            targets = self._compute_targets(features, n_days, n_assets)

        # 3. Walk-forward loop
        all_positions = np.full((n_days, n_assets), np.nan)
        all_returns = np.full((n_days, n_assets), np.nan)
        fold_metrics = []
        final_importance = {}

        fold_idx = 0
        fold_start = self.backbone_cfg.seq_len + self.backbone_cfg.ctx_len + 10  # warm-up

        while fold_start + self.train_window + self.test_window <= n_days:
            train_end = fold_start + self.train_window
            test_end = min(train_end + self.test_window, n_days)

            logger.info(
                "Fold %d: train=[%d:%d], test=[%d:%d]",
                fold_idx, fold_start, train_end, train_end, test_end,
            )

            # --- Phase 1: Pretrain backbone (supervised) ---
            dev = _DEVICE if _HAS_TORCH else "cpu"
            backbone = XTrendBackbone(self.backbone_cfg, feature_names)
            backbone.to(dev)
            pretrain_metrics = backbone.pretrain(
                features, targets, dates,
                train_start=fold_start,
                train_end=train_end,
                epochs=self.pretrain_epochs,
                lr=self.backbone_cfg.lr,
            )
            logger.info(
                "Fold %d pretrain: best_epoch=%d, final_loss=%.4f",
                fold_idx, pretrain_metrics["best_epoch"], pretrain_metrics["final_loss"],
            )

            # Freeze backbone for RL
            backbone.eval()
            for param in backbone.parameters():
                param.requires_grad = False

            # Feature importance from this fold
            final_importance = backbone.get_feature_importance()

            # --- Phase 2: RL Training ---
            env = IntegratedTradingEnv(
                backbone=backbone,
                features=features,
                targets=targets,
                dates=dates,
                symbols=self.symbols,
                reward_lambda_risk=self.rl_cfg.reward_lambda_risk,
                reward_lambda_cost=self.rl_cfg.reward_lambda_cost,
            )

            agent = RLTradingAgent(
                state_dim=env.state_dim,
                n_assets=n_assets,
                d_hidden=self.backbone_cfg.d_hidden,
                thompson_names=self.symbols,
                rl_cfg=self.rl_cfg,
            )
            agent.to(dev)

            # Train RL on train fold
            rl_metrics = self._train_rl(
                env, agent,
                fold_start, train_end,
                num_episodes=self.rl_cfg.num_episodes,
            )
            logger.info(
                "Fold %d RL: episodes=%d, avg_reward=%.4f",
                fold_idx, rl_metrics["n_episodes"], rl_metrics["avg_reward"],
            )

            # --- Evaluate OOS on test fold ---
            oos_positions, oos_returns, oos_info = self._evaluate_oos(
                env, agent, train_end, test_end
            )

            # Store results
            for t in range(test_end - train_end):
                day_idx = train_end + t
                if day_idx < n_days:
                    all_positions[day_idx, :] = oos_positions[t]
                    all_returns[day_idx, :] = oos_returns[t]

            # Update Thompson with realized returns
            for t in range(len(oos_returns)):
                asset_rets = {}
                for a, sym in enumerate(self.symbols):
                    if not np.isnan(oos_returns[t, a]):
                        asset_rets[sym] = float(oos_returns[t, a])
                if asset_rets:
                    agent.update_thompson(asset_rets)

            fold_metrics.append({
                "fold": fold_idx,
                "train_range": (fold_start, train_end),
                "test_range": (train_end, test_end),
                "pretrain": pretrain_metrics,
                "rl": rl_metrics,
                "oos_info": oos_info,
            })

            fold_start += self.step_size
            fold_idx += 1

            # Cleanup
            del backbone, agent, env
            if _HAS_TORCH:
                torch.cuda.empty_cache()

        # 4. Aggregate results
        results = self._aggregate_results(
            all_positions, all_returns, feature_names,
            final_importance, fold_metrics,
        )
        return results

    def _compute_targets(
        self, features: np.ndarray, n_days: int, n_assets: int
    ) -> np.ndarray:
        """Compute vol-scaled next-day returns as targets.

        Uses features[:, :, 0] as proxy for close prices if available,
        otherwise generates synthetic targets from feature momentum.
        """
        targets = np.full((n_days, n_assets), np.nan, dtype=np.float64)

        # Use first feature as close proxy (price/technical group typically first)
        for a in range(n_assets):
            close_proxy = features[:, a, 0]  # first mega feature
            if np.all(close_proxy == 0):
                # Fallback: small random targets for synthetic mode
                targets[:-1, a] = np.random.default_rng(42 + a).normal(0, 0.01, n_days - 1)
                continue

            # Log returns
            valid = close_proxy > 0
            log_close = np.where(valid, np.log(np.maximum(close_proxy, 1e-10)), 0.0)
            for t in range(n_days - 1):
                if valid[t] and valid[t + 1]:
                    ret = log_close[t + 1] - log_close[t]
                    # Vol-scale: divide by rolling 20d vol
                    start_idx = max(0, t - 19)
                    window_rets = np.diff(log_close[start_idx:t + 1])
                    if len(window_rets) >= 5:
                        vol = np.std(window_rets, ddof=1)
                        vol = max(vol, 1e-6)
                        targets[t, a] = ret / vol
                    else:
                        targets[t, a] = ret

        return targets

    def _train_rl(
        self,
        env: IntegratedTradingEnv,
        agent: RLTradingAgent,
        fold_start: int,
        fold_end: int,
        num_episodes: int,
    ) -> dict:
        """Train the RL agent on the environment for a given fold.

        Runs multiple episodes over the train fold, collecting transitions
        and updating the agent via TD Actor-Critic.

        Returns
        -------
        dict with n_episodes, avg_reward, avg_actor_loss, avg_critic_loss
        """
        total_reward = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_updates = 0

        for ep in range(num_episodes):
            state = env.reset(fold_start, fold_end)
            ep_reward = 0.0

            # Collect transitions for this episode
            states, actions, rewards, next_states, dones = [], [], [], [], []

            for step in range(self.rl_cfg.max_steps_per_ep):
                action, log_prob, value = agent.select_action(
                    state,
                    current_drawdown=env._drawdown,
                    portfolio_heat=float(np.sum(np.abs(env._positions))),
                )

                next_state, reward, done, info = env.step(action)
                ep_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(1.0 if done else 0.0)

                state = next_state
                if done:
                    break

            # Update agent with collected transitions
            if len(states) >= 2:
                actor_loss, critic_loss = agent.update(
                    np.array(states, dtype=np.float32),
                    np.array(actions, dtype=np.float32),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states, dtype=np.float32),
                    np.array(dones, dtype=np.float32),
                )
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                n_updates += 1

            total_reward += ep_reward

            if ep % 50 == 0 or ep == num_episodes - 1:
                logger.info(
                    "  RL ep %d/%d: reward=%.4f, steps=%d",
                    ep, num_episodes, ep_reward, len(states),
                )

        return {
            "n_episodes": num_episodes,
            "avg_reward": total_reward / max(num_episodes, 1),
            "avg_actor_loss": total_actor_loss / max(n_updates, 1),
            "avg_critic_loss": total_critic_loss / max(n_updates, 1),
        }

    def _evaluate_oos(
        self,
        env: IntegratedTradingEnv,
        agent: RLTradingAgent,
        test_start: int,
        test_end: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Evaluate the agent OOS on the test fold.

        Parameters
        ----------
        env : IntegratedTradingEnv
        agent : RLTradingAgent
        test_start, test_end : fold boundaries

        Returns
        -------
        positions : (fold_len, n_assets)
        returns : (fold_len, n_assets)
        info : dict with summary statistics
        """
        fold_len = test_end - test_start
        n_assets = env.n_assets
        positions = np.full((fold_len, n_assets), np.nan)
        returns = np.full((fold_len, n_assets), np.nan)

        state = env.reset(test_start, test_end)

        for t in range(fold_len - 1):
            action, _, _ = agent.select_action(
                state, deterministic=True,
                current_drawdown=env._drawdown,
                portfolio_heat=float(np.sum(np.abs(env._positions))),
            )
            next_state, reward, done, info = env.step(action)

            positions[t] = action
            # Strategy returns: position × next-day return − costs
            day_idx = test_start + t + 1
            if day_idx < len(env.targets):
                for a in range(n_assets):
                    ret_a = env.targets[day_idx, a]
                    if not np.isnan(ret_a):
                        cost_frac = abs(action[a] - (positions[t - 1, a] if t > 0 else 0.0))
                        sym = env.symbols[a].upper()
                        cost_pts = cost_frac * env.get_cost_per_leg(sym)
                        spot = INITIAL_SPOTS.get(sym, 20000.0)
                        returns[t, a] = action[a] * ret_a - cost_pts / spot

            state = next_state
            if done:
                break

        # Summary stats
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) > 1:
            sr = (np.mean(valid_returns) / np.std(valid_returns, ddof=1)) * math.sqrt(252)
        else:
            sr = 0.0

        oos_info = {
            "sharpe": sr,
            "total_return": float(np.nansum(valid_returns)),
            "n_oos_days": fold_len,
            "max_drawdown": self._max_drawdown(valid_returns),
        }
        return positions, returns, oos_info

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Compute max drawdown from returns array."""
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
        peak = np.maximum.accumulate(equity)
        dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
        return float(np.max(dd))

    def _aggregate_results(
        self,
        all_positions: np.ndarray,
        all_returns: np.ndarray,
        feature_names: list[str],
        feature_importance: dict[str, float],
        fold_metrics: list[dict],
    ) -> dict:
        """Aggregate results across all folds."""
        n_assets = all_positions.shape[1]

        # Per-asset Sharpe
        per_asset_sharpe = {}
        for a, sym in enumerate(self.symbols):
            valid = all_returns[:, a]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 1:
                sr = (np.mean(valid) / np.std(valid, ddof=1)) * math.sqrt(252)
            else:
                sr = 0.0
            per_asset_sharpe[sym] = sr

        # Total portfolio Sharpe (equal-weight)
        port_returns = np.nanmean(all_returns, axis=1)
        valid_port = port_returns[~np.isnan(port_returns)]
        if len(valid_port) > 1:
            total_sharpe = (np.mean(valid_port) / np.std(valid_port, ddof=1)) * math.sqrt(252)
        else:
            total_sharpe = 0.0

        # Top features
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top_20_features = sorted_importance[:20]

        return {
            "positions": all_positions,
            "returns": all_returns,
            "per_asset_sharpe": per_asset_sharpe,
            "total_sharpe": total_sharpe,
            "feature_importance": feature_importance,
            "top_20_features": top_20_features,
            "fold_metrics": fold_metrics,
        }

    def report(self, results: dict) -> None:
        """Print human-readable report of pipeline results."""
        print(f"\n{'=' * 70}")
        print(f"{'INTEGRATED X-TREND + RL PIPELINE RESULTS':^70}")
        print(f"{'=' * 70}")

        print(f"\nTotal Portfolio Sharpe: {results['total_sharpe']:.4f}")

        print(f"\nPer-Asset Sharpe Ratios:")
        for sym, sr in results["per_asset_sharpe"].items():
            print(f"  {sym:<30} {sr:>8.4f}")

        if results["top_20_features"]:
            print(f"\nTop 20 Features (VSN importance):")
            for i, (name, weight) in enumerate(results["top_20_features"]):
                print(f"  {i + 1:>3}. {name:<50} {weight:.4f}")

        print(f"\nFolds: {len(results['fold_metrics'])}")
        for fm in results["fold_metrics"]:
            tr = fm["test_range"]
            oos = fm.get("oos_info", {})
            print(
                f"  Fold {fm['fold']}: test=[{tr[0]}:{tr[1]}] "
                f"Sharpe={oos.get('sharpe', 0):.3f} "
                f"Return={oos.get('total_return', 0):.4f}"
            )

        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Convenience function for CLI
# ---------------------------------------------------------------------------


def run_integrated_backtest(
    start: str = "2024-01-01",
    end: str = "2026-02-06",
    symbols: Optional[list[str]] = None,
    backbone_cfg: Optional[XTrendConfig] = None,
    rl_cfg: Optional[RLConfig] = None,
) -> dict:
    """Convenience function to run the full integrated backtest.

    Parameters
    ----------
    start, end : str
        Date range.
    symbols : list[str] or None
        Assets to trade. Default: all 4 India indices.
    backbone_cfg : XTrendConfig or None
    rl_cfg : RLConfig or None

    Returns
    -------
    dict of pipeline results
    """
    if symbols is None:
        symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

    if backbone_cfg is None:
        backbone_cfg = XTrendConfig(
            d_hidden=64,
            seq_len=42,
            ctx_len=42,
            n_context=16,
            loss_mode="joint_mle",
            mle_weight=0.1,
        )

    pipeline = IntegratedPipeline(
        symbols=symbols,
        backbone_cfg=backbone_cfg,
        rl_cfg=rl_cfg,
    )

    results = pipeline.run(start, end)
    pipeline.report(results)
    return results
