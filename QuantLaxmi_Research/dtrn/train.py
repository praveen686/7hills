"""DTRN training pipeline — three-stage training.

Stage 1: Self-supervised representation + topology
  - Predict future micro-returns and volatility
  - Topology regularizers (sparsity + stability)

Stage 2: Regime discovery (unsupervised)
  - Temporal smoothness on regime posteriors
  - Encourage distinct, persistent regimes

Stage 3: Trading objective (offline RL-lite)
  - Differentiable PnL proxy
  - Transaction cost penalty
  - Drawdown penalty
"""
from __future__ import annotations

import gc
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .config import DTRNConfig
from .data.loader import load_day, list_available_dates
from .data.features import FeatureEngine
from .model.topology import DynamicTopology
from .model.graph_net import DTRN as DTRNModel
from .model.dtrn import create_dtrn

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_training_data(
    dates: list[date],
    instrument: str,
    config: DTRNConfig,
) -> list[dict]:
    """Prepare training data: features + topology per day.

    Returns list of day dicts with:
    - features: (T, d) np array
    - masks: (T, d) np array
    - adjacencies: (T, d, d) np array
    - weights: (T, d, d) np array
    - prices: (T,) close prices
    - returns: (T,) log returns
    """
    feature_engine = FeatureEngine(config)
    n_features = feature_engine.n_features

    all_days = []

    for d in dates:
        df = load_day(d, instrument)
        if df is None or len(df) < 30:
            continue

        # Compute features
        features, masks = feature_engine.compute_batch(df)
        T = len(features)

        # Compute topology per step
        topology = DynamicTopology(
            d=n_features,
            ewma_span=config.ewma_cov_span,
            top_k=config.top_k_edges,
            tau_on=config.tau_on,
            tau_off=config.tau_off,
            max_flip_rate=config.max_edge_flip_rate,
            precision_reg=config.precision_reg,
        )

        adjs = np.zeros((T, n_features, n_features), dtype=np.float32)
        wgts = np.zeros((T, n_features, n_features), dtype=np.float32)

        for t in range(T):
            topology.update(features[t], masks[t])
            adjs[t] = topology.get_adjacency()
            wgts[t] = topology.get_weights()

        # Prices and returns
        prices = df["close"].values.astype(np.float32)
        returns = np.zeros(T, dtype=np.float32)
        returns[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))

        # Realized volatility (for targets)
        vol = np.zeros(T, dtype=np.float32)
        vol[1:] = np.abs(returns[1:])  # simple |return| as vol proxy

        # Jump flags (|return| > 3 sigma)
        rolling_std = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            window = returns[max(0, t - 60):t]
            if len(window) > 2:
                rolling_std[t] = np.std(window, ddof=1)
        jumps = (np.abs(returns) > 3 * np.maximum(rolling_std, 1e-6)).astype(np.float32)

        all_days.append({
            "date": d,
            "features": features,
            "masks": masks,
            "adjacencies": adjs,
            "weights": wgts,
            "prices": prices,
            "returns": returns,
            "volatility": vol,
            "jumps": jumps,
        })

    return all_days


def train_stage1(
    model: DTRNModel,
    train_data: list[dict],
    config: DTRNConfig,
    epochs: int = 20,
    device: str = DEVICE,
) -> list[float]:
    """Stage 1: Self-supervised — predict returns, volatility, jumps.

    Loss = Huber(return_pred) + lambda_v * Huber(vol_pred) + lambda_j * BCE(jump_pred)
         + lambda_sparse * topology_sparsity + lambda_stable * topology_stability
    """
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    huber = nn.SmoothL1Loss()

    losses = []
    H = config.pred_horizon

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for day in train_data:
            T = len(day["features"])
            if T <= H + 10:
                continue

            # Convert to tensors
            feat = torch.tensor(day["features"], dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, d)
            mask = torch.tensor(day["masks"], dtype=torch.float32, device=device).unsqueeze(0)
            adj = torch.tensor(day["adjacencies"], dtype=torch.float32, device=device)  # (T, d, d)
            wgt = torch.tensor(day["weights"], dtype=torch.float32, device=device)

            returns = torch.tensor(day["returns"], dtype=torch.float32, device=device)
            vol = torch.tensor(day["volatility"], dtype=torch.float32, device=device)
            jumps = torch.tensor(day["jumps"], dtype=torch.float32, device=device)

            # Process in chunks to manage memory (window of ~60 steps)
            window_size = 60
            for start in range(0, T - H - window_size, window_size // 2):
                end = min(start + window_size, T - H)
                if end - start < 10:
                    continue

                f_chunk = feat[:, start:end, :]
                m_chunk = mask[:, start:end, :]
                a_chunk = adj[start:end, :, :]
                w_chunk = wgt[start:end, :, :]

                out = model(f_chunk, m_chunk, a_chunk, w_chunk)
                preds = out["predictions"]

                # Build targets: for each step t in chunk, target is returns[t+1:t+H+1]
                chunk_len = end - start
                ret_targets = torch.zeros(1, chunk_len, H, device=device)
                vol_targets = torch.zeros(1, chunk_len, H, device=device)
                jump_targets = torch.zeros(1, chunk_len, H, device=device)

                for t_local in range(chunk_len):
                    t_global = start + t_local
                    for h in range(H):
                        if t_global + h + 1 < T:
                            ret_targets[0, t_local, h] = returns[t_global + h + 1]
                            vol_targets[0, t_local, h] = vol[t_global + h + 1]
                            jump_targets[0, t_local, h] = jumps[t_global + h + 1]

                # Losses
                loss_ret = huber(preds["returns"], ret_targets)
                loss_vol = huber(preds["volatility"], vol_targets) * config.lambda_vol
                loss_jump = F.binary_cross_entropy_with_logits(
                    preds["jump_logits"], jump_targets
                ) * config.lambda_jump

                # Topology regularizers
                adj_chunk = a_chunk
                loss_sparse = adj_chunk.sum() / max(adj_chunk.numel(), 1) * config.lambda_sparse

                if chunk_len > 1:
                    adj_diff = (adj_chunk[1:] - adj_chunk[:-1]).abs().sum()
                    loss_stable = adj_diff / max(adj_chunk[1:].numel(), 1) * config.lambda_stable
                else:
                    loss_stable = torch.tensor(0.0, device=device)

                loss = loss_ret + loss_vol + loss_jump + loss_sparse + loss_stable

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"  [Stage1] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.6f}  time={elapsed:.1f}s", flush=True)

    return losses


def train_stage2(
    model: DTRNModel,
    train_data: list[dict],
    config: DTRNConfig,
    epochs: int = 10,
    device: str = DEVICE,
) -> list[float]:
    """Stage 2: Regime discovery — temporal smoothness + distinctness.

    Loss = KL(pi_t || pi_{t-1}) smoothness + regime entropy bonus
    """
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr * 0.5, weight_decay=config.weight_decay)

    losses = []
    H = config.pred_horizon

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for day in train_data:
            T = len(day["features"])
            if T <= H + 10:
                continue

            feat = torch.tensor(day["features"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(day["masks"], dtype=torch.float32, device=device).unsqueeze(0)
            adj = torch.tensor(day["adjacencies"], dtype=torch.float32, device=device)
            wgt = torch.tensor(day["weights"], dtype=torch.float32, device=device)

            out = model(feat, mask, adj, wgt)
            regime_probs = out["regime_probs"]  # (1, T, K)

            # Temporal smoothness: KL(p_t || p_{t-1})
            # We want mode-seeking (zero-forcing) KL so regime sharpening isn't penalized.
            # F.kl_div(input=log(Q), target=P) computes KL(P||Q).
            # For KL(p_t || p_{t-1}): P=p_t, Q=p_{t-1} → input=log(p_{t-1}), target=p_t
            p_t = regime_probs[:, 1:, :]
            p_tm1 = regime_probs[:, :-1, :].detach()

            kl_div = F.kl_div(
                torch.log(p_tm1 + 1e-8),  # log(Q) = log(p_{t-1})
                p_t,                        # P = p_t — gradients flow through here
                reduction='batchmean',
                log_target=False,
            )

            # Regime distinctness: encourage non-uniform usage
            avg_regime = regime_probs.mean(dim=1)  # (1, K)
            uniform = torch.ones_like(avg_regime) / config.n_regimes
            usage_loss = -F.kl_div(
                torch.log(avg_regime + 1e-8),
                uniform,
                reduction='batchmean',
                log_target=False,
            )

            loss = kl_div + 0.1 * usage_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"  [Stage2] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.6f}  time={elapsed:.1f}s", flush=True)

    return losses


def train_stage3(
    model: DTRNModel,
    train_data: list[dict],
    config: DTRNConfig,
    epochs: int = 30,
    device: str = DEVICE,
) -> list[float]:
    """Stage 3: Trading objective — differentiable PnL proxy.

    Loss = -E[p_t * r_{t+1}] + lambda_tc * |p_t - p_{t-1}| + lambda_dd * DD_proxy
    """
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr * 0.1, weight_decay=config.weight_decay)

    losses = []

    for epoch in range(epochs):
        t0 = time.time()
        epoch_pnl = 0.0
        epoch_cost = 0.0
        n_batches = 0

        for day in train_data:
            T = len(day["features"])
            if T < 30:
                continue

            feat = torch.tensor(day["features"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(day["masks"], dtype=torch.float32, device=device).unsqueeze(0)
            adj = torch.tensor(day["adjacencies"], dtype=torch.float32, device=device)
            wgt = torch.tensor(day["weights"], dtype=torch.float32, device=device)
            returns = torch.tensor(day["returns"], dtype=torch.float32, device=device)

            out = model(feat, mask, adj, wgt)
            positions = out["position"].squeeze(-1).squeeze(0)  # (T,)
            regime_probs = out["regime_probs"]  # (1, T, K)

            # PnL: position_t * return_{t+1}
            # Scale returns by 1e4 to bring 1-min returns (~1e-4) into gradient-friendly range
            scaled_returns = returns * 1e4
            pnl = positions[:-1] * scaled_returns[1:]

            # Transaction costs: |position_change| * lambda_tc
            # TC should be proportional to PnL magnitude. With 1e4-scaled returns,
            # PnL per step ≈ |pos| * |scaled_ret| ≈ 0.1 * 1.0 = 0.1
            # TC per step should be a fraction of that: pos_change * lambda_tc * scale
            # where scale matches return units. Use 1e2 (not 1e4) to keep TC < PnL.
            pos_change = (positions[1:] - positions[:-1]).abs()
            tc_cost = pos_change * config.lambda_tc * 1e2

            # Net PnL
            net_pnl = pnl - tc_cost

            # Drawdown penalty
            cum_pnl = torch.cumsum(net_pnl, dim=0)
            peak_pnl = torch.cummax(cum_pnl, dim=0).values
            drawdown = (peak_pnl - cum_pnl).mean()

            # Position activity loss: penalize the model for hiding at zero
            # Encourage mean |position| > 0.1
            pos_activity = F.relu(0.1 - positions.abs().mean())

            # Loss: maximize PnL, minimize costs + drawdown, encourage activity
            loss = -net_pnl.mean() + config.lambda_dd * drawdown + 1.0 * pos_activity

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_pnl += net_pnl.mean().item()
            epoch_cost += tc_cost.mean().item()
            n_batches += 1

        avg_pnl = epoch_pnl / max(n_batches, 1)
        avg_cost = epoch_cost / max(n_batches, 1)
        losses.append(-avg_pnl)  # loss is negative PnL
        elapsed = time.time() - t0

        # Check position magnitudes for debugging
        with torch.no_grad():
            sample = train_data[0]
            f = torch.tensor(sample["features"], dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(sample["masks"], dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.tensor(sample["adjacencies"], dtype=torch.float32, device=device)
            w = torch.tensor(sample["weights"], dtype=torch.float32, device=device)
            o = model(f, m, a, w)
            pos_mag = o["position"].abs().mean().item()
            max_regime_p = o["regime_probs"].max(dim=-1).values.mean().item()

        print(f"  [Stage3] Epoch {epoch+1:2d}/{epochs}  "
              f"avg_pnl={avg_pnl:.6f}  avg_tc={avg_cost:.6f}  "
              f"|pos|={pos_mag:.4f}  max_regime_p={max_regime_p:.3f}  time={elapsed:.1f}s", flush=True)

    return losses


def train_full_pipeline(
    config: DTRNConfig = None,
    instrument: str = "NIFTY",
    train_dates: list[date] = None,
    s1_epochs: int = 20,
    s2_epochs: int = 10,
    s3_epochs: int = 30,
    save_dir: Path = None,
    device: str = DEVICE,
) -> tuple:
    """Full three-stage training pipeline.

    Returns (model, training_losses).
    """
    if config is None:
        config = DTRNConfig()

    if save_dir is None:
        save_dir = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/dtrn/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get training dates
    if train_dates is None:
        all_dates = list_available_dates()
        if len(all_dates) > config.train_days:
            train_dates = all_dates[:config.train_days]
        else:
            train_dates = all_dates

    print(f"Preparing training data for {instrument} ({len(train_dates)} days)...")
    train_data = prepare_training_data(train_dates, instrument, config)
    print(f"  Loaded {len(train_data)} valid days")

    if not train_data:
        raise ValueError("No valid training data")

    # Create model
    n_features = train_data[0]["features"].shape[1]
    _, model = create_dtrn(config, n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters, {n_features} features")
    print(f"  Device: {device}")

    all_losses = {}

    # Stage 1: Self-supervised
    print(f"\n{'='*60}")
    print("Stage 1: Self-supervised Prediction")
    print(f"{'='*60}")
    all_losses["stage1"] = train_stage1(model, train_data, config, s1_epochs, device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save stage 1 checkpoint
    torch.save(model.state_dict(), save_dir / "dtrn_stage1.pt")
    print(f"  Stage 1 checkpoint saved")

    # Stage 2: Regime discovery
    print(f"\n{'='*60}")
    print("Stage 2: Regime Discovery")
    print(f"{'='*60}")
    all_losses["stage2"] = train_stage2(model, train_data, config, s2_epochs, device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), save_dir / "dtrn_stage2.pt")
    print(f"  Stage 2 checkpoint saved")

    # Stage 3: Trading objective
    print(f"\n{'='*60}")
    print("Stage 3: Trading Objective")
    print(f"{'='*60}")
    all_losses["stage3"] = train_stage3(model, train_data, config, s3_epochs, device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "n_features": n_features,
            "d_embed": config.d_embed,
            "d_hidden": config.d_hidden,
            "n_message_passes": config.n_message_passes,
            "d_temporal": config.d_temporal,
            "n_regimes": config.n_regimes,
            "pred_horizon": config.pred_horizon,
        },
        "losses": all_losses,
        "instrument": instrument,
        "train_dates": [str(d) for d in train_dates],
    }, save_dir / "dtrn_final.pt")
    print(f"\n  Final model saved to {save_dir / 'dtrn_final.pt'}")

    return model, all_losses
