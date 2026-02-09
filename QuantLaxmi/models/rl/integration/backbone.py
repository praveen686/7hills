"""X-Trend Backbone Wrapper + MegaFeature Adapter.

Provides two components for the RL integration pipeline:

1. **MegaFeatureAdapter**: Converts per-asset MegaFeatureBuilder output into
   a multi-asset aligned tensor (n_days, n_assets, n_features).

2. **XTrendBackbone**: Wraps the XTrendModel as a frozen state encoder,
   exposing `extract_hidden()` to produce d_hidden-dimensional embeddings
   per asset per day.  Also supports supervised pre-training with joint
   MLE+Sharpe loss and walk-forward folds.

Architecture:
    MegaFeatureBuilder (per asset) → MegaFeatureAdapter (align) → XTrendBackbone
    → VSN(~287 → d_hidden) → LSTM → CrossAttention → Decoder → h_t ∈ ℝ^d_hidden
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

# ---------------------------------------------------------------------------
# Symbol mapping: env name → MegaFeatureBuilder symbol
# ---------------------------------------------------------------------------

_SYMBOL_MAP = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}


# ============================================================================
# MegaFeatureAdapter
# ============================================================================


class MegaFeatureAdapter:
    """Convert per-asset MegaFeatureBuilder output to multi-asset tensor.

    Calls ``MegaFeatureBuilder.build(symbol, start, end)`` per asset,
    outer-joins on date, forward-fills, zeros remaining NaN, and returns
    an aligned (n_days, n_assets, n_features) ndarray plus the feature
    name registry.

    Parameters
    ----------
    symbols : list[str]
        Trading symbols (e.g. ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]).
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    def build_multi_asset(
        self, start: str, end: str
    ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Build aligned multi-asset feature tensor.

        Parameters
        ----------
        start, end : str
            Date range "YYYY-MM-DD".

        Returns
        -------
        features : ndarray of shape (n_days, n_assets, n_features)
        names : list[str] — feature column names
        dates : pd.DatetimeIndex
        """
        from features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        per_asset_dfs: list[pd.DataFrame] = []
        feature_names: list[str] = []

        for asset_idx, symbol in enumerate(self.symbols):
            mega_sym = _SYMBOL_MAP.get(symbol.upper(), symbol)
            df, names = builder.build(mega_sym, start, end)
            # df is indexed by date (datetime), one row per trading day
            per_asset_dfs.append(df)
            if not feature_names:
                feature_names = names

        if not per_asset_dfs:
            raise ValueError("No features built — check symbols and data availability")

        # Outer-join all assets on date
        all_dates = sorted(
            set().union(*(df.index for df in per_asset_dfs))
        )
        dates = pd.DatetimeIndex(all_dates)
        n_days = len(dates)
        n_assets = len(self.symbols)
        n_features = len(feature_names)

        features = np.full((n_days, n_assets, n_features), np.nan, dtype=np.float64)

        for asset_idx, df in enumerate(per_asset_dfs):
            aligned = df.reindex(dates)
            # Forward-fill within each asset (causal — only uses past values)
            aligned = aligned.ffill()
            vals = aligned[feature_names].values
            features[:, asset_idx, :] = vals

        # Zero remaining NaN (leading NaN before first valid date)
        features = np.nan_to_num(features, nan=0.0)

        logger.info(
            "MegaFeatureAdapter: %d days × %d assets × %d features",
            n_days, n_assets, n_features,
        )
        return features, feature_names, dates


# ============================================================================
# XTrendBackbone
# ============================================================================

if _HAS_TORCH:

    class XTrendBackbone(nn.Module):
        """Wraps XTrendModel as a frozen state encoder for RL.

        Provides:
        - ``extract_hidden()``: VSN→LSTM→CrossAttention→Decoder→h_t ∈ ℝ^d_hidden
        - ``get_feature_importance()``: reads VSN softmax weights
        - ``pretrain()``: supervised training with joint MLE+Sharpe loss

        Parameters
        ----------
        cfg : XTrendConfig
            Must have ``n_features`` set to the mega feature count.
        feature_names : list[str]
            Feature names for importance mapping.
        """

        def __init__(self, cfg, feature_names: list[str]) -> None:
            super().__init__()
            from models.ml.tft.x_trend import XTrendModel

            self.cfg = cfg
            self.feature_names = feature_names
            self.model = XTrendModel(cfg)

        @property
        def d_hidden(self) -> int:
            return self.cfg.d_hidden

        def extract_hidden(
            self,
            target_seq: torch.Tensor,
            context_set: torch.Tensor,
            target_id: torch.Tensor,
            context_ids: torch.Tensor,
        ) -> torch.Tensor:
            """Extract decoder hidden state (batch, d_hidden) from the backbone.

            Delegates to XTrendModel.extract_hidden() which runs the full
            forward path (VSN→LSTM→CrossAttention→Decoder) but stops before
            the output heads.
            """
            return self.model.extract_hidden(target_seq, context_set, target_id, context_ids)

        def extract_hidden_for_day(
            self,
            features: np.ndarray,
            day_idx: int,
            asset_idx: int,
            rng: np.random.Generator,
        ) -> np.ndarray:
            """Extract hidden state for a single (day, asset) pair.

            Parameters
            ----------
            features : (n_days, n_assets, n_features) — normalized
            day_idx : int — target day index (end of target sequence)
            asset_idx : int — which asset
            rng : numpy RNG for context set sampling

            Returns
            -------
            hidden : (d_hidden,) numpy array
            """
            from models.ml.tft.x_trend import build_context_set

            seq_len = self.cfg.seq_len
            if day_idx < seq_len:
                return np.zeros(self.cfg.d_hidden, dtype=np.float32)

            target_window = features[day_idx - seq_len: day_idx, asset_idx, :]
            if np.any(np.isnan(target_window)):
                return np.zeros(self.cfg.d_hidden, dtype=np.float32)

            ctx_seqs, ctx_ids = build_context_set(
                features,
                target_start=day_idx - seq_len,
                n_context=self.cfg.n_context,
                ctx_len=self.cfg.ctx_len,
                rng=rng,
            )

            # Use the device of the model parameters (handles CPU and CUDA)
            dev = next(self.parameters()).device

            with torch.no_grad():
                tgt_t = torch.tensor(
                    target_window[np.newaxis], dtype=torch.float32, device=dev
                )
                ctx_t = torch.tensor(
                    ctx_seqs[np.newaxis], dtype=torch.float32, device=dev
                )
                tid_t = torch.tensor([asset_idx], dtype=torch.long, device=dev)
                cid_t = torch.tensor(
                    ctx_ids[np.newaxis], dtype=torch.long, device=dev
                )
                h = self.model.extract_hidden(tgt_t, ctx_t, tid_t, cid_t)

            return h.squeeze(0).cpu().numpy()

        def precompute_hidden_states(
            self,
            features: np.ndarray,
            start_idx: int,
            end_idx: int,
            rng: np.random.Generator,
        ) -> np.ndarray:
            """Pre-compute hidden states for a date range, all assets.

            Parameters
            ----------
            features : (n_days, n_assets, n_features) — normalized
            start_idx, end_idx : fold boundaries (inclusive, exclusive)
            rng : numpy RNG

            Returns
            -------
            hidden_states : (end_idx - start_idx, n_assets, d_hidden)
            """
            n_assets = features.shape[1]
            fold_len = end_idx - start_idx
            hidden = np.zeros((fold_len, n_assets, self.cfg.d_hidden), dtype=np.float32)

            self.eval()
            for t in range(fold_len):
                day_idx = start_idx + t
                for a in range(n_assets):
                    hidden[t, a, :] = self.extract_hidden_for_day(
                        features, day_idx, a, rng
                    )
            return hidden

        def get_feature_importance(self) -> dict[str, float]:
            """Read VSN softmax weights and map to feature names.

            Returns
            -------
            dict mapping feature_name → importance weight (sums to ~1.0)
            """
            vsn = self.model.vsn
            # The weight_grn processes (batch, seq_len, n_features) → (batch, seq_len, n_features)
            # Then softmax is applied.  For a quick importance proxy, run a unit
            # input through the weight GRN and take the softmax output.
            n_feat = self.cfg.n_features
            dev = next(self.parameters()).device
            with torch.no_grad():
                dummy_input = torch.ones(1, 1, n_feat, device=dev)
                # Entity context (zeros for neutral)
                raw_weights = vsn.weight_grn(dummy_input)
                weights = torch.softmax(raw_weights, dim=-1)  # (1, 1, n_features)
                w = weights.squeeze().cpu().numpy()

            result = {}
            for i, name in enumerate(self.feature_names):
                if i < len(w):
                    result[name] = float(w[i])
            return result

        def pretrain(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            dates: pd.DatetimeIndex,
            train_start: int,
            train_end: int,
            epochs: int = 50,
            lr: float = 1e-3,
        ) -> dict:
            """Supervised pre-training on a single fold with joint MLE+Sharpe loss.

            Parameters
            ----------
            features : (n_days, n_assets, n_features)
            targets : (n_days, n_assets) — vol-scaled next-day returns
            dates : DatetimeIndex
            train_start, train_end : fold boundaries
            epochs : training epochs
            lr : learning rate

            Returns
            -------
            metrics dict with keys: final_loss, losses, best_epoch
            """
            from models.ml.tft.x_trend import (
                build_context_set,
                joint_loss,
                sharpe_loss,
            )

            cfg = self.cfg
            n_assets = features.shape[1]
            seq_len = cfg.seq_len
            rng = np.random.default_rng(42)

            # Normalize features using train stats
            train_feats = features[train_start:train_end]
            flat = train_feats.reshape(-1, cfg.n_features)
            valid_mask = ~np.any(np.isnan(flat), axis=1)
            if valid_mask.sum() < 30:
                logger.warning("Insufficient valid train samples for pretrain")
                return {"final_loss": float("nan"), "losses": [], "best_epoch": 0}

            feat_mean = np.nanmean(flat[valid_mask], axis=0)
            feat_std = np.nanstd(flat[valid_mask], axis=0, ddof=1)
            feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
            norm_features = (features - feat_mean) / feat_std

            # Build training episodes
            X_target, X_context, X_tid, X_cid, Y = [], [], [], [], []
            for asset_idx in range(n_assets):
                for t in range(train_start + seq_len, train_end):
                    tw = norm_features[t - seq_len: t, asset_idx, :]
                    if np.any(np.isnan(tw)):
                        continue
                    tgt = targets[t, asset_idx]
                    if np.isnan(tgt):
                        continue

                    ctx_seqs, ctx_ids = build_context_set(
                        norm_features,
                        target_start=t - seq_len,
                        n_context=cfg.n_context,
                        ctx_len=cfg.ctx_len,
                        rng=rng,
                    )
                    X_target.append(tw)
                    X_context.append(ctx_seqs)
                    X_tid.append(asset_idx)
                    X_cid.append(ctx_ids)
                    Y.append(tgt)

            if len(X_target) < 10:
                return {"final_loss": float("nan"), "losses": [], "best_epoch": 0}

            X_target_arr = np.array(X_target, dtype=np.float32)
            X_context_arr = np.array(X_context, dtype=np.float32)
            X_tid_arr = np.array(X_tid, dtype=np.int64)
            X_cid_arr = np.array(X_cid, dtype=np.int64)
            Y_arr = np.array(Y, dtype=np.float32)

            dev = next(self.parameters()).device
            self.train()

            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=1e-5
            )

            n_total = len(X_target_arr)
            n_val = max(1, int(n_total * 0.2))
            n_train = n_total - n_val
            batch_size = cfg.batch_size

            losses = []
            best_loss = float("inf")
            best_epoch = 0
            best_state = None

            for epoch in range(epochs):
                self.train()
                perm = rng.permutation(n_train)
                epoch_loss = 0.0
                n_batches = 0

                for batch_start in range(0, n_train, batch_size):
                    batch_idx = perm[batch_start: batch_start + batch_size]
                    tgt_seq = torch.tensor(X_target_arr[batch_idx], device=dev)
                    ctx_set = torch.tensor(X_context_arr[batch_idx], device=dev)
                    tgt_id = torch.tensor(X_tid_arr[batch_idx], device=dev)
                    ctx_id = torch.tensor(X_cid_arr[batch_idx], device=dev)
                    y_batch = torch.tensor(Y_arr[batch_idx], device=dev)

                    # Use joint MLE+Sharpe if mode supports it, else Sharpe
                    if cfg.loss_mode in ("joint_mle", "joint_quantile"):
                        mu, log_sigma = self.model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = joint_loss(mu, log_sigma, y_batch.unsqueeze(-1), cfg.mle_weight)
                    else:
                        positions = self.model(tgt_seq, ctx_set, tgt_id, ctx_id)
                        loss = sharpe_loss(positions.squeeze(-1), y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

            # Restore best
            if best_state is not None:
                self.load_state_dict(best_state)
                self.to(dev)

            return {
                "final_loss": losses[-1] if losses else float("nan"),
                "losses": losses,
                "best_epoch": best_epoch,
            }

else:
    # Stub for no-torch environments
    class XTrendBackbone:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for XTrendBackbone")

    class MegaFeatureAdapter:  # type: ignore[no-redef]
        pass
