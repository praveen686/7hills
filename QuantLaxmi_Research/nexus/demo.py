#!/usr/bin/env python3
"""NEXUS Symposium Demo -- Beautiful visualizations of a novel trading model.

Runs NEXUS on real NSE/crypto market data, trains for 5 epochs (JEPA phase),
then generates 6 publication-quality visualizations demonstrating:
    1. Hyperbolic embeddings on the Poincare disk
    2. Topological persistence diagrams
    3. Imagined future market trajectories
    4. Training loss curves
    5. Regime detection via persistent homology
    6. Architecture summary with parameter counts

All outputs saved to nexus/demo_output/ as PNG files.
Designed to complete in under 2 minutes on T4 GPU.

Usage:
    python demo.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from pathlib import Path

# -- Force non-interactive matplotlib backend before any other imports --------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

import numpy as np
import torch

# -- NEXUS imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus.model import NEXUS, create_nexus
from nexus.config import NexusConfig
from nexus.data import (
    NexusDataLoader,
    NexusDataset,
    create_dataloaders,
    create_synthetic_data,
    create_synthetic_dataloaders,
    ALL_ASSETS,
    MarketFeatureExtractor,
)
from nexus.trainer import NexusTrainer, TrainerConfig
from nexus.topology import (
    TopologicalSensor,
    compute_persistence,
    persistence_entropy,
    takens_embedding,
    persistence_landscape,
)

warnings.filterwarnings("ignore", category=UserWarning)

# -- Constants ---------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "demo_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DARK_BG = "#0d1117"
DARK_FG = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#ff7b72", "#79c0ff"]
ASSET_COLORS = {
    "NIFTY": "#58a6ff",
    "BANKNIFTY": "#f0883e",
    "FINNIFTY": "#3fb950",
    "MIDCPNIFTY": "#bc8cff",
    "BTCUSDT": "#ff7b72",
    "ETHUSDT": "#79c0ff",
}

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def apply_dark_style(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark theme to an axis."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=DARK_FG, labelsize=9)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=DARK_FG, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.15, color=GRID_COLOR, linewidth=0.5)


def save_fig(fig, name):
    """Save figure and close."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load real market data with synthetic fallback."""
    print("[1/5] Loading market data...")
    try:
        loader = NexusDataLoader()
        data_dict = loader.load_all()
        if len(data_dict) >= 2:
            print(f"    Loaded {len(data_dict)} assets from real data:")
            for name, df in data_dict.items():
                print(f"      {name}: {len(df)} days")
            return data_dict, "real"
    except Exception as e:
        print(f"    Real data load failed: {e}")

    print("    Falling back to synthetic data...")
    data_dict = create_synthetic_data(n_assets=6, n_days=500, seed=42)
    for name, df in data_dict.items():
        print(f"      {name}: {len(df)} days (synthetic)")
    return data_dict, "synthetic"


# ---------------------------------------------------------------------------
# 2. Build dataloaders & model
# ---------------------------------------------------------------------------

def build_dataloaders(data_dict, context_len=60, target_len=60, batch_size=16):
    """Build train/val dataloaders from data dict."""
    extractor = MarketFeatureExtractor()

    train_dict, val_dict = {}, {}
    purge_gap = 5
    for name, df in data_dict.items():
        N = len(df)
        train_end = int(N * 0.8)
        val_start = train_end + purge_gap
        if train_end < context_len + target_len:
            continue
        train_dict[name] = df.iloc[:train_end].reset_index(drop=True)
        if val_start < N and (N - val_start) >= context_len + target_len:
            val_dict[name] = df.iloc[val_start:].reset_index(drop=True)

    train_ds = NexusDataset(
        data_dict=train_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=5,
    )
    val_ds = NexusDataset(
        data_dict=val_dict,
        context_len=context_len,
        target_len=target_len,
        feature_extractor=extractor,
        stride=5,
    ) if val_dict else train_ds

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    n_features = train_ds.n_features
    print(f"    Features per timestep: {n_features}")
    print(f"    Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_loader, val_loader, n_features


def build_model(n_features, n_assets=6):
    """Build small NEXUS model for demo."""
    cfg = NexusConfig(
        n_assets=n_assets,
        n_features_daily=max(1, n_features // n_assets),
        d_model=128,
        d_latent=64,
        d_state=32,
        n_layers=3,
        predictor_depth=2,
        d_hyperbolic=32,
        n_heads=4,
        d_action=n_assets,
        n_samples=64,
        n_elites=8,
        n_iterations=2,
        tda_window=20,
        tda_max_dim=0,  # skip H1 to avoid degenerate point cloud crash early in training
        context_len=60,
        target_len=60,
    )
    # Fix total_features to match actual data
    # The config computes total_features = n_assets * n_features_daily
    # but we need it to match the actual feature count from the extractor
    actual_total = cfg.total_features
    if actual_total != n_features:
        cfg.n_features_daily = n_features  # override
        cfg.n_assets = 1  # so total_features = n_features

    model = NEXUS(cfg)
    return model


# ---------------------------------------------------------------------------
# 3. Quick training
# ---------------------------------------------------------------------------

def quick_train(model, train_loader, val_loader, epochs=5):
    """Run 5 epochs of JEPA pre-training."""
    print(f"\n[2/5] Training NEXUS (Phase 1 JEPA, {epochs} epochs on {DEVICE})...")
    t0 = time.time()

    trainer_cfg = TrainerConfig(
        jepa_epochs=epochs,
        jepa_lr=1e-4,
        jepa_warmup_steps=20,
        hyp_weight=0.01,  # small weight to avoid hyperbolic NaN dominating
        use_amp=False,  # disable AMP -- arcosh in hyperbolic loss produces NaN under float16
        device=DEVICE,
        log_every=20,
        save_every=99999,  # no checkpointing during demo
        checkpoint_dir=str(OUTPUT_DIR / "checkpoints"),
    )
    trainer = NexusTrainer(model, trainer_cfg)
    history = trainer.train_jepa(train_loader, epochs=epochs)

    train_time = time.time() - t0
    print(f"    Training completed in {train_time:.1f}s")
    return history, train_time


# ---------------------------------------------------------------------------
# Plot 1: Poincare Disk
# ---------------------------------------------------------------------------

def _prepare_features(df, model, extractor, seq_len=120):
    """Extract features, pad/truncate to model input dim, return tensor."""
    feat_tensor = extractor.extract_tensor(df)
    feat_np = feat_tensor[-seq_len:].numpy()
    n_steps, n_feat = feat_np.shape
    model_input_dim = model.cfg.total_features
    if n_feat < model_input_dim:
        padded = np.zeros((n_steps, model_input_dim), dtype=np.float32)
        padded[:, :n_feat] = feat_np
        feat_np = padded
    elif n_feat > model_input_dim:
        feat_np = feat_np[:, :model_input_dim]
    return feat_np


def plot_poincare_disk(model, data_dict, n_features):
    """Hyperbolic embeddings projected onto the Poincare disk."""
    print("    [Plot 1] Poincare disk embeddings...")
    model.eval()
    extractor = MarketFeatureExtractor()

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.patch.set_facecolor(DARK_BG)
    apply_dark_style(ax, title="Hyperbolic Market Embeddings on Poincare Disk")

    # Draw the unit disk boundary
    circle = Circle((0, 0), 1.0, fill=False, edgecolor="#30363d", linewidth=2, linestyle="--")
    ax.add_patch(circle)

    # Draw faint concentric circles for geodesic distance reference
    for r in [0.3, 0.5, 0.7, 0.9]:
        ref_circle = Circle((0, 0), r, fill=False, edgecolor=GRID_COLOR, linewidth=0.5,
                            linestyle=":", alpha=0.4)
        ax.add_patch(ref_circle)

    # Phase 1: collect ALL Poincare embeddings across assets for joint PCA
    asset_poincare = {}  # name -> (N, d_hyp) raw Poincare coords
    for asset_name, df in data_dict.items():
        if len(df) < 120:
            continue
        feat_np = _prepare_features(df, model, extractor, seq_len=120)
        x = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hyp_emb = model.get_hyperbolic_embeddings(x)
        hyp_np = hyp_emb[0].cpu().numpy()
        x_time = hyp_np[:, 0]
        x_spatial = hyp_np[:, 1:]
        denom = (1.0 + x_time)[:, None]
        poincare_full = x_spatial / np.clip(denom, 1e-6, None)
        asset_poincare[asset_name] = poincare_full

    # Joint PCA across all assets for a coherent 2D projection
    all_points = np.concatenate(list(asset_poincare.values()), axis=0)
    global_mean = all_points.mean(axis=0)
    all_centered = all_points - global_mean
    try:
        _, _, Vt = np.linalg.svd(all_centered, full_matrices=False)
        proj_matrix = Vt[:2].T  # (d_hyp, 2)
    except np.linalg.LinAlgError:
        proj_matrix = np.eye(all_points.shape[1], 2)

    # Phase 2: project each asset and apply tanh scaling for disk spread
    all_proj = all_centered @ proj_matrix
    global_scale = np.percentile(np.abs(all_proj), 95) + 1e-10

    for asset_name, poincare_full in asset_poincare.items():
        color = ASSET_COLORS.get(asset_name, "#8b949e")

        centered = poincare_full - global_mean
        poincare_2d = centered @ proj_matrix
        # tanh scaling: spreads points across the disk, preserving topology
        poincare_2d = np.tanh(poincare_2d / global_scale * 1.8) * 0.88

        # Plot trajectory with fading alpha
        n_pts = len(poincare_2d)
        alphas = np.linspace(0.15, 0.9, n_pts)
        for i in range(n_pts - 1):
            ax.plot(
                poincare_2d[i:i+2, 0], poincare_2d[i:i+2, 1],
                color=color, alpha=alphas[i], linewidth=0.8,
            )

        # Scatter points (last N points larger)
        ax.scatter(
            poincare_2d[:-10, 0], poincare_2d[:-10, 1],
            c=color, s=8, alpha=0.3, zorder=3,
        )
        ax.scatter(
            poincare_2d[-10:, 0], poincare_2d[-10:, 1],
            c=color, s=30, alpha=0.9, edgecolors="white", linewidths=0.5, zorder=4,
        )
        # Label last point
        ax.annotate(
            asset_name,
            (poincare_2d[-1, 0], poincare_2d[-1, 1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)],
        )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincare x1 (PC1)", fontsize=10)
    ax.set_ylabel("Poincare x2 (PC2)", fontsize=10)
    # Add subtitle
    ax.text(
        0, -1.08, "Lorentz H^d -> Poincare Disk via stereographic projection  |  Joint PCA across 6 assets",
        ha="center", fontsize=8, color="#8b949e", style="italic",
    )
    return save_fig(fig, "poincare_disk.png")


# ---------------------------------------------------------------------------
# Plot 2: Persistence Diagram
# ---------------------------------------------------------------------------

def plot_persistence_diagram(model, data_dict):
    """Topological persistence diagrams for each asset."""
    print("    [Plot 2] Persistence diagrams...")
    model.eval()
    extractor = MarketFeatureExtractor()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Topological Persistence Diagrams", fontsize=15, fontweight="bold",
                 color=DARK_FG, y=0.98)

    titles = ["H0 (Connected Components)", "H1 (Loops / Cycles)"]
    for idx, ax in enumerate(axes):
        apply_dark_style(ax, title=titles[idx], xlabel="Birth", ylabel="Death")
        # Diagonal line (points on this line have zero persistence)
        ax.plot([0, 5], [0, 5], "--", color="#484f58", linewidth=1, alpha=0.6)

    asset_entropies = {}
    for asset_name, df in data_dict.items():
        if len(df) < 60:
            continue

        color = ASSET_COLORS.get(asset_name, "#8b949e")

        # Use close prices for Takens embedding
        close = df["close"].values.astype(np.float64)
        log_ret = np.diff(np.log(close + 1e-10))

        # Takens embedding on returns
        try:
            cloud = takens_embedding(log_ret[-100:], dim=3, tau=2)
        except ValueError:
            continue

        persistence = compute_persistence(cloud, max_dim=1)

        entropies = {}
        for dim_idx, hkey in enumerate(["H0", "H1"]):
            pairs = persistence.get(hkey, [])
            ent = persistence_entropy(pairs)
            entropies[hkey] = ent

            finite_pairs = [(b, d) for b, d in pairs if np.isfinite(d) and d > b]
            if not finite_pairs:
                continue
            births = np.array([b for b, _ in finite_pairs])
            deaths = np.array([d for _, d in finite_pairs])
            lifetimes = deaths - births

            # Size proportional to lifetime
            sizes = 20 + 200 * (lifetimes / (lifetimes.max() + 1e-10))

            axes[dim_idx].scatter(
                births, deaths, c=color, s=sizes, alpha=0.7, edgecolors="white",
                linewidths=0.3, label=f"{asset_name} (H={ent:.2f})", zorder=3,
            )

        asset_entropies[asset_name] = entropies

    for ax in axes:
        ax.legend(fontsize=8, loc="lower right", facecolor=DARK_BG, edgecolor=GRID_COLOR,
                  labelcolor=DARK_FG)

    # Add entropy annotation box
    ent_text = "Persistence Entropy:\n"
    for name, ents in asset_entropies.items():
        ent_text += f"  {name}: H0={ents.get('H0', 0):.3f}  H1={ents.get('H1', 0):.3f}\n"
    fig.text(
        0.5, 0.02, ent_text.strip(), ha="center", fontsize=8, color="#8b949e",
        fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                                          edgecolor=GRID_COLOR),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    return save_fig(fig, "persistence_diagram.png")


# ---------------------------------------------------------------------------
# Plot 3: Imagined Futures
# ---------------------------------------------------------------------------

def plot_imagined_futures(model, data_dict, n_features):
    """Imagined market trajectories in latent space."""
    print("    [Plot 3] Imagined futures...")
    model.eval()
    extractor = MarketFeatureExtractor()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Imagined Market Futures (Latent Space, PCA to 2D)", fontsize=15,
                 fontweight="bold", color=DARK_FG, y=0.98)

    # Pick first two assets with enough data
    chosen = []
    for name, df in data_dict.items():
        if len(df) >= 120 and len(chosen) < 2:
            chosen.append((name, df))
    if len(chosen) < 2:
        chosen = list(data_dict.items())[:2]

    for ax_idx, (asset_name, df) in enumerate(chosen):
        ax = axes[ax_idx]
        color = ASSET_COLORS.get(asset_name, "#58a6ff")
        apply_dark_style(ax, title=f"{asset_name} -- 5-step Imagined Trajectories",
                         xlabel="PC1", ylabel="PC2")

        feat_np = _prepare_features(df, model, extractor, seq_len=60)
        x = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Get multiple imagined futures by adding noise to different starting points
        all_trajectories = []
        all_rewards = []
        n_imaginations = 8

        with torch.no_grad():
            for i in range(n_imaginations):
                # Add noise to input for trajectory diversity (scale up for visual spread)
                noise = torch.randn_like(x) * 0.05 * (i + 1)
                x_noisy = x + noise

                imagined_states, imagined_rewards = model.imagine_futures(x_noisy, horizon=5)
                # imagined_states: (1, 5, d_latent)
                all_trajectories.append(imagined_states[0].cpu().numpy())
                all_rewards.append(imagined_rewards[0, :, 0].cpu().numpy())

            # Also get current state for reference
            z_seq = model.encode_sequence(x)  # (1, 60, d_latent)
            context_latent = z_seq[0, -20:].cpu().numpy()  # last 20 steps

        # PCA to 2D: combine context + all imagined
        all_pts = np.concatenate([context_latent] + all_trajectories, axis=0)
        mean = all_pts.mean(axis=0)
        centered = all_pts - mean
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[:2].T
        except np.linalg.LinAlgError:
            proj = centered[:, :2]

        ctx_proj = proj[:20]
        imag_projs = []
        offset = 20
        for traj in all_trajectories:
            imag_projs.append(proj[offset:offset + len(traj)])
            offset += len(traj)

        # Plot context trajectory (fading line)
        for i in range(len(ctx_proj) - 1):
            alpha = 0.2 + 0.6 * (i / len(ctx_proj))
            ax.plot(ctx_proj[i:i+2, 0], ctx_proj[i:i+2, 1],
                    color="#8b949e", alpha=alpha, linewidth=1.5)

        # Mark current state (last context point)
        ax.scatter(ctx_proj[-1, 0], ctx_proj[-1, 1], c="white", s=120,
                   marker="*", zorder=5, edgecolors=color, linewidths=1.5)
        ax.annotate("NOW", (ctx_proj[-1, 0], ctx_proj[-1, 1]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9,
                    fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)])

        # Plot imagined trajectories colored by reward
        # Compute reward range for normalization
        all_avg_rews = [r.mean() for r in all_rewards]
        rew_min = min(all_avg_rews) - 1e-8
        rew_max = max(all_avg_rews) + 1e-8

        for k, (imp, rew) in enumerate(zip(imag_projs, all_rewards)):
            avg_rew = rew.mean()
            # Colormap: red (negative reward) to green (positive)
            norm_rew = np.clip((avg_rew - rew_min) / (rew_max - rew_min), 0, 1)
            traj_color = plt.cm.RdYlGn(norm_rew)

            # Draw from current state to first imagined
            start = ctx_proj[-1]
            ax.plot([start[0], imp[0, 0]], [start[1], imp[0, 1]],
                    color=traj_color, alpha=0.4, linewidth=1, linestyle="--")

            # Draw imagined trajectory
            for i in range(len(imp) - 1):
                alpha = 0.5 + 0.4 * (i / len(imp))
                ax.plot(imp[i:i+2, 0], imp[i:i+2, 1],
                        color=traj_color, alpha=alpha, linewidth=2)

            # End marker
            ax.scatter(imp[-1, 0], imp[-1, 1], c=[traj_color], s=40,
                       edgecolors="white", linewidths=0.5, zorder=4)
            ax.annotate(f"R={avg_rew:.3f}", (imp[-1, 0], imp[-1, 1]),
                        textcoords="offset points", xytext=(5, -10), fontsize=7,
                        color=traj_color, alpha=0.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    return save_fig(fig, "imagined_futures.png")


# ---------------------------------------------------------------------------
# Plot 4: Training Loss Curves
# ---------------------------------------------------------------------------

def plot_training_loss(history):
    """JEPA and hyperbolic loss curves."""
    print("    [Plot 4] Training loss curves...")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    jepa_loss = history.get("jepa_loss", [])
    hyp_loss = history.get("hyperbolic_loss", [])
    steps = np.arange(1, len(jepa_loss) + 1)

    # Smooth with exponential moving average
    def ema_smooth(arr, alpha=0.9):
        out = np.zeros_like(arr, dtype=np.float64)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * out[i - 1] + (1 - alpha) * arr[i]
        return out

    # Left axis: JEPA loss
    apply_dark_style(ax1, title="NEXUS Training Loss Curves (Phase 1: JEPA Pre-training)",
                     xlabel="Training Step", ylabel="JEPA Loss")
    if jepa_loss:
        jepa_arr = np.array(jepa_loss)
        ax1.plot(steps, jepa_arr, color="#58a6ff", alpha=0.2, linewidth=0.5)
        ax1.plot(steps, ema_smooth(jepa_arr), color="#58a6ff", linewidth=2.5,
                 label="JEPA Loss (smoothed)")

    # Right axis: Hyperbolic loss
    ax2 = ax1.twinx()
    ax2.spines["right"].set_color(GRID_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors="#f0883e", labelsize=9)
    ax2.set_ylabel("Hyperbolic Distance Loss", fontsize=10, color="#f0883e")

    if hyp_loss:
        hyp_arr = np.array(hyp_loss)
        ax2.plot(steps, hyp_arr, color="#f0883e", alpha=0.2, linewidth=0.5)
        ax2.plot(steps, ema_smooth(hyp_arr), color="#f0883e", linewidth=2.5,
                 label="Hyperbolic Loss (smoothed)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9,
               facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=DARK_FG)

    # Final loss annotation
    if jepa_loss and hyp_loss:
        final_jepa = ema_smooth(np.array(jepa_loss))[-1]
        final_hyp = ema_smooth(np.array(hyp_loss))[-1]
        ax1.text(
            0.02, 0.95,
            f"Final JEPA: {final_jepa:.5f}\nFinal Hyp:  {final_hyp:.5f}",
            transform=ax1.transAxes, fontsize=9, fontfamily="monospace",
            verticalalignment="top", color=DARK_FG,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=GRID_COLOR),
        )

    plt.tight_layout()
    return save_fig(fig, "training_loss.png")


# ---------------------------------------------------------------------------
# Plot 5: Regime Detection
# ---------------------------------------------------------------------------

def plot_regime_detection(model, data_dict, n_features):
    """Regime detection via topological features overlaid on price."""
    print("    [Plot 5] Regime detection...")
    model.eval()
    extractor = MarketFeatureExtractor()

    # Pick asset with most data
    best_name, best_df = max(data_dict.items(), key=lambda kv: len(kv[1]))

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.2, 1.2, 1], hspace=0.25)

    ax_price = fig.add_subplot(gs[0])
    ax_beta0 = fig.add_subplot(gs[1], sharex=ax_price)
    ax_entropy = fig.add_subplot(gs[2], sharex=ax_price)
    ax_regime = fig.add_subplot(gs[3], sharex=ax_price)

    fig.suptitle(f"Regime Detection via Persistent Homology -- {best_name}",
                 fontsize=15, fontweight="bold", color=DARK_FG, y=0.98)

    close = best_df["close"].values.astype(np.float64)
    N = len(close)
    x_axis = np.arange(N)

    # --- Price panel ---
    apply_dark_style(ax_price, ylabel="Close Price")
    color = ASSET_COLORS.get(best_name, "#58a6ff")
    ax_price.plot(x_axis, close, color=color, linewidth=1.5, alpha=0.9)
    ax_price.fill_between(x_axis, close, close.min(), alpha=0.05, color=color)

    # Compute topological features on sliding windows
    window = 30
    stride = 3
    beta0_series = np.full(N, np.nan)
    entropy_series = np.full(N, np.nan)

    log_ret = np.diff(np.log(close + 1e-10))

    for i in range(window, len(log_ret), stride):
        seg = log_ret[i - window:i]
        try:
            cloud = takens_embedding(seg, dim=3, tau=1)
            persistence = compute_persistence(cloud, max_dim=1)

            # Beta_0 at median radius
            from nexus.topology import _pairwise_distances, betti_numbers
            dists = _pairwise_distances(cloud)
            median_r = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
            betti = betti_numbers(persistence, median_r)

            # +1 because log_ret is shifted by 1 from close
            idx = i + 1
            if idx < N:
                beta0_series[idx] = betti.get("beta_0", 0)
                entropy_series[idx] = persistence_entropy(persistence.get("H0", []))
        except (ValueError, np.linalg.LinAlgError):
            continue

    # --- Beta_0 panel ---
    apply_dark_style(ax_beta0, ylabel="beta_0")
    valid_mask = ~np.isnan(beta0_series)
    ax_beta0.fill_between(x_axis[valid_mask], 0, beta0_series[valid_mask],
                          color="#3fb950", alpha=0.4)
    ax_beta0.plot(x_axis[valid_mask], beta0_series[valid_mask], color="#3fb950",
                  linewidth=1, alpha=0.8)

    # Highlight spikes (regime change candidates)
    if np.any(valid_mask):
        b0_valid = beta0_series[valid_mask]
        b0_mean = np.nanmean(b0_valid)
        b0_std = np.nanstd(b0_valid)
        spike_threshold = b0_mean + 1.5 * b0_std
        spike_mask = beta0_series > spike_threshold
        if np.any(spike_mask):
            ax_price.fill_between(
                x_axis, close.min(), close.max(),
                where=spike_mask, alpha=0.08, color="#ff7b72",
                label="Topo spike (regime change)",
            )
            ax_price.legend(fontsize=8, loc="upper left", facecolor=DARK_BG,
                            edgecolor=GRID_COLOR, labelcolor=DARK_FG)

    # --- Entropy panel ---
    apply_dark_style(ax_entropy, ylabel="H0 Entropy")
    valid_e = ~np.isnan(entropy_series)
    ax_entropy.fill_between(x_axis[valid_e], 0, entropy_series[valid_e],
                            color="#bc8cff", alpha=0.4)
    ax_entropy.plot(x_axis[valid_e], entropy_series[valid_e], color="#bc8cff",
                    linewidth=1, alpha=0.8)

    # --- Model regime predictions ---
    apply_dark_style(ax_regime, xlabel="Time (bar index)", ylabel="Regime")
    regime_labels = ["Bull", "Bear", "Range", "Crisis"]
    regime_colors_map = ["#3fb950", "#ff7b72", "#79c0ff", "#f0883e"]

    # Get model regime predictions
    feat_tensor = extractor.extract_tensor(best_df)
    feat_np = feat_tensor.numpy()
    n_steps, n_feat = feat_np.shape
    model_input_dim = model.cfg.total_features

    window_size = 60
    regime_preds = np.full(N, np.nan)

    with torch.no_grad():
        for i in range(window_size, n_steps, 5):
            seg = feat_np[i - window_size:i]
            if seg.shape[1] < model_input_dim:
                padded = np.zeros((window_size, model_input_dim), dtype=np.float32)
                padded[:, :seg.shape[1]] = seg
                seg = padded
            elif seg.shape[1] > model_input_dim:
                seg = seg[:, :model_input_dim]

            x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, info = model.act(x, use_planning=False)
            pred = info["regime"][0].item()
            regime_preds[i] = pred

    # Plot regime as colored bars
    for i in range(N):
        if not np.isnan(regime_preds[i]):
            r = int(regime_preds[i])
            ax_regime.bar(i, 1, width=5, color=regime_colors_map[r], alpha=0.6)

    ax_regime.set_yticks([0.5])
    ax_regime.set_yticklabels([""])
    ax_regime.set_ylim(0, 1)

    # Legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors_map[i], alpha=0.6, label=regime_labels[i])
        for i in range(4)
    ]
    ax_regime.legend(handles=legend_elements, loc="upper right", ncol=4, fontsize=8,
                     facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=DARK_FG)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_beta0.get_xticklabels(), visible=False)
    plt.setp(ax_entropy.get_xticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "regime_detection.png")


# ---------------------------------------------------------------------------
# Plot 6: Architecture Summary
# ---------------------------------------------------------------------------

def plot_architecture_summary(model, n_features, train_time, data_source, data_dict):
    """Model architecture diagram with parameter counts."""
    print("    [Plot 6] Architecture summary...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    params = model.count_parameters()

    # Title
    ax.text(0.5, 0.97, "NEXUS: Neural Exchange Unified Simulator",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            color=DARK_FG, ha="center", va="top",
            path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)])
    ax.text(0.5, 0.93,
            "JEPA World Model + Mamba-2 Backbone + Hyperbolic Latent Space + TDA Regime Sensor + MPC Planner",
            transform=ax.transAxes, fontsize=10, color="#8b949e", ha="center", va="top",
            style="italic")

    # Architecture flow (left side)
    flow_x = 0.08
    flow_y_start = 0.85
    box_height = 0.08
    box_width = 0.38
    gap = 0.015

    components = [
        ("Market Data", f"{n_features} features/step x {model.cfg.context_len} context",
         "#30363d", None),
        ("Mamba-2 Backbone", f"d_model={model.cfg.d_model}, {model.cfg.n_layers} layers, "
         f"d_state={model.cfg.d_state}, {model.cfg.n_heads} heads", "#1f6feb",
         f"{params['world_model']:,} params"),
        ("JEPA World Model", f"d_latent={model.cfg.d_latent}, predictor_depth="
         f"{model.cfg.predictor_depth}, EMA={model.cfg.ema_decay}", "#238636",
         "(included above)"),
        ("Hyperbolic Space", f"Lorentz H^{model.cfg.d_hyperbolic}, K={model.cfg.curvature}",
         "#8b5cf6", "(included above)"),
        ("Topological Sensor", f"TDA window={model.cfg.tda_window}, dim<={model.cfg.tda_max_dim}, "
         f"8 features", "#f0883e", f"{params['topo_sensor']:,} params"),
        ("MPC Planner", f"CEM: {model.cfg.n_samples} samples, {model.cfg.n_elites} elites, "
         f"horizon={model.cfg.horizon}", "#da3633", f"{params['planner']:,} params"),
        ("Position Vector", f"d_action={model.cfg.d_action} (per-asset positions in [-0.25, 0.25])",
         "#30363d", f"{params['policy_head']:,} + {params['regime_head']:,} params"),
    ]

    for i, (name, desc, color, param_str) in enumerate(components):
        y = flow_y_start - i * (box_height + gap)

        # Box
        rect = plt.Rectangle((flow_x, y - box_height), box_width, box_height,
                              facecolor=color, alpha=0.25, edgecolor=color,
                              linewidth=1.5, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)

        # Name
        ax.text(flow_x + 0.01, y - 0.015, name, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white", va="top")
        # Description
        ax.text(flow_x + 0.01, y - 0.04, desc, transform=ax.transAxes,
                fontsize=8, color="#8b949e", va="top")
        # Param count
        if param_str:
            ax.text(flow_x + box_width - 0.01, y - 0.015, param_str,
                    transform=ax.transAxes, fontsize=8, color="#58a6ff", va="top",
                    ha="right", fontfamily="monospace")

        # Arrow to next (except last)
        if i < len(components) - 1:
            arrow_y = y - box_height - gap / 2
            ax.annotate("", xy=(flow_x + box_width / 2, arrow_y - gap / 2 + 0.002),
                        xytext=(flow_x + box_width / 2, arrow_y + gap / 2 - 0.002),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="#484f58", lw=1.5))

    # Stats panel (right side)
    stats_x = 0.55
    stats_y = 0.85

    stats_title_props = dict(fontsize=12, fontweight="bold", color=DARK_FG, va="top")
    stats_value_props = dict(fontsize=10, color="#58a6ff", va="top", fontfamily="monospace")
    stats_label_props = dict(fontsize=9, color="#8b949e", va="top")

    ax.text(stats_x, stats_y, "Model Statistics", transform=ax.transAxes, **stats_title_props)

    stats = [
        ("Total Parameters", f"{params['total']:,}"),
        ("Model Size (FP32)", f"{params['total'] * 4 / 1024 / 1024:.1f} MB"),
        ("Device", DEVICE.upper()),
        ("Training Time (5 epochs)", f"{train_time:.1f}s"),
        ("Data Source", data_source),
        ("Assets", f"{len(data_dict)}"),
    ]
    for j, (label, value) in enumerate(stats):
        row_y = stats_y - 0.04 - j * 0.035
        ax.text(stats_x, row_y, f"{label}:", transform=ax.transAxes, **stats_label_props)
        ax.text(stats_x + 0.25, row_y, value, transform=ax.transAxes, **stats_value_props)

    # Key innovations panel
    innov_y = stats_y - 0.04 - len(stats) * 0.035 - 0.04
    ax.text(stats_x, innov_y, "Key Innovations", transform=ax.transAxes, **stats_title_props)

    innovations = [
        "JEPA: predict in latent space, not observation space",
        "Mamba-2: O(n) selective state space (not O(n^2) attention)",
        "Lorentz H^d: hyperbolic geometry for hierarchical markets",
        "TDA: persistent homology for topological regime detection",
        "TD-MPC2: CEM trajectory optimization in latent space",
    ]
    for j, text in enumerate(innovations):
        row_y = innov_y - 0.035 - j * 0.03
        ax.text(stats_x, row_y, f"  {j+1}. {text}", transform=ax.transAxes,
                fontsize=8.5, color=DARK_FG, va="top")

    # NEXUS wordmark
    ax.text(0.5, 0.01, "N  E  X  U  S", transform=ax.transAxes, fontsize=28,
            fontweight="bold", color="#21262d", ha="center", va="bottom",
            fontfamily="monospace")

    return save_fig(fig, "architecture_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  NEXUS Symposium Demo")
    print("  Neural Exchange Unified Simulator")
    print(f"  Device: {DEVICE} | PyTorch: {torch.__version__}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()

    # 1. Load data
    data_dict, data_source = load_data()

    # 2. Build dataloaders & model
    print("\n[2/5] Building model and dataloaders...")
    train_loader, val_loader, n_features = build_dataloaders(data_dict)
    model = build_model(n_features, n_assets=len(data_dict))
    params = model.count_parameters()
    print(f"    Model: {params['total']:,} parameters")
    print(f"    Config: d_model={model.cfg.d_model}, d_latent={model.cfg.d_latent}, "
          f"n_layers={model.cfg.n_layers}")

    # 3. Quick training
    history, train_time = quick_train(model, train_loader, val_loader, epochs=5)

    # 4. Generate visualizations
    print(f"\n[4/5] Generating visualizations...")
    paths = []
    paths.append(plot_poincare_disk(model, data_dict, n_features))
    paths.append(plot_persistence_diagram(model, data_dict))
    paths.append(plot_imagined_futures(model, data_dict, n_features))
    paths.append(plot_training_loss(history))
    paths.append(plot_regime_detection(model, data_dict, n_features))
    paths.append(plot_architecture_summary(model, n_features, train_time, data_source, data_dict))

    # 5. Summary
    total_time = time.time() - total_t0
    print(f"\n[5/5] Summary")
    print("=" * 70)
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Data source:    {data_source}")
    print(f"  Assets:         {', '.join(data_dict.keys())}")
    print(f"  Parameters:     {params['total']:,}")
    print(f"  Device:         {DEVICE}")
    print(f"  Visualizations: {len(paths)} PNGs saved to {OUTPUT_DIR}/")
    for p in paths:
        print(f"    - {Path(p).name}")
    print("=" * 70)
    print("  Demo complete.")


if __name__ == "__main__":
    main()
