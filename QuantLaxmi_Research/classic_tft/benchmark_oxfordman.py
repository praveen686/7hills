"""Walk-forward Oxford-Man Volatility Backtest for Classic TFT.

Trains the Classic TFT model on Oxford-Man realized volatility data using
a walk-forward protocol and reports normalized quantile loss metrics
(P50, P90) comparable to the original Google TFT paper results.

Benchmark to beat: P50 = 0.0447, P90 = 0.0213 (Google TF1.x implementation)

Usage
-----
    python -m quantlaxmi.models.ml.tft.classic_tft_backtest

    # Or with custom settings:
    python -m quantlaxmi.models.ml.tft.classic_tft_backtest \\
        --encoder-steps 252 --decoder-steps 5 --epochs 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    _DEVICE = None


DEFAULT_DATA_PATH = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi_Research/oxfordman_long.csv")

# Symbols used in the Google TFT paper for volatility forecasting
BENCHMARK_SYMBOLS = [
    '.SPX', '.FTSE', '.GDAXI', '.N225', '.NSEI',
    '.HSI', '.DJI', '.IXIC', '.FCHI', '.STOXX50E',
]


def load_oxford_man(
    path: Path = DEFAULT_DATA_PATH,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Load and preprocess Oxford-Man realized volatility data.

    Parameters
    ----------
    path : Path to CSV
    symbols : list of symbols to include (default: BENCHMARK_SYMBOLS)

    Returns
    -------
    DataFrame with date index, all derived features
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Handle timezone-aware dates from Oxford-Man CSV
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    if symbols:
        df = df[df['Symbol'].isin(symbols)]

    # Drop rows with missing target
    df = df.dropna(subset=['rv5_ss'])
    df = df[df['rv5_ss'] > 0]

    return df


def run_classic_tft_backtest(
    data_path: Path = DEFAULT_DATA_PATH,
    symbols: list[str] | None = None,
    encoder_steps: int = 252,
    decoder_steps: int = 5,
    d_hidden: int = 160,
    n_heads: int = 4,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 5,
    purge_gap: int = 5,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    save_path: Path | None = None,
) -> dict:
    """Run full walk-forward backtest.

    Split: 70% train / 15% validation / 15% test (per symbol, by time).
    Purge gap of `purge_gap` days between splits.

    Parameters
    ----------
    data_path : Path to Oxford-Man CSV
    symbols : list of symbols (default: BENCHMARK_SYMBOLS)
    encoder_steps : lookback window
    decoder_steps : forecast horizon
    d_hidden : model hidden dimension
    n_heads : attention heads
    dropout : dropout rate
    lr : learning rate
    epochs : max training epochs
    batch_size : training batch size
    patience : early stopping patience
    purge_gap : days between train/val and val/test
    train_frac : fraction of data for training
    val_frac : fraction of data for validation
    save_path : optional path to save results JSON

    Returns
    -------
    dict with keys: p50, p90, rmse, n_test_samples, training_time, etc.
    """
    from classic_tft import (
        ClassicTFTConfig, ClassicTFTModel, QuantileLoss, normalized_quantile_loss,
    )
    from data_formatter import TFTDataFormatter

    if symbols is None:
        symbols = BENCHMARK_SYMBOLS

    t0 = time.time()

    print("=" * 60)
    print("Classic TFT — Oxford-Man Volatility Backtest")
    print("=" * 60)
    print(f"Device: {_DEVICE}")
    if _DEVICE and _DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load data
    print("Loading Oxford-Man data...")
    raw_df = load_oxford_man(data_path, symbols)
    print(f"  {len(raw_df)} rows, {raw_df['Symbol'].nunique()} symbols")

    # Create formatter and prepare data
    formatter = TFTDataFormatter.for_volatility(
        encoder_steps=encoder_steps,
        decoder_steps=decoder_steps,
    )
    prepared = formatter.prepare_oxford_man(raw_df)
    print(f"  Prepared: {len(prepared)} rows after preprocessing")

    # Per-symbol temporal split
    train_dfs = []
    val_dfs = []
    test_dfs = []

    print("\nWalk-forward splits:")
    for sym in symbols:
        sym_df = prepared[prepared['Symbol'] == sym].sort_index()
        n = len(sym_df)
        if n < encoder_steps + decoder_steps + 2 * purge_gap + 10:
            print(f"  {sym}: skipped (only {n} rows)")
            continue

        train_end = int(n * train_frac)
        val_start = train_end + purge_gap
        val_end = val_start + int(n * val_frac)
        test_start = val_end + purge_gap

        train_dfs.append(sym_df.iloc[:train_end])
        val_dfs.append(sym_df.iloc[val_start:val_end])
        test_dfs.append(sym_df.iloc[test_start:])

        print(f"  {sym}: train=[0:{train_end}], val=[{val_start}:{val_end}], "
              f"test=[{test_start}:{n}] ({n - test_start} days)")

    if not train_dfs:
        raise ValueError("No symbols with enough data")

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    # Fit formatter on training data only
    formatter.fit(train_df)

    # Create windowed datasets
    print("\nCreating windowed datasets...")
    train_data = formatter.transform(train_df)
    val_data = formatter.transform(val_df)
    test_data = formatter.transform(test_df, return_raw_targets=True)

    print(f"  Train: {len(train_data['targets'])} windows")
    print(f"  Val:   {len(val_data['targets'])} windows")
    print(f"  Test:  {len(test_data['targets'])} windows")

    # Build model config
    tft_config_dict = formatter.get_tft_config()
    cfg = ClassicTFTConfig(
        d_hidden=d_hidden,
        n_heads=n_heads,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        purge_gap=purge_gap,
        **tft_config_dict,
    )

    print(f"\nModel config:")
    print(f"  d_hidden={cfg.d_hidden}, n_heads={cfg.n_heads}, "
          f"encoder={cfg.encoder_steps}, decoder={cfg.decoder_steps}")
    print(f"  n_observed={cfg.n_observed}, n_known={cfg.n_known}, "
          f"n_static_cat={cfg.n_static_cat}")
    print(f"  Past variables: {formatter.n_past_variables}, "
          f"Future variables: {formatter.n_future_variables}")

    # Create model
    model = ClassicTFTModel(cfg).to(_DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable params: {n_trainable:,}")

    # Training
    criterion = QuantileLoss(cfg.quantiles).to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    use_amp = _DEVICE is not None and _DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Convert to tensors
    def to_tensors(data: dict) -> tuple:
        past = torch.tensor(data['past_inputs'], dtype=torch.float32)
        future = torch.tensor(data['future_inputs'], dtype=torch.float32)
        static_cat = torch.tensor(data['static_cat'], dtype=torch.int64) if data['static_cat'] is not None else None
        targets = torch.tensor(data['targets'], dtype=torch.float32)
        return past, future, static_cat, targets

    train_tensors = to_tensors(train_data)
    val_tensors = to_tensors(val_data)

    n_train = len(train_tensors[0])
    n_batches = math.ceil(n_train / batch_size)

    print(f"\n{'='*60}")
    print("Training Classic TFT")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        t_epoch = time.time()
        model.train()

        # Shuffle
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]

            past_b = train_tensors[0][idx].to(_DEVICE)
            future_b = train_tensors[1][idx].to(_DEVICE)
            static_cat_b = train_tensors[2][idx].to(_DEVICE) if train_tensors[2] is not None else None
            targets_b = train_tensors[3][idx].to(_DEVICE)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds, _ = model(past_b, future_b, static_cat_b)
                    loss = criterion(preds, targets_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds, _ = model(past_b, future_b, static_cat_b)
                loss = criterion(preds, targets_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_past = val_tensors[0].to(_DEVICE)
            val_future = val_tensors[1].to(_DEVICE)
            val_static = val_tensors[2].to(_DEVICE) if val_tensors[2] is not None else None
            val_targets = val_tensors[3].to(_DEVICE)

            # Process in chunks to avoid OOM
            val_preds_list = []
            chunk_size = 1024
            for i in range(0, len(val_past), chunk_size):
                vp = val_past[i:i+chunk_size]
                vf = val_future[i:i+chunk_size]
                vs = val_static[i:i+chunk_size] if val_static is not None else None
                pred_chunk, _ = model(vp, vf, vs)
                val_preds_list.append(pred_chunk)

            val_preds = torch.cat(val_preds_list, dim=0)
            val_loss = criterion(val_preds, val_targets).item()

        elapsed = time.time() - t_epoch
        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train_loss={avg_train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  "
              f"time={elapsed:.1f}s  ({n_batches} batches)")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(_DEVICE)

    training_time = time.time() - t0

    # ==== Evaluation on TEST set ====
    print(f"\n{'='*60}")
    print("Evaluating on TEST set...")
    print(f"{'='*60}")

    test_tensors = to_tensors(test_data)
    model.eval()

    with torch.no_grad():
        test_past = test_tensors[0].to(_DEVICE)
        test_future = test_tensors[1].to(_DEVICE)
        test_static = test_tensors[2].to(_DEVICE) if test_tensors[2] is not None else None
        test_targets = test_tensors[3].to(_DEVICE)

        # Process in chunks
        test_preds_list = []
        for i in range(0, len(test_past), chunk_size):
            tp = test_past[i:i+chunk_size]
            tf = test_future[i:i+chunk_size]
            ts = test_static[i:i+chunk_size] if test_static is not None else None
            pred_chunk, _ = model(tp, tf, ts)
            test_preds_list.append(pred_chunk)

        test_preds = torch.cat(test_preds_list, dim=0)

    # Compute normalized quantile losses
    quantiles = cfg.quantiles
    p50_idx = quantiles.index(0.5) if 0.5 in quantiles else 1
    p90_idx = quantiles.index(0.9) if 0.9 in quantiles else 2
    p10_idx = quantiles.index(0.1) if 0.1 in quantiles else 0

    p50_loss = normalized_quantile_loss(test_preds, test_targets, 0.5, p50_idx)
    p90_loss = normalized_quantile_loss(test_preds, test_targets, 0.9, p90_idx)
    p10_loss = normalized_quantile_loss(test_preds, test_targets, 0.1, p10_idx)

    # RMSE of median predictions
    median_preds = test_preds[:, :, p50_idx:p50_idx+1]
    rmse = float(torch.sqrt(((median_preds - test_targets) ** 2).mean()))

    # Print results
    print(f"\n  P50 (normalized quantile loss): {p50_loss:.4f}")
    print(f"  P90 (normalized quantile loss): {p90_loss:.4f}")
    print(f"  P10 (normalized quantile loss): {p10_loss:.4f}")
    print(f"  RMSE (median predictions):      {rmse:.4f}")

    # Comparison table
    print(f"\n{'='*60}")
    print("Classic TFT PyTorch vs Google TF1.x Benchmark")
    print(f"{'='*60}")
    print(f"{'Metric':<25} | {'Google TF1.x':>12} | {'Our PyTorch':>12} | {'Winner':>12}")
    print(f"{'-'*25}-|{'-'*14}|{'-'*14}|{'-'*13}")

    google_p50 = 0.0447
    google_p90 = 0.0213

    p50_winner = "Ours" if p50_loss < google_p50 else "Google"
    p90_winner = "Ours" if p90_loss < google_p90 else "Google"

    print(f"{'P50 (norm q-loss)':<25} | {google_p50:>12.4f} | {p50_loss:>12.4f} | {p50_winner:>12}")
    print(f"{'P90 (norm q-loss)':<25} | {google_p90:>12.4f} | {p90_loss:>12.4f} | {p90_winner:>12}")
    print(f"{'RMSE':<25} | {'N/A':>12} | {rmse:>12.4f} |")
    print(f"{'Test samples':<25} | {'N/A':>12} | {len(test_preds):>12} |")
    print(f"{'Training time':<25} | {'N/A':>12} | {training_time:>10.0f}s |")
    print(f"{'='*60}")

    results = {
        'p50': p50_loss,
        'p90': p90_loss,
        'p10': p10_loss,
        'rmse': rmse,
        'n_test_samples': len(test_preds),
        'n_train_samples': len(train_tensors[0]),
        'n_val_samples': len(val_tensors[0]),
        'training_time_s': training_time,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'config': {
            'd_hidden': d_hidden,
            'n_heads': n_heads,
            'encoder_steps': encoder_steps,
            'decoder_steps': decoder_steps,
            'dropout': dropout,
            'lr': lr,
            'batch_size': batch_size,
        },
        'benchmark': {
            'google_p50': google_p50,
            'google_p90': google_p90,
            'p50_improvement': (google_p50 - p50_loss) / google_p50 * 100,
            'p90_improvement': (google_p90 - p90_loss) / google_p90 * 100,
        },
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")

        # Save model weights
        if best_state is not None:
            weights_path = save_path.with_suffix('.pt')
            torch.save({
                'model_state_dict': best_state,
                'config': results['config'],
                'p50': p50_loss,
                'p90': p90_loss,
                'best_val_loss': best_val_loss,
            }, weights_path)
            print(f"Model weights saved to {weights_path}")

    # Cleanup
    del model, train_tensors, val_tensors, test_tensors
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Classic TFT Oxford-Man Benchmark")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--encoder-steps", type=int, default=252)
    parser.add_argument("--decoder-steps", type=int, default=5)
    parser.add_argument("--d-hidden", type=int, default=160)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--purge-gap", type=int, default=5)
    parser.add_argument("--save", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_classic_tft_backtest(
        data_path=Path(args.data),
        encoder_steps=args.encoder_steps,
        decoder_steps=args.decoder_steps,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        purge_gap=args.purge_gap,
        save_path=Path(args.save) if args.save else None,
    )

    print(f"\nFinal: P50={results['p50']:.4f}, P90={results['p90']:.4f}")


if __name__ == "__main__":
    main()
