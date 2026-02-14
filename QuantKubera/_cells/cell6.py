# 6. TRAINING & WALK-FORWARD VALIDATION ENGINE

def expanding_normalize(train_df, test_df, feature_cols):
    """Normalize test data using ONLY training statistics (no look-ahead).

    Args:
        train_df: DataFrame with training data
        test_df: DataFrame with test data
        feature_cols: list of feature column names to normalize

    Returns: (train_normalized, test_normalized) DataFrames
    """
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()
    stds = stds.replace(0, 1.0)  # prevent division by zero
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[feature_cols] = (train_df[feature_cols] - means) / stds
    test_norm[feature_cols] = (test_df[feature_cols] - means) / stds
    return train_norm, test_norm


def make_sequences(feature_array, target_array, window_size):
    """Create (window, n_features) sequences with proper target alignment.

    Args:
        feature_array: np.ndarray shape (n, n_feat)
        target_array: np.ndarray shape (n,) -- target_ret (forward return)
        window_size: int W

    Returns:
        X: (n_valid, W, n_feat) float32
        y: (n_valid,) float32
        indices: (n_valid,) int -- row indices in the original array for each
                 sequence's "current time" (i.e., the last row of each window).
                 Use these to recover dates: df.index[indices]

    Alignment:
        X[i] = features[i:i+W] -- "current time" is i+W-1
        y[i] = target[i+W-1]   -- forward return at that time
        indices[i] = i+W-1
    """
    n = len(feature_array)
    n_seq = n - window_size
    if n_seq <= 0:
        return (np.zeros((0, window_size, feature_array.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.array([], dtype=np.int64))

    X = np.array([feature_array[i:i + window_size] for i in range(n_seq)])
    y = target_array[window_size - 1:window_size - 1 + n_seq]
    indices = np.arange(window_size - 1, window_size - 1 + n_seq, dtype=np.int64)

    # Remove NaN targets
    valid = ~np.isnan(y)
    return X[valid].astype(np.float32), y[valid].astype(np.float32), indices[valid]


def _predict_safe(model, X, desc=None):
    """Run model inference one sample at a time.

    Keras 3 (TF 2.20) bakes the batch dimension from the first model() call
    into the traced graph. Since MomentumTransformer is built with batch=1
    (dummy forward pass), subsequent calls with batch>1 collapse the batch
    dimension. Processing one-at-a-time matches the build shape and is
    guaranteed correct. Cost: ~5ms per sample, negligible vs training time.

    Args:
        model: trained MomentumTransformer
        X: np.ndarray shape (n_samples, window, n_features)
        desc: optional tqdm description

    Returns:
        np.ndarray shape (n_samples,) -- signal at last timestep for each sequence
    """
    n = len(X)
    preds = np.empty(n, dtype=np.float32)
    for i in range(n):
        out = model(tf.constant(X[i:i+1]), training=False)
        preds[i] = float(out[0, -1, 0])
    return preds


def walk_forward_train(data_dict, cfg):
    """Universe-mode walk-forward OOS validation with purge gaps.

    Trains ONE model per fold on pooled sequences from ALL tickers,
    then predicts OOS for each ticker separately. This lets the model
    learn cross-asset patterns and benefit from a larger training set.

    Fold schedule is date-based: uses the SHORTEST ticker to define
    fold boundaries, ensuring all tickers have data for every fold.

    Args:
        data_dict: {ticker_name: pd.DataFrame with FEATURE_COLUMNS + 'target_ret'}
        cfg: MonolithConfig

    Returns: dict with:
        'oos_returns': pd.DataFrame (index=dates, columns=tickers)
        'oos_signals': pd.DataFrame (same shape)
        'fold_metrics': list of dicts {ticker, fold, sharpe, n_days, train_days}
        'models': {fold_number: last trained model}
        'vsn_weights': {'universe': np.ndarray of feature importance}
    """
    tickers = list(data_dict.keys())
    n_feat = len(FEATURE_COLUMNS)

    # Use the shortest ticker to define fold boundaries
    min_len = min(len(df) for df in data_dict.values())

    # Pre-compute number of folds for progress bar
    n_folds_est = 0
    _ts = cfg.min_train_days
    while _ts + cfg.window_size < min_len:
        _te = _ts - cfg.purge_gap
        if _te >= cfg.window_size + 10:
            n_folds_est += 1
        _ts += cfg.test_days

    print(f"\n{'=' * 60}")
    print(f"  Universe Walk-Forward: {len(tickers)} tickers, ~{n_folds_est} folds")
    print(f"  Tickers: {tickers}")
    print(f"  Shortest series: {min_len} days")
    print(f"  Est. time: ~{n_folds_est * len(tickers) * 10 // 60} min "
          f"({n_folds_est} folds x {len(tickers)} tickers)")
    print(f"{'=' * 60}")

    all_oos_returns = {t: [] for t in tickers}
    all_oos_signals = {t: [] for t in tickers}
    all_fold_metrics = []
    all_models = {}
    latest_vsn_w = None

    fold = 0
    test_start = cfg.min_train_days
    fold_pbar = tqdm(total=n_folds_est, desc="Walk-Forward", unit="fold",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}, {rate_fmt}]')

    while test_start + cfg.window_size < min_len:
        fold += 1
        train_end = test_start - cfg.purge_gap
        test_end = min(test_start + cfg.test_days, min_len)

        if train_end < cfg.window_size + 10:
            test_start += cfg.test_days
            continue

        print(f"\n  --- Fold {fold} (train: 0-{train_end}, "
              f"purge: {train_end}-{test_start}, "
              f"test: {test_start}-{test_end}) ---")

        # ── Gather train/test data from ALL tickers ──
        X_train_all = []
        y_train_all = []
        # {ticker: (X_test, y_test, valid_indices, test_df)}
        ticker_test_data = {}

        for ticker in tickers:
            df = data_dict[ticker]

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            # Purge: NaN-out last purge_gap+1 training targets (both raw and vol-scaled)
            if cfg.purge_gap > 0 and len(train_df) > cfg.purge_gap + 1:
                train_df.iloc[-(cfg.purge_gap + 1):,
                              train_df.columns.get_loc('target_ret')] = np.nan
                train_df.iloc[-(cfg.purge_gap + 1):,
                              train_df.columns.get_loc('target_train')] = np.nan

            # Normalize each ticker with its OWN training stats (no cross-ticker leakage)
            train_norm, test_norm = expanding_normalize(train_df, test_df, FEATURE_COLUMNS)

            # Training sequences — use vol-scaled target for better gradient signal
            X_tr, y_tr, _ = make_sequences(
                train_norm[FEATURE_COLUMNS].values,
                train_norm['target_train'].values,
                cfg.window_size
            )
            if len(X_tr) > 0:
                X_train_all.append(X_tr)
                y_train_all.append(y_tr)
                print(f"    {ticker}: train_rows={len(train_df)}, "
                      f"train_seqs={len(X_tr)}, "
                      f"target_NaN={int(train_df['target_train'].isna().sum())}")

            # Test sequences — use vol-scaled for alignment, raw returns for OOS eval
            X_te, _, te_idx = make_sequences(
                test_norm[FEATURE_COLUMNS].values,
                test_norm['target_train'].values,
                cfg.window_size
            )
            # Raw forward returns at the same valid indices (for OOS evaluation)
            y_te_raw = test_df['target_ret'].values[te_idx] if len(te_idx) > 0 else np.array([])
            print(f"    {ticker}: test_rows={len(test_df)}, "
                  f"test_seqs={len(X_te)}, "
                  f"target_NaN={int(test_df['target_train'].isna().sum())}")
            if len(X_te) > 0:
                ticker_test_data[ticker] = (X_te, y_te_raw, te_idx, test_df)

        # Combine all tickers' training data into one pool
        if not X_train_all:
            print(f"  Fold {fold}: no training data, skipping")
            test_start += cfg.test_days
            continue

        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)

        # Shuffle the pooled training data (sequences from different tickers
        # are interleaved — this prevents the model from memorizing ticker order)
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        total_test = sum(len(v[0]) for v in ticker_test_data.values())
        if len(X_train) < 50 or total_test < 5:
            print(f"  Fold {fold}: insufficient data "
                  f"(train={len(X_train)}, test={total_test}), skipping")
            test_start += cfg.test_days
            continue

        print(f"    UNIVERSE POOL: {len(X_train)} train sequences "
              f"from {len(tickers)} tickers")

        # ── Build fresh universe model ──
        model = MomentumTransformer(
            cfg.window_size, n_feat, 1,
            cfg.hidden_size, cfg.num_heads, cfg.dropout_rate
        )
        _ = model(np.zeros((1, cfg.window_size, n_feat), dtype=np.float32))
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.learning_rate, clipnorm=cfg.clipnorm),
            loss=SharpeLoss()
        )

        # ── Train on pooled universe data ──
        t0 = time.time()
        model.fit(
            X_train, y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=0.15,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=cfg.early_stop_patience,
                    restore_best_weights=True, verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=cfg.lr_reduce_factor,
                    patience=cfg.lr_reduce_patience, min_lr=cfg.min_lr, verbose=0
                )
            ],
            verbose=0
        )
        train_time = time.time() - t0

        # ── Predict OOS per-ticker ──
        for ticker, (X_test, y_test, valid_idx, test_df) in ticker_test_data.items():
            n_test = len(X_test)

            # Predict one-at-a-time (Keras 3 bakes batch dim from build step;
            # model was built with batch=1, so batch>1 collapses output)
            preds = _predict_safe(model, X_test)

            assert len(preds) == n_test == len(y_test), (
                f"Shape mismatch: preds={len(preds)}, X_test={n_test}, "
                f"y_test={len(y_test)}")

            # Build date index from make_sequences' valid_idx
            test_pred_idx = test_df.index[valid_idx]

            # Use y_test directly (already aligned by make_sequences)
            all_oos_returns[ticker].append(
                pd.Series(y_test, index=test_pred_idx, name=ticker))
            all_oos_signals[ticker].append(
                pd.Series(preds, index=test_pred_idx, name=ticker))

            # Per-ticker fold metrics
            fold_positions = np.sign(preds)
            fold_strat_ret = fold_positions * y_test
            fold_strat_ret = fold_strat_ret[~np.isnan(fold_strat_ret)]
            if len(fold_strat_ret) > 1:
                fold_sharpe = (np.mean(fold_strat_ret)
                               / (np.std(fold_strat_ret, ddof=1) + 1e-9)
                               * np.sqrt(252))
            else:
                fold_sharpe = 0.0

            all_fold_metrics.append({
                'ticker': ticker, 'fold': fold,
                'sharpe': round(fold_sharpe, 2),
                'n_days': len(preds),
                'train_days': len(X_train),
                'train_time': round(train_time / len(tickers), 1)
            })

            print(f"    {ticker}: test={len(preds):3d} Sharpe={fold_sharpe:+.2f}")

        # Extract VSN weights from the universe model (one sample, matching batch=1 build)
        first_ticker_X = list(ticker_test_data.values())[0][0]
        _, weights_dict = model(tf.constant(first_ticker_X[:1]), return_weights=True)
        latest_vsn_w = weights_dict['vsn_weights'].numpy().mean(axis=(0, 1)).flatten()

        all_models[fold] = model

        # Update progress bar with fold summary
        fold_sharpes = [m['sharpe'] for m in all_fold_metrics
                        if m['fold'] == fold]
        avg_s = np.mean(fold_sharpes) if fold_sharpes else 0.0
        fold_pbar.set_postfix_str(
            f"train={train_time:.0f}s, avg_sharpe={avg_s:+.2f}")
        fold_pbar.update(1)

        test_start += cfg.test_days

        # Memory cleanup
        del X_train, y_train, X_train_all, y_train_all, ticker_test_data
        gc.collect()

    fold_pbar.close()

    # ── Assemble per-ticker OOS results ──
    oos_returns_dict = {}
    oos_signals_dict = {}
    for ticker in tickers:
        if all_oos_returns[ticker]:
            combined_ret = pd.concat(all_oos_returns[ticker])
            combined_sig = pd.concat(all_oos_signals[ticker])
            oos_returns_dict[ticker] = combined_ret[~combined_ret.index.duplicated(keep='first')]
            oos_signals_dict[ticker] = combined_sig[~combined_sig.index.duplicated(keep='first')]

    # VSN weights: store under 'universe' key and duplicate per-ticker for viz compatibility
    vsn_weights = {}
    if latest_vsn_w is not None:
        for ticker in tickers:
            vsn_weights[ticker] = latest_vsn_w

    return {
        'oos_returns': pd.DataFrame(oos_returns_dict),
        'oos_signals': pd.DataFrame(oos_signals_dict),
        'fold_metrics': all_fold_metrics,
        'models': all_models,
        'vsn_weights': vsn_weights,
    }
