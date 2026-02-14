# 7. ORCHESTRATION, METRICS & VISUALIZATION

def calculate_metrics(returns, costs=0.0):
    """Full metrics suite for strategy returns.
    
    Args:
        returns: np.ndarray or pd.Series of daily returns (simple, not log)
        costs: total transaction costs already deducted (for reporting only)
    
    Returns: dict with:
        total_return, annual_return, sharpe, sortino, calmar, max_dd,
        win_rate, profit_factor, n_trades, avg_hold_days
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    
    n = len(r)
    if n < 2:
        return {
            'total_return': 0.0, 'annual_return': 0.0, 'sharpe': 0.0,
            'sortino': 0.0, 'calmar': 0.0, 'max_dd': 0.0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'n_days': 0,
        }
    
    # Total and annualized return (simple compounding)
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    n_years = n / 252.0
    if n_years > 0 and equity[-1] > 0:
        annual_return = float(equity[-1] ** (1.0 / n_years) - 1.0)
    else:
        annual_return = 0.0
    
    # Sharpe ratio: sqrt(252) * mean / std(ddof=1)
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    sharpe = float(np.sqrt(252) * mean_r / std_r) if std_r > 1e-9 else 0.0
    
    # Sortino ratio: sqrt(252) * mean / downside_std(ddof=1)
    downside = r[r < 0]
    if len(downside) > 1:
        downside_std = np.std(downside, ddof=1)
        sortino = float(np.sqrt(252) * mean_r / downside_std) if downside_std > 1e-9 else 0.0
    else:
        sortino = 0.0
    
    # Maximum drawdown from equity curve
    peak = np.maximum.accumulate(equity)
    dd_series = (equity - peak) / peak
    max_dd = float(np.min(dd_series))
    
    # Calmar ratio: annual_return / |max_dd|
    calmar = float(annual_return / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0
    
    # Win rate: fraction of positive return days
    win_rate = float(np.mean(r > 0))
    
    # Profit factor: sum(gains) / |sum(losses)|
    gains = r[r > 0]
    losses = r[r < 0]
    sum_gains = float(np.sum(gains)) if len(gains) > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = float(sum_gains / sum_losses) if sum_losses > 1e-12 else 0.0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_days': n,
    }


def run_enhanced_lab(cfg=None):
    """Main orchestrator: Data -> Features -> Walk-Forward -> Backtest -> Visualize.
    
    This is the primary entry point for the notebook.
    """
    if cfg is None:
        cfg = CFG
    
    tickers_to_run = [cfg.tickers[0]] if cfg.quick_mode else cfg.tickers
    
    # ── PHASE 1: DATA ACQUISITION ──
    print("=" * 70)
    print("  PHASE 1: DATA ACQUISITION")
    print("=" * 70)
    auth = KiteAuth()
    kite = auth.get_session()
    if not kite:
        raise RuntimeError("Kite authentication failed. Check .env credentials.")

    fetcher = KiteFetcher(kite)
    raw_data = {}
    for ticker in tqdm(tickers_to_run, desc="Fetching data", unit="ticker"):
        exchange = cfg.exchanges.get(ticker, 'NSE')
        tqdm.write(f"  {ticker} ({exchange})...")
        try:
            raw_data[ticker] = fetcher.fetch_daily(ticker, exchange, cfg.lookback_days)
            tqdm.write(f"    -> {len(raw_data[ticker])} days "
                       f"({raw_data[ticker].index[0].date()} to "
                       f"{raw_data[ticker].index[-1].date()})")
        except Exception as e:
            tqdm.write(f"    SKIP: {ticker} failed ({e})")

    if not raw_data:
        raise RuntimeError("No tickers fetched successfully.")

    # ── PHASE 1.5: CROSS-ASSET DATA (News Sentiment + India VIX) ──
    print("\n" + "=" * 70)
    print("  PHASE 1.5: CROSS-ASSET DATA (News Sentiment + India VIX)")
    print("=" * 70)

    # Build union of all trading dates for alignment
    all_dates = pd.DatetimeIndex([])
    for df in raw_data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    cross_asset_df = fetch_cross_asset_features(
        start_date=all_dates[0].strftime('%Y-%m-%d'),
        end_date=all_dates[-1].strftime('%Y-%m-%d'),
        date_index=all_dates,
        kite=kite,
    )

    # ── PHASE 2: FEATURE ENGINEERING ──
    print("\n" + "=" * 70)
    print(f"  PHASE 2: FEATURE ENGINEERING ({len(FEATURE_COLUMNS)} features)")
    print("=" * 70)
    featured_data = {}
    min_required = cfg.min_train_days + 2 * cfg.test_days
    for ticker, df in tqdm(raw_data.items(), desc="Feature engineering",
                           total=len(raw_data), unit="ticker"):
        try:
            t0 = time.time()
            feat_df = build_all_features(df, cfg)
            elapsed = time.time() - t0
            if len(feat_df) < min_required:
                tqdm.write(f"  SKIP: {ticker} has {len(feat_df)} featured days "
                           f"(need {min_required})")
                continue
            # Merge cross-asset features (same for all tickers)
            if not cross_asset_df.empty:
                feat_df = feat_df.join(cross_asset_df, how='left')
                for col in CROSS_ASSET_COLUMNS:
                    if col in feat_df.columns:
                        feat_df[col] = feat_df[col].ffill(limit=3).fillna(0.0)
            featured_data[ticker] = feat_df
            tqdm.write(f"  {ticker}: {len(feat_df)} days, "
                       f"{len(FEATURE_COLUMNS)} features ({elapsed:.1f}s)")
        except Exception as e:
            tqdm.write(f"  SKIP: {ticker} feature build failed ({e})")

    if not featured_data:
        raise RuntimeError("No tickers survived feature engineering.")

    # Extend FEATURE_COLUMNS with cross-asset features (if available)
    if CROSS_ASSET_COLUMNS:
        for col in CROSS_ASSET_COLUMNS:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)
        print(f"\n  Features extended: {len(FEATURE_COLUMNS)} total "
              f"(+{len(CROSS_ASSET_COLUMNS)} cross-asset: "
              f"{CROSS_ASSET_COLUMNS})")

    print(f"\n  Universe: {len(featured_data)} tickers survived "
          f"(min {min_required} featured days required)")
    
    # ── PHASE 3: WALK-FORWARD TRAINING ──
    print("\n" + "=" * 70)
    print("  PHASE 3: WALK-FORWARD OOS VALIDATION")
    print(f"  Train: {cfg.min_train_days}d | Test: {cfg.test_days}d | "
          f"Purge: {cfg.purge_gap}d | Window: {cfg.window_size}d")
    print("=" * 70)
    results = walk_forward_train(featured_data, cfg)
    
    # ── PHASE 4: STRATEGY CONSTRUCTION & METRICS ──
    print("\n" + "=" * 70)
    print("  PHASE 4: OOS STRATEGY METRICS")
    print("=" * 70)
    
    oos_returns = results['oos_returns']
    oos_signals = results['oos_signals']
    
    if oos_returns.empty:
        print("  WARNING: No OOS data produced. Check data length vs min_train_days.")
        return {'raw_data': raw_data, 'featured_data': featured_data,
                'results': results, 'all_metrics': {}, 'cfg': cfg}
    
    all_metrics = {}
    strat_net_returns = {}  # store for plotting
    
    for ticker in oos_returns.columns:
        ret = oos_returns[ticker].dropna()
        sig = oos_signals[ticker].dropna()
        
        # Align on common dates
        common_idx = ret.index.intersection(sig.index)
        ret = ret.loc[common_idx]
        sig = sig.loc[common_idx]
        
        if len(ret) < 5:
            print(f"  {ticker}: insufficient OOS data ({len(ret)} days), skipping metrics")
            continue
        
        # Strategy returns: position = sign(signal), NOT raw magnitude
        position = np.sign(sig.values)
        strat_ret = position * ret.values
        
        # Transaction costs: deduct |delta_position| * bps_cost
        pos_change = np.abs(np.diff(position, prepend=0))
        costs_arr = pos_change * cfg.bps_cost
        net_ret = strat_ret - costs_arr
        
        metrics = calculate_metrics(net_ret, costs=costs_arr.sum())
        metrics['ticker'] = ticker
        metrics['total_costs'] = float(costs_arr.sum())
        all_metrics[ticker] = metrics
        strat_net_returns[ticker] = pd.Series(net_ret, index=common_idx)
        
        print(f"\n  {ticker} (OOS: {ret.index[0].date()} to "
              f"{ret.index[-1].date()}, {len(ret)} days):")
        print(f"    Sharpe:        {metrics['sharpe']:+.2f}")
        print(f"    Total Return:  {metrics['total_return']:+.2%}")
        print(f"    Annual Return: {metrics['annual_return']:+.2%}")
        print(f"    Max Drawdown:  {metrics['max_dd']:.2%}")
        print(f"    Sortino:       {metrics['sortino']:+.2f}")
        print(f"    Calmar:        {metrics['calmar']:+.2f}")
        print(f"    Win Rate:      {metrics['win_rate']:.1%}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"    Total Costs:   {metrics['total_costs']:.4f}")
    
    # ── PHASE 5: FOLD-BY-FOLD SUMMARY ──
    print("\n" + "=" * 70)
    print("  FOLD-BY-FOLD BREAKDOWN")
    print("=" * 70)
    fold_df = pd.DataFrame(results['fold_metrics'])
    if not fold_df.empty:
        print(fold_df.to_string(index=False))
        
        avg_sharpe = fold_df.groupby('ticker')['sharpe'].mean()
        std_sharpe = fold_df.groupby('ticker')['sharpe'].std(ddof=1)
        n_folds = fold_df.groupby('ticker')['sharpe'].count()
        print(f"\n  Average OOS Sharpe per ticker:")
        for t in avg_sharpe.index:
            se = std_sharpe[t] / np.sqrt(n_folds[t]) if n_folds[t] > 1 else 0.0
            print(f"    {t}: {avg_sharpe[t]:+.2f} +/- {std_sharpe[t]:.2f} "
                  f"(SE={se:.2f}, {int(n_folds[t])} folds)")
    
    # ── PHASE 5.5: ENSEMBLE (TFT + Momentum + Mean Reversion) ──
    print("\n" + "=" * 70)
    print("  PHASE 5.5: ENSEMBLE (TFT + Momentum + MR — majority vote)")
    print("=" * 70)

    ensemble_metrics = {}
    ensemble_net_returns = {}
    for ticker in oos_returns.columns:
        ret = oos_returns[ticker].dropna()
        sig = oos_signals[ticker].dropna()
        common_idx = ret.index.intersection(sig.index)

        if len(common_idx) < 10 or ticker not in featured_data:
            continue

        df_feat = featured_data[ticker]
        feat_idx = common_idx.intersection(df_feat.index)
        if len(feat_idx) < 10:
            continue

        ret_e = ret.loc[feat_idx].values
        sig_e = sig.loc[feat_idx].values

        # Three signals
        tft_pos = np.sign(sig_e)
        mom_pos = np.sign(df_feat.loc[feat_idx, 'norm_ret_21d'].values)
        mr_pos = -np.sign(df_feat.loc[feat_idx, 'mr_zscore'].values)

        # Majority vote: sign of sum (2-of-3 or 3-of-3 agree)
        ens_signal = tft_pos + mom_pos + mr_pos
        ens_pos = np.sign(ens_signal)

        # Ensemble returns with costs
        ens_strat_ret = ens_pos * ret_e
        pos_change = np.abs(np.diff(ens_pos, prepend=0))
        costs = pos_change * cfg.bps_cost
        ens_net = ens_strat_ret - costs

        ens_m = calculate_metrics(ens_net, costs=costs.sum())
        ensemble_metrics[ticker] = ens_m
        ensemble_net_returns[ticker] = pd.Series(ens_net, index=feat_idx)

        # Individual signal Sharpes (gross) for comparison
        tft_sr = (float(np.mean(tft_pos * ret_e))
                  / (float(np.std(tft_pos * ret_e, ddof=1)) + 1e-9)
                  * np.sqrt(252))
        mom_sr = (float(np.mean(mom_pos * ret_e))
                  / (float(np.std(mom_pos * ret_e, ddof=1)) + 1e-9)
                  * np.sqrt(252))
        mr_sr = (float(np.mean(mr_pos * ret_e))
                 / (float(np.std(mr_pos * ret_e, ddof=1)) + 1e-9)
                 * np.sqrt(252))

        # Signal agreement rate
        agree_all = float(np.mean((tft_pos == mom_pos) & (mom_pos == mr_pos)))
        agree_2of3 = float(np.mean(np.abs(ens_signal) >= 2))

        print(f"\n  {ticker} ({len(ret_e)} days):")
        print(f"    TFT Sharpe:        {tft_sr:+.2f}")
        print(f"    Momentum Sharpe:   {mom_sr:+.2f}")
        print(f"    MR Sharpe:         {mr_sr:+.2f}")
        print(f"    Ensemble Sharpe:   {ens_m['sharpe']:+.2f}  (net of costs)")
        print(f"    Ensemble Return:   {ens_m['total_return']:+.2%}")
        print(f"    Ensemble MaxDD:    {ens_m['max_dd']:.2%}")
        print(f"    Agreement (3/3):   {agree_all:.1%}")
        print(f"    Agreement (2+/3):  {agree_2of3:.1%}")

    # ── PHASE 6: EDGE AUDIT ──
    print("\n" + "=" * 70)
    print("  PHASE 6: EDGE AUDIT — Is the alpha real?")
    print("=" * 70)

    print("\n  Target definition:")
    print("    target_ret[t] = (close[t+1] - close[t]) / close[t]  [1-day forward return]")
    print("    signal[t] from features[t-W+1 : t]  [causal window, W=21]")
    print("    strat_ret[t] = sign(signal[t]) * target_ret[t]  [trade at close of day t]")

    for ticker in oos_returns.columns:
        ret = oos_returns[ticker].dropna()
        sig = oos_signals[ticker].dropna()
        common_idx = ret.index.intersection(sig.index)
        ret_a = ret.loc[common_idx]
        sig_a = sig.loc[common_idx]

        if len(ret_a) < 10:
            continue

        r = ret_a.values
        position = np.sign(sig_a.values)
        strat_ret_gross = position * r

        # Turnover stats
        pos_changes = np.abs(np.diff(position, prepend=0))
        n_trades = int(np.sum(pos_changes > 0))
        daily_turnover = np.mean(pos_changes)
        gross_sharpe = (float(np.mean(strat_ret_gross))
                        / (float(np.std(strat_ret_gross, ddof=1)) + 1e-9)
                        * np.sqrt(252))

        print(f"\n  {ticker} ({len(r)} OOS days, {n_trades} position changes, "
              f"daily turnover {daily_turnover:.3f})")
        print(f"    Gross Sharpe (no costs): {gross_sharpe:+.2f}")

        # ── Test 1: Sign-flip ──
        flip_ret = -position * r
        flip_sharpe = (float(np.mean(flip_ret))
                       / (float(np.std(flip_ret, ddof=1)) + 1e-9)
                       * np.sqrt(252))
        verdict_1 = "PASS" if flip_sharpe < 0 else "FAIL (signal is noise or drift)"
        print(f"\n    Test 1 — Sign flip:")
        print(f"      Flipped Sharpe:  {flip_sharpe:+.2f}  [{verdict_1}]")

        # ── Test 2: Delay execution by 1 day ──
        if len(r) > 2:
            # position[t] applied to target_ret[t+1] instead of target_ret[t]
            delay_ret = position[:-1] * r[1:]
            delay_sharpe = (float(np.mean(delay_ret))
                            / (float(np.std(delay_ret, ddof=1)) + 1e-9)
                            * np.sqrt(252))
            decay = gross_sharpe - delay_sharpe
            if delay_sharpe < gross_sharpe * 0.5:
                verdict_2 = "CLEAN (edge decays with delay)"
            elif delay_sharpe < 0:
                verdict_2 = "CLEAN (edge reverses with delay)"
            else:
                verdict_2 = "SUSPICIOUS (edge persists — possible drift capture)"
            print(f"\n    Test 2 — Delay 1 day:")
            print(f"      Delayed Sharpe:  {delay_sharpe:+.2f}  "
                  f"(decay: {decay:+.2f})  [{verdict_2}]")

        # ── Test 3: Dumb baselines ──
        if ticker in featured_data:
            df_feat = featured_data[ticker]
            common_feat_idx = common_idx.intersection(df_feat.index)
            if len(common_feat_idx) > 10:
                ret_bl = ret_a.loc[common_feat_idx].values

                # Momentum: sign(yesterday's normalized return)
                mom_pos = np.sign(
                    df_feat.loc[common_feat_idx, 'norm_ret_1d'].values)
                mom_ret = mom_pos * ret_bl
                mom_sharpe = (float(np.mean(mom_ret))
                              / (float(np.std(mom_ret, ddof=1)) + 1e-9)
                              * np.sqrt(252))

                # Mean reversion: -sign(yesterday's normalized return)
                mr_ret = -mom_pos * ret_bl
                mr_sharpe = (float(np.mean(mr_ret))
                             / (float(np.std(mr_ret, ddof=1)) + 1e-9)
                             * np.sqrt(252))

                best_baseline = max(mom_sharpe, mr_sharpe)
                advantage = gross_sharpe - best_baseline
                if advantage > 0.5:
                    verdict_3 = "PASS"
                elif advantage > 0:
                    verdict_3 = "MARGINAL"
                else:
                    verdict_3 = "FAIL (TFT doesn't beat simple baseline)"
                print(f"\n    Test 3 — Baselines:")
                print(f"      Momentum baseline: {mom_sharpe:+.2f}")
                print(f"      MR baseline:       {mr_sharpe:+.2f}")
                print(f"      TFT advantage:     {advantage:+.2f}  [{verdict_3}]")

        # ── Test 4: Cost sensitivity sweep ──
        print(f"\n    Test 4 — Cost sensitivity:")
        for bps in [0, 5, 10, 15, 20, 30]:
            cost_per_change = bps / 10000.0
            costs = pos_changes * cost_per_change
            net = strat_ret_gross - costs
            net_sharpe = (float(np.mean(net))
                          / (float(np.std(net, ddof=1)) + 1e-9)
                          * np.sqrt(252))
            total_cost_pct = costs.sum() * 100
            marker = " <-- current" if bps == int(cfg.bps_cost * 10000) else ""
            print(f"      {bps:2d} bps: Sharpe {net_sharpe:+.2f}  "
                  f"(total cost: {total_cost_pct:.2f}%){marker}")

        # ── Target alignment: first 5 OOS dates ──
        print(f"\n    Target alignment (first 5 OOS dates):")
        for i, dt in enumerate(common_idx[:5]):
            print(f"      {dt.date()}: signal={sig_a.loc[dt]:+.4f}  "
                  f"target_ret={ret_a.loc[dt]:+.6f}  "
                  f"pos={int(np.sign(sig_a.loc[dt]))}")

    # ── Fold stability ──
    if not fold_df.empty and len(fold_df) > 2:
        print(f"\n  Fold Stability:")
        for ticker in fold_df['ticker'].unique():
            t_folds = fold_df[fold_df['ticker'] == ticker]
            sharpes = t_folds['sharpe'].values
            n_pos = int(np.sum(sharpes > 0))
            n_neg = int(np.sum(sharpes <= 0))
            print(f"    {ticker}: {len(sharpes)} folds, "
                  f"mean={np.mean(sharpes):+.2f}, "
                  f"std={np.std(sharpes, ddof=1):.2f}, "
                  f"min={np.min(sharpes):+.2f}, max={np.max(sharpes):+.2f}, "
                  f"+ve/-ve={n_pos}/{n_neg}")
            if np.std(sharpes, ddof=1) > 3.0:
                print(f"      WARNING: High fold variance — edge is unstable")

    # ── PHASE 7: VISUALIZATION ──
    if not all_metrics:
        print("\n  No metrics to visualize.")
        return {'raw_data': raw_data, 'featured_data': featured_data,
                'results': results, 'all_metrics': all_metrics, 'cfg': cfg}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('QuantKubera Monolith v2 -- Walk-Forward OOS Results',
                 fontsize=14, fontweight='bold')
    
    # 6a. Equity curves (net of costs)
    ax = axes[0, 0]
    for ticker, net_s in strat_net_returns.items():
        equity = (1.0 + net_s).cumprod()
        label = f"{ticker} (Sharpe={all_metrics[ticker]['sharpe']:+.2f})"
        ax.plot(equity.index, equity.values, label=label, linewidth=1.5)
    ax.set_title('OOS Equity Curves (net of costs)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Growth of $1')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
    
    # 6b. Drawdown
    ax = axes[0, 1]
    for ticker, net_s in strat_net_returns.items():
        equity = (1.0 + net_s).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        ax.fill_between(dd.index, dd.values, alpha=0.3, label=ticker)
    ax.set_title('Drawdown')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Drawdown')
    
    # 6c. VSN Feature Importance (first ticker with weights)
    ax = axes[1, 0]
    if results['vsn_weights']:
        first_ticker = list(results['vsn_weights'].keys())[0]
        vsn_w = results['vsn_weights'][first_ticker]
        n_features_to_show = min(15, len(FEATURE_COLUMNS), len(vsn_w))
        sorted_idx = np.argsort(vsn_w)[::-1][:n_features_to_show]
        ax.barh(range(n_features_to_show),
                vsn_w[sorted_idx],
                color='steelblue')
        ax.set_yticks(range(n_features_to_show))
        ax.set_yticklabels([FEATURE_COLUMNS[i] for i in sorted_idx], fontsize=8)
        ax.set_title(f'VSN Feature Importance ({first_ticker})')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Weight')
    else:
        ax.text(0.5, 0.5, 'No VSN weights available',
                ha='center', va='center', transform=ax.transAxes)
    
    # 6d. Monthly returns bar chart (first ticker)
    ax = axes[1, 1]
    first_ticker_key = list(strat_net_returns.keys())[0]
    monthly = strat_net_returns[first_ticker_key].resample('ME').sum()
    colors = ['#d32f2f' if r < 0 else '#388e3c' for r in monthly.values]
    x_pos = range(len(monthly))
    ax.bar(x_pos, monthly.values * 100, color=colors, alpha=0.7)
    ax.set_title(f'Monthly Returns (%) -- {first_ticker_key}')
    ax.set_ylabel('Return %')
    ax.grid(True, alpha=0.3, axis='y')
    # Label x-axis with month abbreviations if not too many
    if len(monthly) <= 36:
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels([d.strftime('%b %y') for d in monthly.index],
                           rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Return results for downstream cells (VBT, etc.)
    return {
        'raw_data': raw_data,
        'featured_data': featured_data,
        'results': results,
        'all_metrics': all_metrics,
        'strat_net_returns': strat_net_returns,
        'cfg': cfg,
    }


# ── RUN THE LAB ──
lab_output = run_enhanced_lab()