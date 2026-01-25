# Spot Crypto Arbitrage: Mathematical Formulations
## Research Note for Implementation Guidance

---

## 1. Triangular Arbitrage

### 1.1 Basic Triangle Structure

**Setup**: Three trading pairs forming a closed loop.

Example: BTC/USDT, ETH/BTC, ETH/USDT

Let:
- P_ab = price of asset A in terms of asset B (how much B per 1 A)
- We start and end with the same asset (e.g., USDT)

**Arbitrage Condition (No-Arbitrage Parity)**:

For a triangle A-B-C-A, no-arbitrage implies:

    P_ab * P_bc * P_ca = 1

Or equivalently with bid/ask:

    (1/P_ask_ab) * (1/P_ask_bc) * (1/P_ask_ca) = 1   [no arbitrage]

**Clockwise Direction** (USDT -> BTC -> ETH -> USDT):

    Start: Q_usdt (quantity of USDT)

    Step 1: Buy BTC with USDT
            Q_btc = Q_usdt / P_ask(BTC/USDT)

    Step 2: Buy ETH with BTC
            Q_eth = Q_btc / P_ask(ETH/BTC)

    Step 3: Sell ETH for USDT
            Q_usdt_final = Q_eth * P_bid(ETH/USDT)

    Profit Ratio (clockwise):
            R_cw = P_bid(ETH/USDT) / (P_ask(BTC/USDT) * P_ask(ETH/BTC))

**Counter-Clockwise Direction** (USDT -> ETH -> BTC -> USDT):

    Step 1: Buy ETH with USDT
            Q_eth = Q_usdt / P_ask(ETH/USDT)

    Step 2: Sell ETH for BTC
            Q_btc = Q_eth * P_bid(ETH/BTC)

    Step 3: Sell BTC for USDT
            Q_usdt_final = Q_btc * P_bid(BTC/USDT)

    Profit Ratio (counter-clockwise):
            R_ccw = P_bid(BTC/USDT) * P_bid(ETH/BTC) / P_ask(ETH/USDT)

**Arbitrage Opportunity Exists When**:

    R_cw > 1  OR  R_ccw > 1

**Equivalent Deviation Metric**:

    delta = R - 1

    Opportunity exists when delta > 0

---

### 1.2 Fee-Adjusted Formulation

**Fee Model**:

Let f = fee rate (e.g., 0.001 for 0.1% taker fee)

Each trade retains (1 - f) of the output quantity.

**Fee-Adjusted Profit Ratio**:

    R_cw_net = R_cw * (1 - f)^3

    R_ccw_net = R_ccw * (1 - f)^3

For 3 trades with fee f each:

    (1 - f)^3 ≈ 1 - 3f    [first-order approximation]

**Example** (f = 0.001):

    (1 - 0.001)^3 = 0.997003

    Fee drag ≈ 0.3% per triangle

**Break-Even Condition**:

    R_gross >= 1 / (1 - f)^3

    R_gross >= 1 + 3f + 3f^2 + f^3  [Taylor expansion]
    R_gross >= 1 + 3f               [first-order]

**Minimum Spread for Profitability**:

    delta_min = 3f / (1 - f)^3 ≈ 3f

    For f = 0.1%:  delta_min ≈ 0.3%
    For f = 0.075% (VIP): delta_min ≈ 0.225%

**Profit Formula**:

    Profit = Q_initial * (R_gross * (1-f)^3 - 1)
           = Q_initial * (R_gross - 1 - 3f*R_gross + O(f^2))
           ≈ Q_initial * (delta_gross - 3f)

---

### 1.3 Latency-Adjusted Variant

**Price Staleness Model**:

Let t_i = timestamp of price quote i
Let t_exec = time of execution attempt
Let tau_i = t_exec - t_i = quote age (staleness)

**Execution Window Concept**:

    T_exec = total time to execute all 3 legs

    Components:
    - t_signal:   time to detect opportunity
    - t_order:    time to construct and send orders
    - t_network:  network round-trip latency
    - t_match:    exchange matching engine latency

    T_exec = t_signal + 3*(t_order + t_network + t_match)

    Typical values (Binance):
    - t_signal:  0.1-1 ms (local processing)
    - t_order:   0.1 ms
    - t_network: 1-50 ms (depends on co-location)
    - t_match:   0.1-1 ms

    Total: 5-150 ms typical

**Expected Slippage Model**:

Assume price follows geometric Brownian motion locally:

    dP/P = sigma * sqrt(dt) * Z,  where Z ~ N(0,1)

Expected absolute price change over interval dt:

    E[|dP/P|] = sigma * sqrt(2/pi) * sqrt(dt)

**Slippage as Function of Latency**:

    S(tau) = k * sigma * sqrt(tau)

    where:
    - k = market-specific constant (order of 1)
    - sigma = volatility (annualized, convert to per-ms)
    - tau = latency in same units as sigma

**Latency-Adjusted Profit**:

    R_latency = R_gross * (1 - f)^3 * (1 - S(T_exec))^3

    Approximation:
    R_latency ≈ R_gross - 3f - 3*S(T_exec)

**Effective Minimum Spread**:

    delta_min_latency = 3f + 3*k*sigma*sqrt(T_exec)

**Example Calculation**:

    sigma_daily = 3% (typical for BTC)
    sigma_per_ms = 0.03 / sqrt(86400 * 1000) ≈ 3.2e-6
    T_exec = 50 ms

    S(50ms) = 1.0 * 3.2e-6 * sqrt(50) ≈ 0.000023 = 0.0023%

    Slippage term: 3 * 0.0023% ≈ 0.007%

    (Relatively small vs fee term, but increases with volatility)

---

### 1.4 Failure Modes

**1. Partial Fills**

    Problem: Order for Q_target gets filled for Q_actual < Q_target

    Fill Rate: r = Q_actual / Q_target,  0 < r <= 1

    Impact on subsequent legs:
    - Leg 2 input reduced by factor r_1
    - Leg 3 input reduced by factor r_1 * r_2

    Residual Position:
        If leg 1 fills fully but leg 2 partial:
        Stuck with (1 - r_2) * Q_leg2_target of intermediate asset

    Mitigation Strategies:
    - Size to minimum book depth across all 3 legs
    - Use IOC (Immediate-or-Cancel) orders
    - Monitor fill rate statistics per pair

**2. Price Movement During Execution**

    Problem: Prices change between legs

    Leg execution is sequential:
        t_0: Observe prices, identify opportunity
        t_1: Execute leg 1
        t_2: Execute leg 2 (price may have moved)
        t_3: Execute leg 3 (price may have moved further)

    Risk Model:
        P_i(t_exec) = P_i(t_obs) * (1 + epsilon_i)

        where epsilon_i ~ N(0, sigma_i * sqrt(dt_i))

    Compound Risk:
        Total deviation ~ sqrt(sum of variances)

    Higher risk for:
    - Longer execution time
    - Higher volatility assets
    - Correlated price moves (if one leg moves, others likely do too)

**3. Inventory Risk**

    Problem: Accumulated position in non-base asset

    Sources:
    - Partial fills leaving residual
    - Intentional position from profitable trades

    Risk: Asset price moves while holding

    Inventory Cost Model:
        C_inventory = Q_held * sigma * sqrt(T_hold) * lambda

        where lambda = risk aversion parameter

    Mitigation:
    - Periodic rebalancing to base asset
    - Size limits per asset
    - Hedging with other instruments

**4. Quote Staleness**

    Problem: Orderbook data is out of date

    Staleness Sources:
    - WebSocket latency
    - Processing delays
    - Snapshot vs incremental update gaps

    Detection:
        If t_now - t_quote > threshold:
            Mark quote as stale, skip opportunity

    Typical thresholds: 10-100 ms depending on strategy speed

    Impact:
        Stale quotes create phantom opportunities
        Execution hits different (worse) prices

---

## 2. Cross-Crypto Statistical Arbitrage

### 2.1 Cointegration-Based Approach

**Setup**: Two or more correlated assets expected to maintain long-run relationship

Example: ETH and SOL (both "smart contract platforms")

**Spread Definition**:

For two assets X and Y:

    S(t) = log(P_x(t)) - beta * log(P_y(t)) - mu

    where:
    - beta = hedge ratio (from cointegration regression)
    - mu = long-run mean of spread

**Hedge Ratio Estimation** (Engle-Granger method):

    Step 1: Run regression
            log(P_x) = alpha + beta * log(P_y) + epsilon

    Step 2: Test residuals for stationarity (ADF test)
            If stationary => cointegrated

    Alternative: Johansen test for multiple assets

**Z-Score Normalization**:

    Z(t) = (S(t) - mu) / sigma_s

    where:
    - mu = rolling mean of S over window W
    - sigma_s = rolling std of S over window W

    Typical W: 20-100 periods (depending on timeframe)

**Trading Signal**:

    Z > +threshold  =>  Short spread (sell X, buy Y)
    Z < -threshold  =>  Long spread (buy X, sell Y)
    |Z| < exit_threshold  =>  Close position

    Common thresholds:
    - Entry: |Z| > 2.0
    - Exit: |Z| < 0.5

**Mean-Reversion Assumption**:

The spread follows Ornstein-Uhlenbeck process:

    dS = theta * (mu - S) * dt + sigma * dW

    where:
    - theta = mean-reversion speed (higher = faster reversion)
    - mu = long-run mean
    - sigma = volatility of spread

**Half-Life of Mean Reversion**:

    t_half = ln(2) / theta

    Estimation:
        Run regression: dS = a + b*S + epsilon
        theta = -b
        t_half = -ln(2) / b

**Condition for Viability**:

    t_half should be:
    - Long enough to execute (> T_exec)
    - Short enough for capital efficiency (< max_holding_period)

    Typical sweet spot: 1 hour to 1 week

---

### 2.2 Fee-Adjusted Formulation

**Round-Trip Cost Calculation**:

To enter and exit a spread trade:

    Entry:
    - Buy X (or Y): pay fee f
    - Sell Y (or X): pay fee f

    Exit:
    - Reverse positions: 2 more fees

    Total fees = 4 * f * notional_per_leg

For dollar-neutral spread (equal notional on each leg):

    Notional per leg = N
    Total capital = 2N
    Total fees = 4 * f * N = 2 * f * (2N) = 2f * Capital

**Minimum Spread for Profitability**:

    Let spread move from entry Z_e to exit Z_x

    Expected P&L (before fees):
        PnL_gross = Capital * |Z_e - Z_x| * sigma_s / 2

    (Factor of 2 because capital split between two legs)

    Break-even condition:
        |Z_e - Z_x| * sigma_s / 2 > 2f

        |Z_e - Z_x| > 4f / sigma_s

**Example**:

    f = 0.1%, sigma_s = 1%
    |Z_e - Z_x| > 4 * 0.001 / 0.01 = 0.4 standard deviations

    With entry at Z=2.0 and exit at Z=0.5:
    |Z_e - Z_x| = 1.5 >> 0.4  =>  Profitable

**Holding Period Considerations**:

Opportunity cost of capital:

    If r = risk-free rate (annualized)
    Holding period = T days

    Opportunity cost = Capital * r * T/365

    Must add to break-even calculation

Funding costs (if using margin):

    Funding_cost = Borrowed_amount * funding_rate * T

---

### 2.3 Latency-Adjusted Variant

**Signal Decay Model**:

Z-score signal decays as other participants act on it:

    Z(t + dt) = Z(t) * exp(-lambda * dt) + noise

    where lambda = signal decay rate

**Half-Life of Signal**:

    t_signal_half = ln(2) / lambda

    Empirical estimation:
    - Measure Z-score at detection
    - Track subsequent evolution
    - Fit exponential decay

**Execution Timing Risk**:

Entry requires 2 simultaneous trades (both legs).

If executed sequentially:

    Leg 1 at time t
    Leg 2 at time t + delta_t

    Risk: hedge ratio may be off

    Realized hedge ratio: beta_realized = beta_target * (1 + epsilon)

    where epsilon ~ N(0, sigma_beta * sqrt(delta_t))

**Partial Fill Impact on Hedge Ratios**:

    Target position:
        Long Q_x of X
        Short Q_y = beta * Q_x of Y

    If leg X fills fully but leg Y fills r * Q_y:

        Actual hedge ratio = r * beta

        Unhedged exposure = (1 - r) * beta * Q_x worth of Y

**Hedge Ratio Drift**:

    Beta is estimated from historical data
    True beta may drift over time

    Model: beta(t) = beta_0 + integral(d_beta)

    Risk: P&L has exposure to beta error
        PnL_error = Q_x * (beta_true - beta_est) * dP_y

---

### 2.4 Failure Modes

**1. Regime Changes (Correlation Breakdown)**

    Problem: Historical relationship no longer holds

    Examples:
    - Chain-specific events (hack, upgrade)
    - Regulatory news affecting one asset
    - Narrative shifts ("ETH killer" rotations)

    Detection:
        - Rolling correlation drops below threshold
        - Cointegration test fails on recent window
        - Spread exceeds N standard deviations (usually 3-4)

    Metrics to monitor:
        corr(X, Y) over rolling window
        ADF test p-value on spread
        Spread kurtosis (fat tails = regime change risk)

    Mitigation:
        - Stop-loss on spread position
        - Automatic pause if cointegration breaks
        - Reduce position size when correlation unstable

**2. Liquidity Asymmetry**

    Problem: One leg of pair has much less liquidity

    Impact:
        - Harder to execute full size
        - More slippage on illiquid leg
        - Wider effective spread

    Measurement:
        Liquidity ratio = depth_X / depth_Y at N bps from mid

        If ratio far from 1, asymmetric risk

    Adjusted sizing:
        Q_trade = min(size_target, min_depth / slippage_tolerance)

**3. Funding Rate Divergence (If Perps Involved)**

    Problem: Spread includes perpetual futures, funding rates diverge

    Impact:
        Funding payment = Position_size * funding_rate * Time

        If long asset with high funding: pay funding
        If short asset with high funding: receive funding

        Can dominate P&L for longer holding periods

    Break-even adjustment:
        Must include expected funding in spread calculation

        Effective_spread = Price_spread - Expected_funding_cost

    Note: This doc focuses on spot, but if perps used for hedging,
          funding is critical consideration

---

## 3. Key Assumptions

### 3.1 Triangular Arbitrage Assumptions

| Assumption | When It Breaks | Impact |
|------------|----------------|--------|
| Prices are executable | Stale quotes, thin books | Worse fills, losses |
| Fees are fixed | Fee tier changes, promos | Model accuracy |
| No position limits | Exchange restrictions | Can't complete triangle |
| Instant execution | Network latency, queue | Price slippage |
| Full fills | Thin liquidity | Residual inventory |
| Independent legs | Correlated movements | Compound risk |

### 3.2 Statistical Arbitrage Assumptions

| Assumption | When It Breaks | Impact |
|------------|----------------|--------|
| Cointegration holds | Regime change, news | Spread diverges, loss |
| Spread is stationary | Structural break | No mean reversion |
| Beta is stable | Market regime shift | Hedge ineffective |
| Volatility is stable | Vol expansion | Position sizing wrong |
| Can execute both legs | Liquidity asymmetry | Unhedged exposure |
| Normal distribution | Fat tails | Underestimate risk |

### 3.3 Impact of Assumption Violations

**Triangular - Quote Staleness**:

    If quotes are T ms old on average:
    Expected adverse selection = k * sigma * sqrt(T)

    For T = 100ms, sigma_daily = 3%:
    Adverse selection ≈ 0.01% per leg

**Statistical Arb - Cointegration Break**:

    If spread goes from Z=2 to Z=5 (instead of reverting):
    Loss = Capital * 3 * sigma_s / 2

    For sigma_s = 1%, Capital = $100k:
    Loss = $100k * 3 * 0.01 / 2 = $1,500

    (Can be much worse if spread keeps diverging)

---

## 4. Observable Metrics for Implementation

### 4.1 Data Quality Metrics

**Triangular Arbitrage**:

| Metric | How to Measure | Threshold |
|--------|----------------|-----------|
| Quote age | t_now - t_quote | < 50ms ideal, < 200ms acceptable |
| Book depth | Sum of qty in top N levels | > 2x trade size at each level |
| Spread (bid-ask) | ask - bid | < 0.05% for majors |
| Update frequency | Messages per second | > 10/sec for active pairs |
| Sequence gaps | Missing sequence numbers | 0 gaps |

**Statistical Arbitrage**:

| Metric | How to Measure | Threshold |
|--------|----------------|-----------|
| Correlation | Rolling Pearson correlation | > 0.7 for pair trading |
| Cointegration p-value | ADF test on spread | < 0.05 |
| Half-life | Regression on spread changes | 1 hour - 1 week |
| Spread stationarity | Variance ratio test | Ratio ≈ 1 |
| Beta stability | Rolling beta std dev | < 10% of beta |

### 4.2 Execution Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Fill rate | Filled_qty / Ordered_qty | > 95% |
| Slippage | Exec_price / Expected_price - 1 | < 0.02% |
| Latency | t_ack - t_send | < 10ms |
| Reject rate | Rejected_orders / Total_orders | < 1% |

### 4.3 Strategy Performance Metrics

**Triangular**:

    Detection rate: Opportunities detected per hour
    Capture rate: Profitable executions / Opportunities detected
    Average profit: Mean profit per successful triangle
    Sharpe (annualized): Mean_daily_return / Std_daily_return * sqrt(365)

**Statistical Arb**:

    Win rate: Profitable trades / Total trades
    Avg win/loss ratio: Avg_profit_on_wins / Avg_loss_on_losses
    Max drawdown: Peak_to_trough decline
    Time in market: Fraction of time with open position

### 4.4 Warning Signals

**Triangular - Stop/Pause Conditions**:

    - Fill rate drops below 80%
    - Average slippage exceeds fee savings
    - Quote latency > 500ms
    - Loss on > 3 consecutive triangles

**Statistical Arb - Stop/Pause Conditions**:

    - Correlation drops below 0.5
    - ADF p-value > 0.10
    - Spread exceeds 4 standard deviations
    - Half-life < 30 minutes (too fast) or > 2 weeks (too slow)
    - Beta changes by > 20% in a week

---

## Summary Formulas Quick Reference

### Triangular Arbitrage

    Profit condition:  R * (1-f)^3 > 1
    Min spread:        delta_min ≈ 3f
    With latency:      delta_min ≈ 3f + 3*k*sigma*sqrt(T_exec)

### Statistical Arbitrage

    Spread:            S(t) = log(P_x) - beta * log(P_y) - mu
    Z-score:           Z(t) = (S(t) - mu) / sigma_s
    Half-life:         t_half = ln(2) / theta
    Min Z-move:        |Z_entry - Z_exit| > 4f / sigma_s

---

*Document Version: 1.0*
*Purpose: Implementation guidance for QuantLaxmi crypto arbitrage strategies*
