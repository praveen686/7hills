// ============================================================
// BRAHMASTRA Type Definitions
// Mirrors FastAPI Pydantic models
// ============================================================

// ---------- Portfolio ----------
// Backend PortfolioOut shape
export interface PortfolioSummary {
  equity: number;
  peak_equity: number;
  cash: number;
  drawdown_pct: number;
  total_exposure: number;
  total_return_pct: number;
  win_rate: number;
  n_positions: number;
  n_closed_trades: number;
  last_scan_date: string;
  last_scan_time: string;
  scan_count: number;
  circuit_breaker_active: boolean;
  last_vix: number;
  last_regime: string;
  positions: PortfolioPosition[];
  recent_trades: ClosedTrade[];
  strategy_equity: StrategyEquity[];

  // Fields that may come from backend or be aliased from other fields
  total_equity?: number;
  day_pnl?: number;
  day_pnl_pct?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  margin_used?: number;
  margin_available?: number;
  positions_count?: number;
  updated_at?: string;
  equity_curve?: EquityPoint[];
}

export interface PortfolioPosition {
  strategy_id: string;
  symbol: string;
  direction: string;
  weight: number;
  instrument_type: string;
  entry_date: string;
  entry_price: number;
  strike?: number;
  expiry?: string;
  current_price?: number;
  unrealized_pnl?: number;
}

export interface ClosedTrade {
  strategy_id: string;
  symbol: string;
  direction: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  weight: number;
  pnl_pct: number;
  instrument_type?: string;
  exit_reason?: string;
}

export interface StrategyEquity {
  strategy_id: string;
  equity: number;
  peak: number;
  drawdown_pct: number;
}

// ---------- Strategy ----------
export type StrategyStatus = "live" | "paused" | "stopped" | "backtest" | "running" | "stale";

export interface StrategySummary {
  // Backend StrategySummaryOut fields
  strategy_id: string;
  name: string;
  status: string;
  equity: number;
  return_pct: number;
  n_open: number;
  n_closed: number;
  win_rate: number;

  // Real backtest metrics from research artefacts
  sharpe?: number;
  max_dd?: number;
  tier?: string;
  best_config?: string;

  // Aliases for UI compatibility
  id?: string;
  cagr?: number;
  max_drawdown?: number;
  day_pnl?: number;
  total_pnl?: number;
  positions_count?: number;
  total_trades?: number;
}

export interface StrategyDetail extends StrategySummary {
  positions: Record<string, unknown>[];
  recent_trades: StrategyTrade[];
  metadata: Record<string, unknown>;

  // Backend research enrichment
  date_range?: string;

  // UI-only fields (may be undefined)
  description?: string;
  instruments?: string[];
  timeframe?: string;
  equity_curve?: EquityPoint[];
  signals?: Signal[];
  params?: Record<string, number | string | boolean>;
}

export interface StrategyTrade {
  symbol: string;
  direction: string;
  entry_date: string;
  exit_date: string;
  pnl_pct: number;
  exit_reason: string;
}

// ---------- Positions & Trades ----------
export interface Position {
  instrument: string;
  symbol: string;
  quantity: number;
  avg_price: number;
  ltp: number;
  pnl: number;
  pnl_pct?: number;
  side: "LONG" | "SHORT";
  strategy_id: string;
}

export interface Trade {
  id: string;
  timestamp: string;
  instrument: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  pnl: number;
  strategy_id: string;
}

// ---------- Signals ----------
export type SignalDirection = "BUY" | "SELL" | "HOLD";

export interface Signal {
  id: string;
  timestamp: string;
  instrument: string;
  symbol: string;
  direction: SignalDirection;
  strength: number;
  strategy_id: string;
  strategy_name: string;
  price: number;
  target?: number;
  stop_loss?: number;

  // WAL event log enrichment
  components?: Record<string, unknown> | null;
  reasoning?: string;
  regime?: string;
  approved?: boolean | null;
}

// ---------- Charts ----------
export interface EquityPoint {
  date: string;
  equity: number;
  drawdown?: number;
  benchmark?: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

// ---------- Market ----------
export interface MarketTick {
  symbol: string;
  ltp: number;
  change: number;
  change_pct: number;
  volume: number;
  oi: number;
  timestamp: string;
}

export interface OptionChainEntry {
  strike: number;
  ce_ltp: number;
  ce_oi: number;
  ce_iv: number;
  ce_delta: number;
  pe_ltp: number;
  pe_oi: number;
  pe_iv: number;
  pe_delta: number;
}

export interface IVSurfacePoint {
  strike: number;
  expiry: string;
  iv: number;
}

// ---------- Risk ----------
// Backend RiskOut shape — fields marked "backend (new)" are being added
// by backend agents and may not be present on older deployments.
export interface RiskMetrics {
  // Core fields always returned by GET /api/risk
  portfolio_drawdown_pct: number;
  total_exposure: number;
  cash: number;
  circuit_breaker_active: boolean;
  last_vpin: number;
  last_vix: number;
  last_regime: string;
  n_positions: number;
  n_strategies: number;
  strategies: StrategyRisk[];
  concentration: ConcentrationData;

  // Backend (new) — real computed Greeks, VaR, drawdown from equity history
  portfolio_delta?: number;
  portfolio_gamma?: number;
  portfolio_theta?: number;
  portfolio_vega?: number;
  var_95?: number;
  var_99?: number;
  portfolio_max_drawdown_pct?: number;
  greeks?: GreeksRow[];
  drawdown_history?: DrawdownPoint[];
  circuit_breaker_reason?: string;

  // UI convenience aliases (derived by fetchRiskMetrics adapter when backend
  // fields above are not yet available)
  max_drawdown?: number;
  current_drawdown?: number;
  margin_utilization?: number;
}

export interface StrategyRisk {
  strategy_id: string;
  equity: number;
  drawdown_pct: number;
  n_positions: number;
  exposure: number;
}

export interface ConcentrationData {
  top_symbols: Record<string, number>;
  long_exposure: number;
  short_exposure: number;
  net_exposure: number;
  gross_exposure: number;
}

export interface GreeksRow {
  instrument: string;
  symbol: string;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  iv: number;
}

// ---------- Backtest ----------
export interface BacktestParams {
  strategy_id: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  params?: Record<string, number | string | boolean>;
}

export interface BacktestStrategyInfo {
  strategy_id: string;
  name: string;
  default_params: Record<string, number | string | boolean>;
}

// Backend BacktestStatusOut shape
export interface BacktestResult {
  backtest_id?: string;
  id?: string;
  strategy_id: string;
  start_date: string;
  end_date: string;
  status?: string;
  created_at?: string;
  completed_at?: string;
  error?: string;
  result?: BacktestMetrics;

  // Flat fields for UI convenience
  initial_capital?: number;
  final_equity?: number;
  total_return?: number;
  cagr?: number;
  sharpe?: number;
  sortino?: number;
  max_drawdown?: number;
  win_rate?: number;
  profit_factor?: number;
  total_trades?: number;
  avg_trade_pnl?: number;
  equity_curve?: EquityPoint[];
  drawdown_curve?: DrawdownPoint[];
  monthly_returns?: MonthlyReturn[];
}

export interface BacktestMetrics {
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  n_trades: number;
  win_rate: number;
  profit_factor: number;
  avg_trade_pnl: number;
  total_costs: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

// ---------- Research ----------
export interface FeatureIC {
  feature: string;
  ic_mean: number;
  ic_std?: number | null;
  icir?: number | null;
  rank_ic?: number | null;
  p_value?: number | null;
  horizon?: string;
  source?: string;
}

export interface WalkForwardResult {
  fold: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  train_sharpe: number;
  test_sharpe: number;
  degradation: number;
}

// ---------- VIX ----------
export interface VIXData {
  value: number;
  change: number;
  change_pct: number;
  timestamp: string;
}

// ---------- Why Panel (Operator Explainability) ----------
export interface SignalContext {
  signal_seq: number;
  ts: string;
  strategy_id: string;
  symbol: string;
  direction: string;
  conviction: number;
  instrument_type: string;
  strike: number;
  expiry: string;
  ttl_bars: number;
  regime: string;
  components: Record<string, unknown>;
  reasoning: string;
}

export interface GateDecision {
  seq: number;
  ts: string;
  gate: string;
  approved: boolean;
  adjusted_weight: number;
  reason: string;
  vpin: number;
  portfolio_dd: number;
  strategy_dd: number;
  total_exposure: number;
}

export interface WhyEvent {
  seq: number;
  ts: string;
  event_type: string;
  strategy_id: string;
  symbol: string;
  payload: Record<string, unknown>;
}

export interface TradeDecisionChain {
  strategy_id: string;
  symbol: string;
  date: string;
  signals: WhyEvent[];
  gates: WhyEvent[];
  orders: WhyEvent[];
  fills: WhyEvent[];
  risk_alerts: WhyEvent[];
  snapshot: WhyEvent | null;
}

// ---------- WebSocket Messages ----------
export interface WSMessage<T = unknown> {
  type: string;
  data: T;
  timestamp: string;
}
