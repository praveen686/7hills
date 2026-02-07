// ============================================================
// BRAHMASTRA API Client
// All REST communication with FastAPI backend
// ============================================================

import type {
  PortfolioSummary,
  StrategySummary,
  StrategyDetail,
  Signal,
  RiskMetrics,
  BacktestParams,
  BacktestResult,
  BacktestStrategyInfo,
  FeatureIC,
  WalkForwardResult,
  VIXData,
  OptionChainEntry,
  SignalContext,
  GateDecision,
  TradeDecisionChain,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------- Generic Fetch Wrapper ----------

async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;

  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const errorBody = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, `API ${res.status}: ${errorBody}`, endpoint);
  }

  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public endpoint: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ---------- Portfolio ----------

// Base capital for converting equity ratio (1.0) to INR value.
// Backend stores equity as a ratio; UI displays in INR.
const BASE_CAPITAL = 1_00_00_000; // ₹1 Crore

export async function fetchPortfolio(): Promise<PortfolioSummary> {
  const data = await apiFetch<PortfolioSummary>("/api/portfolio");

  // Scale equity ratio → INR values
  const equityRatio = data.equity ?? 1.0;
  data.total_equity = equityRatio * BASE_CAPITAL;
  data.total_pnl = (equityRatio - 1.0) * BASE_CAPITAL;
  data.total_pnl_pct = data.total_return_pct;
  data.margin_used = (data.total_exposure ?? 0) * BASE_CAPITAL;
  data.margin_available = (data.cash ?? 0) * BASE_CAPITAL;
  if (data.day_pnl != null) {
    const dayPnlInr = data.day_pnl * BASE_CAPITAL;
    data.day_pnl_pct = equityRatio > 0 ? (data.day_pnl / equityRatio) * 100 : 0;
    data.day_pnl = dayPnlInr;
  }

  // Scale equity curve to INR
  if (data.equity_curve) {
    data.equity_curve = data.equity_curve.map((pt: { date: string; equity: number }) => ({
      ...pt,
      equity: pt.equity * BASE_CAPITAL,
    }));
  }

  if (data.positions_count == null) data.positions_count = data.n_positions;
  if (data.updated_at == null) data.updated_at = data.last_scan_date || new Date().toISOString();
  return data;
}

// ---------- Strategies ----------

export async function fetchStrategies(): Promise<StrategySummary[]> {
  const data = await apiFetch<{ count: number; strategies: StrategySummary[] }>(
    "/api/strategies"
  );
  // Map backend field names to UI convenience aliases (no fabricated values)
  return data.strategies.map((s) => ({
    ...s,
    id: s.id ?? s.strategy_id,
    total_pnl: s.total_pnl ?? s.return_pct,
    total_trades: s.total_trades ?? s.n_closed,
    positions_count: s.positions_count ?? s.n_open,
    max_drawdown: s.max_drawdown ?? s.max_dd,
  }));
}

export async function fetchStrategy(id: string): Promise<StrategyDetail> {
  const data = await apiFetch<StrategyDetail>(`/api/strategies/${id}`);
  data.id = data.strategy_id;
  return data;
}

export async function updateStrategyStatus(
  id: string,
  status: "live" | "paused" | "stopped"
): Promise<void> {
  await apiFetch(`/api/strategies/${id}/status`, {
    method: "PUT",
    body: JSON.stringify({ status }),
  });
}

// ---------- Signals ----------

export async function fetchSignals(): Promise<Signal[]> {
  return apiFetch<Signal[]>("/api/signals");
}

export async function fetchTodaySignals(): Promise<Signal[]> {
  return apiFetch<Signal[]>("/api/signals/today");
}

// ---------- Risk ----------

export async function fetchRiskMetrics(): Promise<RiskMetrics> {
  const data = await apiFetch<RiskMetrics>("/api/risk");

  // Map backend fields to UI convenience aliases only when the backend
  // hasn't already provided them.  Backend agents are adding real Greeks,
  // VaR, and drawdown_history to the response; those take priority.

  // Drawdown: backend always provides portfolio_drawdown_pct
  if (data.current_drawdown == null) {
    data.current_drawdown = data.portfolio_drawdown_pct;
  }
  if (data.max_drawdown == null) {
    // Use dedicated max field from backend if available, else fall back
    data.max_drawdown = data.portfolio_max_drawdown_pct ?? data.portfolio_drawdown_pct;
  }

  // Margin utilization as a 0-1 ratio (not raw exposure).
  if (data.margin_utilization == null) {
    const denom = data.total_exposure + data.cash;
    data.margin_utilization = denom > 0 ? data.total_exposure / denom : 0;
  }

  // portfolio_delta, portfolio_gamma, portfolio_theta, portfolio_vega:
  //   Passed through from backend when present.  UI displays "--" for undefined.

  // var_95, var_99:
  //   Passed through from backend when present.  UI displays "--" for undefined.

  // greeks (per-position array):
  //   Passed through from backend when present.  UI shows "No data" for undefined.

  // drawdown_history:
  //   Passed through from backend when present.  DrawdownChart handles empty/undefined.

  return data;
}

// ---------- Market ----------

export async function fetchVIX(): Promise<VIXData> {
  return apiFetch<VIXData>("/api/market/vix");
}

export async function fetchOptionChain(
  symbol: string,
  expiry?: string
): Promise<OptionChainEntry[]> {
  const params = new URLSearchParams({ symbol });
  if (expiry) params.set("expiry", expiry);
  return apiFetch<OptionChainEntry[]>(
    `/api/market/option-chain?${params.toString()}`
  );
}

// ---------- Backtest ----------

export async function fetchBacktestStrategies(): Promise<BacktestStrategyInfo[]> {
  return apiFetch<BacktestStrategyInfo[]>("/api/backtest/strategies");
}

export async function runBacktest(
  params: BacktestParams
): Promise<BacktestResult> {
  // Launch async backtest — send only the fields the backend expects
  const launch = await apiFetch<{ backtest_id: string; status: string }>(
    "/api/backtest/run",
    {
      method: "POST",
      body: JSON.stringify({
        strategy_id: params.strategy_id,
        start_date: params.start_date,
        end_date: params.end_date,
        initial_capital: params.initial_capital,
        params: params.params ?? {},
      }),
    }
  );

  // Poll until completed or failed (max 5 minutes)
  const maxAttempts = 60;
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 5000));
    const raw = await apiFetch<Record<string, unknown>>(
      `/api/backtest/${launch.backtest_id}/status`
    );
    if (raw.status === "completed" || raw.status === "failed") {
      // Flatten nested result into top-level fields
      const nested = (raw.result ?? {}) as Record<string, unknown>;

      // Backend returns total_return/max_drawdown as fractions (e.g. 0.05 = 5%)
      // UI displays as percentages
      return {
        backtest_id: raw.backtest_id as string,
        strategy_id: raw.strategy_id as string,
        start_date: raw.start_date as string,
        end_date: raw.end_date as string,
        status: raw.status as string,
        created_at: raw.created_at as string,
        completed_at: raw.completed_at as string | undefined,
        error: raw.error as string | undefined,
        total_return: nested.total_return != null ? (nested.total_return as number) * 100 : undefined,
        sharpe: nested.sharpe_ratio as number | undefined,
        sortino: nested.sortino_ratio as number | undefined,
        max_drawdown: nested.max_drawdown != null ? (nested.max_drawdown as number) * 100 : undefined,
        win_rate: nested.win_rate as number | undefined,
        profit_factor: nested.profit_factor as number | undefined,
        total_trades: nested.n_trades as number | undefined,
        avg_trade_pnl: nested.avg_trade_pnl != null ? (nested.avg_trade_pnl as number) * 100 : undefined,
        final_equity: nested.final_equity as number | undefined,
        cagr: undefined,
        equity_curve: nested.equity_curve as BacktestResult["equity_curve"],
        drawdown_curve: nested.drawdown_curve as BacktestResult["drawdown_curve"],
        monthly_returns: nested.monthly_returns as BacktestResult["monthly_returns"],
      } as BacktestResult;
    }
  }

  return {
    backtest_id: launch.backtest_id,
    strategy_id: params.strategy_id,
    start_date: params.start_date,
    end_date: params.end_date,
    status: "timeout",
    error: "Backtest timed out after 5 minutes",
  };
}

export async function fetchBacktestResults(): Promise<BacktestResult[]> {
  return apiFetch<BacktestResult[]>("/api/backtest/results");
}

export async function fetchBacktestResult(id: string): Promise<BacktestResult> {
  return apiFetch<BacktestResult>(`/api/backtest/${id}/status`);
}

// ---------- Research ----------

export async function fetchFeatureIC(): Promise<FeatureIC[]> {
  return apiFetch<FeatureIC[]>("/api/research/feature-ic");
}

export async function fetchWalkForwardResults(): Promise<WalkForwardResult[]> {
  return apiFetch<WalkForwardResult[]>("/api/research/walk-forward");
}

// ---------- Why Panel ----------

export async function fetchSignalContext(
  signalSeq: number,
  date: string
): Promise<SignalContext> {
  return apiFetch<SignalContext>(
    `/api/why/signals/${signalSeq}/context?date=${date}`
  );
}

export async function fetchGateDecisions(
  signalSeq: number,
  date: string
): Promise<GateDecision[]> {
  return apiFetch<GateDecision[]>(
    `/api/why/gates/${signalSeq}?date=${date}`
  );
}

export async function fetchTradeDecisions(
  strategyId: string,
  symbol: string,
  date: string
): Promise<TradeDecisionChain> {
  return apiFetch<TradeDecisionChain>(
    `/api/why/trades/${strategyId}/${symbol}/${date}`
  );
}

export async function fetchWhyDates(): Promise<string[]> {
  return apiFetch<string[]>("/api/why/dates");
}

// ---------- WebSocket URLs ----------

export function getWSUrl(path: string): string {
  const wsBase = BASE_URL.replace(/^http/, "ws");
  return `${wsBase}${path}`;
}

export const WS_ENDPOINTS = {
  signals: "/ws/signals",
  ticks: "/ws/ticks",
  portfolio: "/ws/portfolio",
  risk: "/ws/risk",
} as const;
