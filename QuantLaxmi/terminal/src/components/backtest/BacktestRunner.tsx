import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { BacktestResultData } from "@/components/backtest/BacktestResults";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BacktestStatus = "configuring" | "running" | "complete" | "error";

interface StrategyOption {
  id: string;
  name: string;
  defaultParams: Record<string, number | string>;
}

interface BacktestConfig {
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  params: Record<string, number | string>;
}

interface BacktestRunnerProps {
  onComplete?: (result: BacktestResultData) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map raw backend result object to BacktestResultData. */
function mapBacktestResult(r: any): BacktestResultData {
  return {
    totalReturn: (r.total_return ?? 0) * 100,
    sharpe: r.sharpe_ratio ?? 0,
    sortino: r.sortino_ratio ?? 0,
    maxDD: (r.max_drawdown ?? 0) * 100,
    winRate: (r.win_rate ?? 0) / 100,
    profitFactor: r.profit_factor ?? 0,
    totalTrades: r.n_trades ?? 0,
    equityCurve: (r.equity_curve ?? []).map((p: any) => ({
      time: p.date,
      value: p.equity,
    })),
    drawdownCurve: (r.drawdown_curve ?? []).map((p: any) => ({
      time: p.date,
      value: (p.drawdown ?? 0) * 100,
    })),
    monthlyReturns: (r.monthly_returns ?? []).map((p: any) => ({
      year: p.year,
      month: p.month,
      returnPct: p.return_pct,
    })),
    trades: [],
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BacktestRunner({ onComplete }: BacktestRunnerProps) {
  const [strategies, setStrategies] = useState<StrategyOption[]>([]);
  const [strategiesLoading, setStrategiesLoading] = useState(true);
  const [strategiesError, setStrategiesError] = useState<string | null>(null);
  const [status, setStatus] = useState<BacktestStatus>("configuring");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<BacktestConfig>({
    strategyId: "",
    startDate: "2025-01-01",
    endDate: "2026-02-12",
    initialCapital: 10000000,
    params: {},
  });

  // Fetch strategy list on mount with fallback
  useEffect(() => {
    setStrategiesLoading(true);
    setStrategiesError(null);

    apiFetch<Array<{ strategy_id: string; name: string; default_params: Record<string, any> }>>(
      "/api/backtest/strategies"
    ).then((list) => {
      const mapped = list.map((s) => ({
        id: s.strategy_id,
        name: s.name,
        defaultParams: s.default_params,
      }));
      setStrategies(mapped);
      if (mapped.length > 0) {
        setConfig((prev) => prev.strategyId ? prev : ({
          ...prev,
          strategyId: mapped[0].id,
          params: { ...mapped[0].defaultParams },
        }));
      }
    }).catch(() => {
      // Fallback: try the main strategies endpoint
      return apiFetch<{ strategies: Array<{ strategy_id: string; name: string }> }>(
        "/api/strategies"
      ).then((data) => {
        const mapped = data.strategies.map((s) => ({
          id: s.strategy_id,
          name: s.name,
          defaultParams: {},
        }));
        setStrategies(mapped);
        if (mapped.length > 0) {
          setConfig((prev) => prev.strategyId ? prev : ({
            ...prev,
            strategyId: mapped[0].id,
          }));
        }
      });
    }).catch((err) => {
      setStrategiesError(
        err instanceof Error ? err.message : "Failed to load strategies. Is the API running?"
      );
    }).finally(() => {
      setStrategiesLoading(false);
    });
  }, []);

  // When strategy changes, load the most recent completed backtest result
  useEffect(() => {
    if (!config.strategyId) return;

    apiFetch<Array<{
      backtest_id: string;
      status: string;
      strategy_id: string;
      start_date: string;
      end_date: string;
      created_at: string;
      completed_at: string | null;
      error: string | null;
      progress: number;
      result: any | null;
    }>>("/api/backtest/history")
      .then((history) => {
        const match = history.find(
          (h) =>
            h.status === "completed" &&
            h.strategy_id === config.strategyId &&
            h.result != null
        );
        if (match) {
          onComplete?.(mapBacktestResult(match.result));
        }
      })
      .catch(() => {});
  }, [config.strategyId]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedStrategy = strategies.find((s) => s.id === config.strategyId);

  const handleStrategyChange = useCallback((id: string) => {
    const strat = strategies.find((s) => s.id === id);
    setConfig((prev) => ({
      ...prev,
      strategyId: id,
      params: strat ? { ...strat.defaultParams } : prev.params,
    }));
  }, [strategies]);

  const handleParamChange = useCallback((key: string, value: string) => {
    setConfig((prev) => ({
      ...prev,
      params: { ...prev.params, [key]: isNaN(Number(value)) ? value : Number(value) },
    }));
  }, []);

  const handleRun = useCallback(async () => {
    setStatus("running");
    setProgress(0);
    setError(null);

    try {
      // POST to launch backtest — returns 202 with backtest_id
      const launch = await apiFetch<{ backtest_id: string }>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify({
          strategy_id: config.strategyId,
          start_date: config.startDate,
          end_date: config.endDate,
          initial_capital: config.initialCapital,
          params: config.params,
        }),
      });

      const backtestId = launch.backtest_id;

      // Poll for completion every 2s
      for (let attempt = 0; attempt < 300; attempt++) {
        await new Promise((r) => setTimeout(r, 2000));

        const poll = await apiFetch<{
          status: string;
          error: string | null;
          progress: number;
          result: any | null;
        }>(`/api/backtest/${backtestId}/status`);

        setProgress(poll.progress ?? 0);

        if (poll.status === "completed" && poll.result) {
          const mapped = mapBacktestResult(poll.result);
          setProgress(100);
          setStatus("complete");
          onComplete?.(mapped);
          return;
        }

        if (poll.status === "failed") {
          throw new Error(poll.error || "Backtest failed");
        }
      }

      throw new Error("Backtest timed out after 10 minutes");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
    }
  }, [config, onComplete]);

  const handleReset = useCallback(() => {
    setStatus("configuring");
    setProgress(0);
    setError(null);
  }, []);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
        Backtest Runner
      </h2>

      {/* Strategy selector */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-terminal-muted">Strategy</label>
        {strategiesLoading ? (
          <div className="w-full rounded bg-terminal-surface border border-terminal-border px-3 py-2 text-sm text-terminal-muted animate-pulse">
            Loading strategies...
          </div>
        ) : strategiesError ? (
          <div className="rounded border border-terminal-loss/40 bg-terminal-loss/10 px-3 py-2 text-xs text-terminal-loss">
            {strategiesError}
          </div>
        ) : (
          <select
            value={config.strategyId}
            onChange={(e) => handleStrategyChange(e.target.value)}
            disabled={status === "running"}
            className="w-full rounded bg-terminal-surface border border-terminal-border px-3 py-2 text-sm text-terminal-text focus:outline-none focus:border-terminal-accent disabled:opacity-50"
          >
            {strategies.length === 0 && (
              <option value="">No strategies available</option>
            )}
            {strategies.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        )}
      </div>

      {/* Date range */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs font-medium text-terminal-muted">Start Date</label>
          <input
            type="date"
            value={config.startDate}
            onChange={(e) => setConfig((p) => ({ ...p, startDate: e.target.value }))}
            disabled={status === "running"}
            className="w-full rounded bg-terminal-surface border border-terminal-border px-3 py-2 text-sm text-terminal-text font-mono focus:outline-none focus:border-terminal-accent disabled:opacity-50"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-terminal-muted">End Date</label>
          <input
            type="date"
            value={config.endDate}
            onChange={(e) => setConfig((p) => ({ ...p, endDate: e.target.value }))}
            disabled={status === "running"}
            className="w-full rounded bg-terminal-surface border border-terminal-border px-3 py-2 text-sm text-terminal-text font-mono focus:outline-none focus:border-terminal-accent disabled:opacity-50"
          />
        </div>
      </div>

      {/* Initial capital */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-terminal-muted">Initial Capital (INR)</label>
        <input
          type="number"
          value={config.initialCapital}
          onChange={(e) => setConfig((p) => ({ ...p, initialCapital: Number(e.target.value) }))}
          disabled={status === "running"}
          step={100000}
          className="w-full rounded bg-terminal-surface border border-terminal-border px-3 py-2 text-sm text-terminal-text font-mono focus:outline-none focus:border-terminal-accent disabled:opacity-50"
        />
      </div>

      {/* Strategy-specific parameters */}
      {selectedStrategy && Object.keys(config.params).length > 0 && (
        <div className="space-y-2">
          <label className="text-xs font-medium text-terminal-muted">Strategy Parameters</label>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(config.params).map(([key, val]) => (
              <div key={key} className="space-y-0.5">
                <label className="text-2xs text-terminal-muted">{key}</label>
                <input
                  type={typeof val === "number" ? "number" : "text"}
                  value={val}
                  onChange={(e) => handleParamChange(key, e.target.value)}
                  disabled={status === "running"}
                  step={typeof val === "number" ? (val < 1 ? 0.1 : 1) : undefined}
                  className="w-full rounded bg-terminal-bg border border-terminal-border px-2 py-1.5 text-xs text-terminal-text font-mono focus:outline-none focus:border-terminal-accent disabled:opacity-50"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progress bar */}
      {status === "running" && (
        <div className="space-y-1">
          <div className="flex justify-between text-2xs text-terminal-muted">
            <span>Running backtest...</span>
            <span className="font-mono">{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-terminal-surface border border-terminal-border overflow-hidden">
            <div
              className="h-full rounded-full bg-terminal-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {status === "error" && error && (
        <div className="rounded border border-terminal-loss/40 bg-terminal-loss/10 px-3 py-2 text-xs text-terminal-loss">
          {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 mt-auto pt-2">
        {(status === "configuring" || status === "error") && (
          <button
            onClick={handleRun}
            disabled={!config.strategyId}
            className="flex-1 rounded-md bg-terminal-accent px-4 py-2.5 text-sm font-bold text-white uppercase tracking-wider hover:bg-terminal-accent-dim transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Run Backtest
          </button>
        )}
        {status === "running" && (
          <button
            disabled
            className="flex-1 rounded-md bg-terminal-accent/50 px-4 py-2.5 text-sm font-bold text-white/50 uppercase tracking-wider cursor-not-allowed"
          >
            Running...
          </button>
        )}
        {status === "complete" && (
          <>
            <div className="flex items-center gap-2 flex-1 rounded-md bg-terminal-profit/10 border border-terminal-profit/30 px-4 py-2.5">
              <span className="h-2 w-2 rounded-full bg-terminal-profit" />
              <span className="text-xs font-semibold text-terminal-profit">Backtest Complete</span>
            </div>
            <button
              onClick={handleReset}
              className={cn(
                "rounded-md px-4 py-2.5 text-xs font-semibold",
                "bg-terminal-surface border border-terminal-border text-terminal-text-secondary hover:bg-terminal-panel transition-colors",
              )}
            >
              New Run
            </button>
          </>
        )}
      </div>
    </div>
  );
}
