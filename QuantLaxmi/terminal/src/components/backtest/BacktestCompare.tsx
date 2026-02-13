import { useState, useMemo, useEffect } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StrategyMetrics {
  id: string;
  name: string;
  totalReturn: number;
  sharpe: number;
  sortino: number;
  maxDD: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgHoldDays: number;
  calmarRatio: number;
}

interface BacktestCompareProps {
  availableStrategies?: StrategyMetrics[];
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Metric definitions — which direction is "best"
// ---------------------------------------------------------------------------

interface MetricDef {
  key: keyof StrategyMetrics;
  label: string;
  format: (v: number) => string;
  higherIsBetter: boolean;
}

const METRICS: MetricDef[] = [
  { key: "totalReturn", label: "Total Return (%)", format: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`, higherIsBetter: true },
  { key: "sharpe", label: "Sharpe Ratio", format: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "sortino", label: "Sortino Ratio", format: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "maxDD", label: "Max Drawdown (%)", format: (v) => `${v.toFixed(2)}%`, higherIsBetter: false },
  { key: "winRate", label: "Win Rate", format: (v) => `${(v * 100).toFixed(0)}%`, higherIsBetter: true },
  { key: "profitFactor", label: "Profit Factor", format: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "totalTrades", label: "Total Trades", format: (v) => String(v), higherIsBetter: true },
  { key: "avgHoldDays", label: "Avg Hold (days)", format: (v) => v.toFixed(1), higherIsBetter: false },
  { key: "calmarRatio", label: "Calmar Ratio", format: (v) => v.toFixed(2), higherIsBetter: true },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BacktestCompare({ availableStrategies }: BacktestCompareProps) {
  const [fetchedStrategies, setFetchedStrategies] = useState<StrategyMetrics[]>([]);
  const [loading, setLoading] = useState(!availableStrategies);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (availableStrategies) return;

    // Primary: load from backtest history + strategies name map
    // Fallback: if no completed runs, load from research results
    setLoading(true);
    setError(null);
    (async () => {
      try {
        // Fetch strategy name map and backtest history in parallel
        const [strategyList, historyList] = await Promise.all([
          apiFetch<Array<{ strategy_id: string; name: string; default_params: unknown }>>(
            "/api/backtest/strategies",
          ).catch(() => [] as Array<{ strategy_id: string; name: string; default_params: unknown }>),
          apiFetch<Array<{
            backtest_id: string;
            status: string;
            strategy_id: string;
            start_date: string;
            end_date: string;
            result: {
              total_return: number;
              sharpe_ratio: number;
              sortino_ratio: number;
              max_drawdown: number;
              win_rate: number;
              profit_factor: number;
              n_trades: number;
            } | null;
          }>>("/api/backtest/history").catch(() => []),
        ]);

        // Build name map from strategies endpoint
        const nameMap: Record<string, string> = {};
        for (const s of strategyList) {
          nameMap[s.strategy_id] = s.name;
        }

        // Filter to completed runs with results
        const completed = historyList.filter(
          (h) => h.status === "completed" && h.result != null,
        );

        if (completed.length > 0) {
          // Deduplicate by strategy_id — keep first (latest, backend sorts by created_at desc)
          const seen = new Set<string>();
          const unique = completed.filter((h) => {
            if (seen.has(h.strategy_id)) return false;
            seen.add(h.strategy_id);
            return true;
          });

          setFetchedStrategies(
            unique.map((h) => {
              const r = h.result!;
              return {
                id: h.strategy_id,
                name: nameMap[h.strategy_id] ?? h.strategy_id,
                totalReturn: (r.total_return ?? 0) * 100,
                sharpe: r.sharpe_ratio ?? 0,
                sortino: r.sortino_ratio ?? 0,
                maxDD: (r.max_drawdown ?? 0) * 100,
                winRate: ((r.win_rate ?? 0) > 1 ? (r.win_rate ?? 0) / 100 : (r.win_rate ?? 0)),
                profitFactor: r.profit_factor ?? 0,
                totalTrades: r.n_trades ?? 0,
                avgHoldDays: 0,
                calmarRatio:
                  r.max_drawdown && r.max_drawdown > 0
                    ? (r.sharpe_ratio ?? 0) / (r.max_drawdown * 100)
                    : 0,
              };
            }),
          );
          return;
        }

        // Fallback: load research results when no completed backtest runs
        const researchList = await apiFetch<Array<{
          strategy_id: string;
          name: string;
          best_sharpe: number;
          best_return: number;
          variants: Array<{ max_dd: number; win_rate: number; trades: number }>;
        }>>("/api/research/results").catch(() => []);

        const seen = new Set<string>();
        const uniqueResearch = researchList.filter((s) => {
          if (seen.has(s.strategy_id)) return false;
          seen.add(s.strategy_id);
          return true;
        });

        setFetchedStrategies(
          uniqueResearch.map((s) => {
            const best = s.variants[0];
            return {
              id: s.strategy_id,
              name: s.name,
              totalReturn: s.best_return,
              sharpe: s.best_sharpe,
              sortino: 0,
              maxDD: best?.max_dd ?? 0,
              winRate: (best?.win_rate ?? 0) / 100,
              profitFactor: 0,
              totalTrades: best?.trades ?? 0,
              avgHoldDays: 0,
              calmarRatio: best?.max_dd ? s.best_sharpe / (best.max_dd || 1) : 0,
            };
          }),
        );
      } catch {
        setError("Failed to load strategy data. Is the API running?");
      } finally {
        setLoading(false);
      }
    })();
  }, [availableStrategies]);

  const strategies = availableStrategies ?? fetchedStrategies;
  const [selected, setSelected] = useState<string[]>(["", "", ""]);

  const selectedStrategies = useMemo(
    () => selected.map((id) => strategies.find((s) => s.id === id)).filter(Boolean) as StrategyMetrics[],
    [selected, strategies],
  );

  const handleChange = (index: number, id: string) => {
    setSelected((prev) => {
      const next = [...prev];
      next[index] = id;
      return next;
    });
  };

  // Find best value for each metric among selected
  const bestValues = useMemo(() => {
    const best = new Map<string, number>();
    for (const metric of METRICS) {
      const values = selectedStrategies.map((s) => s[metric.key] as number);
      if (values.length === 0) continue;
      const bestVal = metric.higherIsBetter
        ? Math.max(...values)
        : Math.min(...values);
      best.set(metric.key, bestVal);
    }
    return best;
  }, [selectedStrategies]);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
        Strategy Comparison
      </h2>

      {/* Selectors */}
      <div className="flex gap-3 flex-wrap">
        {[0, 1, 2].map((idx) => (
          <div key={idx} className="space-y-1 min-w-[160px]">
            <label className="text-2xs text-terminal-muted font-medium">
              Strategy {idx + 1}
            </label>
            <select
              value={selected[idx] ?? ""}
              onChange={(e) => handleChange(idx, e.target.value)}
              className="w-full rounded bg-terminal-surface border border-terminal-border px-2 py-1.5 text-xs text-terminal-text-secondary font-mono focus:outline-none focus:border-terminal-accent"
            >
              <option value="">-- None --</option>
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {/* Comparison table */}
      {selectedStrategies.length > 0 && (
        <div className="overflow-x-auto rounded border border-terminal-border">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="bg-terminal-surface">
                <th className="px-3 py-2 text-left text-terminal-muted border-b border-terminal-border min-w-[140px]">
                  Metric
                </th>
                {selectedStrategies.map((s, idx) => (
                  <th
                    key={`${s.id}-${idx}`}
                    className="px-3 py-2 text-right text-terminal-accent border-b border-terminal-border min-w-[100px]"
                  >
                    {s.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS.map((metric) => (
                <tr key={metric.key} className="border-t border-terminal-border hover:bg-terminal-surface/50 transition-colors">
                  <td className="px-3 py-2 text-terminal-text-secondary font-medium">{metric.label}</td>
                  {selectedStrategies.map((s, idx) => {
                    const val = s[metric.key] as number;
                    const isBest = val === bestValues.get(metric.key) && selectedStrategies.length > 1;
                    return (
                      <td
                        key={`${s.id}-${idx}`}
                        className={cn(
                          "px-3 py-2 text-right tabular-nums",
                          isBest ? "text-terminal-accent font-bold" : "text-terminal-text-secondary",
                        )}
                      >
                        {isBest && (
                          <span className="inline-block w-1.5 h-1.5 rounded-full bg-terminal-accent mr-1.5 align-middle" />
                        )}
                        {metric.format(val)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono animate-pulse">
          Loading strategy data...
        </div>
      )}

      {!loading && error && strategies.length === 0 && (
        <div className="flex items-center justify-center h-32 text-terminal-loss text-xs">
          {error}
        </div>
      )}

      {!loading && !error && selectedStrategies.length === 0 && (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
          {strategies.length === 0
            ? "No backtest results yet. Run a backtest first."
            : "Select at least one strategy to compare"}
        </div>
      )}
    </div>
  );
}
