import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";

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

const DEMO_STRATEGIES: StrategyMetrics[] = [
  { id: "s1", name: "S1 VRP Options", totalReturn: 5.59, sharpe: 1.42, sortino: 1.98, maxDD: 2.10, winRate: 0.62, profitFactor: 1.65, totalTrades: 48, avgHoldDays: 4.2, calmarRatio: 2.66 },
  { id: "s4", name: "S4 IV Mean-Rev", totalReturn: 3.07, sharpe: 1.85, sortino: 2.41, maxDD: 1.20, winRate: 0.58, profitFactor: 1.52, totalTrades: 112, avgHoldDays: 1.8, calmarRatio: 2.56 },
  { id: "s5", name: "S5 Hawkes Jump", totalReturn: 4.29, sharpe: 2.14, sortino: 3.07, maxDD: 1.80, winRate: 0.65, profitFactor: 1.88, totalTrades: 67, avgHoldDays: 0.5, calmarRatio: 2.38 },
  { id: "s25", name: "S25 DFF", totalReturn: 6.09, sharpe: 1.87, sortino: 2.64, maxDD: 2.05, winRate: 0.64, profitFactor: 1.78, totalTrades: 93, avgHoldDays: 1.3, calmarRatio: 2.97 },
  { id: "s26", name: "S26 Crypto Flow", totalReturn: -1.50, sharpe: -0.88, sortino: -1.12, maxDD: 4.20, winRate: 0.42, profitFactor: 0.72, totalTrades: 156, avgHoldDays: 0.3, calmarRatio: -0.36 },
  { id: "tft", name: "TFT Ensemble", totalReturn: 8.80, sharpe: 1.97, sortino: 2.85, maxDD: 2.50, winRate: 0.57, profitFactor: 1.91, totalTrades: 210, avgHoldDays: 1.1, calmarRatio: 3.52 },
];

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
  const strategies = availableStrategies ?? DEMO_STRATEGIES;
  const [selected, setSelected] = useState<string[]>([
    strategies[0]?.id ?? "",
    strategies[3]?.id ?? "",
    strategies[5]?.id ?? "",
  ]);

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
      <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
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
              className="w-full rounded bg-terminal-surface border border-terminal-border px-2 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:border-terminal-accent"
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
                {selectedStrategies.map((s) => (
                  <th
                    key={s.id}
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
                  <td className="px-3 py-2 text-gray-300 font-medium">{metric.label}</td>
                  {selectedStrategies.map((s) => {
                    const val = s[metric.key] as number;
                    const isBest = val === bestValues.get(metric.key) && selectedStrategies.length > 1;
                    return (
                      <td
                        key={s.id}
                        className={cn(
                          "px-3 py-2 text-right tabular-nums",
                          isBest ? "text-terminal-accent font-bold" : "text-gray-200",
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

      {selectedStrategies.length === 0 && (
        <div className="flex items-center justify-center h-32 text-terminal-muted text-xs font-mono">
          Select at least one strategy to compare
        </div>
      )}
    </div>
  );
}
