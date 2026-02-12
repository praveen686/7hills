import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WalkForwardFold {
  foldNum: number;
  trainStart: string;
  trainEnd: string;
  testStart: string;
  testEnd: string;
  isSharpe: number;
  oosSharpe: number;
}

interface WalkForwardViewProps {
  folds?: WalkForwardFold[];
  strategyName?: string;
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

const DEFAULT_FOLDS: WalkForwardFold[] = [
  { foldNum: 1, trainStart: "2025-01-01", trainEnd: "2025-04-01", testStart: "2025-04-02", testEnd: "2025-07-01", isSharpe: 2.14, oosSharpe: 1.87 },
  { foldNum: 2, trainStart: "2025-04-02", trainEnd: "2025-07-01", testStart: "2025-07-02", testEnd: "2025-10-01", isSharpe: 2.45, oosSharpe: 1.74 },
  { foldNum: 3, trainStart: "2025-07-02", trainEnd: "2025-10-01", testStart: "2025-10-02", testEnd: "2026-01-01", isSharpe: 1.98, oosSharpe: 2.16 },
  { foldNum: 4, trainStart: "2025-10-02", trainEnd: "2026-01-01", testStart: "2026-01-02", testEnd: "2026-02-12", isSharpe: 2.31, oosSharpe: 1.04 },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sharpeColor(s: number): string {
  if (s >= 1.0) return "text-terminal-profit";
  if (s >= 0.5) return "text-terminal-warning";
  if (s >= 0) return "text-terminal-warning";
  return "text-terminal-loss";
}

function sharpeBgColor(s: number): string {
  if (s >= 1.0) return "bg-terminal-profit/15";
  if (s >= 0.5) return "bg-terminal-warning/15";
  if (s >= 0) return "bg-terminal-warning/10";
  return "bg-terminal-loss/15";
}

function degradation(is_s: number, oos_s: number): number {
  if (is_s === 0) return 0;
  return ((is_s - oos_s) / Math.abs(is_s)) * 100;
}

// ---------------------------------------------------------------------------
// Bar chart — IS vs OOS Sharpe comparison
// ---------------------------------------------------------------------------

function SharpeBarChart({ folds }: { folds: WalkForwardFold[] }) {
  const maxSharpe = Math.max(...folds.flatMap((f) => [f.isSharpe, f.oosSharpe]), 0.1);
  const barScale = 100 / maxSharpe;

  return (
    <div className="flex flex-col gap-2">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted">
        IS vs OOS Sharpe per Fold
      </h4>
      <div className="space-y-2">
        {folds.map((fold) => (
          <div key={fold.foldNum} className="flex items-center gap-3">
            <span className="text-2xs font-mono text-terminal-muted w-10 text-right">
              F{fold.foldNum}
            </span>
            <div className="flex-1 flex flex-col gap-0.5">
              {/* IS bar */}
              <div className="flex items-center gap-2">
                <div className="h-3 rounded-r bg-terminal-accent/50" style={{ width: `${Math.max(fold.isSharpe * barScale, 2)}%` }} />
                <span className="text-2xs font-mono text-terminal-accent">{fold.isSharpe.toFixed(2)} IS</span>
              </div>
              {/* OOS bar */}
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "h-3 rounded-r",
                    fold.oosSharpe >= 1.0 ? "bg-terminal-profit/60" : fold.oosSharpe >= 0.5 ? "bg-terminal-warning/60" : "bg-terminal-loss/60",
                  )}
                  style={{ width: `${Math.max(fold.oosSharpe * barScale, 2)}%` }}
                />
                <span className={cn("text-2xs font-mono", sharpeColor(fold.oosSharpe))}>{fold.oosSharpe.toFixed(2)} OOS</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      {/* Legend */}
      <div className="flex gap-4 text-2xs text-terminal-muted mt-1">
        <span className="flex items-center gap-1">
          <span className="h-2 w-4 rounded bg-terminal-accent/50" /> In-Sample
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-4 rounded bg-terminal-profit/60" /> Out-of-Sample
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function WalkForwardView({ folds, strategyName }: WalkForwardViewProps) {
  const data = folds ?? DEFAULT_FOLDS;
  const avgOOS = data.reduce((s, f) => s + f.oosSharpe, 0) / data.length;
  const avgIS = data.reduce((s, f) => s + f.isSharpe, 0) / data.length;
  const avgDeg = degradation(avgIS, avgOOS);

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
          Walk-Forward Validation{strategyName ? ` — ${strategyName}` : ""}
        </h2>
        <div className="flex items-center gap-3">
          <div className="rounded bg-terminal-surface border border-terminal-border px-3 py-1.5 text-center">
            <div className="text-2xs text-terminal-muted">Avg OOS Sharpe</div>
            <div className={cn("font-mono text-lg font-bold", sharpeColor(avgOOS))}>
              {avgOOS.toFixed(2)}
            </div>
          </div>
          <div className="rounded bg-terminal-surface border border-terminal-border px-3 py-1.5 text-center">
            <div className="text-2xs text-terminal-muted">Avg Degradation</div>
            <div className={cn("font-mono text-lg font-bold", avgDeg <= 30 ? "text-terminal-profit" : "text-terminal-warning")}>
              {avgDeg.toFixed(0)}%
            </div>
          </div>
        </div>
      </div>

      {/* Fold table */}
      <div className="overflow-x-auto rounded border border-terminal-border">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="bg-terminal-surface text-terminal-muted">
              <th className="px-3 py-2 text-center border-b border-terminal-border">Fold</th>
              <th className="px-3 py-2 text-left border-b border-terminal-border">Train Period</th>
              <th className="px-3 py-2 text-left border-b border-terminal-border">Test Period</th>
              <th className="px-3 py-2 text-right border-b border-terminal-border">IS Sharpe</th>
              <th className="px-3 py-2 text-right border-b border-terminal-border">OOS Sharpe</th>
              <th className="px-3 py-2 text-right border-b border-terminal-border">Degradation</th>
            </tr>
          </thead>
          <tbody>
            {data.map((fold) => {
              const deg = degradation(fold.isSharpe, fold.oosSharpe);
              return (
                <tr
                  key={fold.foldNum}
                  className={cn(
                    "border-t border-terminal-border hover:bg-terminal-surface/50 transition-colors",
                    sharpeBgColor(fold.oosSharpe),
                  )}
                >
                  <td className="px-3 py-2 text-center text-gray-200 font-semibold">
                    {fold.foldNum}
                  </td>
                  <td className="px-3 py-2 text-gray-400">
                    {fold.trainStart} → {fold.trainEnd}
                  </td>
                  <td className="px-3 py-2 text-gray-400">
                    {fold.testStart} → {fold.testEnd}
                  </td>
                  <td className="px-3 py-2 text-right text-terminal-accent">
                    {fold.isSharpe.toFixed(2)}
                  </td>
                  <td className={cn("px-3 py-2 text-right font-semibold", sharpeColor(fold.oosSharpe))}>
                    {fold.oosSharpe.toFixed(2)}
                  </td>
                  <td className={cn(
                    "px-3 py-2 text-right",
                    deg <= 20 ? "text-terminal-profit" : deg <= 40 ? "text-terminal-warning" : "text-terminal-loss",
                  )}>
                    {deg.toFixed(0)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="bg-terminal-surface font-semibold">
              <td className="px-3 py-2 text-center text-terminal-accent" colSpan={3}>Average</td>
              <td className="px-3 py-2 text-right text-terminal-accent">{avgIS.toFixed(2)}</td>
              <td className={cn("px-3 py-2 text-right", sharpeColor(avgOOS))}>{avgOOS.toFixed(2)}</td>
              <td className={cn("px-3 py-2 text-right", avgDeg <= 30 ? "text-terminal-profit" : "text-terminal-warning")}>{avgDeg.toFixed(0)}%</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <SharpeBarChart folds={data} />
    </div>
  );
}
