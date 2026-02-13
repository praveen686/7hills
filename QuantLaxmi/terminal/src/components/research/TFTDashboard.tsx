import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FoldMetric {
  fold: number;
  trainSharpe: number;
  valSharpe: number;
  trainLoss: number;
  valLoss: number;
  epochs: number;
}

interface HyperparamSet {
  param: string;
  value: string | number;
}

interface TFTDashboardProps {
  totalFolds?: number;
  completedFolds?: number;
  phase?: string;
  bestHyperparams?: HyperparamSet[];
  foldMetrics?: FoldMetric[];
  status?: "running" | "complete" | "idle" | "error";
}

// ---------------------------------------------------------------------------
// Demo data
// ---------------------------------------------------------------------------

const EMPTY_DEFAULTS: Required<TFTDashboardProps> = {
  totalFolds: 0,
  completedFolds: 0,
  phase: "Idle",
  bestHyperparams: [],
  foldMetrics: [],
  status: "idle",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sharpeColor(s: number): string {
  if (s >= 1.5) return "text-terminal-profit";
  if (s >= 1.0) return "text-terminal-warning";
  return "text-terminal-loss";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TFTDashboard(props: TFTDashboardProps) {
  const [fetched, setFetched] = useState<Partial<TFTDashboardProps>>({});

  useEffect(() => {
    let active = true;
    const fetchStatus = () => {
      apiFetch<{
        status: string;
        total_folds: number;
        completed_folds: number;
        phase: string;
        best_hyperparams: Array<{ param: string; value: string | number }>;
        fold_metrics: Array<{
          fold: number;
          trainSharpe: number;
          valSharpe: number;
          trainLoss: number;
          valLoss: number;
          epochs: number;
        }>;
      }>("/api/research/tft-status").then((data) => {
        if (!active) return;
        setFetched({
          status: data.status as any,
          totalFolds: data.total_folds,
          completedFolds: data.completed_folds,
          phase: data.phase,
          bestHyperparams: data.best_hyperparams,
          foldMetrics: data.fold_metrics,
        });
      }).catch(() => {});
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  const p = { ...EMPTY_DEFAULTS, ...fetched, ...props };
  const progressPct = p.totalFolds > 0 ? (p.completedFolds / p.totalFolds) * 100 : 0;

  const avgTrainSharpe = p.foldMetrics.length > 0
    ? p.foldMetrics.reduce((s, f) => s + f.trainSharpe, 0) / p.foldMetrics.length
    : 0;
  const avgValSharpe = p.foldMetrics.length > 0
    ? p.foldMetrics.reduce((s, f) => s + f.valSharpe, 0) / p.foldMetrics.length
    : 0;

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto scrollbar-thin">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-widest text-terminal-text">
          TFT Training Dashboard
        </h2>
        <span className={cn(
          "px-2 py-0.5 rounded text-2xs font-semibold uppercase",
          p.status === "running" ? "bg-terminal-accent/20 text-terminal-accent animate-pulse" :
          p.status === "complete" ? "bg-terminal-profit/20 text-terminal-profit" :
          p.status === "error" ? "bg-terminal-loss/20 text-terminal-loss" :
          "bg-terminal-surface text-terminal-muted",
        )}>
          {p.status}
        </span>
      </div>

      {/* Phase and progress */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-terminal-accent font-medium">{p.phase}</span>
          <span className="text-2xs font-mono text-terminal-muted">
            {p.completedFolds}/{p.totalFolds} folds
          </span>
        </div>
        <div className="h-2.5 rounded-full bg-terminal-surface border border-terminal-border overflow-hidden">
          <div
            className="h-full rounded-full bg-terminal-accent transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </section>

      {/* Best hyperparams */}
      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-terminal-muted">
          Best Hyperparameters
        </h3>
        <div className="rounded-md bg-terminal-surface border border-terminal-border overflow-hidden">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="bg-terminal-panel">
                <th className="px-3 py-1.5 text-left text-terminal-muted">Parameter</th>
                <th className="px-3 py-1.5 text-right text-terminal-muted">Value</th>
              </tr>
            </thead>
            <tbody>
              {p.bestHyperparams.map((hp) => (
                <tr key={hp.param} className="border-t border-terminal-border/50">
                  <td className="px-3 py-1.5 text-terminal-text-secondary">{hp.param}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-accent tabular-nums">
                    {String(hp.value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-fold metrics */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted">
            Per-Fold Metrics
          </h3>
          <div className="flex items-center gap-3 text-2xs font-mono">
            <span className="text-terminal-muted">
              Avg Train: <span className="text-terminal-accent">{avgTrainSharpe.toFixed(2)}</span>
            </span>
            <span className="text-terminal-muted">
              Avg Val: <span className={sharpeColor(avgValSharpe)}>{avgValSharpe.toFixed(2)}</span>
            </span>
          </div>
        </div>
        <div className="overflow-x-auto rounded-md border border-terminal-border">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="bg-terminal-panel text-terminal-muted">
                <th className="px-3 py-1.5 text-center border-b border-terminal-border">Fold</th>
                <th className="px-3 py-1.5 text-right border-b border-terminal-border">Train Sharpe</th>
                <th className="px-3 py-1.5 text-right border-b border-terminal-border">Val Sharpe</th>
                <th className="px-3 py-1.5 text-right border-b border-terminal-border">Train Loss</th>
                <th className="px-3 py-1.5 text-right border-b border-terminal-border">Val Loss</th>
                <th className="px-3 py-1.5 text-right border-b border-terminal-border">Epochs</th>
              </tr>
            </thead>
            <tbody>
              {p.foldMetrics.map((f) => (
                <tr key={f.fold} className="border-t border-terminal-border/50 hover:bg-terminal-surface/50 transition-colors">
                  <td className="px-3 py-1.5 text-center text-terminal-text-secondary font-semibold">{f.fold}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-accent tabular-nums">{f.trainSharpe.toFixed(2)}</td>
                  <td className={cn("px-3 py-1.5 text-right font-semibold tabular-nums", sharpeColor(f.valSharpe))}>
                    {f.valSharpe.toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-right text-terminal-muted tabular-nums">{f.trainLoss.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-muted tabular-nums">{f.valLoss.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-muted tabular-nums">{f.epochs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
