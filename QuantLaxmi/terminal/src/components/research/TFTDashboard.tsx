import { cn } from "@/lib/utils";

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

const DEMO_HYPERPARAMS: HyperparamSet[] = [
  { param: "hidden_size", value: 64 },
  { param: "attention_heads", value: 4 },
  { param: "dropout", value: 0.15 },
  { param: "learning_rate", value: 0.0008 },
  { param: "batch_size", value: 64 },
  { param: "seq_len", value: 20 },
  { param: "num_layers", value: 2 },
  { param: "gradient_clip", value: 1.0 },
  { param: "weight_decay", value: 0.0001 },
];

const DEMO_FOLDS: FoldMetric[] = [
  { fold: 1, trainSharpe: 2.45, valSharpe: 1.87, trainLoss: 0.0342, valLoss: 0.0418, epochs: 50 },
  { fold: 2, trainSharpe: 2.61, valSharpe: 1.74, trainLoss: 0.0328, valLoss: 0.0445, epochs: 50 },
  { fold: 3, trainSharpe: 2.38, valSharpe: 2.16, trainLoss: 0.0351, valLoss: 0.0402, epochs: 48 },
  { fold: 4, trainSharpe: 2.72, valSharpe: 1.04, trainLoss: 0.0315, valLoss: 0.0489, epochs: 50 },
  { fold: 5, trainSharpe: 2.55, valSharpe: 1.92, trainLoss: 0.0335, valLoss: 0.0425, epochs: 47 },
  { fold: 6, trainSharpe: 2.48, valSharpe: 1.68, trainLoss: 0.0340, valLoss: 0.0451, epochs: 50 },
  { fold: 7, trainSharpe: 2.67, valSharpe: 2.05, trainLoss: 0.0320, valLoss: 0.0412, epochs: 50 },
  { fold: 8, trainSharpe: 2.41, valSharpe: 1.55, trainLoss: 0.0348, valLoss: 0.0462, epochs: 45 },
  { fold: 9, trainSharpe: 2.58, valSharpe: 1.88, trainLoss: 0.0330, valLoss: 0.0430, epochs: 50 },
  { fold: 10, trainSharpe: 2.70, valSharpe: 1.95, trainLoss: 0.0318, valLoss: 0.0410, epochs: 50 },
  { fold: 11, trainSharpe: 2.52, valSharpe: 1.78, trainLoss: 0.0338, valLoss: 0.0440, epochs: 49 },
];

const DEMO_PROPS: Required<TFTDashboardProps> = {
  totalFolds: 16,
  completedFolds: 11,
  phase: "Phase 5 — Production Pass",
  bestHyperparams: DEMO_HYPERPARAMS,
  foldMetrics: DEMO_FOLDS,
  status: "running",
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
  const p = { ...DEMO_PROPS, ...props };
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
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-100">
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
                  <td className="px-3 py-1.5 text-gray-300">{hp.param}</td>
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
                  <td className="px-3 py-1.5 text-center text-gray-300 font-semibold">{f.fold}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-accent tabular-nums">{f.trainSharpe.toFixed(2)}</td>
                  <td className={cn("px-3 py-1.5 text-right font-semibold tabular-nums", sharpeColor(f.valSharpe))}>
                    {f.valSharpe.toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-right text-gray-400 tabular-nums">{f.trainLoss.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-gray-400 tabular-nums">{f.valLoss.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-gray-400 tabular-nums">{f.epochs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
