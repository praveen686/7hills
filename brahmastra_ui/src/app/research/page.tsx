"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchFeatureIC, fetchWalkForwardResults } from "@/lib/api";
import { formatRatio, formatSharpe, formatPct, pnlColor } from "@/lib/formatters";
import { clsx } from "clsx";
import type { FeatureIC, WalkForwardResult } from "@/lib/types";

// ============================================================
// Research Page — Feature IC & Walk-Forward Results
// ============================================================

export default function ResearchPage() {
  const { data: features, isLoading: featuresLoading } = useQuery<FeatureIC[]>({
    queryKey: ["feature-ic"],
    queryFn: fetchFeatureIC,
  });

  const { data: walkForward, isLoading: wfLoading } = useQuery<WalkForwardResult[]>({
    queryKey: ["walk-forward"],
    queryFn: fetchWalkForwardResults,
  });

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-white">Research</h1>
        <p className="text-sm text-gray-500 mt-1">
          Feature analysis, information coefficients, and walk-forward validation
        </p>
      </div>

      {/* Feature IC Table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
            Feature Information Coefficients
          </p>
          <span className="text-xs text-gray-600">
            {features?.length ?? 0} features
          </span>
        </div>

        {featuresLoading ? (
          <div className="animate-pulse space-y-2 p-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-8 bg-gray-800 rounded" />
            ))}
          </div>
        ) : features && features.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="table-header">Feature</th>
                  <th className="table-header text-right">IC Mean</th>
                  <th className="table-header text-right">IC Std</th>
                  <th className="table-header text-right">ICIR</th>
                  <th className="table-header text-right">Rank IC</th>
                  <th className="table-header">IC Bar</th>
                </tr>
              </thead>
              <tbody>
                {features
                  .sort((a, b) => Math.abs(b.icir ?? 0) - Math.abs(a.icir ?? 0))
                  .map((feat, i) => (
                    <tr
                      key={i}
                      className="border-b border-gray-800/50 hover:bg-gray-900/50"
                    >
                      <td className="table-cell text-sm font-medium text-white font-mono">
                        {feat.feature}
                      </td>
                      <td className={clsx("table-cell font-mono text-sm text-right", pnlColor(feat.ic_mean))}>
                        {formatRatio(feat.ic_mean)}
                      </td>
                      <td className="table-cell font-mono text-sm text-right text-gray-400">
                        {formatRatio(feat.ic_std ?? undefined)}
                      </td>
                      <td className={clsx("table-cell font-mono text-sm text-right font-medium", icirColor(feat.icir ?? 0))}>
                        {formatRatio(feat.icir ?? undefined)}
                      </td>
                      <td className={clsx("table-cell font-mono text-sm text-right", pnlColor(feat.rank_ic ?? undefined))}>
                        {formatRatio(feat.rank_ic ?? undefined)}
                      </td>
                      <td className="table-cell">
                        <ICBar value={feat.ic_mean} />
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex items-center justify-center h-48 text-sm text-gray-600">
            <div className="text-center">
              <p>No feature IC data available</p>
              <p className="text-xs text-gray-700 mt-1">
                Run feature analysis from the research pipeline
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Walk-Forward Results */}
      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
            Walk-Forward Validation
          </p>
          <span className="text-xs text-gray-600">
            {walkForward?.length ?? 0} folds
          </span>
        </div>

        {wfLoading ? (
          <div className="animate-pulse space-y-2 p-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-8 bg-gray-800 rounded" />
            ))}
          </div>
        ) : walkForward && walkForward.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="table-header">Fold</th>
                  <th className="table-header">Train Period</th>
                  <th className="table-header">Test Period</th>
                  <th className="table-header text-right">Train Sharpe</th>
                  <th className="table-header text-right">Test Sharpe</th>
                  <th className="table-header text-right">Degradation</th>
                  <th className="table-header">Health</th>
                </tr>
              </thead>
              <tbody>
                {walkForward.map((wf) => (
                  <tr
                    key={wf.fold}
                    className="border-b border-gray-800/50 hover:bg-gray-900/50"
                  >
                    <td className="table-cell font-mono text-sm text-gray-400">
                      #{wf.fold}
                    </td>
                    <td className="table-cell text-xs text-gray-500">
                      {wf.train_start} to {wf.train_end}
                    </td>
                    <td className="table-cell text-xs text-gray-500">
                      {wf.test_start} to {wf.test_end}
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-gray-300">
                      {formatSharpe(wf.train_sharpe)}
                    </td>
                    <td className={clsx("table-cell font-mono text-sm text-right font-medium", sharpeColor(wf.test_sharpe))}>
                      {formatSharpe(wf.test_sharpe)}
                    </td>
                    <td className={clsx("table-cell font-mono text-sm text-right", degradationColor(wf.degradation))}>
                      {formatPct(wf.degradation * 100, 1)}
                    </td>
                    <td className="table-cell">
                      <HealthIndicator degradation={wf.degradation} />
                    </td>
                  </tr>
                ))}
              </tbody>
              {walkForward.length > 1 && (
                <tfoot>
                  <tr className="border-t border-gray-700 bg-gray-900/50">
                    <td className="table-cell font-medium text-white" colSpan={3}>
                      Average
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-gray-300">
                      {formatSharpe(
                        walkForward.reduce((sum, w) => sum + w.train_sharpe, 0) /
                          walkForward.length
                      )}
                    </td>
                    <td className="table-cell font-mono text-sm text-right font-medium text-white">
                      {formatSharpe(
                        walkForward.reduce((sum, w) => sum + w.test_sharpe, 0) /
                          walkForward.length
                      )}
                    </td>
                    <td className="table-cell font-mono text-sm text-right text-gray-400">
                      {formatPct(
                        (walkForward.reduce((sum, w) => sum + w.degradation, 0) /
                          walkForward.length) *
                          100,
                        1
                      )}
                    </td>
                    <td className="table-cell" />
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        ) : (
          <div className="flex items-center justify-center h-48 text-sm text-gray-600">
            <div className="text-center">
              <p>No walk-forward results available</p>
              <p className="text-xs text-gray-700 mt-1">
                Run walk-forward validation from the research pipeline
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------- Helpers ----------

function icirColor(icir: number): string {
  if (Math.abs(icir) > 1.5) return "text-profit";
  if (Math.abs(icir) > 0.5) return "text-accent";
  return "text-gray-400";
}

function sharpeColor(sharpe: number): string {
  if (sharpe > 1.5) return "text-profit";
  if (sharpe > 0.5) return "text-accent";
  if (sharpe > 0) return "text-yellow-400";
  return "text-loss";
}

function degradationColor(deg: number): string {
  if (deg > -0.2) return "text-profit";
  if (deg > -0.5) return "text-yellow-400";
  return "text-loss";
}

function ICBar({ value }: { value: number }) {
  const maxWidth = 60;
  const absValue = Math.min(Math.abs(value), 0.3);
  const width = (absValue / 0.3) * maxWidth;
  const isPositive = value >= 0;

  return (
    <div className="flex items-center gap-1">
      {!isPositive && (
        <div
          className="h-2 rounded-full bg-loss/60"
          style={{ width: `${width}px` }}
        />
      )}
      <div className="w-px h-3 bg-gray-700" />
      {isPositive && (
        <div
          className="h-2 rounded-full bg-profit/60"
          style={{ width: `${width}px` }}
        />
      )}
    </div>
  );
}

function HealthIndicator({ degradation }: { degradation: number }) {
  const health =
    degradation > -0.2
      ? { label: "Healthy", color: "bg-profit", textColor: "text-profit" }
      : degradation > -0.5
        ? { label: "Warning", color: "bg-yellow-500", textColor: "text-yellow-400" }
        : { label: "Overfit", color: "bg-loss", textColor: "text-loss" };

  return (
    <span className={`inline-flex items-center gap-1.5 text-xs ${health.textColor}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${health.color}`} />
      {health.label}
    </span>
  );
}
