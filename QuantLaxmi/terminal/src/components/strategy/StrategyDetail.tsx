import { useEffect, useRef, useState } from "react";
import { useAtomValue } from "jotai";
import {
  createChart,
  type IChartApi,
  type Time,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { useTauriCommand } from "@/hooks/useTauriCommand";
import { cn } from "@/lib/utils";
import { getChartColors, withAlpha } from "@/lib/chartTheme";
import { themeAtom } from "@/stores/workspace";
import type { StrategyStatus } from "@/components/strategy/StrategyPanel";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EquityPoint {
  time: string;
  value: number;
}

interface DrawdownPoint {
  time: string;
  value: number;
}

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  holdingPeriod: string;
}

interface PositionRow {
  symbol: string;
  side: "LONG" | "SHORT";
  quantity: number;
  avgPrice: number;
  ltp: number;
  unrealizedPnl: number;
}

interface SignalHistoryItem {
  timestamp: string;
  direction: "BUY" | "SELL";
  conviction: number;
  acted: boolean;
}

interface StrategyDetailData {
  id: string;
  name: string;
  status: StrategyStatus;
  equity: number;
  returnPct: number;
  sharpe: number;
  sortino: number;
  maxDD: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  equityCurve: EquityPoint[];
  drawdownCurve: DrawdownPoint[];
  recentTrades: Trade[];
  positions: PositionRow[];
  signalHistory: SignalHistoryItem[];
  parameters: Record<string, string | number>;
}

interface StrategyDetailProps {
  strategyId: string;
  onBack?: () => void;
}


// ---------------------------------------------------------------------------
// Chart component (equity + drawdown stacked)
// ---------------------------------------------------------------------------

function EquityChart({ equity, drawdown }: { equity: EquityPoint[]; drawdown: DrawdownPoint[] }) {
  const eqRef = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  const eqChartRef = useRef<IChartApi | null>(null);
  const ddChartRef = useRef<IChartApi | null>(null);
  const theme = useAtomValue(themeAtom);

  useEffect(() => {
    if (!eqRef.current || !ddRef.current) return;

    const c = getChartColors();

    const chartOpts = {
      width: eqRef.current.clientWidth,
      layout: {
        background: { type: ColorType.Solid as const, color: "transparent" },
        textColor: c.text,
        fontSize: 10,
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border, timeVisible: false },
      crosshair: {
        horzLine: { color: c.crosshair, style: LineStyle.Dashed },
        vertLine: { color: c.crosshair, style: LineStyle.Dashed },
      },
    };

    // Equity chart
    const eqChart = createChart(eqRef.current, { ...chartOpts, height: 180 });
    const eqSeries = eqChart.addAreaSeries({
      lineColor: c.accent,
      lineWidth: 2,
      topColor: withAlpha(c.accent, "4d"),
      bottomColor: withAlpha(c.accent, "05"),
      priceFormat: { type: "custom", formatter: (p: number) => `₹${(p / 100000).toFixed(1)}L` },
    });
    eqSeries.setData(equity.map((p) => ({ time: p.time as Time, value: p.value })));
    eqChart.timeScale().fitContent();
    eqChartRef.current = eqChart;

    // Drawdown chart
    const ddChart = createChart(ddRef.current, { ...chartOpts, height: 100 });
    const ddSeries = ddChart.addAreaSeries({
      lineColor: c.loss,
      lineWidth: 2,
      topColor: withAlpha(c.loss, "4d"),
      bottomColor: withAlpha(c.loss, "05"),
      invertFilledArea: true,
      priceFormat: { type: "custom", formatter: (p: number) => `${p.toFixed(2)}%` },
    });
    ddSeries.setData(drawdown.map((p) => ({ time: p.time as Time, value: p.value })));
    ddChart.timeScale().fitContent();
    ddChartRef.current = ddChart;

    // Sync time scales
    eqChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) ddChart.timeScale().setVisibleLogicalRange(range);
    });
    ddChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) eqChart.timeScale().setVisibleLogicalRange(range);
    });

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        eqChart.applyOptions({ width: w });
        ddChart.applyOptions({ width: w });
      }
    });
    ro.observe(eqRef.current);

    return () => {
      ro.disconnect();
      eqChart.remove();
      ddChart.remove();
    };
  }, [equity, drawdown, theme]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-2xs text-terminal-muted uppercase tracking-wider px-1">Equity Curve</span>
      <div ref={eqRef} className="w-full rounded border border-terminal-border bg-terminal-surface" />
      <span className="text-2xs text-terminal-muted uppercase tracking-wider px-1 mt-1">Drawdown</span>
      <div ref={ddRef} className="w-full rounded border border-terminal-border bg-terminal-surface" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metrics row
// ---------------------------------------------------------------------------

function MetricPill({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 rounded bg-terminal-surface px-3 py-1.5 border border-terminal-border">
      <span className="text-2xs text-terminal-muted uppercase">{label}</span>
      <span className={cn("font-mono text-sm font-semibold", color ?? "text-terminal-text")}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StrategyDetail({ strategyId, onBack }: StrategyDetailProps) {
  const { data, execute } = useTauriCommand<StrategyDetailData>("get_strategy_detail");
  const [detail, setDetail] = useState<StrategyDetailData | null>(null);

  useEffect(() => {
    execute({ strategy_id: strategyId }).catch(() => {});
  }, [execute, strategyId]);

  useEffect(() => {
    if (!data) return;
    const d = data as any;
    if (d.id && d.name) setDetail(d);
  }, [data]);

  if (!detail) {
    return (
      <div className="flex flex-col gap-4 p-4 h-full">
        {onBack && (
          <button onClick={onBack} className="rounded p-1.5 hover:bg-terminal-surface transition-colors text-terminal-muted hover:text-terminal-text self-start">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
        <div className="flex items-center justify-center flex-1 text-terminal-muted text-xs font-mono">
          Loading strategy data...
        </div>
      </div>
    );
  }

  const statusColor = {
    live: "text-terminal-profit",
    paused: "text-terminal-warning",
    stopped: "text-terminal-loss",
  }[detail.status];

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        {onBack && (
          <button
            onClick={onBack}
            className="rounded p-1.5 hover:bg-terminal-surface transition-colors text-terminal-muted hover:text-terminal-text"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
        <div className="flex-1">
          <h2 className="text-sm font-bold text-terminal-text font-mono">{detail.id}</h2>
          <span className={cn("text-2xs font-semibold uppercase", statusColor)}>{detail.status}</span>
        </div>
        <span className="font-mono text-lg font-bold text-terminal-text">
          {`₹${(detail.equity / 100000).toFixed(1)}L`}
        </span>
      </div>

      {/* Metrics row */}
      <div className="flex flex-wrap gap-2">
        <MetricPill label="Return" value={`${detail.returnPct >= 0 ? "+" : ""}${detail.returnPct.toFixed(2)}%`} color={detail.returnPct >= 0 ? "text-terminal-profit" : "text-terminal-loss"} />
        <MetricPill label="Sharpe" value={detail.sharpe.toFixed(2)} />
        <MetricPill label="Sortino" value={detail.sortino.toFixed(2)} />
        <MetricPill label="Max DD" value={`${detail.maxDD.toFixed(2)}%`} color="text-terminal-loss" />
        <MetricPill label="Win Rate" value={`${(detail.winRate * 100).toFixed(0)}%`} />
        <MetricPill label="PF" value={detail.profitFactor.toFixed(2)} />
        <MetricPill label="Trades" value={String(detail.totalTrades)} />
      </div>

      {/* Charts */}
      <EquityChart equity={detail.equityCurve} drawdown={detail.drawdownCurve} />

      {/* Positions */}
      {detail.positions.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
            Open Positions
          </h3>
          <div className="overflow-x-auto rounded border border-terminal-border">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-terminal-surface text-terminal-muted">
                  <th className="px-3 py-1.5 text-left">Symbol</th>
                  <th className="px-3 py-1.5 text-left">Side</th>
                  <th className="px-3 py-1.5 text-right">Qty</th>
                  <th className="px-3 py-1.5 text-right">Avg</th>
                  <th className="px-3 py-1.5 text-right">LTP</th>
                  <th className="px-3 py-1.5 text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {detail.positions.map((pos) => {
                  const avg = pos.avgPrice ?? (pos as any).avg_price ?? 0;
                  const ltp = pos.ltp ?? 0;
                  const pnl = pos.unrealizedPnl ?? (pos as any).unrealized_pnl ?? 0;
                  return (
                  <tr key={pos.symbol} className="border-t border-terminal-border hover:bg-terminal-surface/50">
                    <td className="px-3 py-1.5 text-terminal-text-secondary">{pos.symbol}</td>
                    <td className={cn("px-3 py-1.5", pos.side === "LONG" ? "text-terminal-profit" : "text-terminal-loss")}>{pos.side}</td>
                    <td className="px-3 py-1.5 text-right text-terminal-text-secondary">{pos.quantity}</td>
                    <td className="px-3 py-1.5 text-right text-terminal-text-secondary">{avg.toLocaleString("en-IN")}</td>
                    <td className="px-3 py-1.5 text-right text-terminal-text">{ltp.toLocaleString("en-IN")}</td>
                    <td className={cn("px-3 py-1.5 text-right font-semibold", pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                      {pnl >= 0 ? "+" : ""}{pnl.toLocaleString("en-IN")}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Recent trades */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Recent Trades
        </h3>
        <div className="overflow-x-auto rounded border border-terminal-border">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="bg-terminal-surface text-terminal-muted">
                <th className="px-3 py-1.5 text-left">Time</th>
                <th className="px-3 py-1.5 text-left">Symbol</th>
                <th className="px-3 py-1.5 text-left">Side</th>
                <th className="px-3 py-1.5 text-right">Qty</th>
                <th className="px-3 py-1.5 text-right">Entry</th>
                <th className="px-3 py-1.5 text-right">Exit</th>
                <th className="px-3 py-1.5 text-right">P&L</th>
                <th className="px-3 py-1.5 text-right">Hold</th>
              </tr>
            </thead>
            <tbody>
              {detail.recentTrades.map((t) => {
                const entry = t.entryPrice ?? (t as any).entry_price ?? 0;
                const exit = t.exitPrice ?? (t as any).exit_price ?? 0;
                const pnl = t.pnl ?? 0;
                const hold = t.holdingPeriod ?? (t as any).holding_period ?? "--";
                return (
                <tr key={t.id} className="border-t border-terminal-border hover:bg-terminal-surface/50">
                  <td className="px-3 py-1.5 text-terminal-muted">{t.timestamp}</td>
                  <td className="px-3 py-1.5 text-terminal-text-secondary">{t.symbol}</td>
                  <td className={cn("px-3 py-1.5", t.side === "BUY" ? "text-terminal-profit" : "text-terminal-loss")}>{t.side}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-text-secondary">{t.quantity}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-text-secondary">{entry.toLocaleString("en-IN")}</td>
                  <td className="px-3 py-1.5 text-right text-terminal-text-secondary">{exit.toLocaleString("en-IN")}</td>
                  <td className={cn("px-3 py-1.5 text-right font-semibold", pnl >= 0 ? "text-terminal-profit" : "text-terminal-loss")}>
                    {pnl >= 0 ? "+" : ""}{pnl.toLocaleString("en-IN")}
                  </td>
                  <td className="px-3 py-1.5 text-right text-terminal-muted">{hold}</td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Signal history */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Signal History
        </h3>
        <div className="space-y-1">
          {detail.signalHistory.map((sig, i) => (
            <div
              key={i}
              className={cn(
                "flex items-center gap-3 rounded px-3 py-1.5 text-xs",
                sig.acted ? "bg-terminal-surface" : "bg-terminal-surface/40 opacity-60",
              )}
            >
              <span className="font-mono text-terminal-muted min-w-[110px]">{sig.timestamp}</span>
              <span className={cn("font-semibold min-w-[32px]", sig.direction === "BUY" ? "text-terminal-profit" : "text-terminal-loss")}>
                {sig.direction}
              </span>
              <div className="flex-1 h-1 rounded-full bg-terminal-bg overflow-hidden max-w-[100px]">
                <div
                  className="h-full rounded-full bg-terminal-accent"
                  style={{ width: `${sig.conviction * 100}%` }}
                />
              </div>
              <span className="font-mono text-terminal-muted">{(sig.conviction * 100).toFixed(0)}%</span>
              <span className={cn("text-2xs", sig.acted ? "text-terminal-profit" : "text-terminal-muted")}>
                {sig.acted ? "ACTED" : "SKIPPED"}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Parameters */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-terminal-muted mb-2">
          Parameters
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {Object.entries(detail.parameters).map(([key, val]) => (
            <div key={key} className="rounded bg-terminal-surface border border-terminal-border px-3 py-1.5">
              <div className="text-2xs text-terminal-muted">{key}</div>
              <div className="font-mono text-xs text-terminal-text">{String(val)}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
